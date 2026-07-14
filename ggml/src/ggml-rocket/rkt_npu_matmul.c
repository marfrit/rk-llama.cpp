/* SPDX-License-Identifier: MIT
 *
 * rkt_npu_matmul.c — reusable 2D-tiled INT8 matmul on the rocket NPU (see .h).
 */
#define _GNU_SOURCE
#include "rkt_npu_matmul.h"
#include "librocket.h"
#include "rocket_accel.h"
#include "rkt_matmul.h"
#include "rkt_operands.h"
#include "rkt_gemm.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

/* rocket_prep_bo timeout is an ABSOLUTE CLOCK_MONOTONIC deadline. Build one
 * rel_ns in the future, for the job-completion wait so a hung NPU job returns
 * an error instead of blocking forever (the driver watchdog is ~5s). */
static int64_t rkt_deadline_ns(int64_t rel_ns)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (int64_t)t.tv_sec * 1000000000LL + (int64_t)t.tv_nsec + rel_ns;
}

/* env-gated phase profiling (GGML_ROCKET_PROF): where does tiled-matmul time go */
static int rkt_prof_on(void){ static int v=-1; if(v<0) v=getenv("GGML_ROCKET_PROF")?1:0; return v; }
static uint64_t g_setup_ns,g_submit_ns,g_prep_ns,g_free_ns,g_tiles,g_calls;
static inline uint64_t rkt_now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec; }
__attribute__((destructor)) static void rkt_prof_dump(void){
	if(!rkt_prof_on()) return;
	fprintf(stderr,"[rocket-prof] calls=%lu tiles=%lu  setup=%.3fs submit=%.3fs npu_wait=%.3fs free=%.3fs\n",
		(unsigned long)g_calls,(unsigned long)g_tiles,
		g_setup_ns/1e9,g_submit_ns/1e9,g_prep_ns/1e9,g_free_ns/1e9);
}

extern int rkt_g_task_num;

int rkt_build_matmul_regcmd_scaled(uint64_t *out, int out_capacity,
				   uint32_t M, uint32_t N, uint32_t K,
				   uint64_t input_dma, uint64_t weights_dma,
				   uint64_t output_dma,
				   int32_t input_zero_point, int32_t weight_zero_point,
				   int32_t output_zero_point,
				   float input_scale, float weights_scale,
				   float output_scale, uint64_t bias_dma);

struct rkt_bo { uint32_t h; uint64_t dma, off; void *map; uint32_t sz; };

static int bo_alloc(int fd, struct rkt_bo *b, uint32_t size)
{
	b->sz = (size + 0xfffu) & ~0xfffu;
	if (rocket_create_bo(fd, b->sz, &b->h, &b->dma, &b->off))
		return -1;
	b->map = rocket_mmap_bo(fd, b->off, b->sz);
	return b->map == MAP_FAILED ? -1 : 0;
}

/* prep(CPU) -> zero -> copy -> fini(device); prep uses INT64_MAX = block. */
static void bo_write(int fd, struct rkt_bo *b, const void *src, unsigned n)
{
	rocket_prep_bo(fd, b->h, INT64_MAX);
	if (src)
		memcpy(b->map, src, n);
	else
		memset(b->map, 0, n);
	rocket_fini_bo(fd, b->h);
}

/* munmap + GEM_CLOSE a BO (safe on a zero-initialised struct). GEM_CLOSE alone
 * leaks the mapping — the VMA keeps the object pinned until process exit. */
static void bo_free(int fd, struct rkt_bo *b)
{
	if (b->map && b->map != MAP_FAILED)
		rocket_munmap_bo(b->map, b->sz);
	if (b->h)
		rocket_close_bo(fd, b->h);
	b->map = NULL;
	b->h = 0;
}

int rkt_npu_matmul(int fd, const uint8_t *X, const uint8_t *Wc,
		   const int32_t *bias, uint32_t M, uint32_t N, uint32_t K,
		   uint8_t izp, uint8_t wzp, uint8_t ozp,
		   float in_scale, float wt_scale, float out_scale, const float *out_scales,
		   uint32_t tile_m, uint32_t tile_n, uint8_t *Y)
{
	int ret = -1;
	int32_t *bias0 = NULL;
#define RKT_MAXTASKS 32
	struct rkt_bo w = {0}, b = {0};
	struct rkt_bo rin[RKT_MAXTASKS] = {{0}}, ro[RKT_MAXTASKS] = {{0}}, rreg[RKT_MAXTASKS] = {{0}};
	uint8_t *got = NULL;
	if (!bias) {
		bias0 = calloc(N, sizeof(int32_t));
		if (!bias0)
			return -1;
		bias = bias0;
	}

	unsigned max_isz = rkt_raw_input_size(1, tile_m, K);
	unsigned max_wsz = rkt_packed_weights_size(1, 1, K, tile_n);
	uint8_t *Xt = malloc((size_t)tile_m * K);
	uint8_t *Wt = malloc((size_t)tile_n * K);
	uint8_t *ipk = calloc(1, max_isz);
	uint8_t *wpk = calloc(1, max_wsz);
	int32_t *bpk = malloc((size_t)tile_n * sizeof(int32_t));
	struct rkt_gemm_tile *tiles = malloc(
		((M + tile_m - 1) / tile_m) * ((N + tile_n - 1) / tile_n) *
		sizeof(struct rkt_gemm_tile));
	uint64_t *rc = malloc(4096 * sizeof(uint64_t));
	if (!Xt || !Wt || !ipk || !wpk || !bpk || !tiles || !rc)
		goto out;

	int nt = rkt_gemm_plan(M, N, tile_m, tile_n, tiles,
			       (int)(((M + tile_m - 1) / tile_m) *
				     ((N + tile_n - 1) / tile_n)));
	if (nt <= 0)
		goto out;

	/* CBUF weight-reuse: chain a column's row tiles into ONE multi-task job so
	 * only task 0 fetches the column's weights DDR->CBUF and tasks 1..k reuse
	 * the SRAM-resident weights (CNA_CBUF_CON0_WEIGHT_REUSE). ~4x fewer weight
	 * DMAs; tiles are column-major so a column's row tiles are contiguous. */
	if (bo_alloc(fd, &w, rkt_packed_weights_size(1, 1, K, tile_n)) ||
	    bo_alloc(fd, &b, tile_n * sizeof(int32_t)))
		goto out;
	for (int s2 = 0; s2 < RKT_MAXTASKS; s2++) {
		if (bo_alloc(fd, &rin[s2],  rkt_raw_input_size(1, tile_m, K)) ||
		    bo_alloc(fd, &ro[s2],   rkt_raw_output_size(1, tile_m, tile_n)) ||
		    bo_alloc(fd, &rreg[s2], 0x1000))
			goto out;
	}
	got = malloc((size_t)tile_m * tile_n);
	if (!got)
		goto out;

	{
	int t = 0;
	while (t < nt) {
		unsigned cc = tiles[t].col, n = tiles[t].n;
		uint64_t _t = rkt_prof_on()?rkt_now_ns():0;

		for (unsigned j = 0; j < n; j++)
			memcpy(Wt + (size_t)j * K, Wc + (size_t)(cc + j) * K, K);
		rkt_pack_weights(Wt, 1, 1, K, n, wzp, wpk);
		bo_write(fd, &w, wpk, rkt_packed_weights_size(1, 1, K, n));
		rkt_compute_biases(Wt, bias + cc, 1, 1, K, n, wzp, izp, bpk);
		bo_write(fd, &b, bpk, n * sizeof(int32_t));

		/* chunk this column's row tiles into multi-task jobs of <= MAXTASKS */
		while (t < nt && tiles[t].col == cc) {
			struct drm_rocket_task tasks[RKT_MAXTASKS];
			uint32_t ih[2 + 2*RKT_MAXTASKS], oh[RKT_MAXTASKS];
			struct { unsigned r, m, n; } meta[RKT_MAXTASKS];
			int cnt = 0, ni = 0, no = 0;
			ih[ni++] = w.h; ih[ni++] = b.h;

			while (t < nt && tiles[t].col == cc && cnt < RKT_MAXTASKS) {
				unsigned r = tiles[t].row, m = tiles[t].m;
				for (unsigned y = 0; y < m; y++)
					memcpy(Xt + (size_t)y * K, X + (size_t)(r + y) * K, K);
				rkt_pack_input(Xt, 1, m, K, izp, ipk);
				bo_write(fd, &rin[cnt], ipk, rkt_raw_input_size(1, m, K));
				bo_write(fd, &ro[cnt], NULL, rkt_raw_output_size(1, m, n));

				rkt_g_task_num = cnt;   /* task 0 loads weights, 1+ reuse CBUF */
				int nw = rkt_build_matmul_regcmd_scaled(
					rc, 4096, m, n, K, rin[cnt].dma, w.dma, ro[cnt].dma,
					izp, wzp, ozp, in_scale, wt_scale,
					out_scales ? out_scales[cc] : out_scale, b.dma);
				if (nw < 0) { rkt_g_task_num = 0; goto out; }
				bo_write(fd, &rreg[cnt], rc, (unsigned)nw * sizeof(uint64_t));

				tasks[cnt].regcmd = (uint32_t)rreg[cnt].dma;
				tasks[cnt].regcmd_count = (uint32_t)nw;
				ih[ni++] = rin[cnt].h; ih[ni++] = rreg[cnt].h; oh[no++] = ro[cnt].h;
				meta[cnt].r = r; meta[cnt].m = m; meta[cnt].n = n;
				cnt++; t++;
			}
			rkt_g_task_num = 0;

			struct drm_rocket_job job;
			memset(&job, 0, sizeof job);
			job.tasks = (uintptr_t)tasks;
			job.task_count = (uint32_t)cnt;
			job.task_struct_size = sizeof(struct drm_rocket_task);
			job.in_bo_handles = (uintptr_t)ih;
			job.in_bo_handle_count = (uint32_t)ni;
			job.out_bo_handles = (uintptr_t)oh;
			job.out_bo_handle_count = (uint32_t)no;

			if (rkt_prof_on()){g_setup_ns+=rkt_now_ns()-_t; _t=rkt_now_ns();}
			int sr = rocket_submit(fd, &job, 1);
			if (rkt_prof_on()){g_submit_ns+=rkt_now_ns()-_t; _t=rkt_now_ns();}
			if (sr) goto out;
			for (int i = 0; i < cnt; i++) {
				if (rocket_prep_bo(fd, ro[i].h, rkt_deadline_ns(6000000000LL)))
					goto out;
				rkt_unpack_output(ro[i].map, 1, meta[i].m, meta[i].n, got);
				for (unsigned y = 0; y < meta[i].m; y++)
					memcpy(Y + (size_t)(meta[i].r + y) * N + cc,
					       got + (size_t)y * meta[i].n, meta[i].n);
			}
			if (rkt_prof_on()){g_prep_ns+=rkt_now_ns()-_t; g_tiles += cnt;}
		}
	}
	}
	if (rkt_prof_on()) g_calls++;
	ret = 0;
out:
	bo_free(fd, &w); bo_free(fd, &b);
	for (int s2 = 0; s2 < RKT_MAXTASKS; s2++) {
		bo_free(fd, &rin[s2]); bo_free(fd, &ro[s2]); bo_free(fd, &rreg[s2]);
	}
	free(got);
	free(Xt); free(Wt); free(ipk); free(wpk); free(bpk);
	free(tiles); free(rc); free(bias0);
	return ret;
}
