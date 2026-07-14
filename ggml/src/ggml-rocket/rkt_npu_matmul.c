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
	struct rkt_bo in = {0}, w = {0}, b = {0}, o = {0}, reg = {0};
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

	/* Per-call BO pool sized for the largest tile: reuse across every tile.
	 * Thousands of CREATE_BO/mmap/GEM_CLOSE per call were the dominant cost
	 * (NPU compute is a small fraction of per-tile buffer churn). */
	if (bo_alloc(fd, &in,  rkt_raw_input_size(1, tile_m, K)) ||
	    bo_alloc(fd, &w,   rkt_packed_weights_size(1, 1, K, tile_n)) ||
	    bo_alloc(fd, &b,   tile_n * sizeof(int32_t)) ||
	    bo_alloc(fd, &o,   rkt_raw_output_size(1, tile_m, tile_n)) ||
	    bo_alloc(fd, &reg, 0x1000))
		goto out;
	got = malloc((size_t)tile_m * tile_n);
	if (!got)
		goto out;
	{
	uint32_t ih[] = { in.h, w.h, b.h, reg.h }, oh[] = { o.h };
	unsigned last_cc = (unsigned)-1;

	for (int t = 0; t < nt; t++) {
		unsigned r = tiles[t].row, cc = tiles[t].col;
		unsigned m = tiles[t].m, n = tiles[t].n;
		uint64_t _t = rkt_prof_on()?rkt_now_ns():0;

		/* weights + bias depend only on the column tile: pack+write once,
		 * reuse the (device-resident) weight BO across all row tiles. */
		if (cc != last_cc) {
			for (unsigned j = 0; j < n; j++)
				memcpy(Wt + (size_t)j * K, Wc + (size_t)(cc + j) * K, K);
			rkt_pack_weights(Wt, 1, 1, K, n, wzp, wpk);
			bo_write(fd, &w, wpk, rkt_packed_weights_size(1, 1, K, n));
			rkt_compute_biases(Wt, bias + cc, 1, 1, K, n, wzp, izp, bpk);
			bo_write(fd, &b, bpk, n * sizeof(int32_t));
			last_cc = cc;
		}

		for (unsigned y = 0; y < m; y++)
			memcpy(Xt + (size_t)y * K, X + (size_t)(r + y) * K, K);
		rkt_pack_input(Xt, 1, m, K, izp, ipk);
		bo_write(fd, &in, ipk, rkt_raw_input_size(1, m, K));
		bo_write(fd, &o, NULL, rkt_raw_output_size(1, m, n));

		int nw = rkt_build_matmul_regcmd_scaled(
			rc, 4096, m, n, K, in.dma, w.dma, o.dma, izp, wzp, ozp,
			in_scale, wt_scale, out_scales ? out_scales[cc] : out_scale, b.dma);
		if (nw < 0)
			goto out;
		bo_write(fd, &reg, rc, (unsigned)nw * sizeof(uint64_t));

		struct drm_rocket_task task = { .regcmd = (uint32_t)reg.dma,
						.regcmd_count = (uint32_t)nw };
		struct drm_rocket_job job;
		memset(&job, 0, sizeof job);
		job.tasks = (uintptr_t)&task;
		job.task_count = 1;
		job.task_struct_size = sizeof(struct drm_rocket_task);
		job.in_bo_handles = (uintptr_t)ih;
		job.in_bo_handle_count = 4;
		job.out_bo_handles = (uintptr_t)oh;
		job.out_bo_handle_count = 1;

		if(rkt_prof_on()){g_setup_ns+=rkt_now_ns()-_t; _t=rkt_now_ns();}
		int sr = rocket_submit(fd, &job, 1);
		if(rkt_prof_on()){g_submit_ns+=rkt_now_ns()-_t; _t=rkt_now_ns();}
		int wr = sr ? -1 : rocket_prep_bo(fd, o.h, rkt_deadline_ns(6000000000LL));
		if(rkt_prof_on()){g_prep_ns+=rkt_now_ns()-_t; _t=rkt_now_ns();}
		if (!sr && !wr) {
			rkt_unpack_output(o.map, 1, m, n, got);
			for (unsigned y = 0; y < m; y++)
				memcpy(Y + (size_t)(r + y) * N + cc, got + (size_t)y * n, n);
		}
		if(rkt_prof_on()){g_free_ns+=rkt_now_ns()-_t; g_tiles++;}
		if (sr || wr)
			goto out;
	}
	}
	if(rkt_prof_on()) g_calls++;
	ret = 0;
out:
	bo_free(fd, &in); bo_free(fd, &w); bo_free(fd, &b);
	bo_free(fd, &o); bo_free(fd, &reg);
	free(got);
	free(Xt); free(Wt); free(ipk); free(wpk); free(bpk);
	free(tiles); free(rc); free(bias0);
	return ret;
}
