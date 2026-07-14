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
	memset(b->map, 0, b->sz);
	if (src)
		memcpy(b->map, src, n);
	rocket_fini_bo(fd, b->h);
}

int rkt_npu_matmul(int fd, const uint8_t *X, const uint8_t *Wc,
		   const int32_t *bias, uint32_t M, uint32_t N, uint32_t K,
		   uint8_t izp, uint8_t wzp, uint8_t ozp,
		   float in_scale, float wt_scale, float out_scale, const float *out_scales,
		   uint32_t tile_m, uint32_t tile_n, uint8_t *Y)
{
	int ret = -1;
	int32_t *bias0 = NULL;
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

	for (int t = 0; t < nt; t++) {
		unsigned r = tiles[t].row, cc = tiles[t].col;
		unsigned m = tiles[t].m, n = tiles[t].n;

		for (unsigned y = 0; y < m; y++)
			memcpy(Xt + (size_t)y * K, X + (size_t)(r + y) * K, K);
		for (unsigned j = 0; j < n; j++)
			memcpy(Wt + (size_t)j * K, Wc + (size_t)(cc + j) * K, K);
		rkt_pack_input(Xt, 1, m, K, izp, ipk);
		rkt_pack_weights(Wt, 1, 1, K, n, wzp, wpk);
		rkt_compute_biases(Wt, bias + cc, 1, 1, K, n, wzp, izp, bpk);

		struct rkt_bo in = {0}, w = {0}, b = {0}, o = {0}, reg = {0};
		if (bo_alloc(fd, &in, rkt_raw_input_size(1, m, K)) ||
		    bo_alloc(fd, &w, rkt_packed_weights_size(1, 1, K, n)) ||
		    bo_alloc(fd, &b, n * sizeof(int32_t)) ||
		    bo_alloc(fd, &o, rkt_raw_output_size(1, m, n)) ||
		    bo_alloc(fd, &reg, 0x1000))
			goto out;
		bo_write(fd, &in, ipk, rkt_raw_input_size(1, m, K));
		bo_write(fd, &w, wpk, rkt_packed_weights_size(1, 1, K, n));
		bo_write(fd, &b, bpk, n * sizeof(int32_t));
		bo_write(fd, &o, NULL, 0);

		int nw = rkt_build_matmul_regcmd_scaled(
			rc, 4096, m, n, K, in.dma, w.dma, o.dma, izp, wzp, ozp,
			in_scale, wt_scale, out_scales ? out_scales[cc] : out_scale, b.dma);
		if (nw < 0) {
			rocket_close_bo(fd, in.h); rocket_close_bo(fd, w.h);
			rocket_close_bo(fd, b.h); rocket_close_bo(fd, o.h);
			rocket_close_bo(fd, reg.h);
			goto out;
		}
		bo_write(fd, &reg, rc, (unsigned)nw * sizeof(uint64_t));

		struct drm_rocket_task task = { .regcmd = (uint32_t)reg.dma,
						.regcmd_count = (uint32_t)nw };
		uint32_t ih[] = { in.h, w.h, b.h, reg.h }, oh[] = { o.h };
		struct drm_rocket_job job;
		memset(&job, 0, sizeof job);
		job.tasks = (uintptr_t)&task;
		job.task_count = 1;
		job.task_struct_size = sizeof(struct drm_rocket_task);
		job.in_bo_handles = (uintptr_t)ih;
		job.in_bo_handle_count = 4;
		job.out_bo_handles = (uintptr_t)oh;
		job.out_bo_handle_count = 1;

		int sr = rocket_submit(fd, &job, 1);
		int wr = sr ? -1 : rocket_prep_bo(fd, o.h, rkt_deadline_ns(6000000000LL));
		if (!sr && !wr) {
			uint8_t *got = malloc((size_t)m * n);
			if (got) {
				rkt_unpack_output(o.map, 1, m, n, got);
				for (unsigned y = 0; y < m; y++)
					memcpy(Y + (size_t)(r + y) * N + cc,
					       got + (size_t)y * n, n);
				free(got);
			}
		}
		rocket_close_bo(fd, in.h); rocket_close_bo(fd, w.h);
		rocket_close_bo(fd, b.h); rocket_close_bo(fd, o.h);
		rocket_close_bo(fd, reg.h);
		if (sr || wr)
			goto out;
	}
	ret = 0;
out:
	free(Xt); free(Wt); free(ipk); free(wpk); free(bpk);
	free(tiles); free(rc); free(bias0);
	return ret;
}
