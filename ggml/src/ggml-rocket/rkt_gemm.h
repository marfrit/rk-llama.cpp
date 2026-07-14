/* SPDX-License-Identifier: MIT
 *
 * rkt_gemm.h — U6 backend: tile a large INT8 GEMM into rocket-op-sized
 * sub-matmuls, each buildable by rkt_build_matmul_regcmd.
 *
 * A single rocket op has a fixed CBUF budget (12 banks), so a gemma-scale
 * GEMM must be blocked. This layer tiles the M (rows/tokens) and N (output
 * channels) dimensions. K (contraction / input channels) is NOT tiled here:
 * the full K must fit one op — K-splitting needs partial-sum accumulation via
 * the DPU add path and is deferred (see rkt_matmul_coredpu add_tensor path).
 */
#ifndef RKT_GEMM_H
#define RKT_GEMM_H

#include <stdint.h>

struct rkt_gemm_tile {
	uint32_t row; /* first row in M this tile covers */
	uint32_t col; /* first col in N this tile covers */
	uint32_t m;   /* number of rows */
	uint32_t n;   /* number of cols */
};

/*
 * Partition an M x N output (full K) into tiles of at most tile_m x tile_n,
 * row-major. Writes tiles[] and returns the tile count, or -1 on bad args or
 * if more than max_tiles would be produced. Pure geometry — call
 * rkt_gemm_op_fits() to confirm a tile actually fits one rocket op.
 */
int rkt_gemm_plan(uint32_t M, uint32_t N, uint32_t tile_m, uint32_t tile_n,
		  uint32_t col_start, uint32_t col_num,
		  struct rkt_gemm_tile *tiles, int max_tiles);

/* True (1) if an m x n, full-k matmul fits a single rocket op. */
int rkt_gemm_op_fits(uint32_t m, uint32_t n, uint32_t k);

#endif /* RKT_GEMM_H */
