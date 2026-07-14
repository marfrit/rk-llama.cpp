/* SPDX-License-Identifier: MIT
 *
 * rkt_gemm.c — U6 GEMM tiler (see rkt_gemm.h).
 */
#include "rkt_gemm.h"
#include "rkt_matmul.h"

int rkt_gemm_plan(uint32_t M, uint32_t N, uint32_t tile_m, uint32_t tile_n,
		  uint32_t col_start, uint32_t col_num,
		  struct rkt_gemm_tile *tiles, int max_tiles)
{
	if (M == 0 || N == 0 || tile_m == 0 || tile_n == 0)
		return -1;
	uint32_t col_end = col_start + col_num;
	if (col_end > N) col_end = N;

	int t = 0;
	/* column-major over [col_start,col_end): consecutive tiles share the column
	 * tile (weight pack+write cached across a column's row tiles). */
	for (uint32_t c = col_start; c < col_end; c += tile_n) {
		for (uint32_t r = 0; r < M; r += tile_m) {
			if (t >= max_tiles)
				return -1;
			tiles[t].row = r;
			tiles[t].col = c;
			tiles[t].m = (r + tile_m <= M) ? tile_m : (M - r);
			tiles[t].n = (c + tile_n <= col_end) ? tile_n : (col_end - c);
			t++;
		}
	}
	return t;
}

int rkt_gemm_op_fits(uint32_t m, uint32_t n, uint32_t k)
{
	uint64_t scratch[512];
	/* Addresses/zero-points are placeholders — fit is shape-only. */
	return rkt_build_matmul_regcmd(scratch, 512, m, n, k,
				       0x1000, 0x2000, 0x3000,
				       128, 128, 128) >= 0;
}
