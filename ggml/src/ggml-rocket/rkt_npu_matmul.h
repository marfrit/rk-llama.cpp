/* SPDX-License-Identifier: MIT
 *
 * rkt_npu_matmul.h — reusable arbitrary-size INT8 matmul on the rocket NPU.
 *
 * The U6 core primitive: Y[M][N] = X[M][K] * W[K][N] computed by 2D-tiling
 * into single-op rocket matmuls (rkt_gemm_plan + rkt_build_matmul_regcmd +
 * librocket). Intended to be called from a ggml MUL_MAT backend.
 */
#ifndef RKT_NPU_MATMUL_H
#define RKT_NPU_MATMUL_H

#include <stdint.h>

/*
 * fd            open /dev/accel/accel0 (rocket_open).
 * X             [M][K] uint8 row-major activations.
 * Wc            [N][K] uint8 conv-order weights (Wc[out_ch][in_ch]).
 * bias          [N] int32 per-output-channel bias, or NULL for none.
 * izp/wzp/ozp   int8 zero-points (0-255).
 * in/wt/out_scale  per-tensor requant scales.
 * tile_m/tile_n desired tile sizes; each (m x n, full-K) tile must satisfy
 *               rkt_gemm_op_fits (m limited by CBUF input banks, K*n <= ~327k).
 * Y             [M][N] uint8 output (caller-allocated).
 *
 * Returns 0 on success, -1 on error (bad args / tile too big / submit fail).
 * Allocates weights/bias/input/output/regcmd BOs per tile (correctness-first;
 * weight-reuse across M-tiles is a later perf optimization).
 */
int rkt_npu_matmul(int fd, const uint8_t *X, const uint8_t *Wc,
		   const int32_t *bias, uint32_t M, uint32_t N, uint32_t K,
		   uint8_t izp, uint8_t wzp, uint8_t ozp,
		   float in_scale, float wt_scale, float out_scale, const float *out_scales,
		   uint32_t tile_m, uint32_t tile_n,
		   uint32_t col_start, uint32_t col_num, uint8_t *Y);

#endif /* RKT_NPU_MATMUL_H */
