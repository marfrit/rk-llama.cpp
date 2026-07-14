/*
 * rkt_matmul.h — Rockchip "rocket" NPU matmul regcmd generator
 *
 * Builds a register-command stream for a single INT8 matmul
 * Y[M][N] = X[M][K] * W[K][N] expressed as a 1×1 convolution.
 *
 * Faithfully adapted from the upstream Mesa "rocket" Gallium driver
 * (rkt_regcmd.c, rkt_task.c, rkt_ml.c, rkt_registers.h).
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RKT_MATMUL_H
#define RKT_MATMUL_H

#include <stdint.h>

/**
 * rkt_build_matmul_regcmd - build a single-task NPU regcmd for an INT8 matmul
 *
 * The matmul is emitted as a 1×1 convolution on the NPU:
 *   input_channels = K
 *   kernels         = N
 *   height          = M  (conv spatial rows)
 *   width           = 1  (conv spatial columns)
 *
 * @param out           caller-provided output buffer
 * @param out_capacity  max uint64_t entries in @out
 * @param M             number of rows of X (and rows of Y)
 * @param N             number of columns of W (and columns of Y)
 * @param K             number of columns of X / rows of W
 * @param input_dma     bus address of X[M][K] (must be 256-byte aligned)
 * @param weights_dma   bus address of W[K][N] (must be 256-byte aligned)
 * @param output_dma    bus address of Y[M][N] (must be 256-byte aligned)
 * @param input_zero_point   int8 zero-point for input (0-255, usually 128)
 * @param weight_zero_point  int8 zero-point for weights (0-255, usually 128)
 * @param output_zero_point  int8 zero-point for output (0-255, usually 128)
 *
 * @return number of uint64_t commands written, or -1 if capacity exceeded.
 *
 * NOTE: Rate/scale parameters are not exposed.  This generator assumes
 * input_scale = weights_scale = output_scale = 1.0 (identity requantisation).
 * Callers requiring proper quantised output must patch REG_DPU_OUT_CVT_SCALE
 * and REG_DPU_OUT_CVT_SHIFT after construction, or provide pre-computed
 * scale/shift values via an extended API.
 */
int rkt_build_matmul_regcmd_scaled(uint64_t *out, int out_capacity,
                            uint32_t M, uint32_t N, uint32_t K,
                            uint64_t input_dma, uint64_t weights_dma,
                            uint64_t output_dma,
                            int32_t input_zero_point, int32_t weight_zero_point,
                            int32_t output_zero_point,
                            float input_scale, float weights_scale,
                            float output_scale, uint64_t bias_dma, int task_num);

int rkt_build_matmul_regcmd(uint64_t *out, int out_capacity,
                            uint32_t M, uint32_t N, uint32_t K,
                            uint64_t input_dma, uint64_t weights_dma,
                            uint64_t output_dma,
                            int32_t input_zero_point,
                            int32_t weight_zero_point,
                            int32_t output_zero_point);

#endif /* RKT_MATMUL_H */
