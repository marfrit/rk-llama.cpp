#include "rkt_matmul_coredpu.h"
#include "coredpu_defs.h"
#include <stdint.h>

/* Packed register command: (target<<48) | (value<<16) | reg. target = block_enum + 1.
   CORE=0x800->0x801, DPU=0x1000->0x1001, PC=0x100->0x101. */
static int rkt_push(uint64_t *out, int *n, int cap, uint32_t target,
                    uint32_t reg, uint32_t value)
{
    if (*n >= cap)
        return -1;
    out[*n] = ((uint64_t)target << 48) | ((uint64_t)value << 16) | (uint64_t)reg;
    (*n)++;
    return 0;
}

#define EMIT_CORE(reg, val) do { if (rkt_push(out, &n, cap, 0x801,  (reg), (val)) < 0) return -1; } while (0)
#define EMIT_DPU(reg, val)  do { if (rkt_push(out, &n, cap, 0x1001, (reg), (val)) < 0) return -1; } while (0)
#define EMIT_RDMA(reg, val) do { if (rkt_push(out, &n, cap, 0x2001, (reg), (val)) < 0) return -1; } while (0)
#define EMIT_PC(reg, val)   do { if (rkt_push(out, &n, cap, 0x101,  (reg), (val)) < 0) return -1; } while (0)
#define EMIT_RAW(tgt, reg, val) do { if (rkt_push(out, &n, cap, (tgt), (reg), (val)) < 0) return -1; } while (0)

#define F(v, name) (((uint32_t)(v) << name##__SHIFT) & name##__MASK)

/* float -> raw IEEE-754 bits (Mesa fui()) */
static uint32_t fui(float f) { uint32_t u; __builtin_memcpy(&u, &f, sizeof u); return u; }
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define ATOMIC_K_SIZE 16
/* raw 64-bit command append (Mesa util_dynarray_append_typed(regs, uint64_t, v)) */
#define RAW64(v) do { if (n >= cap) return -1; out[n++] = (uint64_t)(v); } while (0)

int rkt_emit_coredpu(uint64_t *out, int cap, const struct coredpu_params *p)
{
    int n = 0;
    int num_tasks = 1;                                   /* single-task port */
    int offset = (int)p->output_zero_point - 0x80;       /* Mesa CNA local */

    /* ==== CORE + DPU + PC emit (filled by slice) ==== */
    uint32_t misc_cfg = F(1, CORE_MISC_CFG_QD_EN);
    if (p->depthwise)
        misc_cfg |= F(1, CORE_MISC_CFG_DW_EN);
    EMIT_CORE(REG_CORE_MISC_CFG, misc_cfg);
    EMIT_CORE(REG_CORE_DATAOUT_SIZE_0,
    F(p->output_height - 1, CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT) |
    F(p->output_width - 1, CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH));
    EMIT_CORE(REG_CORE_DATAOUT_SIZE_1,
    F(p->output_channels - 1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL));
    EMIT_CORE(REG_CORE_CLIP_TRUNCATE,
    F(p->truncate_bits, CORE_CLIP_TRUNCATE_CLIP_TRUNCATE));
    EMIT_RAW(0x801, 0x3030, 0);

    uint32_t feat_mode_cfg =
    F(15, DPU_FEATURE_MODE_CFG_BURST_LEN) | F(2, DPU_FEATURE_MODE_CFG_OUTPUT_MODE);
    if (p->depthwise)
    feat_mode_cfg |= F(3, DPU_FEATURE_MODE_CFG_CONV_MODE);

    EMIT_DPU(REG_DPU_FEATURE_MODE_CFG, feat_mode_cfg);
    EMIT_DPU(REG_DPU_DATA_FORMAT, 0);
    EMIT_DPU(REG_DPU_OFFSET_PEND, 0);
    EMIT_DPU(REG_DPU_DST_BASE_ADDR,
    p->output_addr +
    p->output_offset);
    EMIT_DPU(REG_DPU_DST_SURF_STRIDE,
    F(p->output_surface_stride, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE));
    EMIT_DPU(REG_DPU_DATA_CUBE_WIDTH,
    F(p->output_width - 1, DPU_DATA_CUBE_WIDTH_WIDTH));
    EMIT_DPU(REG_DPU_DATA_CUBE_HEIGHT,
    F(p->output_height - 1, DPU_DATA_CUBE_HEIGHT_HEIGHT));
    EMIT_DPU(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0);
    EMIT_DPU(REG_DPU_DATA_CUBE_CHANNEL,
    F(p->output_channels_real - 1, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL) |
    F(p->output_channels - 1, DPU_DATA_CUBE_CHANNEL_CHANNEL));
    EMIT_DPU(REG_DPU_BS_CFG, F(2, DPU_BS_CFG_BS_ALU_ALGO) | F(1, DPU_BS_CFG_BS_ALU_SRC) |
    F(1, DPU_BS_CFG_BS_RELU_BYPASS) |
    F(1, DPU_BS_CFG_BS_MUL_BYPASS));
    EMIT_DPU(REG_DPU_BS_ALU_CFG, 0);
    EMIT_DPU(REG_DPU_BS_MUL_CFG, 0);
    EMIT_DPU(REG_DPU_BS_RELUX_CMP_VALUE, 0);

    if (p->depthwise) {
    EMIT_DPU(REG_DPU_BS_OW_CFG, F(3, DPU_BS_OW_CFG_SIZE_E_2) |
    F(3, DPU_BS_OW_CFG_SIZE_E_1) |
    F(3, DPU_BS_OW_CFG_SIZE_E_0));
    } else {
    EMIT_DPU(REG_DPU_BS_OW_CFG, F(1, DPU_BS_OW_CFG_SIZE_E_2) |
    F(1, DPU_BS_OW_CFG_SIZE_E_1) |
    F(1, DPU_BS_OW_CFG_SIZE_E_0));
    }

    EMIT_DPU(REG_DPU_BS_OW_OP, F(0x80 - p->weights_zero_point, DPU_BS_OW_OP_OW_OP));

    EMIT_DPU(REG_DPU_WDMA_SIZE_0,
    F(p->output_channels - 1, DPU_WDMA_SIZE_0_CHANNEL_WDMA));
    EMIT_DPU(REG_DPU_WDMA_SIZE_1,
    F(p->output_height - 1, DPU_WDMA_SIZE_1_HEIGHT_WDMA) |
    F(p->output_width - 1, DPU_WDMA_SIZE_1_WIDTH_WDMA));
    EMIT_DPU(REG_DPU_BN_CFG,
    F(1, DPU_BN_CFG_BN_RELU_BYPASS) | F(1, DPU_BN_CFG_BN_MUL_BYPASS) |
    F(1, DPU_BN_CFG_BN_ALU_BYPASS) | F(1, DPU_BN_CFG_BN_BYPASS));
    EMIT_DPU(REG_DPU_BN_ALU_CFG, 0);
    EMIT_DPU(REG_DPU_BN_MUL_CFG, 0);
    EMIT_DPU(REG_DPU_BN_RELUX_CMP_VALUE, 0);

    if (p->add_tensor != -1) {
    EMIT_DPU(REG_DPU_EW_CFG,
    F(1, DPU_EW_CFG_EW_CVT_TYPE) | F(1, DPU_EW_CFG_EW_DATA_MODE) |
    F(1, DPU_EW_CFG_EDATA_SIZE) | F(2, DPU_EW_CFG_EW_ALU_ALGO) |
    F(1, DPU_EW_CFG_EW_RELU_BYPASS) | F(1, DPU_EW_CFG_EW_LUT_BYPASS) |
    F(1, DPU_EW_CFG_EW_OP_SRC));

    /* See http://nvdla.org/hw/v1/ias/precision.html#element-wise */
    EMIT_DPU(REG_DPU_EW_CVT_OFFSET_VALUE, p->addition_offset);

    float add_scale =
    p->addition_scale / (p->input_scale * p->weights_scale);

    uint32_t add_scale_bits = fui(add_scale);
    /* Taken from
    * https://github.com/pytorch/QNNPACK/blob/master/src/qnnpack/requantization.h#L130
    */
    unsigned add_shift = 127 + 31 - 32 - (add_scale_bits >> 23) + 16;

    unsigned scale = ((add_scale_bits >> 9) & 0x7fff);
    if (scale < 1 << 14)
    scale |= 1 << 14;

    EMIT_DPU(REG_DPU_EW_CVT_SCALE_VALUE,
    F(add_shift - 1, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT) |
    F(scale, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE));

    EMIT_DPU(REG_DPU_EW_RELUX_CMP_VALUE, 0x0);

    float out_conv_scale =
    (p->input_scale * p->weights_scale) / p->output_scale;
    uint32_t out_scale_bits = fui(out_conv_scale);
    unsigned out_shift = 127 + 31 - 32 - (out_scale_bits >> 23) + 16;
    if (p->truncate_bits > 0)
    out_shift--;
    unsigned out_scale = ((out_scale_bits >> 9) & 0x7fff) + 1;
    if (out_scale < 1 << 14)
    out_scale |= 1 << 14;

    EMIT_DPU(REG_DPU_OUT_CVT_OFFSET, offset);
    EMIT_DPU(REG_DPU_OUT_CVT_SCALE, F(out_scale, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE));
    EMIT_DPU(REG_DPU_OUT_CVT_SHIFT, F(out_shift - 1, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT));
    } else {
    EMIT_DPU(REG_DPU_EW_CFG,
    F(1, DPU_EW_CFG_EW_RELU_BYPASS) | F(1, DPU_EW_CFG_EW_OP_CVT_BYPASS) |
    F(1, DPU_EW_CFG_EW_LUT_BYPASS) | F(1, DPU_EW_CFG_EW_OP_BYPASS) |
    F(1, DPU_EW_CFG_EW_BYPASS));
    EMIT_DPU(REG_DPU_EW_CVT_OFFSET_VALUE, 0);
    EMIT_DPU(REG_DPU_EW_CVT_SCALE_VALUE, F(1, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE));
    EMIT_DPU(REG_DPU_EW_RELUX_CMP_VALUE, 0);
    EMIT_DPU(REG_DPU_OUT_CVT_OFFSET, offset);

    float conv_scale =
    (p->input_scale * p->weights_scale) / p->output_scale;
    // DBG("conv_scale %f\n", conv_scale);
    uint32_t scale_bits = fui(conv_scale);
    /* Taken from
    * https://github.com/pytorch/QNNPACK/blob/master/src/qnnpack/requantization.h#L130
    */
    unsigned shift = 127 + 31 - 32 - (scale_bits >> 23) + 16;

    if (p->truncate_bits > 0)
    shift--;

    unsigned scale = ((scale_bits >> 9) & 0x7fff) + 1;
    if (scale < 1 << 14)
    scale |= 1 << 14;

    EMIT_DPU(REG_DPU_OUT_CVT_SCALE, F(scale, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE));
    EMIT_DPU(REG_DPU_OUT_CVT_SHIFT, F(shift - 1, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT));
    }

    EMIT_DPU(REG_DPU_EW_OP_VALUE_0, 0);
    EMIT_DPU(REG_DPU_EW_OP_VALUE_1, 0);
    EMIT_DPU(REG_DPU_EW_OP_VALUE_2, 0);
    EMIT_DPU(REG_DPU_EW_OP_VALUE_3, 0);
    EMIT_DPU(REG_DPU_EW_OP_VALUE_4, 0);
    EMIT_DPU(REG_DPU_EW_OP_VALUE_5, 0);
    EMIT_DPU(REG_DPU_EW_OP_VALUE_6, 0);
    EMIT_DPU(REG_DPU_EW_OP_VALUE_7, 0);
    EMIT_DPU(REG_DPU_SURFACE_ADD, F(p->surfaces_per_row, DPU_SURFACE_ADD_SURF_ADD));
    EMIT_RAW(0x1001, 0x40c4, 0);
    EMIT_DPU(REG_DPU_LUT_ACCESS_CFG, 0);
    EMIT_DPU(REG_DPU_LUT_ACCESS_DATA, 0);
    EMIT_DPU(REG_DPU_LUT_CFG, 0);
    EMIT_DPU(REG_DPU_LUT_INFO, 0);
    EMIT_DPU(REG_DPU_LUT_LE_START, 0);
    EMIT_DPU(REG_DPU_LUT_LE_END, 0);
    EMIT_DPU(REG_DPU_LUT_LO_START, 0);
    EMIT_DPU(REG_DPU_LUT_LO_END, 0);
    EMIT_DPU(REG_DPU_LUT_LE_SLOPE_SCALE, 0);
    EMIT_DPU(REG_DPU_LUT_LE_SLOPE_SHIFT, 0);
    EMIT_DPU(REG_DPU_LUT_LO_SLOPE_SCALE, 0);
    EMIT_DPU(REG_DPU_LUT_LO_SLOPE_SHIFT, 0);
    EMIT_RDMA(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
    F(p->output_width - 1, DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH));
    EMIT_RDMA(REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
    F(p->output_height - 1, DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT));
    EMIT_RDMA(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
    F(p->output_channels - 1, DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL));

    if (p->add_tensor != -1) {
    EMIT_RDMA(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
    p->add_tensor_addr +
    p->output_offset);
    } else {
    EMIT_RDMA(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0);
    }

    EMIT_RDMA(REG_DPU_RDMA_RDMA_BRDMA_CFG, F(1, DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE));
    EMIT_RDMA(REG_DPU_RDMA_RDMA_BS_BASE_ADDR,
    p->biases_addr);
    EMIT_RDMA(REG_DPU_RDMA_RDMA_NRDMA_CFG, 0);
    EMIT_RDMA(REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0);

    unsigned ew_stride =
    MAX2(p->output_width * p->output_height, 12);

    if (p->add_tensor != -1) {
    EMIT_RDMA(REG_DPU_RDMA_RDMA_ERDMA_CFG,
    F(1, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE) |
    F(1, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE));
    unsigned ew_base_offset =
    p->output_width * p->output_height * ATOMIC_K_SIZE;
    EMIT_RDMA(REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
    p->add_tensor_addr +
    p->output_offset + ew_base_offset);
    EMIT_RDMA(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
    F(ew_stride, DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE));
    } else {
    EMIT_RDMA(REG_DPU_RDMA_RDMA_ERDMA_CFG, F(1, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE));
    EMIT_RDMA(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0);
    EMIT_RDMA(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 0);
    }

    uint32_t rdma_feat_mode_cfg = 0x0;

    if (p->add_tensor != -1) {
    rdma_feat_mode_cfg |= F(15, DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN) |
    F(5, DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE);
    } else {
    rdma_feat_mode_cfg |= F(15, DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN) |
    F(1, DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE);
    }

    if (p->depthwise)
    rdma_feat_mode_cfg |= F(3, DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE);

    EMIT_RDMA(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feat_mode_cfg);
    EMIT_RDMA(REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0);

    unsigned surf_notch =
    ew_stride +
    p->output_width * (p->output_height - p->output_height);

    if (p->input_width == 3) {
    surf_notch = 15;
    }

    if (p->add_tensor != -1) {
    EMIT_RDMA(REG_DPU_RDMA_RDMA_SURF_NOTCH,
    F(surf_notch, DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR));
    } else {
    EMIT_RDMA(REG_DPU_RDMA_RDMA_SURF_NOTCH, 0);
    }

    EMIT_RDMA(REG_DPU_RDMA_RDMA_PAD_CFG, 0);
    EMIT_RDMA(REG_DPU_RDMA_RDMA_WEIGHT,
    F(1, DPU_RDMA_RDMA_WEIGHT_E_WEIGHT) | F(1, DPU_RDMA_RDMA_WEIGHT_N_WEIGHT) |
    F(1, DPU_RDMA_RDMA_WEIGHT_B_WEIGHT) | F(1, DPU_RDMA_RDMA_WEIGHT_M_WEIGHT));

    if (p->add_tensor != -1) {
    EMIT_RDMA(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH,
    F(surf_notch, DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH));
    } else {
    EMIT_RDMA(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, 0x0);
    }

    if (num_tasks == 1)
    RAW64(0x0);
    else
    EMIT_PC(REG_PC_BASE_ADDRESS, 0);

    EMIT_PC(REG_PC_REGISTER_AMOUNTS, 0);

    /* TRM: before op_en, 64'h0041_xxxx_xxxx_xxxx must be set. */
    RAW64(0x0041000000000000);

    /* TRM: 64'h0081_0000_007f_0008 will set each block's op_en(CNA, CORE, ...,
    * PPU_RDMA). */
    EMIT_RAW(0x81, REG_PC_OPERATION_ENABLE,
    F(14, PC_OPERATION_ENABLE_RESERVED_0) | F(1, PC_OPERATION_ENABLE_OP_EN));

    return n;
}
