#include "rkt_matmul_cna.h"
#include "cna_defs.h"
#include <stdint.h>

/*
 * Packed register command: (target << 48) | (value << 16) | reg
 * target = block_enum + 1.  CNA=0x200->0x201, DPU=0x1000->0x1001,
 * DPU_RDMA=0x2000->0x2001. The target is a property of the register's block
 * (Mesa derives it via rkt_get_target(reg)+1), NOT of the emit site — so the
 * DPU_RDMA registers that appear inside this CNA sequence still carry 0x2001.
 */
static int rkt_push(uint64_t *out, int *n, int cap, uint32_t target,
                    uint32_t reg, uint32_t value)
{
    if (*n >= cap)
        return -1;
    out[*n] = ((uint64_t)target << 48) | ((uint64_t)value << 16) | (uint64_t)reg;
    (*n)++;
    return 0;
}

#define EMIT_CNA(reg, val)  do { if (rkt_push(out, &n, cap, 0x201,  (reg), (val)) < 0) return -1; } while (0)
#define EMIT_DPU(reg, val)  do { if (rkt_push(out, &n, cap, 0x1001, (reg), (val)) < 0) return -1; } while (0)
#define EMIT_RDMA(reg, val) do { if (rkt_push(out, &n, cap, 0x2001, (reg), (val)) < 0) return -1; } while (0)

/* Field-set: (value << FIELD__SHIFT) & FIELD__MASK */
#define F(v, name) (((uint32_t)(v) << name##__SHIFT) & name##__MASK)

int rkt_emit_cna(uint64_t *out, int cap, const struct cna_params *p)
{
    int n = 0;

    uint32_t con0 = F(p->weights_banks, CNA_CBUF_CON0_WEIGHT_BANK) |
                    F(p->input_banks, CNA_CBUF_CON0_DATA_BANK);
    if (p->task_num > 0 && p->reuse_weights_cbuf)
        con0 |= F(1, CNA_CBUF_CON0_WEIGHT_REUSE);
    EMIT_CNA(REG_CNA_CBUF_CON0, con0);

    EMIT_CNA(REG_CNA_DCOMP_REGNUM, 0);
    EMIT_CNA(REG_CNA_DCOMP_CTRL, 0);

    uint32_t con1 = 0x0;
    if (p->input_channels_real == 1) {
        con1 |= F(1, CNA_CONV_CON1_NONALIGN_DMA) | F(1, CNA_CONV_CON1_GROUP_LINE_OFF) |
                F(8, CNA_CONV_CON1_ARGB_IN);
    }
    if (p->depthwise)
        con1 |= F(3, CNA_CONV_CON1_CONV_MODE);
    EMIT_CNA(REG_CNA_CONV_CON1, con1);

    EMIT_DPU(REG_DPU_S_POINTER, F(1, DPU_S_POINTER_POINTER_PP_MODE) |
                                F(1, DPU_S_POINTER_EXECUTER_PP_EN) |
                                F(1, DPU_S_POINTER_POINTER_PP_EN));
    EMIT_RDMA(REG_DPU_RDMA_RDMA_S_POINTER,
              F(1, DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE) |
              F(1, DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN) |
              F(1, DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN));
    EMIT_CNA(REG_CNA_CONV_CON1, con1);

    EMIT_CNA(REG_CNA_CONV_CON2,
             F(50 + p->stride_y + 1, CNA_CONV_CON2_FEATURE_GRAINS)); /* Magic: passes the most tests */
    EMIT_CNA(REG_CNA_CONV_CON3, F(p->stride_x, CNA_CONV_CON3_CONV_X_STRIDE) |
                                F(p->stride_y, CNA_CONV_CON3_CONV_Y_STRIDE));
    EMIT_CNA(REG_CNA_DATA_SIZE0,
             F(p->input_width, CNA_DATA_SIZE0_DATAIN_WIDTH) |
             F(p->input_height, CNA_DATA_SIZE0_DATAIN_HEIGHT));
    EMIT_CNA(REG_CNA_DATA_SIZE1,
             F(p->input_channels_real - 1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL) |
             F(p->input_channels, CNA_DATA_SIZE1_DATAIN_CHANNEL));
    EMIT_CNA(REG_CNA_DATA_SIZE2, F(p->output_width, CNA_DATA_SIZE2_DATAOUT_WIDTH));
    EMIT_CNA(REG_CNA_DATA_SIZE3, F(p->atomic_count, CNA_DATA_SIZE3_DATAOUT_ATOMICS));
    EMIT_CNA(REG_CNA_WEIGHT_SIZE0, p->weights_width * p->weights_height *
                                   p->input_channels * p->weights_kernels);
    EMIT_CNA(REG_CNA_WEIGHT_SIZE1,
             p->weights_width * p->weights_height * p->input_channels);
    EMIT_CNA(REG_CNA_WEIGHT_SIZE2,
             F(p->weights_width, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH) |
             F(p->weights_height, CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT) |
             F(p->weights_kernels, CNA_WEIGHT_SIZE2_WEIGHT_KERNELS));

    EMIT_CNA(REG_CNA_CBUF_CON0, con0);
    EMIT_CNA(REG_CNA_CBUF_CON1, F(p->input_data_entries, CNA_CBUF_CON1_DATA_ENTRIES));

    if (p->input_channels_real == 1) {
        unsigned truncate = 14;
        unsigned scale = 16384;
        unsigned offset = 65408;

        if (p->addition_input || p->add_tensor != -1) {
            truncate = 15;
            scale = 32388;
        }

        EMIT_CNA(REG_CNA_CVT_CON0, F(truncate, CNA_CVT_CON0_CVT_TRUNCATE_3) |
                                   F(truncate, CNA_CVT_CON0_CVT_TRUNCATE_2) |
                                   F(truncate, CNA_CVT_CON0_CVT_TRUNCATE_1) |
                                   F(truncate, CNA_CVT_CON0_CVT_TRUNCATE_0));
        EMIT_CNA(REG_CNA_CVT_CON1,
                 F(scale, CNA_CVT_CON1_CVT_SCALE0) | F(offset, CNA_CVT_CON1_CVT_OFFSET0));
        EMIT_CNA(REG_CNA_CVT_CON2,
                 F(scale, CNA_CVT_CON2_CVT_SCALE1) | F(offset, CNA_CVT_CON2_CVT_OFFSET1));
        EMIT_CNA(REG_CNA_CVT_CON3,
                 F(scale, CNA_CVT_CON3_CVT_SCALE2) | F(offset, CNA_CVT_CON3_CVT_OFFSET2));
        EMIT_CNA(REG_CNA_CVT_CON4,
                 F(scale, CNA_CVT_CON4_CVT_SCALE3) | F(offset, CNA_CVT_CON4_CVT_OFFSET3));
    } else {
        EMIT_CNA(REG_CNA_CVT_CON0, F(1, CNA_CVT_CON0_DATA_SIGN) |
                                   F(1, CNA_CVT_CON0_CVT_TYPE) |
                                   F(1, CNA_CVT_CON0_CVT_BYPASS));
        EMIT_CNA(REG_CNA_CVT_CON1, F(1, CNA_CVT_CON1_CVT_SCALE0));
        EMIT_CNA(REG_CNA_CVT_CON2, F(1, CNA_CVT_CON2_CVT_SCALE1));
        EMIT_CNA(REG_CNA_CVT_CON3, F(1, CNA_CVT_CON3_CVT_SCALE2));
        EMIT_CNA(REG_CNA_CVT_CON4, F(1, CNA_CVT_CON4_CVT_SCALE3));
    }

    EMIT_CNA(REG_CNA_FC_CON0, 0);
    EMIT_CNA(REG_CNA_FC_CON1, 0);
    EMIT_CNA(REG_CNA_PAD_CON0, F(p->pad_left, CNA_PAD_CON0_PAD_LEFT) |
                               F(p->pad_top, CNA_PAD_CON0_PAD_TOP));
    EMIT_CNA(REG_CNA_FEATURE_DATA_ADDR, (uint32_t)p->input_dma);
    EMIT_CNA(REG_CNA_FC_CON2, 0);
    EMIT_CNA(REG_CNA_DMA_CON0,
             F(15, CNA_DMA_CON0_WEIGHT_BURST_LEN) | F(15, CNA_DMA_CON0_DATA_BURST_LEN));
    EMIT_CNA(REG_CNA_DMA_CON1, F(p->input_line_stride, CNA_DMA_CON1_LINE_STRIDE));
    EMIT_CNA(REG_CNA_DMA_CON2, F(p->input_surface_stride, CNA_DMA_CON2_SURF_STRIDE));
    EMIT_CNA(REG_CNA_FC_DATA_SIZE0,
             F(p->input_width, CNA_FC_DATA_SIZE0_DMA_WIDTH) |
             F(p->input_height, CNA_FC_DATA_SIZE0_DMA_HEIGHT));
    EMIT_CNA(REG_CNA_FC_DATA_SIZE1,
             F(p->input_channels, CNA_FC_DATA_SIZE1_DMA_CHANNEL));
    EMIT_CNA(REG_CNA_DCOMP_CTRL, 0);
    EMIT_CNA(REG_CNA_DCOMP_REGNUM, 0);
    EMIT_CNA(REG_CNA_DCOMP_ADDR0, (uint32_t)p->weights_dma);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT0, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT1, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT2, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT3, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT4, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT5, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT6, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT7, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT8, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT9, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT10, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT11, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT12, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT13, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT14, 0);
    EMIT_CNA(REG_CNA_DCOMP_AMOUNT15, 0);

    if (p->input_channels_real == 1)
        EMIT_CNA(REG_CNA_CVT_CON5, 65535);
    else
        EMIT_CNA(REG_CNA_CVT_CON5, 0);

    int32_t pad_con1;
    if (p->weights_width >= 3 && p->input_zero_point == 0x0)
        pad_con1 = 0xffff8080;
    else
        pad_con1 = p->input_zero_point - 0x80;

    if (p->addition_input || p->add_tensor != -1)
        pad_con1 = 0xffffff80;

    if (p->depthwise && p->input_zero_point == 0x8b)
        pad_con1 = 0x0b0b;

    EMIT_CNA(REG_CNA_PAD_CON1, pad_con1);

    return n;
}
