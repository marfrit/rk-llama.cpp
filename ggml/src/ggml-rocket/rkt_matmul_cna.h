#ifndef RKT_MATMUL_CNA_H
#define RKT_MATMUL_CNA_H

#include <stdint.h>

/* All the task-/operation-level fields the Mesa CNA emit reads, flattened. */
struct cna_params {
    uint32_t input_width, input_height, input_channels, input_channels_real;
    uint32_t output_width, atomic_count;
    uint32_t weights_width, weights_height, weights_kernels;
    uint32_t weights_banks, input_banks, input_data_entries;
    uint32_t input_line_stride, input_surface_stride;
    uint64_t input_dma, weights_dma;   /* NPU DMA addresses of the input/weight BOs */
    uint32_t stride_x, stride_y;
    uint32_t pad_left, pad_right, pad_top, pad_bottom;
    int32_t  output_zero_point, weights_zero_point, input_zero_point;
    int      depthwise, reuse_weights_cbuf, addition_input, add_tensor;
    unsigned task_num;   /* 0 for the first/only task */
};

/* Append the CNA-stage register commands (packed uint64_t) into out[] (capacity cap).
   Returns the number of commands written, or -1 if capacity would be exceeded. */
int rkt_emit_cna(uint64_t *out, int cap, const struct cna_params *p);

#endif /* RKT_MATMUL_CNA_H */
