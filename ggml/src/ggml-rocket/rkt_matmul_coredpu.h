#ifndef RKT_MATMUL_COREDPU_H
#define RKT_MATMUL_COREDPU_H

#include <stdint.h>

/* Fields the Mesa CORE/DPU/PC emit reads (task-/operation-level flattened). */
struct coredpu_params {
    uint32_t input_width;
    uint32_t output_width, output_height;
    uint32_t output_channels, output_channels_real;
    uint32_t output_offset, output_surface_stride, surfaces_per_row;
    uint32_t truncate_bits;
    int32_t  output_zero_point, weights_zero_point;
    int      depthwise;
    int32_t  addition_offset;
    int32_t  add_tensor;              /* index, or -1 if none */
    float    input_scale, weights_scale, output_scale, addition_scale;
    uint64_t output_addr, add_tensor_addr, biases_addr; /* pre-computed DMA addresses */
};

/* Append CORE+DPU+PC register commands (packed uint64_t) into out[] (capacity cap).
   Returns number written, or -1 on overflow. */
int rkt_emit_coredpu(uint64_t *out, int cap, const struct coredpu_params *p);

#endif /* RKT_MATMUL_COREDPU_H */
