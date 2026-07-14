/* SPDX-License-Identifier: MIT
 *
 * rkt_operands.h — NPU operand layout ports (see rkt_operands.c).
 * Non-depthwise. Row-major host tensors:
 *   weights_in [oc_real][ww][wh][ic_real], input_in [iw][ih][ic_real],
 *   output_out [oh][ow][oc_real].
 */
#ifndef RKT_OPERANDS_H
#define RKT_OPERANDS_H

#include <stdint.h>

unsigned rkt_raw_input_size(unsigned iw, unsigned ih, unsigned ic_real);
unsigned rkt_raw_output_size(unsigned ow, unsigned oh, unsigned oc_real);
unsigned rkt_packed_weights_size(unsigned ww, unsigned wh, unsigned ic_real,
				 unsigned oc_real);

unsigned rkt_pack_weights(const uint8_t *weights_in, unsigned ww, unsigned wh,
			  unsigned ic_real, unsigned oc_real, uint8_t wzp,
			  uint8_t *out);
void rkt_compute_biases(const uint8_t *weights_in, const int32_t *biases_in,
			unsigned ww, unsigned wh, unsigned ic_real,
			unsigned oc_real, uint8_t wzp, uint8_t izp,
			int32_t *biases_out);
unsigned rkt_pack_input(const uint8_t *input_in, unsigned iw, unsigned ih,
			unsigned ic_real, uint8_t izp, uint8_t *out);
void rkt_unpack_output(const uint8_t *raw, unsigned ow, unsigned oh,
		       unsigned oc_real, uint8_t *out);

#endif /* RKT_OPERANDS_H */
