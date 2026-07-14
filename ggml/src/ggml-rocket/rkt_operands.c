/* SPDX-License-Identifier: MIT
 *
 * rkt_operands.c — NPU operand layout (faithful ports of Mesa rocket).
 *
 * Non-depthwise conv/matmul. Row-major host tensors:
 *   weights_in : [oc_real][ww][wh][ic_real]   (conv weights)
 *   input_in   : [iw][ih][ic_real]
 *   output_out : [oh][ow][oc_real]
 *
 * Ported from Mesa rkt_coefs.c (rkt_fill_weights, rkt_fill_biases) and
 * rkt_ml.c (feature tiling in rkt_ml_subgraph_invoke, output de-tile in
 * rkt_ml_subgraph_read_outputs). Assumes input_zero_point handling as Mesa.
 */
#include "rkt_operands.h"

#define FEATURE_ATOMIC_SIZE 16
#define WEIGHT_ATOMIC_SIZE  32
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define MIN2(a, b) ((a) < (b) ? (a) : (b))
static unsigned alignu(unsigned v, unsigned a) { return ((v + a - 1) / a) * a; }

unsigned rkt_raw_input_size(unsigned iw, unsigned ih, unsigned ic_real)
{
	return iw * ih * (DIV_ROUND_UP(ic_real, FEATURE_ATOMIC_SIZE) * 2) *
	       FEATURE_ATOMIC_SIZE;
}

unsigned rkt_raw_output_size(unsigned ow, unsigned oh, unsigned oc_real)
{
	return ow * oh * (DIV_ROUND_UP(oc_real, FEATURE_ATOMIC_SIZE) * 2) *
	       FEATURE_ATOMIC_SIZE;
}

unsigned rkt_packed_weights_size(unsigned ww, unsigned wh, unsigned ic_real,
				 unsigned oc_real)
{
	unsigned ic = alignu(MAX2(ic_real, FEATURE_ATOMIC_SIZE), FEATURE_ATOMIC_SIZE);
	unsigned oc = alignu(oc_real, 2);
	return ww * wh * oc * alignu(ic, WEIGHT_ATOMIC_SIZE) * 2;
}

/* Mesa rkt_fill_weights (non-depthwise). */
unsigned rkt_pack_weights(const uint8_t *win, unsigned ww, unsigned wh,
			  unsigned ic_real, unsigned oc_real, uint8_t wzp,
			  uint8_t *out)
{
	unsigned ic = alignu(MAX2(ic_real, FEATURE_ATOMIC_SIZE), FEATURE_ATOMIC_SIZE);
	unsigned oc = alignu(oc_real, 2);
	unsigned groups = WEIGHT_ATOMIC_SIZE;
	unsigned ic1n = DIV_ROUND_UP(ic, groups);
	unsigned ic2n = MIN2(ic, groups);
	unsigned n = 0;

	for (unsigned oc1 = 0; oc1 < DIV_ROUND_UP(oc, WEIGHT_ATOMIC_SIZE); oc1++)
		for (unsigned ic1 = 0; ic1 < ic1n; ic1++)
			for (unsigned x = 0; x < ww; x++)
				for (unsigned y = 0; y < wh; y++)
					for (unsigned oc2 = 0;
					     oc2 < MIN2(oc, WEIGHT_ATOMIC_SIZE); oc2++)
						for (unsigned ic2 = 0; ic2 < ic2n; ic2++) {
							unsigned occ = oc1 * WEIGHT_ATOMIC_SIZE + oc2;
							unsigned icc = ic1 * groups + ic2;
							if (oc_real > 2 && occ >= alignu(oc_real, 2))
								continue;
							if (occ >= oc_real) {
								out[n++] = 0x0;
							} else if (icc >= ic_real) {
								if (ic2 < 16 || (ic_real % 32) > 16)
									out[n++] = (uint8_t)(wzp - 0x80);
							} else {
								unsigned idx = ((occ * ww + x) * wh + y) * ic_real + icc;
								out[n++] = (uint8_t)(win[idx] - 0x80);
							}
						}
	return n;
}

/* Mesa rkt_fill_biases (truncate_bits=0 path) + calculate_bias_correction. */
void rkt_compute_biases(const uint8_t *win, const int32_t *bin, unsigned ww,
			unsigned wh, unsigned ic_real, unsigned oc_real,
			uint8_t wzp, uint8_t izp, int32_t *bout)
{
	for (unsigned oc = 0; oc < oc_real; oc++) {
		int32_t corr = 0;
		for (unsigned x = 0; x < ww; x++)
			for (unsigned y = 0; y < wh; y++)
				for (unsigned ic = 0; ic < ic_real; ic++) {
					unsigned idx = ((oc * ww + x) * wh + y) * ic_real + ic;
					corr += ((int32_t)win[idx] - wzp) *
						((int32_t)izp - 0x80);
				}
		bout[oc] = bin[oc] - corr;
	}
}

/* Mesa feature tiling (rkt_ml_subgraph_invoke, input_channels>1 path). */
unsigned rkt_pack_input(const uint8_t *iin, unsigned iw, unsigned ih,
			unsigned ic_real, uint8_t izp, uint8_t *out)
{
	unsigned n = 0;
	for (unsigned u = 0; u < DIV_ROUND_UP(ic_real, FEATURE_ATOMIC_SIZE); u++)
		for (unsigned x = 0; x < iw; x++)
			for (unsigned y = 0; y < ih; y++)
				for (unsigned c = 0; c < FEATURE_ATOMIC_SIZE; c++) {
					unsigned ich = c + u * FEATURE_ATOMIC_SIZE;
					if (ich < ic_real)
						out[n++] = (uint8_t)(iin[(x * ih + y) * ic_real + ich] - 0x80);
					else
						out[n++] = (uint8_t)(izp - 0x80);
				}
	return n;
}

/* Mesa output de-tile (rkt_ml_subgraph_read_outputs). */
void rkt_unpack_output(const uint8_t *raw, unsigned ow, unsigned oh,
		       unsigned oc_real, uint8_t *out)
{
	for (unsigned oc = 0; oc < oc_real; oc++)
		for (unsigned x = 0; x < ow; x++)
			for (unsigned y = 0; y < oh; y++) {
				unsigned c = oc % FEATURE_ATOMIC_SIZE;
				unsigned g = oc / FEATURE_ATOMIC_SIZE;
				out[(y * ow + x) * oc_real + oc] =
					(uint8_t)(raw[((g * oh + y) * ow + x) * FEATURE_ATOMIC_SIZE + c] + 0x80);
			}
}
