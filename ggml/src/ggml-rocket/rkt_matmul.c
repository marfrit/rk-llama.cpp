/* SPDX-License-Identifier: MIT
 *
 * rkt_matmul.c — unified INT8 matmul regcmd builder.
 *
 * Implements rkt_build_matmul_regcmd(): given a plain GEMM shape it derives
 * every low-level CNA / CORE / DPU field via the NVDLA-style CBUF tiling math
 * (ported from Mesa rocket rkt_task.c: fill_task + the single-tile branch of
 * rkt_split_tasks) and drives the two emit stages.
 *
 * The matmul Y[M][N] = X[M][K] * W[K][N] is expressed as a 1x1 convolution:
 *   input_width = 1, input_height = M, input_channels = K,
 *   output_width = 1, output_height = M, output_channels = N,
 *   weights 1x1, stride 1, no padding, no bias, identity requantisation.
 *
 * SCOPE: single-tile only. If the operation would need CBUF task-splitting
 * (input feature map too tall to fit the available input banks) this returns
 * -1 rather than emit a wrong single task. The multi-task splitter and real
 * scale/bias handling are separate follow-on work.
 */
#include "rkt_matmul.h"
#include "rkt_matmul_cna.h"
#include "rkt_matmul_coredpu.h"

#include <string.h>

/* CBUF geometry — verbatim from Mesa rkt_ml.h (NVDLA v1 convolution buffer). */
#define CBUF_BANK_SIZE        32768
#define CBUF_BANKS            12
#define CBUF_ENTRIES_PER_BANK 256
#define CBUF_ENTRY_SIZE       (CBUF_BANK_SIZE / CBUF_ENTRIES_PER_BANK) /* 128 */
#define FEATURE_ATOMIC_SIZE   16
#define ATOMIC_K_SIZE         16

#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))
#define MAX2(a, b)         ((a) > (b) ? (a) : (b))

static unsigned alignu(unsigned v, unsigned a) { return ((v + a - 1) / a) * a; }

/* Fields Mesa's tiling math reads off rkt_operation, specialised to a matmul. */
struct op {
	unsigned stride;
	unsigned input_width, input_height, input_channels;
	unsigned output_width, output_height, output_channels;
	unsigned weights_width, weights_height;
	int depthwise, addition_input, add_tensor;
};

static unsigned calc_line_stride(unsigned width)
{
	return width * ATOMIC_K_SIZE * sizeof(uint8_t);
}

static unsigned calc_entries_per_slice(const struct op *o)
{
	unsigned atomics_per_entry = CBUF_ENTRY_SIZE / FEATURE_ATOMIC_SIZE;
	unsigned total_c_atomics =
		DIV_ROUND_UP(o->input_channels * sizeof(uint8_t), FEATURE_ATOMIC_SIZE);
	unsigned last_c_atomics = total_c_atomics % atomics_per_entry;
	unsigned int_c_entries =
		(total_c_atomics / atomics_per_entry) * o->input_width;
	unsigned frac_c_entries =
		(last_c_atomics == 3)
			? o->input_width
			: DIV_ROUND_UP(last_c_atomics * o->input_width, atomics_per_entry);
	return int_c_entries + frac_c_entries;
}

static unsigned calc_input_banks(const struct op *o)
{
	return DIV_ROUND_UP(calc_entries_per_slice(o) * o->input_height,
			    CBUF_ENTRIES_PER_BANK);
}

static unsigned calc_weights_banks(const struct op *o)
{
	unsigned bytes = o->weights_width * o->weights_height *
			 o->input_channels * sizeof(uint8_t);
	if (!o->depthwise)
		bytes *= o->output_channels;
	unsigned entries = DIV_ROUND_UP(bytes, CBUF_ENTRY_SIZE);
	unsigned banks = DIV_ROUND_UP(entries, CBUF_ENTRIES_PER_BANK);
	banks++; /* Mesa: "the calc above might be wrong on this HW" */
	return banks;
}

int rkt_g_task_num = 0;   /* experiment: CBUF weight-reuse multi-task chaining */

int rkt_build_matmul_regcmd_scaled(uint64_t *out, int out_capacity,
			    uint32_t M, uint32_t N, uint32_t K,
			    uint64_t input_dma, uint64_t weights_dma,
			    uint64_t output_dma,
			    int32_t input_zero_point,
			    int32_t weight_zero_point,
			    int32_t output_zero_point,
			    float input_scale, float weights_scale,
			    float output_scale, uint64_t bias_dma)
{
	if (M == 0 || N == 0 || K == 0)
		return -1;
	/* Hard backstop: the RK3588 int8 matmul K-limit is ~8192 (verified: correct
	 * through 8192, catastrophic garbage at 11008). Weights exceeding CBUF are
	 * streamed (reuse_weights_cbuf=0, emitted below) and are correct up to that
	 * K limit, so gate on K, not on bank count. Backend supports_op also caps
	 * this; the check here protects direct callers. */
	if (K > 8192)
		return -1;

	struct op o = {
		.stride = 1,
		.input_width = 1, .input_height = M, .input_channels = K,
		.output_width = 1, .output_height = M, .output_channels = N,
		.weights_width = 1, .weights_height = 1,
		.depthwise = 0, .addition_input = 0, .add_tensor = -1,
	};

	/* ---- fill_task (Mesa rkt_task.c), stride 1 / no pad / no add ---- */
	unsigned t_input_width = o.input_width; /* width != 8, not doubled */
	unsigned t_input_height = o.input_height;
	unsigned t_input_channels =
		alignu(MAX2(o.input_channels, FEATURE_ATOMIC_SIZE), FEATURE_ATOMIC_SIZE);
	unsigned t_input_channels_real = o.input_channels;

	unsigned t_output_width = o.output_width;
	unsigned t_output_height = o.output_height;
	unsigned t_output_channels_real = o.output_channels;
	unsigned t_output_channels = alignu(MAX2(o.output_channels, 32), 32);
	/* depthwise output_channels fix-ups skipped (depthwise == 0) */

	int t_input_line_stride, t_input_surface_stride;
	if (t_input_channels_real == 1 &&
	    (t_output_channels_real > 1 || o.addition_input || o.add_tensor != -1)) {
		t_input_width = MAX2(t_input_width, FEATURE_ATOMIC_SIZE);
		t_input_line_stride =
			MAX2((int)(calc_line_stride(o.input_width) / FEATURE_ATOMIC_SIZE),
			     (int)FEATURE_ATOMIC_SIZE);
		t_input_surface_stride =
			(int)((float)t_input_line_stride * ((float)t_input_height - 1.0f));
	} else {
		t_input_line_stride = (int)(calc_line_stride(o.input_width) / 4);
		t_input_surface_stride = (int)((float)t_input_line_stride *
					       (((float)t_input_height / 4.0f) - 1.0f));
	}
	/* input_width == 8 && addition branch skipped */

	int output_line_stride = (int)calc_line_stride(o.output_width);
	int t_output_surface_stride = output_line_stride * (int)t_output_height;
	t_output_surface_stride /= FEATURE_ATOMIC_SIZE;

	unsigned t_input_data_entries;
	if (t_input_channels_real == 1)
		t_input_data_entries = t_input_width * t_input_height;
	else if (t_input_width == 40 && t_input_channels_real == 40)
		t_input_data_entries = 40;
	else
		t_input_data_entries = DIV_ROUND_UP(
			t_input_width * 2 *
				DIV_ROUND_UP(t_input_channels_real, FEATURE_ATOMIC_SIZE),
			8);

	unsigned t_weights_kernels =
		o.depthwise ? 1 : alignu(o.output_channels, 2);

	unsigned t_surfaces_per_row = t_output_width * t_output_height * 2;
	/* depthwise surfaces_per_row *= 2 skipped */

	/* ---- rkt_split_tasks, single-tile branch (full weights, full input) ---- */
	unsigned weights_banks_required = calc_weights_banks(&o);
	unsigned input_banks_required = calc_input_banks(&o);
	int reuse_weights_cbuf;
	unsigned available_input_banks;
	if (weights_banks_required + 1 < CBUF_BANKS) {
		reuse_weights_cbuf = 1;
		available_input_banks = CBUF_BANKS - weights_banks_required;
	} else {
		reuse_weights_cbuf = 0;
		available_input_banks = 7;
	}
	if (input_banks_required > available_input_banks)
		return -1; /* needs CBUF task-splitting — unsupported here */

	unsigned t_input_banks = input_banks_required;
	unsigned t_weights_banks = CBUF_BANKS - t_input_banks;
	unsigned t_atomic_count = t_output_width * t_output_height;

	/* ---- CNA params ---- */
	struct cna_params c;
	memset(&c, 0, sizeof c);
	c.input_width = t_input_width;
	c.input_height = t_input_height;
	c.input_channels = t_input_channels;
	c.input_channels_real = t_input_channels_real;
	c.output_width = t_output_width;
	c.atomic_count = t_atomic_count;
	c.weights_width = o.weights_width;
	c.weights_height = o.weights_height;
	c.weights_kernels = t_weights_kernels;
	c.weights_banks = t_weights_banks;
	c.input_banks = t_input_banks;
	c.input_data_entries = t_input_data_entries;
	c.input_line_stride = (uint32_t)t_input_line_stride;
	c.input_surface_stride = (uint32_t)t_input_surface_stride;
	c.input_dma = input_dma;
	c.weights_dma = weights_dma;
	c.stride_x = o.stride;
	c.stride_y = o.stride;
	c.output_zero_point = output_zero_point;
	c.weights_zero_point = weight_zero_point;
	c.input_zero_point = input_zero_point;
	c.depthwise = o.depthwise;
	c.reuse_weights_cbuf = reuse_weights_cbuf;
	c.addition_input = o.addition_input;
	c.add_tensor = o.add_tensor;
	c.task_num = rkt_g_task_num;

	/* ---- CORE/DPU/PC params ---- */
	struct coredpu_params d;
	memset(&d, 0, sizeof d);
	d.input_width = t_input_width;
	d.output_width = t_output_width;
	d.output_height = t_output_height;
	d.output_channels = t_output_channels;
	d.output_channels_real = t_output_channels_real;
	d.output_offset = 0;
	d.output_surface_stride = (uint32_t)t_output_surface_stride;
	d.surfaces_per_row = t_surfaces_per_row;
	d.truncate_bits = 0;
	d.output_zero_point = output_zero_point;
	d.weights_zero_point = weight_zero_point;
	d.depthwise = o.depthwise;
	d.addition_offset = 0;
	d.add_tensor = o.add_tensor;
	d.input_scale = input_scale;
	d.weights_scale = weights_scale;
	d.output_scale = output_scale;
	d.addition_scale = 1.0f;
	d.output_addr = output_dma;
	d.add_tensor_addr = 0;
	d.biases_addr = bias_dma;

	/* ---- emit both stages into one contiguous regcmd buffer ---- */
	int n1 = rkt_emit_cna(out, out_capacity, &c);
	if (n1 < 0)
		return -1;
	int n2 = rkt_emit_coredpu(out + n1, out_capacity - n1, &d);
	if (n2 < 0)
		return -1;
	return n1 + n2;
}

/* Back-compat: identity requant, no bias. */
int rkt_build_matmul_regcmd(uint64_t *out, int out_capacity,
			    uint32_t M, uint32_t N, uint32_t K,
			    uint64_t input_dma, uint64_t weights_dma,
			    uint64_t output_dma,
			    int32_t input_zero_point,
			    int32_t weight_zero_point,
			    int32_t output_zero_point)
{
	return rkt_build_matmul_regcmd_scaled(out, out_capacity, M, N, K,
					      input_dma, weights_dma, output_dma,
					      input_zero_point, weight_zero_point,
					      output_zero_point,
					      1.0f, 1.0f, 1.0f, 0);
}
