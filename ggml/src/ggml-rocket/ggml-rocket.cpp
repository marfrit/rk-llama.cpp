// SPDX-License-Identifier: MIT
//
// ggml-rocket: MUL_MAT backend for the RK3588 NPU via the mainline "rocket"
// DRM-accel driver (/dev/accel/accel0), no vendor RKNPU2 userspace.
//
// Modeled on ggml-blas: an ACCEL device with no weight buffers of its own
// (weights stay in CPU memory); the scheduler offloads MUL_MAT nodes it
// supports and leaves everything else on the CPU backend.
//
// Per MUL_MAT plane: dequantize the weight to F32 (ggml to_float trait),
// requantize per-output-channel to int8, quantize the F32 activation
// per-tensor to int8, run rkt_npu_matmul (int8 GEMM on the NPU with a
// per-channel out_scale array), then dequantize the int8 result to F32.
//
// out_scale is chosen from a Cauchy-Schwarz upper bound on |output| so the
// int8 output never clips: |<w_n, x_m>| <= ||w_n||2 * ||x_m||2. This is the
// pragmatic "int8-first" path; a lossless int32-accumulator readout is the
// planned follow-on (needs an OUT_CVT-bypass builder mode).

#include "ggml-impl.h"
#include "ggml-rocket.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" {
#include "librocket.h"
#include "rkt_npu_matmul.h"
}

// Empirically verified single-op limits on boltzmann's rocket NPU (int8):
//   rows per tile up to 32 correct at K=2560 -> use 16 for margin
//   N per tile up to 256 correct           -> use 128
//   full K correct through 8192, catastrophic at 11008 (vendor int8 K-limit)
#define ROCKET_TILE_M 16
#define ROCKET_TILE_N 128
#define ROCKET_K_MAX  8192
#define ROCKET_MIN_BATCH 32   // below this, CPU wins (decode is bandwidth-bound)

struct ggml_backend_rocket_context {
    int fd = -1;
    // reusable scratch (grown as needed, never shrunk)
    std::vector<float>   wf;   // dequantized weight plane [N*K]
    std::vector<uint8_t> aq;   // quantized activations    [M*K]
    std::vector<uint8_t> wq;   // quantized weights        [N*K]
    std::vector<uint8_t> yq;   // int8 result              [M*N]
    std::vector<float>   ws;   // per-channel weight scale  [N]
    std::vector<float>   os;   // per-channel out scale     [N]
};

static inline uint8_t rocket_q8(float v, float inv_s) {
    int q = (int)lrintf(v * inv_s);
    if (q >  127) q =  127;
    if (q < -127) q = -127;
    return (uint8_t)(q + 128); // symmetric int8 stored as uint8, zp=128
}

static void ggml_backend_rocket_mul_mat(ggml_backend_rocket_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // weight [K, N]
    const struct ggml_tensor * src1 = dst->src[1]; // activ  [K, M]

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;
    const auto * tt = ggml_get_type_traits(type);
    ggml_to_float_t const to_float = tt->to_float;

    const int64_t K = ne00;       // == ne10
    const int64_t N = ne01;       // output channels
    const int64_t M = ne11;       // rows (tokens in batch)

    static const bool dbg = getenv("GGML_ROCKET_DEBUG") != NULL;
    if (dbg) fprintf(stderr, "[rocket] enter mul_mat M=%lld N=%lld K=%lld type=%s to_float=%p\n",
                     (long long)M, (long long)N, (long long)K,
                     ggml_type_name(type), (void*)to_float), fflush(stderr);
    if (to_float == NULL || M == 0 || N == 0 || K == 0) {
        GGML_ABORT("rocket mul_mat: unexpected op (to_float=%p M=%lld N=%lld K=%lld)",
                   (void*)to_float, (long long)M, (long long)N, (long long)K);
    }

    // broadcast of src0 over src1's higher dims (as in ggml-blas)
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    ctx->wf.resize((size_t)N * K);
    ctx->aq.resize((size_t)M * K);
    ctx->wq.resize((size_t)N * K);
    ctx->yq.resize((size_t)M * N);
    ctx->ws.resize((size_t)N);
    ctx->os.resize((size_t)N);

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;

            const char * w_plane = (const char *) src0->data + i02*nb02 + i03*nb03;
            const char * a_plane = (const char *) src1->data + i12*nb12 + i13*nb13;
                  char * d_plane = (      char *)  dst->data + i12*nb2  + i13*nb3;

            // --- dequantize weight plane to F32, per-channel scale + L2 norm ---
            float wn2max = 0.0f; // max_n ||w_n||_2
            for (int64_t n = 0; n < N; n++) {
                float * wrow = ctx->wf.data() + (size_t)n * K;
                to_float((const char *) w_plane + n*nb01, wrow, K);
                float amax = 0.0f, l2 = 0.0f;
                for (int64_t k = 0; k < K; k++) {
                    float v = wrow[k];
                    float a = std::fabs(v);
                    if (a > amax) amax = a;
                    l2 += v * v;
                }
                ctx->ws[n] = (amax > 0.0f ? amax : 1e-6f) / 127.0f;
                l2 = std::sqrt(l2);
                if (l2 > wn2max) wn2max = l2;
            }

            // --- quantize activation plane, per-tensor scale + max L2 row norm ---
            float amax_a = 0.0f, ym2max = 0.0f;
            for (int64_t m = 0; m < M; m++) {
                const float * arow = (const float *) (a_plane + m*nb11);
                float l2 = 0.0f;
                for (int64_t k = 0; k < K; k++) {
                    float v = arow[k];
                    float a = std::fabs(v);
                    if (a > amax_a) amax_a = a;
                    l2 += v * v;
                }
                l2 = std::sqrt(l2);
                if (l2 > ym2max) ym2max = l2;
            }
            const float a_scale = (amax_a > 0.0f ? amax_a : 1e-6f) / 127.0f;
            const float inv_a   = 1.0f / a_scale;
            for (int64_t m = 0; m < M; m++) {
                const float * arow = (const float *) (a_plane + m*nb11);
                uint8_t * qrow = ctx->aq.data() + (size_t)m * K;
                for (int64_t k = 0; k < K; k++) qrow[k] = rocket_q8(arow[k], inv_a);
            }

            // Cauchy-Schwarz bound on |output| -> never clips int8 output.
            float out_scale = (wn2max * ym2max) / 127.0f;
            if (!(out_scale > 0.0f)) out_scale = 1e-6f;

            // The NPU applies ONE OUT_CVT scale per op == per tile of
            // ROCKET_TILE_N output columns. So every column in a tile must be
            // quantized with the SAME weight scale, otherwise off-lead columns
            // ride the tile-lead's scale and can clip. Use the tile's max
            // per-channel scale (block boundaries must match rkt_gemm_plan's
            // column tiling: [0,TILE_N),[TILE_N,2*TILE_N),...).
            for (int64_t c0 = 0; c0 < N; c0 += ROCKET_TILE_N) {
                int64_t c1 = c0 + ROCKET_TILE_N;
                if (c1 > N) c1 = N;
                float wsb = 0.0f;
                for (int64_t n = c0; n < c1; n++)
                    if (ctx->ws[n] > wsb) wsb = ctx->ws[n];
                if (!(wsb > 0.0f)) wsb = 1e-6f;
                const float inv_w = 1.0f / wsb;
                for (int64_t n = c0; n < c1; n++) {
                    const float * wrow = ctx->wf.data() + (size_t)n * K;
                    uint8_t * qrow = ctx->wq.data() + (size_t)n * K;
                    for (int64_t k = 0; k < K; k++) qrow[k] = rocket_q8(wrow[k], inv_w);
                    ctx->os[n] = out_scale / wsb;
                }
            }

            static const bool dbg = getenv("GGML_ROCKET_DEBUG") != NULL;
            if (dbg) fprintf(stderr, "[rocket] mul_mat M=%lld N=%lld K=%lld plane(%lld,%lld) type=%s ...",
                             (long long)M, (long long)N, (long long)K,
                             (long long)i12, (long long)i13, ggml_type_name(type)), fflush(stderr);
            int rc = rkt_npu_matmul(ctx->fd, ctx->aq.data(), ctx->wq.data(), NULL,
                                    (uint32_t)M, (uint32_t)N, (uint32_t)K,
                                    128, 128, 128,
                                    a_scale, 1.0f, out_scale, ctx->os.data(),
                                    ROCKET_TILE_M, ROCKET_TILE_N, ctx->yq.data());
            if (dbg) fprintf(stderr, " rc=%d\n", rc);
            GGML_ASSERT(rc == 0 && "rkt_npu_matmul failed");

            // --- dequantize int8 result into dst (F32) ---
            for (int64_t m = 0; m < M; m++) {
                float * drow = (float *) (d_plane + m*nb1);
                const uint8_t * yrow = ctx->yq.data() + (size_t)m * N;
                for (int64_t n = 0; n < N; n++)
                    drow[n] = ((int)yrow[n] - 128) * out_scale;
            }
        }
    }
}

// backend interface

static const char * ggml_backend_rocket_get_name(ggml_backend_t backend) {
    return "Rocket";
    GGML_UNUSED(backend);
}

static void ggml_backend_rocket_free(ggml_backend_t backend) {
    ggml_backend_rocket_context * ctx = (ggml_backend_rocket_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_rocket_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_rocket_context * ctx = (ggml_backend_rocket_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_rocket_mul_mat(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

static struct ggml_backend_i rocket_backend_i = {
    /* .get_name                = */ ggml_backend_rocket_get_name,
    /* .free                    = */ ggml_backend_rocket_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rocket_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_rocket_guid(void) {
    static ggml_guid guid = { 0x52, 0x6f, 0x63, 0x6b, 0x65, 0x74, 0x4e, 0x50, 0x55, 0x33, 0x35, 0x38, 0x38, 0x01, 0x00, 0x01 };
    return &guid;
}

ggml_backend_t ggml_backend_rocket_init(void) {
    int fd = rocket_open("/dev/accel/accel0");
    if (fd < 0) {
        GGML_LOG_ERROR("%s: failed to open /dev/accel/accel0 (%d)\n", __func__, fd);
        return NULL;
    }

    ggml_backend_rocket_context * ctx = new ggml_backend_rocket_context;
    ctx->fd = fd;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rocket_guid(),
        /* .iface   = */ rocket_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_rocket_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_rocket(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rocket_guid());
}

// device interface

static const char * ggml_backend_rocket_device_get_name(ggml_backend_dev_t dev) {
    return "Rocket";
    GGML_UNUSED(dev);
}

static const char * ggml_backend_rocket_device_get_description(ggml_backend_dev_t dev) {
    return "RK3588 NPU (mainline rocket accel)";
    GGML_UNUSED(dev);
}

static void ggml_backend_rocket_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_rocket_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

static void ggml_backend_rocket_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rocket_device_get_name(dev);
    props->description = ggml_backend_rocket_device_get_description(dev);
    props->type        = ggml_backend_rocket_device_get_type(dev);
    ggml_backend_rocket_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_rocket_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_rocket_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rocket_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();
    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_rocket_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_rocket_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            const int64_t K = src1->ne[0]; // == src0->ne[0]
            const int64_t M = op->ne[1];   // batch rows (tokens)

            // Offload only when the NPU can amortise its per-tile submit
            // overhead: decode (M=1) is bandwidth-bound and strictly faster on
            // CPU, so keep small batches there (mirrors ggml-blas min_batch).
            if (M < ROCKET_MIN_BATCH)             return false;
            if (src1->type != GGML_TYPE_F32)      return false;
            if (!ggml_is_contiguous(src0))        return false;
            if (!ggml_is_contiguous(src1))        return false;
            if (K <= 0 || K > ROCKET_K_MAX)       return false;
            if (K % 16 != 0)                      return false; // feature-atomic align
            // weight must be dequantizable to F32 (F32 src0 has no to_float
            // trait -> leave those on the CPU backend)
            if (ggml_get_type_traits(src0->type)->to_float == NULL) return false;
            return true;
        }

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_rocket_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_rocket_device_i = {
    /* .get_name             = */ ggml_backend_rocket_device_get_name,
    /* .get_description      = */ ggml_backend_rocket_device_get_description,
    /* .get_memory           = */ ggml_backend_rocket_device_get_memory,
    /* .get_type             = */ ggml_backend_rocket_device_get_type,
    /* .get_props            = */ ggml_backend_rocket_device_get_props,
    /* .init_backend         = */ ggml_backend_rocket_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_rocket_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_rocket_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_rocket_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rocket_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_rocket_reg_get_name(ggml_backend_reg_t reg) {
    return "Rocket";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_rocket_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_rocket_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_rocket_device = {
        /* .iface   = */ ggml_backend_rocket_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_rocket_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static const struct ggml_backend_reg_i ggml_backend_rocket_reg_i = {
    /* .get_name         = */ ggml_backend_rocket_reg_get_name,
    /* .get_device_count = */ ggml_backend_rocket_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rocket_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_rocket_reg(void) {
    static struct ggml_backend_reg ggml_backend_rocket_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rocket_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rocket_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_rocket_reg)
