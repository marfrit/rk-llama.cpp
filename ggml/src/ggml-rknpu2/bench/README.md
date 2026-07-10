# RKNPU2 matmul micro-benchmarks

Standalone probes that call the vendor `librknnrt` matmul API directly (no ggml,
no attention), timing only `rknn_matmul_run`. Used to establish the RK3588 int8
GEMM ceiling independent of this backend.

```
make            # builds mm_peak and mm_concurrent
```

The NPU is **single-tenant** — stop any other NPU process (e.g. an llama-server
using this backend) before running, or the kernel can panic.

- `mm_peak` — single-context throughput across shapes, layouts, and quant types.
- `mm_concurrent` — one single-core context per NPU core, submitted concurrently
  (a single context rejects a multi-core mask on RK3588).

## Key results (librknnrt 2.3.2, RK3588 @ 1.0 GHz, 2026-07-10)

- Single-core int8 4096³ = **0.69 TOPS** = 34% of a core's rated 2.0.
- `B_layout` (native vs normal) and `B_quant_type` (per-channel vs per-layer) make
  **no** throughput difference — all 0.69 TOPS.
- 3 concurrent contexts = **1.83 TOPS** (2.77× scaling) = **30.5% of the rated 6.0**.
  The 6 TOPS figure is a convolution number; the GEMM ceiling is silicon-limited.
- K > 8192 (the int8 K-limit) in one context is ~14× slower — split K host-side.

See `docs/rknpu2-campaign-results.md` for the full campaign.
