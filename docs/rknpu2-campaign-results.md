# RKNPU2 perf campaign — results (E1–E6)

Model: ornith-1.0-9b-Q8_0.gguf (Q8_0 → whole model on NPU)
Host: boltzmann.fritz.box (RK3588). Harness: `ulimit -n 65536`, llama-bench, `-t 4`.
Branch: perf-rknpu2 (base compat commit 39622ccb3).

## Throughput (t/s), baseline → each step

| test  | baseline | E1     | E2    | E3        | E4    | E5    | net    |
|-------|----------|--------|-------|-----------|-------|-------|--------|
| pp300 | 30.66    | 37.01  | 35.8  | 35.78     | 36.53 | 36.50 | +19.0% |
| pp512 | 40.42    | 40.25  | 41.1  | 41.56     | 42.09 | 42.10 | +4.2%  |
| pp700 | 40.17    | 40.03  | 40.3  | 40.62     | 42.37 | 41.52 | +3.4%  |
| tg128 |  2.58    |  2.61  |  2.57 | 2.73–2.80 |  2.93 |  2.98 | +15.5% |

Commits: E1=04d2c6e54, E2=c5a47492f, E3=8adda8448, E4=3705e8476, E5=05cf2f9dd.

## What moved, and why
- **pp300 +19%** — E1 M-bucketing: `next_power_of_two(300)=512` wasted 70% of the
  matmul; ceil-to-128 → 384. Biggest single win.
- **tg128 +15.5%** — decode CPU tail: E3 (skip OMP fork/join at M=1), E4 (per-thread
  scratch, no per-row malloc), E5 (NEON 4-wide collect).
- **pp512 / pp700 modest** — already near the NPU compute ceiling; CPU-side opts
  can't move an NPU-bound phase much. E2 was throughput-neutral by design (it
  removed dead work — memset + RMW on the single-K-segment path — not a bottleneck).

## E6 Step-0 measurement gate — CLOSED (no pipelining)

Instrumented the three serial phases (prepA / NPU run / collectC), cumulative:

| regime        | prepA  | run    | collectC | run ÷ (prepA+collectC) | CPU phases % of tracked |
|---------------|--------|--------|----------|------------------------|-------------------------|
| decode (M=1)  | 0.271  | 4.432  | 0.105    | 11.8×                  | 7.8%                    |
| prefill (M=512)| 8.791 | 57.527 | 5.905    | 3.9×                   | 20.3%                   |

Decision rule (guide E6 Step 0): proceed only if `prepA + collectC` is >25% of total.
Both regimes fall below it — decode overwhelmingly so, and that's the metric that
actually hurts (tg ~3 t/s). The percentages are of *tracked* time only; the real
wall-time fraction is smaller still (untracked: context setup, mem syncs, sampling).

**Verdict: NPU-bound. E6 pipelining (medium-high risk, driver kernel-panic hazard)
would recover almost nothing. Not implemented.** Instrumentation reverted; the
E1–E5 gains are the ceiling for this backend without touching the NPU kernel itself.

## NPU vs CPU-only build (same model, same harness)

CPU build = `build-cpu` (`GGML_RKNPU2=OFF`, CPU+NATIVE), same commit 05cf2f9dd,
`-p 300,512,700 -n 128 -t 4`.

| test  | NPU (E5) | CPU   | NPU speedup |
|-------|----------|-------|-------------|
| pp300 | 36.50    | 13.64 | 2.68×       |
| pp512 | 42.10    | 13.65 | 3.08×       |
| pp700 | 41.52    | 13.85 | 3.00×       |
| tg128 |  2.98    |  2.78 | 1.07×       |

- Prefill: NPU ~3× CPU (compute-bound GEMM — the NPU's job).
- Decode: both bandwidth-bound (~8.86 GB streamed per token); NPU only +7%.
- Note: baseline NPU tg was **2.58 — below CPU's 2.78**. The E3/E4/E5 decode work
  (→2.98) is what flipped the NPU backend from a decode *liability* to a marginal
  win. Without those steps you'd be better off on CPU for generation.

## Runtime levers (no code change) — big.LITTLE thread pinning

Topology: cpu0-3 = A55 little @1.8 GHz, cpu4-7 = A76 big @2.3 GHz. Same E5 binary,
same model, `taskset`/`CPUAffinity` only.

| config              | affinity | -t | pp512          | tg128         |
|---------------------|----------|----|----------------|---------------|
| A_sched (unpinned)  | 0-7 sched| 4  | 43.22 ± 0.43   | 2.97 ± 0.03   |
| **B_A76pin**        | **4-7**  | 4  | **51.37 ± 0.07** | **3.35 ± 0.00** |
| C_all8              | 0-7      | 8  | 46.92 ± 0.04   | 2.58 ± 0.05   |
| D_A55pin            | 0-3      | 4  | 16.02 ± 0.01   | 2.58 ± 0.01   |

- **A76-pin (B) vs unpinned scheduler (A): +18.9% prefill, +12.8% decode, zero cost.**
  The single clearest runtime win of the campaign. Stacks on top of the E1–E5 code gains.
- **`-t 8` across all cores (C) is worse than 4 big cores (B)** on both phases: the four
  A55s straggle at the per-token barrier and drag the fast cores. More threads ≠ faster.
- A55-only (D) collapses prefill (3.2×) — confirms prefill rides the big-core GEMM glue.
- **Applied:** `CPUAffinity=4-7`, `-t 4` in `llama-server-ornith-npu.service` (user unit).

## W4A4 NPU pipelines — dead lever

The RKNPU vendor path offers W4A4 (4-bit weight+activation) in standard and Hadamard-rotated
variants. Decode-only (`tg64`, same model, A76-pinned harness):

| pipeline        | tg64        |
|-----------------|-------------|
| W4A4 standard   | 3.43 ± 0.08 |
| W4A4 Hadamard   | 3.17 ± 0.04 |

- Decode is bandwidth-bound, not MAC-bound, so dropping weights/activations to 4-bit buys
  no meaningful throughput over the Q8_0 path (3.35 tg128 pinned) — and costs accuracy.
- The Hadamard variant (the *accuracy-preserving* rotation) is the **slower** of the two
  (−8%): the rotation transform is pure overhead here. So W4A4 trades quality for nothing.
  **Not pursued.**

## KV-cache quantization + flash-attention (depth sweep)

KV cache lives CPU-side (never touches the NPU). This ornith model caches KV on only
**8 layers** (sparse/hybrid attention, n_ctx_train=262144) → ~32 KiB/token. Quantized V
*requires* `-fa 1`. tg128 at increasing context depth:

| KV cfg          | d0 (tg128)  | d8192       | d16384        |
|-----------------|-------------|-------------|---------------|
| f16, no FA      | 3.34 ± 0.00 | 2.75 ± 0.00 | 2.20 ± 0.00   |
| f16, FA         | 3.35 ± 0.03 | 2.18 ± 0.00 | 1.64 ± 0.01   |
| q8_0, FA        | 3.38 ± 0.00 | 2.55 ± 0.01 | 2.09 ± 0.01   |
| q4_0, FA        | 3.37 ± 0.01 | 2.79 ± 0.00 | ≈2.38 \*      |

\* q4_0@d16384 raw table scrolled past the capture window; run completed, delta recorded
as +8.2% over f16-noFA (→ ≈2.38). Not re-measured (would require stopping the live NPU
service).

- **Flash-attention alone is a net regression on rknpu2:** f16+FA vs f16-noFA is −21% at
  8k, −25% at 16k. FA is only worth enabling because quantized-V *needs* it.
- **q8_0 KV is strictly pointless** — pays the FA penalty at every depth, never recovers
  it (KV isn't the bottleneck), stays below f16-noFA throughout.
- **q4_0 KV is a wash until it isn't:** ties f16-noFA at d0/d8192, then the smaller KV
  footprint finally offsets the FA penalty at 16k (+8%). Only pays off past ~16k, and only
  modestly — trade against the quality hit.
- **Shipping default: f16 KV, no FA.** For an 8-KV-layer model the memory savings are tiny
  (32 KiB/token → full 256k ctx is only ~8 GiB f16), so q4_0 isn't worth it below very long
  contexts.

## Speculative decoding — measured, and it loses (2026-07-09)

Target `gemma-4-12b-it-Q8_0` on NPU + A76 (`-Cr 4-7`), draft `gemma-4-E2B-it-Q4_K_M` on the
otherwise-idle A55 cluster (`-Crd 0-3 -td 4`), `-devd none` to keep the draft off the
single-tenant NPU. `-c 8192`, `max_tokens 128`, `temperature 0`, three **distinct** prompts.
Vocabs verified bit-identical (token sha `163d3f6aeefb`, merges sha `2dbf8105a2c2`).

| prompt | baseline | spec `-n-max 6` | Δ | draft acceptance |
|--------|---------:|----------------:|------:|-----------------:|
| q1     | 2.127    | 1.927           | −9.4% | 0.504 |
| q2     | 2.085    | 2.033           | −2.5% | 0.530 |
| q3     | 2.154    | 2.067           | −4.0% | 0.441 |
| mean   | 2.122    | 2.009           | **−5.3%** | 0.492 |

Depth sweep (single cold sample each, baseline 2.085): `-n-max 4` → 1.924, `6` → 1.858,
`8` → 1.789. **Monotonically worse as the draft goes deeper** — the signature of a draft
whose per-token cost isn't amortised by its hit rate.

**Why it loses:** E2B is only ~4× smaller than the 12B target (2.9 GB vs 11.4 GB RKNPU
buffer). At ~0.49 acceptance you win ~2 tokens per cycle but pay 6 draft forward passes on
the slow A55s. The NPU *verify* step is nearly free (it's an `M=6` batch — the regime the NPU
is good at); the loss is entirely draft cost. Speculative decoding is not wrong here, the
**draft ratio** is.

**The config that should win** is the MTP drafter. It did — see below.

## MTP speculative decoding — +65%, and lossless (2026-07-10)

Integrated llama.cpp PR #23398 ("llama : add Gemma4 MTP") by merging `refs/tags/b9549`
(== merge commit `04eb4c446`, i.e. the PR itself) into `perf-rknpu2` → branch `mtp-b9549`,
merge commit `02a883fea`. **Zero conflicts.** Our 6 commits touch only
`ggml/src/ggml-rknpu2/ggml-rknpu2.cpp`; the PR touches zero files under `ggml/`. Now at
upstream b9570 (was b9101). Built into `build-mtp/` (`-DLLAMA_RKNPU2=ON`).

Drafter: `mtp-gemma-4-12b-it-Q8_0.gguf`, 444 MB, `arch=gemma4-assistant`, `block_count=4`,
`nextn_predict_layers=4`, `embedding_length_out=3840` (== target `n_embd`).

Target `gemma-4-12b-it-Q8_0`, `-c 8192 -t 4 -Cr 4-7`, `max_tokens 128`, `temp 0`, 3 distinct prompts:

| config | q1 | q2 | q3 | mean | Δ vs baseline |
|--------|---:|---:|---:|-----:|--------------:|
| baseline (new build) | 2.097 | 2.092 | 2.166 | 2.118 | — |
| `draft-mtp` n-max 2  | 3.301 | 3.200 | 3.446 | 3.316 | **+56.6%** |
| `draft-mtp` n-max 4  | 3.303 | 3.394 | 3.810 | 3.502 | **+65.3%** |

Live service at `-c 131072`: **4.04 tok/s** decode, acceptance 0.57–0.60, tool calls intact.

### The finding: acceptance was never the variable

| draft | size | ratio | acceptance | result |
|-------|-----:|------:|-----------:|-------:|
| gemma-4-E2B Q4_K_M | 2.9 GB | 4× | 0.492 | **−5.3%** |
| MTP head | 0.44 GB | 25× | 0.503 | **+65.3%** |

Statistically identical hit rates, opposite outcomes. What differs is the **price of a wrong
guess**. The NPU's `M=k` batch verify is cheap in both cases; the draft side was always the
binding constraint. Do not reason about speculative decoding from acceptance rate alone.

`nextn_predict_layers=4` caps useful `--spec-draft-n-max` at 4 — there is no head to predict
a 5th token. Do not carry over the 6/8 depths used for draft *models*.

### Correctness gates (both required, both passed)
- **NPU still offloading** after a 448-commit upstream jump: `llama-bench pp512` shows
  `backend RKNPU` on both — old 33.07 ± 0.42, new 31.11 ± 0.09 t/s. Decode alone could never
  have shown this (bandwidth-bound; NPU is only +7% there).
- **Lossless:** greedy (`temp 0`, `seed 42`) output is **byte-identical** between baseline and
  MTP (714 bytes, sha `1f8f6aae1e2deb108206`). This is the gate that catches upstream's
  `TAG_KV_CACHE_SHARE_CELLS` aliasing misbehaving — upstream excludes both `GEMMA4` and
  `GEMMA4_ASSISTANT` from its cross-backend parity fixture (`// FIXME @ngxson`), so a
  corrupted shared cell would emit fluent, plausible, *different* text at a healthy tok/s.

MTP is **not** an independent draft model: `gemma4-assistant`'s graph reads the target's
`tok_embd` and aliases its K/V pointers to the target's last non-SWA and last SWA layers;
`llama_context` throws if `ctx_other` is null. `-devd none` is silently ignored on this path
(log reports `devices=[default]`). The target's KV cache is CPU-resident
(`CPU KV buffer 2048 + 1440 MiB`), so the aliasing is CPU→CPU and costs no NPU tenancy.

### Measurement hazards hit while producing the above
- **Warm-cache mirage.** Re-sending an *identical* prompt at `temperature 0` made every spec
  config report `draft acceptance = 1.00000` and 3–4.6× throughput (up to 8.63 tok/s). The
  baseline showed no such jump. It is cache replay of the previous completion, not drafting
  skill. **Always use distinct prompts when benching speculative decoding.**
- `ulimit -n 65536` is mandatory (RKNPU: `errno 24 Too many open files` → segfault). systemd
  units set `LimitNOFILE`; a bare ssh shell inherits 1024.
- A bench `trap ... EXIT` must kill its own llama-server child *before* restarting the
  systemd unit, or two processes briefly contend for the single-tenant NPU.

## Prompt-lookup / n-gram speculation — measured (2026-07-10)

Same target, `-c 8192 -t 4 -Cr 4-7`, `temp 0`, `seed 42`, each prompt sent **once** per server.
Two agent-shaped, copy-heavy prompts: re-emit a C function with `10`→`20`; re-emit a JSON
config with `port` `8080`→`9090`. Run with `--reasoning-budget 0` to reach the copy phase —
**without it, gemma-4 spends its whole token budget on `reasoning_content` and never copies.**

| config | p1 (C) | p2 (JSON) | mean | Δ | acceptance |
|--------|-------:|----------:|-----:|--:|-----------:|
| baseline              | 2.032 | 2.078 | 2.055 | — | — |
| `ngram-mod`           | 1.951 | 2.277 | 2.114 | +2.9% | 0.203 / 0.289 |
| `ngram-map-k`         | 3.345 | 2.038 | 2.692 | +31.0% | 0.521 / (no draft) |
| **`ngram-simple`**    | 3.309 | 2.783 | 3.046 | **+48.2%** | 0.521 / 0.326 |
| `ngram-cache`         | 4.402 | 1.678 | 3.040 | +47.9% | 0.693 / **0.017** |
| `draft-mtp,ngram-cache` | 5.444 | 4.702 | 5.073 | +146.9% | 0.786 / 0.719 |
| **`draft-mtp`**       | 5.146 | 5.267 | **5.207** | **+153.4%** | 0.941 / 0.900 |

### Three traps found here
1. **`ngram-cache` is NOT prompt-lookup.** It reads a *static cache file* via
   `-lcs/--lookup-cache-static`. With no cache supplied it drafts from an empty corpus:
   **2 accepted / 120 generated (0.017)** on near-verbatim JSON copying. Its numbers above are a
   correctly-measured property of a misconfigured feature. The real prompt-lookup is
   **`ngram-simple`**.
2. **Stacking speculation strategies costs throughput.** `draft-mtp,ngram-cache` = 5.073 vs
   `draft-mtp` alone = 5.207. On the reasoning-heavy variant it was worse still (3.825 vs 4.546,
   −16%, acceptance 0.77→0.47): the n-gram drafter burns budget proposing tokens MTP already
   predicts. They compete, they don't compose.
3. **`ngram-mod` reproducibly diverges from non-speculative greedy** by one whitespace token
   (`"cache": {` vs `"cache":  {` — the latter matches the prompt's original spacing). Verified:
   baseline STABLE run-to-run, ngram-mod STABLE run-to-run, and DIVERGENT from each other every
   time. The server's verify path is exact (drafted tokens are compared against the target's own
   sampled tokens), so the likely mechanism is **batch-shape-dependent FP reduction order flipping
   a near-tie argmax** — speculative decoding is lossless in exact arithmetic, not necessarily
   bit-identical in float. *Likely, not proven* (would need a logit dump). `draft-mtp`,
   `ngram-simple`, `ngram-map-k` were all byte-identical to baseline.

### MTP acceptance is workload-dependent
- prose (100-word explanations): **0.50**
- reasoning traces (structured bullet lists): **0.77–0.83**
- verbatim file echo (the agent workload): **0.90–0.94**

MTP is best on every task shape tested. `ngram-simple` remains the fallback where no MTP head
exists for a model: free, no draft model, no vocab constraint, +48% on copy-heavy work.

## E7 — activation-reuse in prepA: NEGATIVE RESULT (2026-07-10)

A source review of the NPU math path proposed reusing the quantized activation across
`wq`/`wk`/`wv` (which share one `src[1]`) and across `gate`/`up`, skipping the redundant
amax scan + quantize + `rknn_mem_sync`. Estimated +5% prefill / +3% decode.

**Implemented, measured, reverted.** `pp512` 38.35 → 38.34, `tg128` 2.35 → 2.34 — flat.

Instrumented the hit rate rather than guessing why:

| regime  | graph_compute calls | nodes | MUL_MAT | reuse hit | reuse miss |
|---------|--------------------:|------:|--------:|----------:|-----------:|
| prefill | 400                 | 663   | 468     | 66        | 469        |
| decode  | 9200                | 15225 | 10771   | 1538      | 10803      |

**`mm/call ≈ 1.17`.** The backend supports only `GGML_OP_MUL_MAT`, so every intervening op
bounces to CPU and `ggml_backend_sched` splits the graph around it — 563 splits for 1972
nodes. q/k/v arrive in *separate* `graph_compute` invocations, so the cache (which must be
cleared per call, since ggml recycles `ggml_tensor` structs across tokens) tops out at a
**12.3% hit rate**. 12% of a phase that is 12% of prefill ≈ 1.5%: below noise.

Cannot be salvaged by persisting the cache across calls — there is no signal for "the CPU
rewrote this activation", and fingerprinting the buffer costs about what the amax scan costs.

**Lesson:** the review asserted that sched groups consecutive same-backend nodes into one
split. It does not, for a `MUL_MAT`-only backend. `graph splits = 563` was already in the
server's startup log. **Instrument the premise before implementing the optimization.**

Surviving change: `rknpu2: saturate int8 activation quantization` (branch
`rknpu-int8-clamp`). `quantize_fp32_to_int8` cast `roundf(v)` straight to `int8_t`; one ULP
of error in `iscale` rounds 127→128, wrapping to −128. The int4 path already clamped.
**Latent** — greedy output byte-identical before/after (sha `11b2174379e0c2040246`).

Still open, both blocked on facts not in this tree:
- `max_k_limit = 8192` forces `down_proj` (K = n_ff = 15360) into 2 K-segments on every pass,
  costing a `memset` + RMW accumulate + an extra `rknn_matmul_run`. Needs the real int8 K
  limit from the vendor driver / TRM.
- Whether `rknn_matmul_run` serializes on an internal librknnrt lock. If it does, the 3-core
  `omp parallel` N-split buys overhead and nothing. Worth up to 3× prefill if broken.

## int4 (W4A4) — broken numerics + prefill collapse (2026-07-10)

Q4_K_M (6.62 GiB) via the int4 pipelines, A76-pinned, quiet host:

| config | pp512 | tg128 | output |
|--------|------:|------:|--------|
| Q8_0 (deployed)          | 38.34 | 2.34 | correct |
| Q4_K_M `W4A4_STANDARD`   | 5.71  | 4.08 | **token soup** |
| Q4_K_M `W4A4_HADAMARD`   | 5.63  | 3.50 | **degenerate** |
| Q4_K_M forced `W8A8`     | 38.19 | 2.35 | **correct** (`221\nRed, blue, and yellow.`) |

- **Q4_K dequant (`137b4ceeb`) is vindicated:** same file through W8A8 is correct, so the fault
  is the int4 *pipeline*, not our dequant. Cause: B quantized at one scalar per (k_seg×n_seg)
  block — per-tensor-grade granularity at 4 bits — plus an unimplemented int4 repack in the blob
  (`"Convert not imp <int4x2,int4x2>"`).
- **Bandwidth model confirmed by control:** the 6.62 GiB file forced through W8A8 decodes at 2.35
  = identical to the 11.78 GiB Q8_0. File size is irrelevant; NPU-buffer bytes/weight is decisive.
- int4 decode +74% (halved bytes help) but prefill 6.7× SLOWER — decode-only even if numerics fixed.

### int4 is dead at the HARDWARE level (2026-07-10, closed)
Probed which matmul dtypes RK3588 supports (`rknn_matmul_create`, layout-independent):

| dtype | supported |
|-------|-----------|
| W8A8 int8×int8 | ✅ |
| W4A4 int4×int4 | ✅ |
| W8A4 int8×int4 (weight-only) | ❌ `unsupported ... in this platform` |
| W16A4 fp16×int4 (weight-only) | ❌ |
| int8×int4→fp16, fp16×int8 | ❌ |

**There is NO weight-only int4 mode on RK3588.** Every dtype pairing int4 weights with
higher-precision activations — how all practical LLM int4 quant works — is rejected by the
hardware. The only int4 mode is W4A4, which forces **int4 activations**; transformer
activation outliers can't be represented in 4 bits → the garbage we measured. Per-channel
*weight* scaling can't fix it (the problem is the activations). **The "fewer bytes" lever is
closed: you cannot halve weight bytes via int4 on this silicon.** (Clean-room all-ones
arithmetic probe was confounded by native B layout — the W8A8 control also failed — so it's
not cited; the create-support probe above is layout-independent and definitive.)

## Kernel / driver survey — no marginal wins there (2026-07-10)

- **IOMMU domain switching:** ~10.7 switches/forward-pass (6 domains, the NPU's 4 GB windows).
  `rknpu_iommu_switch_domain` does a real detach+attach, but ≤0.25% of a 425 ms decode pass.
- **No `dmc` devfreq:** DDR is fixed at LPDDR5-6400 (3200 MHz), no DVFS. Already the standard max
  (TRM "optimized" is only LPDDR5-5500). No headroom — DDR-clock lever is dead.
- **NPU SRAM = 956 KB:** 0.008% of an 11.9 GB weight set. Irrelevant for LLMs.
- Realized DDR read: 25.6 GB/s (CPU, 4 threads) / ~28 GB/s (NPU decode) = ~55% of 51.2 GB/s peak.

## NPU core scaling — Amdahl-matched (2026-07-10)

Prefill `pp512`: 1 core 18.28 / 2 30.26 / 3 38.13 (2.09×). Decode `tg64`: 1 core 0.89 / 3 2.35
(2.64×). Both match Amdahl on the E6 CPU-serial fractions (20.3% prefill / 7.8% decode) →
predicted 2.14× / 2.60×. `rknn_matmul_run` does NOT serialize; CPU glue caps the scaling.

## GEMM ceiling — clean-room vendor benchmark (2026-07-10)

Standalone C client on `rknn_matmul_api.h` + `librknnrt` (int8, timing only `rknn_matmul_run`):

| config | TOPS | % of rated |
|--------|-----:|-----------:|
| 1 core, 4096³, native+per-channel | 0.69 | 34% of 2.0 |
| 1 core, same, normal+per-layer (our layout) | 0.69 | 34% |
| 3 concurrent contexts (mask 1/2/4) | **1.83** | **30.5% of 6.0** |
| K=15360 (over 8192 limit), 1 ctx | 0.02 | 0.4% (14× slower) |

- **Layout & quant type make ZERO difference** (native==normal, per-channel==per-layer). The
  "use native layout" hypothesis is falsified — no speed there (may still fix int4 accuracy).
- **6 TOPS is a convolution number.** The vendor's own GEMM ceiling is **1.83 TOPS = 30.5%**,
  silicon-limited (conv-tuned MAC array can't feed a GEMM faster). Do NOT reverse-engineer the
  blob's GEMM — the ceiling is proven by Rockchip's own API.
- **Where the remaining prize is:** our backend runs real prefill at ~0.905 TOPS (pp512=38) =
  **~49% of the 1.83 ceiling.** The gap is OURS — CPU glue, 336 small matmuls/pass, down_proj
  K-split — not the blob. ~1.3–1.6× prefill plausibly recoverable in-tree. First step: overlap
  prepA/collectC with `rknn_matmul_run` on the *prefill* path (shelved E6 pipelining; useless for
  decode, but prefill is 20% serial CPU). K>8192 to one context is catastrophic → our manual
  K-segmentation is correct and necessary.

## E8 — prefill CPU/NPU pipelining: NEGATIVE (DDR-bandwidth-bound, 2026-07-10)

Prefill is ~22% CPU-serial (prepA quantize + collectC dequant) around the NPU run
(Amdahl from core-scaling 18.28→38.13 = 2.09×). Idea: chunk M and overlap the CPU
prep of chunk c+1 with the NPU run of chunk c. Ceiling ~1.28× prefill (less e2e:
O(M²) attention isn't pipelined). Designed with the rknpu-specialist; implemented as
a 2-stage M-chunk software pipeline (double-buffered A/C, one context set, NPU run on
a `std::thread` worker so a concurrent OMP team can't serialize it).

**Correct but flat.** Greedy output byte-identical to the non-pipeline build. pp512
38.10 → 37.5–37.8 (all power-of-2 chunks); pp2048 (deep 8-chunk) 26.54 → 26.36. No
gain at any chunk depth. (`kChunkM=192` → 29.5 is self-inflicted: `next_power_of_two(192)=256`
context padding wastes 1.5× NPU rows. Power-of-2 chunks have `ctx_M==kChunkM`, no waste.)

**Why (measured, not inferred):** timed the CPU-prep section with the NPU running vs idle.

| CPU-prep section | mean |
|------------------|-----:|
| NPU idle (overlap off) | ~1973 µs |
| NPU running (overlap on) | ~2827 µs (**+43%**) |

`bind` cost ~100 µs either way (negligible — not the per-chunk bind tax). The overlap
*happens*, but the CPU-prep runs **43% slower while the NPU streams weights**, because
both contend for DDR. The inflation ≈ the time the overlap saves → net flat, by physics.
`rknn_matmul_run` does NOT busy-poll (A76 cores 96–100% idle during pure matmul) — the
cores are free, but the **bus is not**. Idle cores ≠ idle bandwidth.

**Verdict: no-go, reverted.** A 2-context-set rewrite would fix the bind tax but not the
bandwidth wall, so it can't help. Branch `perf-prefill-pipeline` kept local/unpushed as a
correct reference; not merged. This is the campaign's through-line closing: decode
(bandwidth), GEMM ceiling 30.5% (bandwidth), prefill overlap (bandwidth) — **no idle DDR
bandwidth on RK3588 to exploit.** Only levers left: fewer bytes (int4, broken) or fewer
passes (MTP, shipped).

## Campaign verdict — decode levers

Decode (~2.6→3.35 t/s) is bandwidth + CPU-tail bound; the NPU can't be the lever there.
Ranked by realized gain, no-quality-cost first:

1. **A76 pinning (+12.8% tg / +18.9% pp)** — free, applied to the service. The one clear win.
2. **E3/E4/E5 CPU-tail code (2.58→2.98)** — flipped NPU decode from *slower* than CPU to +7%.
3. **KV q4_0** — only past ~16k, +8%, quality cost. Not default.
4. **W4A4 / FA / E6 pipelining** — dead (no gain or net regression).
5. **MTP speculative decoding (+65% prose / +153% file-echo)** — the largest decode win of the
   campaign by far, and lossless. Live on the service. Beats A76-pinning several times over.
6. **`ngram-simple` prompt-lookup (+48% on copy-heavy)** — free fallback for models with no MTP
   head. Do NOT stack it with MTP; they compete for the draft budget.
7. **Speculative decoding w/ E2B draft** — dead (−5.3%); draft ratio too small, not the NPU's fault.
