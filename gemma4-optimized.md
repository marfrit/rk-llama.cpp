# Gemma 4 12B on the RK3588 NPU — optimized serving config

Branch `mtp-b9549` (upstream b9570 + RKNPU2 backend + perf work). Host: boltzmann, RK3588,
31 GiB RAM, 4× A76 (cpu4-7) + 4× A55 (cpu0-3).

## Models

Both required. Place in `~/models/`.

| file | size | role |
|------|-----:|------|
| `gemma-4-12b-it-Q8_0.gguf` | 11.8 GiB | target. `arch=gemma4`, dense, `n_ctx_train=131072` |
| `mtp-gemma-4-12b-it-Q8_0.gguf` | 444 MiB | MTP drafter. `arch=gemma4-assistant`, 4 layers, `nextn_predict_layers=4` |

Both from `unsloth/gemma-4-12b-it-GGUF` (drafter under `MTP/`). The drafter is welded to the
target: its graph reads the target's `tok_embd` and aliases K/V pointers, so it only loads
in-process alongside the matching target.

## systemd unit

`~/.config/systemd/user/llama-server-gemma4-npu.service`, enabled with `loginctl enable-linger`.

```ini
[Service]
Type=simple
LimitNOFILE=65536
CPUAffinity=4-7
WorkingDirectory=/home/mfritsche/npu
ExecStart=/home/mfritsche/src/rk-llama.cpp/build-mtp/bin/llama-server \
     -m /home/mfritsche/models/gemma-4-12b-it-Q8_0.gguf \
     -md /home/mfritsche/models/mtp-gemma-4-12b-it-Q8_0.gguf \
     --spec-type draft-mtp --spec-draft-n-max 4 \
     --alias gemma-4-12b-q8 \
     -c 131072 -t 4 \
     --host 0.0.0.0 --port 8085
Restart=on-failure
TimeoutStartSec=300
```

Build: `cmake -B build-mtp -DLLAMA_RKNPU2=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON`.

Non-obvious, all load-bearing:
- `LimitNOFILE=65536` — RKNPU exhausts the default 1024 fds during init and **segfaults**.
- `CPUAffinity=4-7` — A76 only. `-t 8` across both clusters is *slower*; the A55s straggle at
  the per-token barrier.
- `--spec-draft-n-max 4` — `nextn_predict_layers=4`, so there is no head to predict a 5th token.
- `-c 131072` — `n_ctx_train`. Raising it needs RoPE scaling.
- The NPU is **single-tenant**. Stop this unit before running any other NPU process.

## Measurements (this version, `02a883fea`)

`llama-bench`, `taskset -c 4-7`, `-t 4 -r 3`, no speculation, host idle:

| backend | pp512 | tg128 |
|---------|------:|------:|
| CPU only (`LLAMA_RKNPU2=OFF`) |  8.70 ± 0.01 | 1.91 ± 0.00 |
| RKNPU (model only)            | 38.31 ± 0.03 | 2.35 ± 0.00 |

Taken on an idle host (`-r 3`). NPU: **4.4× prefill**, **+23% decode**. Decode gains little because it is memory-bandwidth
bound, not compute bound — which is what makes speculation the real decode lever.

`llama-server`, `-c 8192`, `temp 0`, distinct prompts, decode t/s:

| config | prose | file echo (`--reasoning-budget 0`) |
|--------|------:|----------------------------------:|
| RKNPU, no speculation | 2.118 | 2.055 |
| RKNPU + MTP (`-n-max 4`) | **3.502** (+65%) | **5.207** (+153%) |

MTP draft acceptance is workload-dependent: 0.50 prose, 0.77–0.83 reasoning traces, 0.90–0.94
verbatim file echo. Live at `-c 131072`: ~4.0–4.7 t/s.

Speculative decoding is lossless — greedy output was verified **byte-identical** to the
non-speculative run.

## Why this configuration

**Dense Q8_0, not MoE.** `ggml-rknpu2` implements `GGML_OP_MUL_MAT` only. It has no
`MUL_MAT_ID`, the op llama.cpp uses for MoE expert routing, so on any MoE every expert FFN
falls back to CPU and the NPU's 4.4× prefill evaporates. Q8_0 is also the NPU's native W8A8
pipeline. A Q8_0 30B MoE would need ~30.3 GiB of weights on a 31 GiB box regardless.

**Gemma 4 12B.** Dense, Q8_0, native 131072 context, and its KV is cheap: 8 global layers at
1 KV head (16 KiB/token) plus a fixed ~320 MiB for the 40 sliding-window layers. ~15 GiB
resident at full context.

**MTP over the alternatives.** Three drafting strategies were measured:

| strategy | draft size | acceptance | result |
|----------|-----------:|-----------:|-------:|
| `gemma-4-E2B` draft model | 2.9 GiB | 0.49 | **−5.3%** |
| `ngram-simple` prompt-lookup | none | 0.33–0.52 | +48% (copy-heavy) |
| **MTP drafter** | 0.44 GiB | 0.50–0.94 | **+65% … +153%** |

Acceptance is not the variable — E2B and MTP accept at the same rate and land 70 points apart.
What decides it is the *cost of a wrong guess*: a 25×-smaller drafter makes rejected tokens
nearly free. `ngram-simple` is the fallback for models without an MTP head.

**Do not stack.** `--spec-type draft-mtp,ngram-cache` is *slower* than `draft-mtp` alone
(5.073 vs 5.207); the two drafters compete for the same budget. Also: `ngram-cache` is not
prompt-lookup — it reads a static cache file (`-lcs`) and drafts from an empty corpus without one.

## Operational notes

- gemma-4 is a **reasoning model**: it fills `reasoning_content` before `content`. A tight
  `max_tokens` returns an empty `content` — that is not a failure. Benchmarks of copy behaviour
  need `--reasoning-budget 0`.
- Tool calls work without `--jinja` (the gguf carries a template with tool support).
- Rollback: `perf-rknpu2` + `build/` + `llama-server-ornith-npu.service` are preserved.
