# Llama 3.2 1B Multi-Precision GPU Baseline (`neuralnetwork v0.12.1`)

This document records the verified CPU-vs-Metal baseline across **FP32**, **BF16**
(native zero-upcast), **FP16**, **Q8_0**, and **Q4_0** weights (with FP32 and
FP16 KV caches) for `github.com/itsmontoya/llama` integrating
`github.com/itsmontoya/neuralnetwork@v0.12.1` (task LL-06).

## Executive Summary

- **CPU remains the default and authoritative baseline.** Resident Metal
  execution is explicitly opt-in (`-tags=metal` with `CGO_ENABLED=1` on Apple
  Silicon macOS, via CLI flags or `LoadHFModelMetalAttention` /
  `LoadHFModelMetalAttentionWithOptions`).
- **Parallel SIMD `RMSNorm` (`v0.12.1`):** Replacing single-thread-per-row
  `RMSNorm` cut 33 per-token dispatches from **`18.51 ms` → `0.22 ms`** on
  Apple M3 (**~85×** faster in end-to-end stage sampling). Even in **FP32**,
  Metal sustained decode improved from `90.10 ms/call` (`11.08 tok/s` on
  `v0.11.3`) to **`72.25 ms/call` (`13.80 tok/s`)**, closing most of the FP32
  gap with 8-core NEON CPU (`67.08 ms/call`, `14.88 tok/s`) while improving
  27–29-token prefill to **−30.6% to −33.1%** vs CPU.
- **Native `BFloat16` is the lossless sweet spot for Llama 3.2 1B:** Uploading
  the BF16 safetensors directly to `gpu.BFloat16` without FP32 upcast halves
  device weights (`4.60 → 2.30 GiB`), cuts peak RSS nearly in half (`6.37 →
  3.31 GiB`), cuts model load time by **33%** (`1.88 → 1.25 s`), delivers
  **2.0×–2.1× faster decode** than CPU NEON (`30.1–31.9` vs `14.9–15.1 tok/s`),
  **3.5× faster 27–29-token prefill** (`113–121` vs `387–425 ms`), **+86%
  shared two-session throughput** (`33.1` vs `17.8 tok/s`), and **100% exact
  greedy parity** across all 33 sustained tokens (including close-tie position
  11 ID `95551`) with **`5.15e-5` max error vs frozen Transformers**.
- **Optional `Float16` KV cache (`bf16-kv16`, `fp16-kv16`):** Halves per-session
  KV device memory (`4.98 → 2.49 MiB` for 38 positions) with negligible latency
  change at short contexts (`30.2 tok/s` sustained) and **100% exact greedy
  parity (33/33)**; logit error vs CPU rises modestly (`p50 ~5e-4`, `max
  ~1.1e-2` over 32 steps).
- **Block-quantized projections (`Q8_0` and `Q4_0`):** Quantizing the 112
  transformer layer projections while keeping the tied `[128256, 2048]`
  embedding/classifier in a single `BFloat16` allocation (for `GatherRows`
  compatibility) and norms in `Float32`:
  - **`Q8_0` (`1.45 GiB` device, `2.49 GiB` peak RSS):** **`43.2–46.2 tok/s`**
    single-session decode (**2.9×–3.1× CPU**), **`49.1 tok/s`** shared two-lane
    (**+178% vs CPU**), **`131–140 ms`** 27–29-token prefill (**−67% to
    −69%**), and **100% exact greedy parity (33/33)** including position 11 ID
    `95551` (margin `0.0118` vs `0.0433` in FP32/BF16).
  - **`Q4_0` (`1.00 GiB` device, `2.06 GiB` peak RSS):** **`56.9–64.1 tok/s`**
    single-session decode (**3.9×–4.3× CPU**), **`67.0 tok/s`** shared two-lane
    (**+281% vs CPU**), **`145–155 ms`** 27–29-token prefill (**−63% to
    −64%**). Matches all 3/3 27/28/29 prefill winners, but diverges at token 3
    of the short/sustained trajectory (`" Paris. The capital of the United
    Kingdom"` vs `" Paris. It is the most populous city"`), reflecting expected
    4-bit quantization loss.
- **Note on `v0.12.0` → `v0.12.1`:** Initial LL-06 testing on `v0.12.0`
  uncovered a Write-After-Read threadgroup-memory race on `shared[0]` in
  `nn_rms_norm_row` under concurrent sessions (`independent_sessions` failed
  4/4). Upstream fixed this in `neuralnetwork#153` and released immutable
  `v0.12.1`, after which `independent_sessions` passed 5/5 and all 378 paired
  campaign processes completed with zero concurrency errors.

---

## Verified Versions, Checksums, and Hardware

| Input | Verified identity |
| --- | --- |
| Llama base `main` | `8a3f4f1f78510cdc9c5c1cec4096e8df2faba44f` (LL-05 merge) |
| `neuralnetwork` release | `github.com/itsmontoya/neuralnetwork@v0.12.1` |
| `neuralnetwork` tag commit | `32f83b827469b384e3a2546dc9ef4a5482e2c479` (`refs/tags/v0.12.1`) |
| `neuralnetwork` module checksum | `h1:y4YSQZ9BaK1tM301y1lqSD+A18j7NAYtYi2i8+4Q0yY=` |
| `neuralnetwork` `go.mod` checksum | `h1:SafbkuDehVErsuJAL5FHFsLhKJbeh/OI3ItvbmwDafw=` |
| Pinned model snapshot | `meta-llama/Llama-3.2-1B` revision `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| Manifest SHA-256 | `3404dfda97c69cf3a9a3cf15dc4db7017cb049da07462c52f2e3004f6e12ab64` |
| Frozen reference SHA-256 | `3a66ddc5cbdef45e397e47a6bd992676ab3d35234a8b389908f721b05466d361` |
| Checkpoint (`model.safetensors`) SHA-256 | `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a` |
| Hardware | MacBook Air `Mac15,13`, Apple M3 (4P+4E CPU, 10-core GPU, Metal 4), 24 GiB unified memory |
| OS and SDK | macOS `26.5.2 (25F84)`, Darwin `arm64`, Command Line Tools SDK `14.4` |
| Go configuration | `go1.27.3 darwin/arm64`, `GOARM64=v8.0`, `GOMAXPROCS=8`, `CGO_ENABLED=1`, `GOWORK=off`, empty `GOFLAGS` |

The historical LL-04 FP32 campaign (`v0.11.2`) is preserved untouched in
[GPU_PERFORMANCE_REPORT.md](GPU_PERFORMANCE_REPORT.md).

---

## Paired CPU vs. Metal Results Across All Precision Modes (`v0.12.1`)

Methodology: 3 alternating fresh-process pairs per workload across all 9
workloads (`short`, `sustained`, `27`, `28`, `29`, `concurrent-prefill`,
`shared`, `independent`, `recovery`) for each of the 7 Metal precision modes
(**189 pairs / 378 fresh processes total**), rotated workload order across
rounds, 2-second cooldown between processes, macOS `/usr/bin/time -l` with a
16 GiB per-process peak RSS cap, and no samples dropped. Each Metal mode is
paired directly against fresh CPU NEON processes in the same campaign.

### Latency, Throughput, and Load Time (Medians Across 3 Pairs per Mode)

| Mode | Load (s) | Short Prefill (ms) | Short Decode (ms / tok/s) | Sustained Decode (ms / tok/s, pos 6–37) | 27 / 28 / 29 Prefill (ms) | Conc. Prefill (ms) | Shared 2-Lane (tok/s) | Indep. 2-Model (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CPU NEON FP32** (pooled baseline) | 1.88 | 102–107 | 67–69 / **14.9** | 67.1 / **14.8** | 387–413 / 406–432 / 425–443 | 154–166 | 17.6–17.9 | 9.2–17.5 |
| **Metal FP32 (`v0.12.1`)** | 1.91 | 81.95 (−23.7%) | 70.41 / 14.18 (+2.9%) | 72.25 / 13.80 (+8.1%) | 280.8 / 281.5 / 301.5 (−31% to −33%) | 142.1 (−16.6%) | 16.54 (−6.0%) | 16.44 (+73.3%) |
| **Metal BF16 (native)** | **1.25** | **36.48 (−64.5%)** | **31.28 / 31.87 (−52.7%)** | **33.10 / 30.12 (−50.6%)** | **113.4 / 115.4 / 121.2 (−71% to −72%)** | 65.65 (−58.3%) | **33.10 (+86.0%)** | **33.38 (+100.8%)** |
| **Metal BF16 + FP16 KV** | **1.28** | **36.22 (−65.8%)** | **31.05 / 32.10 (−53.8%)** | **33.03 / 30.19 (−50.8%)** | **112.1 / 115.1 / 120.9 (−71% to −73%)** | 68.65 (−56.8%) | **32.74 (+83.4%)** | **32.47 (+84.4%)** |
| **Metal FP16** | 3.44 | 36.55 (−65.9%) | 30.99 / 32.16 (−55.1%) | 34.37 / 28.78 (−49.6%) | 112.5 / 115.5 / 120.0 (−72% to −74%) | **63.44 (−60.6%)** | 32.85 (+86.5%) | 32.82 (+95.1%) |
| **Metal FP16 + FP16 KV** | 3.48 | 36.77 (−63.5%) | 31.21 / 31.92 (−53.3%) | 35.07 / 28.31 (−49.0%) | 112.9 / 114.6 / 118.4 (−72% to −73%) | 66.92 (−56.6%) | 32.47 (+83.1%) | 32.31 (+214.1%) |
| **Metal Q8_0** | 3.99 | 36.48 (−64.3%) | **21.56 / 46.15 (−67.4%)** | **23.07 / 43.16 (−65.9%)** | 130.9 / 134.1 / 139.8 (−67% to −69%) | 65.45 (−58.8%) | **49.06 (+177.8%)** | **49.03 (+179.7%)** |
| **Metal Q4_0** | 4.03 | 39.78 (−61.3%) | **15.49 / 64.07 (−77.0%)** | **17.47 / 56.93 (−74.1%)** | 144.9 / 148.4 / 154.7 (−63% to −64%) | 71.33 (−54.9%) | **67.04 (+281.3%)** | **67.18 (+611.4%)** |

Notes:
- Percentages in parentheses are the paired median `Metal/CPU - 1` from each
  mode's alternating campaign (negative latency % = Metal faster; positive
  throughput % = Metal faster).
- `metal-fp16` (`3.44 s`), `metal-q8_0` (`3.99 s`), and `metal-q4_0` (`4.03 s`)
  perform host-side FP32→FP16/Q8_0/Q4_0 conversion/packing at load time from the
  BF16 safetensors file. `metal-bf16` (`1.25 s`) streams raw `uint16` BF16 bits
  directly from disk to Metal without FP32 upcast, cutting load time by **33%**
  vs CPU and **3.2×** vs Q8_0/Q4_0.
- **Recovery workload (all 7 modes):** Every mode rejected out-of-range token
  IDs (`1.6–6.3 µs` warmed) and pre-canceled contexts (`0.5–2.3 µs` warmed),
  preserved the KV cursor, and reproduced its exact deterministic 8-token
  trajectory on retry.

---

## Memory Footprint and Device Residency Accounting

| Mode | Device Model Bytes (Weights + RoPE) | Sustained KV Bytes (2 slots, cap 38) | Short Peak / Post-Close RSS (GiB) | Indep. 2-Model Peak / Post-Close RSS (GiB) | Exact Weight Layout |
| --- | ---: | ---: | ---: | ---: | --- |
| **CPU NEON FP32** | 0 (4.60 GiB Go heap) | 2,490,368 (host, 1 slot) | 6.37 / 6.37 | 11.53 / 9.89 | All FP32 on Go heap |
| **Metal FP32** | `4,943,388,672` (4.60 GiB) | `4,980,736` (4.75 MiB) | 6.60 / 5.62 | 7.82 / 1.06 | All FP32 (tied emb/classifier shared) |
| **Metal BF16** | **`2,471,759,872` (2.30 GiB, −50.0%)** | `4,980,736` (4.75 MiB) | **3.31 / 2.82** | **5.64 / 5.04** | All BF16 (direct upload) + FP32 RoPE (`131,072 B`) |
| **Metal BF16 + FP16 KV** | **`2,471,759,872` (2.30 GiB, −50.0%)** | **`2,490,368` (2.38 MiB, −50%)** | **3.31 / 2.82** | **5.62 / 5.01** | All BF16 + FP16 KV + FP32 RoPE |
| **Metal FP16** | `2,471,759,872` (2.30 GiB, −50.0%) | `4,980,736` (4.75 MiB) | 4.30 / 3.81 | 6.65 / 5.77 | All FP16 + FP32 RoPE |
| **Metal FP16 + FP16 KV** | `2,471,759,872` (2.30 GiB, −50.0%) | `2,490,368` (2.38 MiB, −50%) | 4.30 / 3.81 | 6.65 / 6.00 | All FP16 + FP16 KV + FP32 RoPE |
| **Metal Q8_0** | **`1,559,633,920` (1.45 GiB, −68.5%)** | `4,980,736` (4.75 MiB) | **2.49 / 2.00** | **3.98 / 3.46** | 112 layer projs Q8_0 (`1,033,895,936 B`) + tied emb/cls BF16 (`525,336,576 B`) + 33 norms FP32 (`270,336 B`) + RoPE (`131,072 B`) |
| **Metal Q4_0** | **`1,073,094,656` (1.00 GiB, −78.3%)** | `4,980,736` (4.75 MiB) | **2.06 / 2.03** | **3.13 / 2.64** | 112 layer projs Q4_0 (`547,356,672 B`) + tied emb/cls BF16 (`525,336,576 B`) + 33 norms FP32 (`270,336 B`) + RoPE (`131,072 B`) |

- In `Q8_0` and `Q4_0`, `gpu.Scope.GatherRows` explicitly rejects block-quantized
  tensors (`docs/adr-gpu-quantization.md`), so the tied `[128256, 2048]`
  embedding/classifier table stays in a **single shared `BFloat16` allocation**
  (`525,336,576 B`), uploaded directly from the BF16 checkpoint without FP32
  upcast. This preserves single-allocation tied sharing and protects classifier
  logit accuracy.

---

## Numerical Accuracy and Greedy Token Trajectories

### Greedy Parity and Position 11 (`95551`) Margin

- **Reference 8-token short trajectory:** `[12366, 13, 1102, 374, 279, 1455, 95551, 3363]` (`" Paris. It is the most populous city"`)
- **Reference 33-token sustained trajectory (positions 5–37):** `[12366, 13, 1102, 374, 279, 1455, 95551, 3363, 304, 9822, 323, 279, 3224, 753, 6864, 323, 7928, 3363, 13, 12366, 374, 279, 1455, 95551, 16036, 3158, 304, 279, 7665, 9323, 323, 279, 1917]`
- **Reference 27/28/29 prefill winners:** `[9822]`, `[374]`, `[12366]`

| Mode | Short (8) | Sustained (33) | 27/28/29 | Pos 11 Winner & Margin (frozen: `95551` by `0.043341`) | Frozen Ref Error (p50 / p95 / max / max frac) | Full Paired Sustained vs CPU (p50 / p95 / max / RMSE) |
| --- | :---: | :---: | :---: | --- | --- | --- |
| **CPU NEON FP32** | **8/8** | **33/33** | **3/3** | `95551` by `0.043350` | `6.68e-6` / `2.29e-5` / `4.96e-5` / `0.0316` | baseline |
| **Metal FP32 (`v0.12.1`)** | **8/8** | **33/33** | **3/3** | `95551` by `0.043377` | **`5.66e-6` / `1.91e-5` / `3.81e-5` / `0.0154`** | `1.28e-5` / `5.68e-5` / `1.64e-4` / `2.62e-5` |
| **Metal BF16** | **8/8** | **33/33** | **3/3** | `95551` by `0.043320` | **`6.20e-6` / `2.38e-5` / `5.15e-5` / `0.0254`** | `1.03e-5` / `4.22e-5` / `1.50e-4` / `1.99e-5` |
| **Metal BF16 + FP16 KV** | **8/8** | **33/33** | **3/3** | `95551` by `0.041950` | `4.52e-4` / `1.61e-3` / `2.77e-3` / `1.681` | `7.78e-4` / `3.50e-3` / `1.11e-2` / `1.58e-3` |
| **Metal FP16** | **8/8** | **33/33** | **3/3** | `95551` by `0.043322` | `6.20e-6` / `2.48e-5` / `5.53e-5` / `0.0280` | `1.05e-5` / `4.39e-5` / `1.52e-4` / `2.05e-5` |
| **Metal FP16 + FP16 KV** | **8/8** | **33/33** | **3/3** | `95551` by `0.042166` | `4.32e-4` / `1.47e-3` / `2.16e-3` / `1.422` | `8.86e-4` / `3.54e-3` / `1.29e-2` / `1.67e-3` |
| **Metal Q8_0** | **8/8** | **33/33** | **3/3** | **`95551` by `0.011763`** | `2.77e-2` / `1.20e-1` / `3.40e-1` / `143.2` | `5.97e-2` / `2.66e-1` / `8.46e-1` / `1.24e-1` |
| **Metal Q4_0** | 2/8 | 2/33 | **3/3** | Diverged at step 2 (`578` vs `1102`) | `3.33e-1` / `1.01e+0` / `1.28e+0` / `746.9` (matched prefix steps) | `2.30e+0` / `8.40e+0` / `2.41e+1` / `4.09e+0` (diverged rollout) |

- **FP32 `v0.12.1` vs `v0.11.3` numerical note:** Parallel SIMD `RMSNorm`
  reduces floating-point summation error, making FP32 Metal **closer to the
  frozen Transformers reference** (`max=3.81e-5`, `max_fraction=0.0154`) than
  both `v0.11.3` (`4.77e-5` / `0.0208`) and sequential CPU (`4.96e-5` /
  `0.0316`). Because Metal's reduction order now differs slightly from CPU's
  sequential reduction, full paired sustained decode against CPU has
  `max_fraction = 1.379` (`max = 1.64e-4` on near-zero logits around `-0.009`
  in late steps), while `short`, `27`, `28`, `29`, `concurrent-prefill`, and
  `recovery` remain well under `1.0` (`0.18–0.55`).
- **`Q4_0` greedy trajectory:** Produces `[12366, 13, 578, 6864, 315, 279,
  3723, 15422]` (`" Paris. The capital of the United Kingdom"`) on the 8-token
  short prompt — coherent English that diverges at token 3 (`578` `" The"` vs
  `1102` `" It"`), consistent with standard 4-bit block quantization.

---

## Native Metal Profiling and Per-Stage GPU Attribution

All diagnostic processes across `metal`, `metal-bf16`, `metal-bf16-kv16`,
`metal-q8_0`, and `metal-q4_0` matched their primary campaign `.f32` logit
artifacts byte for byte.

### Warm Sustained Decode GPU Stage Sums (`counters-1`, 32 decode commands/mode, ms)

| Compute Group | Dispatches / tok | FP32 (`v0.11.3`) | FP32 (`v0.12.1`) | BF16 (`v0.12.1`) | BF16+KV16 (`v0.12.1`) | Q8_0 (`v0.12.1`) | Q4_0 (`v0.12.1`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FFN gate, up, down | 48 | 40.95 | 40.67 | 17.99 | 18.76 | **9.89** | **5.25** |
| **RMSNorm** | 33 | **18.51** | **0.22** | **0.20** | **0.20** | **0.21** | **0.20** |
| Attention Q, K, V, O | 64 | 13.61 | 13.38 | 4.11 | 4.10 | **2.42** | **1.36** |
| Classifier (tied) | 1 | 12.07 | 11.75 | 5.67 | 5.68 | 5.75 | 5.73 |
| Causal Attention | 16 | 3.70 | 3.68 | 3.69 | 3.77 | 3.67 | 3.68 |
| Residual / RoPE / SwiGLU / Gather / KV16 | 80–97 | 0.23 | 0.45 | 0.21 | 0.24 | 0.22 | 0.20 |
| **Command-only GPU interval (median)** | — | **90.43** | **70.71** | **32.12** | **32.40** | **23.14** | **16.05** |
| **Sustained peak `currentAllocatedSize`** | — | 4.951 GB | 4.951 GB | **2.479 GB** | **2.476 GB** | **1.567 GB** | **1.080 GB** |

Key takeaways from stage attribution:
1. **`RMSNorm` is no longer a bottleneck:** `18.51 ms` → `0.20–0.22 ms` (**85×**
   reduction across all modes).
2. **In `Q4_0`, the tied BF16 classifier (`5.73 ms`, 35% of GPU time) and
   `CausalAttention` (`3.68 ms`, 23%) now dominate** over the 112 quantized
   transformer projections (`5.25 + 1.36 = 6.61 ms` combined). Future work on
   quantized classifier/gather or FlashAttention would directly target those
   remaining ~9.4 ms.
3. **Every mode maintains 1 command buffer, 1 wait, and 1 logit download
   (`513,024 B`) per step** with zero per-layer host crossings.

---

## Supported Modes, Tradeoffs, and Recommendations

| Mode | Recommended Use | Tradeoff Summary |
| --- | --- | --- |
| **CPU NEON FP32** (default) | Authoritative baseline, portable builds, `ForwardWithTrace` | Zero GPU dependency; `14.8 tok/s` sustained, `6.37 GiB` peak RSS. |
| **Metal FP32** | Strict FP32 GPU reference & multi-row prefill | `13.8 tok/s` sustained, **−31–33% prefill**, `4.60 GiB` device, better than CPU vs frozen ref (`3.81e-5` max). |
| **Metal BF16** (recommended GPU default) | Best overall speed + lossless accuracy on BF16 checkpoints | **30.1 tok/s (2.0× CPU)**, **−71–72% prefill**, **1.25 s load (−33%)**, `2.30 GiB` device, **33/33 greedy**, `5.15e-5` max frozen error. |
| **Metal BF16 + FP16 KV** | Long-context sessions needing half KV memory | Same **30.2 tok/s** throughput, **−50% KV bytes**, **33/33 greedy**, `~1e-3` KV rounding noise. |
| **Metal FP16 (+ FP16 KV)** | FP16 checkpoints or ONNX-aligned FP16 testing | **28.3–28.8 tok/s**, **33/33 greedy**; slower load (`3.44 s`) from BF16 checkpoints due to host conversion. |
| **Metal Q8_0** | High-throughput decode with exact greedy parity | **43.2 tok/s (2.9× CPU)**, `49.1 tok/s` shared 2-lane, **1.45 GiB device**, **33/33 greedy**; `3.99 s` load (host quantization). |
| **Metal Q4_0** | Maximum throughput / minimum memory (1.00 GiB device) | **56.9 tok/s (3.9× CPU)**, `67.0 tok/s` shared 2-lane, **1.00 GiB device**; 4-bit quality tradeoff (diverges at token 3 on short prompt); `4.03 s` load. |

### CPU Fallback and Failure Semantics

- Metal is never selected implicitly by default CPU entrypoints (`LoadHFModel`,
  default CLI flags).
- Explicit Metal requests on unsupported builds (`purego`, `CGO_ENABLED=0`,
  non-Darwin) or hosts without Metal return an explicit `gpu.ErrUnavailable`
  error rather than silently falling back to CPU. Invalid `MetalOptions`
  return `gpu.ErrInvalid`; `Q8_0`/`Q4_0` models with `Dim % 32 != 0` or
  `FFNDim % 32 != 0` return `gpu.ErrInvalid`.

---

## Reproducibility Commands

All raw binaries, `.f32` logit dumps, pprof profiles, and validation logs are
gitignored under `.benchmarks/gpu-ll-06/2026-11-25-m3-v0.12.1/`.

```sh
export GOWORK=off GOMAXPROCS=8 GOARM64=v8.0 CGO_ENABLED=1
unset LLAMA_32_1B_MODEL_DIR LLAMA_PAIRED_BACKEND

# 1. Portability, race, and 1B correctness matrix
go mod verify && go fmt ./... && go vet ./...
go test -p 1 -count=1 -race ./...
CGO_ENABLED=0 go test -p 1 -count=1 ./...
CGO_ENABLED=0 go test -p 1 -count=1 -tags=purego ./...
CGO_ENABLED=0 go test -p 1 -count=1 -tags=metal ./...
CGO_ENABLED=0 go test -p 1 -count=1 -tags='metal purego' ./...
CGO_ENABLED=1 go test -p 1 -count=1 -tags='metal purego' ./...
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build ./...
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -tags=metal ./...
go vet -tags=metal ./...
go test -p 1 -count=1 -race -tags=metal ./...
LLAMA_32_1B_MODEL_DIR="$model_dir" go test -p 1 -count=1 -timeout 20m -v ./...
LLAMA_32_1B_MODEL_DIR="$model_dir" go test -p 1 -count=1 -timeout 20m -v -tags=metal ./...

# 2. Paired CPU vs Metal campaigns and native Metal diagnostics
base=.benchmarks/gpu-ll-06/reproduction
for mode in metal metal-bf16 metal-bf16-kv16 metal-fp16 metal-fp16-kv16 metal-q8_0 metal-q4_0; do
  python3 -B benchmarks/metal/campaign.py "$model_dir" "$base/$mode" --pairs 3 --cooldown 2 --metal-backend "$mode"
  python3 -B benchmarks/metal/summarize.py "$base/$mode" > "$base/$mode/summary.txt"
done
for mode in metal metal-bf16 metal-bf16-kv16 metal-q8_0 metal-q4_0; do
  python3 -B benchmarks/metal/diagnose.py "$base/$mode" "$base/$mode/diagnostics" --repeats 2
  python3 -B benchmarks/metal/summarize_profiles.py "$base/$mode/diagnostics" > "$base/$mode/profile-summary.txt"
done
```
