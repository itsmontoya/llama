# Llama 3.2 1B Multi-Precision GPU Baseline (`neuralnetwork v0.13.3`)

This document records the verified CPU-vs-Metal baseline for
`github.com/itsmontoya/llama` integrating
`github.com/itsmontoya/neuralnetwork@v0.13.3` (task `LL-07`), and compares
directly against the prior `v0.12.1` (`LL-06`) baseline.

## Executive Summary

- **CPU remains the default and authoritative baseline.** Resident Metal
  execution is explicitly opt-in (`-tags=metal` with `CGO_ENABLED=1` on Apple
  Silicon macOS, via CLI flags or `LoadHFModelMetalAttention` /
  `LoadHFModelMetalAttentionWithOptions`).
- **Parallel SIMD `CausalAttention` (`v0.13.3` / `NN-20`):** Replacing the
  single-thread-per-head attention kernel with a threadgroup-cooperative
  parallel SIMD online-softmax kernel cut 16 per-token `CausalAttention`
  dispatches from **`3.67–3.77 ms` → `0.13–0.18 ms`** on Apple M3 (**21×–28×**
  faster across all modes):
  - **FP32 Metal** sustained decode rose from `72.25 ms` (`13.80 tok/s`) on
    `v0.12.1` to **`66.63 ms` (`14.96 tok/s`)** on `v0.13.3`, reaching parity
    with 8-core NEON CPU (`66.74 ms`, `14.96 tok/s`) while remaining closer to
    the frozen Transformers reference (`p50 = 5.25e-6`, `max = 4.08e-5` vs CPU
    `p50 = 6.68e-6`, `max = 4.96e-5`).
  - **Native `BFloat16`** sustained decode rose from `30.12 tok/s` to
    **`34.10 tok/s` (`29.19 ms`, 2.3× CPU)** with **`1.25 s` load time
    (−34% vs CPU)**, **`2.30 GiB` device weights**, **100% exact greedy parity
    (33/33)**, and `6.20e-6` p50 / `5.34e-5` max frozen error.
  - **`BF16 + FP16 KV`** rose from `30.19 tok/s` to **`33.76 tok/s`**, halving
    per-session KV device memory (`4.98 → 2.49 MiB` for 38 positions) with
    **100% exact greedy parity (33/33)**.
- **Quantized tied `[128256, 2048]` embedding/classifier (`v0.13.3` / `NN-19`):**
  Using `Scope.GatherRows` (`Q8_0`/`Q4_0 → Float32` on-the-fly dequantization),
  the tied `[128256, 2048]` table now shares a **single quantized allocation**
  (`279.1 MB` Q8_0 / `147.8 MB` Q4_0 vs `525.3 MB` BF16), eliminating the
  `5.73–5.75 ms` BF16 classifier bottleneck:
  - **`Q8_0` (`1.22 GiB` device, −73.4% vs FP32):** Classifier stage drops
    `5.75 → 3.04 ms` (**1.89×**). Sustained decode jumps from `43.16` to
    **`59.69 tok/s`** (**4.0× CPU**), shared 2-lane hits **`64.48 tok/s`
    (+260.5% vs CPU)**, and **100% exact greedy parity (33/33)** is preserved
    including position 11 ID `95551` (margin `0.0535` vs `0.0434` frozen ref).
    Logit error vs frozen ref rises modestly (`p50 = 0.0312`, `max = 0.347` vs
    `0.0277` / `0.340` in `v0.12.1`) and host-side quantization extends load
    time to `4.71 s`.
  - **`Q4_0` (`0.65 GiB` device, −85.9% vs FP32):** Classifier stage drops
    `5.73 → 1.58 ms` (**3.62×**). Sustained decode jumps from `56.93` to
    **`104.93 tok/s`** (**7.0× CPU**; `8.79 ms` / `>113 tok/s` GPU-only
    interval without counters), shared 2-lane hits **`117.96 tok/s` (+569% vs
    CPU)**, and independent 2-model hits **`118.68 tok/s`**. Matches 3/3
    27/28/29 prefill winners, but diverges at token 3 of the short/sustained
    trajectory (`" Paris. The capital of the United Kingdom"` vs `" Paris. It is
    the most populous city"`), reflecting 4-bit quantization of both projections
    and the tied embedding/classifier (`p50 = 0.339` / `max = 1.466` frozen
    error on matched prefix steps).

---

## Verified Versions, Checksums, and Hardware

| Input | Verified identity |
| --- | --- |
| Llama base `main` | `75bced8` (PR #38) |
| `neuralnetwork` release | `github.com/itsmontoya/neuralnetwork@v0.13.3` |
| `neuralnetwork` tag commit | `2b0a5f7f9fd4f7eff5f99a77d34818d7e3fa524f` (`refs/tags/v0.13.3`) |
| `neuralnetwork` module checksum | `h1:jG/KLENaNXrZO2hKBllVzzPeT25gZlxXmZrNor3CYMM=` |
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

## Paired CPU vs. Metal Results (`v0.13.3` vs `v0.12.1`)

Methodology: 3 alternating fresh-process pairs per workload across all 9
workloads (`short`, `sustained`, `27`, `28`, `29`, `concurrent-prefill`,
`shared`, `independent`, `recovery`) for the 5 primary Metal precision modes
(`metal`, `metal-bf16`, `metal-bf16-kv16`, `metal-q8_0`, `metal-q4_0` — **135
pairs / 270 fresh processes total**), rotated workload order across rounds,
2-second cooldown between processes, macOS `/usr/bin/time -l`, and no samples
dropped. `metal-fp16` and `metal-fp16-kv16` were verified in the correctness
suites (`TestLlama32MetalReducedPrecision`,
`TestMetalAttentionReducedPrecisionSynthetic`) and were not re-benchmarked in
the `v0.13.3` paired campaign (their `v0.12.1` results are noted below for
reference).

### Latency, Throughput, and Load Time (Medians Across 3 Pairs per Mode)

| Mode | Load (s) | Short Prefill (ms) | Short Decode (ms / tok/s) | Sustained Decode (ms / tok/s, pos 6–37) | 27 / 28 / 29 Prefill (ms) | Conc. Prefill (ms) | Shared 2-Lane (tok/s) | Indep. 2-Model (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CPU NEON FP32** (pooled `v0.13.3`) | 1.90 | 100–104 | 66–67 / **15.0** | 66.7 / **15.0** | 386–391 / 412–432 / 424–457 | 158–162 | 17.6–18.2 | 9.5–15.1 |
| **Metal FP32 (`v0.13.3`)** | 1.96 | 80.24 (−20.1%) | 68.51 / 14.57 (+3.1%) | **66.63 / 14.96 (+2.1%)** | 271.1 / 271.2 / 290.9 (−30% to −35%) | 137.1 (−15.2%) | 17.44 (−5.4%) | 17.29 (+13.5%) |
| *Metal FP32 (`v0.12.1`)* | *1.91* | *81.95* | *70.41 / 14.18* | *72.25 / 13.80* | *280.8 / 281.5 / 301.5* | *142.1* | *16.54* | *16.44* |
| **Metal BF16 (`v0.13.3`)** | **1.25** | **35.49 (−65.4%)** | **29.33 / 33.97 (−55.8%)** | **29.19 / 34.10 (−56.4%)** | **107.6 / 108.5 / 115.0 (−72% to −75%)** | 64.74 (−58.9%) | **33.74 (+89.6%)** | **35.19 (+240.8%)** |
| *Metal BF16 (`v0.12.1`)* | *1.25* | *36.48* | *31.28 / 31.87* | *33.10 / 30.12* | *113.4 / 115.4 / 121.2* | *65.65* | *33.10* | *33.38* |
| **Metal BF16 + FP16 KV (`v0.13.3`)** | **1.24** | **36.51 (−65.4%)** | **30.01 / 33.14 (−54.8%)** | **29.51 / 33.76 (−55.8%)** | **108.5 / 110.7 / 116.1 (−72% to −73%)** | 66.86 (−58.2%) | **34.99 (+95.5%)** | **34.99 (+132.1%)** |
| *Metal BF16 + FP16 KV (`v0.12.1`)* | *1.28* | *36.22* | *31.05 / 32.10* | *33.03 / 30.19* | *112.1 / 115.1 / 120.9* | *68.65* | *32.74* | *32.47* |
| *Metal FP16 (`v0.12.1`)* | *3.44* | *36.55* | *30.99 / 32.16* | *34.37 / 28.78* | *112.5 / 115.5 / 120.0* | *63.44* | *32.85* | *32.82* |
| *Metal FP16 + FP16 KV (`v0.12.1`)* | *3.48* | *36.77* | *31.21 / 31.92* | *35.07 / 28.31* | *112.9 / 114.6 / 118.4* | *66.92* | *32.47* | *32.31* |
| **Metal Q8_0 (`v0.13.3`)** | 4.71 | **33.03 (−67.9%)** | **17.07 / 58.13 (−74.3%)** | **16.65 / 59.69 (−75.1%)** | 124.3 / 126.6 / 131.9 (−68% to −69%) | **59.27 (−62.5%)** | **64.48 (+260.5%)** | **64.33 (+571.3%)** |
| *Metal Q8_0 (`v0.12.1`)* | *3.99* | *36.48* | *21.56 / 46.15* | *23.07 / 43.16* | *130.9 / 134.1 / 139.8* | *65.45* | *49.06* | *49.03* |
| **Metal Q4_0 (`v0.13.3`)** | 4.81 | **34.30 (−66.8%)** | **10.10 / 97.11 (−84.8%)** | **9.43 / 104.93 (−86.1%)** | 136.8 / 139.7 / 147.1 (−65% to −68%) | 62.73 (−60.7%) | **117.96 (+569.0%)** | **118.68 (+943.4%)** |
| *Metal Q4_0 (`v0.12.1`)* | *4.03* | *39.78* | *15.49 / 64.07* | *17.47 / 56.93* | *144.9 / 148.4 / 154.7* | *71.33* | *67.04* | *67.18* |

Notes:
- Percentages in parentheses are the paired median `Metal/CPU - 1` from each
  mode's alternating campaign (negative latency % = Metal faster; positive
  throughput % = Metal faster).
- `metal-q8_0` (`4.71 s`) and `metal-q4_0` (`4.81 s`) take ~0.7–0.8 s longer to
  load than in `v0.12.1` (`3.99–4.03 s`) because the `[128256, 2048]` tied
  embedding/classifier table is now quantized on the host at load time instead
  of streamed as raw BF16 bits. `metal-bf16` (`1.25 s`) remains the fastest
  loader.
- **Recovery workload (all 5 modes):** Every mode rejected out-of-range token
  IDs (`1.3–3.5 µs` warmed) and pre-canceled contexts (`0.6–1.0 µs` warmed),
  preserved the KV cursor, and reproduced its exact deterministic 8-token
  trajectory on retry.

---

## Memory Footprint and Device Residency Accounting (`v0.13.3`)

| Mode | Device Model Bytes (Weights + RoPE) | Sustained KV Bytes (2 slots, cap 38) | Short Peak / Post-Close RSS (GiB) | Indep. 2-Model Peak / Post-Close RSS (GiB) | Exact Weight Layout |
| --- | ---: | ---: | ---: | ---: | --- |
| **CPU NEON FP32** | 0 (4.60 GiB Go heap) | 2,490,368 (host, 1 slot) | 6.37 / 6.37 | 11.29 / 9.87 | All FP32 on Go heap |
| **Metal FP32** | `4,943,388,672` (4.60 GiB) | `4,980,736` (4.75 MiB) | 6.60 / 5.62 | 7.90 / 1.76 | All FP32 (tied emb/classifier shared) |
| **Metal BF16** | **`2,471,759,872` (2.30 GiB, −50.0%)** | `4,980,736` (4.75 MiB) | **3.31 / 2.82** | **5.64 / 4.84** | All BF16 (direct upload) + FP32 RoPE (`131,072 B`) |
| **Metal BF16 + FP16 KV** | **`2,471,759,872` (2.30 GiB, −50.0%)** | **`2,490,368` (2.38 MiB, −50%)** | **3.31 / 2.82** | **5.64 / 5.01** | All BF16 + FP16 KV + FP32 RoPE |
| **Metal Q8_0 (`v0.13.3`)** | **`1,313,382,400` (1.22 GiB, −73.4%)** | `4,980,736` (4.75 MiB) | 3.24 / 2.97 | **4.46 / 4.05** | 112 layer projs Q8_0 (`1,033,895,936 B`) + tied emb/cls Q8_0 (`279,085,056 B`) + 33 norms FP32 (`270,336 B`) + RoPE (`131,072 B`) |
| **Metal Q4_0 (`v0.13.3`)** | **`695,508,992` (0.65 GiB, −85.9%)** | `4,980,736` (4.75 MiB) | **2.64 / 2.45** | **3.31 / 3.01** | 112 layer projs Q4_0 (`547,356,672 B`) + tied emb/cls Q4_0 (`147,750,912 B`) + 33 norms FP32 (`270,336 B`) + RoPE (`131,072 B`) |

- In `Q8_0` and `Q4_0`, `gpu.Scope.GatherRows` (`v0.13.3`) dequantizes directly
  from `Q8_0`/`Q4_0` into `Float32`, allowing the tied `[128256, 2048]`
  embedding/classifier table to live in a **single shared quantized allocation**
  (`279,085,056 B` Q8_0 / `147,750,912 B` Q4_0), saving `246.3 MB` (`Q8_0`)
  and `377.6 MB` (`Q4_0`) of device memory vs `v0.12.1`.

---

## Numerical Accuracy and Greedy Token Trajectories (`v0.13.3`)

### Greedy Parity and Position 11 (`95551`) Margin

- **Reference 8-token short trajectory:** `[12366, 13, 1102, 374, 279, 1455, 95551, 3363]` (`" Paris. It is the most populous city"`)
- **Reference 33-token sustained trajectory (positions 5–37):** `[12366, 13, 1102, 374, 279, 1455, 95551, 3363, 304, 9822, 323, 279, 3224, 753, 6864, 323, 7928, 3363, 13, 12366, 374, 279, 1455, 95551, 16036, 3158, 304, 279, 7665, 9323, 323, 279, 1917]`
- **Reference 27/28/29 prefill winners:** `[9822]`, `[374]`, `[12366]`

| Mode | Short (8) | Sustained (33) | 27/28/29 | Pos 11 Winner & Margin (frozen: `95551` by `0.043341`) | Frozen Ref Error (p50 / p95 / max / max frac) | Full Paired Sustained vs CPU (p50 / p95 / max / RMSE) |
| --- | :---: | :---: | :---: | --- | --- | --- |
| **CPU NEON FP32** | **8/8** | **33/33** | **3/3** | `95551` by `0.043350` | `6.68e-6` / `2.29e-5` / `4.96e-5` / `0.0316` | baseline |
| **Metal FP32 (`v0.13.3`)** | **8/8** | **33/33** | **3/3** | `95551` by `0.043362` | **`5.25e-6` / `1.72e-5` / `4.08e-5` / `0.0233`** | `1.11e-5` / `4.86e-5` / `1.69e-4` / `2.24e-5` |
| **Metal BF16 (`v0.13.3`)** | **8/8** | **33/33** | **3/3** | `95551` by `0.043318` | **`6.20e-6` / `2.48e-5` / `5.34e-5` / `0.0255`** | `1.00e-5` / `4.24e-5` / `1.48e-4` / `1.98e-5` |
| **Metal BF16 + FP16 KV (`v0.13.3`)** | **8/8** | **33/33** | **3/3** | `95551` by `0.042242` | `4.34e-4` / `1.61e-3` / `2.72e-3` / `1.852` | `7.75e-4` / `3.28e-3` / `1.12e-2` / `1.53e-3` |
| **Metal Q8_0 (`v0.13.3`, tied Q8_0)** | **8/8** | **33/33** | **3/3** | **`95551` by `0.053505`** | `3.12e-2` / `1.30e-1` / `3.47e-1` / `143.1` | `6.32e-2` / `2.75e-1` / `8.17e-1` / `1.28e-1` |
| **Metal Q4_0 (`v0.13.3`, tied Q4_0)** | 2/8 | 2/33 | **3/3** | Diverged at step 2 (`578` vs `1102`) | `3.39e-1` / `1.15e+0` / `1.47e+0` / `1136.5` (matched prefix steps) | `2.26e+0` / `8.39e+0` / `2.53e+1` / `4.04e+0` (diverged rollout) |

- **FP32 `v0.13.3` numerical note:** Parallel SIMD `CausalAttention` online
  softmax reduces floating-point summation error further, bringing FP32 Metal
  even closer to the frozen Transformers reference (`p50 = 5.25e-6`,
  `p95 = 1.72e-5`) than sequential CPU (`6.68e-6` / `2.29e-5`). Because
  Metal's reduction order differs slightly from CPU's sequential reduction, full
  paired sustained decode against CPU has `max_fraction = 1.314`
  (`max = 1.69e-4` on near-zero logits in late steps), while `short`, `27`,
  `28`, `29`, `concurrent-prefill`, and `recovery` remain well under `1.0`
  (`0.25–0.54`).
- **`Q8_0` / `Q4_0` tied-quantization quality impact:** Quantizing the tied
  `[128256, 2048]` embedding/classifier table increases `Q8_0` frozen `p50`
  only slightly (`0.0277 → 0.0312`) while preserving **33/33 exact greedy
  parity** and strengthening position 11's margin (`0.0535`). In `Q4_0`,
  frozen `max` on matched prefix steps rises from `1.28` (`v0.12.1`) to `1.47`
  (`v0.13.3`) and produces the same coherent English divergence at token 3
  (`[12366, 13, 578, 6864, 315, 279, 3723, 15422]`, `" Paris. The capital of
  the United Kingdom"`).

---

## Native Metal Profiling and Per-Stage GPU Attribution

All diagnostic processes across `metal`, `metal-bf16`, `metal-bf16-kv16`,
`metal-q8_0`, and `metal-q4_0` matched their primary campaign `.f32` logit
artifacts byte for byte.

### Warm Sustained Decode GPU Stage Sums (`counters-1`, 32 decode commands/mode, ms)

| Compute Group | Dispatches / tok | FP32 (`v0.12.1`) | FP32 (`v0.13.3`) | BF16 (`v0.13.3`) | BF16+KV16 (`v0.13.3`) | Q8_0 (`v0.13.3`) | Q4_0 (`v0.13.3`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FFN gate, up, down | 48 | 40.67 | 40.45 | 18.55 | 19.49 | 9.99 | 5.57 |
| **RMSNorm** | 33 | 0.22 | 0.21 | 0.20 | 0.20 | 0.21 | 0.22 |
| Attention Q, K, V, O | 64 | 13.38 | 13.34 | 4.19 | 4.23 | 2.39 | 1.44 |
| **Classifier (tied)** | 1 | 11.75 | 12.06 | 5.68 | 5.84 | **3.04** *(was 5.75)* | **1.58** *(was 5.73)* |
| **Causal Attention** | 16 | **3.68** | **0.13** | **0.13** | **0.18** | **0.13** | **0.14** |
| Residual / RoPE / SwiGLU / Gather / KV16 | 80–97 | 0.45 | 0.26 | 0.20 | 0.23 | 0.20 | 0.23 |
| **Command-only GPU interval (`counters-1`)** | — | 70.71 | 70.29 | 32.32 | 31.96 | **19.64** *(was 23.14)* | **11.98** *(was 16.05)* |
| **Command-only GPU interval (`command-1`, no counters)** | — | — | 66.56 | **28.35** | **28.75** | **15.77** | **8.79** |
| **Sustained peak `currentAllocatedSize`** | — | 4.951 GB | 4.951 GB | 2.479 GB | 2.477 GB | **1.321 GB** *(was 1.567 GB)* | **0.703 GB** *(was 1.080 GB)* |

Key takeaways from stage attribution:
1. **`CausalAttention` is no longer a bottleneck:** `3.67–3.77 ms` →
   **`0.13–0.18 ms`** (**21×–28×** reduction across all modes).
2. **Quantized tied classifier (`Q8_0`/`Q4_0`) cut classifier latency by 1.9×–3.6×**
   (`5.75 → 3.04 ms` in `Q8_0`, `5.73 → 1.58 ms` in `Q4_0`) and device memory by
   `246–378 MB`, while `Q8_0`/`Q4_0 → Float32` embedding `GatherRows` takes
   just `0.003–0.004 ms` (`3–4 µs`) per token.
3. **Every mode maintains 1 command buffer, 1 wait, and 1 logit download
   (`513,024 B`) per step** with zero per-layer host crossings.

---

## Supported Modes, Tradeoffs, and Recommendations

| Mode | Recommended Use | Tradeoff Summary |
| --- | --- | --- |
| **CPU NEON FP32** (default) | Authoritative baseline, portable builds, `ForwardWithTrace` | Zero GPU dependency; `15.0 tok/s` sustained, `6.37 GiB` peak RSS. |
| **Metal FP32** | Strict FP32 GPU reference & multi-row prefill | `15.0 tok/s` sustained, **−30–35% prefill**, `4.60 GiB` device, closer to frozen ref (`5.25e-6` p50) than CPU. |
| **Metal BF16** (recommended GPU default) | Best overall speed + lossless accuracy on BF16 checkpoints | **34.1 tok/s (2.3× CPU)**, **−72–75% prefill**, **1.25 s load (−34%)**, `2.30 GiB` device, **33/33 greedy**, `6.20e-6` p50 frozen error. |
| **Metal BF16 + FP16 KV** | Long-context sessions needing half KV memory | **33.8 tok/s**, **−50% KV bytes**, **33/33 greedy**, `~1e-3` KV rounding noise. |
| **Metal FP16 (+ FP16 KV)** | FP16 checkpoints or ONNX-aligned FP16 testing | **33/33 greedy**; slower load (`~3.4 s`) from BF16 checkpoints due to host conversion. |
| **Metal Q8_0** | High-throughput decode with exact greedy parity | **59.7 tok/s (4.0× CPU)**, `64.5 tok/s` shared 2-lane, **1.22 GiB device (−73%)**, **33/33 greedy**; `4.71 s` load (host quantization). |
| **Metal Q4_0** | Maximum throughput / minimum memory (0.65 GiB device) | **104.9 tok/s (7.0× CPU)**, `118.0 tok/s` shared 2-lane, **0.65 GiB device (−86%)**; 4-bit quality tradeoff (diverges at token 3 on short prompt); `4.81 s` load. |

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
gitignored under `.benchmarks/gpu-ll-07/2026-11-25-m3-v0.13.3/`.

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
base=.benchmarks/gpu-ll-07/reproduction
for mode in metal metal-bf16 metal-bf16-kv16 metal-q8_0 metal-q4_0; do
  python3 -B benchmarks/metal/campaign.py "$model_dir" "$base/$mode" --pairs 3 --cooldown 2 --metal-backend "$mode"
  python3 -B benchmarks/metal/summarize.py "$base/$mode" > "$base/$mode/summary.txt"
  python3 -B benchmarks/metal/diagnose.py "$base/$mode" "$base/$mode/diagnostics" --repeats 2
  python3 -B benchmarks/metal/summarize_profiles.py "$base/$mode/diagnostics" > "$base/$mode/profile-summary.txt"
done
```
