# Llama 3.2 1B Multi-Precision GPU Baseline (`neuralnetwork v0.17.0`)

This document records the verified CPU-vs-Metal baseline for
`github.com/itsmontoya/llama` integrating
`github.com/itsmontoya/neuralnetwork@v0.17.0` (task `LL-08`), and compares
directly against the prior `v0.13.3` (`LL-07`) baseline.

## Executive Summary

- **CPU remains the default and authoritative baseline.** Resident Metal
  execution is explicitly opt-in (`-tags=metal` with `CGO_ENABLED=1` on Apple
  Silicon macOS, via CLI flags or `LoadHFModelMetalAttention` /
  `LoadHFModelMetalAttentionWithOptions`).
- **What `v0.17.0` + `LL-08` delivered on Apple M3:**
  1. **`simdgroup_float8x8` tiled prefill across FP32/FP16/BF16/Q8_0/Q4_0/Q4_1:**
     - **27/28/29-token prefill** dropped **3.5×–3.6×** across FP32
       (`271.1–290.9 ms` → **`73.3–75.2 ms`**, **−80.0% to −82.2% vs CPU**),
       **1.8×–1.9×** in BF16/BF16+KV16 (`107.6–116.1 ms` → **`60.0–61.1 ms`**,
       **−84.1% to −85.7% vs CPU**), **2.8×–2.9×** in `Q8_0`
       (`124.3–131.9 ms` → **`45.2–45.8 ms`**), and **3.2×–3.4×** in `Q4_0`
       (`136.8–147.1 ms` → **`42.4–43.1 ms`**, **−89.0% to −90.3% vs CPU**).
       New **`Q4_1`** hits **`43.1–43.5 ms`** (**−88.9% to −89.9% vs CPU**).
  2. **FP32 `M=1` SIMD GEMV + `MTLComputeCommandEncoder` coalescing + reusable
     `metalSession` scratch (`LL-08`):**
     - **FP32 Metal** sustained decode jumped from `66.63 ms` (`14.96 tok/s`)
       on `v0.13.3` to **`54.99 ms` (`18.14 tok/s`)** on `v0.17.0` (**+21.3%
       vs `v0.13.3`**, **+19.6% paired vs 8-core NEON CPU**), with GPU-only
       interval dropping `66.56 → 56.92 ms`.
     - **242 compute dispatches** per decode step now coalesce into **17
       compute encoders** (down from 242 encoders in `v0.13.3`).
     - **`metalSession` scratch caching** eliminates per-token allocation/free
       churn of 15–16 `MTLBuffer`s across decode steps while preserving
       `Reset()`, `Close()`, and error/cancel recovery.
  3. **−50% `O(1)` in-place `KVCache`:**
     - Single-slot in-place KV cache halves per-session KV device memory across
       all modes (`4.75 → 2.38 MiB` for FP32 KV, `2.38 → 1.19 MiB` for FP16 KV
       at capacity 38).
  4. **Opt-in `gpu.Q4_1` asymmetric 4-bit mode (`metal-q4_1`):**
     - `0.72 GiB` (`772,743,168 B`) device model weights (**−84.4% vs FP32**),
       **`97.80 tok/s`** sustained decode (**10.13 ms/tok**, **−85.0% vs CPU**),
       **`108.14 tok/s`** shared 2-lane (**+507.9% vs CPU**), **`108.11 tok/s`**
       independent 2-model, and **`43.1–43.5 ms`** 27/28/29 prefill (3/3
       prefill winner parity).

---

## Verified Versions, Checksums, and Hardware

| Input | Verified identity |
| --- | --- |
| Llama base `main` | `de4901b` (PR #39) |
| `neuralnetwork` release | `github.com/itsmontoya/neuralnetwork@v0.17.0` |
| `neuralnetwork` tag commit | `0567cf1157df78b38633d9cf7beb90c539204727` (`refs/tags/v0.17.0`) |
| `neuralnetwork` module checksum | `h1:N/Tnl5xq8/E2dIjL3Ui0nYDV+9WkQoKc/+hlGhYmx58=` |
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

## Paired CPU vs. Metal Results (`v0.17.0` vs `v0.13.3`)

Methodology: 3 alternating fresh-process pairs per workload across all 9
workloads (`short`, `sustained`, `27`, `28`, `29`, `concurrent-prefill`,
`shared`, `independent`, `recovery`) for all 6 primary Metal precision modes
(`metal`, `metal-bf16`, `metal-bf16-kv16`, `metal-q8_0`, `metal-q4_1`,
`metal-q4_0` — **162 pairs / 324 fresh processes total**), rotated workload
order across rounds, 2-second cooldown between processes, macOS `/usr/bin/time -l`,
and no samples dropped.

### Latency, Throughput, and Load Time (Medians Across 3 Pairs per Mode)

| Mode | Load (s) | Short Prefill (ms) | Short Decode (ms / tok/s) | Sustained Decode (ms / tok/s, pos 6–37) | 27 / 28 / 29 Prefill (ms) | Conc. Prefill (ms) | Shared 2-Lane (tok/s) | Indep. 2-Model (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CPU NEON FP32** (pooled `v0.17.0`) | 1.88 | 98–107 | 64–69 / **14.9** | 66.6–67.8 / **14.8** | 376–426 | 158–162 | 17.8–18.3 | 11.2–17.7 |
| **Metal FP32 (`v0.17.0`)** | 1.85 | **69.38 (−31.6%)** | **56.17 / 17.76 (−14.9%)** | **54.99 / 18.14 (−16.4%)** | **74.7 / 73.3 / 75.2 (−80.0% to −82.2%)** | **116.2 (−25.0%)** | **18.66 (+1.7%)** | **17.34 (+11.3%)** |
| *Metal FP32 (`v0.13.3`)* | *1.96* | *80.24* | *68.51 / 14.57* | *66.63 / 14.96* | *271.1 / 271.2 / 290.9* | *137.1* | *17.44* | *17.29* |
| **Metal BF16 (`v0.17.0`)** | **1.25** | **36.02 (−65.0%)** | **29.29 / 34.03 (−56.3%)** | **29.32 / 33.99 (−56.3%)** | **60.7 / 60.1 / 61.1 (−84.3% to −85.7%)** | **64.64 (−59.5%)** | **35.55 (+98.9%)** | **35.33 (+216.8%)** |
| *Metal BF16 (`v0.13.3`)* | *1.25* | *35.49* | *29.33 / 33.97* | *29.19 / 34.10* | *107.6 / 108.5 / 115.0* | *64.74* | *33.74* | *35.19* |
| **Metal BF16 + FP16 KV (`v0.17.0`)** | **1.24** | **35.18 (−66.4%)** | **28.96 / 34.41 (−56.8%)** | **28.94 / 34.43 (−56.5%)** | **60.9 / 60.0 / 60.9 (−84.1% to −85.6%)** | **65.13 (−58.9%)** | **35.58 (+98.1%)** | **35.55 (+135.2%)** |
| *Metal BF16 + FP16 KV (`v0.13.3`)* | *1.24* | *36.51* | *30.01 / 33.14* | *29.51 / 33.76* | *108.5 / 110.7 / 116.1* | *66.86* | *34.99* | *34.99* |
| **Metal Q8_0 (`v0.17.0`)** | 4.69 | **32.99 (−68.0%)** | **16.99 / 58.32 (−74.8%)** | **16.70 / 59.53 (−75.0%)** | **45.2 / 45.2 / 45.8** | **59.34** | **64.88 (+261.6%)** | **64.92 (+275.7%)** |
| *Metal Q8_0 (`v0.13.3`)* | *4.71* | *33.03* | *17.07 / 58.13* | *16.65 / 59.69* | *124.3 / 126.6 / 131.9* | *59.27* | *64.48* | *64.33* |
| **Metal Q4_1 (`v0.17.0`)** | 5.84 | **34.17 (−66.8%)** | **10.60 / 92.34 (−84.2%)** | **10.13 / 97.80 (−85.0%)** | **43.3 / 43.5 / 43.1 (−88.9% to −89.9%)** | **62.50 (−60.9%)** | **108.14 (+507.9%)** | **108.11 (+511.8%)** |
| **Metal Q4_0 (`v0.17.0`)** | 4.77 | **34.29 (−67.0%)** | **9.93 / 98.26 (−85.1%)** | **9.35 / 105.82 (−86.2%)** | **42.4 / 42.5 / 43.1 (−89.0% to −90.3%)** | **62.32 (−60.5%)** | **119.14 (+564.1%)** | **118.30 (+917.7%)** |
| *Metal Q4_0 (`v0.13.3`)* | *4.81* | *34.30* | *10.10 / 97.11* | *9.43 / 104.93* | *136.8 / 139.7 / 147.1* | *62.73* | *117.96* | *118.68* |

Notes:
- Percentages in parentheses are the paired median `Metal/CPU - 1` from each
  mode's alternating campaign (negative latency % = Metal faster; positive
  throughput % = Metal faster).
- **`go test -bench` (`BenchmarkLlama32CachedGeneration` vs `BenchmarkLlama32MetalAttentionCachedGeneration`):**
  CPU NEON FP32 `15.53 decode-tok/s` vs Metal FP32 `18.41 decode-tok/s`
  (`+18.5%`).

---

## Memory Footprint and Device Residency Accounting (`v0.17.0`)

| Mode | Device Model Bytes (Weights + RoPE) | Sustained KV Bytes (1-slot in-place, cap 38) | Short Peak / Post-Close RSS (GiB) | Indep. 2-Model Peak / Post-Close RSS (GiB) | Exact Weight Layout |
| --- | ---: | ---: | ---: | ---: | --- |
| **CPU NEON FP32** | 0 (4.60 GiB Go heap) | 2,490,368 (host, 1 slot) | 6.37 / 6.37 | 11.19 / 8.34 | All FP32 on Go heap |
| **Metal FP32** | `4,943,388,672` (4.60 GiB) | **`2,490,368` (2.38 MiB, −50%)** | 6.60 / 5.62 | 11.19 / 8.34 | All FP32 (tied emb/classifier shared) |
| **Metal BF16** | **`2,471,759,872` (2.30 GiB, −50.0%)** | **`2,490,368` (2.38 MiB, −50%)** | **3.31 / 2.82** | **5.64 / 5.03** | All BF16 (direct upload) + FP32 RoPE (`131,072 B`) |
| **Metal BF16 + FP16 KV** | **`2,471,759,872` (2.30 GiB, −50.0%)** | **`1,245,184` (1.19 MiB, −50%)** | **3.31 / 2.82** | **5.65 / 5.04** | All BF16 + FP16 KV + FP32 RoPE |
| **Metal Q8_0** | **`1,313,382,400` (1.22 GiB, −73.4%)** | **`2,490,368` (2.38 MiB, −50%)** | 3.24 / 2.95 | **4.50 / 4.06** | 112 layer projs Q8_0 (`1,033,895,936 B`) + tied emb/cls Q8_0 (`279,085,056 B`) + 33 norms FP32 (`270,336 B`) + RoPE (`131,072 B`) |
| **Metal Q4_1** | **`772,743,168` (0.72 GiB, −84.4%)** | **`2,490,368` (2.38 MiB, −50%)** | **2.72 / 2.50** | **3.49 / 3.18** | 112 layer projs Q4_1 (`608,174,080 B`) + tied emb/cls Q4_1 (`164,167,680 B`) + 33 norms FP32 (`270,336 B`) + RoPE (`131,072 B`) |
| **Metal Q4_0** | **`695,508,992` (0.65 GiB, −85.9%)** | **`2,490,368` (2.38 MiB, −50%)** | **2.64 / 2.43** | **3.30 / 3.02** | 112 layer projs Q4_0 (`547,356,672 B`) + tied emb/cls Q4_0 (`147,750,912 B`) + 33 norms FP32 (`270,336 B`) + RoPE (`131,072 B`) |

---

## Numerical Accuracy and Greedy Token Trajectories (`v0.17.0`)

- **Reference 8-token short trajectory:** `[12366, 13, 1102, 374, 279, 1455, 95551, 3363]` (`" Paris. It is the most populous city"`)
- **Reference 27/28/29 prefill winners:** `[9822]`, `[374]`, `[12366]`

| Mode | Short (8) | Sustained (33) | 27/28/29 | Pos 11 Winner & Margin (frozen: `95551` by `0.043341`) | Frozen Ref Error (p50 / p95 / max / max frac) |
| --- | :---: | :---: | :---: | --- | --- |
| **CPU NEON FP32** | **8/8** | **33/33** | **3/3** | `95551` by `0.043350` | `6.68e-6` / `2.29e-5` / `4.96e-5` / `0.0316` |
| **Metal FP32 (`v0.17.0`)** | **8/8** | **33/33** | **3/3** | `95551` by `0.043320` | **`4.77e-6` / `2.10e-5` / `3.43e-5` / `0.0217`** |
| **Metal BF16 (`v0.17.0`)** | **8/8** | **33/33** | **3/3** | `95551` by `0.043318` | **`6.20e-6` / `2.48e-5` / `5.34e-5` / `0.0255`** |
| **Metal BF16 + FP16 KV (`v0.17.0`)** | **8/8** | **33/33** | **3/3** | `95551` by `0.042242` | `4.34e-4` / `1.61e-3` / `2.72e-3` / `1.852` |
| **Metal Q8_0 (`v0.17.0`)** | **8/8** | **33/33** | **3/3** | **`95551` by `0.053505`** | `3.12e-2` / `1.30e-1` / `3.47e-1` / `143.1` |
| **Metal Q4_1 (`v0.17.0`)** | 1/8 | 2/33 | **3/3** | Diverged at step 1 (`11` `","` vs `13` `"."`) | `4.67e-1` / `1.28e+0` / `1.65e+0` / `604.0` (matched prefix steps) |
| **Metal Q4_0 (`v0.17.0`)** | 2/8 | 2/33 | **3/3** | Diverged at step 2 (`578` vs `1102`) | `3.39e-1` / `1.15e+0` / `1.47e+0` / `1136.5` (matched prefix steps) |

- **`Q4_1` vs `Q4_0` on Llama 3.2 1B:** Both match 3/3 27/28/29 prefill winners
  and diverge on the short trajectory into coherent English (`Q4_1`:
  `[12366, 11, 323, 433, 374, 279, 1455, 6366]`, `" Paris, and it is the most beautiful"`;
  `Q4_0`: `[12366, 13, 578, 6864, 315, 279, 3723, 15422]`, `" Paris. The capital of the United Kingdom"`).
  On synthetic reduced-precision parity (`TestMetalAttentionReducedPrecisionSynthetic`),
  `Q4_1` is tighter (`maxDiffVsCPU=0.232` tied / `0.291` untied vs `0.946` / `0.438` for `Q4_0`).

---

## Native Metal Profiling (`v0.17.0` Command Coalescing)

With `v0.17.0` `MTLComputeCommandEncoder` coalescing, 242 compute dispatches per
decode step coalesce into **17 compute encoders** (down from 242 encoders in
`v0.13.3`):

| Metric (`command-1`, no counters) | FP32 (`v0.17.0`) | BF16 (`v0.17.0`) | BF16+KV16 (`v0.17.0`) | Q8_0 (`v0.17.0`) | Q4_1 (`v0.17.0`) | Q4_0 (`v0.17.0`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warm prefill GPU interval (ms) | **69.74** | **34.71** | **35.02** | **31.49** | **33.02** | **33.07** |
| Warm decode GPU interval (ms) | **56.92** *(was 66.56)* | **28.56** | **28.47** | **16.17** | **9.19** | **8.44** *(was 8.79)* |
| Peak `currentAllocatedSize` | 4.948 GB | 2.477 GB | 2.475 GB | 1.318 GB | 0.778 GB | 0.700 GB |

---

## Reproducibility Commands

All raw binaries, `.f32` logit dumps, pprof profiles, and validation logs are
gitignored under `.benchmarks/gpu-ll-08/2026-11-27-m3-v0.17.0/`.
