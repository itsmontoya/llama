# Llama 3.2 1B Final FP32 GPU Baseline (`neuralnetwork v0.11.3`)

This document records the final verified FP32 CPU-vs-Metal baseline for
`github.com/itsmontoya/llama` integrating the immutable GATE-D upstream release
`github.com/itsmontoya/neuralnetwork@v0.11.3` (task LL-05).

## Executive Summary and NN-08 / GATE-D Context

- **CPU remains the default and authoritative baseline.** Resident Metal
  execution is explicitly opt-in (`-tags=metal` with `CGO_ENABLED=1` on Apple
  Silicon macOS, plus explicit CLI flags or `LoadHFModelMetalAttention`).
- **NN-08 negative result:** Upstream task NN-08 (`neuralnetwork#148`) evaluated
  an `M=1` SIMD-cooperative right-transposed FP32 kernel (`r2s4`) that lowered
  warmed sustained decode latency by 11.47% (`90.933 ms` → `80.917 ms`, 6/6
  pairs) and GPU command interval by 11.91% (`89.35 ms` → `78.71 ms`). However,
  across all 4,232,448 unique logits in 32-step sustained decode, `r2s4` failed
  the predeclared full-row mixed tolerance `1e-4 + 1e-4*abs(CPU)` on 4 near-zero
  logits (`max_fraction = 1.08730` vs `0.96079` for `v0.11.2`). All tested
  parallel-K reduction variants — including 48-bit compensated summation
  (`1.06825`) — exceeded `1.0`, while bit-exact sequential variants missed the
  ≥10% speedup gate. Per the negative-result rule, `v0.11.3` preserves
  `v0.11.2`'s production Metal fast path unchanged while adding expanded
  FP64/tail tests, isolated projection benchmarks, and
  `docs/gpu-nn-08-optimization.md`.
- **Reproduced Apple M3 FP32 tradeoff (`v0.11.3`, 54 pairs / 108 fresh
  processes):**
  - **Multi-row prefill favors Metal:** Warmed 27/28/29-token prefill took
    **26.4–28.7% less wall time** on Metal in **6/6 pairs** per length
    (`405.12 → 299.54 ms`, `423.33 → 300.09 ms`, `436.32 → 319.49 ms`). Warmed
    6-token short prefill was **5.8% faster** (`106.29 → 100.44 ms`, 6/6 pairs).
  - **Single-row decode favors CPU:** Warmed short decode took **88.45 ms/call**
    (`11.29 tok/s`) on Metal vs **66.52 ms/call** (`15.01 tok/s`) on host NEON
    CPU (**+33.9% latency**, Metal lost 6/6 pairs). Warmed 32-step sustained
    decode (positions 6–37) took **90.10 ms/call** (`11.08 tok/s`) on Metal vs
    **68.22 ms/call** (`14.64 tok/s`) on CPU (**+31.5% latency**, Metal lost
    6/6 pairs).
  - **Two-session concurrency:** Shared-model decode favored CPU (**17.66 vs
    14.75 tok/s**, −16.4%, CPU won 6/6 pairs). Independent two-model decode had
    lower total process wall time (**15.87 vs 20.31 s**, −20.0%, Metal won 6/6)
    and lower peak/post-close RSS (**8.73 vs 10.57 GiB** peak; **1.58 vs 9.96
    GiB** post-close) on Metal, while warmed decode throughput favored CPU at
    the paired median (**17.05 vs 14.45 tok/s**, −15.4%, wide CPU range
    `10.15–17.55 tok/s`).
  - **Exact numerical reproducibility:** All 108 primary processes, 22
    diagnostic processes, 286,267,392 paired logits, 29,928 frozen observations
    per backend, and every greedy token ID (including position 11 ID `95551`)
    passed with **0 violations** and reproduced the LL-04 error distributions
    identically.

---

## Verified Versions, Checksums, and Hardware

| Input | Verified identity |
| --- | --- |
| Llama base `main` | `63d5a247d981538dfd7dd695f4f4365c129f4c3d` (LL-04 merge) |
| Llama LL-05 tested commit | `3aed8579c21013537fd7d3640fc9aff588278b06` on `gpu-ll-05` |
| `neuralnetwork` release | `github.com/itsmontoya/neuralnetwork@v0.11.3` (GATE-D) |
| `neuralnetwork` tag commit | `33cae5825c1b6724b0e784eb6293010cbc41d26c` (`refs/tags/v0.11.3`) |
| `neuralnetwork` module checksum | `h1:mPThw+bMgln2u4MAGHxezzLYKh4RPF/qz+0cF0O7lZQ=` |
| `neuralnetwork` `go.mod` checksum | `h1:SafbkuDehVErsuJAL5FHFsLhKJbeh/OI3ItvbmwDafw=` |
| Pinned model snapshot | `meta-llama/Llama-3.2-1B` revision `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| Manifest SHA-256 | `3404dfda97c69cf3a9a3cf15dc4db7017cb049da07462c52f2e3004f6e12ab64` |
| Frozen reference SHA-256 | `3a66ddc5cbdef45e397e47a6bd992676ab3d35234a8b389908f721b05466d361` |
| Checkpoint (`model.safetensors`) SHA-256 | `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a` |
| TinyStories model SHA-256 | `b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696` |
| TinyStories vocabulary SHA-256 | `037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312` |
| Hardware | MacBook Air `Mac15,13`, Apple M3 (4P+4E CPU, 10-core GPU, Metal 4), 24 GiB unified memory |
| OS and SDK | macOS `26.5.2 (25F84)`, Darwin `arm64`, Command Line Tools SDK `14.4` |
| Go configuration | `go1.27.3 darwin/arm64`, `GOARM64=v8.0`, `GOMAXPROCS=8`, `CGO_ENABLED=1`, `GOWORK=off`, empty `GOFLAGS` |

The historical LL-04 campaign on `v0.11.2` is preserved untouched in
[GPU_PERFORMANCE_REPORT.md](GPU_PERFORMANCE_REPORT.md).

---

## Paired CPU vs Metal Results (`neuralnetwork v0.11.3`)

Methodology matches LL-04: 6 alternating fresh-process pairs per workload across
9 workloads (54 pairs / 108 processes), rotated workload order across rounds,
each backend first 3 times, 3-second cooldown between processes, macOS
`/usr/bin/time -l` with a 16 GiB per-process RSS cap, and no samples dropped.
Paired change is the median (and range) of `Metal/CPU - 1` across the 6 pairs.

### Short Single-Session Trajectory (6-Token Prompt → 8 Predictions, KV Capacity 13)

| Metric | CPU median (range) | Metal median (range) | Paired change % (range) |
| --- | ---: | ---: | ---: |
| Process startup, ms | 16.51 (4.83–474.20) | 13.13 (8.10–323.74) | -34.2 (-61.6–353.1) |
| Model load, s | 1.918 (1.889–2.733) | 1.899 (1.852–2.022) | -1.5 (-26.0–0.5) |
| Tokenizer load & encode, ms | 164.70 (158.17–172.57) | 166.37 (161.26–168.37) | 0.8 (-4.5–6.2) |
| First cache setup, ms | 0.015 (0.010–0.029) | 0.211 (0.201–0.246) | 1452.9 (582.2–2054.2) |
| First prefill, ms | 117.97 (110.35–564.20) | 136.48 (125.59–172.96) | 13.1 (-69.3–27.3) |
| First inference TTFT, ms | 118.07 (110.46–564.30) | 136.63 (125.75–173.07) | 13.1 (-69.3–27.3) |
| Launch to first token, s | 2.210 (2.186–3.945) | 2.208 (2.158–2.686) | -0.6 (-31.9–2.7) |
| First decode call, ms | 67.60 (66.18–69.98) | 88.22 (86.78–90.26) | 29.9 (25.8–36.4) |
| First 8 predictions, ms | 592.04 (574.45–1054.9) | 758.90 (733.96–790.25) | 26.6 (-25.1–34.9) |
| Warm prefill, ms | 106.29 (103.95–112.96) | 100.44 (99.09–100.69) | **-5.8** (-11.0–-3.1) |
| Warm inference TTFT, ms | 106.39 (104.06–113.07) | 100.56 (99.58–100.83) | **-5.7** (-11.0–-3.2) |
| Warm decode, ms/call | **66.52** (65.34–68.30) | 88.45 (87.01–90.14) | **+33.9** (27.4–36.2) |
| Warm decode, tokens/s | **15.01** (14.62–15.28) | 11.29 (11.05–11.48) | **-25.4** (-26.8–-21.5) |
| Warm 8 predictions, ms | **572.13** (564.33–591.91) | 720.33 (710.44–734.50) | **+27.0** (20.0–28.2) |
| Whole process, s | 3.314 (3.282–5.139) | 3.673 (3.572–4.113) | +9.4 (-20.0–13.3) |

### Warmed Sustained Decode, Prefill Tails, and Concurrency

| Metric | CPU median (range) | Metal median (range) | Paired change % (range) | Metal wins |
| --- | ---: | ---: | ---: | ---: |
| Sustained decode, ms/call (pos 6–37) | **68.22** (67.03–69.69) | 90.10 (88.66–91.28) | **+31.5** (28.9–36.2) | 0/6 |
| Sustained decode, tokens/s | **14.64** (14.33–14.90) | 11.08 (10.92–11.27) | **-23.9** (-26.7–-22.4) | 0/6 |
| Sustained positions 6–13, ms/token | 68.23 (67.23–72.16) | 88.08 (86.45–90.19) | +29.1 (22.6–34.1) | 0/6 |
| Sustained positions 30–37, ms/token | 68.18 (66.95–69.80) | 92.42 (90.62–93.64) | +34.5 (31.2–38.1) | 0/6 |
| 27-token prefill, ms | 405.12 (388.25–410.25) | **299.54** (295.87–301.30) | **-26.4** (-27.0–-22.4) | **6/6** |
| 28-token prefill, ms | 423.33 (401.64–473.51) | **300.09** (296.94–303.64) | **-28.7** (-36.5–-26.1) | **6/6** |
| 29-token prefill, ms | 436.32 (421.32–451.92) | **319.49** (316.30–322.99) | **-26.7** (-29.1–-24.9) | **6/6** |
| Concurrent 6-token prefill, ms | 161.49 (148.55–183.98) | **151.65** (148.70–153.38) | **-5.5** (-17.7–+0.2) | 5/6 |
| Concurrent 6-token aggregate, pred/s | 12.38 (10.86–13.45) | 13.17 (13.00–13.44) | +5.7 (-0.2–21.2) | 5/6 |
| Concurrent prefill TTFT slower/faster | 1.025 (1.008–1.073) | 1.049 (1.001–1.082) | +0.7 (-2.5–7.0) | — |
| Shared-model decode aggregate, tok/s | **17.66** (16.99–17.79) | 14.75 (14.70–14.87) | **-16.4** (-17.3–-13.3) | 0/6 |
| Shared-model lane 1 / lane 2, tok/s | 8.88 / 8.86 | 7.38 / 7.39 | -16.6 / -16.6 | 0/6 |
| Shared-model slower/faster ratio | 1.002 (1.002–1.004) | 1.003 (1.001–1.004) | +0.0 (-0.3–0.3) | — |
| Independent models load, s | 6.355 (6.160–6.746) | **5.399** (3.820–6.421) | **-15.4** (-38.0–4.1) | 5/6 |
| Independent decode aggregate, tok/s | 17.05 (10.15–17.55) | 14.45 (14.38–14.69) | -15.4 (-16.7–+41.7) | 1/6 |
| Independent whole process, s | 20.31 (16.43–22.59) | **15.87** (13.41–16.86) | **-20.0** (-30.4–-12.7) | **6/6** |
| Independent slower/faster ratio | 1.009 (1.001–1.056) | 1.002 (1.000–1.004) | -0.7 (-4.9–0.2) | — |

### Memory, Go Allocations, and Post-Close Retention

| Metric | CPU median (range) | Metal median (range) | Paired change % (range) |
| --- | ---: | ---: | ---: |
| Short warm Go allocated, MiB | 110.54 (110.54–110.55) | 4.07 (4.07–4.07) | -96.3 (-96.3–-96.3) |
| Short warm Go allocation count | 39,577 (39,559–39,584) | 2,027.5 (2,026–2,029) | -94.9 (-94.9–-94.9) |
| Short peak RSS, GiB | 6.34 (5.69–6.38) | 6.60 (6.60–6.60) | +4.0 (3.5–15.9) |
| Short retained RSS, GiB | 6.34 (5.69–6.38) | 6.60 (6.55–6.60) | +4.1 (3.5–15.9) |
| Short retained Go heap, MiB | 4,714.7 (4,714.7–4,714.7) | 0.35 (0.34–0.35) | -100.0 (-100.0–-100.0) |
| Short post-close RSS, GiB | 6.34 (5.69–6.38) | 5.62 (5.57–6.47) | -11.4 (-11.9–13.7) |
| Sustained peak / post-close RSS, GiB | 6.38 / 6.38 | 6.60 / 5.62 | +3.4 / -11.9 |
| Shared-model peak / post-close RSS, GiB | 6.39 / 6.39 | 6.60 / 5.62 | +3.3 / -12.0 |
| Independent models peak RSS, GiB | 10.57 (9.96–11.88) | **8.73** (6.65–10.84) | **-17.7** (-34.3–-1.8) |
| Independent models post-close RSS, GiB | 9.96 (7.64–10.36) | **1.58** (0.51–8.10) | **-83.8** (-95.0–+6.0) |

### Confidence, Variance, and Workstation State

Across all 54 pairs (plus initial and final probes):
- **Power and thermals:** AC power remained connected in 100% of probes; macOS
  reported no thermal or performance warnings.
- **Memory pressure and swap:** System memory free percentage ranged 68–83%,
  with 13.20–16.02 GiB of existing system swap in use, and **0 per-process
  swaps** in all 108 `/usr/bin/time -l` records. Peak RSS stayed well below the
  16 GiB cap (max 11.88 GiB CPU, 10.84 GiB Metal, both in `independent`).
- **Noise profile:** Warmed single-session latencies and 27/28/29-token prefills
  were tightly clustered (typically ±1–3%). Cold process startup, first prefill,
  and the dual-model `independent` workload exhibited higher workstation
  variance (e.g., CPU `independent` warmed decode ranged `10.15–17.55 tok/s`,
  compared to Metal's tighter `14.38–14.69 tok/s`).

---

## Correctness Tolerances and Numerical Distributions

Every primary process checks greedy IDs at selection, checks the 8 frozen
observations on the short trajectory, and verifies every warm row and concurrent
lane against that backend's first output. Every paired comparison checks all
128,256 FP32 logits per row across both phases and all lanes.

- **Frozen reference bound:** `abs(actual - reference) <= 1e-3 + 1e-3*abs(reference)`
- **Full paired CPU-vs-Metal bound:** `abs(metal - cpu) <= 1e-4 + 1e-4*abs(cpu)`
- **Synthetic oracle bound (unit tests):** `abs(actual - reference) <= 2e-5 + 2e-5*abs(reference)`

All 108 processes and all 54 pairs passed with **0 violations** across
**286,267,392** compared paired logits and **29,928** frozen observations per
backend:

| Comparison | p50 absolute | p95 absolute | Max absolute | RMSE | Max tolerance fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU vs frozen | 6.676e-6 | 2.289e-5 | 4.959e-5 | — | 0.03164 |
| Metal vs frozen | 7.629e-6 | 2.289e-5 | 4.768e-5 | — | 0.02076 |
| Full paired short rows | 7.153e-6 | 2.360e-5 | 8.202e-5 | 1.171e-5 | 0.70488 |
| Full paired sustained rows | 1.132e-5 | 4.244e-5 | 1.214e-4 | 2.002e-5 | 0.96079 |
| Full paired 27-token row | 1.717e-5 | 3.004e-5 | 6.008e-5 | 1.877e-5 | 0.30093 |
| Full paired 28-token row | 7.629e-6 | 2.146e-5 | 5.722e-5 | 1.100e-5 | 0.33227 |
| Full paired 29-token row | 2.345e-5 | 4.083e-5 | 6.914e-5 | 2.580e-5 | 0.62007 |

- **Close greedy decision at position 11 (ID `95551`):** Metal selected `95551`
  by `0.043310`, CPU by `0.043350`, frozen reference by `0.043341`.
- **Recovery workload (6 pairs):** Both backends rejected an out-of-range token
  ID (`4.52 µs` CPU, `6.08 µs` Metal warmed) and a pre-canceled context
  (`1.13 µs` CPU, `1.60 µs` Metal warmed), preserved the KV cursor, and
  reproduced the exact 8-token short trajectory on retry.

---

## Native Metal Profiling, Residency, and Scheduling

All 22 diagnostic processes matched the primary campaign's `.f32` logit outputs
byte for byte.

### Device Residency and Command Layout

- **Model weights + RoPE tables:** `4,943,388,672` requested bytes across 148
  uploads per model (single shared allocation for tied embedding/classifier;
  `9,886,777,344` requested bytes for two independent models).
- **Live buffer high-water marks (model + KV + scratch):**
  - Short: `4,946,633,728` requested B (`4,947,460,096` B `currentAllocatedSize`)
  - Sustained (1 lane): `4,949,910,528` requested B (`4,950,605,824` B `currentAllocatedSize`)
  - Shared (2 lanes): `4,956,432,384` requested B (`4,957,356,032` B `currentAllocatedSize`, peak 2 scopes)
  - Independent (2 models): `9,899,821,056` requested B (`9,900,736,512` B `currentAllocatedSize`, peak 2 scopes)
- **Per-step scheduling:** Exactly **1 command buffer**, **1 wait**, **242
  compute dispatches**, **17 blit encoders** (decode) or **18 blit encoders**
  (multi-row cached prefill), and **1 logit download** (`513,024` B). Zero
  per-layer host downloads or weight uploads during inference.

### Native Metal Command Timestamps and Stage Attribution

Pooled across 3 fresh diagnostic processes per mode (96 warmed sustained decode
commands, 21 warmed short decode commands; GPU/CPU clock calibration ratio
`1.000`):

| Diagnostic command | GPU start to end, ms | Host native wait, ms |
| --- | ---: | ---: |
| First short prefill (n=3) | 112.22 (109.34–114.89) | 127.23 (126.43–131.87) |
| Warm short prefill (n=3) | 100.05 (99.52–100.19) | 100.78 (100.25–100.93) |
| Warm short decode (n=21) | 87.87 (86.70–89.02) | 88.21 (87.01–89.33) |
| Warm sustained decode, command-only (n=96) | 90.43 (86.90–94.64) | 90.77 (87.19–95.23) |
| Warm sustained decode, stage-sampled (n=96) | 92.46 (88.89–109.33) | 93.19 (89.28–109.87) |

Pooled warm sustained decode compute-stage sums (`counters-1..3`, 96 commands):

| Compute group | Dispatches / token | Sampled GPU stage sum, ms |
| --- | ---: | ---: |
| FFN gate, up, and down projections | 48 | 40.949 (39.718–45.724) |
| RMSNorm | 33 | 18.509 (18.044–20.046) |
| Attention Q, K, V, and O projections | 64 | 13.612 (13.057–15.446) |
| Classifier projection | 1 | 12.069 (11.660–13.876) |
| Causal attention | 16 | 3.700 (1.197–6.261) |
| RoPE (`nn_tensor_rotate_pairs`) | 32 | 0.099 (0.081–1.107) |
| Residual adds (`nn_tensor_add`) | 32 | 0.086 (0.069–1.186) |
| SwiGLU (`nn_tensor_swiglu`) | 16 | 0.045 (0.042–0.968) |

Host overlay (`host-1..3`, short workload): device open `37.61 ms`
(`37.36–190.06`), 148 model tensor allocations `3.01 ms`, 148 model uploads
`295.28 ms` (`291.00–325.37`). Warmed short decode per step: scratch allocate
`0.044 ms`, scope create `0.002 ms`, scope encode `0.209 ms`, submit `0.003 ms`,
wait `88.526 ms`, logit download `0.040 ms`, finite validation `0.048 ms`.

Go pprof (`pprof-cpu`, `pprof-metal`): CPU sampled `23.31 s` over `5.26 s`
elapsed (`32.9%` `pthread_cond_wait`, `28.1%` `rightTransposePackedTileARM64`,
`18.5%` `pthread_cond_signal`, `16.7%` `rightTransposeOneRowTileARM64`), with
`4.60 GB` retained Go heap. Metal sampled `60 ms` over `6.12 s` elapsed
(`83.3%` `cgocall`), with `3.59 MB` retained Go heap (device buffers live
outside the Go heap).

---

## Supported Modes, Fallback Behavior, and Known Limitations

### Supported and Unsupported Modes

| Mode / Build | Status | Behavior |
| --- | --- | --- |
| Default CPU (`go build ./...`, `CGO_ENABLED=0` or `1`) | **Default, supported** | Host-only FP32 inference; links no Metal framework; authoritative numerical baseline. |
| Pure Go (`-tags=purego`, `CGO_ENABLED=0` or `1`) | **Supported (CPU only)** | Portable CPU execution; explicit Metal requests return an availability error. |
| Non-Darwin (`GOOS=linux GOARCH=amd64 CGO_ENABLED=0`, with or without `-tags=metal`) | **Supported (CPU only)** | Compiles cleanly; explicit Metal requests return an availability error. |
| Hybrid Metal FFN (`-tags=metal`, `CGO_ENABLED=1`, `-ffn-backend metal` / `LoadHFModelMetal`) | **Experimental, opt-in** | Keeps 3 GiB FFN weights on GPU; per-layer CPU↔GPU crossings; supports `ForwardWithTrace`. |
| Resident Metal Attention + FFN (`-tags=metal`, `CGO_ENABLED=1`, `-ffn-backend metal -attention-backend metal` / `LoadHFModelMetalAttention`) | **Experimental, opt-in** | Full device-resident FP32 prefill and decode (1 command/step, 1 logit download/step). |
| `ForwardWithTrace` on resident Metal attention | **Unsupported** | Returns an error; use CPU or FFN-only Metal mode when attention probability traces are needed. |
| FP16 / BF16 / quantized inference, sharded HF checkpoints, chunked prefill, stochastic sampling | **Out of scope / unsupported** | FP32 single-file checkpoints and greedy decoding only. |

### CPU Fallback and Failure Semantics

- Metal is never selected implicitly by default CPU entrypoints (`LoadHFModel`,
  default CLI flags).
- If a caller explicitly requests Metal on an unsupported build (`purego`,
  `CGO_ENABLED=0`, non-Darwin) or when no Metal device can be opened, Llama
  returns an explicit availability error rather than silently falling back to
  CPU.
- Callers must close Metal sessions (`Session.Close`) and decoders
  (`Decoder.Close`) to release device buffers. Failed or canceled cached steps
  drain any submitted command before returning and leave the committed KV cursor
  untouched.

### Known Limitations

1. **Single-row decode latency:** Resident FP32 Metal decode is ~31–34% slower
   than ARM64 NEON CPU on Apple M3 (`90.10` vs `68.22 ms/call`). Use CPU for
   latency-sensitive single-session decode; Metal currently wins on multi-row
   prefill (27–29 tokens: −26% to −29%).
2. **FP32 memory footprint:** Each resident 1B model requests ~4.60 GiB of
   immutable device buffers; two independent models double this (~9.21 GiB).
3. **Dual-slot KV prefix copies:** To guarantee rollback safety on failure,
   each layer copies the committed KV prefix into its inactive slot per append.
4. **Runtime shader compilation:** Shaders compile at runtime via Metal MSL
   (cached by macOS across processes); no precompiled `.metallib` is shipped.
5. **Experimental API surface:** `neuralnetwork/gpu` and Llama's Metal loaders
   remain experimental pending the owner-only architectural `REVIEW`.

---

## Reproducibility Commands and Artifacts

Compact per-process and per-pair CSVs for this `v0.11.3` campaign are checked in
at:
- [apple-m3-v0.11.3-samples.csv](benchmarks/metal/data/apple-m3-v0.11.3-samples.csv)
- [apple-m3-v0.11.3-pairs.csv](benchmarks/metal/data/apple-m3-v0.11.3-pairs.csv)

Raw binaries, `.f32` logit dumps, pprof profiles, and validation logs stay
gitignored under `.benchmarks/gpu-ll-05/2026-11-24-m3/`.

```sh
# 1. Correctness & portability matrix
export GOWORK=off GOMAXPROCS=8 GOARM64=v8.0 CGO_ENABLED=1
export LLAMA_STORIES260K_MODEL="$stories_dir/stories260K.bin"
export LLAMA_STORIES260K_TOKENIZER="$stories_dir/tok512.bin"
unset LLAMA_32_1B_MODEL_DIR LLAMA_PAIRED_BACKEND

go mod verify
go fmt ./...
go vet ./...
go test -p 1 -count=1 -race -v ./...
CGO_ENABLED=0 go test -p 1 -count=1 ./...
CGO_ENABLED=0 go test -p 1 -count=1 -tags=purego ./...
CGO_ENABLED=0 go test -p 1 -count=1 -tags=metal ./...
CGO_ENABLED=0 go test -p 1 -count=1 -tags='metal purego' ./...
CGO_ENABLED=1 go test -p 1 -count=1 -tags='metal purego' ./...
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build ./...
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -tags=metal ./...
go vet -tags=metal ./...
go test -p 1 -count=1 -race -v -tags=metal ./...
go test -p 1 -race -tags=metal -run '^TestMetal(Attention|FFN)' -count=10 .
LLAMA_32_1B_MODEL_DIR="$model_dir" go test -p 1 -count=1 -timeout 20m -v ./...
LLAMA_32_1B_MODEL_DIR="$model_dir" go test -p 1 -count=1 -timeout 20m -v -tags=metal ./...

# 2. Paired CPU vs Metal campaign and native Metal diagnostics
run=.benchmarks/gpu-ll-05/reproduction
python3 -B benchmarks/metal/campaign.py "$model_dir" "$run"
python3 -B benchmarks/metal/summarize.py "$run" > "$run/summary.txt"
python3 -B benchmarks/metal/diagnose.py "$run" "$run/diagnostics"
python3 -B benchmarks/metal/summarize_profiles.py "$run/diagnostics" \
  > "$run/profile-summary.txt"
```

---

## Inputs for Owner Architectural `REVIEW`

1. **`neuralnetwork/gpu` package boundary:** Llama cleanly consumed
   `gpu.Device`, `gpu.Tensor`, and `gpu.Scope` with zero forked Metal kernels
   and zero impact on default/purego/Linux CPU builds. Keeping `gpu` separate
   from CPU `tensor.Tensor` avoided hidden host/device synchronization and kept
   ownership explicit.
2. **FP32 single-row decode ceiling vs tolerance policy:** NN-08 proved that
   parallel-K `M=1` matmul cuts sustained decode by ~11.5% (`90.9 → 80.9 ms`)
   with lower aggregate RMSE/p50/p95 than `v0.11.2`, but exceeds the `1e-4`
   absolute floor on 4/4,232,448 near-zero logits (`max_fraction = 1.08730`).
   Meanwhile, 33 `RMSNorm` dispatches account for `18.51 ms` (~20% of GPU
   decode time). Even combining both FP32 improvements would likely bring M3
   FP32 decode only near parity with 8-core NEON CPU (`~68 ms`), because each
   FP32 token still reads ~4.94 GB of weights.
3. **Precision roadmap decision:** Because FP32 weight bandwidth (~4.94 GB/token
   for 1B parameters) is the structural bottleneck on Apple M3 unified memory,
   an explicitly scoped **FP16/BF16 or quantized** project is the highest-impact
   lever for GPU decode throughput, whereas additional FP32 tuning should target
   `RMSNorm` (18.5 ms) or require an explicit owner decision on near-zero logit
   tolerance.
4. **Shader distribution:** Runtime MSL compilation took `< 2 ms` total across
   4 libraries and 19 pipelines on this host (benefiting from OS shader caching)
   and requires no build-time Xcode toolchain; `.metallib` packaging is not an
   immediate blocker.
