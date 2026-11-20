# Optimization Pass 8 handoff to neuralnetwork

## Objective

Improve the throughput of the ordered FP32 single-row right-transposed host GEMM used during steady-state Llama 3.2 1B decode. Start with the two FFN shapes, which together account for about 60% of the measured decode loop. Evaluate the tied classifier separately and protect concurrent decoder throughput. The existing host projection loop is the selected optimization. The Llama runtime and its dependency pin are at the Pass 7 baseline.

## Problem and direction

The current four-column scalar loop accounts for 89–92% of labeled sustained decode CPU in two profiles, and projection calls account for 95.7% of the elapsed decode loop in three separate timing runs. The shape sweep, trace, and streaming control support prioritizing throughput within the ordered load/FMA loop. The hot code reads independent weight streams and carries four accumulator dependencies across K. Reusing weights helps, especially for smaller weights. Available software timing and profile evidence leaves the individual contribution of cache misses, memory latency, and execution resources unresolved.

Keep the existing worker selection and weight storage for the first comparison. Benchmark whether more independent output accumulation chains improve load/FMA scheduling on ARM64, with each chain preserving ascending K. Check disassembly for spills, hot-loop bounds checks, loads per iteration, and fused operations. If an ordered lane-wise formulation can improve throughput without a permanent packed copy, it is another way to test the same kernel hypothesis. Use the measured FFN shapes with rotating weights, the classifier, and the full runtime acceptance tests to choose an implementation. Retain the current numerical path if neither experiment demonstrates a win. A 10% reduction in FFN projection wall alone would save approximately 6% of the measured decode wall if other costs were constant.

### Alternatives and tradeoffs

- A bounded dispatch change might reduce helper queueing in concurrent sessions. Both one- and two-session throughput improved at the higher existing process cap, and join times overlap useful arithmetic. Revisit shape-specific scheduling if measurements at a fixed process cap show a real improvement. The classifier has much more work per shard, so it can amortize dispatch differently from FFN and attention projections.
- Packing weights to enable contiguous output-lane loads, or retaining the original BF16 weights for exact conversion on use, could change effective bandwidth. Those alternatives introduce layout or storage-lifetime work, particularly for tied weights. They need separate evidence about preparation, peak memory, and performance under rotating and concurrent workloads.
- Splitting K into independent partial sums can increase parallelism but changes the order of each output's FP32 accumulation. The existing numerical contract excludes that approach.

## Measurement baseline

- Llama baseline `3204dc9694c5d91111485d7597c9c11dadb06146`, the clean Pass 7 merge on `main` and `origin/main`. The only benchmark source addition is `d72b1625696ed7becabf164aed5737a7a9575529`, which adds the concurrent sustained decode workload without changing runtime source.
- Active module `github.com/itsmontoya/neuralnetwork v0.10.17`, checksum `h1:cby0CGQgpK4KniedUtepX2TvGliln1R3EPmokwE0teM=`. The baseline also links `github.com/tphakala/simd v1.3.0` and `golang.org/x/sys v0.45.0`. `go mod verify` passed.
- Go 1.27.3, Darwin ARM64, `GOARM64=v8.0`, `CGO_ENABLED=1`, `GOWORK=off`, empty `GOFLAGS` and default tags. Main series at `GOMAXPROCS=8`. Apple M3 on AC, with four performance and four efficiency cores, 24 GB RAM, macOS 26.5.2 (25F84). Host-reported L1 data/L2 sizes are 128 KiB/16 MiB for the performance level and 64 KiB/4 MiB for the efficiency level.
- `meta-llama/Llama-3.2-1B` revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`. Each process verified all six snapshot files against `testdata/llama32_manifest.json` (SHA-256 `3404dfda97c69cf3a9a3cf15dc4db7017cb049da07462c52f2e3004f6e12ab64`). The checkpoint is 2,471,645,608 bytes, SHA-256 `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`. The frozen reference JSON SHA-256 is `3a66ddc5cbdef45e397e47a6bd992676ab3d35234a8b389908f721b05466d361`. Fixture provisioning is documented in Llama's `testdata/README.md`.

The primary workload is `The capital of France is`, six input tokens including BOS, eight greedy output tokens, context cap 512, and KV capacity 13. TTFT includes prefill and first greedy selection. The short decode metric covers seven steps and their greedy selections. The sustained workload times 32 steps at positions 6–37 with session creation, prefill, and first selection excluded. The 27- and 28-token guards take the first 27 and 28 token IDs of the 29-token prompt in `BenchmarkLlama32SupplementalPrefill`. The existing concurrent prefill guard runs two six-token sessions on one decoder. The two-session sustained benchmark creates independent KV caches, prefills before timing, releases both goroutines at one barrier, and times the same 32 greedy decode steps per session. Its wall includes completion of both lanes, and per-session timing includes scheduling delay after the common start. The ratio of faster to slower session rates measures completion imbalance.

Fresh baseline medians and full ranges follow. Each observation is one timed operation per fresh process with zero inference warm-ups. Process wall and Darwin peak RSS include fixture verification, model, and tokenizer setup. There is no candidate build in this investigation.

| Primary metric, 12 observations | Median (range) |
| --- | ---: |
| Prefill (s) | 0.1834 (0.1725–1.1000) |
| TTFT (s) | 0.1835 (0.1726–1.1010) |
| Decode (s/token) | 0.0805 (0.0773–0.0890) |
| Decode (tokens/s) | 12.415 (11.240–12.930) |
| Cached generation (s/op) | 0.7484 (0.7207–1.6930) |
| Output (tokens/s) | 10.691 (4.725–11.100) |
| Bytes/op | 131,320,272 (131,307,856–131,343,032) |
| Allocs/op | 39,888 (39,869–39,933) |
| Process wall (s) | 3.8898 (3.8186–5.8262) |
| Peak RSS (GiB) | 6.3738 (5.6899–6.3745) |
| Model load (s) | 1.8770 (1.8330–1.9130) |
| Tokenizer setup (s) | 0.1794 (0.1619–1.1070) |

| Workload | Trials | Wall in the timed scope (s) | Decode throughput (tokens/s) |
| --- | ---: | ---: | ---: |
| Single-session sustained | 7 | 2.7358 (2.5312–3.1462) | 11.70 (10.17–12.64) |
| Two-session sustained | 7 | 4.271 (4.181–5.314) | 14.98 (12.04–15.31) |
| Prefill 27 | 5 | 0.7128 (0.6849–0.7479) | — |
| Prefill 28 | 5 | 1.4717 (0.7560–1.8889) | — |
| Prefill 29 | 5 | 0.9953 (0.7336–1.2892) | — |
| Two-session prefill | 7 | 0.2767 (0.2716–1.1834) | — |

The sustained single-session median was 85.5 ms/token with 329,808,696 B/op and 149,554 allocs/op. The concurrent lane medians were 7.591 and 7.492 tokens/s, or 131.7 and 133.5 ms/token. The higher/lower rate ratio was 1.0051 (1.0033–1.0132). Aggregate throughput was about 28% higher than the separate single-session median, with lower throughput per session. These are workload medians from separate observations. The concurrent benchmark measured a median of 659,520,888 B/op, 298,918 allocs/op, and a 6.3674 GiB process RSS peak. The 28- and 29-token wall measurements vary substantially across processes, so compare these guards with alternating pairs and investigate noisy trials without discarding them.

## Fresh phase profiles

These are exclusive sampled CPU totals at the named function and phase. CPU seconds sum across running goroutines. Profiled wall time uses separate phase timers inside the diagnostic binary. The normal timing series runs without CPU profiling or the per-projection shape label overlay. The 27- and 28-token guards retain the Pass 7 diagnostic benchmark.

| Scope | Labeled CPU | Main kernel sample | Condition signaling | Profiled elapsed |
| --- | ---: | ---: | ---: | ---: |
| Primary prefill, six calls | 2.15 s | 1.59 s three-row, 0.36 s one-row | — | 1.081 s |
| Short decode, 42 steps | 13.18 s | 12.26 s single-row (93.0%) | 0.80 s | 4.608 s |
| Sustained decode, 64 steps | 22.09 s | 20.30 s single-row (91.9%) | 1.38 s | 7.740 s |
| Sustained repeat, 64 steps | 13.62 s | 12.18 s single-row (89.4%) | 1.17 s | 5.545 s |
| Prefill 27, three calls | 10.32 s | 8.87 s three-row (85.9%) | 0.30 s | 3.436 s |
| Prefill 28, three calls | 6.83 s | 5.62 s three-row (82.3%) | 0.27 s | 2.525 s |
| Prefill 29, three calls | 5.31 s | 4.14 s three-row (78.0%) | 0.24 s | 2.254 s |

The 27-token prompt has nine complete three-row groups and no remainder. At 28 tokens, of the 0.65 s in the one-row loop, 0.20 s was the classifier and 0.45 s was in block projections. At 29 tokens, the two-row tail accounted for 0.48 s and the classifier for 0.14 s.

Within the first and second sustained samples, single-row gate/up took 10.19 and 6.41 CPU seconds, down took 3.50 and 1.17, and the classifier took 5.04 and 4.05. The two profiled runs varied in wall and aggregate CPU while indicating the same dominant loop. Greedy selection across 64 decode steps took 6.49 and 6.58 ms wall, measured separately. Primary labeled model loading took 1.842 s elapsed. The concurrent CPU profile is whole-process, with 24.61 sampled seconds in `rightTransposeOneRow` and 4.58 s in condition signaling across two runs of two sessions, including setup.

The Pass 7 release measured 12.79 sampled seconds in standalone single-row projection (89.3%) for 64 sustained steps, with 3.96 s in the classifier and 1.41 s in condition signaling. The 32-step projection probe measured 2.411 s. Its primary TTFT median was 0.1840 s and sustained decode median was 12.27 tokens/s. Its 28- and 29-token prefill medians fell from 1.6163 to 0.7522 s and 2.5750 to 0.8488 s respectively after the remainder-kernel release. These historical timings provide context for the fresh investigation. Compare a future candidate using fresh pairs.

## Projection wall time and logical weight traffic

A separate Llama timer measured each sequential `project` call around the real public API. The table gives medians for three complete 32-step runs. The times include that call's allocation, dispatch, worker join, and destination commit.

| Projection | (M,K,N) | Calls across 32 steps | FP32 bytes per call | Wall across 32 steps | Logical weight GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Attention K/V | (1,2048,512) | 1,024 | 4 MiB | 0.1675 s | 25.6 |
| Attention Q/O | (1,2048,2048) | 1,024 | 16 MiB | 0.3026 s | 56.8 |
| FFN gate/up | (1,2048,8192) | 1,024 | 64 MiB | 1.0408 s | 66.0 |
| FFN down | (1,8192,2048) | 512 | 64 MiB | 0.4943 s | 69.5 |
| Tied classifier | (1,2048,128256) | 32 | 1,002 MiB | 0.4626 s | 72.7 |

Total projection time was **2.4608 s** (2.4538–2.6296 s), or **95.7%** of the measured 2.5723 s decode loop. FFN gate/up/down account for about 60% of the loop, and the classifier about 18%. Each token visits about 4.943 GB of projection weights (158.176 GB across 32 steps, or 64.3 logical GB/s within the projection timers). These rates use `4*K*N` logical weight bytes divided by wall time. Hardware cache and DRAM bytes, miss rates, and latency were unavailable. The classifier traverses one immutable 1,002 MiB tensor once per token. The FFN traverses 48 different 64 MiB tensors per token. The activation is 8 KiB for gate/up and the classifier, and 32 KiB for down. Those activations fit the reported per-level L1 data capacities. The weights exceed the reported L2 sizes. The loops stream four separate ranges from the row-major storage and reuse the input across output columns.

At P=8, the 4 MiB attention K/V projection uses two participating goroutines, including the caller, and each receives 256 output columns and 2 MiB of weights. Q/O uses eight, each with 256 columns and 2 MiB. FFN gate/up uses eight with 1,024 columns and 8 MiB per shard. Down uses eight with 256 columns and the same 8 MiB per shard. The classifier uses eight with 16,032 columns and 125.25 MiB per shard. Selection is capped at eight and by `GOMAXPROCS` and budgets 2 MiB of full output-column tiles per worker. Each qualifying call creates helpers for all shards except the caller's last one and uses one `WaitGroup` join. At P=1 it calls the same arithmetic serially. Helpers live for one call. Two simultaneous calls can offer up to 16 participating goroutines to the eight Go execution slots, including the callers.

## Worker and locality experiments

`GOMAXPROCS` is the existing control for these kernels. The sweep used three fresh processes per setting, in orders 1/2/4/8, 8/4/2/1, and 4/1/8/2. It also ran the unchanged public `GemmInto` with real checkpoint tensors, a preallocated destination, and 48 timed M=1 operations per case. Reuse applies one immutable tensor repeatedly. Rotate uses 32 different gate/up tensors or 16 different down tensors. The pinned model has one tied classifier, so the classifier uses the same tensor per call. Its in-model timing includes the intervening block projections.

| GOMAXPROCS | Single-session tokens/s | Two-session aggregate tokens/s | Two-session wall (s) | Faster/slower rate ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.829 (2.824–2.909) | 2.813 (2.811–2.921) | 22.750 (21.910–22.770) | 1.001 (1.001–1.002) |
| 2 | 5.258 (4.603–5.661) | 5.743 (5.687–5.816) | 11.140 (11.000–11.250) | 1.001 (1.001–1.002) |
| 4 | 8.852 (7.716–9.185) | 10.140 (8.549–10.320) | 6.314 (6.203–7.486) | 1.004 (1.002–1.005) |
| 8 | 12.500 (10.620–12.510) | 14.710 (13.260–15.150) | 4.351 (4.225–4.825) | 1.004 (1.004–1.004) |

| GOMAXPROCS | Gate/up median ms, reuse / rotate | Down median ms, reuse / rotate | Classifier median ms |
| ---: | ---: | ---: | ---: |
| 1 | 4.483 / 4.979 | 4.339 / 4.646 | 71.612 |
| 2 | 2.360 / 2.876 | 2.249 / 2.743 | 37.301 |
| 4 | 1.505 / 1.627 | 1.158 / 1.352 | 20.076 |
| 8 | 0.967 / 1.081 | 0.861 / 0.968 | 14.779 |

The P=8 rotating FFN medians are 12% slower than reuse for both shapes. There are outliers in the rotated results at lower process limits. The P=8 classifier rates are similar in isolation (14.78 ms/call) and in decode (14.46 ms/call). That large classifier amortizes one fan-out/join over about 125 MiB per shard, versus 8 MiB per FFN shard. Q/O at P=8 measured 0.249/0.314 ms on reused/rotating weights. K/V measured 0.148/0.170 ms and showed little change past P=2 because its projection already reaches the two-shard budget. At P=8 the gate/up, down, and classifier cases all improved over P=4. These process-wide sweeps change Go scheduling and the upper bound of kernel parallelism together. They leave open the effect of changing one shape's worker count at the same P. They support preserving the current dispatch in the first arithmetic experiment.

### Streaming control and scheduling

A separate process measured three trials of eight passes over each prefaulted 1 GiB buffer, split into disjoint contiguous spans at P=1, 2, 4, and 8. The Go standard library's vectorized byte count read 57.8 GB/s at P=1 and 96.9 GB/s at P=8 (P=8 range 96.1–97.5). The disjoint buffer copy measured 81.5 logical read-plus-write GB/s at P=8 (81.4–82.3). Single-worker projections move roughly 14–16 logical weight GB/s across several shapes. At P=8 the real projection path moves 64.3 logical weight GB/s. The different access patterns and instruction mixes make this a streaming reference. The measured gap supports investigating load and arithmetic throughput together with locality. It gives no direct measurement of projection DRAM saturation, cache misses, or FMA stalls.

The Go execution traces observed 19,168 helper goroutines in 32 single-session decode steps and 38,336 in 64 two-session steps. The first timing below is creation to first scheduled execution, and the second is time in Go's running state. Each cell gives median and 95th percentile per helper.

| Path | Creation to first run, one session | Running, one session | Creation to first run, two sessions | Running, two sessions |
| --- | ---: | ---: | ---: | ---: |
| Block projections | 31.6 / 164.9 µs | 615.4 / 883.7 µs | 185.6 / 871.6 µs | 615.2 / 1,002.5 µs |
| Classifier | 45.4 / 1,582.7 µs | 13.16 / 14.63 ms | 509.2 / 11,152.7 µs | 11.78 / 15.55 ms |

Across the single-session trace, helpers accumulated 12.32 s in the Go running state over 2.605 s of decode, equivalent to about 4.73 helpers running on average. The corresponding two-session trace was 25.74 helper-seconds over 4.262 s, or about 6.04. These are Go scheduler states for spawned helpers. Caller shards are additional, and the totals can include time while the OS deschedules a thread. Aggregate blocked time at the join was 0.800 s across the single caller and 4.600 s summed across both concurrent callers. The wait includes the tail of useful work, so only a fraction can be removed by improving dispatch. Condition-signaling CPU, helper runnable delay, and wall time measure different phenomena. Every observed helper exited, and helper waiting time after launch was negligible. The higher two-session runnable delays make aggregate throughput and both lane measurements important candidate guards.

## Remaining candidates and ranking

1. **Ordered single-row host GEMM, neuralnetwork.** The sustained loop dominates both elapsed wall and labeled CPU on every generated token. The controls support testing for higher load/FMA throughput with the current immutable layout and worker contract. Preserve the classifier and concurrent guards as part of this change.
2. **Cold checkpoint staging, Llama or cross-repository.** Loading accounts for 9,758 MiB, or 91.6%, of cumulative sampled primary-process allocations. The allocation stacks attribute about 4,723 MiB to `ReadFloat32`, 4,714 MiB to the required tensor ownership allocation and copy, and 320 MiB to rotary row relayout. A separate three-process loader probe measured medians of 1.868 s loading, 0.932 s conversion-loop time including first writes, 0.154 s reads, 0.022 s reader allocation, 0.441 s nonfinite scanning, and 0.290 s private tensor construction and copy. Snapshot hashing runs before loading and can warm OS file pages. The per-tensor sampled heap peak was about 6.60 GB, post-GC retained heap was 4.94 GB, and probe process peak RSS was 6.62 GB. Reducing startup work and transient heap has value, with separate ownership review before changing private copies or tied weights. Startup is paid once per decoder.
3. **Complete-group prefill, neuralnetwork.** Three-row projection accounted for 78.0% of labeled 29-token prefill CPU and 85.9% at 27 tokens. The block projection timers measured 0.752 s and 0.566 s respectively across independent runs. The shorter controls and the remaining one- and two-row tails need the same guard set. This is a sizeable longer-prompt cost with less reach in sustained generation.
4. **SwiGLU staging, Llama and neuralnetwork.** The allocation profiles found 261.1 MiB across three 29-token prefills and 190.8 MiB across 64 decode steps, including 93.8 MiB in sigmoid during decode. The selected decode profiles sampled only 0.22–0.32 CPU seconds in zeroing overall. The single-session trace recorded one GC concurrent mark phase of about 0.69 ms in the measured decode interval. This trace and the CPU results give allocation traffic lower priority for a latency pass. A longer-running memory-focused experiment can test a different allocation or GC regime. One-row dispatch accounts for just 3.5 MiB of the 64-step sampled decode traffic.

## Ownership and call path

Llama owns decoder orchestration, sessions, and immutable model weights. Neuralnetwork owns the host GEMM adapter, private product loop, output-column tiles, worker dispatch, and tensor staging. Work in `github.com/itsmontoya/neuralnetwork` from tag `v0.10.17` at `cfda2464435c568c46b71b58fd5518514e72c72f`. Use Llama unchanged for the release comparison, apart from the benchmark commit and the temporary module pin used to link the candidate.

- In Llama, `(*Session).Decode` in `session.go` invokes `(*Decoder).forwardBlock` in `decoder.go` for each of 16 blocks. The attention path calls `headProjection` for Q/K/V and `project` for O. The FFN calls `project` for gate/up, computes `swiGLU`, then calls `project` for down.
- `(*Session).commit` calls `(*Decoder).projectLogits`. It final-normalizes the single hidden row and calls `project` against `d.output`. The loader sets `d.output = d.embedding` for this snapshot.
- Llama's `project` in `tensor_ops.go` constructs a disjoint FP32 destination after `tensor.GemmShape` and calls `tensor.GemmInto(input, weight, nil, out, tensor.GemmConfig{TransposeB: true, Alpha: 1})`. The default build and `purego` use `tensor.matrixProductValues` → `tensor.matrixProductInto` in `tensor/matrix_product_host.go` → `internal/float32product.RightTranspose`. The optional Metal build has its own adapter.
- At `M=1` and `N>=4`, `RightTranspose` calls `dispatchRightTransposeOneRow`, then `rightTransposeOneRow` directly or via `rightTransposeOneRowParallel`. Each output covers K products in ascending order. The main loop loads one input scalar and four weight scalars, advances four accumulators, and writes four output values. The weights are contiguous within each of four separate K-long streams. A partial column tile uses the serial scalar tail. At `M>1`, `dispatchRightTransposeRows` owns complete groups and one- or two-row tails. The one-row remainder also calls `rightTransposeOneRow` inside the existing shard, so verify it if the common arithmetic changes.

## Numerical contract

Preserve ascending-K FP32 accumulation per output, initial positive zero, existing architecture/build FMA behavior, and exact FP32 bits for finite ordered-reference cases. On the measured ARM64 build the inner loop emits four scalar `FMADDS` instructions per K iteration. Independent output columns can be interleaved without changing order within a column. K-split partial sums and horizontal reductions change the accumulation order. Retain current nonfinite and signed-zero behavior, all alpha/beta and bias semantics, empty contractions, zero-sized output behavior, and scalar column tails. Use existing Llama tolerances unchanged.

## Ownership and lifetime

Inputs are read-only. The raw kernel overwrites every assigned element in a disjoint, possibly dirty output and returns with no retained buffers. Public `GemmInto` validates shapes, aliases, types, bias, and destination. It stages the result, applies alpha and bias, and copies it to the caller's destination. Every returned error leaves a valid disjoint destination unchanged. Workers must finish before the call-owned temporary storage goes out of scope. Preserve retries, independent destinations, and concurrently shared immutable weights. In Llama, callers may retain returned rows and trace tensors. A failed session step leaves position uncommitted, and retry overwrites provisional KV before attention reads it.

## Concurrency, allocation, and memory

Llama sessions share only decoder resources and immutable weights. A session owns its KV cache, scratch, position, and returned logits. Preserve deterministic results, bounded worker fan-out, caller participation where applicable, and a join before staged outputs are visible. The host GEMM API is synchronous and has no cancellation parameter. Preserve that contract and the cancellation behavior of upstream callers, with no abandoned helper goroutines or background writes after error or return.

Treat workspace and goroutine allocation counts as diagnostics. Avoid new model-size retained buffers in the first candidate. A permanent FP32 copy would add 64 MiB per FFN tensor or 1,002 MiB for the tied classifier. Any later experiment that changes layout must preserve immutable tensor and tied-weight semantics, report amortization and steady-state/peak memory, and pass the same real-workload acceptance criteria. The current GEMM product workspace remains call-owned.

## Upstream verification

Run default and `purego` tests, targeted race tests, and `go vet`. Cover `TestRightTranspose`, `TestRightTransposeOneRowWorkerCount`, `TestRightTransposeOneRowAgainstOrderedReference`, `TestRightTransposeConcurrent`, `TestRightTransposeNonfiniteAndSignedZero`, `TestRightTransposeGroupWorkerCount`, `TestRightTransposeGroupsAgainstOrderedReference`, `Test_Gemm_RightTransposeSingleRowConcurrentRetry`, `Test_Gemm_RightTransposeGroupsConcurrentRetry`, and existing GEMM alias, bias, and empty-contraction tests. Add targeted regression cases for each new tile/fallback boundary and uneven output partition. Include varied K, full and partial column tiles, dirty pre-existing outputs, cancellation or failure paths exposed by callers, retries, and concurrent calls sharing the same weights but owning separate destinations. Compare P=1 with parallel outputs bit-for-bit on the same build. If using an architecture-specific path, verify the portable fallback and disassemble the optimized path to confirm the intended FMA and footprint.

Benchmark `(M,K,N)` of `(1,2048,8192)` for gate/up, `(1,8192,2048)` for down, and `(1,2048,128256)` for the classifier. Also cover `(1,2048,2048)` and `(1,2048,512)` for attention, small columns and irregular tails. Use the recorded real checkpoint tensors to compare repeated and rotating FFN/right-side weights. Measure the classifier in isolation and in full decode. Sweep `GOMAXPROCS=1,2,4,8` through public `tensor.GemmInto`, record actual worker counts, and compare two shared-decoder callers at the same process settings. A microbenchmark improvement alone does not meet acceptance.

## Llama integration verification

Use the baseline Llama commit above plus its Pass 8 concurrent benchmark commit. Build both pins with Go 1.27.3 and the same verified fixture, default tags, `GOARM64=v8.0`, `GOMAXPROCS=8`, `CGO_ENABLED=1`, `GOWORK=off`, and empty `GOFLAGS` on the same host. Use separate temporary module files to compare the released baseline to the immutable candidate while keeping the Llama source constant. Compare one timed operation per fresh process with no inference warm-up, with alternating baseline/candidate order. Keep the historical timing boundaries and the same setup and token semantics within each pair. Run at least 12 primary, seven single-session sustained, seven two-session sustained, five each for 27/28/29-token prefill, and the existing two-session prefill guard. Report each build's median and range, the median and range of the individual paired deltas and ratios, and improved-pair counts. Keep outliers in the record and repeat inconsistent workloads with the same methodology.

```sh
export GOTOOLCHAIN=go1.27.3 GOARM64=v8.0 GOMAXPROCS=8 CGO_ENABLED=1 GOFLAGS= GOWORK=off
export LLAMA_32_1B_MODEL_DIR=/path/to/verified/4e20de362430cd3b72f300e6b0f18e50e7166e08
export LLAMA_STORIES260K_MODEL=/path/to/stories260K.bin
export LLAMA_STORIES260K_TOKENIZER=/path/to/tok512.bin
go test -mod=readonly -count=1 -p 1 -v ./...
CGO_ENABLED=0 go test -mod=readonly -tags=purego -count=1 -p 1 -v ./...
env -u LLAMA_32_1B_MODEL_DIR go test -mod=readonly -race -count=1 ./...
go vet -mod=readonly ./...
out=$(mktemp -d)
go test -mod=readonly -c -o "$out/llama.test" .
"$out/llama.test" -test.run '^$' -test.bench '^BenchmarkLlama32SustainedDecode$' -test.benchtime 1x -test.benchmem -test.v
"$out/llama.test" -test.run '^$' -test.bench '^BenchmarkLlama32ConcurrentSustainedDecode$' -test.benchtime 1x -test.benchmem -test.v
```

The Transformers comparison is `TestLlama32Logits` with `atol=rtol=1e-3`. Cached/full-prefix parity is `TestLlama32CachedParity` with `atol=rtol=1e-4`. Require frozen greedy winners and continuation IDs, byte-identical default and purego CLI output in `cmd/generate:TestLlama32Generation`, TinyStories logits/tokenizer/generation, synthetic MHA/GQA/MQA traces, shared-decoder session tests, retained output checks, and failed-step retries. The established race run includes the TinyStories and local fixtures. Cover the real concurrent benchmark under race if the host can run it without memory pressure.

| Pinned default ARM64 logit scope | SHA-256 of little-endian FP32 |
| --- | --- |
| Prefill 27 | `0b087b0839da36ec06a3a8f777d4c3c944ccc7439107fcad39b6ad0239ed9afc` |
| Prefill 28 | `cc93319c3c0701fd9cd3fdc2080afe43d2f82ba7cd236a698d19518836854e8c` |
| Prefill 29 | `6ffb4dc93003480aebdd51e9f1fee47ae3836eea9fb35ce2508f5d618a9871a5` |
| Sustained terminal row, both sessions | `b72d983e8b5c45f83ad858b52f1d27b87a7ad8118be6c5d315db2ea6fc6fd84a` |
| Two concurrent six-token prefills | `f2e070e41d1eac82f0159284cfdd0d070aa06ef703c833fa6fb781a5b7b487c1` |

Both sustained benchmarks print the seed selection followed by 32 greedy selections. The pinned sequence follows.

```text
12366 13 1102 374 279 1455 95551 3363 304 9822 323 279 3224 753 6864 323 7928
3363 13 12366 374 279 1455 95551 16036 3158 304 279 7665 9323 323 279 1917
```

The concurrent benchmark also prints both session rates. The primary benchmark checks its suffix and decoded bytes against `testdata/llama32_reference.json`. Report prefill, TTFT, short decode seconds/token and tokens/s, primary cached generation wall and eight-output throughput, sustained decode, per-session concurrent latency and aggregate throughput, bytes/op, allocs/op, process wall, and peak RSS. Collect matched post-candidate CPU and allocation profiles. Compare aggregate sampled CPU, elapsed wall, arithmetic loop, and worker coordination independently.

## Acceptance, completion, and return

Prioritize a repeatable real single-session sustained improvement, with a working target of at least 5% in paired median decode latency and improvement in a majority of pairs. Treat a repeatable degradation over 5% in two-session aggregate throughput, either session's latency, primary TTFT, or a prefill guard as a regression to resolve or flag for review before recommending adoption. Report changes in fairness using per-session completion latency and the ratio between the two rates. Require unchanged pinned results and ownership behavior, bounded parallelism, no races or goroutine leaks, and no unexplained memory or concurrency cost. If the candidate loses these comparisons or the timing direction remains inconclusive, return the evidence and state that the candidate did not qualify for integration.

Keep commits small and logical. Separate measurement infrastructure from the selected implementation and its result record when that helps review. Finish with a reviewable upstream implementation and passing checks, paired real-Llama results, and an immutable candidate. Return commit hashes, test commands and results, paired raw logs and summary, CPU/allocation profiles and scheduler evidence, the published module version or immutable commit and its module checksum, and exact Llama integration instructions. Include reproducible benchmark commands, supported configurations, known tradeoffs, and any remaining validation gaps. Llama integration begins after the upstream result is returned.

## Fresh baseline validation

The pinned default and `purego` Llama suites passed, including Transformers logits, cached/full-prefix parity, the byte-exact CLI continuation, TinyStories, synthetic MHA/GQA/MQA traces, retained outputs, retries, and shared-decoder sessions. The largest Transformers absolute logit error was `4.95910645e-05` and the largest tolerance fraction was `0.0316354157` in both builds. The checked cached/full-prefix rows matched exactly. The standard race suite passed with TinyStories and local fixtures, and `go vet` passed. The additional full-size 32-step concurrent race benchmark was stopped after five minutes. Its log records the start of the pinned workload and no completed race result. All 60 supplemental, sustained, and concurrent timing logs passed the frozen suffix or digest checks across the main series and worker sweeps. The 12 primary runs checked their output against the frozen generation reference.

## Reproducing the evidence

Extract `/private/tmp/llama-pass8-evidence.I9Q7GPj6.tar.gz`. Update fixture paths and choose a writable Go cache in the included `env.sh`. Run from the Llama source root at the benchmark commit or the handoff commit. The baseline source and module pin are recorded above. The scripts regenerate diagnostic overlays for Llama source only. If the Llama checkout is at the Pass 7 merge and lacks the benchmark commit, apply the archive's `concurrent-benchmark.patch` in Llama before building.

```sh
out=/path/to/extracted/llama-pass8.I9Q7GPj6
. "$out/env.sh"
python3 "$out/prepare-overlays.py" "$PWD"
python3 "$out/prepare-pass7.py" "$PWD"
sh "$out/build.sh" "$out"
go test -mod=readonly -c -o "$out/concurrent.test" .
sh "$out/collect.sh" "$out"
sh "$out/diagnose.sh" "$out"
sh "$out/more.sh" "$out"
sh "$out/analyze.sh" "$out"
python3 "$out/check-digests.py" "$out"
```

`CPU-*.pprof` and `CPU-*.raw.txt` are sampled CPU profiles. `CleanAlloc-*.pprof` and `Alloc-*.raw.txt` are allocation profiles from binaries without the CPU shape diagnostic. `ProjectionWall-*.log` times Llama's sequential projection calls separately. `Trace-Sustained.out`, `Trace-ConcurrentDecode.out`, and `Trace-*.pprof` carry scheduler and synchronization evidence. `baseline-summary.txt`, `profile-summary.txt`, `trace-summary.txt`, `trace-concurrent-summary.txt`, and per-run logs provide the extracted metrics. To compare a returned candidate, keep the same Llama source and use temporary module files. Set `nn_candidate` to the returned immutable version or commit and rebuild both binaries from the same benchmark sources.

```sh
for variant in before after; do
    mkdir -p "$out/$variant"
    cp go.mod "$out/$variant/go.mod"
    cp go.sum "$out/$variant/go.sum"
done
go get -modfile="$out/after/go.mod" "github.com/itsmontoya/neuralnetwork@$nn_candidate"
go mod tidy -modfile="$out/after/go.mod"
for variant in before after; do
    dir="$out/$variant"
    go test -mod=readonly -modfile="$dir/go.mod" -c -o "$dir/llama.test" .
    go test -mod=readonly -modfile="$dir/go.mod" -overlay "$out/overlay.json" -c -o "$dir/diagnostic.test" .
    go version -m "$dir/llama.test"
done
sh "$out/paired-later.sh" "$out"
```

Profile the same workloads and scopes on the candidate after the paired timing series.
