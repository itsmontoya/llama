# Neuralnetwork handoff for Llama Optimization Pass 7

## Objective

Reduce host FP32 right-transposed GEMM latency for one- and two-row remainders after complete three-row groups. Use `neuralnetwork v0.10.16` at `222fdc71d51eef23bc4a7edaeaeacc18e5ca7b7a` as the upstream baseline. Keep the public API, output bits, and backend contracts. Evaluate the same scalar two-row path at standalone `M=2` as part of this change.

## Problem

Llama's 27-, 28-, and 29-token prefills all project nine complete three-row groups per block projection. At 28 and 29 tokens the caller waits for one or two serial scalar rows after the group workers join. In five fresh runs per length, median prefill rose from 0.689 s to 1.628 s and 2.619 s. The one-row and two-row scalar loops separately accounted for 1.63 s and 3.90 s of aggregate CPU across three profiled prefills per length. The six-token primary prompt has two complete groups and zero remainder rows. It serves as a regression guard, as do sustained decode and concurrent six-token prefill.

## Fresh Profiling Evidence

### Baseline and workload

Llama `5fb56f5bd22ec0f8157b3664fd2ef0d914dbd785` includes the completed Pass 6 integration and README results. Its Go 1.27.3 build uses `neuralnetwork v0.10.16` with sum `h1:T4UdQg2b1p0K8g4uRkV8xeTvTvCNst59gFfGeMiTirc=`, `github.com/tphakala/simd v1.3.0`, and `golang.org/x/sys v0.45.0`. The module cache verified successfully. Measurements used Apple M3 (four performance and four efficiency cores, 24 GB RAM) on AC power, macOS 26.5.2 (25F84), Darwin arm64, default build tags, `GOARM64=v8.0`, `GOMAXPROCS=8`, `CGO_ENABLED=1`, empty `GOFLAGS`, and `GOWORK=off`.

Use `meta-llama/Llama-3.2-1B` snapshot `4e20de362430cd3b72f300e6b0f18e50e7166e08`. Each process verifies the sizes and SHA-256 hashes of all six files against `testdata/llama32_manifest.json`. The primary workload is `The capital of France is`, six tokens with BOS and eight greedy selections using one prefill and seven decode calls. Context cap is 512, and session capacity is 13. TTFT includes the first greedy selection and excludes cache creation. The decode metric includes greedy selection for the remaining seven outputs. Sustained decode times positions 6–37, including greedy selection with session creation and initial prefill outside the timed interval.

The 29-token prompt is `France is a country in Western Europe. It shares borders with Belgium, Germany, Switzerland, Italy, and Spain. The capital of France is`. Tokenize it once with BOS and use its first 27 or 28 IDs for those diagnostics. Their timed intervals cover prefill and greedy selection. The two-session guard launches both six-token prefills against one shared decoder with independent caches. Fresh processes ran one timed operation at `-test.benchtime 1x`, with zero inference warm-ups. The primary, sustained, and 29-token results use the tracked benchmarks. The 27/28 and concurrent runs reuse the Pass 6 diagnostic overlay. Whole-process wall and Darwin peak RSS include manifest verification, loading, tokenizer setup, and inference.

| Primary metric, 12 trials | Median | Range |
| --- | ---: | ---: |
| Prefill (s) | 0.1838 | 0.1736–1.727 |
| TTFT (s) | 0.1839 | 0.1737–1.727 |
| Decode (s/token) | 0.08091 | 0.07754–0.14657 |
| Decode (tokens/s) | 12.36 | 6.820–12.90 |
| Cached generation (s/op) | 0.7502 | 0.7237–2.5893 |
| Output (tokens/s) | 10.66 | 3.09–11.05 |
| Bytes/op | 131,321,784 | 131,309,936–131,334,112 |
| Allocs/op | 39,896 | 39,869–39,915 |
| Process wall (s) | 3.831 | 3.754–6.013 |
| Peak RSS (GiB) | 6.374 | 6.170–6.375 |
| Model load (s), excluding snapshot verification | 1.859 | 1.823–2.931 |
| Tokenizer setup (s) | 0.1681 | 0.1592–0.5309 |

| Workload | Trials | Timed s/op, median (range) | B/op, median (range) | Allocs/op, median (range) | Process wall s, median (range) | RSS GiB, median (range) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sustained decode, 32 steps | 7 | 2.631 (2.510–4.074) | 329,811,232 (329,804,320–329,824,952) | 149,563 (149,548–149,587) | 5.964 (5.773–7.399) | 6.376 (6.363–6.378) |
| 27-token prefill | 5 | 0.689 (0.687–1.439) | 261,768,768 (261,765,264–261,793,144) | 7,058 (7,053–7,106) | 3.790 (3.761–5.645) | 6.373 (5.626–6.374) |
| 28-token prefill | 5 | 1.628 (1.627–2.465) | 271,349,696 (271,343,352–271,353,528) | 7,046 (7,037–7,058) | 4.775 (4.669–6.125) | 6.373 (6.046–6.374) |
| 29-token prefill | 5 | 2.619 (2.514–3.494) | 282,514,808 (282,492,352–282,527,536) | 7,081 (7,032–7,105) | 5.733 (5.537–7.492) | 6.372 (5.959–6.372) |
| Two concurrent six-token prefills | 3 | 0.377 (0.285–1.249) | 116,467,976 (116,462,224–116,479,320) | 13,982 (13,963–14,019) | 3.447 (3.364–4.573) | 6.372 (6.351–6.374) |

Sustained decode separately reached 12.16 tokens/s (7.854–12.75), or 0.08222 s/token (0.07844–0.12730). Across the five rounds, the 28-token prompt took a median 0.940 s longer than the 27-token prompt (0.841–1.776 s). At 29 tokens the corresponding difference was 1.931 s (1.075–2.795 s). All five rounds showed both increases. The differences also include the ordinary cost of processing more tokens in attention and activations.

### CPU and wall time

A Llama-only overlay applied pprof labels to manifest verification, loading, tokenizer setup, prefill, first greedy selection, short decode, sustained decode, and decode greedy selection. A nested projection label followed workers and distinguished classifier, KV, Q/O, gate/up, and down. Table entries for kernels are exclusive CPU samples and their fraction of phase CPU. CPU seconds sum sampling across workers. Elapsed seconds come from phase timers, whose profiles may affect latency. Each CPU sample represented 10 ms.

| Phase | Labeled CPU | Complete groups | One-row remainder | Two-row remainder | Standalone M=1 blocks | M=1 classifier | Profiled elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary prefill ×6 | 7.33 s | 6.25 s (85.3%) | — | — | — | 0.85 s (11.6%) | 1.867 s |
| Primary short decode, 42 steps | 9.62 s | — | — | — | 5.89 s (61.2%) | 2.71 s (28.2%) | 3.565 s |
| Sustained decode, 64 steps | 14.09 s | — | — | — | 8.69 s (61.7%) | 3.92 s (27.8%) | 5.124 s |
| Prefill 27 ×3 | 10.92 s | 9.46 s (86.6%) | — | — | — | 0.67 s (6.1%) | 3.360 s |
| Prefill 28 ×3 | 12.81 s | 9.13 s (71.3%) | 1.63 s (12.7%) | — | — | 0.40 s (3.1%) | 5.854 s |
| Prefill 29 ×3 | 11.97 s | 6.09 s (50.9%) | — | 3.90 s (32.6%) | — | 0.52 s (4.3%) | 8.212 s |

Gate/up and down together accounted for 1.48 of the 1.63 s at 28 tokens and 3.34 of the 3.90 s at 29 tokens. Primary load used 1.95 sampled CPU seconds and 2.910 elapsed seconds in its profiling process. Manifest verification was 0.96 sampled and 1.010 elapsed seconds in that process. Across six primary generations, first greedy selection took 0.614 ms in total, and decode greedy selection took 4.328 ms in total. Across both 32-step sustained iterations, decode greedy selection took 6.622 ms. Both selection phases were below CPU sampling resolution.

A timing-only overlay measured wall inside the normal Llama `project` calls, including product staging and dispatch. Three rounds ran in separate processes.

| Actual Llama projections | Median wall (range) |
| --- | ---: |
| Primary prefill, M=6 blocks | 0.147 s (0.145–0.152) |
| Primary prefill, M=1 classifier | 0.015 s (0.014–0.017) |
| 27-token prefill, M=27 blocks | 0.576 s (0.550–1.633) |
| 28-token prefill, M=28 blocks | 1.505 s (1.495–2.372) |
| 29-token prefill, M=29 blocks | 2.413 s (2.410–2.709) |
| Sustained decode, M=1 blocks across 32 steps | 1.958 s (1.950–1.969) |
| Sustained decode, M=1 classifier across 32 steps | 0.463 s (0.452–0.483) |
| Sustained decode, all projections across 32 steps | 2.432 s (2.402–2.442) |

The median of round-by-round block projection differences was +0.945 s for 28 versus 27 tokens (−0.128 to +1.795 s, two of three positive) and +1.834 s for 29 versus 27 tokens (+1.076 to +1.863 s, all three positive). Each comparison includes the same number of groups and the rest of the public projection work. The CPU profile attributes the scalar loop directly. The wall differences estimate its cost from whole block projection timings and include scheduling variance.

A separate diagnostic benchmark rotated actual checkpoint weights through the public GEMM API and reused its destination. Below are ranges across two P=8 sweeps, in ms per projection. M=2 exposes the scalar path with zero complete groups. M=4 and M=5 add a remainder to the same one group as M=3.

| Shape | M=1 | M=2 | M=3 | M=4 | M=5 | M=6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention K/V | 0.161–0.182 | 1.980–1.982 | 0.114–0.117 | 1.114–1.128 | 2.106–2.112 | 0.185–0.186 |
| Attention Q/O | 0.300–0.488 | 7.922–7.948 | 0.326–0.331 | 4.323–4.334 | 8.282–8.302 | 0.580–0.601 |
| FFN gate/up | 1.047–4.254 | 31.701–31.865 | 1.187–1.201 | 17.044–17.051 | 32.888–32.962 | 2.344–2.375 |
| FFN down | 1.058–2.721 | 32.755–33.239 | 1.095–1.208 | 17.472–17.516 | 33.829–33.871 | 2.139–2.178 |
| Classifier, one tied weight reused | 14.342–20.623 | — | — | — | — | — |

In the FFN cases M4−M3 was 15.84–16.38 ms and M5−M3 was 31.70–32.73 ms. At P=1, M=1 took 5.364 ms for gate/up and 4.755 ms for down. M4−M3 was 15.83 and 16.37 ms respectively, showing room to evaluate tiling even before introducing parallel dispatch. With P=4, M4−M3 was 15.88 ms for gate/up and 16.39 ms for down. These are comparisons among existing paths, using synthetic activations and checkpoint weights. Wall differences and CPU profiles in the real model establish workload relevance. Candidate end-to-end gains remain to be measured.

### Allocation traffic

Allocation profiles ran in separate processes at 64 KiB sampling. Rows use sampled cumulative bytes attributed by call stack. GEMM includes dispatch and the staged product. Projection includes GEMM plus the Llama destination and shape work. SwiGLU includes sigmoid. Benchmark B/op and allocs/op above use the benchmark's direct counters.

| Scope | Phase total MiB | Llama projection MiB | GEMM MiB | SwiGLU MiB | Sigmoid staging MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary prefill ×6 | 334.48 | 112.87 | 56.93 | 107.74 | 53.28 |
| Primary short decode, 42 steps | 410.44 | 166.93 | 84.53 | 126.48 | 63.68 |
| Sustained decode, 64 steps | 623.64 | 251.41 | 127.17 | 189.43 | 95.41 |
| Prefill 27 ×3 | 747.69 | 235.81 | 118.19 | 243.00 | 121.50 |
| Prefill 28 ×3 | 774.57 | 244.77 | 122.52 | 252.00 | 126.00 |
| Prefill 29 ×3 | 810.50 | 255.59 | 127.24 | 261.00 | 130.50 |

The scalar remainder loop shows zero separate allocation traffic. The larger M shapes allocate larger outputs and activations. For the 29-token process, complete-group coordination sampled 0.375 MiB and roughly 3,074 objects across three iterations. In the sustained profile, the standalone one-row dispatch sampled 4.065 MiB of coordination allocation. In the primary process `LoadHFModel` attributed 9,758.67 MiB out of 10,652.86 MiB of cumulative allocations (91.61%). `ReadFloat32` accounted for 4,723.82 MiB and the ownership copy in `tensor.FromFloat32` for 4,714.59 MiB. A separate P=8 diagnostic forced GC immediately after loading with the decoder live and observed 4,943,540,048 bytes (4.604 GiB) of live Go heap. Primary process peak RSS was 6.374 GiB at the median. These are separate measures of allocation traffic, live objects, and process peak.

The Pass 6 release measured 19.6% and 37.3% labeled prefill CPU in the one- and two-row remainders, on different runs. The CPU table reports fresh Pass 7 samples. Its lower percentages reflect the measured phase totals, which include CPU consumed in parallel groups. Both passes found a substantial serial cost at 29 rows.

## Repository Ownership

Neuralnetwork owns dispatch, the contraction loops, and validated FP32 storage. Llama supplies whole tensors and the same public GEMM operation for block and classifier projections. A reusable remainder fix belongs in `internal/float32product`. It can reach existing tensor and pure matrix callers through their current adapters.

## Exact Call Path

1. Llama `session.go` runs `(*Session).Prefill` → `(*Decoder).forwardPrefix` → `(*Decoder).forwardBlock` in `decoder.go`. `attention.go:headProjection` calls `project` for Q, K, and V before reshaping for heads. The attention output, FFN gate, up, and down projections call `project` directly from `forwardBlock`. Session prefill gathers the last hidden row and `session.go:(*Session).commit` calls `decoder.go:(*Decoder).projectLogits` for the classifier.
2. `session.go:(*Session).Decode` passes one row through `forwardBlock` and uses the same `commit` classifier. `decoder.go:(*Decoder).Forward` goes through `forwardPrefix` and `projectLogits` for full-prefix logits. Its classifier can receive multiple rows.
3. `tensor_ops.go:project` calls `tensor.GemmShape`, allocates the FP32 destination with `tensor.New`, then calls `tensor.GemmInto(input, weight, nil, out, config)` with `TransposeB: true, Alpha: 1`.
4. In `github.com/itsmontoya/neuralnetwork/tensor`, `gemm.go:GemmInto` validates dtype, shape, aliasing, and bias. The default/purego implementation in `matrix_product_host.go` calls `matrixProductValues` → `matrixProductInto` → `float32product.RightTranspose`. It stages FP32 products and `GemmInto` applies coefficients and copies into the caller destination. The separate Metal-tagged adapter in `matrix_product.go` routes through `matrix.(*Matrix).MatMulRightTransposeInto`. The host fallback `matrix/matmul_pure.go:matMulRightTransposeIntoPure` also calls `float32product.RightTranspose`.

## Current Behavior

In `internal/float32product/product.go`, `RightTranspose` returns on empty output, clears for zero contraction width, and dispatches a standalone row with `N >= 4` via `dispatchRightTransposeOneRow`. Other nonempty shapes go through `dispatchRightTransposeRows`. It calls `dispatchRightTransposeGroups` for `M/3` complete groups and then `rightTransposeScalarRows` for any remaining rows. `rightTransposeScalarRow` traverses each output column and uses `scalarDotProduct` in ascending K. At `M=2` there is no complete group, so both rows take this loop. For `M=28/29` the group workers have already joined before the scalar loop starts.

In `right_transpose_parallel.go`, group workers shard complete four-column tiles with stride N. One worker runs on the caller, consumes the column tail, and joins the other workers before returning. The group worker cap is eight, limited by `GOMAXPROCS`, tile count, and a budget of 1 MiB of logical FP32 weight visits per worker. Standalone M=1 uses its own four-column tiling and 2 MiB per-worker budget with the same cap. Its serial and parallel kernels remain available to evaluate for a one-row remainder.

## Desired Behavior

Use ordered host kernels for the remainder and enough parallel work to materially lower 28- and 29-token Llama prefill latency. Preserve the dispatch performance at M=1, M=3, M=6, and M=27, including the repeated calls at those sizes in the primary, sustained, and concurrent benchmarks. Retain a serial path for low work or one scheduler thread.

## Proposed Implementation Direction

Start by benchmarking reuse of the existing tiled single-row kernel on the one-row remainder slices. Measure both its serial path and cost-gated sharding. For two rows, compare two existing single-row computations against a two-row tile that reuses weight loads across both rows. Compare a single sharded dispatch handling both rows against two dispatches. For mixed group/remainder calls, also measure handling all rows within one existing output-column shard and joining once. Keep each output in its original column and row stride, and handle an incomplete column tile. Select the simplest variant with a stable gain in kernel, public GEMM, and real Llama measurements.

## Alternatives Considered

- **Single-row decode dispatch and bandwidth.** Standalone block and classifier kernels account for 89.5% of fresh sustained decode CPU, and condition signaling accounts for 1.37 s (9.7%). Three alternating real decode comparisons at P=4 and P=8 reached 9.898 versus 11.28, 7.809 versus 9.239, and 9.922 versus 12.61 tokens/s. Reducing P to four decreased throughput by a median paired 15.5% (12.3–21.3%). This global scheduler sweep leaves room for per-shape tuning, but offers less direct evidence for a change than the measured gap between scalar and tiled paths.
- **Checkpoint staging.** A fresh process spends a median 1.859 s inside `LoadHFModel`. The cumulative allocation is dominated by conversion and copying into private tensor storage. The recurring prefill cost now has a smaller reusable implementation scope. Eliminating the loader-to-tensor copy would require an ownership-preserving API or independent evidence for a safe loader-local change.
- **SwiGLU temporaries.** Sigmoid staging accounts for about half of sampled SwiGLU bytes in sustained decode and 29-token prefill. The 29-token profile sampled 0.39 s (3.3%) in zeroing across all prefill stacks, and sustained decode sampled 0.09 s (0.6%). Reduced allocation traffic may help live memory and GC. These runs give stronger evidence for elapsed latency in the scalar remainder.
- **Remainder implementation choices.** Reusing the standalone M=1 kernel is the smallest experiment. A tiled two-row path could share weight loads. Fusing group and remainder dispatch could avoid a second worker launch. Padding M to a multiple of three would increase arithmetic and staged output size. Measure per-call overhead and independent sessions before choosing among those approaches.

## Numerical Correctness

Preserve one FP32 accumulator initialized to positive zero per output, ascending K, and the effective rounding and FMA behavior of the same Go build and architecture. Go 1.27.3 arm64 disassembly shows `FMADDS` in the current scalar remainder, one-row, and three-row loops. Partition full dot products by output columns or rows and preserve coefficient, bias, and copy order. Require bitwise parity with the ordered v0.10.16 path for finite inputs and signed zero within each build configuration. Match existing NaN/Inf expectations. Keep the pinned Llama tolerances, winners, and logit digests.

## Ownership and Lifetime

The raw kernel reads immutable left and right storage, overwrites every output element even if its prior value was dirty, and retains no buffers. Public GEMM validates all aliases and operands and leaves a disjoint destination unchanged on any returned error. Keep the product workspace call-owned, preserve output staging and error/retry behavior, and finish every write before releasing a workspace. Llama's callers may retain returned logits or trace tensors across steps. An unsuccessful step leaves the session position uncommitted, and the next attempt overwrites any provisional cache data before it becomes visible.

## Concurrency

Independent sessions may share one decoder and its read-only weights. Give each invocation its own destination and scratch. Bound workers by the current scheduler budget and the amount of work, test concurrent invocations and the P=1 fallback, and let the caller compute one shard. Workers must terminate and release references before the operation returns. The current API is synchronous and has no cancellation parameter. Preserve that lifecycle so application-level cancellation or error cannot leave workers using released buffers.

## Allocation and Memory

The raw scalar remainder allocates no payload or coordination objects. Preserve the existing GEMM staging and destination semantics. Bound any new coordination by participating workers and measure it separately from payloads, using raw B/op, public B/op, real prefill B/op, and alloc_objects. Avoid per-row or per-tile allocations and additional output clearing or weight copies. Quantify zeroing and coordination CPU and compare peak RSS in independent sessions sharing weights. Inspect the full working set and releases before considering any persistent buffer.

## Representative Shapes

A projection uses input `[M,K]`, checkpoint weight `[N,K]`, and output `[M,N]`. The pinned model has 16 layers, hidden width 2048, intermediate width 8192, 32 query heads, eight KV heads, head width 64, and 128256 tied embedding/classifier rows.

| Projection | K | N | Calls per layer or session step |
| --- | ---: | ---: | --- |
| Q and attention output | 2048 | 2048 | Two per layer |
| K and V | 2048 | 512 | Two per layer |
| Gate and up | 2048 | 8192 | Two per layer |
| Down | 8192 | 2048 | One per layer |
| Tied classifier | 2048 | 128256 | Once per session step, M=1 |

Use M=3/4/5 for one complete group plus zero/one/two rows, M=6 for two complete groups, and M=27/28/29 for nine complete groups plus zero/one/two rows. Include M=1 and M=2 as standalone controls. M=30 is the next complete-group guard at the long-prompt scale. For the classifier focus on M=1 and include a small mixed-M check for full-prefix callers.

## Benchmark Plan

- Extend the existing `tensor/right_transpose_benchmark_test.go:BenchmarkRightTranspose` and `tensor/right_transpose_groups_benchmark_test.go:BenchmarkRightTransposeGroups` coverage for raw `float32product.RightTranspose` and public `tensor.GemmInto`. Allocate operands and destinations outside the timed loop. Use each block K/N pair, repeated weights and rotating weights larger than cache. Record raw and public times, exact memory per operation, and logical weight visits separately.
- Measure standalone M=2 and matched M4−M3 and M5−M3 deltas at the same K/N/P. Include 27/28/29 and compare complete groups, remainder contributions, and whole-call overhead. Use P=1/2/4/8, widths around dispatch thresholds, N=0–5, N=513/8193, and short or odd K. Exercise parallel calls with shared weights and disjoint outputs.
- Build clean baseline/candidate binaries using Go 1.27.3 and the same GOARM64, CGO, and tags. Alternate within pairs in fresh processes. Report medians, ranges, paired deltas and improvement counts, sample projection and scheduler CPU, and track allocations. Test the resulting immutable candidate in Llama before claiming an end-to-end gain.

## Correctness Plan

Run upstream `TestRightTranspose`, `TestRightTransposeConcurrent`, `TestRightTransposeNonfiniteAndSignedZero`, `TestEmptyContraction`, `TestRightTransposeGroupWorkerCount`, `TestRightTransposeGroupsAgainstOrderedReference`, `TestRightTransposeOneRowWorkerCount`, and `TestRightTransposeOneRowAgainstOrderedReference`. Keep tensor `Test_Products_AliasesAndEmptyContractions`, `Test_Gemm_BiasValidation`, `Test_Gemm_RightTranspose`, `Test_Gemm_RightTransposeGroupsConcurrentRetry`, and `Test_Gemm_RightTransposeSingleRowConcurrentRetry` green. Keep matrix `Test_MatMulRightTransposeInto_AscendingOrder` green.

Add coverage that triggers the remainder path and compares every value with an ordered baseline. Include M=2/4/5/28/29, complete groups alone, serial fallback, threshold and cap behavior, odd K, small and non-multiple-of-four N, dirty output, shared read-only weights, and independent destinations. Verify completeness after workers join, deterministic output across worker counts, alias errors, destination retention, and valid retries. Run the upstream default, purego, and race suites and vet. On supported Darwin/arm64, verify the Metal matrix and tensor routes with the Metal toolchain.

```sh
go test -mod=readonly ./...
CGO_ENABLED=0 go test -mod=readonly -tags=purego ./...
go test -mod=readonly -race ./...
go vet -mod=readonly ./...
go test -mod=readonly -tags=metal ./tensor ./matrix
```

## Performance Acceptance Criteria

On paired real Llama measurements at P=8, target at least 15% lower median elapsed prefill at 28 tokens and 25% at 29 tokens, with improvement in at least four of five pairs at each size. Use the fresh runs as context and collect a new same-session baseline alongside the candidate. Verify lower elapsed time and less scalar remainder work for the FFN shapes. An improvement should persist with rotating weights and through public GEMM.

At 27 tokens, primary TTFT and cached generation, sustained decode, and the two-session guard, investigate any repeatable median regression above 5%. Include a P=1 guard and investigate a repeatable small-shape penalty. Increase repeats when ranges or paired directions leave the result inconclusive. Keep new allocation traffic below 1% of the corresponding prefill B/op, with bounded coordination objects and no material repeatable RSS increase. Gate acceptance on the Llama timings and correctness checks.

## Commit Discipline

Use small descriptive commits with one purpose each. Separate independent benchmark or test infrastructure from the implementation when each can be reviewed alone. Put regression tests with the change when they are needed to review its behavior together. Keep a single versionable candidate containing all required code and tests.

## Completion Criteria

An immutable candidate has the targeted remainder change, upstream correctness and concurrency coverage, paired raw and public results, and a report showing it meets the acceptance criteria. Supply evidence for every regression guard and identify any outstanding measurements so Llama can verify them before integration.

## Return Contract

Return the exact commit hashes and purpose of each, the published tag and module checksum or immutable pseudo-version, toolchain and build settings, test commands and outcomes, bitwise and nonfinite validation, raw benchmark observations with paired summaries, and CPU/allocation profiles. Include the dependency integration command and any relevant backend or configuration details. State which Llama acceptance runs were performed and supply their raw results.

## Llama Validation

Use Llama `5fb56f5bd22ec0f8157b3664fd2ef0d914dbd785` for both builds and verified fixtures from `testdata/README.md`. Compare `v0.10.16` to the immutable upstream candidate by changing only the dependency in temporary Go module files. Run commands from the Llama source root so the tests resolve `testdata`.

```sh
export GOTOOLCHAIN=go1.27.3 GOARM64=v8.0 GOMAXPROCS=8 CGO_ENABLED=1 GOFLAGS= GOWORK=off
export LLAMA_32_1B_MODEL_DIR=/path/to/verified/4e20de362430cd3b72f300e6b0f18e50e7166e08
export LLAMA_STORIES260K_MODEL=/path/to/stories260K.bin
export LLAMA_STORIES260K_TOKENIZER=/path/to/tok512.bin
out=$(mktemp -d "${TMPDIR:-/tmp}/llama-pass7-compare.XXXXXXXX")
for build in before after; do
    cp go.mod "$out/$build.mod"
    cp go.sum "$out/$build.sum"
done
candidate=SET_VERIFIED_TAG_OR_PSEUDOVERSION
go get -modfile="$out/after.mod" "github.com/itsmontoya/neuralnetwork@$candidate"
for build in before after; do
    go test -mod=readonly -modfile="$out/$build.mod" -c -o "$out/$build.test" .
    go version -m "$out/$build.test"
    go test -mod=readonly -modfile="$out/$build.mod" -count=1 -p 1 ./...
    CGO_ENABLED=0 go test -mod=readonly -modfile="$out/$build.mod" -tags=purego -count=1 -p 1 ./...
    env -u LLAMA_32_1B_MODEL_DIR go test -mod=readonly -modfile="$out/$build.mod" -race -count=1 ./...
    go vet -mod=readonly -modfile="$out/$build.mod" ./...
done
```

Check `TestLlama32Logits` (Transformers, `atol=rtol=1e-3`), `TestLlama32CachedParity` (complete logits, `atol=rtol=1e-4`), and `cmd/generate:TestLlama32Generation` (frozen continuation and byte-identical cached/full-prefix stdout). Check TinyStories logits and output, synthetic MHA/GQA/MQA traces, retained outputs, failed-step retries, and shared-decoder sessions. Run the default and purego fixture suites with the pinned Meta fixture. The race invocation above covers local fixtures and TinyStories without the large opt-in Meta fixture.

Require the following SHA-256 hashes of little-endian FP32 logit bits. The diagnostics already produced these on the fresh pass.

| Scope | SHA-256 | Greedy winner where reported |
| --- | --- | ---: |
| 27-token prefill | `0b087b0839da36ec06a3a8f777d4c3c944ccc7439107fcad39b6ad0239ed9afc` | 9822 |
| 28-token prefill | `cc93319c3c0701fd9cd3fdc2080afe43d2f82ba7cd236a698d19518836854e8c` | 374 |
| 29-token prefill | `6ffb4dc93003480aebdd51e9f1fee47ae3836eea9fb35ce2508f5d618a9871a5` | 12366 |
| Sustained terminal decode row | `b72d983e8b5c45f83ad858b52f1d27b87a7ad8118be6c5d315db2ea6fc6fd84a` | 1917 |
| Two concurrent six-token prefills | `f2e070e41d1eac82f0159284cfdd0d070aa06ef703c833fa6fb781a5b7b487c1` | — |

Alternate baseline/candidate order within pairs. Collect 12 primary pairs, seven sustained decode pairs, five pairs at each of 27/28/29 tokens, and at least three concurrent pairs. For each process call the relevant binary with `-test.run '^$' -test.bench '<exact benchmark regexp>' -test.benchtime 1x -test.benchmem -test.timeout 15m -test.v`. Use Python `os.posix_spawn`, monotonic `time.perf_counter`, and `os.wait4` for process wall and Darwin peak RSS in bytes. Use the tracked Llama benchmarks `BenchmarkLlama32CachedGeneration`, `BenchmarkLlama32SustainedDecode`, and `BenchmarkLlama32SupplementalPrefill` for the primary, sustained, and 29-token workload. Use the archived Llama-only diagnostic overlay for the 27/28 and concurrent cases. The overlays call the candidate's normal public GEMM entry point.

Collect CPU and heap profiles for both builds in independent processes. Use six primary iterations, two sustained iterations, and three iterations per supplemental length. Use `-test.cpuprofile` with `BenchmarkLlama32Pass7Phases` and the profiling overlay to label prefill, greedy selection, loading, and both decode workloads. Use `-test.memprofile` and `-test.memprofilerate 65536` in separate processes for allocation stacks. Compare exclusive CPU for groups, remainders, standalone single-row blocks, and classifier, along with stage elapsed time. Report per-build medians and ranges, median paired differences and percentages from individual pairs, and the number of pairs that improved. Report inconclusive effects explicitly where variance prevents attribution.

### Fresh baseline validation

The pinned baseline passed the default and purego suites with the Meta and TinyStories fixtures, the race suite with TinyStories and local fixtures, and `go vet`. The largest Transformers tolerance fraction was 0.0316354 in each fixture build. Cached and full-prefix logits matched exactly at the checked positions. Both CLI paths matched the frozen continuation bytes in the fixture suites.

### Raw evidence and replay

The host archive `/private/tmp/llama-pass7-evidence.Fzf5laYk.tar.gz` contains per-trial logs, CPU and allocation profiles, symbolized output, ARM64 disassembly, and the exact diagnostic sources and scripts. The extracted directory contains `env.sh`, `measure.py`, `baseline.sh`, `profile.sh`, `analyze.sh`, and `validate.sh`. From Llama's root, source `env.sh` after updating fixture paths if needed, run both overlay preparation scripts with the repo root, and build the binaries as follows.

```sh
evidence=/path/to/extracted/evidence
. "$evidence/env.sh"
python3 "$evidence/prepare-overlays.py" "$PWD"
python3 "$evidence/prepare-pass7.py" "$PWD"
go test -mod=readonly -c -o "$evidence/llama.test" .
go test -mod=readonly -overlay "$evidence/overlay.json" -c -o "$evidence/diagnostic.test" .
go test -mod=readonly -overlay "$evidence/wall-overlay.json" -c -o "$evidence/timing.test" .
go test -mod=readonly -overlay "$evidence/pass7-overlay.json" -c -o "$evidence/phases.test" .
sh "$evidence/baseline.sh" "$evidence"
sh "$evidence/profile.sh" "$evidence"
sh "$evidence/analyze.sh" "$evidence"
sh "$evidence/validate.sh" "$evidence"
```

Add the relevant `-modfile` to each build when measuring the candidate. See `summary.txt` in the archive for extracted observations. Pass 6 reproduction and evidence remain in Llama history at `422bfa41277d9135c5badcd803320de8f43585fe:NEURALNETWORK_OPTIMIZATION_HANDOFF.md`.
