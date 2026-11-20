# Neuralnetwork handoff for Llama Optimization Pass 6

## Objective

Reduce cached Llama 3.2 1B time to first token by accelerating evaluation of complete three-row groups in the host FP32 right-transposed matrix product. Start by testing bounded output-column parallelism for sufficiently large products. Deliver one optimization upstream. Llama is paused pending the result.

## Baseline and environment

Llama `cd74e6d48064f4eec20549a0fccd5c64d4fed14d` pins neuralnetwork `v0.10.15`, which resolves to `59e0dd572621490bf5b9cd374c47f6595235bce7`. `go mod verify` passed. The freshly compiled test binary reported Go 1.27.3 and module checksum `h1:PV7ZmnGCA9uGV43XoMEoZ9NBc4gbFvexngFH6nmdNJA=`.

The host was an Apple M3 on AC power, macOS 26.5.2 (25F84), darwin/arm64, `GOARM64=v8.0`, `GOMAXPROCS=8`, `CGO_ENABLED=1`, default tags, and no workspace overrides. The Meta snapshot revision was `4e20de362430cd3b72f300e6b0f18e50e7166e08`. The benchmarks verify all six fixture files against Llama's manifest on every process. Each unprofiled observation ran a single timed operation in a fresh process with zero inference warm-ups. The process wall clock and Darwin peak RSS include verified snapshot loading and setup. TTFT starts at prefill and ends at the first greedy selection. Session creation has its own timer.

### Six-token primary benchmark

`BenchmarkLlama32CachedGeneration` uses `The capital of France is`, context cap 512, cache capacity 13, one prefill and seven decode steps to generate eight tokens. It verifies the frozen IDs and decoded bytes. Output rate is eight tokens divided by the timed generation operation. Decode rate uses seven steps.

| Metric | Median | Range, seven runs |
| --- | ---: | ---: |
| Prefill (s) | 1.0020 | 0.6149–1.3180 |
| TTFT (s) | 1.0020 | 0.6150–1.3180 |
| Decode (s/token) | 0.08234 | 0.07679–0.13749 |
| Decode (tokens/s) | 12.14 | 7.27–13.02 |
| Cached generation (s/op) | 1.5672 | 1.1526–2.2413 |
| Output (tokens/s) | 5.105 | 3.569–6.941 |
| Bytes/op | 131,221,784 | 131,213,192–131,223,888 |
| Allocs/op | 38,990 | 38,972–38,998 |
| Process wall (s) | 5.2443 | 4.1414–5.8853 |
| Peak RSS (GiB) | 6.3345 | 6.0800–6.3737 |

In run order, TTFT was 1.0020, 1.2000, 0.6378, 1.2790, 0.6282, 1.3180, and 0.6150 s. Decode was 0.08073, 0.08961, 0.07786, 0.13749, 0.08234, 0.10769, and 0.07679 s/token. Cached generation was 1.5672, 1.8270, 1.1829, 2.2413, 1.2047, 2.0716, and 1.1526 s/op. Report paired distributions on return. The spread in the fresh primary data makes isolated best runs or a comparison to historical medians inadequate.

### Sustained decode and 29-token prefill

`BenchmarkLlama32SustainedDecode` times 32 calls to `Session.Decode` and greedy selection at positions 6–37. Its timer excludes snapshot verification, model load, tokenizer and session creation, and prefill. Eight-step windows ranged from 0.613 to 0.649 s. `BenchmarkLlama32SupplementalPrefill` times only prefill and greedy selection for `France is a country in Western Europe. It shares borders with Belgium, Germany, Switzerland, Italy, and Spain. The capital of France is`.

| Metric | Sustained median (range, three runs) | 29-token median (range, three runs) |
| --- | ---: | ---: |
| s/op | 2.5115 (2.4806–2.5418) | 5.1383 (5.0225–5.2879) |
| Decode s/token | 0.07848 (0.07752–0.07943) | — |
| Decode tokens/s | 12.74 (12.59–12.90) | — |
| Bytes/op | 329,900,064 (329,891,744–329,903,696) | 282,316,136 (282,316,024–282,321,520) |
| Allocs/op | 149,742 (149,725–149,749) | 5,975 (5,974–5,983) |
| Process wall (s) | 6.2326 (6.1804–6.2496) | 8.2413 (8.1999–8.4481) |
| Peak RSS (GiB) | 6.3764 (6.3762–6.3765) | 6.3562 (6.2972–6.3710) |

The 29-token winner was `12366` and its little-endian FP32 logit SHA-256 was `6ffb4dc93003480aebdd51e9f1fee47ae3836eea9fb35ce2508f5d618a9871a5`. The sustained digest was `b72d983e8b5c45f83ad858b52f1d27b87a7ad8118be6c5d315db2ea6fc6fd84a`, with the same IDs in every repetition, starting with the prefill winner.

```text
12366 13 1102 374 279 1455 95551 3363 304 9822 323 279 3224 753 6864 323 7928 3363 13 12366 374 279 1455 95551 16036 3158 304 279 7665 9323 323 279 1917
```

## Problem and selection evidence

Fresh CPU profiles used runtime phase labels around `Session.Prefill` plus greedy selection and around the decode loop. Goroutines created within those calls inherit the label, so the aggregate includes the single-row GEMM workers. The table uses self samples at the named kernel or the source lines of the scalar row loop, divided by the sampled CPU attributed to the phase. Total projection self samples are the sum of those disjoint entries. Inclusive time, including scheduler samples below a kernel, can differ. Parallel worker CPU is summed across threads. Separate Llama-side wall measurements used a timing-only test overlay on the `project` helper.

| Profile scope | Phase CPU | Three-row groups | One scalar row | Two scalar rows | Single-row kernel | All projection kernels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary prefill, 6 iterations | 3.44 s | 2.64 s (76.7%) | — | — | 0.38 s (11.0%) | 3.02 s (87.8%) |
| Primary decode, 42 steps | 9.24 s | — | — | — | 8.31 s (89.9%) | 8.31 s (89.9%) |
| Sustained decode, 64 steps | 14.40 s | — | — | — | 13.16 s (91.4%) | 13.16 s (91.4%) |
| 27-token prefill, 3 iterations | 7.25 s | 5.83 s (80.4%) | — | — | 0.29 s (4.0%) | 6.12 s (84.4%) |
| 28-token prefill, 3 iterations | 10.12 s | 5.99 s (59.2%) | 2.27 s (22.4%) | — | 0.60 s (5.9%) | 8.86 s (87.5%) |
| 29-token prefill, 3 iterations | 12.40 s | 6.08 s (49.0%) | — | 4.16 s (33.5%) | 0.57 s (4.6%) | 10.81 s (87.2%) |

The 27- and 28-token diagnostics are prefixes of the same encoded 29-token workload. All three run nine complete groups per block projection. Their elapsed time under profiling was 2.751, 3.961, and 4.868 s/op. Use the per-line samples and shape experiments for attribution because the three elapsed results come from separate processes. In the 28- and 29-token profiles, the scalar time is itemized on the fallback loop in `product.go:66–78`. The final classifier uses the tuned M=1 kernel in every cached prefill.

Across three timing-only primary trials, prefill wall was 0.619–0.676 s. The complete M=6 block projections occupied 0.584–0.632 s, including 0.482–0.523 s across gate, up and down combined. The M=1 classifier occupied 0.014–0.024 s. Across three separate 32-step trials, all decode projections took 95.6–95.7% of elapsed decode, leaving 0.111–0.149 s total for the rest of the phase. The decode projection path is already parallel. At 0.07848 s/token, one logical visit to 4.94 GB of weights per token corresponds to about 63.0 GB/s of weight payload. This calculation describes logical work per second. Labeled sustained decode also sampled 1.17 s (8.1%) in `runtime.pthread_cond_signal`. The stack-filtered caller profile sampled 0.060 s under `attendCached` and 0.170 s of memory zeroing across 64 steps. Caller-stack totals exclude parallel workers. The 38-position maximum in this workload bounds the conclusion about attention and KV access.

### Allocation and setup evidence

Unmodified benchmark binaries captured allocation profiles in separate processes at a 65,536-byte sampling rate. The sampled counts are estimates with considerable variance for small objects. The benchmark allocation counters above are the totals for individual timed runs. Profile sizes below use MiB. Inclusive stacks overlap.

| Scope | Sampled allocation | Attribution |
| --- | ---: | --- |
| Primary, whole process, 2 loads / 6 generations | 20,554.78 MiB | `LoadHFModel` 19,516.30 MiB (94.95%). BF16 conversion and the FP32 ownership copy dominate. Loader CPU was 3.18 s across the two processes. |
| Primary, 6 prefills | 328.87 MiB | `tensor.New` 208.51 MiB, GEMM staging 53.60 MiB, SwiGLU 108.33 MiB including 54.86 MiB in sigmoid staging. |
| Primary, 42 decode steps | 409.21 MiB | `tensor.New` 222.42 MiB, GEMM staging 83.05 MiB including worker overhead, SwiGLU 123.81 MiB, `Session.commit` 63.03 MiB including classifier and caller-owned logit copy. |
| Sustained, 64 decode steps | 635.18 MiB | `tensor.New` 343.05 MiB, GEMM staging 131.94 MiB including worker overhead, SwiGLU 190.36 MiB, `Session.commit` 95.53 MiB including classifier and caller-owned logit copy. |
| Supplemental, 3 prefills | 807.84 MiB | `tensor.New` 527.23 MiB, GEMM staging 127.23 MiB, SwiGLU 261.00 MiB including 130.50 MiB of sigmoid adapter allocations. |

Across the sustained thread sweep, the timed allocation count went from 126,624 at P=1 to 149,739 at P=8. The single-row worker dispatch accounted for an estimated 58,737 objects (23.91%) and 3.82 MiB (0.6%) in the sustained decode allocation profile. It creates call-owned worker closures and a wait group. This identifies a material source of the allocation-count tradeoff for Pass 5 parallelism. Remaining downstream attribution includes `Tensor.Float32Values` copies, `matrix.FromSlice`, and `Matrix.Values` under sigmoid. `Decoder.NewSession` retains bounded K/V capacity and reusable attention-score scratch per session. KV payload is 851,968 bytes for the primary workload and 2,490,368 bytes for sustained decode. Model loading retains 4,943,257,600 bytes of weights, including one shared embedding/classifier tensor. RSS is a process high-water mark. In the labeled 29-token prefill, memory zeroing sampled 0.63 s (5.1%) against 10.81 s of projection kernel self time. GC marking was below the displayed threshold in the phase profiles. Zeroing and copying were visible, with a smaller elapsed opportunity than projection. The wall timings bound independent Llama work. Preserve storage semantics and treat staging, sigmoid, loader ownership, and allocation count as separate candidates.

### Representative shapes and experiments

The matrices are left `[M,K]`, stored weights `[N,K]`, and output `[M,N]`. The model has 16 blocks, 32 query heads, 8 KV heads, head width 64, hidden width 2048, FFN width 8192, vocabulary 128256, and tied embeddings.

| Site | K | N | FP32 weights per call | Calls per phase iteration or decode step |
| --- | ---: | ---: | ---: | ---: |
| Attention K and V | 2048 | 512 | 4 MiB | 32 |
| Attention Q and O | 2048 | 2048 | 16 MiB | 32 |
| FFN gate and up | 2048 | 8192 | 64 MiB | 32 |
| FFN down | 8192 | 2048 | 64 MiB | 16 |
| Tied classifier | 2048 | 128256 | 1002 MiB | 1 |

M is six or 29 for the cached block projections and one for the cached classifier. The full-prefix `Forward` path applies the classifier to every row. Include its multi-row case as a regression and performance guard.

The following is the existing public `GemmInto` with preallocated operands/destination and the real checkpoint weights, over three fresh processes with 125 ms minimum per subbenchmark. Reuse repeatedly applies one weight tensor. Rotate cycles through the model's disjoint block weights per shape, covering 128 MiB, 512 MiB, 2 GiB and 1 GiB, respectively. The classifier has one weight tensor. Times are median ms/op and parenthesized ranges. M4−M3 and M5−M3 are paired differences within each process. They estimate the additional cost of one or two scalar rows, including the incremental staging and output work. M2 directly exercises two scalar rows without a group. Standalone M1 follows the tuned single-row dispatch.

| Site | M3 reuse | M3 rotate | M6 rotate | M4−M3 rotate | M5−M3 rotate | M2 rotate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K/V | 0.283 (0.282–0.285) | 0.317 (0.313–0.318) | 0.597 (0.596–0.659) | 0.991 (0.987–0.992) | 1.974 (1.972–2.166) | 1.976 (1.974–1.985) |
| Q/O | 1.151 (1.144–1.173) | 1.260 (1.256–1.265) | 2.498 (2.476–2.560) | 3.953 (3.952–4.008) | 7.935 (7.901–7.990) | 7.970 (7.891–8.037) |
| Gate/up | 4.931 (4.901–4.931) | 5.020 (5.019–5.035) | 10.036 (10.018–10.088) | 15.930 (15.889–15.930) | 31.635 (31.580–31.640) | 31.614 (31.585–31.647) |
| Down | 4.563 (4.557–4.628) | 4.639 (4.629–4.747) | 9.244 (9.229–9.245) | 16.266 (16.223–16.440) | 32.503 (32.419–33.341) | 32.586 (32.505–32.678) |

The M=1 classifier reused its single tensor at 17.806 ms/op (14.222–21.495). Public M=6 products used three allocations and 12,384, 49,248, 196,704 and 49,248 B/op, respectively. M=1 rotating attention K/V and Q/O medians were 0.161 and 0.320 ms. Rotating M=1 FFN results varied from 1.006–3.232 ms for gate/up and 0.919–2.522 ms for down. Use the sustained end-to-end test to assess decode.

On Go 1.27.3 arm64, the sampled full-tile inner loop contains seven scalar FP32 loads per contraction index (four weights and three activations), twelve scalar `FMADDS` instructions on independent ordered accumulator chains, and increment, compare and branch. The compiler keeps all twelve accumulators in FP registers with no inner-loop spills. The twelve raw output writes occur after the contraction. At K=2048/8192, output stores and outer tile setup are infrequent. The rotating M=3 FFN tests were about 2% slower than reuse. One caller processed logical weight payload at about 13–14 GB/s on those shapes. Three trials of independent existing M=6 gate/up calls, rotating through real weights with one caller per scheduler thread and disjoint destinations, showed aggregate payload rates of 6.4–12.4 GB/s at P=1, 24.9–25.2 at P=2, 45.9–46.5 at P=4, and 67.3–68.2 at P=8. The P=1 trials were noisy. This experiment demonstrates headroom beyond one core in the current serial three-row loop. It measures independent products and logical weight visits, and gives no direct measure of one parallel call, DRAM transactions, or cache misses. The strongest diagnosis is serial inner-loop throughput, with load and fused arithmetic both material. Distinguishing load stalls from FP issue limits or cache effects requires hardware counters or controlled measurements. A wider tile needs its own evidence.

## Ownership and exact call path

The reusable implementation belongs in neuralnetwork. Llama builds shape-checked, private tensors and sends each projection through a public GEMM API. Neuralnetwork owns the row-major weight slices, the validation boundary, the host dispatch, and the arithmetic. This change can use the existing API.

1. Llama `generation.go:Decoder.GenerateGreedyCached` calls `session.go:Session.Prefill`. The primary benchmark exercises the same session path directly. Prefill calls `decoder.go:Decoder.forwardPrefix` and `Decoder.forwardBlock`. Q, K and V call `attention.go:headProjection` and then `project`. O, FFN gate, up and down call `project` directly. `Session.Prefill` gathers the last hidden row and calls `Session.commit` → `Decoder.projectLogits` → `project`. `weights.go:Decoder.loadWeights` ties the classifier to the embedding.
2. Llama `tensor_ops.go:project` calls `tensor.GemmShape`, allocates a disjoint FP32 tensor with `tensor.New`, and calls `tensor.GemmInto(input, weight, nil, out, tensor.GemmConfig{TransposeB: true, Alpha: 1})`. `Session.Decode` reaches the same projections with M=1. Uncached `Decoder.Forward` and `ForwardWithTrace` use the full-prefix path and project all final rows.
3. Neuralnetwork `tensor/gemm.go:GemmInto` validates the operands, shapes, optional bias, destination storage and aliasing. In default and purego builds, `tensor/matrix_product_host.go:matrixProductValues` allocates call-owned staging and calls `matrixProductInto` → `internal/float32product/product.go:RightTranspose`. The kernel writes the staged result. GEMM then applies Alpha and optional Beta/bias in its existing order and copies to the public destination. The platform-specific Metal configuration uses `tensor/matrix_product.go` → `matrix.Matrix.MatMulRightTransposeInto`, through a separate adapter.
4. Host `RightTranspose` handles empty contractions first. For M=1 and N≥4 it selects `rightTransposeOneRow` or `right_transpose_parallel.go:rightTransposeOneRowParallel`. For M≥3 it walks complete groups serially, invoking `rightTransposeThreeRows` once per group. It then computes each remaining row with the scalar loop local to `RightTranspose`. That loop also handles M=2 directly. The three-row kernel tiles four output columns and has its own scalar-column cleanup when N is not a multiple of four.

## Desired behavior and implementation direction

Parallelize sufficiently large complete-group work in a host right-transposed product while preserving all output bits and the public contract. Use the following hypothesis, then select a worker cap and threshold from data.

1. Test splitting contiguous right-hand rows and the corresponding output columns across a bounded number of workers. Align full ranges to the existing four-column tile. Each worker should cover its range across the complete groups in the call, with the original destination row stride and exclusive output columns for each row. Preserve the original handling of an incomplete output-column tile. Reuse the ordered tile arithmetic and existing worker conventions where they fit. Joining once per GEMM gives one coordination cost across the complete groups. Retain the serial path for small calls and P=1.
2. Estimate eligibility from weight footprint and number of complete groups. Sweep P=1, 2, 4 and 8, and quantify thresholds for 4, 16 and 64 MiB weights, groups at M=3/M=6/M=27, and the uncached classifier. The Pass 5 one-row implementation's 2 MiB-per-worker heuristic is a comparison point. Cost per worker and thread contention may differ for the three-row kernel.
3. Keep the existing one- and two-row scalar remainder handling and the Pass 5 M=1 kernel as independent paths. Track three-row CPU, projection wall and remainder behavior so a speedup on M=29 can be attributed to the intended groups, even when the remainder becomes a larger percentage. Preserve Metal dispatch.

### Alternatives considered

- **Retune the 3×4 tile or vectorize independent outputs.** The loop already reuses activations and weights with twelve independent accumulators and no inner-loop spills. Extra chains or different loads need controlled evidence for execution throughput and register pressure. Evaluate only if the bounded parallel approach fails its completion criteria. Preserve each output's ascending contraction and fused behavior.
- **Shard by complete three-row groups.** Workers would independently read the full right matrix. This may be simpler for contiguous writes, but shares parallelism across at most two groups on the primary workload and duplicates the concurrent weight streams. Compare against output-column sharding if the worker topology proves significant. Select a single approach.
- **Accelerate the scalar remainder.** It accounts for 33.5% of supplemental prefill CPU and is a separate opportunity. Six-row prefill has zero scalar remainder.
- **Continue improving M=1 decode.** It still dominates decode CPU. In a fresh one-trial-per-setting diagnostic, sustained decode delivered 2.96, 4.67, 9.94 and 12.52 tokens/s at P=1, 2, 4 and 8. Its logical payload rate is already close to the aggregate rotating FFN experiment, and the remaining memory, scheduling and per-core headroom needs separate evidence. The serial complete-group path has demonstrated multicore headroom and directly affects TTFT.
- **GEMM staging, sigmoid or checkpoint loading.** These are visible in allocations and peak-memory behavior, with independent ownership and lifetime concerns. Improving them addresses a different part of the workload.

## Numerical, allocation and ownership contracts

- Preserve ascending K, initial FP32 zero, and the same effective FP32 rounding and fusion in each build. Go 1.27.3 arm64 emits `FMADDS`. Require bitwise equality to the v0.10.15 path for finite outputs within the same build and architecture, including signed zero. Cover NaN and Inf behavior under existing test expectations. Preserve empty contractions, output tails, Alpha, Beta, bias order and all existing tolerances. Use disjoint output partitions of complete dot products. Splitting K changes the accumulation order.
- Retain public GEMM shape and dtype validation, input/destination/bias disjointness, and destination immutability on returned errors. The raw kernel overwrites dirty output, and inputs remain private and read-only. Keep staging and coordination call-owned, join all work before returning, and release references at completion. Preserve independently retained session logits, trace tensors, and retry semantics. A single `Session` is caller-synchronized, and many sessions and forwards can share one decoder.
- The existing serial raw three-row kernel allocates zero bytes. The public path allocates its current staging and shape objects. Any parallel coordination allocations should be bounded by participating workers per call and measured separately. Avoid per-row-group allocations, weight duplication, shared mutable scratch and global kernel state. Document measured B/op, allocs/op, RSS and concurrent caller impact, including a one-thread and small-work guard.

## Neuralnetwork microbenchmarks and tests

Build the baseline from `v0.10.15` and the candidate from its implementation commit, in separate neuralnetwork worktrees if helpful. Run the existing `tensor/right_transpose_benchmark_test.go:BenchmarkRightTranspose` under the pinned Go environment. Record build metadata, repeat each measurement in fresh processes, and alternate before/after order.

```sh
export GOTOOLCHAIN=go1.27.3 GOARM64=v8.0 GOMAXPROCS=8 CGO_ENABLED=1 GOFLAGS= GOWORK=off
nn_out=$(mktemp -d "${TMPDIR:-/tmp}/neuralnetwork-pass6.XXXXXXXX")
go test -mod=readonly -c -o "$nn_out/nn.test" ./tensor
go version -m "$nn_out/nn.test"
"$nn_out/nn.test" -test.run '^$' -test.bench '^BenchmarkRightTranspose$' -test.benchtime 200ms -test.benchmem
```

Extend raw `float32product.RightTranspose` and public `tensor.GemmInto` benchmarks using the K/N pairs above. Measure M=1–6 and M=27, 28, 29 and 30, with focused subsets for the large vocabulary costs. Include M=3 and M=6 for all shapes. Record the remainder-only M=2 and paired M4−M3 and M5−M3 differences, alongside full projection time. Use repeated and rotating weight operands, with disjoint sets larger than cache and a documented logical-byte definition. Allocate operands and destinations outside the timed loop. Test divisibility, dimensions near the worker threshold, N=0–5 and N=513 or N=8193, and short/odd K. Sweep P and worker caps. Repeat with parallel callers sharing weights and disjoint destinations, and profile scheduling overhead and allocation changes. Compare raw versus public cost and report the range over at least three pairs.

Keep `internal/float32product` tests `TestRightTranspose`, `TestRightTransposeConcurrent`, `TestRightTransposeNonfiniteAndSignedZero`, and `TestEmptyContraction` green. Preserve public tensor product reference coverage and `Test_Products_AliasesAndEmptyContractions`, `Test_Gemm_BiasValidation`, `Test_Gemm_RightTranspose`, and `Test_Gemm_RightTransposeSingleRowConcurrentRetry`. Add regression coverage exercising the actual parallel path against an ordered serial reference with K=2048 and 8192, irregular split and output-column tails, odd K, mixed group/remainder cases, cancellation, dirty destinations, shared immutable weights and concurrent exclusive destinations. Include fallback at P=1. Verify alias errors, independently retained results and clean retries in the public API. Run:

```sh
go test -mod=readonly ./...
CGO_ENABLED=0 go test -mod=readonly -tags=purego ./...
go test -mod=readonly -race ./...
go vet -mod=readonly ./...
# On Darwin/arm64 with the Metal toolchain:
go test -mod=readonly -tags=metal ./tensor ./matrix
```

## Llama validation

Use Llama `cd74e6d` for both builds. Download and verify the pinned Meta snapshot and TinyStories model/tokenizer using `testdata/README.md`. The frozen Transformers comparisons use `atol=rtol=1e-3`, and cached versus full-prefix uses `atol=rtol=1e-4` over the complete logits. Keep greedy winners, reference continuation IDs, and decoded bytes identical. Preserve the supplemental and sustained FP32 hashes in the same build and architecture. The benchmark reproduction for Pass 5 remains available at `git show cd74e6d:NEURALNETWORK_OPTIMIZATION_HANDOFF.md`.

From the Llama root, use temporary module files containing the same Llama baseline and only the verified neuralnetwork candidate dependency change. Confirm both build records contain Go 1.27.3, the pinned expected dependency version and checksum, and the same tags and platform. Use an immutable candidate tag or Go pseudo-version. Alternate order inside pairs and keep the model snapshot, hardware, timer and profile settings fixed.

```sh
export GOTOOLCHAIN=go1.27.3 GOARM64=v8.0 GOMAXPROCS=8 CGO_ENABLED=1 GOFLAGS= GOWORK=off
export LLAMA_32_1B_MODEL_DIR=/path/to/verified/4e20de362430cd3b72f300e6b0f18e50e7166e08
export LLAMA_STORIES260K_MODEL=/path/to/stories260K.bin
export LLAMA_STORIES260K_TOKENIZER=/path/to/tok512.bin
out=$(mktemp -d "${TMPDIR:-/tmp}/llama-pass6.XXXXXXXX")
cp go.mod "$out/before.mod"
cp go.sum "$out/before.sum"
cp go.mod "$out/after.mod"
cp go.sum "$out/after.sum"
candidate=SET_VERIFIED_TAG_OR_PSEUDOVERSION
go get -modfile="$out/after.mod" "github.com/itsmontoya/neuralnetwork@$candidate"
for build in before after; do
  go test -mod=readonly -modfile="$out/$build.mod" -c -o "$out/$build.test" .
  go version -m "$out/$build.test" > "$out/$build.build.txt"
done
```

Run seven pairs for `CachedGeneration` and at least three pairs each for `SupplementalPrefill` and `SustainedDecode`. Run from the Llama source directory to resolve `testdata`. For Darwin, the wrapper below records process wall time and peak RSS bytes using the same monotonic clock and one-process `wait4` method. Elsewhere, document OS-specific RSS units and record them consistently.

```python
# Save as $out/measure.py
import os
import sys
import time

path, *command = sys.argv[1:]
with open(path, "wb", buffering=0) as log:
    start = time.perf_counter()
    pid = os.posix_spawn(command[0], command, os.environ, file_actions=[
        (os.POSIX_SPAWN_DUP2, log.fileno(), 1),
        (os.POSIX_SPAWN_DUP2, log.fileno(), 2),
    ])
    _, status, usage = os.wait4(pid, 0)
    log.write(f"process_wall_s={time.perf_counter() - start:.6f} peak_rss_bytes={usage.ru_maxrss}\n".encode())
sys.exit(os.waitstatus_to_exitcode(status))
```

```sh
set -e
for workload in CachedGeneration SupplementalPrefill SustainedDecode; do
  trials=3
  if [ "$workload" = CachedGeneration ]; then trials=7; fi
  for trial in $(seq "$trials"); do
    order='before after'
    if [ "$((trial % 2))" -eq 0 ]; then order='after before'; fi
    for build in $order; do
      python3 "$out/measure.py" "$out/$workload-$build-$trial.log" "$out/$build.test" \
        -test.run '^$' -test.bench "^BenchmarkLlama32${workload}$" \
        -test.benchtime 1x -test.benchmem -test.timeout 15m -test.v
    done
  done
done
```

Profile both builds in separate, matched processes. Use primary 3x in two processes, supplemental 3x, and sustained 2x for unmodified benchmark CPU and allocation profiles. Collect with `-test.cpuprofile`, `-test.memprofile` and `-test.memprofilerate 65536`. Use `go tool pprof -alloc_space` and `-alloc_objects` with `-focus='\(\*Session\)\.Prefill'` or `Decode` for allocation attribution. Compare `LoadHFModel` independently of inference. Time repeated `project` calls inside the actual Llama run by shape to measure elapsed kernel plus staging/dispatch cost. For CPU attribution, tag the phase around `Session.Prefill` plus greedy selection, and separately around the decode loop with `pprof.Do(ctx, pprof.Labels("phase", "prefill"), fn)` or `"decode"`. Include spawned workers. Run six primary iterations, two 32-step sustained iterations, and three iterations per 27/28/29 row case. `Decoder.NewSession` is outside the labels. For the 27/28 diagnostics, tokenize the exact supplemental prompt and pass its first 27/28 IDs. Use `go tool pprof -top -relative_percentages -tagfocus=phase=prefill` and `-list='float32product\.RightTranspose$'` to account for the complete groups, separate remainder lines and M=1 classifier. On the candidate include newly attributed worker stacks and report both sampled CPU and elapsed time. Parallel execution can increase sampled CPU while decreasing wall time.

Run full fixture suites with both module files and real fixtures. This covers `TestLlama32Logits`, `TestLlama32CachedParity`, `cmd/generate:TestLlama32Generation` for byte-identical cached and uncached CLI stdout, `TestStories260KLogits`, `cmd/generate:TestStories260KGeneration`, synthetic MHA/GQA/MQA intermediate traces, retained logits and traces, failed-step retries, and concurrent decoders and sessions. The race suite can exclude the large opt-in Meta fixture while covering local models and TinyStories. Confirm both default and purego.

```sh
for build in before after; do
  go test -mod=readonly -modfile="$out/$build.mod" -count=1 -p 1 ./...
  CGO_ENABLED=0 go test -mod=readonly -modfile="$out/$build.mod" -tags=purego -count=1 -p 1 ./...
  env -u LLAMA_32_1B_MODEL_DIR go test -mod=readonly -modfile="$out/$build.mod" -race -count=1 ./...
  go vet -mod=readonly -modfile="$out/$build.mod" ./...
done
```

### Profile evidence and replay files

The fresh unmodified benchmark logs, raw pprof files, labeled diagnostic source, timing probe, disassembly and summaries are in `/private/tmp/llama-pass6-evidence.UMQziOl7.tar.gz` on the measurement host. The standard Llama benchmarks and the commands above reproduce primary, sustained and supplemental measurements independently of the archive. For the diagnostic archive, extract to a temporary directory, supply its `env.sh` with the fixture exports above, and run `python3 "$out/prepare-overlays.py" "$PWD"` from the Llama root. Build `$out/llama.test` with `go test -mod=readonly -c -o "$out/llama.test" .` before running the profile script. In the archive, `profile.sh` and `analyze.sh` take the extracted directory as their argument. Build `wall.test` with `go test -mod=readonly -overlay "$out/wall-overlay.json" -c -o "$out/wall.test" .`. Run `BenchmarkLlama32Pass6ProjectionWall/Primary`, `/Sustained` and `/Prefill29` at 1x with that binary. Run `BenchmarkLlama32Pass6Projection` with `diagnostic.test` at 125 ms and `BenchmarkLlama32Pass6IndependentFFN` with `wall.test` at 400 ms. The phase, shape and independent caller experiments were implemented as Llama-only test overlays and call the existing neuralnetwork implementation.

Fresh baseline validation passed on this host with the pinned Meta and TinyStories fixtures in the default and purego suites. The race suite passed with TinyStories and local tests, with the opt-in Meta fixture disabled. `go vet -mod=readonly ./...` passed.

## Commit discipline and completion criteria

- Produce small, concise, logical commits. Group the performance implementation with its atomic regression coverage. Separate meaningful benchmark infrastructure or measured documentation when useful. Inspect the final diff and history.
- Show a repeatable improvement on complete three-row groups and on Llama's paired six-token prefill/TTFT, with ranges and raw observations. Report the effect on overall generation, throughput, bytes/op, allocs/op, whole-process wall and peak RSS. Use the 27/28/29 row series and source profiles to explain the supplemental effect. Track sustained decode and concurrent callers as guards. Report inconclusive effects as such and collect more same-method pairs when needed.
- Preserve the tested FP32 results, public error/alias semantics, returned-output lifetime and concurrent use. Keep small shapes, purego, and Metal behavior covered. Show that the targeted elapsed projection work decreased on Llama, with CPU sample counts and percentages reported alongside wall time. A microbenchmark gain alone leaves end-to-end validation outstanding.

## Return contract and Llama integration

Return the upstream commit(s) and their purposes, immutable version or exact commit for the Llama pin, candidate dependency checksum, correctness results, commands, paired raw and summarized benchmark results, profile changes and artifact locations, and any known tradeoffs. Include the threshold/cap rationale and integration instructions.

The Llama agent will inspect the upstream change against this handoff, verify the returned version and module checksum, update the pin in a focused commit, and rerun the complete Llama validation above on the same hardware. Pass 6 concludes after verifying the targeted bottleneck decreased in the real workload and documenting the paired end-to-end effects. Future candidates will be ranked from the post-integration evidence.
