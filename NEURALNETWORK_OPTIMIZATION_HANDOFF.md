# Neuralnetwork handoff for Llama Optimization Pass 5

## Objective

Reduce real Llama 3.2 1B decode latency by accelerating the host FP32, single-row, right-transposed product in neuralnetwork. Investigate bounded parallelism across independent output columns as the first implementation direction. Keep this to one optimization. Llama is paused at the handoff with Go 1.27.3 and neuralnetwork v0.10.14. The Llama benchmark commit is `1e10d4b`, based on `b587204`. Neuralnetwork v0.10.14 resolves to `2a75a34561834915f773ba15464db5241fcd40c2`.

## Evidence

### CPU and projection sites

Two fresh primary profiles each covered one process with three generations. With the profiles merged and percentages normalized within the stage stack:

| Stage | Sampled CPU | Kernel CPU self | Kernel share |
| --- | ---: | ---: | ---: |
| Primary decode, 42 steps | 12.06 s | 9.43 s, single-row | 78.19% |
| Sustained decode, 64 steps | 18.36 s | 15.65 s, single-row | 85.24% |
| Primary prefill, six prompts | 3.61 s | 2.52 s, three-row | 69.81% |
| 29-token prefill, three prompts | 11.27 s | 5.58 s, three-row | 49.51% |
| 29-token prefill, same profile | 11.27 s | 4.11 s, two scalar fallback rows | 36.47% |

Of the 9.43 s of primary decode kernel self samples, FFN gate/up/down account for 2.29/2.16/1.56 s, respectively (63.7% combined). Vocabulary accounts for 2.40 s (25.5%). Q/K/V/O account for 0.41/0.09/0.11/0.41 s (10.8% combined). The classifier also uses the single-row kernel after cached prefill, accounting for 0.37 s of the six primary prefill calls. The multi-row kernel and scalar prefill tails remain separate opportunities.

### Allocation, zeroing, and copies

Heap data used 64 KiB sampling. These are estimates in MiB. Inclusive stacks overlap.

| Stack | Six primary generations | 64 sustained decode steps |
| --- | ---: | ---: |
| Inference allocations | 745.80 | 624.78 |
| `tensor.newTensor` allocations | 445.07 | 351.44 |
| Host GEMM result staging in `matrixProductValues` | 131.75 | 122.04 |
| `swiGLU` inclusive | 231.70 | 189.97 |
| Sigmoid matrix adapter within SwiGLU | 117.65 | 96.15 |
| All `Tensor.Float32Values` copies | 33.43 | 43.96 |
| Cached attention inclusive | 15.62 | 24.46 |

The primary SwiGLU stack includes ~38 MiB each for the allocated sigmoid, SiLU, and gated destinations, in addition to sigmoid adapter staging. The matrix adapter uses `matrix.FromSlice`, `matrix.New`, and `Matrix.Values`. The primary `Session.commit` copy accounts for 23.63 MiB of the `Float32Values` total and returns caller-owned logits. The session uses call-created KV storage and reuses its own `scores` buffer. `NewSession` accounts for 4.88 MiB across the six generations. The profile attributed 1.55 s of primary decode CPU self to zeroing, including 1.51 s under `tensor.New` and 0.01 s under GEMM result staging. All `runtime.memmove` samples under primary decode total 0.14 s. Under sustained decode, zeroing accounts for 0.95 s (5.17%) and `memmove` for 0.28 s (1.53%). The sampled cached-attention callback accounts for 0.19 s inclusive in sustained decode. Its context only reaches 38 positions. The separate 29-token profile attributes 808.38 MiB to prefill, including 126.76 MiB of GEMM staging and 130.50 MiB of sigmoid staging.

`LoadHFModel` accounts for 19,516.35 MiB across two primary profiling processes (94.9% of whole-process cumulative allocation). Most is BF16→FP32 conversion and the tensor ownership copy. The model retains 4,943,257,600 bytes of FP32 parameters, with one shared embedding/classifier tensor. Peak RSS below is a process high-water mark including checkpoint verification and loading. The exit heap snapshot was taken after benchmark state became unreachable, so it does not measure peak live model memory. The projection kernel itself allocates zero bytes. Treat loader ownership, destination reuse, staged GEMM, and sigmoid as separate changes.

### What limits single-row projection

The Go 1.27.3 arm64 disassembly shows five scalar FP32 loads and four dependent `FMADDS` updates per contraction index across four independent output accumulators, plus loop control. Fresh working-set measurements below show a modest reuse advantage on small matrices and only about 1–4% on the 64 MiB FFN and 1002 MiB classifier. Even the 32 KiB shape delivered about 16 GB/s of weight payload. Rotating large FFN weights delivered 14.76–15.52 GB/s, rotating vocabulary weights 14.12–14.24 GB/s, and the sequential-read control delivered 59.77–64.32 GB/s at those sizes. Our definition of effective bandwidth is logical weight bytes divided by elapsed time. It does not count hardware memory transactions, and the control has a different load stream. All large weights in a decode step account for 4,942,987,264 logical bytes (4,714 MiB), about 14.55 GB/s at the primary median. Raw output writes account for about 2.02 MB per step.

The existing raw kernel, run on eight independent 64 MiB projections, delivered 14.77–14.98 GB/s with one worker, 28.80–28.89 GB/s with two, 51.73–53.95 GB/s with four, and 68.93–73.23 GB/s with eight. These are aggregate rates for independent jobs with disjoint inputs/destinations except for a read-only left row. The probe does not predict single-projection scaling or Llama latency. The results rule out saturation of the whole host's memory bandwidth by one serial projection on this host. They support investigating per-core execution and load-stream limits, with much more bandwidth available across workers. They do not isolate accumulator latency, instruction issue, and per-thread memory-level parallelism. Hardware counter access was unavailable in this run. `xctrace` required full Xcode, DTrace CPC initialization was denied, and `powermetrics` required root. Use trustworthy counters if available on your host.

The near-linear width sweep, comparable 512- and 513-column times, and raw versus public timings show that repeated arithmetic and weight loads dominate dispatch, scalar output tails, and output writes at the real shapes. All real output widths are multiples of four. FFN accounts for the largest directly affected CPU share. Vocabulary is the largest single call and already streams from a roughly 1 GiB allocation on every step. Optimize and measure each of those cases explicitly.

## Ownership

Neuralnetwork owns the shared algorithm, CPU dispatch, and the private validated weight slices on this path. Llama supplies tensors through `tensor.GemmInto` and has no direct view into their FP32 storage. A host kernel change can benefit all matching callers with the existing API, allocation, and ownership behavior. Use the existing Llama integration and measure it against your candidate.

## Call path and relevant code

- Llama `generation.go:Decoder.GenerateGreedyCached` calls `Session.Prefill` and `Session.Decode` in `session.go`. `Decode` calls `Decoder.forwardBlock` for every layer, then `Session.commit` → `Decoder.projectLogits` in `decoder.go`. `Prefill` uses `Decoder.forwardPrefix`, gathers only the last prompt hidden row, and calls the same classifier through `commit`.
- In `decoder.go`, Q/K/V use `headProjection` in `attention.go`, which calls `project`. O, gate, up, down, and classifier call `project` directly. Llama's `tensor_ops.go:project` computes `tensor.GemmShape`, allocates the disjoint FP32 destination with `tensor.New`, and calls `tensor.GemmInto(input, weight, nil, out, GemmConfig{TransposeB:true, Alpha:1})`. `weights.go:loadWeights` sets `d.output = d.embedding` when tied.
- Neuralnetwork `tensor/gemm.go:GemmInto` validates types, shapes, bias, storage and destination aliasing. The default CPU and purego build resolve `tensor/matrix_product_host.go:matrixProductValues` → `matrixProductInto` → `internal/float32product/product.go:RightTranspose` → `rightTransposeOneRow`. Metal builds have a separate adapter in `tensor/matrix_product.go`. Keep its dispatch and error behavior intact.
- Relevant test and benchmark files are `internal/float32product/product_test.go`, `tensor/products_test.go`, and `tensor/right_transpose_benchmark_test.go`.

## Current behavior and desired behavior

`RightTranspose` uses `rightTransposeOneRow` for one validated nonempty row and at least four columns. The serial kernel consumes four adjacent weight rows at a time in ascending contraction order, reuses each left load across the four rows, then processes an output tail scalarly. The right operand is physically row-major `[N,K]`. The host computes into a call-owned staging slice, then `GemmInto` applies Alpha and optional bias in its existing order and copies the result to the caller's destination.

Reduce elapsed time for large one-row products while preserving validation, output bits and storage ownership. Keep the serial path for small work and a stable serial fallback, including when only one scheduler thread is available. Keep the changes limited to the targeted single-row path. Both the FFN projections and the tied classifier need to be covered in the measurements.

## Proposed direction

1. Test splitting sufficiently large single-row products into contiguous ranges of output columns. Each worker reads the same left row, reads its own contiguous right-row range, and overwrites only its exclusive output range. Reuse the existing ordered serial kernel within a range. Prefer splits on the existing four-column tile boundary to preserve main-loop and tail behavior. Join every worker before returning.
2. Sweep bounded worker counts and an empirically chosen minimum work or weight footprint per worker. Test 1, 2, 4, and 8 workers on this host, including `GOMAXPROCS=1` and higher settings, small gemv calls, 4 MiB K/V, 16 MiB Q/O, 64 MiB FFN, and 1002 MiB vocabulary. Profile goroutine/scheduler costs, per-call allocation overhead and contention with concurrent projections and Llama sessions. The independent-job probe justifies testing this direction. Validate single-call sharding before carrying it into the candidate.
3. Choose a threshold and cap that survive the real Llama A/B and concurrent workload. Preserve the existing public API. Keep coordination bounded and call-owned. A solution that wins only on the isolated classifier or by using unbounded workers fails the completion criteria.

## Alternatives

- **Ordered serial kernel tuning.** Measure instruction issue, downstream load stalls, and accumulator dependencies before changing tile width or adding accumulators. Vectorizing independent outputs can preserve contraction order if loads and FP32 fusion also match the target build. Row-major weight layout complicates cross-output loads, and packing a second copy is expensive for the tied vocabulary. Consider a serial alternative only if bounded parallel dispatch is a poor fit, and land one approach.
- **Split along K or use a horizontal SIMD reduction.** These change the reduction order. They are incompatible with the bitwise contract unless the effective order and FP32 fusion are preserved.
- **GEMM staging or SwiGLU.** These have substantial allocation traffic. The timed public-versus-raw benchmarks and copy/zeroing stacks indicate a smaller direct opportunity for projection latency. Keep their API and memory behavior as-is for this change.
- **Multi-row prefill tail work.** The 29-token profile gives direct evidence for this separate candidate. A tail optimization primarily helps prompts with incomplete three-row tiles. The primary decode proof takes precedence for this pass.
- **Permanent weight packing, a second vocabulary layout, or a Llama-side backend.** These introduce weight ownership, peak RSS and backend concerns beyond this pass. Keep the row-major inputs and current host API.

## Numerical contract

Maintain ascending K for every output, the same zero initialization, and the effective float32 multiply/add fusion for the same compiler, architecture, and build configuration. Go 1.27.3 emitted fused `FMADDS` for the current arm64 arithmetic. Keep bitwise equality to the existing kernel on finite results within each supported build. Preserve signed zero, NaN classification and Inf behavior as covered by the existing tests. Preserve empty contractions, output and inner tails, Alpha scaling, and optional Beta/bias behavior. Run the same numerical checks with the serial and parallel dispatch paths. Keep existing reference tolerances.

## Ownership and lifetime contract

Inputs are private/read-only through the operation. Retain the public destination validation, disjointness, and all-or-nothing behavior on returned errors. The raw kernel overwrites the supplied range even if it was previously dirty. A failed/error call leaves the existing destination usable and retries must produce correct results. Keep staging and coordination within each call, complete every write before returning, and release all input, output, and scratch references when work completes. Preserve caller-owned logits, retained trace tensors, and existing Llama retry semantics. Use no full-size new weight layout or extra copy of input payloads.

## Concurrency contract

Independent calls may share read-only inputs and must be safe with distinct destinations. Each worker owns only its output range, and all workers finish before any public result is committed. Bound scheduling work per call and test concurrent callers under oversubscription. Preserve the dependency's race-free default and purego paths and the shared Llama decoder / independent session model. Keep allocation and RSS impact bounded even when several calls run at once.

## Representative Llama shapes

For a single row, left is `[1,K]`, stored weights `[N,K]`, and raw output `[1,N]`. The pinned model has 16 layers, head width 64, 32 Q heads, and 8 KV heads.

| Site | K | N | FP32 weight per call | Calls per decode |
| --- | ---: | ---: | ---: | ---: |
| Q and O | 2048 | 2048 | 16 MiB | 32 |
| K and V | 2048 | 512 | 4 MiB | 32 |
| FFN gate and up | 2048 | 8192 | 64 MiB | 32 |
| FFN down | 8192 | 2048 | 64 MiB | 16 |
| Tied vocabulary | 2048 | 128256 | 1002 MiB / 1,050,673,152 bytes | 1 |

For cached prefill use M=6 and M=29 for all block projections, and M=1 for the final classifier. Include M=3 and M=6 guard cases, and M=27–30 if benchmarking the supplemental tail neighborhood. Preserve behavior on small and irregular shapes, including N=0–5 and right-column tails such as N=513. Use realistic K=2048 and 8192, plus short, empty and odd K cases.

## Baseline

### Llama primary

Hardware was Apple M3, darwin/arm64, macOS 26.5.2 (25F84), AC power, Go 1.27.3, `GOARM64=v8.0`, `CGO_ENABLED=1`, default tags and `GOMAXPROCS=8`. The pinned Meta snapshot is revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`, verified against the Llama repo manifest each run. The primary workload is the six-token prompt `The capital of France is`, cap 512, cache capacity 13 with 851,968 KV payload bytes, one prefill and seven decode steps. Every iteration verifies eight frozen IDs and decoded bytes. TTFT here is prefill plus greedy first-token selection, with cache setup measured separately. Process wall and Darwin peak RSS include verification, model loading, tokenizer setup, and inference. Timed operations use one iteration per fresh process with zero inference warm-ups. The seven results below represent the baseline build only.

| Metric | Median | Range |
| --- | ---: | ---: |
| TTFT (s) | 0.7246 | 0.6636–1.5460 |
| Prefill (s) | 0.7245 | 0.6635–1.5450 |
| Decode (s/token) | 0.3396 | 0.3313–0.4039 |
| Decode (tokens/s) | 2.945 | 2.476–3.018 |
| Cached generation (s/op) | 3.0568 | 2.9924–4.0463 |
| Output (tokens/s) | 2.6171 | 1.9771–2.6734 |
| B/op | 130,639,968 | 130,639,952–130,645,272 |
| Allocs/op | 33,669 | 33,669–33,675 |
| Process wall (s) | 6.0740 | 5.9460–7.2457 |
| Peak RSS (GiB) | 6.3721 | 6.2767–6.3730 |

Decode s/token is `decode-s/op / 7`. Output tok/s is `8 / cached-generation s/op`. The TTFT spread is substantial even within this measurement run. Use fresh alternating pairs for before/after claims.

### Supplemental workloads

Three fresh processes per workload ran the benchmarks in Llama's `llama32_supplemental_benchmark_test.go`. Timed prefill includes greedy selection and excludes session creation. Sustained decode includes 32 calls to `Session.Decode` and greedy selection at positions 6–37 on one session, with model load, tokenizer, session creation and prefill excluded from the benchmark timer. It logs four eight-step windows. In these trials the last two windows were 2.687–2.730 s each. Whole-process wall includes setup for both workloads.

| Metric | 29-token prefill median (range) | 32 decode steps median (range) |
| --- | ---: | ---: |
| s/op | 5.0424 (4.8079–5.1696) | 11.5311 (11.4941–12.0822) |
| Decode (s/token) | — | 0.3603 (0.3592–0.3776) |
| Decode (tokens/s) | — | 2.775 (2.649–2.784) |
| B/op | 282,311,912 (282,311,704–282,311,912) | 327,835,408 (327,835,408–327,840,728) |
| Allocs/op | 5,961 (5,958–5,961) | 126,624 (126,624–126,630) |
| Process wall (s) | 8.7855 (7.7884–9.4763) | 15.3271 (15.2645–15.8211) |
| Peak RSS (GiB) | 6.2487 (5.6010–6.3719) | 6.3710 (6.3709–6.3716) |

The 29-token workload preserved winner `12366` and the Pass 3/4 SHA-256 `6ffb4dc93003480aebdd51e9f1fee47ae3836eea9fb35ce2508f5d618a9871a5` over the little-endian FP32 logits. Compare the sustained benchmark's logged suffix and final-row digest against the paired baseline on the same platform. All baseline trials produced digest `b72d983e8b5c45f83ad858b52f1d27b87a7ad8118be6c5d315db2ea6fc6fd84a` and the suffix below, which includes the prefill winner.

```text
12366 13 1102 374 279 1455 95551 3363 304 9822 323 279 3224 753 6864 323 7928 3363 13 12366 374 279 1455 95551 16036 3158 304 279 7665 9323 323 279 1917
```

### Neuralnetwork existing benchmark

Three fresh processes per series, `200ms` per subbenchmark, preallocated tensors/slices, reused weights. Raw allocations were zero. Times are median (range) in ms/op.

| Real shape | Existing raw kernel | Existing public GemmInto | Public B/op | Public allocs/op |
| --- | ---: | ---: | ---: | ---: |
| K/V | 0.257 (0.256–0.268) | 0.257 (0.257–0.261) | 2,144 | 3 |
| Q/O | 1.040 (1.032–1.049) | 1.042 (1.036–1.136) | 8,288 | 3 |
| Gate/up | 4.453 (4.446–4.468) | 4.449 (4.448–4.479) | 32,864 | 3 |
| Down | 4.312 (4.302–4.325) | 4.374 (4.303–4.617) | 8,288 | 3 |
| Vocabulary | 71.096 (70.796–78.320) | 74.105 (70.926–74.278) | 516,192 | 3 |

### Working-set probe

Separate process per trial, three trials and `200ms` per subbenchmark. The probe used three modes. Reuse repeatedly applies the kernel to one preallocated right operand. Rotate cycles through disjoint right operands totaling 256 MiB, or about 2 GiB for vocabulary. Scan uses the rotating sets and searches raw bytes for an absent value. This proxy measures equivalent bytes for a read-only contiguous traversal. It does not flush caches or measure physical DRAM bandwidth.

| Shape | Reuse ms/op | Rotate ms/op | Rotating scan ms/op |
| --- | ---: | ---: | ---: |
| K=2048, N=4 | 0.00202 (0.00200–0.00216) | — | — |
| K=2048, N=64 | 0.03214 (0.03118–0.03223) | 0.03519 (0.03418–0.03560) | 0.00826 (0.00816–0.00839) |
| K/V | 0.257 (0.252–0.261) | 0.283 (0.277–0.367) | 0.0657 (0.0653–0.0669) |
| K=2048, N=513 | 0.259 (0.254–0.265) | — | — |
| Q/O | 1.041 (1.016–1.054) | 1.132 (1.109–1.143) | 0.264 (0.261–0.264) |
| Gate/up | 4.431 (4.407–4.495) | 4.508 (4.431–4.546) | 1.048 (1.043–1.060) |
| Down | 4.307 (4.261–4.308) | 4.355 (4.323–4.368) | 1.047 (1.045–1.056) |
| Vocabulary | 71.066 (70.597–71.066) | 73.894 (73.797–74.414) | 17.514 (17.183–17.579) |

Raw logs, profiles, disassembly, and scripts from these measurements are at `/private/tmp/llama-pass5.aEeivJdn` on the profiling machine. The commands and probe source below can recreate them with a new output directory.

## Benchmarks

In the neuralnetwork checkout, build the baseline test binary from v0.10.14 and candidate from the implementation commit. Use separate worktrees if needed. These commands compile and run the existing tests without changing the checked-out files:

```sh
export GOTOOLCHAIN=go1.27.3 GOMAXPROCS=8 GOARM64=v8.0 CGO_ENABLED=1 GOFLAGS=
go test -mod=readonly -c -o "$bench_dir/nn-baseline.test" ./tensor
go version -m "$bench_dir/nn-baseline.test"
"$bench_dir/nn-baseline.test" -test.run '^$' \
  -test.bench '^BenchmarkRightTranspose$/(Kernel|GemmInto)/(QAndOutput|KAndV|FFNGateAndUp|FFNDown|Vocabulary|SingleRowTail)/^M1$' \
  -test.benchmem -test.benchtime 200ms -test.count 1
```

Build the candidate the same way. Alternate order across at least three fresh pairs, report distributions, and increase trial count or time if needed. Use the working-set probe below to repeat hot/rotating and worker-count checks. Extend the upstream benchmarks to cover the worker decision threshold, realistic work per worker, all five real shape groups, exact-fit and irregular partition tails, M=3/M=6 guards, and simultaneous invocations sharing read-only weights with distinct destinations. Include raw kernel and public `GemmInto`. Test `GOMAXPROCS=1`, and 2, 4, and 8 where available. Track allocation counts, bytes, elapsed time, effective weight bandwidth, and scheduler cost. Treat the large working sets and complete Llama model as the primary performance checks.

## Tests

- Preserve raw `TestRightTranspose`, `TestRightTransposeConcurrent`, `TestRightTransposeNonfiniteAndSignedZero`, and `TestEmptyContraction`. Keep public product reference, alias/empty contraction, bias validation, right-transpose, and dynamic-weight/concurrency tests green.
- Add bitwise comparison to an ascending serial reference for inputs both below and above the chosen threshold, including K=2048 and 8192, odd K, a partial worker range, output widths around tile/partition boundaries, nonfinite values, signed zero, and cancellation that would expose a reordered reduction. Verify overwrites of dirty outputs, unchanged inputs, separately retained outputs across calls, and retry after validation errors. Ensure tests exercise the path with workers active, including multiple concurrent calls with common weight storage.
- Run `go test -mod=readonly ./...`, `CGO_ENABLED=0 go test -mod=readonly -tags=purego ./...`, `go test -mod=readonly -race ./...`, and `go vet -mod=readonly ./...`. On supported Darwin/arm64 hardware also run `go test -mod=readonly -tags=metal ./tensor ./matrix`. Keep all existing tolerances.

## Commit discipline

Use small, logical commits with concise messages. Separate reproducible benchmark support, the validated kernel or dispatch change, and performance evidence when appropriate. Keep atomic regression tests with the code they protect or in a focused test commit. Review the resulting history and diff before handing the work back. Supply the implementation commit or release and the exact baseline/candidate version, commands, raw benchmark data, and correctness results.

## Completion criteria

- There is one scoped neuralnetwork optimization on the host single-row product path. The fast path and fallback preserve FP32 output, public validation and errors, retention, retries, and concurrency.
- Measurements cover the real FFN and classifier cases, guard small calls and multi-row prefill, and show how working set and worker count affect results. Evaluate thread contention and overhead with concurrent callers.
- Paired Llama measurements establish a repeatable decode improvement in both the primary and sustained workloads. Report the change in end-to-end generation, TTFT and supplemental prefill, allocation metrics, wall time, peak RSS, and any tradeoffs. If noise makes an effect inconclusive, extend the measurement or report it as inconclusive. Preserve material performance and memory bounds on the unaffected workload and concurrency paths.
- Re-run profiles and verify that the projection bottleneck improves in elapsed time. With parallel work, account for worker stacks as well as caller stacks. The reduction in time spent waiting or computing per projection matters even if summed CPU time stays constant or rises. A microbenchmark improvement alone leaves this work incomplete.

## Llama validation

Run from the Llama root using the source that contains `1e10d4b`. Use the verified Meta and TinyStories fixtures in `testdata/README.md`. The default benchmark and both supplemental benchmarks below must use the same source, snapshot, tags, Go environment and hardware for both dependency versions.

```sh
export GOTOOLCHAIN=go1.27.3 GOMAXPROCS=8 GOARM64=v8.0 CGO_ENABLED=1 GOFLAGS=
export LLAMA_32_1B_MODEL_DIR=/path/to/verified/4e20de362430cd3b72f300e6b0f18e50e7166e08
export LLAMA_STORIES260K_MODEL=/path/to/stories260K.bin
export LLAMA_STORIES260K_TOKENIZER=/path/to/tok512.bin
bench_dir=$(mktemp -d /private/tmp/llama-pass5-validation.XXXXXXXX)
git show b587204:go.mod > "$bench_dir/before.mod"
git show b587204:go.sum > "$bench_dir/before.sum"
cp "$bench_dir/before.mod" "$bench_dir/after.mod"
cp "$bench_dir/before.sum" "$bench_dir/after.sum"
candidate=SET_VALIDATED_TAG_OR_PSEUDOVERSION
go get -modfile="$bench_dir/after.mod" "github.com/itsmontoya/neuralnetwork@$candidate"
for build in before after; do
  go test -mod=readonly -modfile="$bench_dir/$build.mod" -c -o "$bench_dir/$build.test" .
  go version -m "$bench_dir/$build.test"
done
```

Check that both builds show Go 1.27.3 and the intended neuralnetwork pins. Keep the baseline module checksum and verify the candidate. Use one timed operation per fresh process with zero inference warm-ups. The baseline and candidate must alternate within pairs. The following uses the same Darwin `wait4` RSS units and monotonic clock as the baseline.

```sh
cat > "$bench_dir/measure.py" <<'PY'
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
    elapsed = time.perf_counter() - start
    log.write(f"process_wall_s={elapsed:.6f} peak_rss_bytes={usage.ru_maxrss}\n".encode())
sys.exit(os.waitstatus_to_exitcode(status))
PY
for workload in CachedGeneration SupplementalPrefill SustainedDecode; do
  trials='1 2 3'
  if [ "$workload" = CachedGeneration ]; then trials='1 2 3 4 5 6 7'; fi
  for trial in $trials; do
    order='before after'
    if [ "$((trial % 2))" -eq 0 ]; then order='after before'; fi
    for build in $order; do
      python3 "$bench_dir/measure.py" "$bench_dir/$workload-$build-$trial.log" \
        "$bench_dir/$build.test" -test.run '^$' -test.bench "^BenchmarkLlama32${workload}$" \
        -test.benchtime 1x -test.benchmem -test.timeout 15m -test.v || exit 1
    done
  done
done
```

Retain per-trial results, paired differences and ranges. Report the ten primary metrics from the baseline table, and report both supplemental workloads separately. Run fresh profiles on the same paired builds after collecting timing data.

```sh
for workload in CachedGeneration SupplementalPrefill SustainedDecode; do
  trials='1'
  runs=3x
  if [ "$workload" = CachedGeneration ]; then trials='1 2'; fi
  if [ "$workload" = SustainedDecode ]; then runs=2x; fi
  for trial in $trials; do
    order='before after'
    if [ "$((trial % 2))" -eq 0 ]; then order='after before'; fi
    for build in $order; do
      prefix="$bench_dir/$workload-$build-profile-$trial"
      "$bench_dir/$build.test" -test.run '^$' -test.bench "^BenchmarkLlama32${workload}$" \
        -test.benchtime "$runs" -test.benchmem -test.timeout 15m -test.v \
        -test.memprofilerate 65536 -test.cpuprofile "$prefix.cpu.pprof" \
        -test.memprofile "$prefix.heap.pprof" > "$prefix.log" 2>&1 || exit 1
    done
  done
done

for build in before after; do
  go tool pprof -top -relative_percentages -focus='\(\*Session\)\.Decode' \
    "$bench_dir/$build.test" "$bench_dir"/CachedGeneration-"$build"-profile-*.cpu.pprof
  go tool pprof -top -cum -alloc_space -relative_percentages -focus='BenchmarkLlama32' \
    -ignore='LoadHFModel|LoadLlama3|verifyLlama32Snapshot' \
    "$bench_dir/$build.test" "$bench_dir"/CachedGeneration-"$build"-profile-*.heap.pprof
done
```

Repeat with `Prefill` and `-alloc_objects`, and inspect the whole profiles. The `Session.Decode` stack filter matches caller work. Worker samples can appear on separate goroutine stacks. For a parallel candidate, capture a decode-only profile or run a profiling-only benchmark with pprof phase labels inherited by/tagged on workers. A lower filtered caller CPU share without worker attribution is not evidence of reduced work. Compare elapsed decode, profile totals and scheduling separately, including the 29-token prefill and one-row classifier.

Run the following with both modfiles. The 1B model is used in the default and purego suites. The race run uses the synthetic and TinyStories fixtures to exercise concurrent decoder/session behavior and ownership.

```sh
for build in before after; do
  modfile="$bench_dir/$build.mod"
  go test -mod=readonly -modfile="$modfile" -p=1 -count=1 -timeout 45m -v ./... || exit 1
  CGO_ENABLED=0 go test -mod=readonly -modfile="$modfile" -tags=purego \
    -p=1 -count=1 -timeout 45m -v ./... || exit 1
  LLAMA_32_1B_MODEL_DIR= go test -mod=readonly -modfile="$modfile" -race -p=1 -count=1 ./... || exit 1
  go vet -mod=readonly -modfile="$modfile" ./... || exit 1
  go build -mod=readonly -modfile="$modfile" -o "$bench_dir/generate-$build-default" ./cmd/generate || exit 1
  CGO_ENABLED=0 go build -mod=readonly -modfile="$modfile" -tags=purego \
    -o "$bench_dir/generate-$build-purego" ./cmd/generate || exit 1
  for tags in default purego; do
    for mode in cached uncached; do
      (
        set --
        if [ "$mode" = uncached ]; then set -- -uncached; fi
        prefix="$bench_dir/cli-$build-$tags-$mode"
        "$bench_dir/generate-$build-$tags" -hf-model-dir "$LLAMA_32_1B_MODEL_DIR" \
          -context 512 -prompt 'The capital of France is' -max-new-tokens 8 "$@" \
          > "$prefix.stdout" 2> "$prefix.stderr" || exit 1
        printf '%s\n' ' Paris. It is the most populous city' | cmp - "$prefix.stdout" || exit 1
        test ! -s "$prefix.stderr"
      ) || exit 1
    done
  done
done
```

Keep `TestLlama32Logits` Transformers comparison at `atol=rtol=1e-3`, all frozen winners, and all five `TestLlama32CachedParity` rows at `atol=rtol=1e-4`. Verify the suffix `[12366, 13, 1102, 374, 279, 1455, 95551, 3363]` and byte-identical CLI output above. Keep synthetic MHA/GQA/MQA traces, TinyStories logits/generation, retry, ownership and concurrent session/decoder tests green. Compare the long-prefill and sustained-decode hashes and sustained suffix with the paired baseline.

### Baseline correctness already run

Both default and `purego` suites passed with the pinned model and TinyStories fixtures. Transformers' maximum absolute logit error was `4.95910645e-05` and cached/full-prefix maximum absolute error was zero. All four standalone CLI runs (default/purego and cached/uncached) matched the frozen bytes and emitted empty stderr. Race tests passed with synthetic and TinyStories fixtures, and `go vet` passed. Repeat these checks against the returned dependency.

## Working-set probe reproduction

Use a standalone temporary module to call the existing internal kernel. The import prefix below allows Go's internal package check. Record the neuralnetwork checkout to measure in `nn_checkout`, create the probe module, and save the code below as `$probe_dir/projection_test.go`.

```sh
export GOTOOLCHAIN=go1.27.3 GOMAXPROCS=8 GOARM64=v8.0 CGO_ENABLED=1 GOFLAGS=
nn_checkout=/absolute/path/to/checked-out-neuralnetwork
probe_dir=$(mktemp -d /private/tmp/llama-pass5-probe.XXXXXXXX)
# Save projection_test.go in $probe_dir before the next commands.
(
  cd "$probe_dir" || exit 1
  go mod init github.com/itsmontoya/neuralnetwork/pass5probe || exit 1
  go mod edit -go=1.27.3 -require=github.com/itsmontoya/neuralnetwork@v0.10.14 \
    -replace="github.com/itsmontoya/neuralnetwork=$nn_checkout" || exit 1
  cp "$nn_checkout/go.sum" . || exit 1
  go mod tidy || exit 1
  go test -mod=readonly -c -o "$probe_dir/probe.test" . || exit 1
  go version -m "$probe_dir/probe.test"
  for trial in 1 2 3; do
    "$probe_dir/probe.test" -test.run '^$' -test.bench '^(BenchmarkWorkingSet|BenchmarkIndependentFFN)$' \
      -test.benchmem -test.benchtime 200ms -test.count 1 > "$probe_dir/probe-$trial.log" || exit 1
  done
)
```

Use the v0.10.14 checkout for the baseline and the candidate checkout for the paired measurements. The original probe used the published v0.10.14 directly. Compare `Workers1` to `Workers2/4/8` on the baseline for independent-job scaling. The candidate's internal dispatch may parallelize each job, so interpret those candidate rates as oversubscription results.

```go
package pass5probe

import (
	"bytes"
	"fmt"
	"math"
	"sync"
	"testing"
	"unsafe"

	"github.com/itsmontoya/neuralnetwork/internal/float32product"
)

func weights(size int) []float32 {
	values := make([]float32, size)
	for i := range values {
		// The byte 0xff is absent, so Scan reads the whole slice.
		values[i] = math.Float32frombits(0x3f010101 + uint32(i%63))
	}
	return values
}

func BenchmarkWorkingSet(b *testing.B) {
	for _, shape := range []struct {
		name string
		k, n int
	}{
		{"N4", 2048, 4}, {"N64", 2048, 64},
		{"KV", 2048, 512}, {"KVTail", 2048, 513},
		{"QOut", 2048, 2048}, {"GateUp", 2048, 8192},
		{"Down", 8192, 2048}, {"Vocab", 2048, 128256},
	} {
		for _, mode := range []string{"Reuse", "Rotate", "Scan"} {
			if (shape.n == 4 || shape.n == 513) && mode != "Reuse" {
				continue
			}
			b.Run(shape.name+"/"+mode, func(b *testing.B) {
				size := shape.k * shape.n
				nsets := 1
				if mode != "Reuse" {
					nsets = max(2, (256<<20)/(size*4))
				}
				left, output := weights(shape.k), make([]float32, shape.n)
				right := make([][]float32, nsets)
				for i := range right {
					right[i] = weights(size)
				}
				b.SetBytes(int64(size * 4))
				b.ReportAllocs()
				set := 0
				for b.Loop() {
					values := right[set]
					if mode == "Scan" {
						raw := unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), size*4)
						if bytes.IndexByte(raw, 0xff) != -1 {
							b.Fatal("weight contains scan marker")
						}
					} else {
						float32product.RightTranspose(left, values, output, 1, shape.k, shape.n)
					}
					set++
					if set == nsets {
						set = 0
					}
				}
			})
		}
	}
}

func BenchmarkIndependentFFN(b *testing.B) {
	const k, n, jobs = 2048, 8192, 8
	for _, workers := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("Workers%d", workers), func(b *testing.B) {
			left := weights(k)
			var right, output [jobs][]float32
			for i := range right {
				right[i], output[i] = weights(k*n), make([]float32, n)
			}
			start := make([]chan struct{}, workers)
			var finished, stopped sync.WaitGroup
			for worker := range workers {
				start[worker] = make(chan struct{})
				stopped.Go(func() {
					for range start[worker] {
						for job := worker; job < jobs; job += workers {
							float32product.RightTranspose(left, right[job], output[job], 1, k, n)
						}
						finished.Done()
					}
				})
			}
			b.SetBytes(jobs * k * n * 4)
			b.ReportAllocs()
			for b.Loop() {
				finished.Add(workers)
				for _, ch := range start {
					ch <- struct{}{}
				}
				finished.Wait()
			}
			for _, ch := range start {
				close(ch)
			}
			stopped.Wait()
		})
	}
}
```
