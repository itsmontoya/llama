# Improve the CPU right-transposed FP32 product

## Problem and ownership

Optimize `internal/float32product.RightTranspose` in
`github.com/itsmontoya/neuralnetwork`. The right operand is already laid out
as contiguous weight rows. The current loop reads one element from each row
before moving to the next contraction element. On the pinned Llama 3.2 1B
checkpoint, these reads have an 8 KiB stride for most projections and a 32 KiB
stride for the FFN down projection. Each multiply-add also reads and writes an
element of the output row.

The normal Llama path calls `tensor.GemmInto` with `GemmConfig{TransposeB:
true, Alpha: 1}` and nil bias. The host tensor backend already sends the
validated weight payload straight to the product kernel. The access order
belongs in the shared CPU kernel, which also serves the default CPU
`matrix.Matrix.MatMulRightTransposeInto` path. Llama stores weights as
`[output,input]`, and its tied classifier shares the contiguous embedding
rows used by token lookup. A separate transposed classifier would add
1,050,673,152 bytes of FP32 weights for this model alone.

| Repository and file | Relevant path |
| --- | --- |
| Llama `session.go` | `Session.Prefill` calls `Decoder.forwardPrefix`; `Session.Decode` calls `Decoder.forwardBlock` for one row. Both call `Session.commit` to get final logits. |
| Llama `decoder.go`, `attention.go`, `tensor_ops.go` | `forwardBlock` and `projectLogits` use `project`, with `headProjection` wrapping the Q, K, and V calls. `project` allocates the destination and calls `tensor.GemmInto`. |
| neuralnetwork `tensor/gemm.go`, `tensor/matrix_product_host.go` | `GemmInto` validates the inputs, bias and destination, then uses `matrixProductValues` → `matrixProductInto` → `float32product.RightTranspose`. It applies coefficients and commits the staged output. |
| neuralnetwork `internal/float32product/product.go`, `matrix/matmul_default.go`, `matrix/matmul_pure.go` | The shared scalar kernel is called directly by the host tensor backend and through the default matrix backend. The `metal` tensor build has its own adapter. |

## Measured baseline

Llama `v0.1.0` (`6e53cb6c9f5f1458d56e5c27b597b1940a8327c7`) pins
neuralnetwork `v0.10.10` (`2ffa5bd9fd2034382c49d14efc732ea4d026aa09`).
The host was an Apple M3 with four performance and four efficiency cores,
24 GB RAM, macOS 26.5.2 (25F84), Go 1.27.2 darwin/arm64, default build tags,
`GOARM64=v8.0`, `CGO_ENABLED=1`, and `GOMAXPROCS=8`.

The pinned `meta-llama/Llama-3.2-1B` revision is
`4e20de362430cd3b72f300e6b0f18e50e7166e08`. Its safetensors SHA-256 is
`68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`.
The loader converts BF16 to FP32 and retains 4,943,257,600 weight payload
bytes. The benchmark uses the frozen prompt `The capital of France is`
(six tokens including BOS), a 512-token context cap, and eight greedy
predictions. It uses one prefill and seven decode calls. The cache holds
13 positions and has 851,968 bytes of KV payload. Every trial verifies the
snapshot, prompt IDs, suffix IDs, final position, and decoded text.

| Metric | Trial 1 | Trial 2 | Trial 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Model load (s) | 2.657 | 2.703 | 2.732 | 2.703 |
| Tokenizer load plus prompt encoding (s) | 0.1838 | 0.1764 | 0.1721 | 0.1764 |
| Session allocation (µs) | 21.0 | 18.2 | 11.0 | 18.2 |
| Prefill / first token (s) | 18.32 | 18.83 | 18.73 | 18.73 |
| Prefill (prompt tokens/s) | 0.328 | 0.319 | 0.320 | 0.320 |
| Seven decode steps (s) | 28.00 | 27.96 | 28.26 | 28.00 |
| Mean decode step (s/token) | 4.000 | 3.994 | 4.037 | 4.000 |
| Decode throughput (tokens/s) | 0.2500 | 0.2504 | 0.2477 | 0.2500 |
| Cached generation (s/op) | 46.315 | 46.792 | 46.992 | 46.792 |
| Cached generation (output tokens/s) | 0.173 | 0.171 | 0.170 | 0.171 |
| Cached generation (B/op) | 198,723,344 | 198,717,784 | 198,723,328 | 198,723,328 |
| Cached generation (allocs/op) | 39,872 | 39,863 | 39,872 | 39,872 |
| Benchmark process wall (s) | 50.300 | 50.785 | 51.025 | 50.785 |
| Benchmark process peak RSS (GiB) | 6.119 | 5.469 | 5.418 | 5.469 |

Each trial ran in a fresh process with one measured operation and zero
inference warm-ups. Snapshot verification can warm the filesystem cache.
Cached generation metrics cover session creation, prefill, greedy selection,
decode, and output verification including suffix decoding. Model load, tokenizer
setup, and cached generation account for a median 49.672 s together before
adding snapshot verification and test process overhead. The prefill and
first-token latencies round to the same numbers. Decode includes greedy
selection and the reported seconds per token average seven steps at successive
cache positions. Process wall time and RSS include checkpoint hashing, model
loading, and tokenizer setup. Peak RSS was collected from macOS `wait4`.
The profile was collected in a separate process with the same workload.

## CPU and heap evidence

| CPU profile scope | Sampled CPU | `RightTranspose` self CPU | Share of scope |
| --- | ---: | ---: | ---: |
| Entire benchmark process | 43.69 s | 37.85 s | 86.63% |
| Stacks under `Session.Prefill` | 16.22 s | 15.47 s | 95.38% |
| Stacks under `Session.Decode` | 24.40 s | 22.38 s | 91.72% |

The kernel's inclusive time was 39.43 s (90.25% of process samples).
`product.go:51`, the strided weight load and output update, held 36.24 s
of self samples. The ARM64 disassembly shows a scalar fused multiply-add
and an output load and store in that inner loop. The three FFN projection
call sites held 29.89 s of cumulative CPU. The vocabulary projection
held another 7.74 s under `projectLogits`. The short prompt put little
sampled CPU in attention, KV-cache access, RoPE, RMSNorm, or greedy selection.

The sampled allocation profile attributed 9,758.60 of 10,080.09 MiB
(96.81%) of cumulative allocations to model loading. That includes 4,723.53
MiB under safetensors conversion, 4,714.55 MiB under the retained tensor
copies, and 320.02 MiB under the Q/K row permutation. These are allocation
totals over the run. The exit-time in-use profile no longer held the model,
so the table uses process peak RSS.

After filtering out setup, the profile estimates 174.39 MiB on benchmark stacks.
`tensor.newTensor` had 66.41 MiB self (38.08%) and about 75% of the sampled
object count. The matrix adapter's `FromSlice`, `New`, and `Values` had
79.41 MiB self combined (45.54%), on binary arithmetic and activation
stacks. Host `GemmInto` result staging was 17.25 MiB self (9.89%).
`swiGLU` accounted for 75.62 MiB cumulative (43.36%), overlapping the
tensor and adapter allocations. Use the unprofiled `-benchmem` results for
total allocation comparisons.

## Implementation direction

1. Benchmark a portable loop that computes each output as a dot product of
   two contiguous rows, using a local FP32 accumulator and one output store.
   Keep each output's contraction order ascending for the first candidate.
   Measure one input row and the six-row prompt separately. If prefill needs
   more reuse, benchmark computing several prompt rows per weight row.
2. Keep the change inside the shared host product kernel and keep existing
   tensor validation, result staging, alpha and bias arithmetic, and
   dispatch. The existing `matrix/dot_product_benchmark_test.go` has a
   private dot-product candidate with smaller shapes. It can guide the
   benchmarks. If a SIMD variant is warranted by measured results, use
   architecture dispatch with a portable fallback and document any change
   in the reduction order or fused arithmetic.
3. Use call-local accumulators and the supplied weights on each call. A
   decoder can share immutable inputs across concurrent sessions, and other
   library callers can supply different weights on later calls. Preserve
   disjoint destinations, read-only inputs, call-scoped access to operands,
   and the public guarantee that errors leave the destination unchanged.
   Include irregular sizes, empty outputs, zero contraction, signed zero,
   and nonfinite input cases. Preserve existing `GemmInto` alpha and beta
   semantics, including zero beta with validated bias.

The isolated change targets about 93% of sampled prefill-plus-decode CPU
by self time. A twofold speedup in that code alone would imply about 1.9×
the combined sampled CPU throughput if the rest of the profile held steady.
Wall-clock improvement remains to be measured. Working set and allocation
counts should stay comparable for a loop-order change.

## Benchmarks and correctness in neuralnetwork

Benchmark both `float32product.RightTranspose` with a preallocated output and
the public `tensor.GemmInto` call with a preallocated destination and Llama's
config and bias. Use deterministic, signed finite FP32 values. Allocate and
initialize operands outside the timer. Use `M=1` for decode and `M=6` for
prefill except where indicated.

| Projection | Right operand `[N,K]` | Input rows `M` |
| --- | --- | --- |
| Q and attention output | `[2048,2048]` | 1, 6 |
| K and V | `[512,2048]` | 1, 6 |
| FFN gate and up | `[8192,2048]` | 1, 6 |
| FFN down | `[2048,8192]` | 1, 6 |
| Tied vocabulary | `[128256,2048]` | 1 in cached prefill and decode; 6 for full-prefix regression |

Include small, irregular cases to detect regressions in other matrix
callers. Collect at least five samples per case with `-benchmem` and
compare both revisions using `benchstat`. Report latency, allocation
counts and bytes, with the CPU, Go version, tags, and thread settings. The
exact-shape kernel baseline should be collected on v0.10.10 before changing
it. The table above records the actual Llama application baseline.

Run the existing tests in `internal/float32product/product_test.go`,
`tensor/products_test.go`, and `matrix/matrix_test.go`. Extend their coverage
with ascending-order dot-product reference cases for single-row, multi-row
and irregular dimensions, checking repeated writes into an already populated
destination. Exercise shared inputs with separate destinations and use
the race detector. Run default and `CGO_ENABLED=0 -tags=purego` suites,
and check the tagged Metal suite on a supported Darwin host if shared
fallback dispatch changes. Check optimized results against the existing
references without changing their tolerances.

## Reproduce and verify in Llama

Set `LLAMA_32_1B_MODEL_DIR` to the local snapshot directory.
[Pinned snapshot setup](testdata/README.md#llama-32-1b-transformers-reference)

Run from the Llama repository root. Use `/usr/bin/time -l` for wall time and
peak RSS on macOS.

```sh
bench_dir=$(mktemp -d)
export CGO_ENABLED=1 GOMAXPROCS=8
go test -c -o "$bench_dir/llama.test" .
for trial in 1 2 3; do
  /usr/bin/time -l "$bench_dir/llama.test" \
    -test.run '^$' -test.bench '^BenchmarkLlama32CachedGeneration$' \
    -test.benchtime 1x -test.benchmem -test.timeout 15m -test.v \
    >"$bench_dir/trial-$trial.log" 2>&1
done
"$bench_dir/llama.test" \
  -test.run '^$' -test.bench '^BenchmarkLlama32CachedGeneration$' \
  -test.benchtime 1x -test.benchmem -test.timeout 15m -test.v \
  -test.cpuprofile "$bench_dir/cpu.pprof" -test.memprofile "$bench_dir/heap.pprof"
go tool pprof -top "$bench_dir/llama.test" "$bench_dir/cpu.pprof"
for stage in Prefill Decode; do
  go tool pprof -top -focus="\\(\\*Session\\)\\.$stage" -relative_percentages \
    "$bench_dir/llama.test" "$bench_dir/cpu.pprof"
done
go tool pprof -list='float32product.RightTranspose' \
  "$bench_dir/llama.test" "$bench_dir/cpu.pprof"
go tool pprof -top -alloc_space "$bench_dir/llama.test" "$bench_dir/heap.pprof"
go tool pprof -top -alloc_space -relative_percentages \
  -focus=BenchmarkLlama32CachedGeneration \
  -ignore='LoadHFModel|LoadLlama3|verifyLlama32Snapshot' \
  "$bench_dir/llama.test" "$bench_dir/heap.pprof"

go test -p=1 -count=1 -timeout 45m -v ./...
```

After the library change, build the same Llama revision against the candidate
via a temporary Go workspace or module replacement. Compare fresh-process
medians and trial ranges for prefill, decode, generation throughput,
allocation counts and bytes, and peak RSS. Reprofile to confirm the hot
kernel accounts for less CPU. Run the reference command on both builds with
the snapshot enabled. It checks sampled frozen Transformers logits and greedy
winners (`atol=rtol=1e-3`), cached/full-prefix parity across all 128,256
logits on five positions (`atol=rtol=1e-4`), checkpoint inventory, tokenizer
IDs and bytes, and cached and uncached CLI suffix IDs and stdout. The expected
suffix IDs are `[12366, 13, 1102, 374, 279, 1455, 95551, 3363]`. The
continuation is ` Paris. It is the most populous city` and the CLI adds a
trailing newline. The standard suite checks synthetic traces, retries,
retained output ownership, and concurrent sessions.

## Follow-up candidates

1. Reduce SwiGLU temporaries and the elementwise and activation adapter
   copies on inference stacks. Use the new kernel's profile to prioritize
   fusion or destination reuse.
2. Reduce transient FP32 copies during checkpoint loading. Measure peak RSS
   and load latency while retaining validated, privately owned weights.
3. Evaluate host GEMM result staging and its copy after the product path is
   faster. Preserve coefficient and bias behavior and destination atomicity.
4. Profile longer prompts and cache histories to evaluate prefill's GQA
   materialization and attention planning. The six-token workload only
   attributed 7.79 MiB of sampled allocations to prefill attention.
