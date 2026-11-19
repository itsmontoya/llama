# Optimization Pass 3 neuralnetwork handoff

## Target and ownership

Optimize the **multi-row CPU FP32 right-transposed product**, specifically
the three-input-row tile used by Llama's prefill projections.
The implementation belongs in `github.com/itsmontoya/neuralnetwork`
at `v0.10.12` (`1efde389dc71f21b37580d4325d802ce91c23620`).
The Llama baseline is `20a45d9`. Keep Llama's tensor calls and the
one-row projection path as they are for this pass.

## Fresh Go 1.27.3 baseline

Measurements used an Apple M3 (four performance and four efficiency cores),
24 GB RAM, macOS 26.5.2 (25F84), darwin/arm64, Go 1.27.3, `GOARM64=v8.0`,
`GOMAXPROCS=8`, `CGO_ENABLED=1`, and default tags. The benchmark verifies the
pinned Llama 3.2 1B snapshot, loads it once, and generates eight tokens from
the six-token prompt `The capital of France is` with context 512. The
session has 13 cache positions and runs one prefill and seven decode calls.
These are five unprofiled, one-operation fresh processes without inference
warm-ups. The paired comparison should remeasure the baseline in alternating
order with the candidate. Wall time and peak RSS include setup.

| Metric | Median | Range |
| --- | ---: | ---: |
| Prefill / first token (s) | 2.107 | 1.838–2.492 |
| Decode (s/token) | 0.369 | 0.319–0.464 |
| Decode (tokens/s) | 2.711 | 2.154–3.137 |
| Cached generation (s/op) | 4.784 | 4.072–5.301 |
| Output (tokens/s) | 1.672 | 1.509–1.964 |
| Generation bytes/op | 198,723,312 | 198,717,992–198,723,328 |
| Generation allocs/op | 39,872 | 39,866–39,872 |
| Process wall (s) | 7.970 | 7.001–9.297 |
| Process peak RSS (GiB) | 6.345 | 6.205–6.373 |

## Profiling evidence

A separate three-operation run of the same workload used
`-test.cpuprofile`, `-test.memprofile`, and `-test.memprofilerate 65536`.
CPU values are sampled self time unless indicated.

| Stack filter | Total CPU | Three-row kernel | One-row kernel |
| --- | ---: | ---: | ---: |
| Entire process | 13.99 s | 3.23 s (23.09%) | 3.98 s (28.45%) |
| `Session.Prefill` | 4.70 s | 3.23 s (68.72%) | 0.22 s (4.68%) |
| `Session.Decode` | 6.50 s | 0 | 3.76 s (57.85%) |

The three-row loop accounts for 3.02 of its 3.23 sampled self seconds at
`product.go:131–134` and 3.73 s inclusive. The ARM64 inner loop emits four
scalar loads and three scalar `fmadd` instructions. Each accumulator
depends on the previous contraction's result. Bounds checks are already
outside the inner loop. Multiple independent columns can let the CPU
overlap those dependencies and reuse the three activation loads. The
one-row kernel is larger in the aggregate profile and already uses four
independent output columns. Prefill offers a similarly large CPU target
with an observable opportunity to reuse loads and increase instruction
parallelism. Prompts of six tokens run two three-row tiles per block
projection. Longer prompts also use the tile, with a scalar remainder
when their row count is not divisible by three.

The whole-process allocation profile estimates 10,474.94 MiB cumulative,
of which model loading accounts for 9,758.26 MiB (93.16%). With loading,
tokenizer setup, and snapshot hashing filtered out, benchmark stacks
account for 570.48 MiB. `elementwise` accounts for 298.64 MiB inclusive,
including 197.19 MiB under the binary tensor adapter. `swiGLU` accounts for
236.51 MiB inclusive, overlapping elementwise and activation allocations.
`matrixProductValues` itself allocates an estimated 67.74 MiB (11.87%).
The allocation-object profile attributes about half of generation objects
to `elementwise` stacks. Use the benchmark's unprofiled bytes/op and
allocs/op to compare the actual totals.

## Call path and current behavior

- Pinned benchmark `BenchmarkLlama32CachedGeneration` in
  `llama32_benchmark_test.go`, or `Decoder.GenerateGreedyCached` in
  `generation.go` → `Session.Prefill` in `session.go` →
  `Decoder.forwardPrefix` → `Decoder.forwardBlock` in `decoder.go`.
- Q/K/V take the `headProjection` route through `attention.go`. Attention
  output and FFN gate/up/down call `project` directly. All seven block
  projections use `project` in `tensor_ops.go`.
- `project` calls `tensor.GemmShape` and `tensor.GemmInto` with
  `GemmConfig{TransposeB: true, Alpha: 1}` and nil bias. That validated
  rank-two API calls `matrixProductValues` → `matrixProductInto` in
  `tensor/matrix_product_host.go` → `float32product.RightTranspose`
  → `rightTransposeThreeRows` in `internal/float32product/product.go`.
  The host adapter uses call-owned staging and commits the result after
  coefficient and bias processing.
- CPU matrix callers also reach the kernel through
  `Matrix.MatMulRightTransposeInto` and `matMulRightTransposeIntoPure` in
  `matrix/matmul_pure.go`. Full-prefix Llama evaluation uses the same block
  projections. Cached prefill gathers its last hidden row before the
  vocabulary projection, so its classifier and all cached decode
  projections use the one-row path.
- The product reads activations `[M,K]` and contiguous weight rows
  `[N,K]`, writing row-major `[M,N]`. The three-row helper loads one
  weight value for three input rows, advancing one output column at a time.
  One or two rows remaining after complete tiles use the scalar loop.
  Llama block projection `(K,N)` shapes are `(2048,2048)`,
  `(2048,512)`, `(2048,8192)`, and `(8192,2048)`.

## Proposed implementation and constraints

Prototype a small output-column tile within the three-row helper. Start
with two columns and compare a wider tile only if measurements warrant it.
Keep separate scalar FP32 accumulators for each output, retain the
single-column tail, and prefer a portable Go implementation. Compare the
generated ARM64 loop for input reuse, unneeded bounds checks, and spills.
Choose the smallest change that improves steady prefill across the main
projection shapes without a material regression elsewhere. Preserve
the general layout and existing tensor and matrix APIs. The kernel
should require zero workspace allocations.

Initialize each accumulator at positive FP32 zero and add terms in
ascending contraction order. Preserve the existing Go compiler's fused
multiply-add semantics where applicable. Avoid reassociation, split
partial sums, horizontal reductions across contraction lanes, or wider
intermediate precision. Preserve ordinary FP32 handling of cancellations,
nonfinite values, and signed zero. Exercise empty output, zero contraction,
small and odd dimensions, output-column tails, row remainders, and reused
destinations. Keep the source slices read-only and overwrite each
validated output cell. Retain no storage. Concurrent callers may share
immutable input/weight slices with disjoint destinations. Preserve
existing alias validation and error atomicity in the public APIs.

## neuralnetwork benchmarks and tests

Use Go 1.27.3 for both revisions. Build separate test binaries and
alternate repeated runs with the same tags and CPU settings. Run
`BenchmarkRightTranspose` in `./tensor` for both its raw kernel and
public `GemmInto` cases. Compare M=6 for all four block-projection shapes,
M=1 for decode regressions, and the existing small and irregular cases.
Use M=3/5/12 or a moderately longer row count to cover a complete tile,
row remainder, and a prefill longer than six tokens. Include an odd
output-column count. Check `Benchmark_MatMulRightTransposeIntoShapes`
in `./matrix` for the matrix caller. Report elapsed time and allocations.

Run `go test -count=1 ./internal/float32product ./tensor ./matrix`,
repeat with `CGO_ENABLED=0` and `-tags=purego`, run the same package tests
with `-race`, and run `go vet ./...`. Cover the new tile and its tail with
bitwise comparisons to the ordered scalar reference. Use the existing
public GEMM and matrix tests for destination/alias behavior and the
ascending-order test. If available on darwin/arm64, also run the three
packages with `-tags=metal` to check the shared CPU fallback.

## Validate the result in Llama

Use a temporary alternate modfile with a local replacement for preliminary
testing. For integration, build with the exact published tag or commit
pseudo-version, update the real `go.mod` and `go.sum`, and verify both
binaries with `go version -m`. Keep the original pin and sums from
`20a45d9` in a separate before modfile. From the Llama root with
`LLAMA_32_1B_MODEL_DIR` set to the
[pinned snapshot](testdata/README.md#llama-32-1b-transformers-reference):

```sh
export GOTOOLCHAIN=go1.27.3 GOARM64=v8.0 GOMAXPROCS=8 CGO_ENABLED=1
bench_dir=$(mktemp -d)
git show 20a45d9:go.mod > "$bench_dir/before.mod"
git show 20a45d9:go.sum > "$bench_dir/before.sum"
go test -mod=readonly -modfile="$bench_dir/before.mod" -c -o "$bench_dir/before.test" .
go test -mod=readonly -c -o "$bench_dir/after.test" .
```

Run at least five alternating fresh-process pairs. Use
`-test.run '^$' -test.bench '^BenchmarkLlama32CachedGeneration$'
-test.benchtime 1x -test.benchmem -test.timeout 15m -test.v` for each
binary. Record process wall time and peak RSS from macOS `wait4`.
Compute decode seconds/token as decode seconds/op divided by seven
and output throughput as eight divided by generation seconds/op.
Compare medians and observed ranges. Profile separately at
`-test.benchtime 3x -test.memprofilerate 65536` with CPU and heap output
files. Use `pprof -top`, `pprof -list`, and `pprof -disasm` for the
multi-row kernel, `-focus='\(\*Session\)\.Prefill'` and
`-focus='\(\*Session\)\.Decode'` for CPU, and both `-alloc_space`
and `-alloc_objects` with and without setup filters for allocation.
Check the kernel's absolute sampled CPU per operation and both stages'
end-to-end times.

Enable `LLAMA_STORIES260K_MODEL` and `LLAMA_STORIES260K_TOKENIZER`
with verified [TinyStories artifacts](README.md#generate-text-with-tinystories-260k).
Run `go test -mod=readonly -p=1 -count=1 -timeout 45m -v ./...` for
each revision, then repeat with `CGO_ENABLED=0` and `-tags=purego`.
Run `go vet ./...` and `go test -race -p=1 -count=1 ./...` with the
Llama 3.2 model variable unset for the race run and TinyStories enabled.
The existing suite checks synthetic traces, pinned Transformers logits
and exact greedy winners (`atol=rtol=1e-3`), cached/full-prefix parity
for five full logit rows (`atol=rtol=1e-4`), TinyStories, tokenizer,
and both CLI modes. Preserve the suffix IDs
`[12366, 13, 1102, 374, 279, 1455, 95551, 3363]`. Build both CLI
configurations and compare stdout byte for byte with
` Paris. It is the most populous city` followed by one newline.
