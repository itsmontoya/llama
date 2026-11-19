# Optimization Pass 2 — speed up single-row FP32 projections

## Target and ownership

Optimize the single-input-row case of `float32product.RightTranspose` in
`github.com/itsmontoya/neuralnetwork`. Seven cached decode steps and the
classifier at the end of prefill use this case. Pass 1 already changed the
kernel to read contiguous weight rows and reuse each weight load across three
input rows. The next change should improve the single-row dot products.

| Repository and file | Call path and responsibility |
| --- | --- |
| Llama `session.go`, `decoder.go` | `Session.Prefill` calls `forwardPrefix` for six rows. `Session.Decode` calls `forwardBlock` for one row. Both call `Session.commit`, which projects the final hidden row through `projectLogits`. |
| Llama `decoder.go`, `attention.go`, `tensor_ops.go` | Q/K/V, attention output, FFN gate/up/down, and vocabulary projection all reach `project`. It allocates the destination and calls `tensor.GemmInto` with `GemmConfig{TransposeB: true, Alpha: 1}` and nil bias. |
| neuralnetwork `tensor/gemm.go`, `tensor/matrix_product_host.go` | `GemmInto` checks the shape, types, bias, and destination. `matrixProductValues` stages the result, and `matrixProductInto` dispatches to `float32product.RightTranspose`. |
| neuralnetwork `internal/float32product/product.go`, `matrix/matmul_default.go`, `matrix/matmul_pure.go` | The shared CPU kernel serves the host tensor path and the default CPU `Matrix.MatMulRightTransposeInto` path. The tagged Metal backend has its own tensor adapter. |

The kernel receives contiguous `[N,K]` FP32 weights from both callers.
Llama's classifier shares its `[128256,2048]` tensor with the embedding
lookup. Duplicating those weights in another layout would add 1,050,673,152
bytes for this model. The shared kernel can work directly with the existing
layout and cover both callers.

## Go 1.27.3 pre-optimization baseline

Llama `06f34750334e579cafa7ad5d6733306a0134e101` pins neuralnetwork
`v0.10.11` (`b5a97884d85aad9dcbe5b86362f09e867b0b005d`). The run used Go
1.27.3, macOS 26.5.2 (25F84) darwin/arm64, Apple M3 (four performance and
four efficiency cores), 24 GB RAM, `GOARM64=v8.0`, `CGO_ENABLED=1`,
`GOMAXPROCS=8`, and default build tags.

`BenchmarkLlama32CachedGeneration` verifies the six files of the pinned
`meta-llama/Llama-3.2-1B` revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`.
The safetensors SHA-256 is
`68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`.
The frozen prompt `The capital of France is` uses six tokens including BOS.
The context cap is 512. Eight greedy predictions use one prefill and seven
decode steps, with 13 cache positions and 851,968 bytes of KV payload.
Each trial checks the prompt IDs, suffix IDs, final position, and decoded
bytes.

Five fresh processes each measured one operation with zero inference
warm-ups. Snapshot hashing can warm the file cache. Model load and
tokenizer setup have separate timers. The generation timer covers session
creation, inference, greedy selection, and suffix verification. Process wall
and peak RSS also include checkpoint hashing and model/tokenizer setup.
Peak RSS came from macOS `wait4`.

| Metric | Median | Trial range |
| --- | ---: | ---: |
| Model load (s) | 1.894 | 1.803–2.694 |
| Tokenizer load and prompt encoding (s) | 0.1878 | 0.1684–0.2232 |
| Session allocation (µs) | 8.0 | 6.2–27.3 |
| Prefill / first token (s) | 2.940 | 2.750–2.972 |
| Prefill (prompt tokens/s) | 2.041 | 2.019–2.182 |
| Seven decode steps (s) | 7.971 | 7.874–8.437 |
| Decode (s/token) | 1.139 | 1.125–1.205 |
| Decode (tokens/s) | 0.8782 | 0.8297–0.8890 |
| Cached generation (s/op) | 10.940 | 10.817–11.187 |
| Cached generation (output tokens/s) | 0.7313 | 0.7151–0.7396 |
| Generation bytes/op | 198,723,312 | 198,723,104–198,723,344 |
| Generation allocations/op | 39,872 | 39,869–39,872 |
| Benchmark process wall (s) | 14.036 | 13.975–15.187 |
| Benchmark process peak RSS (GiB) | 6.258 | 5.590–6.340 |

### Baseline correctness

The full default and `purego` suites passed with both pinned models
enabled. The Llama 3.2 tests agreed with every frozen greedy winner and
recorded a maximum sampled Transformers logit error of
`4.95910645e-05`. Cached and full-prefix logits matched exactly at all five
checked positions. Both CLI builds produced the frozen continuation byte
for byte. All five benchmark trials passed their own generation checks.
`go vet` and the race suite also passed. The race run used the synthetic
and TinyStories tests.

### Historical Go 1.27.2 comparison

With the same dependency version, Pass 1 ended at 2.085 s to first token,
1.119 s/token for decode, and 10.222 s/op for generation. The new medians
were 2.940 s, 1.139 s/token, and 10.940 s/op. The first-token trial ranges
overlap (2.011–2.982 s historically, 2.750–2.972 s here), and allocation
counts were within six at the median. The older and newer measurements
came from separate runs. Use paired Go 1.27.3 measurements for every
reported Pass 2 improvement.

## CPU and allocation evidence

A separate profile run used the same workload. CPU numbers are sampled
self time. `RightTranspose` self time comes from its one-row loop in
this workload. Its inclusive samples also cover the three-row helper and
runtime preemption.

| Scope | Total sampled CPU | Single-row `RightTranspose` self | `rightTransposeThreeRows` self |
| --- | ---: | ---: | ---: |
| Whole process | 12.03 s | 5.64 s (46.88%) | 1.87 s (15.54%) |
| `Session.Prefill` | 2.49 s | 0.41 s (16.47%) | 1.87 s (75.10%) |
| `Session.Decode` | 6.61 s | 5.23 s (79.12%) | 0 |
| Prefill and decode | 9.10 s | 5.64 s (61.98%) | 1.87 s (20.55%) |

`GemmInto` accounts for 8.19 s inclusive across prefill and decode. The FFN
gate, up, and down projection call sites account for 1.86 s, 1.99 s, and
1.87 s inclusive, and the vocabulary projection accounts for 1.73 s,
including 1.32 s during decode. At `product.go:62–63`, the single-row
inner loop accounts for 5.63 s of self samples. The ARM64 disassembly
shows two scalar FP32 loads and an `fmadd` feeding the same accumulator
each iteration. There is one store per output. Both the load pattern and
the accumulator dependency motivate testing a column tile.

The `alloc_space` profile estimates 10,110.47 MiB of cumulative process
allocations. Loading the model accounts for 9,762.94 MiB (96.56%),
including BF16 conversion, retained tensor copies, and the Q/K row
permutation. The retained FP32 model payload is 4,943,257,600 bytes.
Filtering setup out leaves 195.48 MiB on benchmark stacks. `tensor.newTensor`
accounts for 72.65 MiB self. `matrix.FromSlice`, `matrix.New`, and
`Matrix.Values` account for 94.92 MiB self combined. `swiGLU` accounts for
78.79 MiB inclusive and overlaps those adapters. `matrixProductValues`
accounts for 21.75 MiB self. Use unprofiled `-benchmem` totals to compare
allocation bytes and counts. `alloc_space` measures cumulative activity,
while process peak RSS measures resident memory.

## Implementation and correctness in neuralnetwork

Benchmark an output-column tile for `M=1`. Compute two or four dot
products at once, reusing each input activation across that many
contiguous weight-row streams. Give each output an independent FP32
accumulator, with the original ascending contraction order and fused
arithmetic on ARM64. This exposes independent work to the CPU and
reduces input-row loads. Measure the trade-off from reading multiple
weight streams, especially for the vocabulary projection. Start with
portable Go and inspect the ARM64 output if the compiler does not
generate the expected loop. Keep a scalar tail for odd output widths.
Limit the change to the single-row kernel and its dispatch.

- Keep the kernel allocation-free and use call-local state, disjoint
  destinations, and read-only inputs. Exercise repeated calls with
  changed operands and concurrent calls with shared inputs and separate
  outputs. Keep the portable and `purego` builds working.
- Keep the existing bitwise ascending-order tests green. Exercise both
  tile and tail paths with signed finite and cancellation-sensitive
  values, realistic contraction lengths, irregular widths, one and
  multiple input rows, and reused destinations. Cover signed zero,
  NaN, infinities, empty output, and zero contraction. To test the
  optimized path, use shapes that reach its dispatch threshold.
- Keep `GemmInto` validation, destination atomicity on error, and alpha
  and beta behavior, including zero beta with supplied validated bias.
  Run `internal/float32product/product_test.go`, `tensor/products_test.go`,
  and `matrix/matrix_test.go` via the normal, race, and
  `CGO_ENABLED=0 -tags=purego` suites. Run the Metal suite on a supported
  host if its dispatch or shared fallback changes.

### Kernel and public API benchmarks

Reuse `tensor/right_transpose_benchmark_test.go`. It already has a
preallocated-output kernel benchmark and a public `GemmInto` benchmark
with Llama's config, nil bias, and deterministic signed inputs initialized
outside the timer.

| Projection | Weight shape `[N,K]` | Input rows |
| --- | --- | --- |
| Q and attention output | `[2048,2048]` | 1, 6 |
| K and V | `[512,2048]` | 1, 6 |
| FFN gate and up | `[8192,2048]` | 1, 6 |
| FFN down | `[2048,8192]` | 1, 6 |
| Tied vocabulary | `[128256,2048]` | 1 for cached generation, 6 for full-prefix regression |

From neuralnetwork, collect at least five samples per case on `v0.10.11`
and on the candidate using the same Go 1.27.3 settings. Include the small
and irregular cases and add a one-row tail case if the existing suite
misses it. Compare latency, bytes, and allocations with `benchstat`.

```sh
GOTOOLCHAIN=go1.27.3 GOARM64=v8.0 CGO_ENABLED=1 GOMAXPROCS=8 \
  go test ./tensor -run '^$' -bench '^BenchmarkRightTranspose$' -benchmem -count=5
```

## Llama reproduction and acceptance

Work from the same Llama source on both sides. Use
[`LLAMA_32_1B_MODEL_DIR` and the pinned snapshot](testdata/README.md#llama-32-1b-transformers-reference).
Once the candidate exists, use an alternate modfile pointing to its local
checkout. This keeps the tracked dependency pin out of the experiment.

```sh
export GOTOOLCHAIN=go1.27.3 GOARM64=v8.0 CGO_ENABLED=1 GOMAXPROCS=8
bench_dir=$(mktemp -d)
go version
go test -mod=readonly -c -o "$bench_dir/before.test" .
go version -m "$bench_dir/before.test"

# After implementing the candidate, set nn_checkout to its absolute path.
cp go.mod "$bench_dir/candidate.mod"
cp go.sum "$bench_dir/candidate.sum"
go mod edit -modfile="$bench_dir/candidate.mod" \
  -replace="github.com/itsmontoya/neuralnetwork=$nn_checkout"
go test -mod=readonly -modfile="$bench_dir/candidate.mod" \
  -c -o "$bench_dir/after.test" .
go version -m "$bench_dir/after.test"
```

Run five one-operation fresh processes per binary. Alternate their order
to limit drift. On macOS, this `wait4` wrapper records wall time and peak
RSS in bytes. Run from the Llama root for fixture lookup.

```sh
measure() {
  python3 - "$@" <<'PY'
import os
import subprocess
import sys
import time

start = time.perf_counter()
process = subprocess.Popen(sys.argv[1:])
_, status, usage = os.wait4(process.pid, 0)
process.returncode = os.waitstatus_to_exitcode(status)
print(f"wall_seconds={time.perf_counter() - start:.6f} peak_rss_bytes={usage.ru_maxrss}")
sys.exit(process.returncode)
PY
}
for trial in 1 2 3 4 5; do
  if [ "$((trial % 2))" -eq 1 ]; then
    order="before after"
  else
    order="after before"
  fi
  for build in $order; do
    measure "$bench_dir/$build.test" \
      -test.run '^$' -test.bench '^BenchmarkLlama32CachedGeneration$' \
      -test.benchtime 1x -test.benchmem -test.timeout 15m -test.v \
      >"$bench_dir/$build-$trial.log" 2>&1 || exit 1
  done
done

for build in before after; do
  "$bench_dir/$build.test" \
    -test.run '^$' -test.bench '^BenchmarkLlama32CachedGeneration$' \
    -test.benchtime 1x -test.benchmem -test.timeout 15m -test.v \
    -test.cpuprofile "$bench_dir/$build.cpu.pprof" \
    -test.memprofile "$bench_dir/$build.heap.pprof" || exit 1
  pprof -top "$bench_dir/$build.cpu.pprof"
  for stage in Prefill Decode; do
    pprof -top -focus="\\(\\*Session\\)\\.$stage" -relative_percentages \
      "$bench_dir/$build.cpu.pprof"
  done
  pprof -list='float32product\.(RightTranspose|rightTranspose)' \
    "$bench_dir/$build.cpu.pprof"
  pprof -disasm='float32product\.(RightTranspose|rightTranspose)' \
    "$bench_dir/$build.cpu.pprof"
  pprof -top -cum -alloc_space "$bench_dir/$build.heap.pprof"
  for metric in alloc_space alloc_objects; do
    pprof -top "-$metric" -relative_percentages -focus=BenchmarkLlama32CachedGeneration \
      -ignore='LoadHFModel|LoadLlama3|verifyLlama32Snapshot' "$bench_dir/$build.heap.pprof"
  done
done
```

Use standalone `pprof` on this Go installation. Compare unprofiled
medians and ranges for prefill, decode, full generation, allocations and
bytes, and process peak RSS. Calculate decode seconds/token as decode
seconds/op divided by seven, and generation throughput as eight divided
by seconds/op. Report improvement percentages from Go 1.27.3 before to
Go 1.27.3 after. Confirm with the separate profiles that decode projection
CPU fell and check for regressions in the multi-row cases and prefill.

With the official snapshot and pinned TinyStories 260K artifacts enabled,
run `go test -p=1 -count=1 -timeout 45m -v ./...` on both builds. Pass
`-modfile="$bench_dir/candidate.mod"` for the candidate. Repeat with
`CGO_ENABLED=0` and `-tags=purego`. Run `go vet ./...` and
`go test -race ./...` with the Llama 3.2 model environment variable
unset for the race run. The real-model tests verify sampled Transformers
logits and greedy winners (`atol=rtol=1e-3`), cached/full-prefix parity
across all 128,256 logits at five positions (`atol=rtol=1e-4`), checkpoint
inventory, tokenizer parity, and cached and uncached CLI output. The
expected suffix IDs are `[12366, 13, 1102, 374, 279, 1455, 95551, 3363]`.
Expected CLI stdout is ` Paris. It is the most populous city` followed by
one newline. The synthetic tests cover retained outputs, retries, and
concurrent sessions.

Keep the change focused on the measured kernel. Commit independently
useful benchmark or reference coverage separately when it can stand on
its own. Keep kernel dispatch, implementation, and tests of that
dispatch together when they depend on one another. Stop after validating
and remeasuring the single-row change.

## Ranked candidates for Optimization Pass 3

Re-rank these with the post-change profile and an explicit latency or
memory objective.

1. **Three-input-row prefill kernel (neuralnetwork).** It takes 1.87 s
   self, 75.10% of sampled prefill CPU. Keep weight-load reuse when
   investigating first-token latency. This profile uses six prompt tokens.
2. **Elementwise and SwiGLU allocations (neuralnetwork and Llama).**
   Matrix adapter allocations account for 94.92 MiB self and SwiGLU
   for 78.79 MiB inclusive, with overlapping stacks. Zeroing newly
   allocated memory takes 0.91 s self on decode stacks. Trace the
   allocation sites and assess direct host tensor arithmetic first.
3. **Checkpoint conversion and copies (Llama and neuralnetwork).**
   Model load takes 1.894 s at the median and accounts for 9,762.94 MiB
   of cumulative allocations. Measure load latency and peak RSS while
   preserving validated, privately owned weights.
4. **Host GEMM result staging (neuralnetwork).** The staging allocation
   accounts for 21.75 MiB self. Evaluate reuse alongside the copy and
   coefficient pass while preserving the destination contract.
5. **Longer-context attention and KV access (Llama).** Prefill attention
   accounts for 7.84 MiB inclusive and cached attention for 3.02 MiB
   inclusive. Measure longer prompts and cache histories before choosing
   a change.
