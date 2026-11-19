# Optimization Pass 2 — speed up single-row FP32 projections

## Integration result

`neuralnetwork v0.10.12` (`1efde389dc71f21b37580d4325d802ce91c23620`)
implements a four-column tile for `M=1` and uses the original dot product
for the tail. Across five alternating fresh-process trials per build on
Go 1.27.3, median decode fell from 1.117 to 0.327 s/token (−70.8%) and
cached eight-token generation fell from 10.102 to 4.107 s/op (−59.35%).
The comparison uses a fresh v0.10.11 run from the same session. The earlier
Go 1.27.3 pre-optimization measurements are recorded below. [Full benchmark results](README.md#optimization-pass-2-go-1273)

One separate CPU and allocation profile per build used the same workload.
Times below are sampled CPU. The single-row entries use self time.

| CPU profile scope | v0.10.11 | v0.10.12 |
| --- | ---: | ---: |
| Process total | 11.64 s | 6.35 s |
| Prefill and decode total | 8.71 s | 3.48 s |
| Single-row kernel, prefill and decode | 5.51 s | 1.74 s |
| Single-row kernel, decode | 5.14 s | 1.68 s |
| Single-row kernel, prefill classifier | 0.37 s | 0.06 s |
| Three-row kernel self, prefill | 1.66 s | 1.24 s |

The single-row work in the decode profile fell by 67.3%. Prefill also
uses the single-row kernel for its final vocabulary projection. The
three-row sample totals varied across the two profile runs. Both builds
measured 198,718,008 bytes/op and 39,866 allocs/op at the median in the
unprofiled trials. The v0.10.12 allocation profile estimates 189.08 MiB
on generation stacks and 9,760.14 MiB under checkpoint loading.

Both versions passed the full Llama test suite in default and `purego`
builds with the pinned Llama 3.2 and TinyStories checkpoints. With v0.10.12,
all five cached logit rows still matched their full-prefix rows exactly,
the maximum sampled Transformers logit error was `4.95910645e-05`, and
both CLI builds printed the frozen continuation byte for byte. Llama
passed `go vet`. The race and Metal-tagged suites passed with synthetic
and TinyStories coverage. The dependency's float32 product, tensor, and matrix
suites passed in default, `purego`, race, and Metal-tagged builds.

## Target and ownership

Pass 2 targeted the single-input-row case of `float32product.RightTranspose` in
`github.com/itsmontoya/neuralnetwork`. Seven cached decode steps and the
classifier at the end of prefill use this case. Pass 1 already changed the
kernel to read contiguous weight rows and reuse each weight load across three
input rows.

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

## Kernel design and correctness requirements

The portable Go implementation computes four dot products at once for
`M=1`, reusing each input activation across four contiguous weight-row
streams. Each output uses an independent FP32 accumulator in the original
ascending contraction order. Go 1.27.3 emits scalar fused multiply-adds
on ARM64. A scalar tail covers the remaining columns.

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

From neuralnetwork, use the same Go 1.27.3 settings to collect at least
five samples per case on `v0.10.11` and `v0.10.12`. The benchmark-only
commit `66eadf8` adds the single-row tail case for the baseline. Include
the small and irregular cases. Compare latency, bytes, and allocations
with `benchstat`.

```sh
GOTOOLCHAIN=go1.27.3 GOARM64=v8.0 CGO_ENABLED=1 GOMAXPROCS=8 \
  go test ./tensor -run '^$' -bench '^BenchmarkRightTranspose$' -benchmem -count=5
```

## Reproduce the Llama comparison

Build both binaries from the same Llama checkout after integration. Use an
alternate modfile from the pre-integration commit for v0.10.11. Set
[`LLAMA_32_1B_MODEL_DIR` to the pinned snapshot](testdata/README.md#llama-32-1b-transformers-reference).

```sh
export GOTOOLCHAIN=go1.27.3 GOARM64=v8.0 CGO_ENABLED=1 GOMAXPROCS=8
bench_dir=$(mktemp -d)
go version
git show bccba37:go.mod > "$bench_dir/before.mod"
git show bccba37:go.sum > "$bench_dir/before.sum"
go test -mod=readonly -modfile="$bench_dir/before.mod" \
  -c -o "$bench_dir/before.test" .
go version -m "$bench_dir/before.test"
go test -mod=readonly -c -o "$bench_dir/after.test" .
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
`-modfile="$bench_dir/before.mod"` for v0.10.11. Repeat with
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

## Ranked candidates for Optimization Pass 3

The v0.10.12 profile puts the three-row kernel first for first-token
latency and the tiled single-row kernel first for ongoing decode.

1. **Three-input-row prefill kernel (neuralnetwork).** It takes 1.24 s
   self, 80.0% of sampled prefill CPU on the six-token prompt. Preserve
   weight-load reuse while evaluating first-token latency.
2. **Remaining single-row projection cost (neuralnetwork).** The tiled
   kernel still takes 1.68 s self, 87.0% of sampled decode CPU. Measure
   sustained memory bandwidth on the FFN and vocabulary shapes to assess
   how much headroom remains for further kernel work.
3. **Elementwise and SwiGLU allocations (neuralnetwork and Llama).**
   Matrix adapter allocations account for 82.03 MiB self and SwiGLU
   for 75.83 MiB inclusive, with overlapping stacks. Assess how much
   direct host tensor arithmetic can reduce this allocation traffic.
4. **Checkpoint conversion and copies (Llama and neuralnetwork).**
   Model load takes 1.803 s at the median and accounts for 9,760.14 MiB
   of cumulative allocations. Measure load latency and peak RSS while
   preserving validated, privately owned weights.
5. **Host GEMM result staging (neuralnetwork).** The staging allocation
   accounts for 20.12 MiB self. Evaluate reuse alongside the copy and
   coefficient pass while preserving the destination contract.
6. **Longer-context attention and KV access (Llama).** Prefill attention
   accounts for 7.84 MiB inclusive and cached attention for 3.53 MiB
   inclusive. Measure longer prompts and cache histories before choosing
   a change.
