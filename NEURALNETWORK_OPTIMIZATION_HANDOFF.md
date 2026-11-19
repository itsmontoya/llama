# Optimization Pass 4 neuralnetwork handoff

## Objective and ownership

Remove the copy-and-stage adapter from equal-shape FP32 tensor arithmetic on
host storage. Write each element directly to the validated caller-supplied
destination and reuse the existing host arithmetic kernels where practical.
Llama exercises `tensor.AddInto` and `tensor.MulInto`. `tensor.SubInto` uses
the same adapter and can share the replacement if that keeps the change
cohesive. This is one **neuralnetwork-owned** optimization.

Llama already supplies a disjoint tensor to the `Into` API. The redundant
payloads are created inside neuralnetwork after validation. The tensor
package owns the storage and can operate on it privately. Public accessors
copy values as part of their ownership contract. The optimized dependency
should work with the existing Llama call sites and API.

Pass 4 is paused pending this upstream change. The Llama baseline is
`3a9ce0d4ee556fdc8eaa3aeaf5cb5dc14905696d`. The dependency baseline is
`v0.10.13` at `a426793e8af0d6064730c2d2e76c23cabc0981cc`.
Use Go 1.27.3 for before and after measurements.

## Baseline and selection evidence

Seven unprofiled fresh processes on Apple M3, macOS 26.5.2 (25F84),
darwin/arm64, used default tags, `GOARM64=v8.0`, `CGO_ENABLED=1`, and
`GOMAXPROCS=8`. Each process did one benchmark iteration with zero inference
warm-ups. The pinned workload starts with six prompt tokens, produces eight
tokens with one prefill and seven decode calls, and uses a context cap of
512. Each iteration checked the frozen continuation and decoded bytes.
Process wall time and RSS include snapshot verification, model loading,
and tokenizer setup. Timing varied across processes, including a split
between roughly 0.7 s and 1.4–1.6 s for first token.

| Primary metric | Median | Observed range |
| --- | ---: | ---: |
| Prefill (s) | 1.360 | 0.6702–1.622 |
| First token (s) | 1.361 | 0.6703–1.622 |
| Decode (s/token) | 0.3334 | 0.3279–0.4911 |
| Decode (tokens/s) | 2.999 | 2.036–3.050 |
| Cached generation (s/op) | 3.754 | 2.981–4.143 |
| Output (tokens/s) | 2.131 | 1.931–2.684 |
| Generation B/op | 198,717,992 | 198,716,624–198,723,344 |
| Generation allocs/op | 39,866 | 39,861–39,872 |
| Process wall (s) | 7.337 | 5.957–8.113 |
| Peak RSS (GiB) | 6.138 | 5.694–6.373 |

Two fresh primary profile processes ran three generations each using a
64 KiB heap sampling rate. The filtered stack totals and whole-run sampled
CPU are below. Inclusive paths overlap. Allocation bytes and objects are
sampling estimates. Use the unprofiled benchmark for exact per-op totals.

| Primary stack or metric | Samples across six generations |
| --- | ---: |
| All generation allocation bytes after filtering setup | 1,139.02 MiB |
| `llama.elementwise` inclusive | 589.53 MiB (51.8%) |
| `llama.swiGLU` inclusive | 471.06 MiB (41.4%) |
| `tensor.matrixBinaryInto` inclusive | 394.93 MiB (34.7%) |
| `tensor.matrixBinaryInto` prefill / decode | 178.74 / 216.19 MiB |
| `tensor.matrixBinaryInto` sampled objects | 32,341 of 156,637 (20.6%) |
| `tensor.matrixProductValues` GEMM staging | 137.56 MiB (12.1%) |
| `tensor.activationFloat32Into` inside sigmoid | 116.52 MiB (10.2%) |
| `LoadHFModel` including setup | 19,516.87 of 20,949.99 MiB (93.2% of whole process) |
| `Session.Prefill` CPU | 3.92 s |
| `Session.Decode` CPU | 12.88 s |
| `rightTransposeThreeRows` prefill CPU self | 2.36 s (60.2% of prefill) |
| `rightTransposeOneRow` decode CPU self | 7.82 s (60.7% of decode) |
| `runtime.memclrNoHeapPointers` prefill / decode CPU self | 0.68 / 4.12 s |
| `tensor.matrixBinaryInto` CPU inclusive | 0.43 s, including 0.35 s in zeroing |
| Whole process sampled CPU | 22.49 s |

The two primary runs independently attributed 197.86 and 197.07 MiB to
the adapter. Within its stacks, about 260 MiB went to the two
`matrix.FromSlice` calls and 135 MiB to `matrix.New`. More than half of
its allocation bytes came from the two SwiGLU multiplies. The standalone
adapter measurements below include allocator and copy costs when inputs
and destination are already available. The primary CPU profile directly
attributes about 72 ms per generation to the adapter stack, including
about 58 ms of sampled zeroing. Some other allocator activity appears
elsewhere in runtime stacks. An end-to-end timing improvement remains to
be established.

### Supplemental 29-token prefill

The Pass 3 supplemental prompt, checkpoint, and context cap were reused.
Three fresh unprofiled processes timed only prefill and first-token
selection, with a median of 4.621 s (4.543–5.244), 433,743,208 B/op, and
6,735 allocs/op. All chose token `12366`. Their SHA-256 over the 128,256
little-endian FP32 logits was
`6ffb4dc93003480aebdd51e9f1fee47ae3836eea9fb35ce2508f5d618a9871a5`.
A separate three-iteration profile attributed 435.79 of 1,242.62 MiB
(35.1%) to the adapter with 0.07 s sampled CPU, including 0.06 s zeroing
and 0.01 s copying. Prefill had 12.45 sampled CPU seconds, including
5.86 s self in the three-row kernel and 4.13 s self in
`RightTranspose` with its two leftover rows. Keep these results separate
from the primary workload.

### Alternatives considered

- **Projection kernels.** They dominate CPU. For all primary one-row
  projection stacks, 4.99 of 8.89 sampled seconds inclusive were in the
  FFN gate, up, and down calls and 2.99 s were in the vocabulary projection.
  The ARM64 inner loop uses five scalar loads and four scalar FMADDs per
  step across four ordered accumulators. The real output widths are all
  divisible by four. The one-row kernel overwrites every output and
  allocates zero bytes. Repeated raw one-row kernel medians were 0.291 ms
  for K/V, 1.806 ms for Q/output, 4.590/4.352 ms for FFN gate or up/down,
  and 70.072 ms for vocabulary. The FFN weights are 64 MiB each and the
  vocabulary matrix is about 1 GiB. The FFN and vocabulary timings imply
  about 15 GB/s of logical weight reads. Those timings and disassembly
  leave both instruction throughput and weight traffic in play. They
  provide no hardware-counter evidence to separate cache misses, bandwidth
  saturation, or accumulator latency. Tail CPU is a separate opportunity
  for the longer prompt. The adapter has a directly attributable amount of
  removable traffic and a smaller numerical risk than a new product kernel.
- **SwiGLU destinations and sigmoid adapter.** Llama creates three
  destination tensors around sigmoid and the two multiplies. About 1.61 s
  of primary profile CPU fell under allocation of the sigmoid destination
  in `llama.unary`, outside the selected adapter. The separate dependency
  activation adapter contributed 116.52 MiB. Reusing Llama destinations
  would still leave the binary copies and would require scratch lifetimes
  across calls and distinct trace handling. Fusing sigmoid and the two
  multiplies would add a new operation and require demonstrating identical
  rounding and live output behavior. The selected path addresses a larger
  isolated allocation source across SwiGLU, normalization, residuals, and
  rotary addition.
- **GEMM staging and model loading.** GEMM staging is a smaller generation
  allocation source and has its own failure and device contracts. The model
  loader dominates cold-process allocation and includes BF16 conversion and
  owned model storage. Work there primarily addresses startup. Keep the
  shared elementwise optimization focused on steady-state inference.

## Exact call path and current behavior

1. In Llama, `(*Session).Prefill` calls
   `(*Decoder).forwardPrefix`, which invokes `(*Decoder).forwardBlock`
   in `decoder.go`. `(*Session).Decode` calls `forwardBlock`
   directly for each cached token. Both reach `(*Session).commit` and
   `(*Decoder).projectLogits`, which performs a final one-row `rmsNorm`.
   Full-prefix `Forward` and `ForwardWithTrace` use the same block code.
   [session.go](session.go) · [decoder.go](decoder.go)
2. `forwardBlock` uses `tensor.AddInto` for its two residuals.
   `normalization.go` uses `tensor.MulInto` for
   `rmsNorm`'s square and for both SwiGLU products, `gate * sigmoid(gate)`
   and `silu * up`. `rotary.go` uses `tensor.AddInto` to combine
   the two rotary terms for Q and K. The helper `elementwise` in
   `tensor_ops.go` uses `BroadcastShape` and `tensor.New`
   to allocate each operation's destination before calling the operator.
   [normalization.go](normalization.go) · [rotary.go](rotary.go) · [tensor_ops.go](tensor_ops.go)
3. In neuralnetwork `tensor/arithmetic.go`, `AddInto`, `SubInto`, and
   `MulInto` pass a matrix method into `binaryFloat32Into`. After tensor
   type, broadcast shape, destination shape and alias validation, equal
   nonempty shapes take `matrixBinaryInto`. Its calls to
   `matrix.FromSlice` in `matrix/matrix.go` each allocate owned storage
   and copy a flattened input. `matrix.New` zero-fills a third buffer.
   `(*Matrix).AddInto`, `SubtractInto`, or `MultiplyElementsInto`
   computes into it. `(*Matrix).ValuesInto` then copies to the supplied
   tensor. This round trip allocates three payloads, copies both inputs and
   the output, and constructs three matrix headers for a single binary op.
4. Matrix host dispatch lives in `matrix/elementwise_arm64.go`,
   `matrix/elementwise_amd64.go`, `matrix/elementwise_default.go`, and
   `matrix/elementwise_pure.go`. The adapters construct ordinary host
   matrices with no active device execution. `matrix.AddInto` can
   dispatch to a device for genuinely execution-bound Matrix callers.
   Preserve that behavior.

## Desired behavior and implementation direction

Keep the public API and the existing validation sequence, including for
scalars and empty results. For a validated nonempty equal-shape operation,
apply a non-failing slice kernel to private tensor storage, writing
directly into `destination.floats`. Consider factoring only the shared
host add, subtract, and multiply kernels into a dependency-internal
package used by Matrix and Tensor. Preserve the existing SIMD dispatch,
length thresholds, and scalar fallback wherever the extraction supports
them. Start with the narrowest shared implementation. Benchmark a simple
contiguous loop if sharing the existing kernels proves intrusive, and
check its performance on wide operations. The evidence, tests, and
benchmarks should determine the final implementation.

Keep broadcasting, DivInto, unary activations, GEMM, and projection math
outside this optimization. For example, `[m,2048] * [2048]` in RMS norm
uses unequal shapes and should retain its existing path. There is no
need to expose tensor payloads, add a public borrowed Matrix constructor,
or retain scratch buffers for the equal-shape case.

## Numerical, ownership, and concurrency requirements

- Preserve the existing float32 operation and operand order per element,
  including signed zero, infinities, subnormals, and NaN behavior under
  the current tests and API. Keep finite values and signed zeros bitwise
  identical for an equivalent build. Respect the existing NaN comparison
  contract. Preserve SIMD tails and pure Go behavior. The optimization
  leaves sigmoid and both FP32 SwiGLU multiply boundaries in their current
  order. Product accumulation order and tolerances stay as pinned.
- Keep `Into` destination identity, shape metadata, and backing allocation
  stable. Validate dtype, shape, and disjoint storage before any write,
  including shallow tensor aliases and empty destinations. All returned
  errors must leave destination bytes intact. Identical left and right
  inputs remain valid, including the RMS square.
- Inputs remain read-only. Outputs remain owned by callers and survive
  later operations, session resets, failed steps, and retries. Preserve
  the copying behavior of public Matrix constructors and Tensor accessors.
  Preserve Llama's independent trace and returned logit tensors and its
  caller-owned session logit slices.
- Concurrent calls may share read-only inputs with distinct destinations.
  Keep all implementation state local to the operation. If a different
  design genuinely requires scratch, account for capacity, peak live bytes,
  reuse, and release. A pool that removes cumulative allocations by
  retaining their equivalent payloads would miss the objective.

## Representative shapes and allocation targets

Llama 3.2 1B has 16 layers, hidden width 2048, FFN width 8192, 32 query
heads, eight KV heads, head width 64, and vocabulary 128256. Use row counts
`m=1,6,29`. Residual and RMS-square operations use `[m,2048]`. SwiGLU
multiplies use `[m,8192]`. Rotary addition uses `[m,32,64]` and
`[m,8,64]`. These apply 8 equal-shape operations per layer, plus the
single-row final RMS square per inference step. Six prompt rows plus seven
decoded rows produce 1,032 adapter calls per generation. The three
payloads total **67,928,064 bytes** by shape, before Matrix metadata.
The 29-token prefill accounts for 151,117,824 payload bytes by the same
calculation. Both agree with the approximately 35% sampled share. The
calculation estimates the removable allocation budget.

The preallocated-destination microbenchmarks should reach 0 B/op and
0 allocs/op on these shapes. The caller's destination and Llama's public
outputs remain allocated as required. Large-rank validation metadata,
if needed, is distinct from adapter payloads. With six measured
allocations per adapter call the expected reduction is about 6,192
objects per primary generation. Compare this estimate with exact
unprofiled B/op and allocs/op and with filtered `alloc_space` and
`alloc_objects` profiles. Record peak RSS and live memory if any
storage is retained. Whole-process RSS is largely set by model loading.
The test process writes its heap profile after the benchmark returns.
Its live-heap snapshot is an end-of-run observation. Use process peak
RSS and an in-operation live-heap measurement if an implementation retains
working storage.

### Dependency microbenchmark baseline

Five independent samples used 200 ms per subbenchmark. Inputs were
deterministic finite FP32, and disjoint destinations were reused.
Times are medians in microseconds with the observed ranges.

| Operation and shape | µs/op | Range | B/op | allocs/op |
| --- | ---: | ---: | ---: | ---: |
| Add 1×2048 | 2.101 | 2.087–2.299 | 24,720 | 6 |
| Add 6×2048 | 9.967 | 9.855–10.085 | 147,600 | 6 |
| Add 29×2048 | 45.434 | 43.658–46.684 | 712,849 | 6 |
| Multiply 1×2048 | 2.138 | 2.095–2.160 | 24,720 | 6 |
| Multiply 6×2048 | 10.087 | 10.011–10.133 | 147,600 | 6 |
| Multiply 29×2048 | 46.244 | 44.725–46.872 | 712,849 | 6 |
| Multiply 1×8192 | 7.038 | 6.970–7.113 | 98,448 | 6 |
| Multiply 6×8192 | 39.483 | 38.226–40.108 | 589,968 | 6 |
| Multiply 29×8192 | 137.797 | 134.627–139.203 | 2,850,965 | 6 |
| Sigmoid 1×8192 (comparison) | 56.326 | 55.727–56.639 | 98,400 | 5 |
| Sigmoid 6×8192 (comparison) | 338.133 | 335.263–342.137 | 589,920 | 5 |
| Sigmoid 29×8192 (comparison) | 1,599.071 | 1,582.006–1,606.223 | 2,850,913 | 5 |

## Reproduce and compare

All recipes use the same default environment unless a validation variant
specifies otherwise. Set the model variable to the verified Meta
checkpoint at revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`.
[Snapshot verification](testdata/README.md#llama-32-1b-transformers-reference)

Run from the Llama root. The first binary uses the published baseline
dependency. A local replacement in a separate modfile is suitable for
testing your upstream commit before release. Set `nn_checkout` to the
absolute path of the upstream candidate checkout before running these commands.

```sh
export GOTOOLCHAIN=go1.27.3 GOARM64=v8.0 GOMAXPROCS=8 CGO_ENABLED=1 GOFLAGS=
bench_dir=$(mktemp -d)
baseline=3a9ce0d4ee556fdc8eaa3aeaf5cb5dc14905696d
git show "$baseline:go.mod" > "$bench_dir/before.mod"
git show "$baseline:go.sum" > "$bench_dir/before.sum"
cp "$bench_dir/before.mod" "$bench_dir/after.mod"
cp "$bench_dir/before.sum" "$bench_dir/after.sum"
go mod edit -modfile="$bench_dir/after.mod" \
  -replace="github.com/itsmontoya/neuralnetwork=$nn_checkout"
for build in before after; do
  go test -mod=readonly -modfile="$bench_dir/$build.mod" -c -o "$bench_dir/$build.test" .
  go version -m "$bench_dir/$build.test"
done
git -C "$nn_checkout" rev-parse HEAD
```

Run seven alternating before/after pairs, reversing pair order each time.
Each invocation starts a fresh process. On macOS use `/usr/bin/time -l`
or a parent that records elapsed wall time and Darwin `wait4`
`ru_maxrss` in bytes. The figures above use the latter. Divide
`decode-s/op` by seven and divide eight by generation seconds/op.
Compare medians, observed ranges, and paired differences.

```sh
# Set build to before or after for each invocation.
/usr/bin/time -l "$bench_dir/$build.test" -test.run '^$' \
  -test.bench '^BenchmarkLlama32CachedGeneration$' -test.benchtime 1x \
  -test.benchmem -test.timeout 15m -test.v

for trial in 1 2; do
  "$bench_dir/$build.test" -test.run '^$' \
    -test.bench '^BenchmarkLlama32CachedGeneration$' -test.benchtime 3x \
    -test.benchmem -test.timeout 15m -test.v -test.memprofilerate 65536 \
    -test.cpuprofile "$bench_dir/$build-$trial.cpu.pprof" \
    -test.memprofile "$bench_dir/$build-$trial.heap.pprof"
done
for mode in alloc_space alloc_objects; do
  go tool pprof "-$mode" -top -cum -nodecount=50 \
    -focus='BenchmarkLlama32' -ignore='LoadHFModel|LoadLlama3|verifyLlama32Snapshot' \
    "$bench_dir/$build.test" "$bench_dir/$build-1.heap.pprof" "$bench_dir/$build-2.heap.pprof"
done
for path in '\(\*Session\)\.Prefill' '\(\*Session\)\.Decode' matrixBinaryInto; do
  go tool pprof -top -focus="$path" "$bench_dir/$build.test" \
    "$bench_dir/$build-1.cpu.pprof" "$bench_dir/$build-2.cpu.pprof"
done
```

For allocation attribution, also use `-focus=matrixBinaryInto` on the
heap profiles and `-list='tensor\.matrixBinaryInto'` on the baseline.
Record before/after absolute allocation totals and sampled CPU per
iteration. Confirm that eliminated adapter work did not move to a new
allocation stack. Inspect heap setup stacks separately and measure RSS in
the unprofiled processes.

### Microbenchmarks

Reuse this public-API workload in the dependency's tensor benchmarks
with package `tensor_test`, or run it in Llama with package `llama` and
an overlay. Add SubInto for the same-shape case if you change the shared
subtract dispatch. The existing tensor `BenchmarkAddInto` checks
broadcast and small shapes. Run it as a regression check along with
the new shape benchmark. For projection context the baseline used
`BenchmarkRightTranspose` subtests `Kernel` and `GemmInto` for
`QAndOutput`, `KAndV`, `FFNGateAndUp`, `FFNDown`, and `Vocabulary`
at `M1`, with three samples of three iterations.

```go
package tensor_test

import (
	"fmt"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func BenchmarkPass4TensorArithmetic(b *testing.B) {
	for _, operation := range []struct {
		name  string
		width int64
		call  func(left, right, dst *tensor.Tensor) error
	}{
		{"Add", 2048, tensor.AddInto},
		{"Multiply", 2048, tensor.MulInto},
		{"Multiply", 8192, tensor.MulInto},
		{"Sigmoid", 8192, func(left, _, dst *tensor.Tensor) error {
			return tensor.SigmoidInto(left, dst)
		}},
	} {
		for _, rows := range []int64{1, 6, 29} {
			b.Run(fmt.Sprintf("%s/%dx%d", operation.name, rows, operation.width), func(b *testing.B) {
				shape := []int64{rows, operation.width}
				input := func(multiplier int) *tensor.Tensor {
					values := make([]float32, rows*operation.width)
					for i := range values {
						values[i] = float32((i*multiplier)%127-63) / 8
					}
					result, err := tensor.FromFloat32(shape, values)
					if err != nil {
						b.Fatal(err)
					}
					return result
				}
				left, right := input(3), input(5)
				dst, err := tensor.New(tensor.Float32, shape)
				if err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				for b.Loop() {
					if err := operation.call(left, right, dst); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
```

Run `-test.run '^$' -test.bench '^BenchmarkPass4TensorArithmetic$'
-test.benchmem -test.benchtime 200ms -test.count 1` in five fresh
processes per build. Alternate before and after. Use the same compiler,
tags, hardware, and GOGC settings in both builds.

### Supplemental prefill benchmark

Add the following via an overlay to the Llama test build. Map an unused
absolute Llama path ending in `_test.go` to the absolute temporary
source path in a JSON file of the form `{"Replace":{"from":"to"}}`.
Compile the same before and after commands with `-overlay` added.
Use `BenchmarkLlama32SupplementalPrefill` with three fresh one-iteration
processes per build and one three-iteration profile per build. Keep
the primary results separate.

```go
package llama

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/itsmontoya/llama/internal/tokenizer"
)

func BenchmarkLlama32SupplementalPrefill(b *testing.B) {
	dir := os.Getenv("LLAMA_32_1B_MODEL_DIR")
	if dir == "" {
		b.Fatal("supplemental prefill requires the pinned snapshot")
	}
	verifyLlama32Snapshot(b, dir)
	const prompt = "France is a country in Western Europe. It shares borders with Belgium, Germany, Switzerland, Italy, and Spain. The capital of France is"
	model, settings, err := LoadHFModel(dir, 512)
	if err != nil {
		b.Fatal(err)
	}
	vocab, err := tokenizer.LoadLlama3(dir, model.Config().VocabSize, settings.BOSID, settings.StopIDs)
	if err != nil {
		b.Fatal(err)
	}
	ids, err := vocab.Encode(prompt)
	if err != nil || len(ids) != 29 {
		b.Fatalf("encode: %v, %v", ids, err)
	}
	var logits []float32
	var winner int64
	for b.Loop() {
		b.StopTimer()
		session, err := model.NewSession(int64(len(ids)))
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		logits, err = session.Prefill(ids)
		if err != nil {
			b.Fatal(err)
		}
		winner, err = greedyToken(logits)
		if err != nil {
			b.Fatal(err)
		}
	}
	payload := make([]byte, 4*len(logits))
	for i, value := range logits {
		binary.LittleEndian.PutUint32(payload[4*i:], math.Float32bits(value))
	}
	b.Logf("greedy=%d FP32 logits SHA256=%x", winner, sha256.Sum256(payload))
}
```

## Tests and integration back into Llama

- Run the neuralnetwork default suite and relevant purego and race suites,
  plus `go vet ./...`. Run `go test -tags=metal ./tensor ./matrix` on
  supported darwin/arm64 hardware. Keep `Test_Arithmetic_EqualShapeMatrixBits`,
  `Test_Arithmetic_EqualShapeConcurrentReaders`, the arithmetic argument,
  alias and output-lifetime tests, and the broadcast tests green.
  `Test_MatrixBinaryInto_OperationError` injects failure through the
  private adapter. If the direct path removes the adapter, preserve that
  check for any remaining fallible path and verify the public
  argument-error atomicity tests against the new path.
- Add a regression that detects renewed payload allocation on the real
  preallocated destination shapes. Extend bitwise coverage for the
  representative shapes and vector tails if existing cases miss a
  modified dispatch path. Include shared-input concurrent reads, identical
  operands, disjoint retained outputs, and retry after rejected destination
  aliases if the existing tests do not cover the path.
- Test both Llama builds using the pinned snapshot and verified TinyStories
  artifacts. From the Llama root, export `LLAMA_32_1B_MODEL_DIR`,
  `LLAMA_STORIES260K_MODEL`, and `LLAMA_STORIES260K_TOKENIZER`. Run
  `go test -mod=readonly -p=1 -count=1 -timeout 45m -v ./...` with the
  respective modfile. For the purego build, set `CGO_ENABLED=0` and add
  `-tags=purego`. Run `go vet ./...` and
  `go test -race -p=1 -count=1 ./...` with the same modfile, default CGO,
  and the 1B model variable unset for the race run. The race run should
  exercise concurrent decoder and session calls, output ownership and retry.
  [TinyStories fixtures](README.md#generate-text-with-tinystories-260k)
- Keep `TestLlama32Logits` Transformers tolerances at `atol=rtol=1e-3`
  with every frozen greedy winner. `TestLlama32CachedParity` compares
  five full logit rows at `atol=rtol=1e-4`. Preserve synthetic MHA/GQA/MQA
  traces and TinyStories checks. Cached and uncached `TestLlama32Generation`
  must preserve `[12366, 13, 1102, 374, 279, 1455, 95551, 3363]` and
  stdout equal to ` Paris. It is the most populous city` with one newline.
  Build and check the default and purego CLI binaries with
  `-hf-model-dir "$LLAMA_32_1B_MODEL_DIR" -context 512
  -prompt 'The capital of France is' -max-new-tokens 8` in both modes
  (add `-uncached` for uncached), comparing output bytes and exit status.
  The supplemental winner and logits digest should match as well.

## Commit discipline and completion criteria

Make small, concise, logical commits. Check the working tree before
each one. Group the shared host implementation and the correctness
checks needed for it to stand alone. Commit independently meaningful
benchmark coverage or related changes separately when appropriate.
Review the commit series before returning it.

The upstream work is ready when the relevant equal-shape calls validate
and write directly with unchanged output bits and ownership, the
representative microbenchmarks show zero timed allocations, and the
dependency tests and vet pass. The primary Llama benchmark must show a
material fall in exact allocation bytes and objects consistent with
removing the adapter, with the targeted profile stacks eliminated or
reduced. Check for timing regressions over paired samples, report any
timing gain only when it is distinguishable from observed variance,
and record wall time and RSS. Keep the supplemental results and full
correctness checks with the primary data.

Return the upstream commit IDs, a release tag or pseudo-version when
available, exact benchmark commands and raw samples, profile artifacts
and an analysis summary, and the correctness results. Llama integration
should update the real `go.mod` and `go.sum` to the returned version,
verify the release or pseudo-version in the binaries with `go version -m`,
and repeat the paired Llama benchmark, fresh profiles, and correctness
checks using the published module. The preliminary local replacement and
the published-module run both need an independently identified baseline.
