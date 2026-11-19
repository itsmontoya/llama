# Optimization Pass 3

Pass 3 moved cached Llama 3.2 1B generation from **4.216 to 3.060 s/op
(−27.4%)**. Time to first token fell from **1.881 to 0.742 s (−60.6%)**.
The change is in the shared CPU product kernel in
`neuralnetwork`, commit `e378860ebdb6efbe835c194d80fab012c60ed256`.

Llama pins the published module version
`v0.10.13-0.20261119195252-e378860ebdb6`. Go verified the public module
through its checksum database. The module and `go.mod` checksums match
the archive used for the paired measurements.
[Published upstream branch](https://github.com/itsmontoya/neuralnetwork/tree/optimization-pass-3)

## Selected optimization

The fresh pre-optimization profile attributed 3.23 of 4.70 sampled prefill
CPU seconds (68.7%) to the three-row product. The compiler emitted four
scalar loads and three fused multiply-adds per contraction step, each
feeding one accumulation chain. The single-row kernel accounted for more
aggregate sampled CPU and already used four independent accumulators.
Three-row projection offered a similarly large target with a measurable
opportunity to reuse input loads and run more independent accumulation
chains. [Profile evidence, call path, and handoff](NEURALNETWORK_OPTIMIZATION_HANDOFF.md)

The CPU helper now computes four output columns for each tile of three
input rows. It shares weight loads across rows and activation loads across
columns. On ARM64, seven scalar loads feed 12 independent fused
multiply-adds without loop bounds branches or register spills. Each output
still accumulates in ascending contraction order. An ordinary scalar loop
handles leftover columns. Six alternating dependency benchmark pairs showed
66.6–69.7% lower `GemmInto` time across the four six-row Llama projection
shapes and 66.8% lower time for the 24-row FFN gate/up projection. The
raw tile allocates zero bytes.

## Paired Llama measurements

Seven alternating before/after pairs ran on an Apple M3 with 24 GB RAM,
macOS 26.5.2 (25F84), darwin/arm64, and Go 1.27.3. Both binaries used
`GOARM64=v8.0`, `CGO_ENABLED=1`, default tags, and `GOMAXPROCS=8`.
Each fresh process performed one benchmark iteration with zero inference
warm-ups. The pinned workload used six prompt tokens, one prefill, seven
decode calls, eight generated tokens, and a context cap of 512. The Go
binaries report `neuralnetwork v0.10.12` and
`v0.10.13-0.20261119195252-e378860ebdb6`.

| Median metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Prefill / first token (s) | 1.881 / 1.881 | 0.7418 / 0.7419 | −60.6% |
| Decode (s/token) | 0.3357 | 0.3311 | −1.4% |
| Decode (tokens/s) | 2.978 | 3.020 | +1.4% |
| Cached generation (s/op) | 4.216 | 3.060 | −27.4% |
| Output (tokens/s) | 1.898 | 2.615 | +37.8% |
| Generation bytes/op | 198,718,008 | 198,723,104 | +0.003% |
| Generation allocs/op | 39,866 | 39,869 | +0.008% |
| Process wall (s) | 7.225 | 6.370 | −11.8% |
| Peak RSS (GiB) | 6.3729 | 6.3728 | −0.002% |

Generation improved in all seven pairs. The observed ranges were
4.159–5.804 s/op before and 2.990–3.668 s/op after. First-token ranges
were 1.854–2.723 s and 0.676–1.354 s. Decode, allocation, and RSS
ranges overlapped. Process wall and RSS include snapshot verification,
checkpoint loading, and tokenizer setup. Median tokenizer setup was
0.167 s before and 0.198 s after, with overlapping observed ranges.

## Supplemental 29-token prompt

Three alternating pairs timed prefill and first-token selection on the
same checkpoint and runtime cap. The prompt was “France is a country in
Western Europe. It shares borders with Belgium, Germany, Switzerland,
Italy, and Spain. The capital of France is”. Median time fell from 10.045 s
(range 9.931–10.071 s) to 4.571 s (range 4.570–4.609 s), a 54.5%
reduction. All six trials chose token `12366` and had the same SHA-256
over the 128,256 little-endian FP32 logits,
`6ffb4dc93003480aebdd51e9f1fee47ae3836eea9fb35ce2508f5d618a9871a5`.

## Paired profiles

Separate three-operation profile runs used a 64 KiB heap sampling rate.
The three-row product fell from **3.73 to 1.58 sampled CPU self seconds
(−57.6%)**, or 1.243 to 0.527 sampled seconds per operation.
Total prefill CPU fell from 4.70 to 2.15 sampled seconds. Whole-process
CPU samples fell from 14.20 to 11.12 seconds. The decode projection
remained the largest consumer, with 4.70 sampled self seconds out of
6.17 seconds on post-change decode stacks.

The filtered allocation profile estimated 569.86 MiB before and
565.32 MiB after. In the after profile, elementwise stacks account for
291.08 MiB and 52.3% of sampled objects. SwiGLU accounts for 237.44 MiB
inclusive and overlaps elementwise and activation stacks. GEMM's
temporary product accounts for 66.83 MiB. Checkpoint loading accounts
for 9,757.32 MiB, or 93.2% of cumulative process allocations.
Unprofiled bytes/op and allocs/op provide the comparison for generation
allocation totals.

## Correctness

Both dependency versions passed the full default and `purego` Llama
test suites on Go 1.27.3, including a rerun with the published version
in Llama's `go.mod`. The pinned Transformers comparison preserved
every checked greedy winner. The largest absolute logit error was
`4.95910645e-05`. All 128,256 logits matched between cached and full-prefix
evaluation at each of the five checked positions. All numerical checks
used the existing tolerances.

The frozen completion IDs are
`[12366, 13, 1102, 374, 279, 1455, 95551, 3363]`. Cached and uncached
CLI tests checked those IDs and all stdout bytes. The standalone default
and `purego` binaries printed ` Paris. It is the most populous city`
with one trailing newline. The suites also check the pinned checkpoint
inventory, Llama tokenizer, TinyStories logits, tokenizer and generation,
and the synthetic MHA, GQA, and MQA reference traces.

The Llama race run covered synthetic and TinyStories tests, including
concurrent decoder calls and sessions. The 1B checkpoint was exercised
in the default and `purego` suites. The dependency passed its float32
product, tensor and matrix suites in default, `purego`, race and Metal
configurations. `go vet ./...` passed in each repository.

## Optimization Pass 4 candidates

1. **Single-row CPU projection.** It accounts for 4.70 of 6.17 sampled
   decode CPU seconds (76.2%) and 4.88 of 11.12 whole-process CPU seconds.
   Measure the classifier and FFN shapes independently to distinguish
   memory traffic from instruction throughput.
2. **Elementwise and SwiGLU allocations.** Elementwise accounts for
   51.5% of generation bytes and 52.3% of sampled objects. The binary
   tensor adapter alone accounts for 192.58 MiB across three operations.
3. **Remaining multi-row projection cost.** The updated kernel still
   accounts for 1.58 of 2.15 prefill CPU seconds (73.5%). Measure
   shorter tails and longer prompt sizes separately.
4. **Checkpoint conversion and weight ownership.** Loading accounts
   for 93.2% of cumulative process allocations and 1.827 s of median
   setup wall time. This is the leading allocation target for cold
   command startup.
5. **Host GEMM result staging.** The temporary product represents
   11.8% of generation allocation bytes. Any change must preserve
   destination and error semantics.
6. **Longer-context attention and KV cache.** Cached attention had
   little sampled CPU time with a 13-position cache. A sustained
   long-context profile would establish its priority for that workload.

## Reproduction

Set `LLAMA_32_1B_MODEL_DIR` to the verified
[Llama 3.2 1B snapshot](testdata/README.md#llama-32-1b-transformers-reference).
Use an Apple M3 or report the hardware variation. Run from the Llama root.
Keep setup, benchmark count, compiler, and build tags identical across both
binaries.

```sh
export GOTOOLCHAIN=go1.27.3 GOARM64=v8.0 GOMAXPROCS=8 CGO_ENABLED=1
bench_dir=$(mktemp -d)
git show 20a45d9:go.mod > "$bench_dir/before.mod"
git show 20a45d9:go.sum > "$bench_dir/before.sum"
cp "$bench_dir/before.mod" "$bench_dir/after.mod"
cp "$bench_dir/before.sum" "$bench_dir/after.sum"

go get -modfile="$bench_dir/after.mod" \
  github.com/itsmontoya/neuralnetwork@v0.10.13-0.20261119195252-e378860ebdb6
go mod tidy -modfile="$bench_dir/after.mod"

for build in before after; do
  go test -mod=readonly -modfile="$bench_dir/$build.mod" \
    -c -o "$bench_dir/$build.test" .
  go version -m "$bench_dir/$build.test"
done
```

Run the following benchmark once in each of 14 fresh processes, reversing
the before/after order every pair. Use macOS `wait4` (also used by
`/usr/bin/time -l`) to record process wall time and peak RSS in bytes.
Derive decode seconds/token by dividing benchmark decode seconds/op by
seven. Derive output throughput by dividing eight by generation seconds/op.

```sh
/usr/bin/time -l "$bench_dir/$build.test" -test.run '^$' \
  -test.bench '^BenchmarkLlama32CachedGeneration$' -test.benchtime 1x \
  -test.benchmem -test.timeout 15m -test.v
```

Collect the two fresh profile sets using the same invocation with
`-test.benchtime 3x -test.memprofilerate 65536
-test.cpuprofile "$bench_dir/$build.cpu.pprof"
-test.memprofile "$bench_dir/$build.heap.pprof"` and no wall/RSS wrapper.
Filter CPU by `\(\*Session\)\.Prefill` and
`\(\*Session\)\.Decode`. For inference allocation attribution use
`-focus=BenchmarkLlama32 -ignore='LoadHFModel|LoadLlama3|verifyLlama32Snapshot'`
with both `-alloc_space` and `-alloc_objects`.
[Full correctness commands and fixture requirements](NEURALNETWORK_OPTIMIZATION_HANDOFF.md#validate-the-result-in-llama)
