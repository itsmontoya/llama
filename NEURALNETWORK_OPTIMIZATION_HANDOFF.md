# Neuralnetwork handoff for Llama Optimization Pass 9

## Objective and ownership

Improve sustained CPU decode by testing one change in the host right-transposed FP32 GEMM. Prepare the immutable FFN weights in eight-output K-major tiles so the kernel can consume vectors across outputs directly. Start with gate/up, then exercise down through the same layout. Neuralnetwork owns the weight representation, dispatch and ARM64 kernel. Llama owns only the later integration of a proven prepared-weight path.

Use the released `neuralnetwork v0.10.18` at `ce91677fa12ff521309764e4664a296f57a52427` as the upstream baseline. Its module checksum is `h1:j1PoyjYzxYhlwaWOUp/KlSh4V2xu8Upt/9kyIcnK/AU=`. The consuming Llama baseline is `github.com/itsmontoya/llama` at `b07f4ae100d0ada87a8a4485f294bdacb7fe137e`. The original model is `meta-llama/Llama-3.2-1B` revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`. Use Go 1.27.3, Darwin ARM64 on Apple M3, `GOARM64=v8.0`, `GOMAXPROCS=8`, `CGO_ENABLED=1`, default tags, empty `GOFLAGS` and `GOWORK=off` for the release timing series.

## Problem and fresh evidence

The published pin's fresh 48-process timing series on the pinned fixture measured a sustained decode median of 3.1376 s and 10.200 tokens/s over 32 steps, with a wide 2.2027–4.1594 s latency range. Two simultaneous 32-step sessions measured 3.768 s median wall, 16.990 tok/s aggregate, 8.525 and 8.493 tok/s per lane, and a 1.0038 faster/slower rate ratio. Median process peak RSS was 6.323 GiB single and 6.371 GiB concurrent. Primary generation measured 0.8327 s median, with two later timing builds required for any release claim.

The table combines separate fresh probes. Single-session independent projection wall is the median across three 32-step loops. The total decode loop median was 2.5037 s, of which a median of 92.9% was inside projection. Single CPU and allocation scopes are 64 steps. Concurrent scopes are 128 total steps across two pairs of synchronized 32-step sessions. Concurrent projection wall sums the durations across both lanes and can overlap.

| Projection (M,K,N) | Single wall (s) | Single CPU (s, % of labeled phase) | Single traffic (MiB allocated) | Concurrent summed wall (s) | Concurrent CPU (s, % of labeled phase) | Concurrent traffic (MiB allocated) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention K/V (1,2048,512) | 0.1206 | 0.05 (0.4%) | 10.5 | 0.8234 | 0.52 (2.0%) | 18.3 |
| Attention Q/O (1,2048,2048) | 0.2879 | 0.28 (2.2%) | 33.0 | 2.0702 | 2.57 (9.7%) | 65.3 |
| FFN gate/up (1,2048,8192) | 1.0179 | 8.40 (64.7%) | 128.1 | 6.1850 | 10.37 (38.9%) | 257.7 |
| FFN down (1,8192,2048) | 0.4730 | 1.31 (10.1%) | 17.3 | 2.9437 | 4.69 (17.6%) | 34.0 |
| Tied classifier (1,2048,128256) | 0.4463 | 2.86 (22.0%) | 63.0 | 2.3114 | 6.70 (25.2%) | 125.6 |
| All projections | 2.3259 | 12.90 (99.4%) | 252.0 | 14.3336 | 24.85 (93.3%) | 500.8 |

The single-row NEON function alone consumed 11.02 of 12.98 sampled seconds in the labeled sustained phase and 20.42 of 26.63 in the concurrent phase. Total profiled phase wall was 4.575 s sustained and 7.511 s in the concurrent series. Phase allocation was 633.7 and 1,249.8 MiB respectively. Worker dispatch accounted for about 3.8 and 7.1 MiB of that traffic. Complete three-row prefill used 85.0%, 81.4% and 82.8% of labeled CPU at 27, 28 and 29 tokens, respectively. The one-row remainder at 28 used 0.21 sampled CPU seconds, excluding the classifier. The two-row remainder at 29 used 0.40 s. A load-only profile measured 1.70 s of labeled CPU independently of 1.965 s of wall. Loading dominated sampled primary-process allocation traffic, with approximately 4.60 GiB retained heap after a forced GC.

Fresh Go trace counts were 19,168 spawned helpers for 32 single-session steps and 38,336 for two concurrent 32-step sessions. Median block creation-to-first-run delay changed from 30.1 to 140.7 µs, and classifier delay from 35.0 to 210.8 µs. Labeled condition signaling was 1.84 sampled CPU seconds in the 64-step sustained profile and 3.67 in the 128-step concurrent profile. Caller join waits include outstanding useful computation. Across 3.871 s of summed two-session wait time, a corresponding helper was in the Go Running state except during 1.829 ms with pending runnable helpers. Summed concurrent waits can overlap, and Go Running can include OS descheduling and synchronization.

## Limiting-factor analysis

The best-supported classification is mixed, with material weight-access sensitivity. Single-row projections visit 4.60 GiB of logical FP32 weights per token, including 3 GiB for FFN gate/up and down. The probes imply 67.5 GB/s logical weight read throughput for gate/up, 72.6 for down, 75.3 for the classifier and 68.0 across projections. The separate contiguous read control reached 95.9 GB/s at P=8. Logical rates include different compute and cache behavior. The host could not collect PMU measurements for cache line refills, physical bandwidth or stalls.

At equal shape and public GEMM dispatch, a repeated 64 MiB gate/up matrix measured 2.658 ms median at P=1 and 1.083 ms at P=4. Rotating actual layer weights gave 2.984 and 1.311 ms, respectively. There are no helper goroutines at P=1. In the separate test with two callers, holding shape and dispatch fixed and making the readers share the same rotating operand reduced elapsed time by paired medians of 22.55% at P=4 and 16.00% at P=8. This occurred in six of six and five of six pairs. Simultaneously sharing one operand is an experimental control and may not reproduce the real sessions' cache overlap.

The controls show weight locality matters even without scheduler work. They do not measure which level of the cache hierarchy supplied the bytes or how close the kernel was to a physical bandwidth or instruction throughput ceiling. Source inspection shows both a fragmented stream geometry and substantial register shuffling. This supports testing a layout that reduces those costs together while preserving the weight count, and measuring the value at the pinned Llama workload.

## Exact call path and current implementation

- Llama `session.go` `(*Session).Decode` gathers the current embedding, sets up one-row RoPE, and invokes `(*Decoder).forwardBlock` for every layer. `forwardBlock` in `decoder.go` uses `headProjection` in `attention.go` for Q/K/V and `project` for attention output, FFN gate/up and FFN down. `(*Session).commit` calls `(*Decoder).projectLogits` for final normalization and the classifier. It returns a caller-owned logit slice after finiteness validation and then advances the committed position.
- Llama `(*Session).Prefill` calls `forwardPrefix` to process all prompt rows, gathers the last hidden row and calls the same `commit`. The block projections use multi-row inputs and the classifier uses M=1. `(*Decoder).Forward` for full prefix uses the same block projections and can use a multi-row classifier.
- Llama `tensor_ops.go` `project` calls `tensor.GemmShape`, allocates a disjoint FP32 tensor, then calls `tensor.GemmInto(input, weight, nil, out, GemmConfig{TransposeB: true, Alpha: 1})`. The host `tensor/gemm.go` validates types, shape, bias and destination. `tensor/matrix_product_host.go` allocates product staging in `matrixProductValues`, then `matrixProductInto` calls `internal/float32product.RightTranspose`. `GemmInto` applies alpha and any bias and copies the product to the destination.
- `internal/float32product/product.go` `RightTranspose` calls `dispatchRightTransposeOneRow` for M=1 and at least four columns. `right_transpose_parallel.go` sets workers to the minimum of eight, `GOMAXPROCS`, the number of four-column tiles, and the number of complete 2 MiB weight budgets. `rightTransposeOneRowParallel` gives disjoint contiguous output ranges to workers, starts one goroutine per helper, computes the last range on the caller and joins via `sync.WaitGroup`. The three-row group path enters through `dispatchRightTransposeRows` and `rightTransposeRows`, with distinct three-row, two-row, and one-row paths.
- `internal/float32product/right_transpose_one_row_arm64.go` `rightTransposeOneRow` selects `rightTransposeOneRowTileARM64` in `right_transpose_one_row_arm64.s` for complete eight-column tiles at K ≥ 64 and divisible by four. The tile initializes two four-lane accumulators to positive zero. Per four K values it uses eight 16-byte weight loads from eight rows, one 16-byte activation load, four activation broadcasts, sixteen ZIPs, and eight vector FMAs. Every output lane accumulates in ascending K. The Go four-output and scalar loops cover ineligible contraction lengths, output tails, other architectures, `purego`, and race builds. `GOARM64=v8.0` is part of the baseline.
- The immutable right operand uses canonical dense row-major `[N,K]`. Within each stream the tile reads contiguous bytes and eventually consumes the full cache lines. Its eight independent streams start 8 KiB or 32 KiB apart. The row-to-K-major transpose is register-local. Weights are read once per M=1 projection and the left vector is reread per eight outputs. The classifier reads a 1,002 MiB tensor, which Llama ties to its token embedding by pointer in `weights.go` `(*Decoder).loadWeights`.
- Llama `hf_model.go` `loadHFCheckpoint` validates the checkpoint inventory, reads BF16 as FP32 with `internal/safetensors.(*File).ReadFloat32`, interleaves Q/K rotary rows, checks finiteness in `weightTensor` and passes a private copy to `tensor.FromFloat32`. The active checkpoint has 16 layers and tied embeddings.

## Proposed direction

Prototype a prepared FFN weight layout that stores consecutive K positions with their eight output weights together inside each output tile. Preserve the logical `[N,K]` order at the API boundary. Each K iteration should be able to load the two four-output vectors directly and keep the existing ascending-K FMA chains. Begin with K=2048 and N=8192 to compare the 64 MiB weights across layers. Exercise the down shape through the same representation. Keep worker policy and the ordinary dense path fixed for this experiment.

Validate the hypothesis with four comparisons using the same operation and dispatch. Measure the existing and prepared tiles over a repeated weight, rotating checkpoint weights, shared versus distinct weights across two callers, and the pinned one- and two-session Llama workloads. Confirm the disassembly removes the row-to-K register transpose in the prepared hot loop and record how the changed streams affect locality. Measure preparation separately. A single authoritative backing representation for prepared FFN tensors is preferred. Prototype subset storage only within the working-set budget and report exactly what it retains. Keep complete-group prefill and row remainders working with the same immutable weights. If they cannot use the prepared data efficiently within the memory and correctness constraints, record that result before proposing adoption. Initially leave the tied classifier on its existing path.

Compare with three alternatives during design review. A dense-layout FMA or prefetch adjustment could avoid startup cost while retaining eight row streams and the existing API. Changing output tile width alters load streams, register pressure, shard alignment and scalar tail behavior at the same time. Bounded worker reuse could address condition signaling and queueing, but the serial rotation penalty persists without helpers and the traces show useful work during joins. Treat these as alternative explanations or future work and keep this experiment focused on a prepared layout.

## Numerical correctness

Preserve bitwise ascending-K FP32 FMA accumulation for the eligible production path, including the initial positive zero and operation and operand order. Keep scalar/fallback and prefill remainder arithmetic under their existing contracts. Preserve results for deterministic finite random data, actual checkpoint tensors, NaNs including payload propagation, infinities, signed zeros and subnormals where covered by current semantics. Exercise tile and contraction boundaries, output tails, uneven cuts, empty contractions, bias and alpha paths, and dirty destinations. Keep the existing reference tolerance and pinned logit hashes. Confirm serialized/checkpoint inputs remain usable under the existing format.

## Ownership, lifetime and concurrency

Keep immutable weights shared by sessions and keep the tied embedding and classifier logically unified. Preserve constructor ownership, destination alias checks, error atomicity, disjoint scratch, caller-owned retained logits and failed-step retries. Preserve the public ordinary tensor contract across shape queries, values, gather, GEMM, serialization and non-eligible paths. Any preparation must be deterministic, preserve source ownership on failure, and be safe for concurrent use. Validate cleanup and reclaimability on preparation failures and decoder disposal. Preserve bounded worker counts, synchronous completion and any cancellation contract. Compare both lanes for deterministic continuation, whether used serially or concurrently. Run race tests on all path-independent lifetime code and exercise ARM64 dispatch through default builds.

## Memory and startup

Record cold load, any preparation wall and CPU, cumulative allocation, loaded live heap after forced GC, process RSS high-water and steady-state RSS on the actual checkpoint. Baseline was 1.8975 s median model load in the unprofiled primary series and approximately 4.60 GiB retained Go heap in the load-only probe. Report weight footprint by class and temporary bytes at preparation peak. Release transient memory promptly and confirm it is reclaimable. Measure any lazy preparation on cold TTFT and under two simultaneous sessions. Measure tied storage, serialization and repeated model load/unload. A persistent second model-sized or FFN-sized representation needs measured costs and an explicit acceptance decision.

## Benchmark and correctness plan

1. Use the public host `GemmInto` for M=1 with (K,N) of (2048,512), (2048,2048), (2048,8192), (8192,2048), and (2048,128256). Sweep one, two, four and eight scheduler processors, and compare repeat against rotation on real checkpoint layer weights. Include a shared/disjoint two-caller control and inspect linked ARM64 disassembly. If counters are available, report physical memory traffic, cache refills, execution stalls and relevant per-core activity. Otherwise retain the logical rates and trace limitations.
2. Pair the real Llama baselines with the candidate. Keep model, tokenizer, compiler, tags, CGO, runtime cap and warm-up fixed. Build separate, checksum-verified binaries. Alternate which binary runs first in each workload. Use 12 primary, seven sustained, seven concurrent sustained, five each of the three prefill lengths, and seven concurrent prefill pairs. Time exactly one operation per process with zero inference warm-ups. Keep all valid observations and profile separately.
3. Primary generation uses `The capital of France is`, six prompt tokens including BOS, eight greedy outputs, context cap 512 and KV capacity 13. Record prefill, TTFT, decode seconds/token and tokens/s, cached generation and output tokens/s, B/op, allocs/op, full process wall and peak RSS. Run the 32-step sustained decode at positions 6–37, with setup, prefill and first selection excluded. Then run two independent sessions on one decoder with the common start barrier for their 32 timed steps. Record aggregate wall and throughput, both lanes' wall and throughput, fairness ratio, B/op, allocs/op and peak RSS. Guard 27-, 28- and 29-token prefill and two concurrent six-token prefills. Attribute complete groups, each remainder type, and the classifier separately.
4. Run upstream `go test ./internal/float32product ./tensor -run 'RightTranspose|EmptyContraction|Gemm_Bias'` under default, `purego` and race configurations, and the upstream affected-package suites and `go vet`. Retain `TestRightTransposeOneRowAgainstOrderedReference`, `TestRightTransposeOneRowNaNOperands`, `TestRightTransposeConcurrent`, `Test_Gemm_RightTransposeSingleRowConcurrentRetry` and the group and remainder coverage. Add focused regression tests for any new tile, layout, preparation, fallback or lifetime edges that those tests miss. Require clean race runs for concurrent shared input, different destinations, retained outputs, failures/retries and cleanup.
5. In Llama, run fixture-enabled default and `purego` suites, `go vet`, and the established race suite with TinyStories and local traces, leaving the Meta model unset for the full race suite as in Pass 8. Confirm the pinned Transformers numerical tolerance and greedy winners, cached/full-prefix parity, frozen token IDs and CLI bytes, TinyStories, synthetic MHA/GQA/MQA traces, retained outputs and step retries. Check all supplemental hashes and both sustained lanes. For the full fixture, use `LLAMA_32_1B_MODEL_DIR`, `LLAMA_STORIES260K_MODEL` and `LLAMA_STORIES260K_TOKENIZER` as documented in Llama's fixture instructions.

## Acceptance and completion

Require a repeatable, material real-Llama sustained decode improvement with a consistent majority of paired gains and no material concurrent aggregate regression, per-lane regression or fairness loss. Similar timings across noisy pairs are inconclusive. Report per-build medians, full ranges, median paired absolute and percentage deltas and improved-pair counts. Measure startup, retained and peak memory and allocation effects at the same model pin. Use the Pass 8 reference of a 5% paired-median prefill regression as an investigation threshold for each prefill guard, including concurrent prefill. Review unfavorable tails independently. Preserve all correctness requirements and compare CPU and allocation profiles after the change. Complete the upstream work only after demonstrating the real workload effect or documenting that the tested hypothesis failed.

Use small, logical commits for the diagnostic, implementation, and regression coverage as appropriate. Keep the scope to the optimization described above.

## Return contract and Llama validation

Return immutable implementation and regression commit hashes, exact tests and results, raw and summarized paired real-Llama and GEMM measurements, pre/post CPU and allocation profiles, worker and memory evidence, and the published module version with checksums or an immutable commit for review. Include any new API usage, activation conditions, fallback, lifetime and integration instructions. State explicitly whether the hypothesis held and which acceptance criteria passed. For a negative result, include the measurements and the most direct reason implementation should stop.

The Llama integration should verify the returned source, version and checksum, then pin the dependency and add only the necessary integration. Rerun the correctness suite and all paired workloads using the existing harness, and gather matched post-change CPU and allocation profiles. Reassess the limiting factor and rank the remaining candidates from those profiles. Upstream dimensional wins alone do not establish a Pass 9 release result.

The manifest and frozen Transformers reference live in Llama's `testdata/llama32_manifest.json` and `testdata/llama32_reference.json`. Their SHA-256 digests are `3404dfda97c69cf3a9a3cf15dc4db7017cb049da07462c52f2e3004f6e12ab64` and `3a66ddc5cbdef45e397e47a6bd992676ab3d35234a8b389908f721b05466d361`, respectively. The frozen benchmark hashes below use little-endian FP32 logits.

| Workload | SHA-256 |
| --- | --- |
| Prefill 27 | `0b087b0839da36ec06a3a8f777d4c3c944ccc7439107fcad39b6ad0239ed9afc` |
| Prefill 28 | `cc93319c3c0701fd9cd3fdc2080afe43d2f82ba7cd236a698d19518836854e8c` |
| Prefill 29 | `6ffb4dc93003480aebdd51e9f1fee47ae3836eea9fb35ce2508f5d618a9871a5` |
| Sustained terminal row, each session | `b72d983e8b5c45f83ad858b52f1d27b87a7ad8118be6c5d315db2ea6fc6fd84a` |
| Concurrent six-token prefills | `f2e070e41d1eac82f0159284cfdd0d070aa06ef703c833fa6fb781a5b7b487c1` |

The complete 33-token sustained suffix including the first selection is `[12366 13 1102 374 279 1455 95551 3363 304 9822 323 279 3224 753 6864 323 7928 3363 13 12366 374 279 1455 95551 16036 3158 304 279 7665 9323 323 279 1917]`. The primary suffix is the first eight of those IDs. The existing primary benchmark verifies the decoded bytes.

The Llama investigation commit is `0af6bf0b26b48edda3099eb9baec368e6b1f7de9`. Its report includes the exact profiling and timing reproduction commands. The fresh evidence archive is `/private/tmp/llama-pass9-evidence.hqiIiZXs.tar.gz`. The evidence archive contains runnable Llama-only diagnostic overlays and the raw logs. [Pass 9 investigation](OPTIMIZATION_PASS_9.md)

---

# Llama LL-01 results and NN-06 handoff

LL-01 integrates `github.com/itsmontoya/neuralnetwork v0.11.0` at
`95e3339bbb44615137c32599f89841aad1c088fb`. The module and module-file
checksums are `h1:+QMn9KogezZ8IbCL1d4xSVynQtbswxe+5tdqLB5qoto=` and
`h1:SafbkuDehVErsuJAL5FHFsLhKJbeh/OI3ItvbmwDafw=`. The Llama branch is
`gpu-ll-01`, based on Llama `main` at
`bdc22d2f6d23512788f26143144bd7d157941215`. The production change is
`65bde40` and the CLI and test entry points are in `fcf1190`.
[NN-05 released FFN contract](https://github.com/itsmontoya/neuralnetwork/pull/145)

## Resident FFN and correctness

The hybrid FFN starts at the CPU post-attention RMSNorm result `[M,2048]`.
One scope records gate and up projections with logical row-major weights
`[8192,2048]`, SwiGLU into a distinct `[M,8192]` tensor, and the down
projection with weight `[2048,8192]` into a distinct `[M,2048]` tensor.
Each layer uploads one normalized input and downloads one down result after
completion for the CPU residual. Scope close drains outstanding work on
cancellation. A cached step publishes its position only after logit validation.
The same model device serves independent calls, with disjoint scratch for each
active call. The attention, normalization, residual, KV and classification
paths use the existing Llama tensor operations and cache layout.

On Apple M3, the six-token prompt IDs were
`[128000 791 6864 315 9822 374]` and both cached and uncached CLI modes
returned `[12366 13 1102 374 279 1455 95551 3363]` and
`" Paris. It is the most populous city\n"`. The maximum sampled absolute
error against the frozen Transformers observations was `5.09023666e-5`
with a maximum tolerance fraction of `0.0309471734` under the existing
`1e-3 + 1e-3*abs(reference)` bound. Position 11 retained the frozen winner
with a margin of `0.0433597565` versus `0.043340683` in Transformers.
The untagged CPU maximum sampled absolute error was `4.95910645e-5`.
Across all 128,256 logits, the observed maximum cached/full-prefix hybrid
absolute error was zero across the eight greedy steps and the three
27/28/29-token prefill tails under the existing `1e-4` absolute and relative
bounds. Two concurrent sessions gave the same results as their corresponding
full-prefix executions. Synthetic Llama traces had maximum absolute error
`4.76837158e-7` against CPU, and the released realistic-size synthetic FFN
had maximum absolute error `1.52e-6` against FP64 across M=1/27/28/29.
The tests include partial loading failure, scope recording failure, cancellation
after submission, retry at the same position, and close. Frozen artifacts
use the revision and hashes documented above.

## Measured boundaries

Four short fresh-process pairs alternated the prepared-default and explicit
Metal FFN modes of one Metal-tagged binary. Companion fresh processes used
the untagged host-only integration binary and the unchanged untagged Llama
base. All measured outputs matched the frozen eight IDs and sampled logits.
Each process loaded the same checkpoint and took one six-token prefill and
seven timed decode steps without an inference warmup. The fixture was verified
before the campaign. The driver used frozen token IDs, so load plus generation
excludes tokenization. Process wall includes test setup, correctness checks,
post-run GC and RSS observation. Timings below are medians and ranges.

| Mode | Model load, s | Prefill, ms | Mean decode, ms/token | Load + generation, s | Peak RSS, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Starting main, untagged | 1.851 (1.810–1.877) | 109.3 (105.1–124.1) | 65.09 (65.02–68.42) | 2.420 (2.371–2.475) | 6.274 (6.273–6.274) |
| Integration CPU, untagged | 1.870 (1.855–1.910) | 113.1 (109.2–128.8) | 67.15 (64.84–73.96) | 2.464 (2.435–2.522) | 6.274 (6.274–6.274) |
| Integration default FFN, tagged | 1.831 (1.796–1.887) | 3341 (1302–3492) | 1115 (926–1331) | 11.951 (11.632–14.697) | 14.101 (14.066–14.121) |
| Integration Metal FFN, tagged | 1.792 (1.766–1.988) | 335.0 (315.3–385.6) | 263.66 (259.06–268.56) | 3.988 (3.918–4.202) | 8.625 (8.623–9.474) |

The untagged build measures host execution. The tagged default uses the
existing prepared interface with the ordinary release tensor adapter. The tag
disables prepared packing and allows its matrix operations to dispatch to
Metal. The matched tagged test controls the build and model, while the
untagged test is the actual CPU baseline. The median paired Metal/untagged
ratios were 2.96 for prefill, 3.98 for mean decode and 1.61 for load plus
generation. Eight-output generation including greedy selection took 2.183 s
on Metal and 0.581 s on the untagged CPU. Mean decode across positions 6–37
in two matched longer runs was 315.0 ms on Metal versus 66.66 ms on untagged
CPU. The longer runs greedily checked the first eight predictions, then used
fixed replay inputs beyond the frozen trajectory.

A separate three-process host diagnostic measured `OpenDefault` at
28.19 ms (27.42–48.57), 48 buffer allocations at 1.75 ms (1.54–1.82),
and 3 GiB of weight uploads at 183.65 ms (181.86–195.72). Each short
generation records 128 scopes and 512 FFN dispatches. Across these scopes,
median create/encode/submit/close totals were 0.493/1.978/0.579/0.081 ms.
Wait accounted for 408.61 ms (398.81–411.47) of 411.87 ms
(402.85–414.56) of FFN command wall time. Wait includes GPU execution,
queueing and Go scheduling. Published `gpu` APIs expose neither GPU kernel
timestamps nor separate lazy pipeline compilation times. The first prefill
includes first-use pipeline preparation. The two boundary transfers copied
1,703,936 bytes per direction across the short generation. Upload and
download took 0.214 and 0.271 ms in total at the medians. The surrounding
CPU tensor copies took 0.205 ms to extract and 0.504 ms to rewrap. Treat
numbers from the diagnostic binaries as separate from the uninstrumented
paired medians. The two copies per layer are part of the current API
contract. Most hybrid wall time was outside the timed FFN scopes.

The checkpoint retains 3 GiB across 48 Metal FFN tensors (192 MiB per
layer), with no independently prepared CPU FFN copy on this path.
The other FP32 model tensors account for 1,722,032,128 bytes.
One shape-matched scratch set requests `4*M*(2*2048+3*8192)` bytes,
or 112 KiB at M=1 and 3.172 MiB at M=29. Host staging includes one
converted checkpoint tensor at a time, extra Q/K rotary permutation storage,
and a copied activation input and caller-owned output. The largest decoded
tensor is the 1,050,673,152-byte embedding. Post-GC Go heap and process RSS
medians were 1.605 GiB and 8.140 GiB for Metal versus 4.605 GiB and
6.274 GiB for the untagged CPU. Across short runs, observed per-generation Go
allocations totaled 12.905 GiB for Metal versus 0.108 GiB for the untagged
CPU. The existing tagged matrix adapter copies ordinary operands, including
the tied embedding/classifier, during execution. Unified memory permits CPU
and GPU accounting to overlap. Peak RSS and requested native bytes describe
different measurements.

## Generic operations and storage for NN-06

- Keep the existing rank-two right-transposed Q/K/V and output projections.
  For the pinned model Q and O weights are `[2048,2048]` and K/V weights are
  `[512,2048]`. The resident projection results must support a checked
  layout transformation from `[M,2048]` or `[M,512]` to the logical
  `[M,32,64]` and `[M,8,64]` head layouts. Preserve the existing grouped
  output order on the path back to `[M,2048]`.
- Provide adjacent-pair rotary multiplication for Q and K, using
  caller-supplied coefficients or frequencies at absolute positions. Llama
  already normalizes checkpoint Q/K row order and computes the scaled
  coefficients. The numeric step is
  `(x0*cos - x1*sin, x1*cos + x0*sin)` with per-position coefficients
  and a broadcast head dimension. V stays in projection order.
- Provide device-resident stable masked attention math for M=1 and prefill.
  For head width 64 and the 32:8 GQA mapping, queries in group `h/4` read
  that group's K/V head. Support arbitrary compatible MHA, GQA and MQA.
  The operation needs grouped/batched score and value products or their fused
  equivalent, multiplication by `1/sqrt(head_width)`, an absolute-position
  causal mask excluding future keys before stable softmax, and accumulation
  over populated values only. Avoid materializing repeated K/V heads as
  physical copies if a head mapping can express the same contract.
- Add a checked device-side append/read-range primitive or a precise layout
  view for session-owned K/V planes `[capacity, KV_heads, head_width]`.
  Prefill writes a prefix and decode writes one row at the next absolute
  position. Llama owns the committed length. Failed or canceled recordings
  must not publish the new length or damage prior rows. A retry must overwrite
  the staged rows before attention can read them. Support disjoint mutable
  sessions sharing read-only weights. Existing full-buffer leases and
  failed-write invalidation need a documented contract for protecting
  committed history across partial writes and device failures.
- RMSNorm over the last axis needs exact row-broadcast handling for a
  `[2048]` norm weight and a supplied epsilon. Together with existing Add,
  keeping normalization and residual operations resident avoids recreating
  an attention-to-FFN host round trip.

For upstream acceptance, encode a complete attention-shaped sequence
without intermediate host transfers and check both single-row and multi-row
inputs, position offsets, uneven tails, MHA/GQA/MQA, empty history,
cancellation and concurrent KV ownership. Return operation shapes and
failure semantics, scratch and KV bytes, scope counts and actual Metal
diagnostics. The Llama layer remains the owner of checkpoint shapes,
frequency policy, KV capacity and transactions.

## Reproduce the downstream boundary

Follow [the snapshot and Metal test instructions](testdata/README.md#experimental-metal-ffn-parity).
On the measured Apple M3 Mac15,13 (24 GiB, 10 GPU cores, macOS 26.5.2
25F84, SDK 14.4 and Go 1.27.3), use `GOWORK=off`, `CGO_ENABLED=1`,
`GOARM64=v8.0`, and `GOMAXPROCS=8`. Runtime MSL compilation works.
The offline `metal` compiler is unavailable on this host.

Build the tagged benchmark once, then run the two named benchmarks in
separate fresh processes and alternate their order:

```sh
GOWORK=off CGO_ENABLED=1 go test -c -tags=metal -o "$bin_dir/llama-metal.test" .
GOWORK=off CGO_ENABLED=1 go test -c -o "$bin_dir/llama-cpu.test" .
LLAMA_32_1B_MODEL_DIR="$model_dir" GOMAXPROCS=8 /usr/bin/time -l \
  "$bin_dir/llama-metal.test" -test.run '^$' \
  -test.bench '^BenchmarkLlama32CachedGeneration$' -test.benchtime=1x -test.benchmem
LLAMA_32_1B_MODEL_DIR="$model_dir" GOMAXPROCS=8 /usr/bin/time -l \
  "$bin_dir/llama-metal.test" -test.run '^$' \
  -test.bench '^BenchmarkLlama32MetalCachedGeneration$' -test.benchtime=1x -test.benchmem
LLAMA_32_1B_MODEL_DIR="$model_dir" GOMAXPROCS=8 /usr/bin/time -l \
  "$bin_dir/llama-cpu.test" -test.run '^$' \
  -test.bench '^BenchmarkLlama32CachedGeneration$' -test.benchtime=1x -test.benchmem
```

The committed benchmark also checks tokenization and decoded bytes. Its
reported generation metrics separate tokenizer setup and verify the model
files before starting the timed load. Local raw samples, exact paired driver,
instrumentation overlay, benchmark binary, test logs and summarizer are in
`.benchmarks/gpu-ll-01/2026-11-23-m3/` under the isolated Llama checkout.
The paired driver is reproducible there with `zsh run.zsh` from the
repository root using the script's repository-relative path. The diagnostic
overlay instruments Llama only and consumes the released dependency.
