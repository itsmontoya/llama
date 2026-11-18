# neuralnetwork requirements for a Go Llama runtime

Writing a Go-native Llama inference runtime on top of
[`github.com/itsmontoya/neuralnetwork`](https://github.com/itsmontoya/neuralnetwork)
is feasible. **v0.10.10** supplies the tensor arithmetic on a released revision.
The tiny FP32 decoder now composes existing operations with runtime-built
rotary tables and masks. The remaining library work is to make that computation
reusable, support efficient cached decoding, and handle realistic weight sizes.

This document is a proposed implementation backlog for **neuralnetwork**.
Checkpoint parsing, tokenization, model assembly, and generation belong in
**llama**. The review itself added no library implementation. Checked items in
the library backlog identify verified neuralnetwork capabilities. The subsequent
runtime numerical proof is recorded below; unchecked library tasks remain
proposals, including tasks that can initially compose existing primitives.

## Runtime milestone completed: tiny FP32 numerical proof

Implemented on **2026-11-13**, following the v0.10.10 reassessment:

- llama's module now pins `github.com/itsmontoya/neuralnetwork v0.10.10`, with no
  local replacement. `NewDecoder` copies and validates model weights;
  `Forward` evaluates full batch-one prefixes and returns owned logits.
- The reference path composes existing tensor primitives for RMSNorm, ordinary
  adjacent-pair RoPE, causal MHA/GQA/MQA, residuals, and SwiGLU. It supports
  configurable dimensions/limits and tied or untied embedding/output weights.
- Three frozen synthetic models cover head widths 2/4/6, one/two layers,
  irregular FFN widths, and single/multiple token inputs. Every traced
  intermediate and logits matches an independent Python binary64 scalar oracle
  within `atol=rtol=2e-5`; the maximum observed absolute error in the default
  build is approximately **4.85e-7**. Hashes pin the model, generator, and tokens.
- Tests cover causal prefixes, future-token isolation, invalid arguments and
  context boundaries, constructor/output ownership, and concurrent calls.
  `go run ./cmd/tiny` demonstrates next-token logits from synthetic weights.
- Validation passed with `go1.27.2 darwin/arm64`: default, cgo-disabled,
  cgo-disabled `purego`, race, and vet checks, plus module integrity verification.
  All six model/reference artifacts regenerate byte-for-byte with Python 3.14.2.

See [README.md](README.md), [decoder.go](decoder.go), and
[fixture provenance](testdata/README.md). This closes delivery step 1, not the
reusable cached-backend milestone. No neuralnetwork source was changed; the
dedicated library kernels in LLNN-01 through LLNN-04 are still open. This
runtime currently recomputes full prefixes, expands GQA heads, and materializes
attention scores. Cached decoding and real-checkpoint parity remain next.

## Review baseline

Reassessed on **2026-11-13** against **v0.10.10**, using the clean local
`../neuralnetwork` checkout and the separately cached Go module for that version:

| Revision | What was inspected | Availability implication |
| --- | --- | --- |
| [`2ffa5bd9fd2034382c49d14efc732ea4d026aa09`][reviewed-tree], tag `v0.10.10`, local `main` and recorded `origin/main` | Tensor stack, completed NN-13 CPU product adapter, subsequent tensor optimizations, and the latest arithmetic adapter change | The tensor package is consumable as `github.com/itsmontoya/neuralnetwork v0.10.10`; no branch pin is needed. |
| Cached module `github.com/itsmontoya/neuralnetwork@v0.10.10` | Origin metadata identifies the same commit and `refs/tags/v0.10.10`; module checksum `h1:CcoQQN8w9aGodz61FQGKSWtbnKzAHf/gtXbukwVUkj0=` | Consumer verification uses the tagged module without a local replacement. |

GitHub's tag/release pages could not be fetched during this review. Version
identity is supported by the local tag and cached module origin/checksum;
live remote HEAD and whether a newer release exists were not verified.
No files in neuralnetwork were changed. At reassessment time llama contained
only `NEEDS.md`. The subsequent numerical-proof implementation above adds the
module pin and synthetic decoder harness; checkpoint loading and tokenization
remain unimplemented.

The module declares Go 1.26.1. Existing CPU SIMD and the optional Darwin/cgo
Metal path can remain accelerators; the initial runtime should also work with
`CGO_ENABLED=0` and the library's `purego` build tag.

## What changed since the November 6 assessment

- **LLNN-00's library availability blocker is resolved.** NN-02 through NN-12
  are merged and tagged. Begin the runtime against v0.10.10.
- **The CPU product portion of LLNN-05 is implemented.** NN-13 now directly
  reads private tensor inputs through shared CPU product loops. Gemm stages one
  output; batched MatMul stages the complete output. These CPU paths no longer
  copy the full weight matrix into Matrix objects on each projection. The
  Metal-tagged path retains its copying adapter.
- **v0.10.10 removes another elementwise output allocation/copy.** Equal-shape
  Add/Sub/Mul now use `Matrix.ValuesInto` to publish their private result. Input
  matrix copies and the private output matrix remain. The checked-in
  [adapter report][matrix-adapters] records `[1,10]` Add falling from 336 B /
  7 allocations to 288 B / 6 allocations. These are existing measurements,
  not a new Llama benchmark.
- Broadcast metadata/addition, convolution, pooling, and ReLU have also been
  optimized. Their evidence does not establish Llama projection or attention
  throughput. The CPU product implementation still uses scalar Go loops.
- **The transformer-specific gap is essentially unchanged:** no dedicated
  RMSNorm, SiLU/SwiGLU, RoPE, causal/GQA attention, or bounded KV-cache API;
  no half/quantized tensor storage, reusable public workspace, or transformer
  device-resident execution.

The tiny FP32 reference decoder is now implemented above. The next milestone
is the reusable cached-attention backend. General tensor availability and the
old NN-13 dependency blocker no longer need to precede that work.

### Verification performed for this reassessment

An isolated temporary module required **v0.10.10**, with no `replace` directive
or Go workspace. Its consumer checked Int64 embedding Gather, bias-free Gemm
with stored `[out,in]` weights, reuse of that same tensor for embedding/output,
MatMul, broadcast Add, explicit-axis softmax, and unchanged shared weights.
It passed in default and `CGO_ENABLED=0 -tags=purego` builds.

The tagged module's `tensor` and `internal/float32product` package suites also
passed in default, `CGO_ENABLED=0 -tags=purego`, and race builds, using
`go1.27.2 darwin/arm64`. Dependencies were already cached; checks used
`GOWORK=off GOPROXY=off GOSUMDB=off` and a temporary build cache. This verifies
the pinned local artifact, without a fresh network/checksum-database lookup.
The full repository suite and Metal execution were not rerun. No whole-model
or generation-performance result is established by these checks.

## Initial target and ownership

Start with a **dense, decoder-only Llama 2/3-style transformer**, batch size one,
FP32 CPU execution, explicit context limits, prompt prefill, and one-token
decode. Use a tiny deterministic model first, then one explicitly selected
checkpoint. This scope does not establish compatibility with every model bearing
the Llama name. Multimodal layers, mixtures of experts, training, and adapters
require separate work.

The architecture reference is Meta's
[Llama 3 implementation][meta-llama3], with
[Llama 2][meta-llama2] and the newer
[scaled-RoPE implementation][meta-scaled-rope] for variant differences.
The runtime must resolve configuration rather than hardcode dimensions,
normalization epsilon, vocabulary size, context size, or RoPE parameters.

| neuralnetwork owns | llama owns |
| --- | --- |
| Numeric storage, weight representations, forward kernels, shape/byte planning, bounded scratch, and optional acceleration | Model configuration, checkpoint loading and tensor-name mapping, decoder block assembly, and checkpoint compatibility |
| Generic cache storage/update/read operations usable by attention | Per-generation sessions, per-layer cache ownership, positions, reset policy, cancellation, and scheduling |
| Embedding lookup arithmetic and the output projection | Tokenizer, special-token policy, prompt/chat formatting, sampling, stop conditions, streaming, CLI/API |
| Kernel/reference fixtures and backend parity | Whole-model logits and token parity, model artifact provenance, and end-to-end performance |

Use additive forward-only APIs in `tensor` or a focused inference subpackage.
Keep neuralnetwork independent of llama, ONNX, checkpoint formats, and tokenizers.
Existing `layer.Layer`, training APIs, and serialized Sequential models should
retain their contracts. A general graph executor or automatic differentiation
engine is not required: llama can explicitly execute its decoder blocks.

## Capabilities to reuse

These observations refer to v0.10.10 unless otherwise stated.

| Area | Existing source | Implication |
| --- | --- | --- |
| Typed tensors | [`tensor/tensor.go`][tensor-storage], [`data_type.go`][tensor-types], and the [tensor contract][tensor-design] | Owned contiguous row-major FP32 and Int64 storage, concrete shapes, scalars/empty values, checked sizes, copying constructors/accessors. No FP16, BF16, or quantized storage. |
| Linear algebra | [`tensor/matmul.go`][tensor-matmul], [`gemm.go`][tensor-gemm] | Batched/broadcast MatMul and rank-two Gemm with transpose flags already exist. Bias-free `[out,in]` projections use `GemmConfig{TransposeB: true, Alpha: 1}` and nil bias. No trainable Dense layer is needed. |
| Arithmetic and normalization ingredients | [`arithmetic.go`][tensor-arithmetic], [`reduction.go`][tensor-reduction], [`math.go`][tensor-math] | Broadcast Add/Sub/Mul/Div, reductions, square root, and powers can compose residual addition and RMSNorm. Dedicated RMSNorm is missing. |
| Activations and softmax | [`activation.go`][tensor-activation], [`softmax.go`][tensor-softmax] | Sigmoid plus multiplication can compose SiLU. Stable softmax accepts an explicit axis. Dedicated SiLU/SwiGLU and attention masking are missing. |
| Embeddings and transforms | [`gather.go`][tensor-gather], [`reshape.go`][tensor-reshape], [`transpose.go`][tensor-transpose], [`concat.go`][tensor-concat] | Gather already implements embedding lookup. Reshape/transpose/concat materialize outputs; they do not provide cheap views or cache append. |
| Existing inference sessions | [`model/inference_session.go`][inference-session] | Separate sessions already share parameters and isolate mutable Sequential state. Their reused-output and concurrency contracts are useful precedents; they do not implement transformer KV caches. |
| Training-oriented Dense | [`layer/dense.go`][dense], [`optimizer/parameter.go`][parameter] | Dense carries bias and forward/backward state; Parameter copies values and allocates gradients. Avoid these containers for pretrained inference weights. |
| CPU and device execution | [SIMD design][simd-design], [Metal design][metal-design] | Acceleration already exists. The resident device graph targets Dense/ReLU/Softmax, rather than a transformer block. Existing SIMD support does not establish optimized Llama GEMV/GEMM. |
| CPU tensor products | [`matrix_product_host.go`][tensor-products-host], [`internal/float32product/product.go`][cpu-products], [NN-13 handoff][nn13] | Direct read-only input slices eliminate per-call weight copies. Products still allocate output staging and copy to the destination; there is no reusable public workspace or prepared-weight API. |
| Other execution costs | [`matrix_product.go`][tensor-products], [`arithmetic.go`][tensor-arithmetic], [`activation.go`][tensor-activation] | Metal-tagged products retain the Matrix adapter. Equal-shape Add/Sub/Mul retain two input copies and a private result, but avoid the extra `Values()` allocation. Sigmoid/Tanh still use copying adapters. `Into` does not imply allocation-free execution. |
| Persistence | [`model/serialization_binary.go`][binary-format] | Compact Sequential serialization already exists. It is not a Llama checkpoint reader or a packed quantized-weight format. |

Embedding, residual addition, general tensor storage, basic matrix multiplication,
and softmax should therefore **not** be treated as entirely missing features.
The principal work is the contracts and execution paths below.

## Decoder computation the backend must support

Let `B` be batch size, `T` the incoming chunk length, `D` model width, `F` FFN
width, `Hq` query heads, `Hkv` KV heads, and `Dh = D/Hq`. Use stored projection
weights shaped `[output, input]` in this document:

```text
tokens [B,T] -> gather embedding [vocab,D] -> x [B,T,D]

for each decoder block:
    n = RMSNorm(x, attention_norm_weight, epsilon)
    q = linear(n, Wq) -> [B,T,Hq,Dh]
    k = linear(n, Wk) -> [B,T,Hkv,Dh]
    v = linear(n, Wv) -> [B,T,Hkv,Dh]
    q, k = RoPE(q, k, absolute_positions, resolved_rope_parameters)
    append rotated k and unrotated v to this layer's KV cache
    a = causal_grouped_attention(q, populated_cache) -> [B,T,Hq,Dh]
    h = x + linear(merge_heads(a), Wo)
    n = RMSNorm(h, ffn_norm_weight, epsilon)
    x = h + linear(SiLU(linear(n, Wgate)) * linear(n, Wup), Wdown)

hidden = RMSNorm(x, final_norm_weight, epsilon)
logits = linear(selected_hidden_rows, output_weight) -> [...,vocab]
```

The runtime chooses whether to project every token or only the final prompt
position. Return logits before vocabulary softmax so sampling policy remains in
llama. The loader resolves weight orientation and any tied embedding/output
storage explicitly.

## Library backlog

`LLNN-*` identifiers are local to this Llama backlog and do not rename the
existing ONNX-oriented `NN-01` through `NN-13` tasks.

**P0** establishes a reusable, correct cached FP32 backend. A tiny mathematical
prototype can use compositions before dedicated kernels land. **P1** makes
selected real model sizes practical on a measured CPU/memory budget. **P2**
adds optional acceleration and serving throughput. Status below distinguishes
released capabilities from the remaining acceptance work.

| ID | Priority | Work | v0.10.10 assessment |
| --- | --- | --- | --- |
| LLNN-00 | P0 | Make the existing tensor stack consumable | **Satisfied.** Tagged tensor package is available and llama now pins v0.10.10. |
| LLNN-01 | P0 | Forward RMSNorm and SiLU/SwiGLU | **Open.** Compose existing operations for the first numerical proof. |
| LLNN-02 | P0 | Rotary position kernel and explicit layout contract | **Open.** Explicit tables plus gather/arithmetic can serve a tiny prototype. |
| LLNN-03 | P0 | Causal attention with grouped KV heads | **Open.** MatMul/masks/softmax provide ingredients, not cached attention semantics. |
| LLNN-04 | P0 | Bounded incremental KV storage and access | **Open.** No efficient public append/prefix-read storage contract. |
| LLNN-05 | P1 | Reusable immutable weights, projections, and workspace | **Partly satisfied.** Bias-free projections, tied FP32 tensor reuse, and removal of CPU weight copies exist. Workspace, transform/load copies, and preparation remain. |
| LLNN-06 | P1 | Reduced-precision and packed quantized weights | **Open.** Storage types remain FP32 and Int64. |
| LLNN-07 | P1 | Bounded-memory prefill attention | **Open.** Chunked/tiled attention and scratch planning still need implementation. |
| LLNN-08 | P1 | Representative CPU kernels and performance evidence | **Open for Llama.** Generic benchmarks and CPU adapter evidence exist; generation workloads are unmeasured. |
| LLNN-09 | P0/P1 | Reference conformance and integration handoff | **Generic foundation exists.** Transformer fixtures, cached decoder parity, and real-model evidence remain. |
| LLNN-10 | P2 | Transformer device residency and advanced execution | **Open, optional.** Portable CPU execution remains the baseline. |

### LLNN-00: Consume the existing tensor work

- [x] A reviewed, tagged revision contains NN-02 through NN-12:
  `github.com/itsmontoya/neuralnetwork v0.10.10`.
- [x] Existing tensor ownership, disjoint destinations, error atomicity, and
  concurrent-read contracts are documented and tested.
- [x] Add llama's `go.mod` with the exact v0.10.10 dependency pin. The runtime
  numerical proof now consumes that tag directly.
- [ ] Preserve those contracts in the new forward-only transformer APIs and
  document their extension boundary. No changes to CNN/RNN training contracts
  or Sequential persistence are needed for this milestone.

**Acceptance:** a minimal consumer can import the tensor package and exercise
Int64 Gather, FP32 products, broadcasting, and axis softmax on the chosen
revision in ordinary and cgo-disabled builds. Both the initial temporary
consumer and the subsequent synthetic decoder use the tagged package.

### LLNN-01: RMSNorm, SiLU, and SwiGLU

- [ ] Add a destination-writing RMSNorm over the final feature axis:
  `y[...,i] = x[...,i] * weight[i] / sqrt(mean(x*x, last_axis) + epsilon)`.
  Require a matching width and positive finite epsilon. Use FP32 reduction for
  the initial path and at least FP32 accumulation for future low-precision input.
  This normalization does not subtract a mean or use running statistics.
- [ ] Add destination-writing SiLU, `x * sigmoid(x)`, using the existing stable
  sigmoid formula. Define finite-extreme and nonfinite input behavior.
- [ ] Make the gated operation `SiLU(gate) * up` available without training state.
  It can initially be SiLU plus Mul; fuse it only when measurements justify it.
  The runtime supplies the actual FFN width, rather than assuming `4*D`.

**Acceptance:** independently generated reference vectors cover multiple token
rows, irregular widths, zeros, large finite values, invalid shapes/epsilon, and
destination ownership. Composite implementations are acceptable first; backward
passes and optimizer integration are unnecessary. RMSNorm follows
[Meta's normalization][meta-llama2]; the gated block follows
[its Llama 3 feed-forward path][meta-llama3].

### LLNN-02: RoPE with checkpoint-compatible coordinates

- [ ] Add a forward rotary kernel for query/key head tensors and explicit
  absolute positions, with validated even rotary width and supported shapes.
  Rotate Q and K; do not rotate V. A single input/destination pair per call is
  sufficient, with the runtime applying it to Q and K.
- [ ] Accept resolved sine/cosine tables or a validated numeric configuration.
  A table-based interface avoids requiring a general tensor trigonometry API.
  Plan/share tables once instead of recomputing them for every layer and token.
- [ ] Define the rotation pairing convention explicitly. Meta's adjacent-pair
  representation and split-half checkpoint conventions require compatible Q/K
  weight conversion or a corresponding kernel mode. Shape agreement alone is
  insufficient; record that conversion in llama's loader.
- [ ] Support ordinary RoPE first. Add the selected checkpoint's scaled-RoPE
  rule before claiming that model variant works. Resolve base, scaling factors,
  and original context length from metadata rather than hardcoding a universal
  Llama default. Unsupported variants must produce an explicit runtime error.

**Acceptance:** reference fixtures cover nonzero offsets, chunk boundaries,
Q/K head-count differences, position limits, both supported layout/conversion
paths, and scaled-frequency transition regions. Full-sequence and chunked
rotation agree within a documented FP32 tolerance. See
[Meta's rotary and scaling implementation][meta-scaled-rope] and
[model arguments][meta-args].

### LLNN-03: Causal scaled dot-product attention and GQA

- [ ] Provide a forward attention operation with an explicit shape/layout
  contract. One useful boundary is Q `[B,Tq,Hq,Dh]`, K/V
  `[B,Tk,Hkv,Dh]`, output `[B,Tq,Hq,Dh]`, and a query start position.
- [ ] Compute `softmax(Q*K^T / sqrt(Dh) + mask) * V`. Validate
  `Hq % Hkv == 0`; route query head `h` to KV head
  `h / (Hq/Hkv)`. Cover ordinary multi-head attention (`Hq == Hkv`),
  grouped-query attention, and the single-KV-head case.
- [ ] Apply the causal condition using absolute positions: with a cache starting
  at position zero, chunk query `i` may read key `j` only when
  `j <= startPosition+i`. An unshifted triangular mask is incorrect after
  the first chunk. Read populated cache positions only.
- [ ] Normalize over the key/time axis with max subtraction and FP32
  accumulation. Reject an invalid mask configuration that leaves a computed
  query with no valid keys, before writing outputs; do not silently return NaNs.
- [ ] Keep a straightforward copying/reference path. The practical GQA path
  should select shared KV heads directly instead of expanding the whole cache
  to the query-head count. Padding/per-row lengths need an explicit contract
  when batched prompts are introduced; they are not implicit in tensor shapes.

**Acceptance:** compare against an independent reference for full prefill,
single-token decode, multi-token cached chunks, MHA/GQA, and masked future
tokens. Changing a future token must not change earlier outputs. Existing
axis softmax is reusable, but by itself does not establish these semantics.
[Meta's attention implementation][meta-llama3] is the architecture reference.

### LLNN-04: KV cache without copying the history on append

Current tensors have private fixed-size storage and no range-write/read-view
API. `ConcatInto` creates a complete new logical value; `GatherInto` copies its
selection. Neither is an efficient growing cache.

- [ ] Add a generic bounded cache container or validated range-write plus
  read-range primitives. A dedicated cache type is the smaller initial scope;
  it need not introduce arbitrary mutable strided views.
- [ ] Allocate per-layer K/V capacity once, for example `[B,maxContext,Hkv,Dh]`.
  Append only incoming rows, track populated length, and let attention consume
  that populated prefix without materializing or copying the complete prefix.
  Cache population work should scale with the appended payload; attention's
  necessary reads of history are a separate cost.
- [ ] Expose checked capacity/byte planning before allocation. Validate positions,
  shapes, types, and `start+count` before writes. Preserve the existing prefix
  and logical length on argument errors; never attend to unused capacity.
- [ ] Support reset and tail truncation so llama can discard a conversation or
  roll back an uncommitted chunk. The initial policy can reject context overflow;
  sliding-window eviction is a separate feature with different semantics.
- [ ] Specify single-session mutation and read lifetimes. Separate sessions share
  weights, but never mutable cache/scratch. Avoid global caches and pools that
  allow one session to overwrite another's history.

**Acceptance:** full-prefix logits match prefill-plus-decode logits in the runtime
harness. Library tests verify append boundaries, reset/truncate, capacity errors,
prefix preservation, independent caches, and constant-capacity storage over many
appends. llama owns committing a token across all layers; on a later layer
failure it must truncate/reset all affected caches before reusing the session.

### LLNN-05: Immutable inference weights, projections, and workspace

**Partially implemented in v0.10.10.** Revise this task around the remaining
output/workspace/transform/load costs; per-call CPU weight copies are resolved.

- [x] Support rank-two FP32 bias-free projections with stored `[out,in]`
  weights, caller-owned output, and one or many input rows, without optimizer,
  gradient, or training state. Existing Gemm provides this boundary; set
  `TransposeB: true, Alpha: 1`, with nil bias. Its zero config scales by zero.
- [x] Eliminate per-call full CPU weight copies. The host product adapter reads
  private tensor slices directly. This does not apply to Metal-tagged products.
- [x] Allow the same read-only FP32 Tensor to serve Gather and output Gemm when
  weights are tied. Validate `0 <= tokenID < vocab` in llama because generic
  Gather also accepts negative indices.
- [ ] Reduce measured remaining elementwise/activation adapter copies.
  v0.10.10 removes the extra equal-shape arithmetic output allocation, but
  Add/Sub/Mul still copy inputs into matrices and allocate a private result;
  Sigmoid/Tanh still copy through their adapters. Keep reference parity.
- [ ] Evaluate preparing/packing weights once for the selected projection
  workloads. A new prepared-weight API is justified by measurements, not needed
  merely to share existing immutable FP32 tensors.
- [ ] Give callers explicit workspace sizing/reuse and output lifetime rules.
  Keep workspace per session/call; distinguish capacity from current token count.
  Support head reshaping, permutations, and selected final-token rows without
  requiring large copies merely to change metadata or select a contiguous range.
  Introduce any borrowed/read-only views under a separately documented lifetime
  and overlap contract rather than weakening existing Tensor ownership.
- [ ] Provide bounded bulk initialization or ownership transfer if measured load
  peaks require it. Constructors currently copy; a decoder buffer plus tensor
  copy plus packed copy can materially exceed final model size. Streaming into
  owned private storage is a valid first solution. Memory mapping is optional.

**Acceptance:** warmed projection makes no allocation/copy proportional to the
full weight size per token. Shared weights remain unchanged under concurrent
sessions. Record initialization peaks, packed-weight memory, copied payload
bytes, allocations, and numerical parity before/after optimization.

[NN-13][nn13] is complete for its measured CPU product adapter. Its ONNX RT-12
dependency has been resolved, and its evidence is available in the
[published measurement report][nn13-measurements]. Reuse that implementation;
justify further packing/workspace work with Llama fixtures. Gemm/MatMul still
allocate output staging each call to preserve destination-on-error behavior.

### LLNN-06: Lower-precision and quantized weight execution

- [ ] Define FP16/BF16 weight representation and conversion semantics if needed
  for the selected artifact. Loading either into FP32 is an acceptable first
  runtime; native half-precision arithmetic is not a correctness prerequisite.
  Retaining two-byte weights is a separate backend capability, with FP32
  accumulation and portable conversion/fallback behavior.
- [ ] For a practical quantized runtime, choose one precisely specified packed
  format first. If llama adopts GGUF, Q8_0 followed by Q4_0 is a reasonable
  incremental path; GGUF parsing alone does not implement those kernels.
- [ ] Specify block sizes, scales/zero points where applicable, row alignment,
  tails, checked byte counts, and supported operand combinations. Implement
  packed-weight GEMV and prefill multiplication, dequantizing tiles/blocks
  instead of expanding the entire model to FP32 on every call.
- [ ] Support packed embedding row lookup and a tied output projection when
  applicable, or document an explicit mixed-storage representation. Norm weights
  and intermediate arithmetic can remain FP32 initially.

**Acceptance:** fixed known blocks decode exactly as the chosen format defines;
products match a dequantized reference within format-specific tolerances. Report
model memory, scratch, throughput, and full-model logit/quality differences from
FP32. Unsupported tensor encodings fail explicitly. The
[GGUF specification][gguf] describes container metadata and tensor types; llama
owns reading it and resolving its dimensions into backend layout.

### LLNN-07: Attention with bounded prefill memory

- [ ] Add blocked/tiled attention with stable online softmax, or another measured
  implementation that avoids retaining an entire `[B,Hq,Tq,Tk]` score matrix.
  Do not require a GPU FlashAttention implementation for the CPU milestone.
- [ ] Expose scratch planning and support chunked prefill with a populated cache.
  GQA must continue to avoid full KV-head expansion.

**Acceptance:** outputs match LLNN-03's reference for multiple chunk sizes,
including a partial final chunk. Demonstrate peak workspace versus context
length; distinguish scratch from persistent KV storage. The runtime checks its
memory budget before launching a chunk.

### LLNN-08: CPU performance that represents generation

- [ ] Benchmark decode GEMV, prompt GEMM, normalization, rotary transforms,
  embedding lookup, cache append, attention, and the vocabulary projection.
  Use widths and head counts from the selected checkpoint, plus small fixtures.
- [ ] Profile allocation/copy costs separately from arithmetic. The host tensor
  path now calls [`internal/float32product`][cpu-products] directly; its product
  kernels use scalar nested loops. Generic SIMD helpers, optimized Conv/ReLU,
  and the old copying-adapter profile do not establish projection performance.
  Measure decode GEMV and prefill GEMM separately, including `[out,in]` layout.
- [ ] Add tiling, prepared weights, SIMD, or bounded parallelism only where those
  profiles show a useful improvement. Bound worker counts and avoid nested
  per-layer/per-session oversubscription. Keep the portable fallback.

**Acceptance:** publish load time, first-token latency, prompt tokens/second,
decode tokens/second, bytes/op, allocations/op, and peak live memory. Identify
hardware, Go/library/runtime revisions, build tags, cgo, thread settings,
checkpoint hash, quantization, and prompt/context sizes. Report prefill and
decode separately. Establish a hardware-specific target after the baseline;
this review makes no throughput claim.

### LLNN-09: Conformance and a usable backend handoff

**Existing foundation:** checked-in independent fixtures and ownership/error/
concurrency tests cover the generic tensor operations. The tagged package
suites passed during this review. Those fixtures target arithmetic, products,
indexing, transforms, activations, reductions, and other existing operations;
they do not yet cover Llama normalization, rotation, attention, or cached logits.

- [ ] Check in small independently generated numerical fixtures for the new
  kernels, recording reference implementation revision, configuration, dtype,
  input/weight hashes, and generator versions. Pin those references when
  implementing; the architecture links here are not frozen fixture provenance.
- [ ] Validate shapes, checked sizes, invalid arguments, destination aliases,
  unchanged inputs, retained outputs, and separate concurrent sessions. Existing
  `Into` calls require disjoint destinations; any new in-place or cache mutation
  operation must explicitly define its different contract.
- [ ] Cover no-cache, full prefill, chunked prefill, and cached decode against
  the same tiny model. Keep the whole-model harness in llama and numeric fixtures
  in neuralnetwork, avoiding a library-to-runtime dependency.
- [ ] Set explicit absolute/relative FP32 tolerances and separate quantized
  error budgets. Compare logits and intermediate block outputs; generated text
  alone can conceal a wrong mask, rotation layout, or cache offset.
- [ ] Run relevant package tests and existing compatibility tests, then default,
  `CGO_ENABLED=0`/`purego`, and race checks for implemented backend changes.
  Exercise Metal parity on supported hardware if that path changes.

**Acceptance:** hand off the exact consumable library revision, public API and
ownership documentation, passing checks, and a minimal cached-decoder example.
For P1, add one real checkpoint with loader/tokenizer parity and reproducible
resource/performance evidence. For cancellation, document the current kernel
granularity: llama can stop between chunks/blocks, while existing synchronous
kernels do not promise in-kernel interruption.

### LLNN-10: Optional accelerators and throughput extensions

- [ ] Extend device execution to keep projections, normalization, RoPE,
  attention, gating, and KV state resident across a decoder block. A sequence of
  GPU multiplications separated by full host copies is not sufficient evidence
  of a useful device path.
- [ ] Define host/device coherence, cache lifetime, fallback synchronization,
  packed-weight support, session isolation, and resource release. Keep
  `internal/device` private unless a reviewed public abstraction is necessary.
- [ ] Consider batched decode, per-row lengths, cache-slot updates, or paged
  cache storage after the batch-one implementation is measured. Continuous
  batching, prefix reuse, and request scheduling remain llama responsibilities.

**Acceptance:** full-block parity and end-to-end evidence against the CPU path,
including transfer volume and memory. Metal/CUDA/Vulkan, multi-device execution,
and speculative decoding are separate optional projects, not initial blockers.

## Memory makes the priority order concrete

The following are calculations, not benchmark results or a claim about the exact
parameter count of any named checkpoint. For `P` stored parameters, weight
payload is approximately `P * bytesPerWeight`, before metadata and packing:

| Illustrative 8 billion parameters | Payload only |
| --- | --- |
| FP32 | 32 GB / 29.8 GiB |
| FP16 or BF16 | 16 GB / 14.9 GiB |
| Ideal 4-bit storage | 4 GB / 3.7 GiB, plus block scales and other overhead |

A full FP32 gradient would add another roughly 29.8 GiB at that size, which is
why pretrained inference weights should avoid `optimizer.Parameter`.

For ordinary full-context attention, an uncompressed cache needs:

```text
KV bytes = 2 * layers * batch * context * KVHeads * headDim * bytesPerElement
```

For 32 layers, batch one, 8,192 tokens, 8 KV heads, and head dimension 128, that
is **2 GiB in FP32** or **1 GiB in FP16**, in addition to weights and scratch.
Materializing one FP32 attention-score tensor with 32 query heads and an
8,192-token square prompt adds **8 GiB**. GQA cache storage must use KV heads,
not query heads; tiled/chunked prefill addresses the score-matrix peak.

Budget weights, prepared copies, cache, activations, attention workspace, output
logits, loader buffers, and concurrent sessions separately. Checked integer
arithmetic alone is not a memory budget. This explains why LLNN-05 through
LLNN-08 matter before promising useful inference at multi-billion-parameter
sizes, even though a tiny FP32 proof needs none of their optimizations.

## Runtime work required alongside this library backlog

These are integration dependencies, not requests to add unrelated facilities to
neuralnetwork:

- Select one exact model/configuration and one initial artifact format. A
  converted FP32 artifact or FP32 conversion while reading
  [safetensors][safetensors] is sufficient initially. GGUF is a separate loader
  option; neither existing Sequential format reads these checkpoints.
- Validate tensor names, shapes, dtypes, FFN width, head counts, norm/RoPE
  settings, tied weights, and all file/shard offsets. Convert checkpoint weight
  orientation and rotary layout deliberately. Reject unsupported architecture
  features before inference rather than guessing from a model filename.
- Implement the selected tokenizer exactly. The
  [Llama 2 reference uses SentencePiece][tokenizer-llama2]; the
  [Llama 3 reference uses tiktoken BPE][tokenizer-llama3]. Match special tokens,
  encode/decode behavior, and chat formatting for the chosen checkpoint.
- Assemble the decoder, own one mutable session/cache per generation, and
  enforce context/output/memory limits. Start with greedy decoding, then add
  seeded sampling, temperature/top-k/top-p, stop tokens, and streaming as needed.
- Compare tokenizer IDs, intermediate tensors, next-token logits, and cached
  generation with the selected reference. Checkpoint permissions/provenance and
  artifact downloading remain runtime/distribution concerns.

## Suggested delivery sequence

1. **Numerical proof (complete):** llama now pins **v0.10.10** and evaluates
   tiny deterministic FP32 models using existing tensor primitives, explicit
   rotary tables, and masks. Frozen independent references verify intermediate
   tensors and logits. See the runtime milestone above.
2. **Reusable cached backend (next):** deliver LLNN-01 through LLNN-04 with the P0
   conformance work. Prove full-prefix, chunked-prefill, and decode equivalence.
3. **One real checkpoint:** implement its loader/tokenizer/configuration in
   llama, then complete LLNN-05's remaining workspace, transform, and load-memory
   work as measurements require. CPU full-weight copying is already resolved.
   Add that checkpoint's scaled RoPE before claiming compatibility if needed.
4. **Practical CPU execution:** complete the selected precision/quantization,
   attention-memory, and measured CPU work from LLNN-06 through LLNN-09.
5. **Optional acceleration:** pursue LLNN-10 only against an established
   correctness and performance baseline.

[reviewed-tree]: https://github.com/itsmontoya/neuralnetwork/tree/2ffa5bd9fd2034382c49d14efc732ea4d026aa09
[tensor-storage]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/tensor.go
[tensor-types]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/data_type.go
[tensor-design]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/docs/tensor-design.md
[tensor-matmul]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/matmul.go
[tensor-gemm]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/gemm.go
[tensor-arithmetic]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/arithmetic.go
[tensor-reduction]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/reduction.go
[tensor-math]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/math.go
[tensor-activation]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/activation.go
[tensor-softmax]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/softmax.go
[tensor-gather]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/gather.go
[tensor-reshape]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/reshape.go
[tensor-transpose]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/transpose.go
[tensor-concat]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/concat.go
[tensor-products]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/matrix_product.go
[inference-session]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/model/inference_session.go
[dense]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/layer/dense.go
[parameter]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/optimizer/parameter.go
[simd-design]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/docs/simd.md
[metal-design]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/docs/metal.md
[binary-format]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/model/serialization_binary.go
[nn13]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/docs/handoffs/NN-13.md
[meta-llama2]: https://github.com/meta-llama/llama/blob/main/llama/model.py
[meta-llama3]: https://github.com/meta-llama/llama3/blob/main/llama/model.py
[meta-scaled-rope]: https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py
[meta-args]: https://github.com/meta-llama/llama-models/blob/main/models/llama3/args.py
[tokenizer-llama2]: https://github.com/meta-llama/llama/blob/main/llama/tokenizer.py
[tokenizer-llama3]: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
[gguf]: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
[safetensors]: https://github.com/safetensors/safetensors

[tensor-products-host]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/tensor/matrix_product_host.go
[cpu-products]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/internal/float32product/product.go
[matrix-adapters]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/Benchmarks_matrix_adapters.md
[nn13-measurements]: https://github.com/itsmontoya/neuralnetwork/blob/2ffa5bd9fd2034382c49d14efc732ea4d026aa09/docs/performance/nn13/README.md
