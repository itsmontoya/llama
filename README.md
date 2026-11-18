# llama

A tiny FP32 Llama-style decoder built on
[`neuralnetwork v0.10.10`](https://github.com/itsmontoya/neuralnetwork/tree/v0.10.10).
This is the numerical baseline for a Go inference runtime. It evaluates a
complete token prefix with batch size one and returns logits for every position.

The decoder composes the tensor package's Gather, Gemm/MatMul, broadcast
arithmetic, reductions, Sigmoid, reshape/transpose, and axis Softmax. It supports
RMSNorm, adjacent-pair RoPE with optional Llama 3 scaling, causal MHA/GQA/MQA,
residual blocks, and SwiGLU. Dimensions, FFN width, epsilon, RoPE base, context
limit, and tied output weights are explicit configuration.

## Generate text with Llama 3.2 1B

Use Go 1.26.1 or later and an authenticated Hugging Face account with access to
`meta-llama/Llama-3.2-1B` at revision
`4e20de362430cd3b72f300e6b0f18e50e7166e08`. Keep the six snapshot files outside
the checkout. Set `model_dir` to that directory and `reference_python` to the
local Python interpreter used for snapshot verification. [Pinned download and verification recipe](testdata/README.md#llama-32-1b-transformers-reference)

From the repository root, check the snapshot against the manifest and generate
an eight-token continuation:

```sh
"$reference_python" -B testdata/generate_llama32_reference.py \
  --model-dir "$model_dir" --verify-only

go run ./cmd/generate \
  -hf-model-dir "$model_dir" -context 512 \
  -prompt 'The capital of France is' -max-new-tokens 8
```

The full-prefix FP32 CPU path recomputes the prompt and its accumulated suffix
for each prediction and serves as the numerical reference baseline. The command
prints this continuation with one leading space and a trailing newline:

```text
 Paris. It is the most populous city
```

`-context` defaults to 512 in Hugging Face mode and must be positive and within
the model's advertised limit. The encoded prompt, including BOS, shares this cap
with the generated tokens. Generation stops before a selected stop token from
the checkpoint metadata. A zero budget or full-context prompt prints an empty
line. Use `-model` and `-tokenizer` together for legacy mode.

## Generate text with TinyStories 260K

From this directory, with Go 1.26.1 or later, download the pinned FP32
checkpoint and its matching 512-piece vocabulary into a temporary directory.
Verify both SHA-256 checksums before running the command.

```sh
artifact_dir=$(mktemp -d)
model_path="$artifact_dir/stories260K.bin"
tokenizer_path="$artifact_dir/tok512.bin"
artifact_url=https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K

curl -fL -o "$model_path" "$artifact_url/stories260K.bin"
curl -fL -o "$tokenizer_path" "$artifact_url/tok512.bin"
printf '%s  %s\n' \
  b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696 "$model_path" \
  037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312 "$tokenizer_path" \
  | shasum -a 256 -c
```

```sh
go run ./cmd/generate \
  -model "$model_path" -tokenizer "$tokenizer_path" \
  -prompt 'Once upon a time' -max-new-tokens 16

CGO_ENABLED=0 go run -tags=purego ./cmd/generate \
  -model "$model_path" -tokenizer "$tokenizer_path" \
  -prompt 'Once upon a time' -max-new-tokens 16
```

Both CPU builds print the continuation followed by a newline.

```text
, there was a little girl named Lily. She loved to play
```

`-max-new-tokens` defaults to 16. The encoded prompt, including its initial
BOS token, and the generated tokens share this checkpoint's 512-token context.
A prompt longer than 512 tokens is an error. Generation stops at the first BOS
prediction or when either token limit is reached. A full-context prompt or a
zero budget prints an empty line. An empty `-prompt` starts from BOS alone.

## Run the numerical example

From this directory, with Go 1.26.1 or later:

```sh
go run ./cmd/tiny
go run ./cmd/tiny -tokens 1,4,2
CGO_ENABLED=0 go run -tags=purego ./cmd/tiny
```

The command loads `testdata/tiny_gqa.model.json` and prints the last position's
logits and greedy next-token ID. Its synthetic vocabulary has no text tokenizer;
these IDs demonstrate arithmetic, not language quality. `-model` selects another
synthetic fixture in the same JSON schema.

## Decoder API

```go
decoder, err := llama.NewDecoder(config, weights)
if err != nil {
    return err
}
logits, err := decoder.Forward([]int64{1, 4, 2})
```

`Weights` contains row-major FP32 slices with matrices shaped `[output,input]`.
The constructor validates shapes and finite values and copies all weight
payloads. With `TieEmbeddings`, leave `Weights.Output` empty: the embedding
tensor also supplies the output projection.

`Forward` accepts a nonempty prefix of at most `MaxContext` token IDs, each in
`[0,VocabSize)`. Every call starts positions at zero and recomputes the entire
prefix. The decoder holds only immutable tensors, so separate calls may run
concurrently. Outputs belong to each call and survive later calls. Callers
synchronize writes to token slices and returned tensors.

`ForwardWithTrace` additionally returns named intermediate tensors for numerical
comparison. The trace includes embeddings; each block's normalized inputs,
Q/K/V, rotated Q/K, attention probabilities and merged output, residuals, gate/up
projections, SwiGLU output, and block output; final hidden state; and logits.
`trace["logits"]` is the returned logits tensor. Errors return no partial result.

## Numerical checks

```sh
go test ./...
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go test -tags=purego ./...
go test -race ./...
go vet ./...
```

Three frozen synthetic models cover MHA, GQA, MQA, head widths 2/4/6, irregular
FFN widths, one/two layers, tied/untied output weights, and single/multiple token
prefixes. Tests compare every traced intermediate and logits against an
independent Python standard-library scalar oracle. Inputs are binary32;
reference calculations use binary64, `math.fsum`, and complex rotary arithmetic.
The FP32 tolerance is `abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`.

Tests also check causal prefix equivalence, future-token isolation, constructor
copying, retained outputs, concurrent calls, invalid models/tokens, and context
limits. Go tests use checked-in JSON and need neither Python nor network access
once Go dependencies are cached. See [fixture provenance](testdata/README.md).

## Scope and next milestone

This path deliberately copies head reshapes, expands KV heads for GQA, and
materializes full attention-score matrices. Configuration checks integer/byte
representability, not a total memory budget. Finite input weights still follow
ordinary FP32 overflow/nonfinite arithmetic during execution.

KV caching, chunked prefill with offsets, sharded checkpoints, additional
tokenizer formats, stochastic sampling, reduced precision, and practical model
performance remain future work. Compatibility is verified for the pinned
TinyStories 260K and Llama 3.2 1B checkpoints and tokenizers. The next backend
milestone is bounded per-layer KV storage and cached/full-prefix parity.
The detailed remaining library work is tracked in [NEEDS.md](NEEDS.md).
