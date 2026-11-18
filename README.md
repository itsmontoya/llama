# llama

A tiny FP32 Llama-style decoder built on
[`neuralnetwork v0.10.10`](https://github.com/itsmontoya/neuralnetwork/tree/v0.10.10).
This is the numerical baseline for a Go inference runtime. It evaluates a
complete token prefix with batch size one and returns logits for every position.

The decoder composes the tensor package's Gather, Gemm/MatMul, broadcast
arithmetic, reductions, Sigmoid, reshape/transpose, and axis Softmax. It supports
RMSNorm, ordinary adjacent-pair RoPE, causal MHA/GQA/MQA, residual blocks, and
SwiGLU. Dimensions, FFN width, epsilon, RoPE base, context limit, and tied output
weights are explicit configuration.

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

KV caching, chunked prefill with offsets, scaled/split-half RoPE, checkpoint
formats, tokenization, sampling policies, reduced precision, and practical model
performance remain future work. Synthetic fixture parity establishes this
computation only; no real Llama checkpoint compatibility is claimed. The next
backend milestone is bounded per-layer KV storage and cached/full-prefix parity.
The detailed remaining library work is tracked in [NEEDS.md](NEEDS.md).
