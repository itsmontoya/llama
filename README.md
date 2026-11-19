# llama

A tiny FP32 Llama-style decoder built on
[`neuralnetwork v0.10.11`](https://github.com/itsmontoya/neuralnetwork/tree/v0.10.11).
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

bin_dir=$(mktemp -d)
go build -o "$bin_dir/generate" ./cmd/generate
"$bin_dir/generate" \
  -hf-model-dir "$model_dir" -context 512 \
  -prompt 'The capital of France is' -max-new-tokens 8
```

Hugging Face mode prefills the prompt once and decodes with an FP32 KV cache.
For the full-prefix numerical baseline, pass `-uncached` to the same command:

```sh
"$bin_dir/generate" -uncached \
  -hf-model-dir "$model_dir" -context 512 \
  -prompt 'The capital of France is' -max-new-tokens 8
```

Both modes print this continuation with one leading space and a trailing newline:

```text
 Paris. It is the most populous city
```

`-context` defaults to 512 in Hugging Face mode and must be positive and within
the model's advertised limit. The encoded prompt, including BOS, shares this cap
with the generated tokens. Generation stops before a selected stop token from
the checkpoint metadata. A zero budget or full-context prompt prints an empty
line. Use `-model` and `-tokenizer` together for legacy mode.

For the cgo-disabled CPU build, run the same cached command with this binary:

```sh
CGO_ENABLED=0 go build -tags=purego -o "$bin_dir/generate-purego" ./cmd/generate
"$bin_dir/generate-purego" \
  -hf-model-dir "$model_dir" -context 512 \
  -prompt 'The capital of France is' -max-new-tokens 8
```

### Ask questions in the terminal

Run the interactive example with the same verified model directory:

```sh
go run ./cmd/example -hf-model-dir "$model_dir"
```

Type a question such as `What is the capital of France?` and press Enter.
Press Ctrl-D to exit. The example loads the model once and treats each question
as a fresh completion. It also accepts one question per line from a pipe.
Use `-max-new-tokens` to change the answer budget, which defaults to 16.

### CPU performance

The pinned prompt encodes to six tokens, including BOS. The eight-token output
budget uses one prefill and seven decode calls. The runtime cap is 512, and the
cache allocates 13 positions because the last selected token requires no decode.
The FP32 model payload estimate is 4,943,257,600 bytes and the KV payload estimate
is 851,968 bytes. Peak RSS covers the full benchmark process, including loader
scratch, tokenizer, temporary tensors, and Go runtime overhead.

The comparison uses `BenchmarkLlama32CachedGeneration` on Go 1.27.2,
macOS 26.5.2 (25F84) darwin/arm64, Apple M3 (4 performance and 4 efficiency
cores), 24 GB RAM, `CGO_ENABLED=1`, default build tags, and `GOMAXPROCS=8`.
Both builds use the same Llama source. The baseline dependency pin and
checksums come from the handoff commit. Each version runs five times in
alternating order, with one operation per fresh process and zero inference
warm-ups. Each trial verifies the snapshot first, so the load measurements
can benefit from the filesystem cache.

```sh
git show 486be90:go.mod >"$bin_dir/v0.10.10.mod"
git show 486be90:go.sum >"$bin_dir/v0.10.10.sum"
CGO_ENABLED=1 go test -mod=readonly -modfile="$bin_dir/v0.10.10.mod" \
  -c -o "$bin_dir/llama-v0.10.10.test" .
CGO_ENABLED=1 go test -mod=readonly -c -o "$bin_dir/llama-v0.10.11.test" .
for trial in 1 2 3 4 5; do
  if [ "$((trial % 2))" -eq 1 ]; then
    first_version=v0.10.10 second_version=v0.10.11
  else
    first_version=v0.10.11 second_version=v0.10.10
  fi
  for version in "$first_version" "$second_version"; do
    LLAMA_32_1B_MODEL_DIR="$model_dir" GOMAXPROCS=8 \
      /usr/bin/time -l "$bin_dir/llama-$version.test" -test.run '^$' \
      -test.bench '^BenchmarkLlama32CachedGeneration$' -test.benchtime 1x \
      -test.benchmem -test.timeout 15m -test.v
  done
done
```

`model-load-s` measures checkpoint loading. `tokenizer-setup-s` covers tokenizer
loading and prompt encoding. Cache allocation is reported separately. Prefill
returns the final prompt logit row, and first-token time adds greedy selection.
Decode throughput counts only the seven predictions after prefill, including
greedy selection. The generation timer includes session allocation and suffix
decoding. Process wall time and peak RSS also include checkpoint hashing,
model loading, and tokenizer setup. RSS for this comparison came from macOS
`wait4`, which is also used by `/usr/bin/time -l`.

| Median metric | v0.10.10 | v0.10.11 | Change |
| --- | ---: | ---: | ---: |
| Prefill / first token (s) | 18.620 | 2.085 | −88.8% |
| Decode (s/token) | 4.054 | 1.119 | −72.4% |
| Decode (tokens/s) | 0.2466 | 0.8940 | +262.5% |
| Cached generation (s/op) | 47.213 | 10.222 | −78.4% |
| Cached generation (output tokens/s) | 0.1694 | 0.7827 | +361.9% |
| Generation bytes/op | 198,717,800 | 198,717,992 | <0.01% |
| Generation allocs/op | 39,863 | 39,866 | <0.01% |
| Benchmark process wall (s) | 51.076 | 13.317 | −73.9% |
| Benchmark process peak RSS (GiB) | 6.23 | 6.37 | Overlapping ranges |

Cached generation ranged from 46.03–47.27 s with v0.10.10 and 9.82–10.81 s
with v0.10.11. Model-load times ranged from 1.81–2.87 s and 1.83–2.58 s,
respectively. The sample ranges for peak RSS were 5.74–6.37 GiB and
5.80–6.37 GiB.

In separate CPU profiles, self time in the right-transposed product and its
three-row helper fell from 37.85 s to 7.18 s. The new kernel walks contiguous
weight rows and reuses each weight load for three input rows during prefill.
[Baseline profile and optimization handoff](NEURALNETWORK_OPTIMIZATION_HANDOFF.md)

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

`decoder.NewSession(capacity)` allocates bounded FP32 KV storage for a prompt and
its continuation. Call `Prefill(tokens)` once to get the final prompt logit row,
then `Decode(token)` for each subsequent token. Both methods return caller-owned
rows. `Position()` counts committed tokens and gives the next absolute position.
A failed step leaves that position intact so it can be retried. `Reset()`
discards the history and allows a new prompt. Separate sessions can share the
same decoder concurrently. Callers synchronize use of each session.

`decoder.GenerateGreedyCached(prompt, maxNewTokens, stopIDs...)` handles prefill,
incremental decode, and greedy selection for a caller-owned continuation.
`decoder.GenerateGreedy` uses full-prefix execution with the same token budget,
stop IDs, and lowest-ID tie break.

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

## Scope and follow-up work

Full-prefix and prefill paths copy head reshapes, expand KV heads for GQA,
and materialize full attention-score matrices. Single-token decode reads its
session's populated cache directly and uses one score row of scratch storage.
Configuration checks that tensor and cache sizes fit their integer indices.

Chunked prefill with offsets, sharded checkpoints, additional tokenizer formats,
stochastic sampling, and reduced precision remain future work. Compatibility is
verified for the pinned TinyStories 260K and Llama 3.2 1B checkpoints and
tokenizers.
The detailed remaining library work is tracked in [NEEDS.md](NEEDS.md).
