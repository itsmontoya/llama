# Tiny decoder fixtures

These are original synthetic models and numerical references. The architecture
follows the RMSNorm, rotary attention, residual, and gated feed-forward equations
in [Meta's Llama 3 model](https://github.com/meta-llama/llama3/blob/main/llama/model.py).
The references are produced by our independent scalar oracle, not by running
Meta's implementation or a real checkpoint.

| Model | Dim / FFN | Layers | Q / KV heads | Head width | Output weights |
| --- | --- | --- | --- | --- | --- |
| `tiny_gqa` | 16 / 19 | 2 | 4 / 2 | 4 | Tied to embedding |
| `tiny_mha` | 8 / 11 | 1 | 4 / 4 | 2 | Separate |
| `tiny_mqa` | 18 / 17 | 1 | 3 / 1 | 6 | Separate |

Each model has a multi-token case and a single-token case. All projection
matrices use `[output,input]`; rotary coordinates pair adjacent features and
rotate Q/K only. Trace shapes are `[T,D]` for hidden states and merged attention,
`[T,H,Dh]` for Q/K/V and rotated heads, `[H,T,T]` for probabilities, `[T,F]` for
FFN intermediates, and `[T,V]` for logits. Positions start at zero. The oracle
never reads future keys, whereas Go applies an explicit negative-infinity mask.
The oracle selects each head's KV group directly; Go materializes head expansion.

Weights use SHA-256-derived dyadic values; norm weights are centered on one.
There is no random seed or external model download. Python's `struct` rounds
weights and epsilon to binary32. The scalar oracle then uses binary64
`math.fsum` reductions/products, complex-number rotary multiplication, and
per-query attention loops. It does not import Go, neuralnetwork, NumPy, or Torch.
This independent arithmetic is compared with the composed Go FP32 tensor path,
with absolute and relative tolerances of `2e-5` each. Frozen arrays are not
copied from the implementation under test.

Regenerate from the repository root with **CPython 3.14.2**:

```sh
python3 testdata/generate_reference.py
go test -run TestDecoderReference -v ./...
```

Each `.reference.json` records the generator's SHA-256, the complete corresponding
`.model.json` file's SHA-256, Python version, arithmetic, tolerances, and SHA-256
of each input token sequence encoded as little-endian signed int64. Tests verify
these hashes before using expected values. The generator's content hash pins
the exact reference implementation; the Meta link is an architecture reference.
The script prints model/reference file hashes and is byte-for-byte deterministic
in the recorded Python environment. Other Python versions may change metadata
or math-library rounding; review and regenerate fixtures intentionally.

## TinyStories 260K checkpoint logits

`stories260k_logits.json` contains all 512 logits at each position of the
prefixes `[1]` and `[1, 3, 42, 3]`. The test driver calls `build_transformer`
and `forward` from [llama2.c at `350e04fe35433e6d2941dce5a1f53308f87058eb`](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb),
with positions starting from zero for each prefix. The Go comparison uses an
absolute tolerance of `1e-4` plus a relative tolerance of `1e-4`.
On darwin/arm64, the maximum observed absolute errors were `4.76837158e-06`
for `[1]` and `7.62939453e-06` for `[1, 3, 42, 3]`.

Run these commands from the repository root on a little-endian host. The
checkpoint, upstream source, and compiled driver are stored in a temporary
directory outside the repository.

```sh
artifact_dir=$(mktemp -d)
model_path="$artifact_dir/stories260K.bin"
reference_dir="$artifact_dir/llama2.c"
reference_exe="$artifact_dir/stories260k_reference"

curl -fL -o "$model_path" \
  https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K/stories260K.bin
printf '%s  %s\n' b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696 \
  "$model_path" | shasum -a 256 -c

git clone https://github.com/karpathy/llama2.c.git "$reference_dir"
git -C "$reference_dir" switch --detach 350e04fe35433e6d2941dce5a1f53308f87058eb
cc -O2 -ffp-contract=off -I "$reference_dir" testdata/stories260k_reference.c -o "$reference_exe" -lm
"$reference_exe" logits "$model_path" > testdata/stories260k_logits.json

LLAMA_STORIES260K_MODEL="$model_path" go test . -run '^TestStories260KLogits$' -count=1 -v
```

## TinyStories 260K tokenizer

`stories260k_tokenizer.json` records exact prompt IDs and decoded bytes for
empty, plain-text, whitespace, and Unicode prompts. The driver uses the pinned
llama2.c revision above to encode with BOS and decode the resulting IDs after
BOS. It writes prompts and decoded bytes in hex to preserve whitespace and raw
byte-fallback output before terminal filtering.

From the repository root, keep the same temporary artifact directory and
compiled driver from the checkpoint instructions above.

```sh
tokenizer_path="$artifact_dir/tok512.bin"
curl -fL -o "$tokenizer_path" \
  https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K/tok512.bin
printf '%s  %s\n' 037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312 \
  "$tokenizer_path" | shasum -a 256 -c

"$reference_exe" tokenizer "$tokenizer_path" > testdata/stories260k_tokenizer.json
LLAMA_STORIES260K_TOKENIZER="$tokenizer_path" \
  go test ./internal/tokenizer -run '^TestStories260KTokenizerParity$' -count=1 -v
```

## TinyStories 260K text generation

`stories260k_generation.json` records the reference prompt IDs, greedy suffix
IDs, and raw decoded suffix in hex for `Once upon a time`. The driver feeds the
prompt IDs from position zero, then uses llama2.c's greedy argmax on the logits.
It generates at most 16 new tokens within the remaining context and stops before
a predicted BOS. The upstream CLI's `-n` counts forward steps, including forced
prompt tokens. The driver decodes the suffix from the final prompt ID, and the Go
test also checks that command stdout ends with the added newline.

Use the pinned artifacts and the compile command above to rebuild the driver
before regenerating the fixture.

```sh
"$reference_exe" generation "$model_path" "$tokenizer_path" > testdata/stories260k_generation.json
LLAMA_STORIES260K_MODEL="$model_path" LLAMA_STORIES260K_TOKENIZER="$tokenizer_path" \
  go test ./cmd/generate -run '^TestStories260KGeneration$' -count=1 -v
```

## Llama 3.2 1B Transformers reference

The generator uses the Meta base checkpoint `meta-llama/Llama-3.2-1B` at
revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`. The six file sizes, Git
blob IDs of the JSON files, and SHA-256 of `model.safetensors` come from the
[official revision inventory](https://huggingface.co/api/models/meta-llama/Llama-3.2-1B/revision/4e20de362430cd3b72f300e6b0f18e50e7166e08?blobs=true).

Use CPython 3.12.12 on macOS arm64 and install the hashed dependency lock.
Run from the repository root with `uv` available. The checkpoint and
environment live outside the checkout. If the snapshot is already local,
set `model_dir` to its directory and omit `--download` and the login command.

```sh
reference_dir=$(mktemp -d)
model_dir="$reference_dir/model"
uv python install 3.12.12
uv venv --python 3.12.12 "$reference_dir/env"
reference_python="$reference_dir/env/bin/python"
uv pip sync --python "$reference_python" --require-hashes --only-binary :all: \
  testdata/llama32-reference-requirements.lock

# Use a local Hugging Face account with approved access to the model.
"$reference_dir/env/bin/huggingface-cli" login
"$reference_python" testdata/generate_llama32_reference.py \
  --model-dir "$model_dir" --download --verify-only

for run in first second; do
  "$reference_python" testdata/generate_llama32_reference.py \
    --model-dir "$model_dir" --output-dir "$reference_dir/$run"
done
for name in llama32_manifest.json llama32_reference.json; do
  cmp "$reference_dir/first/$name" "$reference_dir/second/$name" || exit 1
done
# Freeze both fixtures after reviewing the generated records.
cp "$reference_dir/first/llama32_manifest.json" "$reference_dir/first/llama32_reference.json" testdata/
```

`llama32_manifest.json` records SHA-256 for each verified file, the raw and
resolved configuration, the safetensors header inventory, and the effective
fast tokenizer components. `llama32_reference.json` records the exact host
and environment, IDs and decoded bytes for tokenizer cases, the scaled
rotary frequencies and coefficients at fixed positions, and logits from
the prompt and the greedy trajectory. Each observed row records the winning
ID, the top eight logits with the top-two margin, and logits at 32 evenly
spaced vocabulary IDs plus the special IDs and top candidates. Generated
step prefixes are stored verbatim so a failed position can be replayed.

The pinned file has 146 BF16 tensors with 1,235,814,400 parameters. The
classifier shares the embedding weights. The frozen reference uses one CPU
thread for each of intra-op and inter-op work on an Apple M3 running macOS
26.5.2. Its eight-token continuation of `The capital of France is` is
`Paris. It is the most populous city`, preceded by one space.

The oracle uses CPU FP32 with eager attention and recomputes the full
prefix for each prediction. It selects the highest raw logit, breaks ties
by lowest token ID, stops before a selected EOS, and decodes only the suffix.
The fixture uses BOS insertion, no truncation or padding, and decoding with
special tokens skipped and whitespace cleanup disabled. For opt-in tests,
run `--verify-only` against the checked-in manifest before setting
`LLAMA_32_1B_MODEL_DIR` to the verified directory.

### Safetensors inventory

After verifying the snapshot against the manifest, compare the reader's
descriptors with the official file inventory:

```sh
LLAMA_32_1B_MODEL_DIR="$model_dir" \
  go test ./internal/safetensors -run '^TestLlama32SafetensorsInventory$' -count=1 -v
```

The pinned snapshot has a single `model.safetensors` file. Sharded Hugging
Face checkpoints use `model.safetensors.index.json` with a `weight_map`
from tensor names to shard filenames such as
`model-00001-of-00002.safetensors`. When a directory has only an index or
shard files, the model-directory loader should report
`sharded safetensors checkpoints are unsupported (expected model.safetensors)`.
An index-aware loader can open each resolved shard with the same single-file reader.
[Hugging Face checkpoint index format](https://huggingface.co/docs/transformers/v4.51.3/en/models#sharded-checkpoints)

### Uncached checkpoint parity

Run the official checkpoint's full-prefix logit check in one test process:

```sh
LLAMA_32_1B_MODEL_DIR="$model_dir" go test . -run '^TestLlama32Logits$' -count=1 -v
```

The test verifies the snapshot hashes before loading, checks every compared
row for finite logits and the exact winning token, and compares each frozen
sample with `abs(got - want) <= 1e-3 + 1e-3*abs(want)`. It reports the worst
absolute error and tolerance fraction with their positions and token IDs.

`LoadHFModel` validates the declared tensor inventory before reading weights.
It copies each tensor into private decoder storage. For tied models the
embedding also serves as the classifier. The pinned checkpoint's FP32 payload
is 4,943,257,600 bytes. Live staging holds one converted tensor at a time.
Q and K row permutations use a second FP32 buffer, and file reads use 64 KiB of
scratch. The largest staging tensor is the 1,050,673,152-byte embedding.
Process RSS also depends on garbage collection and runtime overhead.

### Cached decode parity

Verify the six snapshot files as above, then run the cache check in one test
process:

```sh
LLAMA_32_1B_MODEL_DIR="$model_dir" go test . -run '^TestLlama32CachedParity$' -count=1 -v
```

The test prefills the frozen six-token prompt and replays its first four
generated IDs. At each step it compares all 128,256 logits with the last row
from a fresh full-prefix forward using `atol = rtol = 1e-4`. Both paths must
select the frozen winning ID. The test checks snapshot hashes and reports
maximum absolute errors and tolerance fractions with positions and token IDs.

### Llama 3 tokenizer parity

After verifying the snapshot against the manifest, compare prompt IDs and
decoded bytes with the pinned fast tokenizer, including the generation prompt
and frozen suffix:

```sh
LLAMA_32_1B_MODEL_DIR="$model_dir" \
  go test ./internal/tokenizer -run '^TestLlama32TokenizerParity$' -count=1 -v
```

The Go tokenizer uses the Split regex's letter and number ranges from the
pinned Tokenizers 0.21.1 environment. Its Unicode classifications are frozen
independently of Go's Unicode version. Check or regenerate the table with
the same reference environment used above:

```sh
"$reference_python" -B testdata/generate_llama32_unicode.py |
  cmp - internal/tokenizer/llama3_unicode.go
```
