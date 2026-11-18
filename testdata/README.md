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
