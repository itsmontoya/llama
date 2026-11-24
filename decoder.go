package llama

import (
	"context"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/gpu"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

// NewDecoder validates and copies the supplied weights into private tensors.
// The resulting decoder is read-only and may be shared across Forward calls.
func NewDecoder(config Config, weights Weights) (*Decoder, error) {
	if err := config.validate(); err != nil {
		return nil, err
	}
	if int64(len(weights.Layers)) != config.Layers {
		return nil, fmt.Errorf("llama: got %d weight layers, want %d", len(weights.Layers), config.Layers)
	}
	if config.TieEmbeddings && len(weights.Output) != 0 {
		return nil, fmt.Errorf("llama: tied embeddings require empty output weights")
	}
	decoder := &Decoder{config: config, blocks: make([]blockWeights, len(weights.Layers))}
	if err := decoder.loadWeights(context.Background(), weights, func(_ int, weight decoderWeight) (string, []float32, error) {
		return weight.name, weight.values, nil
	}); err != nil {
		return nil, err
	}
	return decoder, nil
}

// Decoder owns immutable model weights for batch-one inference. Its zero value
// is invalid.
type Decoder struct {
	config                  Config
	embedding, norm, output *tensor.Tensor
	blocks                  []blockWeights
	metal                   *metalModel
	metalEmbedding          *gpu.Tensor
	metalNorm, metalOutput  *gpu.Tensor
}

// Config returns the decoder's resolved configuration by value.
func (d *Decoder) Config() Config {
	return d.config
}

// Forward returns owned logits [len(tokens),VocabSize], before vocabulary
// softmax. Tokens must be in [0,VocabSize), with 1 <= len <= MaxContext.
// Positions start at zero on every call. Outputs survive later calls; callers
// must synchronize any mutation of their token slice while Forward reads it.
func (d *Decoder) Forward(tokens []int64) (*tensor.Tensor, error) {
	return d.ForwardContext(context.Background(), tokens)
}

// ForwardContext is Forward with cancellation. An outstanding Metal command
// drains before the call returns and releases its scratch. CPU tensor
// operations finish before the next cancellation check.
func (d *Decoder) ForwardContext(ctx context.Context, tokens []int64) (out *tensor.Tensor, err error) {
	return d.forward(ctx, tokens, nil)
}

// Close waits for active Metal calls, then releases device weights and any
// remaining session caches. It can be called repeatedly, including through
// copies of the decoder handle. CPU decoders need no cleanup.
func (d *Decoder) Close() (err error) {
	if d == nil {
		return nil
	}
	return d.metal.close()
}

// ForwardWithTrace also retains named intermediate tensors for conformance.
// Every tensor is owned by this call; trace["logits"] is the returned logits.
// No partial result or trace is returned on error. This diagnostic path retains
// considerably more memory than Forward. Metal attention models return an
// error because their fused attention operation does not expose probabilities.
func (d *Decoder) ForwardWithTrace(tokens []int64) (*tensor.Tensor, map[string]*tensor.Tensor, error) {
	trace := make(map[string]*tensor.Tensor)
	logits, err := d.forward(context.Background(), tokens, trace)
	if err != nil {
		return nil, nil, err
	}
	return logits, trace, nil
}

func (d *Decoder) validatePrefix(tokens []int64) error {
	if d == nil || (d.embedding == nil && d.metalEmbedding == nil) {
		return fmt.Errorf("llama: decoder is uninitialized")
	}
	if len(tokens) == 0 || int64(len(tokens)) > d.config.MaxContext {
		return fmt.Errorf("llama: prefix length must be in [1,%d]", d.config.MaxContext)
	}
	for position, token := range tokens {
		if token < 0 || token >= d.config.VocabSize {
			return fmt.Errorf("llama: token at position %d is out of range: %d", position, token)
		}
	}
	return nil
}

func (d *Decoder) forward(ctx context.Context, tokens []int64, trace map[string]*tensor.Tensor) (out *tensor.Tensor, err error) {
	if err = d.validatePrefix(tokens); err != nil {
		return nil, err
	}
	if d.metal != nil && d.metal.attention {
		if trace != nil {
			return nil, fmt.Errorf("llama: ForwardWithTrace requires attention probabilities, which Metal attention does not expose")
		}
		return d.forwardMetalPrefix(ctx, tokens)
	}
	x, err := d.forwardPrefix(ctx, tokens, trace, nil)
	if err != nil {
		return nil, err
	}
	if out, err = d.projectLogits(x, trace); err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// forwardPrefix evaluates all blocks for an already validated prefix.
func (d *Decoder) forwardPrefix(ctx context.Context, tokens []int64, trace map[string]*tensor.Tensor, capture func(int, *tensor.Tensor, *tensor.Tensor) error) (out *tensor.Tensor, err error) {
	var scratch *metalFFNScratch
	if scratch, err = d.metal.begin(ctx, d.config, int64(len(tokens))); err != nil {
		return nil, err
	}
	defer scratch.close()
	x, err := gather(d.embedding, tokens, 0)
	if err != nil {
		return nil, fmt.Errorf("llama: embedding: %w", err)
	}
	record(trace, "embedding", x)
	cosine, sine, err := rotaryTables(0, int64(len(tokens)), d.config)
	if err != nil {
		return nil, err
	}
	mask, err := causalMask(int64(len(tokens)))
	if err != nil {
		return nil, err
	}
	for index, block := range d.blocks {
		attend := func(q, k, v *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
			if capture != nil {
				if captureErr := capture(index, k, v); captureErr != nil {
					return nil, nil, captureErr
				}
			}
			return attention(q, k, v, mask, d.config)
		}
		if x, err = d.forwardBlock(ctx, x, block, cosine, sine, attend, scratch, trace, fmt.Sprintf("layer.%d.", index)); err != nil {
			return nil, fmt.Errorf("llama: layer %d: %w", index, err)
		}
	}
	return x, nil
}

func (d *Decoder) projectLogits(x *tensor.Tensor, trace map[string]*tensor.Tensor) (*tensor.Tensor, error) {
	hidden, err := rmsNorm(x, d.norm, d.config.NormEpsilon)
	if err != nil {
		return nil, fmt.Errorf("llama: final norm: %w", err)
	}
	record(trace, "hidden", hidden)
	return d.classify(hidden, trace)
}

func (d *Decoder) classify(hidden *tensor.Tensor, trace map[string]*tensor.Tensor) (*tensor.Tensor, error) {
	logits, err := project(hidden, d.output)
	if err != nil {
		return nil, fmt.Errorf("llama: output projection: %w", err)
	}
	record(trace, "logits", logits)
	return logits, nil
}

type blockAttention func(q, k, v *tensor.Tensor) (attended, probabilities *tensor.Tensor, err error)

func (d *Decoder) forwardBlock(ctx context.Context, x *tensor.Tensor, block blockWeights, cosine, sine *tensor.Tensor, attend blockAttention, scratch *metalFFNScratch, trace map[string]*tensor.Tensor, prefix string) (out *tensor.Tensor, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	n, err := rmsNorm(x, block.attentionNorm, d.config.NormEpsilon)
	if err != nil {
		return nil, err
	}
	record(trace, prefix+"attention_norm", n)
	headWidth := d.config.Dim / d.config.Heads
	q, err := headProjection(n, block.wq, d.config.Heads, headWidth)
	if err != nil {
		return nil, err
	}
	k, err := headProjection(n, block.wk, d.config.KVHeads, headWidth)
	if err != nil {
		return nil, err
	}
	v, err := headProjection(n, block.wv, d.config.KVHeads, headWidth)
	if err != nil {
		return nil, err
	}
	record(trace, prefix+"q", q)
	record(trace, prefix+"k", k)
	record(trace, prefix+"v", v)
	rotatedQ, err := rotate(q, cosine, sine)
	if err != nil {
		return nil, err
	}
	rotatedK, err := rotate(k, cosine, sine)
	if err != nil {
		return nil, err
	}
	record(trace, prefix+"rope_q", rotatedQ)
	record(trace, prefix+"rope_k", rotatedK)
	attended, probabilities, err := attend(rotatedQ, rotatedK, v)
	if err != nil {
		return nil, err
	}
	record(trace, prefix+"attention_probs", probabilities)
	record(trace, prefix+"attention", attended)
	projected, err := project(attended, block.wo)
	if err != nil {
		return nil, err
	}
	h, err := elementwise(x, projected, tensor.AddInto)
	if err != nil {
		return nil, err
	}
	record(trace, prefix+"attention_residual", h)
	n, err = rmsNorm(h, block.ffnNorm, d.config.NormEpsilon)
	if err != nil {
		return nil, err
	}
	record(trace, prefix+"ffn_norm", n)
	var down *tensor.Tensor
	if scratch == nil {
		down, err = block.forwardCPUFFN(n, trace, prefix)
	} else {
		down, err = scratch.run(ctx, n, block.metal, trace, prefix)
	}
	if err != nil {
		return nil, err
	}
	if out, err = elementwise(h, down, tensor.AddInto); err != nil {
		return nil, err
	}
	record(trace, prefix+"output", out)
	return out, nil
}

func record(trace map[string]*tensor.Tensor, name string, value *tensor.Tensor) {
	if trace != nil {
		trace[name] = value
	}
}
