package llama

import (
	"fmt"

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
	embedding, err := weightTensor("token_embedding", []int64{config.VocabSize, config.Dim}, weights.TokenEmbedding)
	if err != nil {
		return nil, err
	}
	norm, err := weightTensor("final_norm", []int64{config.Dim}, weights.FinalNorm)
	if err != nil {
		return nil, err
	}
	output := embedding
	if !config.TieEmbeddings {
		if output, err = weightTensor("output", []int64{config.VocabSize, config.Dim}, weights.Output); err != nil {
			return nil, err
		}
	}
	blocks := make([]blockWeights, len(weights.Layers))
	for index, layerWeights := range weights.Layers {
		if blocks[index], err = loadBlock(config, layerWeights); err != nil {
			return nil, fmt.Errorf("llama: layer %d: %w", index, err)
		}
	}
	return &Decoder{config: config, embedding: embedding, norm: norm, output: output, blocks: blocks}, nil
}

// Decoder evaluates full token prefixes at batch size one. It has no KV cache
// or mutable session state. Its zero value is invalid; use NewDecoder.
type Decoder struct {
	config                  Config
	embedding, norm, output *tensor.Tensor
	blocks                  []blockWeights
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
	return d.forward(tokens, nil)
}

// ForwardWithTrace also retains named intermediate tensors for conformance.
// Every tensor is owned by this call; trace["logits"] is the returned logits.
// No partial result or trace is returned on error. This diagnostic path retains
// considerably more memory than Forward.
func (d *Decoder) ForwardWithTrace(tokens []int64) (*tensor.Tensor, map[string]*tensor.Tensor, error) {
	trace := make(map[string]*tensor.Tensor)
	logits, err := d.forward(tokens, trace)
	if err != nil {
		return nil, nil, err
	}
	return logits, trace, nil
}

func (d *Decoder) validatePrefix(tokens []int64) error {
	if d == nil || d.embedding == nil {
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

func (d *Decoder) forward(tokens []int64, trace map[string]*tensor.Tensor) (*tensor.Tensor, error) {
	if err := d.validatePrefix(tokens); err != nil {
		return nil, err
	}
	x, err := gather(d.embedding, tokens, 0)
	if err != nil {
		return nil, fmt.Errorf("llama: embedding: %w", err)
	}
	record(trace, "embedding", x)
	cosine, sine, err := rotaryTables(int64(len(tokens)), d.config.Dim/d.config.Heads, d.config.RopeTheta)
	if err != nil {
		return nil, err
	}
	mask, err := causalMask(int64(len(tokens)))
	if err != nil {
		return nil, err
	}
	for index, block := range d.blocks {
		if x, err = d.forwardBlock(x, block, cosine, sine, mask, trace, fmt.Sprintf("layer.%d.", index)); err != nil {
			return nil, fmt.Errorf("llama: layer %d: %w", index, err)
		}
	}
	hidden, err := rmsNorm(x, d.norm, d.config.NormEpsilon)
	if err != nil {
		return nil, fmt.Errorf("llama: final norm: %w", err)
	}
	record(trace, "hidden", hidden)
	logits, err := project(hidden, d.output)
	if err != nil {
		return nil, fmt.Errorf("llama: output projection: %w", err)
	}
	record(trace, "logits", logits)
	return logits, nil
}

func (d *Decoder) forwardBlock(x *tensor.Tensor, block blockWeights, cosine, sine, mask *tensor.Tensor, trace map[string]*tensor.Tensor, prefix string) (*tensor.Tensor, error) {
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
	attended, probabilities, err := attention(rotatedQ, rotatedK, v, mask, d.config)
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
	gate, err := project(n, block.gate)
	if err != nil {
		return nil, err
	}
	up, err := project(n, block.up)
	if err != nil {
		return nil, err
	}
	record(trace, prefix+"gate", gate)
	record(trace, prefix+"up", up)
	gated, err := swiGLU(gate, up)
	if err != nil {
		return nil, err
	}
	record(trace, prefix+"swiglu", gated)
	down, err := project(gated, block.down)
	if err != nil {
		return nil, err
	}
	out, err := elementwise(h, down, tensor.AddInto)
	if err != nil {
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
