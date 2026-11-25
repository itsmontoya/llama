package llama

import (
	"context"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/gpu"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

// Weights supplies row-major FP32 weights. NewDecoder copies every slice.
// Matrices are stored [output,input]. Output must be empty when embeddings are
// tied, and [VocabSize,Dim] otherwise. Norm weights have shape [Dim].
type Weights struct {
	TokenEmbedding []float32      `json:"token_embedding"`
	Layers         []LayerWeights `json:"layers"`
	FinalNorm      []float32      `json:"final_norm"`
	Output         []float32      `json:"output,omitempty"`
}

// LayerWeights supplies one bias-free attention and SwiGLU block.
// WQ/WO are [Dim,Dim], WK/WV are [KVHeads*headWidth,Dim], WGate/WUp are
// [FFNDim,Dim], and WDown is [Dim,FFNDim]. Q/K use adjacent rotary pairs.
type LayerWeights struct {
	AttentionNorm []float32 `json:"attention_norm"`
	WQ            []float32 `json:"wq"`
	WK            []float32 `json:"wk"`
	WV            []float32 `json:"wv"`
	WO            []float32 `json:"wo"`
	FFNNorm       []float32 `json:"ffn_norm"`
	WGate         []float32 `json:"w_gate"`
	WUp           []float32 `json:"w_up"`
	WDown         []float32 `json:"w_down"`
}

type blockWeights struct {
	attentionNorm, wq, wk, wv, wo *tensor.Tensor
	ffnNorm                       *tensor.Tensor
	gate, up, down                *tensor.PreparedRightTranspose
	metal                         metalFFNWeights
	metalAttention                metalAttentionWeights
}

type decoderWeight struct {
	name     string
	shape    []int64
	values   []float32
	target   **tensor.Tensor
	prepared **tensor.PreparedRightTranspose
	metal    **gpu.Tensor
}

func (w decoderWeight) store(name string, values []float32, model *metalModel) (err error) {
	for index, value := range values {
		if !finite(float64(value)) {
			return fmt.Errorf("llama: %s contains nonfinite value at %d", name, index)
		}
	}
	switch {
	case w.metal != nil && model != nil && (model.attention || w.prepared != nil):
		*w.metal, err = model.uploadWithDType(w.shape, values, model.weightDTypeFor(w.name))
	case w.prepared != nil:
		*w.prepared, err = tensor.PrepareRightTranspose(w.shape, values)
	default:
		*w.target, err = tensor.FromFloat32(w.shape, values)
	}
	if err != nil {
		return fmt.Errorf("llama: %s: %w", name, err)
	}
	return nil
}

func (w decoderWeight) storeBFloat16Bits(name string, bits []uint16, model *metalModel) (err error) {
	for index, value := range bits {
		if (value & 0x7f80) == 0x7f80 {
			return fmt.Errorf("llama: %s contains nonfinite value at %d", name, index)
		}
	}
	*w.metal, err = model.uploadBFloat16Bits(w.shape, bits)
	if err != nil {
		return fmt.Errorf("llama: %s: %w", name, err)
	}
	return nil
}

// visitWeights defines the shape and destination of each private tensor.
// Model-wide weights use layer -1. File loaders supply their values later.
func (d *Decoder) visitWeights(w Weights, visit func(layer int, weight decoderWeight) error) error {
	c := d.config
	vocabShape := []int64{c.VocabSize, c.Dim}
	normShape := []int64{c.Dim}
	for _, weight := range []decoderWeight{
		{name: "token_embedding", shape: vocabShape, values: w.TokenEmbedding, target: &d.embedding, metal: &d.metalEmbedding},
		{name: "final_norm", shape: normShape, values: w.FinalNorm, target: &d.norm, metal: &d.metalNorm},
	} {
		if err := visit(-1, weight); err != nil {
			return err
		}
	}
	if !c.TieEmbeddings {
		if err := visit(-1, decoderWeight{name: "output", shape: vocabShape, values: w.Output, target: &d.output, metal: &d.metalOutput}); err != nil {
			return err
		}
	}
	headWidth := c.Dim / c.Heads
	qShape := []int64{c.Heads * headWidth, c.Dim}
	kvShape := []int64{c.KVHeads * headWidth, c.Dim}
	upShape, downShape := []int64{c.FFNDim, c.Dim}, []int64{c.Dim, c.FFNDim}
	for layer := range d.blocks {
		block := &d.blocks[layer]
		var source LayerWeights
		if w.Layers != nil {
			source = w.Layers[layer]
		}
		for _, weight := range []decoderWeight{
			{name: "attention_norm", shape: normShape, values: source.AttentionNorm, target: &block.attentionNorm, metal: &block.metalAttention.attentionNorm},
			{name: "wq", shape: qShape, values: source.WQ, target: &block.wq, metal: &block.metalAttention.q},
			{name: "wk", shape: kvShape, values: source.WK, target: &block.wk, metal: &block.metalAttention.k},
			{name: "wv", shape: kvShape, values: source.WV, target: &block.wv, metal: &block.metalAttention.v},
			{name: "wo", shape: []int64{c.Dim, c.Dim}, values: source.WO, target: &block.wo, metal: &block.metalAttention.o},
			{name: "ffn_norm", shape: normShape, values: source.FFNNorm, target: &block.ffnNorm, metal: &block.metalAttention.ffnNorm},
			{name: "w_gate", shape: upShape, values: source.WGate, prepared: &block.gate, metal: &block.metal.gate},
			{name: "w_up", shape: upShape, values: source.WUp, prepared: &block.up, metal: &block.metal.up},
			{name: "w_down", shape: downShape, values: source.WDown, prepared: &block.down, metal: &block.metal.down},
		} {
			if err := visit(layer, weight); err != nil {
				return fmt.Errorf("llama: layer %d: %w", layer, err)
			}
		}
	}
	return nil
}

func (d *Decoder) loadWeights(ctx context.Context, w Weights, read func(layer int, weight decoderWeight) (name string, values []float32, err error)) (err error) {
	if err = d.visitWeights(w, func(layer int, weight decoderWeight) (readErr error) {
		if readErr = ctx.Err(); readErr != nil {
			return readErr
		}
		name, values, readErr := read(layer, weight)
		if readErr != nil {
			return readErr
		}
		return weight.store(name, values, d.metal)
	}); err != nil {
		return err
	}
	if d.config.TieEmbeddings {
		d.output = d.embedding
		// The classifier borrows the embedding allocation owned by metalModel.
		d.metalOutput = d.metalEmbedding
	}
	return nil
}

func (block blockWeights) forwardCPUFFN(n *tensor.Tensor, trace map[string]*tensor.Tensor, prefix string) (out *tensor.Tensor, err error) {
	gate, err := projectPrepared(n, block.gate)
	if err != nil {
		return nil, err
	}
	up, err := projectPrepared(n, block.up)
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
	return projectPrepared(gated, block.down)
}
