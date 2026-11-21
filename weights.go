package llama

import (
	"fmt"

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
}

type decoderWeight struct {
	name     string
	shape    []int64
	values   []float32
	target   **tensor.Tensor
	prepared **tensor.PreparedRightTranspose
}

func (w decoderWeight) store(name string, values []float32) error {
	for index, value := range values {
		if !finite(float64(value)) {
			return fmt.Errorf("llama: %s contains nonfinite value at %d", name, index)
		}
	}
	var err error
	if w.prepared != nil {
		*w.prepared, err = tensor.PrepareRightTranspose(w.shape, values)
	} else {
		*w.target, err = tensor.FromFloat32(w.shape, values)
	}
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
		{name: "token_embedding", shape: vocabShape, values: w.TokenEmbedding, target: &d.embedding},
		{name: "final_norm", shape: normShape, values: w.FinalNorm, target: &d.norm},
	} {
		if err := visit(-1, weight); err != nil {
			return err
		}
	}
	if !c.TieEmbeddings {
		if err := visit(-1, decoderWeight{name: "output", shape: vocabShape, values: w.Output, target: &d.output}); err != nil {
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
			{name: "attention_norm", shape: normShape, values: source.AttentionNorm, target: &block.attentionNorm},
			{name: "wq", shape: qShape, values: source.WQ, target: &block.wq},
			{name: "wk", shape: kvShape, values: source.WK, target: &block.wk},
			{name: "wv", shape: kvShape, values: source.WV, target: &block.wv},
			{name: "wo", shape: []int64{c.Dim, c.Dim}, values: source.WO, target: &block.wo},
			{name: "ffn_norm", shape: normShape, values: source.FFNNorm, target: &block.ffnNorm},
			{name: "w_gate", shape: upShape, values: source.WGate, prepared: &block.gate},
			{name: "w_up", shape: upShape, values: source.WUp, prepared: &block.up},
			{name: "w_down", shape: downShape, values: source.WDown, prepared: &block.down},
		} {
			if err := visit(layer, weight); err != nil {
				return fmt.Errorf("llama: layer %d: %w", layer, err)
			}
		}
	}
	return nil
}

func (d *Decoder) loadWeights(w Weights, read func(layer int, weight decoderWeight) (name string, values []float32, err error)) error {
	if err := d.visitWeights(w, func(layer int, weight decoderWeight) error {
		name, values, err := read(layer, weight)
		if err != nil {
			return err
		}
		return weight.store(name, values)
	}); err != nil {
		return err
	}
	if d.config.TieEmbeddings {
		d.output = d.embedding
	}
	return nil
}
