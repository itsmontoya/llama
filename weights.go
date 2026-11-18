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
	ffnNorm, gate, up, down       *tensor.Tensor
}

func weightTensor(name string, shape []int64, values []float32) (*tensor.Tensor, error) {
	for index, value := range values {
		if !finite(float64(value)) {
			return nil, fmt.Errorf("llama: %s contains nonfinite value at %d", name, index)
		}
	}
	out, err := tensor.FromFloat32(shape, values)
	if err != nil {
		return nil, fmt.Errorf("llama: %s: %w", name, err)
	}
	return out, nil
}

func loadBlock(c Config, w LayerWeights) (blockWeights, error) {
	var block blockWeights
	kvWidth := c.KVHeads * (c.Dim / c.Heads)
	for _, item := range []struct {
		name   string
		shape  []int64
		values []float32
		target **tensor.Tensor
	}{
		{"attention_norm", []int64{c.Dim}, w.AttentionNorm, &block.attentionNorm},
		{"wq", []int64{c.Dim, c.Dim}, w.WQ, &block.wq},
		{"wk", []int64{kvWidth, c.Dim}, w.WK, &block.wk},
		{"wv", []int64{kvWidth, c.Dim}, w.WV, &block.wv},
		{"wo", []int64{c.Dim, c.Dim}, w.WO, &block.wo},
		{"ffn_norm", []int64{c.Dim}, w.FFNNorm, &block.ffnNorm},
		{"w_gate", []int64{c.FFNDim, c.Dim}, w.WGate, &block.gate},
		{"w_up", []int64{c.FFNDim, c.Dim}, w.WUp, &block.up},
		{"w_down", []int64{c.Dim, c.FFNDim}, w.WDown, &block.down},
	} {
		value, err := weightTensor(item.name, item.shape, item.values)
		if err != nil {
			return blockWeights{}, err
		}
		*item.target = value
	}
	return block, nil
}
