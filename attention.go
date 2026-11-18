package llama

import (
	"math"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func causalMask(tokens int64) (*tensor.Tensor, error) {
	shape := []int64{tokens, tokens}
	elements, _, err := tensor.ValidateShape(shape, tensor.Float32)
	if err != nil {
		return nil, err
	}
	values := make([]float32, elements)
	for query := int64(0); query < tokens; query++ {
		for key := query + 1; key < tokens; key++ {
			values[query*tokens+key] = float32(math.Inf(-1))
		}
	}
	return tensor.FromFloat32(shape, values)
}

func headProjection(input, weight *tensor.Tensor, heads, headWidth int64) (*tensor.Tensor, error) {
	projected, err := project(input, weight)
	if err != nil {
		return nil, err
	}
	return reshape(projected, []int64{input.Shape()[0], heads, headWidth})
}

// attention deliberately materializes GQA heads and the full score matrix.
// This copying reference path will be replaced by the later cached backend.
func attention(q, k, v, mask *tensor.Tensor, c Config) (*tensor.Tensor, *tensor.Tensor, error) {
	tokens, headWidth := q.Shape()[0], c.Dim/c.Heads
	indices := make([]int64, c.Heads)
	for head := int64(0); head < c.Heads; head++ {
		indices[head] = head / (c.Heads / c.KVHeads)
	}
	keys, err := gather(k, indices, 1)
	if err != nil {
		return nil, nil, err
	}
	values, err := gather(v, indices, 1)
	if err != nil {
		return nil, nil, err
	}
	queries, err := transpose(q, []int64{1, 0, 2})
	if err != nil {
		return nil, nil, err
	}
	keys, err = transpose(keys, []int64{1, 2, 0})
	if err != nil {
		return nil, nil, err
	}
	values, err = transpose(values, []int64{1, 0, 2})
	if err != nil {
		return nil, nil, err
	}
	scores, err := matmul(queries, keys)
	if err != nil {
		return nil, nil, err
	}
	scale, err := scalar(float32(1 / math.Sqrt(float64(headWidth))))
	if err != nil {
		return nil, nil, err
	}
	scaled, err := elementwise(scores, scale, tensor.MulInto)
	if err != nil {
		return nil, nil, err
	}
	masked, err := elementwise(scaled, mask, tensor.AddInto)
	if err != nil {
		return nil, nil, err
	}
	probabilities, err := tensor.New(tensor.Float32, masked.Shape())
	if err != nil {
		return nil, nil, err
	}
	if err = tensor.SoftmaxInto(masked, probabilities, -1); err != nil {
		return nil, nil, err
	}
	attended, err := matmul(probabilities, values)
	if err != nil {
		return nil, nil, err
	}
	ordered, err := transpose(attended, []int64{1, 0, 2})
	if err != nil {
		return nil, nil, err
	}
	merged, err := reshape(ordered, []int64{tokens, c.Dim})
	if err != nil {
		return nil, nil, err
	}
	return merged, probabilities, nil
}
