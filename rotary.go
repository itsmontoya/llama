package llama

import (
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

// rotaryTables are shared across layers of one Forward call. The signed sine
// table and a pair-swapping Gather implement adjacent-pair rotation.
func rotaryTables(tokens, headWidth int64, theta float64) (*tensor.Tensor, *tensor.Tensor, error) {
	shape := []int64{tokens, 1, headWidth}
	elements, _, err := tensor.ValidateShape(shape, tensor.Float32)
	if err != nil {
		return nil, nil, err
	}
	cosine, sine := make([]float32, elements), make([]float32, elements)
	for position := int64(0); position < tokens; position++ {
		for feature := int64(0); feature < headWidth; feature += 2 {
			angle := float64(position) / math.Pow(theta, float64(feature)/float64(headWidth))
			if !finite(angle) {
				return nil, nil, fmt.Errorf("llama: nonfinite rotary angle at position %d feature %d", position, feature)
			}
			s, c := math.Sincos(angle)
			offset := position*headWidth + feature
			cosine[offset], cosine[offset+1] = float32(c), float32(c)
			sine[offset], sine[offset+1] = -float32(s), float32(s)
		}
	}
	cosTensor, err := tensor.FromFloat32(shape, cosine)
	if err != nil {
		return nil, nil, err
	}
	sinTensor, err := tensor.FromFloat32(shape, sine)
	if err != nil {
		return nil, nil, err
	}
	return cosTensor, sinTensor, nil
}

func rotate(input, cosine, sine *tensor.Tensor) (*tensor.Tensor, error) {
	width := input.Shape()[2]
	indices := make([]int64, width)
	for feature := int64(0); feature < width; feature += 2 {
		indices[feature], indices[feature+1] = feature+1, feature
	}
	swapped, err := gather(input, indices, 2)
	if err != nil {
		return nil, err
	}
	first, err := elementwise(input, cosine, tensor.MulInto)
	if err != nil {
		return nil, err
	}
	second, err := elementwise(swapped, sine, tensor.MulInto)
	if err != nil {
		return nil, err
	}
	return elementwise(first, second, tensor.AddInto)
}
