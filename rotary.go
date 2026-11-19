package llama

import (
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func rotaryInverseFrequencies(config Config) ([]float64, error) {
	headWidth := config.Dim / config.Heads
	frequencies := make([]float64, headWidth/2)
	for pair := range frequencies {
		frequency := 1 / math.Pow(config.RopeTheta, float64(2*pair)/float64(headWidth))
		if config.RopeScaling.Type == "llama3" {
			frequency = llama3Frequency(frequency, config.RopeScaling)
		}
		if frequency <= 0 || !finite(frequency) {
			return nil, fmt.Errorf("llama: invalid rotary inverse frequency at pair %d: %g", pair, frequency)
		}
		frequencies[pair] = frequency
	}
	return frequencies, nil
}

func llama3Frequency(frequency float64, scaling RopeScaling) float64 {
	wavelength := 2 * math.Pi / frequency
	originalContext := float64(scaling.OriginalMaxPositionEmbeddings)
	switch {
	case wavelength < originalContext/scaling.HighFreqFactor:
		return frequency
	case wavelength > originalContext/scaling.LowFreqFactor:
		return frequency / scaling.Factor
	default:
		weight := (originalContext/wavelength - scaling.LowFreqFactor) / (scaling.HighFreqFactor - scaling.LowFreqFactor)
		return (1-weight)*(frequency/scaling.Factor) + weight*frequency
	}
}

// rotaryTables builds adjacent-pair coefficients for consecutive absolute
// positions. The signed sine table and a pair-swapping Gather rotate each pair.
func rotaryTables(start, tokens int64, config Config) (*tensor.Tensor, *tensor.Tensor, error) {
	if start < 0 || tokens <= 0 || start > math.MaxInt64-(tokens-1) {
		return nil, nil, fmt.Errorf("llama: invalid rotary position range: start %d, count %d", start, tokens)
	}
	headWidth := config.Dim / config.Heads
	shape := []int64{tokens, 1, headWidth}
	elements, _, err := tensor.ValidateShape(shape, tensor.Float32)
	if err != nil {
		return nil, nil, err
	}
	frequencies, err := rotaryInverseFrequencies(config)
	if err != nil {
		return nil, nil, err
	}
	cosine, sine := make([]float32, elements), make([]float32, elements)
	for row := range tokens {
		position := start + row
		for pair, frequency := range frequencies {
			angle := float64(position) * frequency
			if !finite(angle) {
				return nil, nil, fmt.Errorf("llama: nonfinite rotary angle at position %d pair %d", position, pair)
			}
			s, c := math.Sincos(angle)
			offset := row*headWidth + int64(2*pair)
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
