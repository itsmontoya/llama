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
	var cosine, sine, err = rotaryCoefficients(start, tokens, config, false)
	if err != nil {
		return nil, nil, err
	}
	var shape = []int64{tokens, 1, config.Dim / config.Heads}
	var cosTensor *tensor.Tensor
	if cosTensor, err = tensor.FromFloat32(shape, cosine); err != nil {
		return nil, nil, err
	}
	var sinTensor *tensor.Tensor
	if sinTensor, err = tensor.FromFloat32(shape, sine); err != nil {
		return nil, nil, err
	}
	return cosTensor, sinTensor, nil
}

// Metal takes one coefficient per pair and applies the subtraction itself.
// The host rotate path repeats cosine and uses opposite signs for the two
// sine entries because it applies the coefficients to a pair-swapped tensor.
func rotaryCoefficients(start, tokens int64, config Config, packed bool) (cosine, sine []float32, err error) {
	if start < 0 || tokens <= 0 || start > math.MaxInt64-(tokens-1) {
		return nil, nil, fmt.Errorf("llama: invalid rotary position range: start %d, count %d", start, tokens)
	}
	var width = config.Dim / config.Heads
	if packed {
		width /= 2
	}
	var elements int64
	if elements, _, err = tensor.ValidateShape([]int64{tokens, width}, tensor.Float32); err != nil {
		return nil, nil, err
	}
	var frequencies []float64
	if frequencies, err = rotaryInverseFrequencies(config); err != nil {
		return nil, nil, err
	}
	cosine, sine = make([]float32, elements), make([]float32, elements)
	for row := range tokens {
		var position = start + row
		for pair, frequency := range frequencies {
			var angle = float64(position) * frequency
			if !finite(angle) {
				return nil, nil, fmt.Errorf("llama: nonfinite rotary angle at position %d pair %d", position, pair)
			}
			var s, c = math.Sincos(angle)
			if packed {
				var offset = row*width + int64(pair)
				cosine[offset], sine[offset] = float32(c), float32(s)
				continue
			}
			var offset = row*width + int64(2*pair)
			cosine[offset], cosine[offset+1] = float32(c), float32(c)
			sine[offset], sine[offset+1] = -float32(s), float32(s)
		}
	}
	return cosine, sine, nil
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
