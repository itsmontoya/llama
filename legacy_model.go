package llama

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

// LoadLegacyModel loads a little-endian llama2.c legacy v0 FP32 checkpoint.
func LoadLegacyModel(path string) (*Decoder, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("llama: open legacy checkpoint: %w", err)
	}
	defer file.Close()
	decoder, err := readLegacyModel(file)
	if err != nil {
		return nil, fmt.Errorf("llama: legacy checkpoint %q: %w", path, err)
	}
	return decoder, nil
}

type legacyTensorGroup struct {
	name          string
	rows, columns int64
	repeat        int64
	target        func(int64) *[]float32
}

func readLegacyModel(file *os.File) (*Decoder, error) {
	var header [7]int32
	reader := bufio.NewReader(file)
	if err := binary.Read(reader, binary.LittleEndian, &header); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	vocabSize := int64(header[5])
	config := Config{
		Dim:           int64(header[0]),
		FFNDim:        int64(header[1]),
		Layers:        int64(header[2]),
		Heads:         int64(header[3]),
		KVHeads:       int64(header[4]),
		VocabSize:     max(vocabSize, -vocabSize),
		MaxContext:    int64(header[6]),
		NormEpsilon:   1e-5,
		RopeTheta:     10000,
		TieEmbeddings: vocabSize > 0,
	}
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid header: %w", err)
	}

	var weights Weights
	dim, ffnDim, layers := config.Dim, config.FFNDim, config.Layers
	headWidth := dim / config.Heads
	kvWidth := config.KVHeads * headWidth
	groups := []legacyTensorGroup{
		{"token embedding", config.VocabSize, dim, 1, func(int64) *[]float32 { return &weights.TokenEmbedding }},
		{"attention norm", dim, 1, layers, func(i int64) *[]float32 { return &weights.Layers[i].AttentionNorm }},
		{"WQ", dim, dim, layers, func(i int64) *[]float32 { return &weights.Layers[i].WQ }},
		{"WK", kvWidth, dim, layers, func(i int64) *[]float32 { return &weights.Layers[i].WK }},
		{"WV", kvWidth, dim, layers, func(i int64) *[]float32 { return &weights.Layers[i].WV }},
		{"WO", dim, dim, layers, func(i int64) *[]float32 { return &weights.Layers[i].WO }},
		{"FFN norm", dim, 1, layers, func(i int64) *[]float32 { return &weights.Layers[i].FFNNorm }},
		{"WGate", ffnDim, dim, layers, func(i int64) *[]float32 { return &weights.Layers[i].WGate }},
		{"WDown", dim, ffnDim, layers, func(i int64) *[]float32 { return &weights.Layers[i].WDown }},
		{"WUp", ffnDim, dim, layers, func(i int64) *[]float32 { return &weights.Layers[i].WUp }},
		{"final norm", dim, 1, 1, func(int64) *[]float32 { return &weights.FinalNorm }},
		// The decoder derives its own rotary tables from the resolved configuration.
		{"legacy rotary tables", config.MaxContext, headWidth / 2, 2, nil},
	}
	if !config.TieEmbeddings {
		groups = append(groups, legacyTensorGroup{"output", config.VocabSize, dim, 1, func(int64) *[]float32 { return &weights.Output }})
	}

	expectedSize := int64(binary.Size(header))
	for _, group := range groups {
		_, size, err := tensor.ValidateShape([]int64{group.repeat, group.rows, group.columns}, tensor.Float32)
		if err != nil {
			return nil, fmt.Errorf("%s group size: %w", group.name, err)
		}
		if size > math.MaxInt64-expectedSize {
			return nil, fmt.Errorf("file size overflow at %s", group.name)
		}
		expectedSize += size
	}
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	if info.Size() != expectedSize {
		return nil, fmt.Errorf("file size: got %d bytes, expected %d", info.Size(), expectedSize)
	}

	weights.Layers = make([]LayerWeights, layers)
	for _, group := range groups {
		count := group.rows * group.columns // The full group size was validated above.
		if group.target == nil {
			if _, err := io.CopyN(io.Discard, reader, group.repeat*count*4); err != nil {
				return nil, fmt.Errorf("read %s: %w", group.name, err)
			}
			continue
		}
		for index := range group.repeat {
			values := make([]float32, count)
			if err := binary.Read(reader, binary.LittleEndian, values); err != nil {
				return nil, fmt.Errorf("read %s tensor %d: %w", group.name, index, err)
			}
			*group.target(index) = values
		}
	}
	if _, err := reader.ReadByte(); err != io.EOF {
		if err != nil {
			return nil, fmt.Errorf("check checkpoint end: %w", err)
		}
		return nil, fmt.Errorf("unexpected data after checkpoint weights")
	}

	decoder, err := NewDecoder(config, weights)
	if err != nil {
		return nil, fmt.Errorf("construct decoder: %w", err)
	}
	return decoder, nil
}
