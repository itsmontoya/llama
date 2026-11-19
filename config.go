package llama

import (
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

// Config resolves every dimension and numeric setting used by the decoder.
// RoPE rotates adjacent feature pairs over the whole head.
// FFNDim is the actual gated hidden width, not a multiplier of Dim.
type Config struct {
	Dim           int64       `json:"dim"`
	FFNDim        int64       `json:"ffn_dim"`
	Layers        int64       `json:"layers"`
	Heads         int64       `json:"heads"`
	KVHeads       int64       `json:"kv_heads"`
	VocabSize     int64       `json:"vocab_size"`
	MaxContext    int64       `json:"max_context"`
	NormEpsilon   float32     `json:"norm_epsilon"`
	RopeTheta     float64     `json:"rope_theta"`
	RopeScaling   RopeScaling `json:"rope_scaling,omitzero"`
	TieEmbeddings bool        `json:"tie_embeddings"`
}

// RopeScaling selects a rotary frequency schedule. The zero value uses the
// unscaled schedule. The llama3 schedule uses every field.
type RopeScaling struct {
	Type                          string  `json:"rope_type"`
	Factor                        float64 `json:"factor"`
	LowFreqFactor                 float64 `json:"low_freq_factor"`
	HighFreqFactor                float64 `json:"high_freq_factor"`
	OriginalMaxPositionEmbeddings int64   `json:"original_max_position_embeddings"`
}

func (s RopeScaling) validate() error {
	switch s.Type {
	case "", "default":
		if s.Factor != 0 || s.LowFreqFactor != 0 || s.HighFreqFactor != 0 || s.OriginalMaxPositionEmbeddings != 0 {
			return fmt.Errorf("llama: unscaled RoPE must have no scaling parameters")
		}
	case "llama3":
		if s.Factor <= 0 || !finite(s.Factor) || s.LowFreqFactor <= 0 || !finite(s.LowFreqFactor) ||
			s.HighFreqFactor <= s.LowFreqFactor || !finite(s.HighFreqFactor) || s.OriginalMaxPositionEmbeddings <= 0 {
			return fmt.Errorf("llama: llama3 RoPE requires positive finite scaling factors, high_freq_factor > low_freq_factor, and a positive original context")
		}
	default:
		return fmt.Errorf("llama: unsupported rope_type %q", s.Type)
	}
	return nil
}

func (c Config) validate() error {
	if c.Dim <= 0 || c.FFNDim <= 0 || c.Layers <= 0 || c.Heads <= 0 || c.KVHeads <= 0 || c.VocabSize <= 0 || c.MaxContext <= 0 {
		return fmt.Errorf("llama: all dimensions and limits must be positive")
	}
	if c.Dim%c.Heads != 0 || c.Heads%c.KVHeads != 0 {
		return fmt.Errorf("llama: dim must be divisible by heads, and heads by kv_heads")
	}
	if (c.Dim/c.Heads)%2 != 0 {
		return fmt.Errorf("llama: adjacent-pair RoPE requires an even head width")
	}
	if c.NormEpsilon <= 0 || !finite(float64(c.NormEpsilon)) || c.RopeTheta <= 0 || !finite(c.RopeTheta) {
		return fmt.Errorf("llama: norm epsilon and RoPE theta must be positive and finite")
	}
	if err := c.RopeScaling.validate(); err != nil {
		return err
	}
	// Check the largest individual tensor shapes before creating any payload.
	// These representability checks do not enforce a total memory budget.
	for _, shape := range [][]int64{
		{c.VocabSize, c.Dim}, {c.FFNDim, c.Dim}, {c.Dim, c.Dim},
		{c.MaxContext, c.Dim}, {c.MaxContext, c.FFNDim},
		{c.MaxContext, c.VocabSize}, {c.Heads, c.MaxContext, c.MaxContext},
	} {
		if _, _, err := tensor.ValidateShape(shape, tensor.Float32); err != nil {
			return fmt.Errorf("llama: configuration tensor size: %w", err)
		}
	}
	return nil
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
