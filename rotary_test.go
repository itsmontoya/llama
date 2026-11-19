package llama

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"
)

func TestLlama3RotaryReference(t *testing.T) {
	var manifest struct {
		Architecture struct {
			HeadWidth int64       `json:"head_dim"`
			Theta     float64     `json:"rope_theta"`
			Scaling   RopeScaling `json:"rope_scaling"`
		} `json:"architecture"`
	}
	var reference struct {
		RoPE struct {
			Positions   []int64     `json:"positions"`
			Frequencies []float64   `json:"inv_freq"`
			Cosine      [][]float64 `json:"cos"`
			Sine        [][]float64 `json:"sin"`
		} `json:"rope"`
	}
	for name, output := range map[string]any{
		"testdata/llama32_manifest.json":  &manifest,
		"testdata/llama32_reference.json": &reference,
	} {
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, output); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	width := manifest.Architecture.HeadWidth
	config := Config{
		Dim: width, Heads: 1, MaxContext: 512,
		RopeTheta: manifest.Architecture.Theta, RopeScaling: manifest.Architecture.Scaling,
	}
	want := reference.RoPE
	if width <= 0 || config.RopeScaling.Type != "llama3" || len(want.Positions) < 2 ||
		len(want.Positions) != len(want.Cosine) || len(want.Positions) != len(want.Sine) {
		t.Fatal("invalid Llama 3 rotary reference fixture")
	}
	frequencies, err := rotaryInverseFrequencies(config)
	if err != nil {
		t.Fatal(err)
	}
	if len(frequencies) != len(want.Frequencies) {
		t.Fatalf("got %d frequencies, want %d", len(frequencies), len(want.Frequencies))
	}
	var maxFrequencyError float64
	for pair, expected := range want.Frequencies {
		// This schedule includes all three bands. Use a relative bound for the
		// lowest frequencies so a missing division by the factor cannot pass.
		difference := math.Abs(frequencies[pair] - expected)
		if expected <= 0 || !finite(expected) || difference > 1e-6*expected {
			t.Fatalf("inverse frequency %d: got %.12g, want %.12g", pair, frequencies[pair], expected)
		}
		maxFrequencyError = max(maxFrequencyError, difference)
	}
	lastPosition := want.Positions[len(want.Positions)-1]
	cosine, sine, err := rotaryTables(0, lastPosition+1, config)
	if err != nil {
		t.Fatal(err)
	}
	cosines, sines := tensorValues(t, cosine), tensorValues(t, sine)
	var maxTableError float64
	for row, position := range want.Positions {
		if position < 0 || position > lastPosition || len(want.Cosine[row]) != len(frequencies) || len(want.Sine[row]) != len(frequencies) {
			t.Fatalf("invalid rotary reference row at position %d", position)
		}
		for pair := range frequencies {
			offset := position*width + int64(2*pair)
			for _, point := range []struct {
				got  float32
				want float64
			}{
				{cosines[offset], want.Cosine[row][pair]}, {cosines[offset+1], want.Cosine[row][pair]},
				{sines[offset], -want.Sine[row][pair]}, {sines[offset+1], want.Sine[row][pair]},
			} {
				difference := math.Abs(float64(point.got) - point.want)
				if !finite(float64(point.got)) || !finite(point.want) || difference > 2e-6 {
					t.Fatalf("position %d, pair %d: got %.9g, want %.9g", position, pair, point.got, point.want)
				}
				maxTableError = max(maxTableError, difference)
			}
		}
	}
	start := want.Positions[1]
	config.MaxContext = lastPosition + 1
	cosine, sine, err = rotaryTables(start, lastPosition-start+1, config)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tensorValues(t, cosine), cosines[start*width:]) || !slices.Equal(tensorValues(t, sine), sines[start*width:]) {
		t.Fatal("nonzero-offset coefficients changed with the operational cap")
	}
	t.Logf("maximum absolute errors against Transformers: inverse frequency %.9g, coefficient %.9g", maxFrequencyError, maxTableError)
}

func TestRotaryTablesInvalidRange(t *testing.T) {
	config := Config{Dim: 4, Heads: 1, RopeTheta: 10000}
	for _, interval := range [][2]int64{{-1, 1}, {0, 0}, {0, -1}, {math.MaxInt64, 2}} {
		cosine, sine, err := rotaryTables(interval[0], interval[1], config)
		if err == nil || cosine != nil || sine != nil {
			t.Fatalf("rotaryTables(%d, %d) returned coefficients for an invalid range", interval[0], interval[1])
		}
	}
}
