package llama

import (
	"math"
	"slices"
	"strings"
	"testing"
)

func newGreedyTestDecoder(t *testing.T, lastOutput [2]float32) *Decoder {
	t.Helper()
	config := Config{
		Dim: 2, FFNDim: 2, Layers: 1, Heads: 1, KVHeads: 1,
		VocabSize: 5, MaxContext: 5, NormEpsilon: 1e-5, RopeTheta: 10000,
	}
	// Zero attention and FFN weights leave each embedding on the residual path.
	// The output projection makes the first four token IDs form a cycle.
	zeros := make([]float32, 4)
	ones := []float32{1, 1}
	weights := Weights{
		TokenEmbedding: []float32{1, 0, 0, 1, -1, 0, 0, -1, 1, 1},
		Layers: []LayerWeights{{
			AttentionNorm: ones, WQ: zeros, WK: zeros, WV: zeros, WO: zeros,
			FFNNorm: ones, WGate: zeros, WDown: zeros, WUp: zeros,
		}},
		FinalNorm: ones,
		Output:    []float32{0, -1, 1, 0, 0, 1, -1, 0, lastOutput[0], lastOutput[1]},
	}
	decoder, err := NewDecoder(config, weights)
	if err != nil {
		t.Fatal(err)
	}
	return decoder
}

func TestGenerateGreedyDeterministicAndOwned(t *testing.T) {
	// For prompt [0], tokens 1 and 4 have equal highest logits.
	decoder := newGreedyTestDecoder(t, [2]float32{1, 0})
	backing := []int64{0, 9, 9, 9, 9}
	prompt := backing[:1]
	want := []int64{1, 2, 3, 0}

	first, err := decoder.GenerateGreedy(prompt, 4)
	if err != nil || !slices.Equal(first, want) {
		t.Fatalf("first continuation = %v, error = %v, want %v", first, err, want)
	}
	again, err := decoder.GenerateGreedy(prompt, 4)
	if err != nil || !slices.Equal(again, want) {
		t.Fatalf("repeated continuation = %v, error = %v, want %v", again, err, want)
	}
	other, err := decoder.GenerateGreedy([]int64{1}, 4)
	if err != nil || !slices.Equal(other, []int64{2, 3, 0, 1}) {
		t.Fatalf("different prompt continuation = %v, error = %v", other, err)
	}
	again[0] = 4
	if !slices.Equal(first, want) {
		t.Fatalf("earlier continuation changed to %v", first)
	}
	if !slices.Equal(backing, []int64{0, 9, 9, 9, 9}) {
		t.Fatalf("prompt backing array changed to %v", backing)
	}
}

func TestGenerateGreedyLimitsAndStops(t *testing.T) {
	decoder := newGreedyTestDecoder(t, [2]float32{1, 0})
	for _, test := range []struct {
		name   string
		prompt []int64
		budget int
		stops  []int64
		want   []int64
	}{
		{name: "new token limit", prompt: []int64{0}, budget: 2, want: []int64{1, 2}},
		{name: "remaining context", prompt: []int64{3, 0, 1}, budget: math.MaxInt, want: []int64{2, 3}},
		{name: "exact context", prompt: []int64{0, 1, 2, 3, 0}, budget: 3},
		{name: "zero budget", prompt: []int64{0}},
		{name: "first token is a stop", prompt: []int64{0}, budget: 4, stops: []int64{1}},
		{name: "later token is a stop", prompt: []int64{0}, budget: 4, stops: []int64{0, 3}, want: []int64{1, 2}},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := decoder.GenerateGreedy(test.prompt, test.budget, test.stops...)
			if err != nil || !slices.Equal(got, test.want) {
				t.Fatalf("continuation = %v, error = %v, want %v", got, err, test.want)
			}
		})
	}
}

func TestGenerateGreedyRejectsInvalidInputs(t *testing.T) {
	decoder := newGreedyTestDecoder(t, [2]float32{1, 0})
	for _, test := range []struct {
		name    string
		decoder *Decoder
		prompt  []int64
		budget  int
		stops   []int64
		wantErr string
	}{
		{name: "nil decoder", prompt: []int64{0}, wantErr: "uninitialized"},
		{name: "zero decoder", decoder: &Decoder{}, prompt: []int64{0}, wantErr: "uninitialized"},
		{name: "empty prompt", decoder: decoder, wantErr: "prefix length"},
		{name: "negative prompt ID", decoder: decoder, prompt: []int64{0, -1}, wantErr: "token at position 1"},
		{name: "large prompt ID", decoder: decoder, prompt: []int64{5}, wantErr: "token at position 0"},
		{name: "over context", decoder: decoder, prompt: make([]int64, 6), wantErr: "prefix length"},
		{name: "negative budget", decoder: decoder, prompt: []int64{0}, budget: -1, wantErr: "max new tokens"},
		{name: "negative stop ID", decoder: decoder, prompt: []int64{0}, stops: []int64{-1}, wantErr: "stop token"},
		{name: "large stop ID", decoder: decoder, prompt: make([]int64, 5), budget: 1, stops: []int64{0, 5}, wantErr: "stop token"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := test.decoder.GenerateGreedy(test.prompt, test.budget, test.stops...)
			if got != nil || err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("continuation = %v, error = %v, want nil and error containing %q", got, err, test.wantErr)
			}
		})
	}
}

func TestGenerateGreedyRejectsNonfiniteLastRow(t *testing.T) {
	// A finite output weight overflows to negative infinity for token 1,
	// even though the winning logit remains finite.
	decoder := newGreedyTestDecoder(t, [2]float32{0, -math.MaxFloat32})
	first, err := decoder.GenerateGreedy([]int64{0}, 1)
	if err != nil || !slices.Equal(first, []int64{1}) {
		t.Fatalf("single-step continuation = %v, error = %v", first, err)
	}
	got, err := decoder.GenerateGreedy([]int64{0}, 2)
	if got != nil || err == nil || !strings.Contains(err.Error(), "nonfinite logit") {
		t.Fatalf("two-step continuation = %v, error = %v, want nil and nonfinite logit error", got, err)
	}
	// The first row still overflows, but only the final row is needed.
	got, err = decoder.GenerateGreedy([]int64{1, 0}, 1)
	if err != nil || !slices.Equal(got, first) {
		t.Fatalf("continuation with nonfinite earlier row = %v, error = %v", got, err)
	}
}
