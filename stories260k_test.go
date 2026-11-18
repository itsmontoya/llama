package llama

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestStories260KLogits(t *testing.T) {
	modelPath := os.Getenv("LLAMA_STORIES260K_MODEL")
	if modelPath == "" {
		t.Skip("set LLAMA_STORIES260K_MODEL to the pinned stories260K.bin")
	}
	modelBytes, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	assertHash(t, modelBytes, "b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696")
	decoder, err := LoadLegacyModel(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	wantConfig := Config{Dim: 64, FFNDim: 172, Layers: 5, Heads: 8, KVHeads: 4, VocabSize: 512, MaxContext: 512, NormEpsilon: 1e-5, RopeTheta: 10000, TieEmbeddings: true}
	if got := decoder.Config(); got != wantConfig {
		t.Fatalf("config = %+v, want %+v", got, wantConfig)
	}
	fixture, err := os.ReadFile(filepath.Join("testdata", "stories260k_logits.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Tokens []int64     `json:"tokens"`
		Logits [][]float32 `json:"logits"`
	}
	if err := json.Unmarshal(fixture, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) != 2 || !slices.Equal(cases[0].Tokens, []int64{1}) || !slices.Equal(cases[1].Tokens, []int64{1, 3, 42, 3}) {
		t.Fatal("expected single-token and multi-token prefixes")
	}
	const vocabSize = 512
	for _, test := range cases {
		logits, err := decoder.Forward(test.Tokens)
		if err != nil {
			t.Fatal(err)
		}
		if len(test.Logits) != len(test.Tokens) || !slices.Equal(logits.Shape(), []int64{int64(len(test.Tokens)), vocabSize}) {
			t.Fatalf("prefix %v: incorrect reference row count %d or output shape %v", test.Tokens, len(test.Logits), logits.Shape())
		}
		values := tensorValues(t, logits)
		var maximum float64
		for position, row := range test.Logits {
			if len(row) != vocabSize {
				t.Fatalf("prefix %v, position %d: got %d reference logits, want %d", test.Tokens, position, len(row), vocabSize)
			}
			for token, value := range row {
				got := values[position*vocabSize+token]
				want := float64(value)
				difference := math.Abs(float64(got) - want)
				if !finite(float64(got)) || !finite(want) || difference > 1e-4+1e-4*math.Abs(want) {
					t.Fatalf("prefix %v, position %d, token %d: got %.9g, want %.9g, absolute error %.9g", test.Tokens, position, token, got, want, difference)
				}
				maximum = max(maximum, difference)
			}
		}
		t.Logf("prefix %v: maximum absolute logit error %.9g", test.Tokens, maximum)
	}
}
