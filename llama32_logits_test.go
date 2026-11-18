package llama

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

const (
	llama32Revision  = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
	llama32LogitATol = 1e-3
	llama32LogitRTol = 1e-3
)

type llama32LogitSample struct {
	ID    int     `json:"token_id"`
	Logit float64 `json:"logit"`
}

type llama32LogitObservation struct {
	Position int                  `json:"position"`
	ArgmaxID int                  `json:"argmax_id"`
	Samples  []llama32LogitSample `json:"sampled_logits"`
	Top      []llama32LogitSample `json:"top_candidates"`
	Margin   float64              `json:"top_two_margin"`
	InputIDs []int64              `json:"input_ids"`
}

type llama32LogitDiscrepancy struct {
	label                 string
	position, token       int
	got, want, difference float64
	fraction              float64
}

func TestLlama32Logits(t *testing.T) {
	dir := os.Getenv("LLAMA_32_1B_MODEL_DIR")
	if dir == "" {
		t.Skip("set LLAMA_32_1B_MODEL_DIR to the pinned Meta snapshot")
	}
	verifyLlama32Snapshot(t, dir)
	var reference struct {
		Revision   string `json:"revision"`
		Schema     int    `json:"schema_version"`
		LogitCases []struct {
			Name         string                    `json:"name"`
			InputIDs     []int64                   `json:"input_ids"`
			Observations []llama32LogitObservation `json:"observations"`
		} `json:"logit_cases"`
		Generation struct {
			Steps []llama32LogitObservation `json:"steps"`
		} `json:"generation"`
	}
	if err := readHFJSON(filepath.Join("testdata", "llama32_reference.json"), &reference); err != nil {
		t.Fatal(err)
	}
	if reference.Revision != llama32Revision || reference.Schema != 1 || len(reference.LogitCases) < 2 || len(reference.Generation.Steps) == 0 {
		t.Fatal("invalid pinned Llama 3.2 reference coverage")
	}
	decoder, _, err := LoadHFModel(dir, 512)
	if err != nil {
		t.Fatal(err)
	}
	vocab := int(decoder.Config().VocabSize)
	var maxError, maxFraction llama32LogitDiscrepancy
	observe := func(label string, prefix []int64, observations []llama32LogitObservation) {
		if len(observations) == 0 {
			t.Fatalf("%s has no frozen observations", label)
		}
		logits, err := decoder.Forward(prefix)
		if err != nil {
			t.Fatalf("%s, prefix %v: %v", label, prefix, err)
		}
		if shape := logits.Shape(); !slices.Equal(shape, []int64{int64(len(prefix)), int64(vocab)}) {
			t.Fatalf("%s: got logit shape %v, want [%d %d]", label, shape, len(prefix), vocab)
		}
		values := tensorValues(t, logits)
		for _, observation := range observations {
			position := observation.Position
			if position < 0 || position >= len(prefix) {
				t.Fatalf("%s: observation position %d is outside prefix %v", label, position, prefix)
			}
			largest, fraction := compareLlama32Row(t, label, prefix, values[position*vocab:(position+1)*vocab], observation)
			if maxError.label == "" || largest.difference > maxError.difference {
				maxError = largest
			}
			if maxFraction.label == "" || fraction.fraction > maxFraction.fraction {
				maxFraction = fraction
			}
		}
	}
	for _, test := range reference.LogitCases {
		observe(test.Name, test.InputIDs, test.Observations)
	}
	for index, step := range reference.Generation.Steps {
		observe(fmt.Sprintf("generation step %d", index+1), step.InputIDs, []llama32LogitObservation{step})
	}
	t.Logf("maximum absolute logit error %.9g at %s, position %d, token %d (Go %.9g, reference %.9g)",
		maxError.difference, maxError.label, maxError.position, maxError.token, maxError.got, maxError.want)
	t.Logf("maximum tolerance fraction %.9g at %s, position %d, token %d (Go %.9g, reference %.9g)",
		maxFraction.fraction, maxFraction.label, maxFraction.position, maxFraction.token, maxFraction.got, maxFraction.want)
}

func compareLlama32Row(t *testing.T, label string, prefix []int64, row []float32, want llama32LogitObservation) (largest, fraction llama32LogitDiscrepancy) {
	t.Helper()
	if len(row) < 2 || len(want.Samples) < 32 || len(want.Top) != 8 || want.Top[0].ID != want.ArgmaxID || !finite(want.Margin) {
		t.Fatalf("%s position %d: incomplete reference observation", label, want.Position)
	}
	best, second := -1, -1
	for token, value := range row {
		if !finite(float64(value)) {
			t.Fatalf("%s prefix %v, position %d: nonfinite Go logit for token %d", label, prefix, want.Position, token)
		}
		if best < 0 || value > row[best] {
			best, second = token, best
		} else if second < 0 || value > row[second] {
			second = token
		}
	}
	for index, sample := range want.Samples {
		if sample.ID < 0 || sample.ID >= len(row) || !finite(sample.Logit) {
			t.Fatalf("%s position %d: invalid reference sample %+v", label, want.Position, sample)
		}
		got := float64(row[sample.ID])
		difference := math.Abs(got - sample.Logit)
		point := llama32LogitDiscrepancy{
			label: label, position: want.Position, token: sample.ID, got: got, want: sample.Logit, difference: difference,
			fraction: difference / (llama32LogitATol + llama32LogitRTol*math.Abs(sample.Logit)),
		}
		if index == 0 || point.difference > largest.difference {
			largest = point
		}
		if index == 0 || point.fraction > fraction.fraction {
			fraction = point
		}
	}
	margin := float64(row[best]) - float64(row[second])
	t.Logf("%s position %d: winner %d, Go/reference margin %.9g/%.9g, max abs error %.9g (token %d), tolerance fraction %.9g (token %d)",
		label, want.Position, best, margin, want.Margin, largest.difference, largest.token, fraction.fraction, fraction.token)
	if best != want.ArgmaxID || fraction.fraction > 1 {
		t.Errorf("%s prefix %v, position %d: Go top two [%d:%.9g, %d:%.9g], reference top eight %+v; worst sample token %d: Go %.9g, reference %.9g, tolerance fraction %.9g",
			label, prefix, want.Position, best, row[best], second, row[second], want.Top,
			fraction.token, fraction.got, fraction.want, fraction.fraction)
	}
	return largest, fraction
}

func verifyLlama32Snapshot(t *testing.T, dir string) {
	t.Helper()
	var manifest struct {
		Revision string `json:"revision"`
		Files    map[string]struct {
			Size   int64  `json:"size"`
			SHA256 string `json:"sha256"`
		} `json:"files"`
	}
	if err := readHFJSON(filepath.Join("testdata", "llama32_manifest.json"), &manifest); err != nil {
		t.Fatal(err)
	}
	names := []string{"config.json", "generation_config.json", hfCheckpointName, "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"}
	if manifest.Revision != llama32Revision || len(manifest.Files) != len(names) {
		t.Fatal("invalid pinned Llama 3.2 snapshot manifest")
	}
	for _, name := range names {
		want, ok := manifest.Files[name]
		if !ok || want.Size <= 0 {
			t.Fatalf("snapshot manifest missing %q", name)
		}
		path := filepath.Join(dir, name)
		sum, err := hashLlama32File(path, want.Size)
		if err != nil {
			t.Fatal(err)
		}
		if sum != want.SHA256 {
			t.Fatalf("%s SHA-256: got %s, want %s", path, sum, want.SHA256)
		}
	}
}

func hashLlama32File(path string, size int64) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("%s: expected a regular snapshot file", path)
	}
	if info.Size() != size {
		return "", fmt.Errorf("%s size: got %d, want %d", path, info.Size(), size)
	}
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	digest := sha256.New()
	n, err := io.Copy(digest, file)
	if err != nil {
		return "", fmt.Errorf("hash %s: %w", path, err)
	}
	if n != size {
		return "", fmt.Errorf("hash %s: read %d bytes, want %d", path, n, size)
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}
