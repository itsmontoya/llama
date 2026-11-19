package llama

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestLlama32CachedParity(t *testing.T) {
	dir := os.Getenv("LLAMA_32_1B_MODEL_DIR")
	if dir == "" {
		t.Skip("set LLAMA_32_1B_MODEL_DIR to the pinned Meta snapshot")
	}
	verifyLlama32Snapshot(t, dir)
	var reference struct {
		Revision   string `json:"revision"`
		Schema     int    `json:"schema_version"`
		Generation struct {
			PromptIDs []int64                   `json:"prompt_ids"`
			SuffixIDs []int64                   `json:"suffix_ids"`
			Steps     []llama32LogitObservation `json:"steps"`
		} `json:"generation"`
	}
	if err := readHFJSON(filepath.Join("testdata", "llama32_reference.json"), &reference); err != nil {
		t.Fatal(err)
	}
	const decodeSteps = 4
	generation := reference.Generation
	if reference.Revision != llama32Revision || reference.Schema != 1 || len(generation.PromptIDs) == 0 ||
		len(generation.SuffixIDs) <= decodeSteps || len(generation.Steps) <= decodeSteps {
		t.Fatal("invalid cached replay coverage in the pinned Llama 3.2 reference")
	}
	tokens := append(slices.Clone(generation.PromptIDs), generation.SuffixIDs[:decodeSteps]...)
	for step := range decodeSteps + 1 {
		length := len(generation.PromptIDs) + step
		observation := generation.Steps[step]
		if observation.Position != length-1 || observation.ArgmaxID != int(generation.SuffixIDs[step]) ||
			!slices.Equal(observation.InputIDs, tokens[:length]) {
			t.Fatalf("generation step %d disagrees with the frozen replay prefix or next ID", step)
		}
	}

	const capacity = 512
	decoder, _, err := LoadHFModel(dir, capacity)
	if err != nil {
		t.Fatal(err)
	}
	session, err := decoder.NewSession(capacity)
	if err != nil {
		t.Fatal(err)
	}
	var maxError, maxFraction cachedRowError
	for step := range decodeSteps + 1 {
		length := len(generation.PromptIDs) + step
		var logits []float32
		if step == 0 {
			logits, err = session.Prefill(tokens[:length])
		} else {
			logits, err = session.Decode(tokens[length-1])
		}
		if err != nil {
			t.Fatalf("step %d, position %d: %v", step, length-1, err)
		}
		if session.Position() != int64(length) {
			t.Fatalf("step %d committed %d tokens, want %d", step, session.Position(), length)
		}
		largest, fraction, winner := compareCachedRow(t, decoder, tokens[:length], logits)
		if winner != generation.Steps[step].ArgmaxID {
			t.Errorf("step %d, position %d: winner %d, want frozen winner %d", step, length-1, winner, generation.Steps[step].ArgmaxID)
		}
		t.Logf("step %d, position %d: exact winner %d, maximum absolute error %.9g (token %d), tolerance fraction %.9g (token %d)",
			step, length-1, winner, largest.abs, largest.token, fraction.fraction, fraction.token)
		if step == 0 || largest.abs > maxError.abs {
			maxError = largest
		}
		if step == 0 || fraction.fraction > maxFraction.fraction {
			maxFraction = fraction
		}
	}
	t.Logf("maximum absolute logit error %.9g at position %d, token %d (cached %.9g, full %.9g)",
		maxError.abs, maxError.position, maxError.token, maxError.got, maxError.want)
	t.Logf("maximum tolerance fraction %.9g at position %d, token %d (cached %.9g, full %.9g)",
		maxFraction.fraction, maxFraction.position, maxFraction.token, maxFraction.got, maxFraction.want)
}
