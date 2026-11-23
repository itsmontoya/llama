//go:build darwin && arm64 && cgo && metal && !purego

package llama

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestLlama32Metal(t *testing.T) {
	var dir = os.Getenv("LLAMA_32_1B_MODEL_DIR")
	if dir == "" {
		t.Skip("set LLAMA_32_1B_MODEL_DIR to the pinned Meta snapshot")
	}
	verifyLlama32Snapshot(t, dir)
	var decoder = loadTestMetalModel(t, dir, 512)
	t.Run("frozen_logits", func(t *testing.T) {
		checkLlama32Logits(t, decoder)
	})

	var reference struct {
		Generation struct {
			PromptIDs []int64                   `json:"prompt_ids"`
			SuffixIDs []int64                   `json:"suffix_ids"`
			Steps     []llama32LogitObservation `json:"steps"`
		} `json:"generation"`
	}
	if err := readHFJSON(filepath.Join("testdata", "llama32_reference.json"), &reference); err != nil {
		t.Fatal(err)
	}
	var generation = reference.Generation
	if len(generation.PromptIDs) != 6 || len(generation.SuffixIDs) != 8 || len(generation.Steps) != 8 {
		t.Fatal("invalid pinned generation coverage")
	}
	t.Run("cached_greedy", func(t *testing.T) {
		var session, err = decoder.NewSession(13)
		if err != nil {
			t.Fatal(err)
		}
		var row []float32
		if row, err = session.Prefill(generation.PromptIDs); err != nil {
			t.Fatal(err)
		}
		var prefix = slices.Clone(generation.PromptIDs)
		var got []int64
		for step, observation := range generation.Steps {
			if session.Position() != int64(len(prefix)) || !slices.Equal(prefix, observation.InputIDs) {
				t.Fatalf("step %d: got prefix %v at committed position %d, want %v", step, prefix, session.Position(), observation.InputIDs)
			}
			compareLlama32Row(t, "cached Metal", prefix, row, observation)
			largest, fraction, winner := compareCachedRow(t, decoder, prefix, row)
			t.Logf("position %d: cached/full maximum absolute error %.9g, tolerance fraction %.9g", largest.position, largest.abs, fraction.fraction)
			got = append(got, int64(winner))
			prefix = append(prefix, int64(winner))
			if step+1 < len(generation.Steps) {
				if row, err = session.Decode(int64(winner)); err != nil {
					t.Fatal(err)
				}
			}
		}
		if !slices.Equal(got, generation.SuffixIDs) {
			t.Fatalf("cached generated IDs %v, want %v", got, generation.SuffixIDs)
		}
		t.Logf("prompt %v, eight generated IDs %v", generation.PromptIDs, got)
	})

	t.Run("prefill_tails", func(t *testing.T) {
		for _, size := range []int{27, 28, 29} {
			t.Run(fmt.Sprint(size), func(t *testing.T) {
				var prefix = make([]int64, size)
				prefix[0] = generation.PromptIDs[0]
				for i := 1; i < size; i++ {
					prefix[i] = generation.PromptIDs[1+(i-1)%5]
				}
				var session, err = decoder.NewSession(int64(size))
				if err != nil {
					t.Fatal(err)
				}
				var row []float32
				if row, err = session.Prefill(prefix); err != nil {
					t.Fatal(err)
				}
				largest, fraction, winner := compareCachedRow(t, decoder, prefix, row)
				t.Logf("winner %d: cached/full maximum absolute error %.9g, tolerance fraction %.9g", winner, largest.abs, fraction.fraction)
			})
		}
	})

	t.Run("independent_sessions", func(t *testing.T) {
		checkConcurrentMetalSessions(t, decoder, generation.PromptIDs, generation.SuffixIDs)
	})
}

func checkConcurrentMetalSessions(t *testing.T, decoder *Decoder, prompt, suffix []int64) {
	t.Helper()
	type result struct {
		session            *Session
		prefilled, decoded []float32
		err                error
	}
	var (
		inputs  = [][]int64{slices.Clone(prompt), slices.Clone(prompt)}
		results [2]result
		start   = make(chan struct{})
		done    = make(chan struct{}, len(results))
	)
	inputs[1][len(prompt)-1] = suffix[1]

	for i := range results {
		go func() {
			defer func() { done <- struct{}{} }()
			<-start
			var current = &results[i]
			current.session, current.err = decoder.NewSession(int64(len(prompt) + 1))
			if current.err != nil {
				return
			}
			current.prefilled, current.err = current.session.Prefill(inputs[i])
			if current.err != nil {
				return
			}
			current.decoded, current.err = current.session.Decode(suffix[i])
		}()
	}
	close(start)
	for range results {
		<-done
	}
	for i, current := range results {
		if current.err != nil {
			t.Fatalf("session %d: %v", i, current.err)
		}
		if current.session.Position() != int64(len(prompt)+1) {
			t.Fatalf("session %d committed %d positions", i, current.session.Position())
		}
		compareCachedRow(t, decoder, inputs[i], current.prefilled)
		compareCachedRow(t, decoder, append(inputs[i], suffix[i]), current.decoded)
	}
	if slices.Equal(results[0].decoded, results[1].decoded) {
		t.Fatal("distinct inputs produced identical session outputs")
	}
}

func BenchmarkLlama32MetalCachedGeneration(b *testing.B) {
	benchmarkLlama32CachedGeneration(b, true)
}
