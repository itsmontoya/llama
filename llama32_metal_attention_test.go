//go:build darwin && arm64 && cgo && metal && !purego

package llama

import (
	"os"
	"runtime"
	"slices"
	"testing"

	"github.com/itsmontoya/llama/internal/tokenizer"
)

func TestLlama32MetalAttention(t *testing.T) {
	var dir = os.Getenv("LLAMA_32_1B_MODEL_DIR")
	if dir == "" {
		t.Skip("set LLAMA_32_1B_MODEL_DIR to the pinned Meta snapshot")
	}
	verifyLlama32Snapshot(t, dir)
	var decoder = loadTestMetalAttention(t, dir, 512)
	checkLlama32Metal(t, decoder)

	var model, err = readHFConfig(dir, 512)
	if err != nil {
		t.Fatal(err)
	}
	var vocab *tokenizer.Llama3Tokenizer
	if vocab, err = tokenizer.LoadLlama3(dir, decoder.Config().VocabSize, model.generation.BOSID, model.generation.StopIDs); err != nil {
		t.Fatal(err)
	}
	var prompt []int64
	if prompt, err = vocab.Encode(llama32ShortPrompt); err != nil || len(prompt) != 6 {
		t.Fatalf("short prompt IDs %v, error %v", prompt, err)
	}
	t.Run("sustained_decode", func(t *testing.T) {
		// Positions 6–37 are the established host benchmark trajectory. The
		// first eight selections also form the pinned Transformers suffix.
		var want = []int64{
			12366, 13, 1102, 374, 279, 1455, 95551, 3363, 304, 9822, 323, 279,
			3224, 753, 6864, 323, 7928, 3363, 13, 12366, 374, 279, 1455, 95551,
			16036, 3158, 304, 279, 7665, 9323, 323, 279, 1917,
		}
		var capacity = int64(len(prompt) + len(want) - 1)
		var session = newTestMetalAttentionSession(t, decoder, capacity)
		var row, prefillErr = session.Prefill(prompt)
		if prefillErr != nil {
			t.Fatal(prefillErr)
		}
		var suffix []int64
		for step, expected := range want {
			var winner, stepErr = greedyToken(row)
			if stepErr != nil || winner != expected {
				t.Fatalf("position %d: winner %d, want %d, error %v", session.Position()-1, winner, expected, stepErr)
			}
			suffix = append(suffix, winner)
			if step+1 < len(want) {
				if row, stepErr = session.Decode(winner); stepErr != nil {
					t.Fatal(stepErr)
				}
			}
		}
		assertMetalCachePosition(t, session, capacity)
		var prefix = append(slices.Clone(prompt), suffix[:len(suffix)-1]...)
		var largest, fraction, _ = compareCachedRow(t, decoder, prefix, row)
		t.Logf("positions 6–37: exact winners %v, final cached/full maximum error %.9g, tolerance fraction %.9g", suffix, largest.abs, fraction.fraction)
	})

	t.Run("independent_decoders", func(t *testing.T) {
		// Reclaim preceding temporary host classifier operands before loading
		// the second decoder. The model weights and device handles stay live.
		runtime.GC()
		var other = loadTestMetalAttention(t, dir, 512)
		if decoder.metal.device == other.metal.device || decoder.embedding == other.embedding {
			t.Fatal("independent decoders share a logical Metal device or embedding")
		}
		checkConcurrentMetalSessions(t, [2]*Decoder{decoder, other}, prompt, []int64{12366, 13})
	})
}

func BenchmarkLlama32MetalAttentionCachedGeneration(b *testing.B) {
	benchmarkLlama32CachedGeneration(b, LoadHFModelMetalAttention)
}
