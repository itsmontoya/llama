//go:build darwin && arm64 && cgo && metal && !purego

package llama

import (
	"os"
	"slices"
	"testing"

	"github.com/itsmontoya/llama/internal/tokenizer"
	"github.com/itsmontoya/neuralnetwork/gpu"
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
	t.Run("generation_limits", func(t *testing.T) {
		for _, mode := range [...]struct {
			name     string
			generate func([]int64, int, ...int64) ([]int64, error)
		}{
			{name: "cached", generate: decoder.GenerateGreedyCached},
			{name: "full_prefix", generate: decoder.GenerateGreedy},
		} {
			t.Run(mode.name, func(t *testing.T) {
				var stopped, stopErr = mode.generate(prompt, 8, 13)
				if stopErr != nil || !slices.Equal(stopped, []int64{12366}) {
					t.Fatalf("generation before stop token 13: IDs %v, error %v", stopped, stopErr)
				}
				var full = make([]int64, decoder.Config().MaxContext)
				var limited, limitErr = mode.generate(full, 8)
				if limitErr != nil || limited == nil || len(limited) != 0 {
					t.Fatalf("full context generation: IDs %v, error %v", limited, limitErr)
				}
				if limited, limitErr = mode.generate(append(full, 0), 8); limited != nil || limitErr == nil {
					t.Fatalf("over context generation: IDs %v, error %v", limited, limitErr)
				}
			})
		}
	})
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
		var other = loadTestMetalAttention(t, dir, 512)
		if decoder.metal.device == other.metal.device || decoder.metalEmbedding == other.metalEmbedding || decoder.metalOutput == other.metalOutput {
			t.Fatal("independent decoders share a logical Metal device or embedding")
		}
		checkConcurrentMetalSessions(t, [2]*Decoder{decoder, other}, prompt, []int64{12366, 13})
	})
}

func BenchmarkLlama32MetalAttentionCachedGeneration(b *testing.B) {
	benchmarkLlama32CachedGeneration(b, LoadHFModelMetalAttention)
}

func TestLlama32MetalReducedPrecision(t *testing.T) {
	var dir = os.Getenv("LLAMA_32_1B_MODEL_DIR")
	if dir == "" {
		t.Skip("set LLAMA_32_1B_MODEL_DIR to the pinned Meta snapshot")
	}
	verifyLlama32Snapshot(t, dir)
	var model, err = readHFConfig(dir, 512)
	if err != nil {
		t.Fatal(err)
	}
	var vocab *tokenizer.Llama3Tokenizer
	if vocab, err = tokenizer.LoadLlama3(dir, model.decoder.VocabSize, model.generation.BOSID, model.generation.StopIDs); err != nil {
		t.Fatal(err)
	}
	var prompt []int64
	if prompt, err = vocab.Encode(llama32ShortPrompt); err != nil || len(prompt) != 6 {
		t.Fatalf("short prompt IDs %v, error %v", prompt, err)
	}
	var wantShort = []int64{12366, 13, 1102, 374, 279, 1455, 95551, 3363}

	for _, tc := range []struct {
		name           string
		opts           MetalOptions
		wantModelBytes uint64
		requireExact   bool
	}{
		{"bf16_kv32", MetalOptions{WeightDType: gpu.BFloat16, KVCacheDType: gpu.Float32}, 2_471_759_872, true},
		{"bf16_kv16", MetalOptions{WeightDType: gpu.BFloat16, KVCacheDType: gpu.Float16}, 2_471_759_872, true},
		{"fp16_kv32", MetalOptions{WeightDType: gpu.Float16, KVCacheDType: gpu.Float32}, 2_471_759_872, true},
		{"fp16_kv16", MetalOptions{WeightDType: gpu.Float16, KVCacheDType: gpu.Float16}, 2_471_759_872, true},
		{"q8_0_kv32", MetalOptions{WeightDType: gpu.Q8_0, KVCacheDType: gpu.Float32}, 1_313_382_400, true},
		{"q4_0_kv32", MetalOptions{WeightDType: gpu.Q4_0, KVCacheDType: gpu.Float32}, 695_508_992, false},
		{"q4_1_kv32", MetalOptions{WeightDType: gpu.Q4_1, KVCacheDType: gpu.Float32}, 772_743_168, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var decoder, _, loadErr = LoadHFModelMetalAttentionWithOptions(t.Context(), dir, 512, tc.opts)
			if loadErr != nil {
				t.Fatal(loadErr)
			}
			defer func() {
				if closeErr := decoder.Close(); closeErr != nil {
					t.Error(closeErr)
				}
			}()
			var totalBytes uint64
			for _, w := range decoder.metal.weights {
				totalBytes += w.Bytes()
			}
			if totalBytes != tc.wantModelBytes {
				t.Fatalf("device model bytes = %d, want %d", totalBytes, tc.wantModelBytes)
			}
			var got, genErr = decoder.GenerateGreedyCached(prompt, 8, model.generation.StopIDs...)
			if genErr != nil {
				t.Fatal(genErr)
			}
			var text, decErr = vocab.Decode(got)
			if decErr != nil {
				t.Fatal(decErr)
			}
			t.Logf("%s: model_bytes=%d suffix=%v text=%q", tc.name, totalBytes, got, text)
			if tc.requireExact && !slices.Equal(got, wantShort) {
				t.Fatalf("%s: got %v (%q), want %v", tc.name, got, text, wantShort)
			}
		})
	}
}
