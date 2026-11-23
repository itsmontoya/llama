package llama

import (
	"context"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"

	"github.com/itsmontoya/llama/internal/tokenizer"
	"github.com/itsmontoya/neuralnetwork/gpu"
)

func BenchmarkLlama32CachedGeneration(b *testing.B) {
	benchmarkLlama32CachedGeneration(b, nil)
}

func benchmarkLlama32CachedGeneration(b *testing.B, loadMetal func(context.Context, string, int64) (*Decoder, HFGenerationConfig, error)) {
	b.Helper()
	dir := os.Getenv("LLAMA_32_1B_MODEL_DIR")
	if dir == "" {
		b.Skip("set LLAMA_32_1B_MODEL_DIR to the pinned Meta snapshot")
	}
	verifyLlama32Snapshot(b, dir)
	var reference struct {
		Revision   string `json:"revision"`
		Schema     int    `json:"schema_version"`
		Generation struct {
			Prompt       string  `json:"input_text"`
			PromptIDs    []int64 `json:"prompt_ids"`
			SuffixIDs    []int64 `json:"suffix_ids"`
			MaxNewTokens int     `json:"max_new_tokens"`
			DecodedHex   string  `json:"decoded_bytes_hex"`
		} `json:"generation"`
	}
	if err := readHFJSON(filepath.Join("testdata", "llama32_reference.json"), &reference); err != nil {
		b.Fatal(err)
	}
	want := reference.Generation
	if reference.Revision != llama32Revision || reference.Schema != 1 || len(want.SuffixIDs) < 2 {
		b.Fatal("benchmark requires the pinned generation reference and at least one decode step")
	}

	const contextCap = 512
	var (
		decoder  *Decoder
		settings HFGenerationConfig
		err      error
		start    = time.Now()
	)
	if loadMetal != nil {
		decoder, settings, err = loadMetal(b.Context(), dir, contextCap)
	} else {
		decoder, settings, err = LoadHFModel(dir, contextCap)
	}
	modelLoad := time.Since(start)
	if errors.Is(err, gpu.ErrUnavailable) {
		b.Skip(err)
	}
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		if closeErr := decoder.Close(); closeErr != nil {
			b.Error(closeErr)
		}
	})
	config := decoder.Config()
	start = time.Now()
	vocab, err := tokenizer.LoadLlama3(dir, config.VocabSize, settings.BOSID, settings.StopIDs)
	if err != nil {
		b.Fatal(err)
	}
	prompt, err := vocab.Encode(want.Prompt)
	tokenizerSetup := time.Since(start)
	if err != nil || !slices.Equal(prompt, want.PromptIDs) {
		b.Fatalf("encode prompt: got %v (%v), want %v", prompt, err, want.PromptIDs)
	}
	budget, err := decoder.generationBudget(prompt, want.MaxNewTokens, settings.StopIDs)
	if err != nil || budget != int64(len(want.SuffixIDs)) {
		b.Fatalf("generation budget: got %d (%v), want %d", budget, err, len(want.SuffixIDs))
	}
	capacity := int64(len(prompt)) + budget - 1
	cacheBytes := 2 * config.Layers * capacity * config.KVHeads * (config.Dim / config.Heads) * 4
	if decoder.metal != nil && decoder.metal.attention {
		cacheBytes *= 2
	}
	b.Logf("GOMAXPROCS=%d prompt=%q prompt_tokens=%d output_cap=%d context_cap=%d cache_capacity=%d KV_payload_bytes=%d",
		runtime.GOMAXPROCS(0), want.Prompt, len(prompt), want.MaxNewTokens, contextCap, capacity, cacheBytes)

	var cacheSetup, prefill, firstToken, decode time.Duration
	for b.Loop() {
		start = time.Now()
		session, sessionErr := decoder.NewSession(capacity)
		cacheSetup += time.Since(start)
		if sessionErr != nil {
			b.Fatal(sessionErr)
		}
		start = time.Now()
		row, stepErr := session.Prefill(prompt)
		prefill += time.Since(start)
		if stepErr != nil {
			b.Fatal(stepErr)
		}
		next, stepErr := greedyToken(row)
		firstToken += time.Since(start)
		if stepErr != nil {
			b.Fatal(stepErr)
		}
		suffix := []int64{next}
		start = time.Now()
		for range budget - 1 {
			row, stepErr = session.Decode(next)
			if stepErr != nil {
				b.Fatal(stepErr)
			}
			next, stepErr = greedyToken(row)
			if stepErr != nil {
				b.Fatal(stepErr)
			}
			suffix = append(suffix, next)
		}
		decode += time.Since(start)
		if !slices.Equal(suffix, want.SuffixIDs) || session.Position() != capacity {
			b.Fatalf("got suffix %v at position %d, want %v at %d", suffix, session.Position(), want.SuffixIDs, capacity)
		}
		if sessionErr = session.Close(); sessionErr != nil {
			b.Fatal(sessionErr)
		}
		text, decodeErr := vocab.Decode(suffix)
		if decodeErr != nil || hex.EncodeToString([]byte(text)) != want.DecodedHex {
			b.Fatalf("decode suffix: got %q (%v), want hex %s", text, decodeErr, want.DecodedHex)
		}
	}
	trials := float64(b.N)
	b.ReportMetric(modelLoad.Seconds(), "model-load-s")
	b.ReportMetric(tokenizerSetup.Seconds(), "tokenizer-setup-s")
	b.ReportMetric(cacheSetup.Seconds()/trials, "cache-setup-s/op")
	b.ReportMetric(prefill.Seconds()/trials, "prefill-s/op")
	b.ReportMetric(firstToken.Seconds()/trials, "first-token-s/op")
	b.ReportMetric(decode.Seconds()/trials, "decode-s/op")
	b.ReportMetric(float64(budget-1)*trials/decode.Seconds(), "decode-tok/s")
}
