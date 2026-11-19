package llama

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"

	"github.com/itsmontoya/llama/internal/tokenizer"
)

func BenchmarkLlama32CachedGeneration(b *testing.B) {
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
	start := time.Now()
	decoder, settings, err := LoadHFModel(dir, contextCap)
	modelLoad := time.Since(start)
	if err != nil {
		b.Fatal(err)
	}
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
	b.Logf("GOMAXPROCS=%d prompt=%q prompt_tokens=%d output_cap=%d context_cap=%d cache_capacity=%d KV_payload_bytes=%d",
		runtime.GOMAXPROCS(0), want.Prompt, len(prompt), want.MaxNewTokens, contextCap, capacity, cacheBytes)

	var cacheSetup, prefill, firstToken, decode time.Duration
	for b.Loop() {
		start = time.Now()
		session, err := decoder.NewSession(capacity)
		cacheSetup += time.Since(start)
		if err != nil {
			b.Fatal(err)
		}
		start = time.Now()
		row, err := session.Prefill(prompt)
		prefill += time.Since(start)
		if err != nil {
			b.Fatal(err)
		}
		next, err := greedyToken(row)
		firstToken += time.Since(start)
		if err != nil {
			b.Fatal(err)
		}
		suffix := []int64{next}
		start = time.Now()
		for range budget - 1 {
			row, err = session.Decode(next)
			if err != nil {
				b.Fatal(err)
			}
			next, err = greedyToken(row)
			if err != nil {
				b.Fatal(err)
			}
			suffix = append(suffix, next)
		}
		decode += time.Since(start)
		if !slices.Equal(suffix, want.SuffixIDs) || session.Position() != capacity {
			b.Fatalf("got suffix %v at position %d, want %v at %d", suffix, session.Position(), want.SuffixIDs, capacity)
		}
		text, err := vocab.Decode(suffix)
		if err != nil || hex.EncodeToString([]byte(text)) != want.DecodedHex {
			b.Fatalf("decode suffix: got %q (%v), want hex %s", text, err, want.DecodedHex)
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
