package llama

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
	"os"
	"testing"
	"time"

	"github.com/itsmontoya/llama/internal/tokenizer"
)

func BenchmarkLlama32SupplementalPrefill(b *testing.B) {
	const prompt = "France is a country in Western Europe. It shares borders with Belgium, Germany, Switzerland, Italy, and Spain. The capital of France is"
	model, ids := setupLlama32SupplementalBenchmark(b, prompt, 29)

	var logits []float32
	var winner int64
	for b.Loop() {
		b.StopTimer()
		session, err := model.NewSession(int64(len(ids)))
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		logits, err = session.Prefill(ids)
		if err != nil {
			b.Fatal(err)
		}
		winner, err = greedyToken(logits)
		if err != nil {
			b.Fatal(err)
		}
	}
	if winner != 12366 {
		b.Fatalf("greedy token = %d, want 12366", winner)
	}
	b.Logf("greedy=%d FP32 logits SHA256=%x", winner, llama32LogitDigest(logits))
}

func BenchmarkLlama32SustainedDecode(b *testing.B) {
	const prompt = "The capital of France is"
	const window = 8
	const steps = 4 * window
	model, ids := setupLlama32SupplementalBenchmark(b, prompt, 6)
	capacity := int64(len(ids) + steps)

	var logits []float32
	var suffix [1 + steps]int64
	var decodeTime [steps / window]time.Duration
	for b.Loop() {
		b.StopTimer()
		session, err := model.NewSession(capacity)
		if err != nil {
			b.Fatal(err)
		}
		logits, err = session.Prefill(ids)
		if err != nil {
			b.Fatal(err)
		}
		suffix[0], err = greedyToken(logits)
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		for group := range decodeTime {
			start := time.Now()
			for step := group * window; step < (group+1)*window; step++ {
				logits, err = session.Decode(suffix[step])
				if err != nil {
					b.Fatal(err)
				}
				suffix[step+1], err = greedyToken(logits)
				if err != nil {
					b.Fatal(err)
				}
			}
			decodeTime[group] += time.Since(start)
		}
		if session.Position() != capacity {
			b.Fatalf("session position = %d, want %d", session.Position(), capacity)
		}
	}
	var elapsed time.Duration
	for _, duration := range decodeTime {
		elapsed += duration
	}
	tokens := float64(steps) * float64(b.N)
	b.ReportMetric(elapsed.Seconds()/tokens, "decode-s/token")
	b.ReportMetric(tokens/elapsed.Seconds(), "decode-tok/s")
	b.Logf("decode steps=%d positions=%d..%d", steps, len(ids), capacity-1)
	b.Logf("total time per %d-step window across %d iterations: %v", window, b.N, decodeTime)
	b.Logf("suffix=%v final FP32 logits SHA256=%x", suffix, llama32LogitDigest(logits))
}

func llama32LogitDigest(logits []float32) [sha256.Size]byte {
	payload := make([]byte, 4*len(logits))
	for index, value := range logits {
		binary.LittleEndian.PutUint32(payload[4*index:], math.Float32bits(value))
	}
	return sha256.Sum256(payload)
}

func setupLlama32SupplementalBenchmark(b *testing.B, prompt string, wantTokens int) (*Decoder, []int64) {
	b.Helper()
	dir := os.Getenv("LLAMA_32_1B_MODEL_DIR")
	if dir == "" {
		b.Skip("set LLAMA_32_1B_MODEL_DIR to the pinned Meta snapshot")
	}
	verifyLlama32Snapshot(b, dir)
	model, settings, err := LoadHFModel(dir, 512)
	if err != nil {
		b.Fatal(err)
	}
	vocab, err := tokenizer.LoadLlama3(dir, model.Config().VocabSize, settings.BOSID, settings.StopIDs)
	if err != nil {
		b.Fatal(err)
	}
	ids, err := vocab.Encode(prompt)
	if err != nil || len(ids) != wantTokens {
		b.Fatalf("encode prompt: got %v (%v), want %d tokens", ids, err, wantTokens)
	}
	b.Logf("prompt=%q tokens=%d", prompt, len(ids))
	return model, ids
}
