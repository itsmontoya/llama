package llama

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
	"os"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/itsmontoya/llama/internal/tokenizer"
)

const llama32ShortPrompt = "The capital of France is"
const llama32LongPrompt = "France is a country in Western Europe. It shares borders with Belgium, Germany, Switzerland, Italy, and Spain. The capital of France is"

func BenchmarkLlama32SupplementalPrefill(b *testing.B) {
	benchmarkLlama32Prefill(b, 29, 12366)
}

func BenchmarkLlama32SupplementalPrefill27(b *testing.B) {
	benchmarkLlama32Prefill(b, 27, 9822)
}

func BenchmarkLlama32SupplementalPrefill28(b *testing.B) {
	benchmarkLlama32Prefill(b, 28, 374)
}

func benchmarkLlama32Prefill(b *testing.B, rows int, wantWinner int64) {
	model, ids := setupLlama32SupplementalBenchmark(b, llama32LongPrompt, 29)
	ids = ids[:rows]

	var logits []float32
	var winner int64
	var prefill time.Duration
	for b.Loop() {
		b.StopTimer()
		session, err := model.NewSession(int64(len(ids)))
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		start := time.Now()
		logits, err = session.Prefill(ids)
		if err != nil {
			b.Fatal(err)
		}
		winner, err = greedyToken(logits)
		if err != nil {
			b.Fatal(err)
		}
		prefill += time.Since(start)
	}
	if winner != wantWinner {
		b.Fatalf("greedy token = %d, want %d", winner, wantWinner)
	}
	b.ReportMetric(prefill.Seconds()/float64(b.N), "prefill-s/op")
	b.Logf("greedy=%d FP32 logits SHA256=%x", winner, llama32LogitDigest(logits))
}

func BenchmarkLlama32SustainedDecode(b *testing.B) {
	const window = 8
	const steps = 4 * window
	model, ids := setupLlama32SupplementalBenchmark(b, llama32ShortPrompt, 6)
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

func BenchmarkLlama32ConcurrentSustainedDecode(b *testing.B) {
	benchmarkLlama32ConcurrentSustainedDecode(b, false)
}

func BenchmarkLlama32IndependentSustainedDecode(b *testing.B) {
	benchmarkLlama32ConcurrentSustainedDecode(b, true)
}

func benchmarkLlama32ConcurrentSustainedDecode(b *testing.B, independent bool) {
	const callers, steps = 2, 32
	model, ids := setupLlama32SupplementalBenchmark(b, llama32ShortPrompt, 6)
	models := [callers]*Decoder{model, model}
	if independent {
		var err error
		models[1], _, err = LoadHFModel(os.Getenv("LLAMA_32_1B_MODEL_DIR"), 512)
		if err != nil {
			b.Fatal(err)
		}
		if models[0].embedding == models[1].embedding {
			b.Fatal("independent decoders share embedding storage")
		}
	}
	capacity := int64(len(ids) + steps)
	type result struct {
		logits []float32
		suffix [1 + steps]int64
		err    error
		wall   time.Duration
	}
	var results [callers]result
	var elapsed time.Duration
	var sessionTime [callers]time.Duration
	for b.Loop() {
		b.StopTimer()
		var sessions [callers]*Session
		for i := range sessions {
			session, err := models[i].NewSession(capacity)
			if err != nil {
				b.Fatal(err)
			}
			row, err := session.Prefill(ids)
			if err != nil {
				b.Fatal(err)
			}
			winner, err := greedyToken(row)
			if err != nil {
				b.Fatal(err)
			}
			sessions[i] = session
			results[i].suffix[0] = winner
		}

		start := make(chan struct{})
		var ready, done sync.WaitGroup
		ready.Add(callers)
		var released time.Time
		for i, session := range sessions {
			done.Go(func() {
				ready.Done()
				<-start
				result := &results[i]
				for step := range steps {
					result.logits, result.err = session.Decode(result.suffix[step])
					if result.err != nil {
						return
					}
					result.suffix[step+1], result.err = greedyToken(result.logits)
					if result.err != nil {
						return
					}
				}
				result.wall = time.Since(released)
			})
		}
		ready.Wait()
		b.StartTimer()
		released = time.Now()
		close(start)
		done.Wait()
		elapsed += time.Since(released)
		b.StopTimer()

		for i, result := range results {
			if result.err != nil || sessions[i].Position() != capacity {
				b.Fatalf("session %d ended at position %d, want %d: %v", i, sessions[i].Position(), capacity, result.err)
			}
			sessionTime[i] += result.wall
		}
		if results[0].suffix != results[1].suffix || !slices.Equal(results[0].logits, results[1].logits) {
			b.Fatal("concurrent decode sessions disagree")
		}
		b.StartTimer()
	}
	tokens := float64(steps) * float64(b.N)
	b.ReportMetric(elapsed.Seconds()/float64(b.N), "decode-wall-s/op")
	b.ReportMetric(callers*tokens/elapsed.Seconds(), "aggregate-tok/s")
	b.ReportMetric(tokens/sessionTime[0].Seconds(), "session0-tok/s")
	b.ReportMetric(tokens/sessionTime[1].Seconds(), "session1-tok/s")
	b.ReportMetric(sessionTime[0].Seconds()/tokens, "session0-s/token")
	b.ReportMetric(sessionTime[1].Seconds()/tokens, "session1-s/token")
	b.Logf("callers=%d decode steps per session=%d positions=%d..%d", callers, steps, len(ids), capacity-1)
	b.Logf("suffix=%v final FP32 logits SHA256=%x", results[0].suffix, llama32LogitDigest(results[0].logits))
}

func BenchmarkLlama32ConcurrentPrefill(b *testing.B) {
	const callers = 2
	model, ids := setupLlama32SupplementalBenchmark(b, llama32ShortPrompt, 6)
	var baseline []float32
	var rows [callers][]float32
	for b.Loop() {
		b.StopTimer()
		var sessions [callers]*Session
		for i := range sessions {
			session, err := model.NewSession(int64(len(ids)))
			if err != nil {
				b.Fatal(err)
			}
			sessions[i] = session
		}
		b.StartTimer()
		var group sync.WaitGroup
		var errs [callers]error
		for i, session := range sessions {
			group.Go(func() { rows[i], errs[i] = session.Prefill(ids) })
		}
		group.Wait()
		b.StopTimer()
		for _, err := range errs {
			if err != nil {
				b.Fatal(err)
			}
		}
		if !slices.Equal(rows[0], rows[1]) {
			b.Fatal("concurrent prefill sessions disagree")
		}
		if baseline == nil {
			baseline = rows[0]
		} else if !slices.Equal(rows[0], baseline) {
			b.Fatal("concurrent prefill differs from retained first result")
		}
		b.StartTimer()
	}
	b.Logf("callers=%d prefill SHA256=%x", callers, llama32LogitDigest(rows[0]))
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
