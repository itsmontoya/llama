package llama

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
)

const cachedATol = 1e-4
const cachedRTol = 1e-4

type cachedRowError struct {
	position int
	token    int
	got      float32
	want     float32
	abs      float64
	fraction float64
}

func compareCachedRow(t *testing.T, decoder *Decoder, prefix []int64, got []float32) (largest, fraction cachedRowError, winner int) {
	t.Helper()
	full, err := decoder.Forward(prefix)
	if err != nil {
		t.Fatal(err)
	}
	vocab := int(decoder.Config().VocabSize)
	values := tensorValues(t, full)
	if len(got) != vocab || len(values) != len(prefix)*vocab {
		t.Fatalf("position %d: cached/full logit sizes %d/%d, want %d/%d", len(prefix)-1, len(got), len(values), vocab, len(prefix)*vocab)
	}
	want := values[len(values)-vocab:]
	position := len(prefix) - 1
	var fullWinner int
	for id, value := range got {
		if !finite(float64(value)) || !finite(float64(want[id])) {
			t.Fatalf("position %d, token %d: nonfinite cached/full logits %g/%g", position, id, value, want[id])
		}
		difference := math.Abs(float64(value) - float64(want[id]))
		bound := cachedATol + cachedRTol*math.Abs(float64(want[id]))
		point := cachedRowError{position, id, value, want[id], difference, difference / bound}
		if id == 0 || point.abs > largest.abs {
			largest = point
		}
		if id == 0 || point.fraction > fraction.fraction {
			fraction = point
		}
		if value > got[winner] {
			winner = id
		}
		if want[id] > want[fullWinner] {
			fullWinner = id
		}
	}
	if fraction.fraction > 1 {
		t.Errorf("position %d, token %d: cached %g, full %g, tolerance fraction %.9g", position, fraction.token, fraction.got, fraction.want, fraction.fraction)
	}
	if winner != fullWinner {
		t.Errorf("position %d: cached winner %d differs from full-prefix winner %d", position, winner, fullWinner)
	}
	return largest, fraction, winner
}

func TestSessionSyntheticParity(t *testing.T) {
	for _, name := range fixtureNames {
		t.Run(name, func(t *testing.T) {
			model, fixture := readFixture(t, name)
			model.Config.RopeScaling = RopeScaling{
				Type: "llama3", Factor: 8, LowFreqFactor: 1, HighFreqFactor: 4, OriginalMaxPositionEmbeddings: 8,
			}
			decoder, err := NewDecoder(model.Config, model.Weights)
			if err != nil {
				t.Fatal(err)
			}
			prompt := fixture.Cases[0].Tokens
			tokens := make([]int64, model.Config.MaxContext)
			for index := range tokens {
				tokens[index] = prompt[index%len(prompt)]
			}
			for _, prefill := range []int{1, len(prompt), len(tokens)} {
				t.Run(fmt.Sprintf("prefill_%d", prefill), func(t *testing.T) {
					session, err := decoder.NewSession(int64(len(tokens)))
					if err != nil {
						t.Fatal(err)
					}
					keys := &session.layers[0].keys[0]
					values := &session.layers[0].values[0]
					logits, err := session.Prefill(tokens[:prefill])
					if err != nil || session.Position() != int64(prefill) {
						t.Fatalf("prefill position %d: %v", session.Position(), err)
					}
					compareCachedRow(t, decoder, tokens[:prefill], logits)
					for index, token := range tokens[prefill:] {
						position := prefill + index
						logits, err = session.Decode(token)
						if err != nil || session.Position() != int64(position+1) {
							t.Fatalf("decode position %d: %v", session.Position(), err)
						}
						compareCachedRow(t, decoder, tokens[:position+1], logits)
					}
					assertSessionFailure(t, session, int64(len(tokens)), "capacity", func() ([]float32, error) {
						return session.Decode(tokens[0])
					})
					if keys != &session.layers[0].keys[0] || values != &session.layers[0].values[0] {
						t.Fatal("decode replaced the cache backing storage")
					}
				})
			}
		})
	}
}

func assertSessionFailure(t *testing.T, session *Session, position int64, message string, call func() ([]float32, error)) {
	t.Helper()
	got, err := call()
	if got != nil || err == nil || !strings.Contains(err.Error(), message) {
		t.Fatalf("got logits %v and error %v, want nil logits and error containing %q", got, err, message)
	}
	if session != nil && session.Position() != position {
		t.Fatalf("failed call left position %d, want %d", session.Position(), position)
	}
}

func TestSessionCapacityValidation(t *testing.T) {
	model, _ := readFixture(t, "tiny_gqa")
	decoder, err := NewDecoder(model.Config, model.Weights)
	if err != nil {
		t.Fatal(err)
	}
	var nilDecoder *Decoder
	for _, test := range []struct {
		d        *Decoder
		capacity int64
		message  string
	}{
		{nilDecoder, 1, "uninitialized"},
		{&Decoder{}, 1, "uninitialized"},
		{decoder, 0, "capacity"},
		{decoder, -1, "capacity"},
		{decoder, model.Config.MaxContext + 1, "capacity"},
	} {
		session, err := test.d.NewSession(test.capacity)
		if session != nil || err == nil || !strings.Contains(err.Error(), test.message) {
			t.Fatalf("capacity %d: got session %v and error %v, want %q error", test.capacity, session, err, test.message)
		}
	}
	// Exercise the allocation guard independently of constructing these
	// impractically large model fixtures.
	kvWidth := model.Config.KVHeads * (model.Config.Dim / model.Config.Heads)
	for _, layers := range []int64{math.MaxInt64, int64(math.MaxInt)/(2*kvWidth*4) + 1} {
		oversized := *decoder
		oversized.config.Layers = layers
		session, err := oversized.NewSession(1)
		if session != nil || err == nil || !strings.Contains(err.Error(), "session KV storage") {
			t.Fatalf("layers %d: got session %v and error %v, want a checked storage failure", layers, session, err)
		}
	}
}

func TestSessionValidationResetAndOwnership(t *testing.T) {
	model, fixture := readFixture(t, "tiny_gqa")
	decoder, err := NewDecoder(model.Config, model.Weights)
	if err != nil {
		t.Fatal(err)
	}
	var nilSession *Session
	assertSessionFailure(t, nilSession, 0, "uninitialized", func() ([]float32, error) {
		return nilSession.Prefill([]int64{0})
	})
	emptySession := &Session{}
	assertSessionFailure(t, emptySession, 0, "uninitialized", func() ([]float32, error) {
		return emptySession.Decode(0)
	})

	const capacity = 4
	session, err := decoder.NewSession(capacity)
	if err != nil {
		t.Fatal(err)
	}
	assertSessionFailure(t, session, 0, "prefilled", func() ([]float32, error) { return session.Decode(0) })
	prompt := fixture.Cases[0].Tokens[:3]
	for _, test := range []struct {
		tokens  []int64
		message string
	}{
		{nil, "prefix length"},
		{[]int64{prompt[0], -1}, "out of range"},
		{[]int64{model.Config.VocabSize}, "out of range"},
		{make([]int64, capacity+1), "capacity"},
	} {
		assertSessionFailure(t, session, 0, test.message, func() ([]float32, error) { return session.Prefill(test.tokens) })
	}
	first, err := session.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	compareCachedRow(t, decoder, prompt, first)
	first[0] = 123
	assertSessionFailure(t, session, 3, "empty session", func() ([]float32, error) { return session.Prefill(prompt[:1]) })
	for _, invalid := range []int64{-1, model.Config.VocabSize} {
		assertSessionFailure(t, session, 3, "out of range", func() ([]float32, error) { return session.Decode(invalid) })
	}
	decoded, err := session.Decode(prompt[0])
	if err != nil {
		t.Fatal(err)
	}
	compareCachedRow(t, decoder, append(slices.Clone(prompt), prompt[0]), decoded)
	saved := slices.Clone(decoded)
	session.Reset()
	if session.Position() != 0 {
		t.Fatal("reset left committed tokens")
	}
	newPrompt := []int64{prompt[1], prompt[2]}
	reset, err := session.Prefill(newPrompt[:1])
	if err != nil {
		t.Fatal(err)
	}
	compareCachedRow(t, decoder, newPrompt[:1], reset)
	reset, err = session.Decode(newPrompt[1])
	if err != nil {
		t.Fatal(err)
	}
	compareCachedRow(t, decoder, newPrompt, reset)
	if first[0] != 123 || !slices.Equal(saved, decoded) {
		t.Fatal("later calls overwrote caller-owned logits")
	}
}

func TestSessionRetriesAfterNonfiniteProjection(t *testing.T) {
	config := Config{Dim: 2, FFNDim: 2, Layers: 1, Heads: 1, KVHeads: 1, VocabSize: 2, MaxContext: 2, NormEpsilon: 1e-5, RopeTheta: 10000}
	identity, zero, norm := []float32{1, 0, 0, 1}, make([]float32, 4), []float32{1, 1}
	decoder, err := NewDecoder(config, Weights{
		TokenEmbedding: identity,
		Layers: []LayerWeights{{
			AttentionNorm: norm, WQ: identity, WK: identity, WV: identity, WO: zero,
			FFNNorm: norm, WGate: zero, WUp: zero, WDown: zero,
		}},
		FinalNorm: norm, Output: []float32{1, 0, 0, math.MaxFloat32},
	})
	if err != nil {
		t.Fatal(err)
	}
	session, err := decoder.NewSession(config.MaxContext)
	if err != nil {
		t.Fatal(err)
	}
	assertSessionFailure(t, session, 0, "nonfinite logit", func() ([]float32, error) {
		return session.Prefill([]int64{1})
	})
	if _, err = session.Prefill([]int64{0}); err != nil {
		t.Fatal(err)
	}
	cache := session.layers[0]
	keys, values := slices.Clone(cache.keys[:2]), slices.Clone(cache.values[:2])
	assertSessionFailure(t, session, 1, "nonfinite logit", func() ([]float32, error) { return session.Decode(1) })
	if !slices.Equal(keys, cache.keys[:2]) || !slices.Equal(values, cache.values[:2]) {
		t.Fatal("failed decode overwrote committed history")
	}
	if cache.values[3] == 0 {
		t.Fatal("test did not reach the provisional cache row before the failure")
	}
	logits, err := session.Decode(0)
	if err != nil || session.Position() != 2 {
		t.Fatalf("retry position %d: %v", session.Position(), err)
	}
	compareCachedRow(t, decoder, []int64{0, 0}, logits)
	if cache.values[3] != 0 {
		t.Fatal("retry left a value from the failed decode")
	}
}

func TestSessionsShareDecoderConcurrently(t *testing.T) {
	model, _ := readFixture(t, "tiny_gqa")
	decoder, err := NewDecoder(model.Config, model.Weights)
	if err != nil {
		t.Fatal(err)
	}
	for worker := range int64(4) {
		t.Run(fmt.Sprintf("worker_%d", worker), func(t *testing.T) {
			t.Parallel()
			session, err := decoder.NewSession(3)
			if err != nil {
				t.Fatal(err)
			}
			tokens := []int64{worker, worker + 1, worker + 2}
			logits, err := session.Prefill(tokens[:1])
			if err != nil {
				t.Fatal(err)
			}
			compareCachedRow(t, decoder, tokens[:1], logits)
			for index, token := range tokens[1:] {
				logits, err = session.Decode(token)
				if err != nil {
					t.Fatal(err)
				}
				compareCachedRow(t, decoder, tokens[:index+2], logits)
			}
		})
	}
}
