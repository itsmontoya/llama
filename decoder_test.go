package llama

import (
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func TestDecoderCausalPrefixes(t *testing.T) {
	for _, name := range fixtureNames {
		t.Run(name, func(t *testing.T) {
			model, fixture := readFixture(t, name)
			decoder, err := NewDecoder(model.Config, model.Weights)
			if err != nil {
				t.Fatal(err)
			}
			tokens := fixture.Cases[0].Tokens
			full, err := decoder.Forward(tokens)
			if err != nil {
				t.Fatal(err)
			}
			fullValues := tensorValues(t, full)
			for length := 1; length <= len(tokens); length++ {
				prefix, err := decoder.Forward(tokens[:length])
				if err != nil {
					t.Fatal(err)
				}
				assertClose(t, fmt.Sprintf("prefix %d", length), tensorValues(t, prefix), asFloat64(fullValues[:int64(length)*model.Config.VocabSize]))
			}
			changed := slices.Clone(tokens)
			changed[len(changed)-1] = (changed[len(changed)-1] + 1) % model.Config.VocabSize
			future, err := decoder.Forward(changed)
			if err != nil {
				t.Fatal(err)
			}
			futureValues := tensorValues(t, future)
			boundary := int64(len(tokens)-1) * model.Config.VocabSize
			assertClose(t, "future-token isolation", futureValues[:boundary], asFloat64(fullValues[:boundary]))
			if slices.Equal(futureValues[boundary:], fullValues[boundary:]) {
				t.Fatal("fixture's last-token logits did not respond to a changed token")
			}
		})
	}
}

func TestDecoderOwnsWeightsAndOutputs(t *testing.T) {
	model, fixture := readFixture(t, "tiny_gqa")
	decoder, err := NewDecoder(model.Config, model.Weights)
	if err != nil {
		t.Fatal(err)
	}
	tokens := fixture.Cases[0].Tokens
	logits, trace, err := decoder.ForwardWithTrace(tokens)
	if err != nil {
		t.Fatal(err)
	}
	saved := make(map[string][]float32)
	for name, value := range trace {
		saved[name] = tensorValues(t, value)
	}
	// Mutate all constructor-owned input slices after successful initialization.
	payloads := [][]float32{model.Weights.TokenEmbedding, model.Weights.FinalNorm, model.Weights.Output}
	for _, block := range model.Weights.Layers {
		payloads = append(payloads, block.AttentionNorm, block.WQ, block.WK, block.WV,
			block.WO, block.FFNNorm, block.WGate, block.WUp, block.WDown)
	}
	for _, payload := range payloads {
		for index := range payload {
			payload[index] = float32(math.NaN())
		}
	}
	if _, err = decoder.Forward([]int64{0}); err != nil {
		t.Fatal(err)
	}
	for name, value := range trace {
		if !slices.Equal(saved[name], tensorValues(t, value)) {
			t.Fatalf("later call overwrote %s", name)
		}
	}
	// Mutating a returned tensor must not expose the decoder's private weights.
	for _, value := range []*tensor.Tensor{trace["embedding"], logits} {
		zeros, err := tensor.New(tensor.Float32, value.Shape())
		if err != nil {
			t.Fatal(err)
		}
		if err = tensor.AddInto(zeros, zeros, value); err != nil {
			t.Fatal(err)
		}
	}
	again, err := decoder.Forward(tokens)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "private weights after external writes", tensorValues(t, again), fixture.Cases[0].Trace["logits"].Values)
	if decoder.output != decoder.embedding {
		t.Fatal("tied output projection duplicated embedding storage")
	}
}

func TestDecoderConcurrentForward(t *testing.T) {
	model, fixture := readFixture(t, "tiny_gqa")
	decoder, err := NewDecoder(model.Config, model.Weights)
	if err != nil {
		t.Fatal(err)
	}
	var group sync.WaitGroup
	errors := make(chan error, 4)
	for worker := 0; worker < 4; worker++ {
		group.Go(func() {
			for _, test := range fixture.Cases {
				logits, err := decoder.Forward(test.Tokens)
				if err != nil {
					errors <- err
					return
				}
				actual, err := logits.Float32Values()
				if err != nil {
					errors <- err
					return
				}
				for index, value := range actual {
					expected := test.Trace["logits"].Values[index]
					if !finite(float64(value)) || math.Abs(float64(value)-expected) > referenceTolerance+referenceTolerance*math.Abs(expected) {
						errors <- fmt.Errorf("concurrent output %d differs from reference", index)
						return
					}
				}
			}
		})
	}
	group.Wait()
	close(errors)
	for err := range errors {
		t.Fatal(err)
	}
}

func TestNewDecoderValidation(t *testing.T) {
	for _, test := range []struct {
		name   string
		modify func(*modelFixture)
	}{
		{"zero dimension", func(m *modelFixture) { m.Config.Dim = 0 }},
		{"negative FFN", func(m *modelFixture) { m.Config.FFNDim = -1 }},
		{"zero layers", func(m *modelFixture) { m.Config.Layers = 0 }},
		{"zero heads", func(m *modelFixture) { m.Config.Heads = 0 }},
		{"zero KV heads", func(m *modelFixture) { m.Config.KVHeads = 0 }},
		{"indivisible width", func(m *modelFixture) { m.Config.Dim = 17 }},
		{"invalid GQA", func(m *modelFixture) { m.Config.KVHeads = 3 }},
		{"odd rotary width", func(m *modelFixture) { m.Config.Dim = 12 }},
		{"zero vocabulary", func(m *modelFixture) { m.Config.VocabSize = 0 }},
		{"zero context", func(m *modelFixture) { m.Config.MaxContext = 0 }},
		{"overflow", func(m *modelFixture) { m.Config.MaxContext = math.MaxInt64 }},
		{"zero epsilon", func(m *modelFixture) { m.Config.NormEpsilon = 0 }},
		{"NaN epsilon", func(m *modelFixture) { m.Config.NormEpsilon = float32(math.NaN()) }},
		{"infinite epsilon", func(m *modelFixture) { m.Config.NormEpsilon = float32(math.Inf(1)) }},
		{"zero theta", func(m *modelFixture) { m.Config.RopeTheta = 0 }},
		{"NaN theta", func(m *modelFixture) { m.Config.RopeTheta = math.NaN() }},
		{"infinite theta", func(m *modelFixture) { m.Config.RopeTheta = math.Inf(1) }},
		{"layer count", func(m *modelFixture) { m.Weights.Layers = nil }},
		{"embedding shape", func(m *modelFixture) { m.Weights.TokenEmbedding = nil }},
		{"norm shape", func(m *modelFixture) { m.Weights.FinalNorm = nil }},
		{"attention norm shape", func(m *modelFixture) { m.Weights.Layers[0].AttentionNorm = nil }},
		{"query shape", func(m *modelFixture) { m.Weights.Layers[0].WQ = nil }},
		{"key shape", func(m *modelFixture) { m.Weights.Layers[0].WK = nil }},
		{"value shape", func(m *modelFixture) { m.Weights.Layers[0].WV = nil }},
		{"attention output shape", func(m *modelFixture) { m.Weights.Layers[0].WO = nil }},
		{"FFN norm shape", func(m *modelFixture) { m.Weights.Layers[0].FFNNorm = nil }},
		{"gate shape", func(m *modelFixture) { m.Weights.Layers[0].WGate = nil }},
		{"up shape", func(m *modelFixture) { m.Weights.Layers[0].WUp = nil }},
		{"down shape", func(m *modelFixture) { m.Weights.Layers[0].WDown = nil }},
		{"ambiguous tied output", func(m *modelFixture) { m.Weights.Output = []float32{1} }},
		{"missing untied output", func(m *modelFixture) { m.Config.TieEmbeddings = false }},
		{"NaN embedding", func(m *modelFixture) { m.Weights.TokenEmbedding[0] = float32(math.NaN()) }},
		{"infinite norm", func(m *modelFixture) { m.Weights.FinalNorm[0] = float32(math.Inf(1)) }},
		{"nonfinite block", func(m *modelFixture) { m.Weights.Layers[0].WGate[0] = float32(math.Inf(-1)) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			model, _ := readFixture(t, "tiny_gqa")
			test.modify(&model)
			decoder, err := NewDecoder(model.Config, model.Weights)
			if err == nil || decoder != nil {
				t.Fatal("invalid model accepted")
			}
		})
	}
}

func TestDecoderForwardValidationAndRecovery(t *testing.T) {
	model, fixture := readFixture(t, "tiny_gqa")
	decoder, err := NewDecoder(model.Config, model.Weights)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name    string
		decoder *Decoder
		tokens  []int64
	}{
		{"nil decoder", nil, []int64{0}},
		{"zero decoder", &Decoder{}, []int64{0}},
		{"empty prefix", decoder, nil},
		{"negative ID", decoder, []int64{-1}},
		{"large ID", decoder, []int64{model.Config.VocabSize}},
		{"invalid later ID", decoder, []int64{0, 1, -1}},
		{"context overflow", decoder, make([]int64, model.Config.MaxContext+1)},
	} {
		t.Run(test.name, func(t *testing.T) {
			logits, trace, err := test.decoder.ForwardWithTrace(test.tokens)
			if err == nil || logits != nil || trace != nil {
				t.Fatal("invalid call returned a result or no error")
			}
		})
	}
	boundary, err := decoder.Forward(make([]int64, model.Config.MaxContext))
	if err != nil || !slices.Equal(boundary.Shape(), []int64{model.Config.MaxContext, model.Config.VocabSize}) {
		t.Fatalf("exact context limit rejected: %v", err)
	}
	valid, err := decoder.Forward(fixture.Cases[0].Tokens)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "after invalid calls", tensorValues(t, valid), fixture.Cases[0].Trace["logits"].Values)
}

func TestRMSNormAnchor(t *testing.T) {
	x, err := tensor.FromFloat32([]int64{2, 2}, []float32{0, 0, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	weight, err := tensor.FromFloat32([]int64{2}, []float32{2, 0.5})
	if err != nil {
		t.Fatal(err)
	}
	result, err := rmsNorm(x, weight, 3.5)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "RMSNorm anchor", tensorValues(t, result), []float64{0, 0, 1.5, 0.5})
}
