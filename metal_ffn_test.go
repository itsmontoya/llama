//go:build darwin && arm64 && cgo && metal && !purego

package llama

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/itsmontoya/llama/internal/safetensors"
	"github.com/itsmontoya/neuralnetwork/gpu"
)

func loadTestMetalModel(t *testing.T, dir string, maxContext int64) (out *Decoder) {
	t.Helper()
	var decoder, _, err = LoadHFModelMetal(t.Context(), dir, maxContext)
	if errors.Is(err, gpu.ErrUnavailable) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if closeErr := decoder.Close(); closeErr != nil {
			t.Error(closeErr)
		}
	})
	return decoder
}

func TestMetalFFNModelTrace(t *testing.T) {
	var dir = t.TempDir()
	var config, weights, entries = makeHFCheckpointFixture(t, dir, true)
	writeHFFixtureTensors(t, dir, safetensors.BF16, entries)
	var metal = loadTestMetalModel(t, dir, config.MaxContext)
	var cpu, err = NewDecoder(config, weights)
	if err != nil {
		t.Fatal(err)
	}
	for i, block := range metal.blocks {
		if block.gate != nil || block.up != nil || block.down != nil {
			t.Fatalf("layer %d retained prepared CPU FFN weights", i)
		}
		if block.metal.gate == nil || block.metal.up == nil || block.metal.down == nil {
			t.Fatalf("layer %d is missing Metal FFN weights", i)
		}
	}
	var prefix = []int64{1, 10, 4, 7}
	var logits, metalTrace, traceErr = metal.ForwardWithTrace(prefix)
	if traceErr != nil {
		t.Fatal(traceErr)
	}
	var _, cpuTrace, cpuErr = cpu.ForwardWithTrace(prefix)
	if cpuErr != nil {
		t.Fatal(cpuErr)
	}
	if len(metalTrace) != len(cpuTrace) || metalTrace["logits"] != logits {
		t.Fatal("Metal trace changed the available intermediates or logits ownership")
	}
	var maxError float64
	for name, want := range cpuTrace {
		var got = metalTrace[name]
		if got == nil || !slices.Equal(got.Shape(), want.Shape()) {
			t.Fatalf("%s missing or wrong shape", name)
		}
		var gotValues = tensorValues(t, got)
		for i, value := range tensorValues(t, want) {
			var actual = float64(gotValues[i])
			var delta = math.Abs(actual - float64(value))
			maxError = max(maxError, delta)
			if !finite(actual) || delta > 2e-5+2e-5*math.Abs(float64(value)) {
				t.Fatalf("%s[%d] = %g, want %g", name, i, actual, value)
			}
		}
	}
	t.Logf("maximum absolute error versus CPU trace %.9g", maxError)
}

func TestMetalFFNFailureAndRetry(t *testing.T) {
	var dir = t.TempDir()
	var config, _, entries = makeHFCheckpointFixture(t, dir, true)
	var last = entries[len(entries)-1]
	var previous = last.values[0]
	last.values[0] = float32(math.NaN())
	writeHFFixtureTensors(t, dir, safetensors.F32, entries)
	var decoder, _, err = LoadHFModelMetal(t.Context(), dir, config.MaxContext)
	if errors.Is(err, gpu.ErrUnavailable) {
		t.Skip(err)
	}
	if decoder != nil || err == nil || !strings.Contains(err.Error(), last.name+" contains nonfinite") {
		t.Fatalf("partial load: got decoder %v, error %v, want nonfinite %s failure", decoder, err, last.name)
	}
	last.values[0] = previous
	writeHFFixtureTensors(t, dir, safetensors.F32, entries)
	decoder = loadTestMetalModel(t, dir, config.MaxContext)
	var session *Session
	if session, err = decoder.NewSession(5); err != nil {
		t.Fatal(err)
	}
	var prefix = []int64{1, 10, 4, 7}
	var canceled, cancel = context.WithCancel(t.Context())
	cancel()

	var row []float32
	if row, err = session.PrefillContext(canceled, prefix); row != nil || !errors.Is(err, context.Canceled) || session.Position() != 0 {
		t.Fatalf("canceled prefill: logit count %d, position %d, error %v", len(row), session.Position(), err)
	}
	if row, err = session.Prefill(prefix); err != nil {
		t.Fatal(err)
	}
	var prefilled, retained = row, slices.Clone(row)

	var committed, cancelCommitted = context.WithCancel(t.Context())
	defer cancelCommitted()
	if row, err = session.DecodeContext(cancelAtMetalWait{Context: committed, cancel: cancelCommitted}, 0); row != nil ||
		!errors.Is(err, context.Canceled) || session.Position() != 4 {
		t.Fatalf("canceled decode: logit count %d, position %d, error %v", len(row), session.Position(), err)
	}
	// Poison the last layer's recording after earlier valid encodes and a
	// completed first layer. Restore the weight before retrying the same token.
	var block = &decoder.blocks[len(decoder.blocks)-1]
	var down = block.metal.down
	block.metal.down = nil
	row, err = session.Decode(0)
	block.metal.down = down
	if row != nil || !errors.Is(err, gpu.ErrInvalid) || session.Position() != 4 {
		t.Fatalf("failed recording: logit count %d, position %d, error %v", len(row), session.Position(), err)
	}
	if row, err = session.Decode(0); err != nil {
		t.Fatal(err)
	}
	compareCachedRow(t, decoder, append(prefix, 0), row)
	if !slices.Equal(prefilled, retained) {
		t.Fatal("decode mutated retained logits")
	}
	if err = decoder.Close(); err != nil {
		t.Fatal(err)
	}
	if logits, forwardErr := decoder.Forward(prefix); logits != nil || !errors.Is(forwardErr, gpu.ErrClosed) {
		t.Fatalf("closed model: logits %v, error %v", logits, forwardErr)
	}
}

// The released scope first asks for Done after a successful submission.
type cancelAtMetalWait struct {
	context.Context
	cancel context.CancelFunc
}

func (c cancelAtMetalWait) Done() (out <-chan struct{}) {
	c.cancel()
	return c.Context.Done()
}
