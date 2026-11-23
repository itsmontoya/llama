//go:build darwin && arm64 && cgo && metal && !purego

package llama

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/itsmontoya/llama/internal/safetensors"
	"github.com/itsmontoya/neuralnetwork/gpu"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

func loadTestMetalAttention(t *testing.T, dir string, capacity int64) (out *Decoder) {
	t.Helper()
	var decoder, _, err = LoadHFModelMetalAttention(t.Context(), dir, capacity)
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

func newTestMetalAttentionSession(t *testing.T, decoder *Decoder, capacity int64) (out *Session) {
	t.Helper()
	var session, err = decoder.NewSession(capacity)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if closeErr := session.Close(); closeErr != nil {
			t.Error(closeErr)
		}
	})
	return session
}

func TestMetalAttentionSynthetic(t *testing.T) {
	for _, name := range fixtureNames {
		t.Run(name, func(t *testing.T) {
			var fixture, reference = readFixture(t, name)
			var device, err = gpu.OpenDefault()
			if errors.Is(err, gpu.ErrUnavailable) {
				t.Skip(err)
			}
			if err != nil {
				t.Fatal(err)
			}
			var model = &metalModel{device: device, attention: true}
			t.Cleanup(func() {
				if closeErr := model.close(); closeErr != nil {
					t.Error(closeErr)
				}
			})
			var config = fixture.Config
			var decoder = &Decoder{config: config, blocks: make([]blockWeights, config.Layers), metal: model}
			if err = decoder.loadWeights(t.Context(), fixture.Weights, func(_ int, weight decoderWeight) (string, []float32, error) {
				return weight.name, weight.values, nil
			}); err != nil {
				t.Fatal(err)
			}
			if err = model.loadRotary(t.Context(), config); err != nil {
				t.Fatal(err)
			}
			var largest float64
			for _, check := range reference.Cases {
				var logits *tensor.Tensor
				if logits, err = decoder.Forward(check.Tokens); err != nil {
					t.Fatal(err)
				}
				var want = check.Trace["logits"]
				if !slices.Equal(logits.Shape(), want.Shape) {
					t.Fatalf("%s: got shape %v, want %v", check.Name, logits.Shape(), want.Shape)
				}
				largest = max(largest, assertClose(t, check.Name, tensorValues(t, logits), want.Values))
				var session = newTestMetalAttentionSession(t, decoder, int64(len(check.Tokens)))
				var row []float32
				if row, err = session.Prefill(check.Tokens[:1]); err != nil {
					t.Fatal(err)
				}
				var vocab = int(config.VocabSize)
				largest = max(largest, assertClose(t, check.Name+" prefill", row, want.Values[:vocab]))
				for i, token := range check.Tokens[1:] {
					if row, err = session.Decode(token); err != nil {
						t.Fatal(err)
					}
					largest = max(largest, assertClose(t, check.Name+" decode", row, want.Values[(i+1)*vocab:(i+2)*vocab]))
				}
			}
			t.Logf("maximum FP64 logit difference %.9g under existing synthetic tolerance", largest)
		})
	}
}

func TestMetalAttentionFailureAndRetry(t *testing.T) {
	var dir = t.TempDir()
	var config, weights, entries = makeHFCheckpointFixture(t, dir, true)
	var last = entries[len(entries)-1]
	var saved = last.values[0]
	last.values[0] = float32(math.NaN())
	writeHFFixtureTensors(t, dir, safetensors.F32, entries)
	var decoder, _, err = LoadHFModelMetalAttention(t.Context(), dir, config.MaxContext)
	if errors.Is(err, gpu.ErrUnavailable) {
		t.Skip(err)
	}
	if decoder != nil || err == nil || !strings.Contains(err.Error(), last.name+" contains nonfinite") {
		t.Fatalf("partial attention load: decoder %v, error %v", decoder, err)
	}
	last.values[0] = saved
	writeHFFixtureTensors(t, dir, safetensors.F32, entries)
	decoder = loadTestMetalAttention(t, dir, config.MaxContext)
	if decoder.output != decoder.embedding || decoder.norm != nil || decoder.metalNorm == nil {
		t.Fatal("tied output or final norm has the wrong ownership")
	}
	for layer, block := range decoder.blocks {
		if block.wq != nil || block.wk != nil || block.wv != nil || block.wo != nil ||
			block.attentionNorm != nil || block.ffnNorm != nil || block.gate != nil {
			t.Fatalf("layer %d has duplicate CPU weights", layer)
		}
	}
	if logits, trace, traceErr := decoder.ForwardWithTrace([]int64{1}); logits != nil || trace != nil || traceErr == nil ||
		!strings.Contains(traceErr.Error(), "attention probabilities") {
		t.Fatalf("unexpected result for unsupported trace: %v %v %v", logits, trace, traceErr)
	}

	var prefix = []int64{1, 10, 4, 7}
	var session = newTestMetalAttentionSession(t, decoder, 5)
	if session.layers != nil || session.scores != nil {
		t.Fatal("Metal session allocated CPU KV or score storage")
	}
	var canceled, cancel = context.WithCancel(t.Context())
	cancel()
	var row []float32
	row, err = session.PrefillContext(canceled, prefix)
	if row != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled prefill: %d logits, error %v", len(row), err)
	}
	assertMetalCachePosition(t, session, 0)
	var cpu *Decoder
	if cpu, err = NewDecoder(config, weights); err != nil {
		t.Fatal(err)
	}
	if row, err = session.Prefill(prefix); err != nil {
		t.Fatal(err)
	}
	compareCachedRow(t, cpu, prefix, row)
	var prefilled, retained = row, slices.Clone(row)
	assertMetalCachePosition(t, session, 4)

	var submitted, cancelSubmitted = context.WithCancel(t.Context())
	defer cancelSubmitted()
	row, err = session.DecodeContext(cancelAtMetalWait{Context: submitted, cancel: cancelSubmitted}, 2)
	if row != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled decode: %d logits, error %v", len(row), err)
	}
	assertMetalCachePosition(t, session, 4)

	var block = &decoder.blocks[len(decoder.blocks)-1]
	var down = block.metal.down
	block.metal.down = nil
	row, err = session.Decode(3)
	block.metal.down = down
	if row != nil || !errors.Is(err, gpu.ErrInvalid) {
		t.Fatalf("failed final layer: %d logits, error %v", len(row), err)
	}
	assertMetalCachePosition(t, session, 4)

	// Fail the final classifier after the Metal command successfully wrote all
	// staging slots. Each slot is still unpublished and cancellable.
	var output = decoder.output
	var invalid = make([]float32, config.VocabSize*config.Dim)
	invalid[0] = float32(math.NaN())
	if decoder.output, err = tensor.FromFloat32([]int64{config.VocabSize, config.Dim}, invalid); err != nil {
		t.Fatal(err)
	}
	row, err = session.Decode(5)
	decoder.output = output
	if row != nil || err == nil || !strings.Contains(err.Error(), "nonfinite logit") {
		t.Fatalf("invalid classifier output: %d logits, error %v", len(row), err)
	}
	assertMetalCachePosition(t, session, 4)

	if row, err = session.Decode(0); err != nil {
		t.Fatal(err)
	}
	compareCachedRow(t, decoder, append(prefix, 0), row)
	assertMetalCachePosition(t, session, 5)
	assertSessionFailure(t, session, 5, "capacity", func() ([]float32, error) { return session.Decode(0) })
	if !slices.Equal(prefilled, retained) {
		t.Fatal("decode mutated retained logits")
	}
	var alias = *session
	alias.Reset()
	assertMetalCachePosition(t, session, 0)
	if row, err = session.Prefill(prefix); err != nil || !slices.Equal(row, retained) {
		t.Fatalf("reset did not restore same prefill: %d logits, error %v", len(row), err)
	}
	if err = alias.Close(); err != nil {
		t.Fatal(err)
	}
	if row, err = session.Decode(0); row != nil || !errors.Is(err, gpu.ErrClosed) {
		t.Fatalf("decode after closing copied session handle: %d logits, error %v", len(row), err)
	}
}

func assertMetalCachePosition(t *testing.T, session *Session, want int64) {
	t.Helper()
	if session.Position() != want {
		t.Fatalf("session committed %d, want %d", session.Position(), want)
	}
	for index, cache := range session.metal.caches {
		if cache.Len() != want {
			t.Fatalf("layer %d committed %d, want %d", index, cache.Len(), want)
		}
	}
}

type holdMetalWait struct {
	context.Context
	waiting chan struct{}
	resume  chan struct{}
}

func (c holdMetalWait) Done() <-chan struct{} {
	close(c.waiting)
	<-c.resume
	return c.Context.Done()
}

func TestMetalAttentionCloseDuringStep(t *testing.T) {
	var dir = t.TempDir()
	var config, _, entries = makeHFCheckpointFixture(t, dir, false)
	writeHFFixtureTensors(t, dir, safetensors.F32, entries)
	var decoder = loadTestMetalAttention(t, dir, config.MaxContext)
	var active = newTestMetalAttentionSession(t, decoder, 5)
	var other = newTestMetalAttentionSession(t, decoder, 5)
	var wait = holdMetalWait{Context: t.Context(), waiting: make(chan struct{}), resume: make(chan struct{})}
	var stepped = make(chan error, 1)
	go func() {
		var _, err = active.PrefillContext(wait, []int64{1, 10, 4, 7})
		stepped <- err
	}()
	select {
	case <-wait.waiting:
	case err := <-stepped:
		t.Fatalf("step returned before reaching Metal Wait: %v", err)
	}
	var resumed bool
	defer func() {
		if !resumed {
			close(wait.resume)
		}
	}()

	var otherClosed = make(chan error, 1)
	go func() { otherClosed <- other.Close() }()
	select {
	case err := <-otherClosed:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("closing an idle session waited for a different session")
	}

	var copied = *decoder
	var modelClosed = make(chan error, 1)
	go func() { modelClosed <- copied.Close() }()
	select {
	case err := <-modelClosed:
		t.Fatalf("model closed before active step drained: %v", err)
	case <-time.After(20 * time.Millisecond):
	}
	close(wait.resume)
	resumed = true
	if err := <-stepped; err != nil {
		t.Fatal(err)
	}
	if err := <-modelClosed; err != nil {
		t.Fatal(err)
	}
	if active.Position() != 4 || active.metal.caches != nil {
		t.Fatal("model close interrupted publication or retained the active session")
	}
	if _, err := active.Decode(0); !errors.Is(err, gpu.ErrClosed) {
		t.Fatalf("closed session decode: %v", err)
	}
	if _, err := decoder.NewSession(5); !errors.Is(err, gpu.ErrClosed) {
		t.Fatalf("closed original decoder new session: %v", err)
	}
	if _, err := decoder.Forward([]int64{1}); !errors.Is(err, gpu.ErrClosed) {
		t.Fatalf("closed original decoder forward: %v", err)
	}
}
