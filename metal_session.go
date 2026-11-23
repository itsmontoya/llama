package llama

import (
	"context"
	"fmt"
	"sync"

	"github.com/itsmontoya/neuralnetwork/gpu"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

func (m *metalModel) newSession(config Config, capacity int64) (out *metalSession, err error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return nil, fmt.Errorf("llama: Metal attention: %w", gpu.ErrClosed)
	}
	var caches []*gpu.KVCache
	if caches, err = newMetalCaches(m.device, config, capacity); err != nil {
		return nil, err
	}
	var state = &metalSession{caches: caches}
	m.sessionsMu.Lock()
	defer m.sessionsMu.Unlock()
	if m.sessions == nil {
		m.sessions = make(map[*metalSession]struct{})
	}
	m.sessions[state] = struct{}{}
	return state, nil
}

func newMetalCaches(device *gpu.Device, config Config, capacity int64) (out []*gpu.KVCache, err error) {
	var width = config.Dim / config.Heads
	if _, _, err = tensor.ValidateShape([]int64{4, config.Layers, capacity, config.KVHeads, width}, tensor.Float32); err != nil {
		return nil, fmt.Errorf("llama: Metal session KV storage: %w", err)
	}
	var caches = make([]*gpu.KVCache, config.Layers)
	for layer := range caches {
		if caches[layer], err = gpu.NewKVCache(device, capacity, config.KVHeads, width); err != nil {
			closeMetalCaches(caches)
			return nil, fmt.Errorf("llama: allocate Metal KV cache for layer %d: %w", layer, err)
		}
	}
	return caches, nil
}

func closeMetalCaches(caches []*gpu.KVCache) {
	for _, cache := range caches {
		_ = cache.Close()
	}
}

// metalSession serializes steps, position reads, resets and cleanup. It is
// shared by any copies of the Session handle. A nil cache slice means closed.
type metalSession struct {
	mu       sync.Mutex
	position int64
	caches   []*gpu.KVCache
}

func (s *metalSession) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	closeMetalCaches(s.caches)
	s.caches = nil
}

func (s *metalSession) reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.caches == nil {
		return
	}
	for _, cache := range s.caches {
		// Step and close exclusion makes each private cache idle and open.
		_ = cache.Reset()
	}
	s.position = 0
}

func (s *Session) runMetalStep(ctx context.Context, tokens []int64, prefill bool) (out []float32, err error) {
	var d, state = s.decoder, s.metal
	d.metal.mu.RLock()
	defer d.metal.mu.RUnlock()
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.caches == nil {
		return nil, fmt.Errorf("llama: Metal session: %w", gpu.ErrClosed)
	}
	if prefill {
		err = s.validatePrefill(tokens, state.position)
	} else {
		err = s.validateDecode(tokens[0], state.position)
	}
	if err != nil {
		return nil, err
	}

	var updates = make([]*gpu.KVAppend, len(state.caches))
	defer discardMetalAppends(updates)
	var logits *tensor.Tensor
	if logits, err = d.metalLogits(ctx, tokens, state.position, state.caches, updates, true); err != nil {
		return nil, err
	}
	var next = state.position + int64(len(tokens))
	if out, err = sessionLogits(ctx, logits, next); err != nil {
		return nil, err
	}

	// A successful Wait made every append publishable. Both locks exclude
	// close, reset and concurrent steps through publication and cursor update.
	for layer, update := range updates {
		if err = update.Publish(); err != nil {
			return nil, fmt.Errorf("llama: publish Metal KV cache for layer %d: %w", layer, err)
		}
	}
	state.position = next
	return out, nil
}
