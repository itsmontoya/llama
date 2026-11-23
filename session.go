package llama

import (
	"context"
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

// Session holds bounded, batch-one FP32 history for a single prompt and its
// continuation. Separate sessions can use the same decoder concurrently.
// A session must not be copied, and its callers must synchronize any concurrent
// use of that session. Use Decoder.NewSession to construct one.
type Session struct {
	decoder  *Decoder
	capacity int64
	position int64
	layers   []layerCache
	scores   []float32
}

// Each plane stores [position, KV head, head width]. Keys have already been
// rotated. Values retain their projection order.
type layerCache struct {
	keys, values []float32
}

// NewSession allocates space for capacity tokens, including the prompt.
// Capacity must be in [1, Config().MaxContext].
func (d *Decoder) NewSession(capacity int64) (*Session, error) {
	if d == nil || d.embedding == nil {
		return nil, fmt.Errorf("llama: decoder is uninitialized")
	}
	c := d.config
	if capacity <= 0 || capacity > c.MaxContext {
		return nil, fmt.Errorf("llama: session capacity must be in [1,%d], got %d", c.MaxContext, capacity)
	}
	headWidth := c.Dim / c.Heads
	elements, _, err := tensor.ValidateShape([]int64{2, c.Layers, capacity, c.KVHeads, headWidth}, tensor.Float32)
	if err != nil {
		return nil, fmt.Errorf("llama: session KV storage: %w", err)
	}
	storage := make([]float32, elements)
	plane := capacity * c.KVHeads * headWidth
	layers := make([]layerCache, c.Layers)
	for index := range layers {
		start := 2 * int64(index) * plane
		middle, end := start+plane, start+2*plane
		layers[index] = layerCache{keys: storage[start:middle:middle], values: storage[middle:end:end]}
	}
	return &Session{decoder: d, capacity: capacity, layers: layers, scores: make([]float32, capacity)}, nil
}

// Position returns the number of committed tokens and the absolute position
// of the next token. A failed call leaves Position and existing history intact.
func (s *Session) Position() int64 {
	return s.position
}

// Reset discards the logical history so the session can prefill another
// prompt. Neither reset nor later steps modify earlier returned logits.
func (s *Session) Reset() {
	s.position = 0
}

// Prefill evaluates tokens as one causal prompt starting at position zero.
// It returns the caller-owned final logit row and commits the entire prompt
// on success. The session must be empty. Callers must synchronize any
// mutation of tokens while Prefill runs.
func (s *Session) Prefill(tokens []int64) ([]float32, error) {
	return s.PrefillContext(context.Background(), tokens)
}

// PrefillContext is Prefill with cancellation. It drains outstanding Metal
// work before returning. A canceled step leaves Position unchanged.
func (s *Session) PrefillContext(ctx context.Context, tokens []int64) (out []float32, err error) {
	if s == nil || s.decoder == nil {
		return nil, fmt.Errorf("llama: session is uninitialized")
	}
	if s.position != 0 {
		return nil, fmt.Errorf("llama: prefill requires an empty session")
	}
	if int64(len(tokens)) > s.capacity {
		return nil, fmt.Errorf("llama: prompt length %d exceeds session capacity %d", len(tokens), s.capacity)
	}
	if err = s.decoder.validatePrefix(tokens); err != nil {
		return nil, err
	}
	x, err := s.decoder.forwardPrefix(ctx, tokens, nil, func(layer int, k, v *tensor.Tensor) error {
		return s.storeKV(layer, 0, k, v)
	})
	if err != nil {
		return nil, err
	}
	position := int64(len(tokens))
	last, err := gather(x, []int64{position - 1}, 0)
	if err != nil {
		return nil, fmt.Errorf("llama: final prompt row: %w", err)
	}
	return s.commit(ctx, last, position)
}

// Decode appends a token at Position and returns its caller-owned logit row.
// It requires a successful prefill and remaining capacity. After a failure,
// retry at the same position or call Reset. Any provisional rows beyond the
// committed history are overwritten before attention can read them again.
func (s *Session) Decode(token int64) ([]float32, error) {
	return s.DecodeContext(context.Background(), token)
}

// DecodeContext is Decode with cancellation. It drains outstanding Metal
// work before returning. A canceled step leaves Position unchanged.
func (s *Session) DecodeContext(ctx context.Context, token int64) (out []float32, err error) {
	if s == nil || s.decoder == nil {
		return nil, fmt.Errorf("llama: session is uninitialized")
	}
	if s.position == 0 {
		return nil, fmt.Errorf("llama: decode requires a prefilled session")
	}
	if s.position == s.capacity {
		return nil, fmt.Errorf("llama: session has reached capacity %d", s.capacity)
	}
	d := s.decoder
	if token < 0 || token >= d.config.VocabSize {
		return nil, fmt.Errorf("llama: token at position %d is out of range: %d", s.position, token)
	}
	var scratch *metalFFNScratch
	if scratch, err = d.metal.begin(ctx, d.config, 1); err != nil {
		return nil, err
	}
	defer scratch.close()
	x, err := gather(d.embedding, []int64{token}, 0)
	if err != nil {
		return nil, fmt.Errorf("llama: embedding: %w", err)
	}
	cosine, sine, err := rotaryTables(s.position, 1, d.config)
	if err != nil {
		return nil, err
	}
	next := s.position + 1
	for index, block := range d.blocks {
		attend := func(q, k, v *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
			if storeErr := s.storeKV(index, s.position, k, v); storeErr != nil {
				return nil, nil, storeErr
			}
			result, attentionErr := attendCached(q, s.layers[index], s.scores[:next], d.config)
			return result, nil, attentionErr
		}
		if x, err = d.forwardBlock(ctx, x, block, cosine, sine, attend, scratch, nil, ""); err != nil {
			return nil, fmt.Errorf("llama: layer %d: %w", index, err)
		}
	}
	return s.commit(ctx, x, next)
}

func (s *Session) storeKV(layer int, position int64, k, v *tensor.Tensor) error {
	keys, err := k.Float32Values()
	if err != nil {
		return fmt.Errorf("read rotated keys: %w", err)
	}
	values, err := v.Float32Values()
	if err != nil {
		return fmt.Errorf("read values: %w", err)
	}
	c := s.decoder.config
	start := position * c.KVHeads * (c.Dim / c.Heads)
	copy(s.layers[layer].keys[start:], keys)
	copy(s.layers[layer].values[start:], values)
	return nil
}

func (s *Session) commit(ctx context.Context, x *tensor.Tensor, next int64) (out []float32, err error) {
	logits, err := s.decoder.projectLogits(x, nil)
	if err != nil {
		return nil, err
	}
	values, err := logits.Float32Values()
	if err != nil {
		return nil, fmt.Errorf("llama: read session logits: %w", err)
	}
	for id, value := range values {
		if !finite(float64(value)) {
			return nil, fmt.Errorf("llama: nonfinite logit at position %d for ID %d", next-1, id)
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	s.position = next
	return values, nil
}

// attendCached computes one query row. The length of scores is the populated
// position count, including the staged current row. Scratch is reused across
// heads and layers, and K/V are read directly in their stored layout.
func attendCached(q *tensor.Tensor, history layerCache, scores []float32, c Config) (*tensor.Tensor, error) {
	queries, err := q.Float32Values()
	if err != nil {
		return nil, fmt.Errorf("read rotated queries: %w", err)
	}
	width := c.Dim / c.Heads
	stride := c.KVHeads * width
	headsPerGroup := c.Heads / c.KVHeads
	scale := attentionScale(width)
	attended := make([]float32, c.Dim)
	for head := range c.Heads {
		query := queries[head*width : (head+1)*width]
		group := (head / headsPerGroup) * width
		for position := range scores {
			start := int64(position)*stride + group
			keys := history.keys[start : start+width]
			var dot float32
			for feature, value := range query {
				dot += value * keys[feature]
			}
			scores[position] = dot * scale
		}
		maximum := scores[0]
		for _, score := range scores[1:] {
			maximum = max(maximum, score)
		}
		var sum float32
		for position, score := range scores {
			scores[position] = float32(math.Exp(float64(score - maximum)))
			sum += scores[position]
		}
		output := attended[head*width : (head+1)*width]
		for position, score := range scores {
			start := int64(position)*stride + group
			values := history.values[start : start+width]
			probability := score / sum
			for feature, value := range values {
				output[feature] += probability * value
			}
		}
	}
	return tensor.FromFloat32([]int64{1, c.Dim}, attended)
}
