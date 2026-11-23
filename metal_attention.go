package llama

import (
	"context"
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/gpu"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

type metalAttentionWeights struct {
	attentionNorm, q, k, v, o, ffnNorm *gpu.Tensor
}

func newMetalAttentionScratch(device *gpu.Device, config Config, rows int64) (out *metalAttentionScratch, err error) {
	var (
		scratch metalAttentionScratch
		hidden  = []int64{rows, config.Dim}
		kv      = []int64{rows, config.KVHeads * (config.Dim / config.Heads)}
		wide    = []int64{rows, config.FFNDim}
	)
	for _, item := range [...]struct {
		dst   **gpu.Tensor
		shape []int64
	}{
		{&scratch.hidden, hidden},
		{&scratch.normalized, hidden},
		{&scratch.q, hidden},
		{&scratch.k, kv},
		{&scratch.v, kv},
		{&scratch.rotatedQ, hidden},
		{&scratch.rotatedK, kv},
		{&scratch.attended, hidden},
		{&scratch.projected, hidden},
		{&scratch.residual, hidden},
		{&scratch.ffn.gate, wide},
		{&scratch.ffn.up, wide},
		{&scratch.ffn.activated, wide},
		{&scratch.ffn.output, hidden},
	} {
		if *item.dst, err = device.NewFloat32(item.shape); err != nil {
			scratch.close()
			return nil, err
		}
	}
	return &scratch, nil
}

// The command orders all reads before later layers overwrite the same scratch.
// The per-layer KV slots survive that reuse, with separate slots per session.
type metalAttentionScratch struct {
	hidden, normalized          *gpu.Tensor
	q, k, v, rotatedQ, rotatedK *gpu.Tensor
	attended, projected         *gpu.Tensor
	residual                    *gpu.Tensor
	ffn                         metalFFNActivations
}

func (s *metalAttentionScratch) close() {
	for _, value := range [...]*gpu.Tensor{
		s.hidden, s.normalized, s.q, s.k, s.v, s.rotatedQ, s.rotatedK,
		s.attended, s.projected, s.residual, s.ffn.gate, s.ffn.up, s.ffn.activated, s.ffn.output,
	} {
		_ = value.Close()
	}
}

func (d *Decoder) forwardMetalPrefix(ctx context.Context, tokens []int64) (out *tensor.Tensor, err error) {
	var m = d.metal
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return nil, fmt.Errorf("llama: Metal attention: %w", gpu.ErrClosed)
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	var caches []*gpu.KVCache
	if caches, err = newMetalCaches(m.device, d.config, int64(len(tokens))); err != nil {
		return nil, err
	}
	defer closeMetalCaches(caches)
	var updates = make([]*gpu.KVAppend, len(caches))
	defer discardMetalAppends(updates)
	return d.metalLogits(ctx, tokens, 0, caches, updates, false)
}

// metalLogits is shared by full-prefix inference, prefill and decode. The caller
// owns the model read lock, private caches and cleanup of each staged append.
func (d *Decoder) metalLogits(ctx context.Context, tokens []int64, position int64, caches []*gpu.KVCache, updates []*gpu.KVAppend, lastOnly bool) (out *tensor.Tensor, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	var rows = int64(len(tokens))
	var scratch *metalAttentionScratch
	if scratch, err = newMetalAttentionScratch(d.metal.device, d.config, rows); err != nil {
		return nil, fmt.Errorf("llama: allocate Metal attention scratch: %w", err)
	}
	defer scratch.close()

	var embedding *tensor.Tensor
	if embedding, err = gather(d.embedding, tokens, 0); err != nil {
		return nil, fmt.Errorf("llama: embedding: %w", err)
	}
	var values []float32
	if values, err = embedding.Float32Values(); err != nil {
		return nil, fmt.Errorf("llama: read embedding: %w", err)
	}
	if err = scratch.hidden.Upload(values); err != nil {
		return nil, fmt.Errorf("llama: upload embedding: %w", err)
	}
	if err = d.runMetalLayers(ctx, scratch, position, caches, updates); err != nil {
		return nil, err
	}

	// The ordinary tensor classifier is the boundary of the resident sequence.
	// DownloadInto copies the entire tensor, even when only the tail is needed.
	if err = scratch.normalized.DownloadInto(values); err != nil {
		return nil, fmt.Errorf("llama: download final hidden rows: %w", err)
	}
	var dim = d.config.Dim
	if lastOnly {
		values = values[(rows-1)*dim:]
		rows = 1
	}
	var hidden *tensor.Tensor
	if hidden, err = tensor.FromFloat32([]int64{rows, dim}, values); err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if out, err = d.classify(hidden, nil); err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func (d *Decoder) runMetalLayers(ctx context.Context, scratch *metalAttentionScratch, position int64, caches []*gpu.KVCache, updates []*gpu.KVAppend) (err error) {
	var scope *gpu.Scope
	if scope, err = d.metal.device.NewScope(); err != nil {
		return err
	}
	defer func() {
		// Finish an outstanding command before the caller discards its appends
		// or releases scratch. A canceled Wait leaves those appends unpublished.
		if closeErr := scope.Close(); closeErr != nil {
			err = errors.Join(err, closeErr)
		}
	}()
	for index, block := range d.blocks {
		if err = ctx.Err(); err != nil {
			return err
		}
		if updates[index], err = d.encodeMetalBlock(scope, scratch, block, caches[index], position); err != nil {
			return fmt.Errorf("llama: layer %d: %w", index, err)
		}
	}
	if err = scope.RMSNorm(scratch.hidden, d.metalNorm, scratch.normalized, d.config.NormEpsilon); err != nil {
		return fmt.Errorf("llama: final norm: %w", err)
	}
	if err = scope.Submit(ctx); err != nil {
		return err
	}
	return scope.Wait(ctx)
}

func (d *Decoder) encodeMetalBlock(scope *gpu.Scope, s *metalAttentionScratch, block blockWeights, cache *gpu.KVCache, position int64) (update *gpu.KVAppend, err error) {
	var (
		w         = block.metalAttention
		m         = d.metal
		epsilon   = d.config.NormEpsilon
		headWidth = d.config.Dim / d.config.Heads
	)
	if err = scope.RMSNorm(s.hidden, w.attentionNorm, s.normalized, epsilon); err != nil {
		return nil, err
	}
	if err = scope.MatMulRightTranspose(s.normalized, w.q, s.q); err != nil {
		return nil, err
	}
	if err = scope.MatMulRightTranspose(s.normalized, w.k, s.k); err != nil {
		return nil, err
	}
	if err = scope.MatMulRightTranspose(s.normalized, w.v, s.v); err != nil {
		return nil, err
	}
	if err = scope.RotatePairs(s.q, m.cosine, m.sine, s.rotatedQ, headWidth, position, 0); err != nil {
		return nil, err
	}
	if err = scope.RotatePairs(s.k, m.cosine, m.sine, s.rotatedK, headWidth, position, 0); err != nil {
		return nil, err
	}
	if update, err = scope.AppendKV(cache, s.rotatedK, s.v, position); err != nil {
		return nil, err
	}
	if err = scope.CausalAttention(s.rotatedQ, cache, s.attended, position); err != nil {
		return update, err
	}
	if err = scope.MatMulRightTranspose(s.attended, w.o, s.projected); err != nil {
		return update, err
	}
	if err = scope.Add(s.hidden, s.projected, s.residual); err != nil {
		return update, err
	}
	if err = scope.RMSNorm(s.residual, w.ffnNorm, s.normalized, epsilon); err != nil {
		return update, err
	}
	if err = s.ffn.encode(scope, s.normalized, block.metal); err != nil {
		return update, err
	}
	return update, scope.Add(s.residual, s.ffn.output, s.hidden)
}

func discardMetalAppends(updates []*gpu.KVAppend) {
	for _, update := range updates {
		_ = update.Close()
	}
}
