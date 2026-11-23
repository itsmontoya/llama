package llama

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/itsmontoya/neuralnetwork/gpu"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

// metalFFNModel owns a logical device and its immutable projection weights.
// The read lock pins the model through one call's use and release of scratch.
type metalFFNModel struct {
	mu      sync.RWMutex
	device  *gpu.Device
	weights []*gpu.Tensor
	closed  bool
}

func (m *metalFFNModel) upload(shape []int64, values []float32) (out *gpu.Tensor, err error) {
	var weight *gpu.Tensor
	if weight, err = m.device.NewFloat32(shape); err != nil {
		return nil, err
	}
	if err = weight.Upload(values); err != nil {
		return nil, errors.Join(err, weight.Close())
	}
	m.weights = append(m.weights, weight)
	return weight, nil
}

func (m *metalFFNModel) begin(ctx context.Context, config Config, rows int64) (scratch *metalFFNScratch, err error) {
	if err = ctx.Err(); err != nil || m == nil {
		return nil, err
	}
	m.mu.RLock()
	if m.closed {
		m.mu.RUnlock()
		return nil, fmt.Errorf("llama: Metal FFN: %w", gpu.ErrClosed)
	}
	if scratch, err = newMetalFFNScratch(m, rows, config.Dim, config.FFNDim); err != nil {
		m.mu.RUnlock()
		return nil, fmt.Errorf("llama: allocate Metal FFN scratch: %w", err)
	}
	return scratch, nil
}

func (m *metalFFNModel) close() (err error) {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	for _, weight := range m.weights {
		err = errors.Join(err, weight.Close())
	}
	m.weights = nil
	return errors.Join(err, m.device.Close())
}

type metalFFNWeights struct {
	gate, up, down *gpu.Tensor
}

func newMetalFFNScratch(model *metalFFNModel, rows, dim, ffnDim int64) (out *metalFFNScratch, err error) {
	var scratch metalFFNScratch
	scratch.model = model
	scratch.shape = []int64{rows, dim}
	var wide = []int64{rows, ffnDim}
	for _, item := range [...]struct {
		dst   **gpu.Tensor
		shape []int64
	}{
		{&scratch.input, scratch.shape},
		{&scratch.gate, wide},
		{&scratch.up, wide},
		{&scratch.activated, wide},
		{&scratch.output, scratch.shape},
	} {
		if *item.dst, err = model.device.NewFloat32(item.shape); err != nil {
			scratch.closeTensors()
			return nil, err
		}
	}
	scratch.hostOutput = make([]float32, rows*dim)
	return &scratch, nil
}

// metalFFNScratch belongs to one prefix or decode call and is reused by that
// call's layers. Releasing it bounds GPU retention independently of KV capacity.
type metalFFNScratch struct {
	model                              *metalFFNModel
	input, gate, up, activated, output *gpu.Tensor
	shape                              []int64
	hostOutput                         []float32
}

func (s *metalFFNScratch) close() {
	if s == nil {
		return
	}
	s.closeTensors()
	s.model.mu.RUnlock()
}

func (s *metalFFNScratch) closeTensors() {
	// gpu.Tensor.Close is idempotent and has no failure path in v0.11.0.
	for _, value := range [...]*gpu.Tensor{s.input, s.gate, s.up, s.activated, s.output} {
		_ = value.Close()
	}
}

func (s *metalFFNScratch) run(ctx context.Context, input *tensor.Tensor, weights metalFFNWeights, trace map[string]*tensor.Tensor, prefix string) (out *tensor.Tensor, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	var values []float32
	if values, err = input.Float32Values(); err != nil {
		return nil, err
	}
	if err = s.input.Upload(values); err != nil {
		return nil, err
	}
	if err = s.execute(ctx, weights); err != nil {
		return nil, err
	}
	if err = s.output.DownloadInto(s.hostOutput); err != nil {
		return nil, err
	}
	if trace != nil {
		if err = s.recordTrace(trace, prefix); err != nil {
			return nil, err
		}
	}
	return tensor.FromFloat32(s.shape, s.hostOutput)
}

func (s *metalFFNScratch) execute(ctx context.Context, weights metalFFNWeights) (err error) {
	var scope *gpu.Scope
	if scope, err = s.model.device.NewScope(); err != nil {
		return err
	}
	defer func() {
		// Drain a submitted command before the caller can release or reuse scratch.
		if closeErr := scope.Close(); closeErr != nil {
			err = errors.Join(err, closeErr)
		}
	}()
	if err = scope.MatMulRightTranspose(s.input, weights.gate, s.gate); err != nil {
		return err
	}
	if err = scope.MatMulRightTranspose(s.input, weights.up, s.up); err != nil {
		return err
	}
	if err = scope.SwiGLU(s.gate, s.up, s.activated); err != nil {
		return err
	}
	if err = scope.MatMulRightTranspose(s.activated, weights.down, s.output); err != nil {
		return err
	}
	if err = scope.Submit(ctx); err != nil {
		return err
	}
	return scope.Wait(ctx)
}

func (s *metalFFNScratch) recordTrace(trace map[string]*tensor.Tensor, prefix string) (err error) {
	var (
		shape  = s.gate.Shape()
		values = make([]float32, shape[0]*shape[1])
	)
	for _, item := range [...]struct {
		name  string
		value *gpu.Tensor
	}{
		{"gate", s.gate},
		{"up", s.up},
		{"swiglu", s.activated},
	} {
		if err = item.value.DownloadInto(values); err != nil {
			return err
		}
		var snapshot *tensor.Tensor
		if snapshot, err = tensor.FromFloat32(shape, values); err != nil {
			return err
		}
		trace[prefix+item.name] = snapshot
	}
	return nil
}
