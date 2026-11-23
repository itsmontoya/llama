package llama

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/itsmontoya/neuralnetwork/gpu"
)

// metalModel owns a logical device and the model's immutable device tensors.
// The read lock pins the device and its weights through an inference call.
type metalModel struct {
	mu      sync.RWMutex
	device  *gpu.Device
	weights []*gpu.Tensor
	closed  bool

	attention    bool
	cosine, sine *gpu.Tensor

	// Changes to the session set also hold mu for reading. Closing the model
	// holds mu for writing, which excludes use and registration.
	sessionsMu sync.Mutex
	sessions   map[*metalSession]struct{}
}

func (m *metalModel) upload(shape []int64, values []float32) (out *gpu.Tensor, err error) {
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

func (m *metalModel) close() (err error) {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	for session := range m.sessions {
		session.close()
	}
	m.sessions = nil
	for _, weight := range m.weights {
		err = errors.Join(err, weight.Close())
	}
	m.weights = nil
	return errors.Join(err, m.device.Close())
}

func (m *metalModel) loadRotary(ctx context.Context, config Config) (err error) {
	if err = ctx.Err(); err != nil {
		return err
	}
	var cosine, sine []float32
	if cosine, sine, err = rotaryCoefficients(0, config.MaxContext, config, true); err != nil {
		return err
	}
	var shape = []int64{config.MaxContext, config.Dim / config.Heads / 2}
	if err = ctx.Err(); err != nil {
		return err
	}
	if m.cosine, err = m.upload(shape, cosine); err != nil {
		return fmt.Errorf("llama: upload Metal RoPE cosine: %w", err)
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	if m.sine, err = m.upload(shape, sine); err != nil {
		return fmt.Errorf("llama: upload Metal RoPE sine: %w", err)
	}
	return nil
}
