package llama

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/itsmontoya/neuralnetwork/gpu"
)

// MetalOptions configures opt-in reduced-precision Metal storage.
// Zero values default to gpu.Float32.
type MetalOptions struct {
	// WeightDType selects the device storage dtype for transformer projection
	// weights (Q, K, V, O, gate, up, down, and untied output). Valid values:
	// 0 (defaults to gpu.Float32), gpu.Float32, gpu.Float16, gpu.BFloat16,
	// gpu.Q8_0, and gpu.Q4_0.
	WeightDType gpu.DType

	// KVCacheDType selects the device storage dtype for per-session KV caches.
	// Valid values: 0 (defaults to gpu.Float32), gpu.Float32, and gpu.Float16.
	KVCacheDType gpu.DType
}

func (o MetalOptions) resolve() (resolved MetalOptions, err error) {
	resolved = o
	if resolved.WeightDType == 0 {
		resolved.WeightDType = gpu.Float32
	}
	if resolved.KVCacheDType == 0 {
		resolved.KVCacheDType = gpu.Float32
	}
	switch resolved.WeightDType {
	case gpu.Float32, gpu.Float16, gpu.BFloat16, gpu.Q8_0, gpu.Q4_0:
	default:
		return MetalOptions{}, fmt.Errorf("llama: unsupported Metal weight dtype %v: %w", resolved.WeightDType, gpu.ErrInvalid)
	}
	switch resolved.KVCacheDType {
	case gpu.Float32, gpu.Float16:
	default:
		return MetalOptions{}, fmt.Errorf("llama: unsupported Metal KV cache dtype %v: %w", resolved.KVCacheDType, gpu.ErrInvalid)
	}
	return resolved, nil
}

// metalModel owns a logical device and the model's immutable device tensors.
// The read lock pins the device and its weights through an inference call.
type metalModel struct {
	mu      sync.RWMutex
	device  *gpu.Device
	weights []*gpu.Tensor
	closed  bool

	attention    bool
	weightDType  gpu.DType
	kvCacheDType gpu.DType
	cosine, sine *gpu.Tensor

	// Changes to the session set also hold mu for reading. Closing the model
	// holds mu for writing, which excludes use and registration.
	sessionsMu sync.Mutex
	sessions   map[*metalSession]struct{}
}

func (m *metalModel) weightDTypeFor(name string) gpu.DType {
	if !m.attention || m.weightDType == 0 || m.weightDType == gpu.Float32 {
		return gpu.Float32
	}
	if m.weightDType.IsQuantized() {
		switch name {
		case "attention_norm", "ffn_norm", "final_norm":
			return gpu.Float32
		}
	}
	return m.weightDType
}

func (m *metalModel) effectiveKVCacheDType() gpu.DType {
	if m.kvCacheDType == 0 {
		return gpu.Float32
	}
	return m.kvCacheDType
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

func (m *metalModel) uploadWithDType(shape []int64, values []float32, dtype gpu.DType) (out *gpu.Tensor, err error) {
	if dtype == 0 || dtype == gpu.Float32 {
		return m.upload(shape, values)
	}
	var weight *gpu.Tensor
	if weight, err = m.device.NewTensorWithDType(shape, dtype); err != nil {
		return nil, err
	}
	if err = weight.UploadFloat32As(values); err != nil {
		return nil, errors.Join(err, weight.Close())
	}
	m.weights = append(m.weights, weight)
	return weight, nil
}

func (m *metalModel) uploadBFloat16Bits(shape []int64, bits []uint16) (out *gpu.Tensor, err error) {
	var weight *gpu.Tensor
	if weight, err = m.device.NewTensorWithDType(shape, gpu.BFloat16); err != nil {
		return nil, err
	}
	if err = weight.UploadBits(bits); err != nil {
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
