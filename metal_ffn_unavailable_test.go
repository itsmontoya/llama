//go:build !darwin || !arm64 || !cgo || !metal || purego

package llama

import (
	"errors"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/gpu"
)

func TestLoadHFModelMetalUnavailable(t *testing.T) {
	var decoder, _, err = LoadHFModelMetal(t.Context(), t.TempDir(), 8)
	if decoder != nil || !errors.Is(err, gpu.ErrUnavailable) || !strings.Contains(err.Error(), "-tags=metal") {
		t.Fatalf("got decoder %v and error %v, want a Metal availability error", decoder, err)
	}
}

func TestLoadHFModelMetalAttentionUnavailable(t *testing.T) {
	var decoder, _, err = LoadHFModelMetalAttention(t.Context(), t.TempDir(), 8)
	if decoder != nil || !errors.Is(err, gpu.ErrUnavailable) {
		t.Fatalf("got decoder %v and error %v, want a Metal availability error", decoder, err)
	}
}

func TestLoadHFModelMetalAttentionWithOptionsUnavailableAndInvalid(t *testing.T) {
	var decoder, _, err = LoadHFModelMetalAttentionWithOptions(t.Context(), t.TempDir(), 8, MetalOptions{
		WeightDType:  gpu.BFloat16,
		KVCacheDType: gpu.Float16,
	})
	if decoder != nil || !errors.Is(err, gpu.ErrUnavailable) {
		t.Fatalf("got decoder %v and error %v, want a Metal availability error", decoder, err)
	}
	if decoder, _, err = LoadHFModelMetalAttentionWithOptions(t.Context(), t.TempDir(), 8, MetalOptions{
		WeightDType: gpu.DType(99),
	}); decoder != nil || !errors.Is(err, gpu.ErrInvalid) {
		t.Fatalf("invalid weight dtype: got decoder %v, err %v, want ErrInvalid", decoder, err)
	}
	if decoder, _, err = LoadHFModelMetalAttentionWithOptions(t.Context(), t.TempDir(), 8, MetalOptions{
		KVCacheDType: gpu.BFloat16,
	}); decoder != nil || !errors.Is(err, gpu.ErrInvalid) {
		t.Fatalf("invalid KV dtype: got decoder %v, err %v, want ErrInvalid", decoder, err)
	}
}
