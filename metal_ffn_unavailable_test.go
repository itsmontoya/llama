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
