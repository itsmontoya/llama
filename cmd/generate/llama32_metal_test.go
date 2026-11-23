//go:build darwin && arm64 && cgo && metal && !purego

package main

import (
	"errors"
	"testing"

	"github.com/itsmontoya/neuralnetwork/gpu"
)

func requireMetalGenerationDevice(t *testing.T) {
	t.Helper()
	device, err := gpu.OpenDefault()
	if errors.Is(err, gpu.ErrUnavailable) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err = device.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestLlama32MetalGeneration(t *testing.T) {
	requireMetalGenerationDevice(t)
	checkLlama32Generation(t, []string{"-ffn-backend", "metal"})
}

func TestLlama32MetalAttentionGeneration(t *testing.T) {
	requireMetalGenerationDevice(t)
	checkLlama32Generation(t, []string{"-ffn-backend", "metal", "-attention-backend", "metal"})
}
