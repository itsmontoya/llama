// Package safetensors validates a single local safetensors file.
package safetensors

import (
	"cmp"
	"encoding/binary"
	"fmt"
	"io"
	"maps"
	"os"
	"slices"
)

// The safetensors format limits the JSON header to 100 MB.
const maxHeaderBytes = 100_000_000

// DType is the on-disk element type of a tensor.
type DType string

const (
	BF16 DType = "BF16"
	F32  DType = "F32"
)

func (d DType) bytes() int {
	switch d {
	case BF16:
		return 2
	case F32:
		return 4
	default:
		return 0
	}
}

// Tensor describes a contiguous, row-major tensor. Offsets are relative to
// the start of the payload and the end offset is exclusive.
type Tensor struct {
	DType   DType
	Shape   []int64
	Offsets [2]int64
}

// File is an open checkpoint with a validated tensor inventory.
type File struct {
	file      *os.File
	payloadAt int64
	tensors   map[string]Tensor
	metadata  map[string]string
}

// Open opens a BF16 or F32 safetensors file and validates the entire header
// and payload layout before returning. The caller must Close the file.
func Open(path string) (*File, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: %w", err)
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("safetensors: %q: expected a regular file", path)
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: %w", err)
	}
	archive, err := openFile(file)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("safetensors: %q: %w", path, err)
	}
	return archive, nil
}

func openFile(file *os.File) (*File, error) {
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("expected a regular file")
	}
	var prefix [8]byte
	if _, err := io.ReadFull(file, prefix[:]); err != nil {
		return nil, fmt.Errorf("read header length: %w", err)
	}
	length := binary.LittleEndian.Uint64(prefix[:])
	if length == 0 || length > maxHeaderBytes {
		return nil, fmt.Errorf("header length %d is outside [1,%d]", length, maxHeaderBytes)
	}
	payloadAt := int64(len(prefix)) + int64(length)
	if payloadAt > info.Size() {
		return nil, fmt.Errorf("header length %d exceeds file size %d", length, info.Size())
	}
	header := make([]byte, int(length))
	if _, err := io.ReadFull(file, header); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	payloadSize := info.Size() - payloadAt
	tensors, metadata, err := parseHeader(header, payloadSize)
	if err != nil {
		return nil, fmt.Errorf("header: %w", err)
	}
	if err := validateCoverage(tensors, payloadSize); err != nil {
		return nil, err
	}
	return &File{file: file, payloadAt: payloadAt, tensors: tensors, metadata: metadata}, nil
}

func validateCoverage(tensors map[string]Tensor, payloadSize int64) error {
	names := slices.Collect(maps.Keys(tensors))
	slices.SortFunc(names, func(a, b string) int {
		left, right := tensors[a].Offsets, tensors[b].Offsets
		if order := cmp.Compare(left[0], right[0]); order != 0 {
			return order
		}
		if order := cmp.Compare(left[1], right[1]); order != 0 {
			return order
		}
		return cmp.Compare(a, b)
	})
	var next int64
	for _, name := range names {
		offsets := tensors[name].Offsets
		switch {
		case offsets[0] < next:
			return fmt.Errorf("tensor %q starts at %d and overlaps the preceding range ending at %d", name, offsets[0], next)
		case offsets[0] > next:
			return fmt.Errorf("payload has a hole from %d to %d before tensor %q", next, offsets[0], name)
		}
		next = offsets[1]
	}
	if next != payloadSize {
		return fmt.Errorf("tensors cover %d bytes of the %d-byte payload", next, payloadSize)
	}
	return nil
}

// Tensors returns a copy of the validated descriptors and their shapes.
func (f *File) Tensors() map[string]Tensor {
	tensors := make(map[string]Tensor, len(f.tensors))
	for name, descriptor := range f.tensors {
		descriptor.Shape = slices.Clone(descriptor.Shape)
		tensors[name] = descriptor
	}
	return tensors
}

// Metadata returns a copy of the optional string metadata.
func (f *File) Metadata() map[string]string {
	return maps.Clone(f.metadata)
}

// Close closes the checkpoint file.
func (f *File) Close() error {
	return f.file.Close()
}
