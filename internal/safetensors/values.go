package safetensors

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

const readBufferBytes = 64 * 1024

// ReadFloat32 reads one named tensor as a caller-owned row-major slice.
// The checkpoint must remain open for the duration of the read.
func (f *File) ReadFloat32(name string) ([]float32, error) {
	descriptor, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: %q: unknown tensor %q", f.file.Name(), name)
	}
	width := descriptor.DType.bytes()
	start, end := descriptor.Offsets[0], descriptor.Offsets[1]
	span := end - start
	// Open checked the range, its source byte size, and the FP32 allocation size.
	values := make([]float32, int(span/int64(width)))
	reader := io.NewSectionReader(f.file, f.payloadAt+start, span)
	var buffer [readBufferBytes]byte
	for offset := 0; offset < len(values); {
		count := min(len(values)-offset, len(buffer)/width)
		chunk := buffer[:count*width]
		if _, err := io.ReadFull(reader, chunk); err != nil {
			return nil, fmt.Errorf("safetensors: %q: read tensor %q at element %d: %w", f.file.Name(), name, offset, err)
		}
		out := values[offset : offset+count]
		switch descriptor.DType {
		case BF16:
			for i := range out {
				bits := binary.LittleEndian.Uint16(chunk[i*2:])
				out[i] = math.Float32frombits(uint32(bits) << 16)
			}
		case F32:
			for i := range out {
				bits := binary.LittleEndian.Uint32(chunk[i*4:])
				out[i] = math.Float32frombits(bits)
			}
		}
		offset += count
	}
	return values, nil
}

// ReadBFloat16Bits reads one named BF16 tensor directly as caller-owned
// little-endian uint16 bit patterns without FP32 upcast. Returns an error if
// the tensor does not exist or is not stored as BF16.
func (f *File) ReadBFloat16Bits(name string) ([]uint16, error) {
	descriptor, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: %q: unknown tensor %q", f.file.Name(), name)
	}
	if descriptor.DType != BF16 {
		return nil, fmt.Errorf("safetensors: %q: tensor %q has dtype %s, want BF16", f.file.Name(), name, descriptor.DType)
	}
	const width = 2
	start, end := descriptor.Offsets[0], descriptor.Offsets[1]
	span := end - start
	values := make([]uint16, int(span/width))
	reader := io.NewSectionReader(f.file, f.payloadAt+start, span)
	var buffer [readBufferBytes]byte
	for offset := 0; offset < len(values); {
		count := min(len(values)-offset, len(buffer)/width)
		chunk := buffer[:count*width]
		if _, err := io.ReadFull(reader, chunk); err != nil {
			return nil, fmt.Errorf("safetensors: %q: read tensor %q at element %d: %w", f.file.Name(), name, offset, err)
		}
		out := values[offset : offset+count]
		for i := range out {
			out[i] = binary.LittleEndian.Uint16(chunk[i*2:])
		}
		offset += count
	}
	return values, nil
}
