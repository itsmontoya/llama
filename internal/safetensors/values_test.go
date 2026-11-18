package safetensors

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"testing"
)

func TestReadFloat32ExpandsBF16Bits(t *testing.T) {
	cases := []struct {
		bf16 uint16
		fp32 uint32
	}{
		{0x0000, 0x00000000}, // Positive zero.
		{0x8000, 0x80000000}, // Negative zero.
		{0x3f80, 0x3f800000},
		{0xc1c0, 0xc1c00000},
		{0x0001, 0x00010000}, // Positive subnormal.
		{0x807f, 0x807f0000}, // Negative subnormal.
		{0x0080, 0x00800000},
		{0x7f7f, 0x7f7f0000},
		{0x7f80, 0x7f800000}, // Positive infinity.
		{0xff80, 0xff800000}, // Negative infinity.
		{0x7fc1, 0x7fc10000}, // Quiet NaN.
		{0xff81, 0xff810000}, // Signaling NaN.
	}
	var payload []byte
	for _, test := range cases {
		payload = binary.LittleEndian.AppendUint16(payload, test.bf16)
	}
	header := fmt.Sprintf(`{"values":{"dtype":"BF16","shape":[%d],"data_offsets":[0,%d]}}`, len(cases), len(payload))
	archive := openArchive(t, writeArchive(t, header, payload))
	values, err := archive.ReadFloat32("values")
	if err != nil || len(values) != len(cases) {
		t.Fatalf("got %d values and error %v, want %d values", len(values), err, len(cases))
	}
	for i, test := range cases {
		if bits := math.Float32bits(values[i]); bits != test.fp32 {
			t.Errorf("BF16 0x%04x expanded to 0x%08x, want 0x%08x", test.bf16, bits, test.fp32)
		}
	}
}

func TestReadFloat32ReadsNamedRangesAcrossChunks(t *testing.T) {
	const chunkElements = readBufferBytes / 4
	want := make([]uint32, 2*chunkElements+7)
	for i := range want {
		want[i] = 0x3f800000 + uint32(i)
	}
	want[chunkElements-1] = 0x80000000
	want[chunkElements] = 0x7fc12345
	payload := binary.LittleEndian.AppendUint16(nil, 0x4000)
	for _, bits := range want {
		payload = binary.LittleEndian.AppendUint32(payload, bits)
	}
	header := fmt.Sprintf(`{
		"scalar":{"dtype":"BF16","shape":[],"data_offsets":[0,2]},
		"large":{"dtype":"F32","shape":[%d],"data_offsets":[2,%d]},
		"empty":{"dtype":"F32","shape":[0],"data_offsets":[2,2]}
	}`, len(want), len(payload))
	archive := openArchive(t, writeArchive(t, header, payload))
	// A caller can edit its descriptors without affecting the validated read.
	invent := archive.Tensors()
	invent["large"].Shape[0] = 0
	delete(invent, "scalar")
	values, err := archive.ReadFloat32("large")
	if err != nil || len(values) != len(want) {
		t.Fatalf("got %d values and error %v, want %d values", len(values), err, len(want))
	}
	for i, expected := range want {
		if bits := math.Float32bits(values[i]); bits != expected {
			t.Fatalf("element %d: got 0x%08x, want 0x%08x", i, bits, expected)
		}
	}
	values[0] = -1
	second, err := archive.ReadFloat32("large")
	if err != nil || len(second) != len(want) || math.Float32bits(second[0]) != want[0] {
		t.Fatalf("edited result changed a later read, error %v", err)
	}
	scalar, err := archive.ReadFloat32("scalar")
	if err != nil || len(scalar) != 1 || scalar[0] != 2 {
		t.Fatalf("got scalar %v and error %v, want [2]", scalar, err)
	}
	empty, err := archive.ReadFloat32("empty")
	if err != nil || len(empty) != 0 {
		t.Fatalf("got empty tensor %v and error %v", empty, err)
	}
	if missing, err := archive.ReadFloat32("absent"); missing != nil || err == nil || !strings.Contains(err.Error(), `unknown tensor "absent"`) {
		t.Fatalf("got %v, %v for missing tensor", missing, err)
	}
}

func TestReadFloat32RejectsShortReads(t *testing.T) {
	const payloadSize = readBufferBytes + 4
	header := fmt.Sprintf(`{"values":{"dtype":"BF16","shape":[%d],"data_offsets":[0,%d]}}`, payloadSize/2, payloadSize)
	for _, test := range []struct {
		name string
		size int64
		want error
	}{
		{"missing payload", 0, io.EOF},
		{"late partial chunk", payloadSize - 1, io.ErrUnexpectedEOF},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := writeArchive(t, header, make([]byte, payloadSize))
			archive := openArchive(t, path)
			if err := os.Truncate(path, int64(8+len(header))+test.size); err != nil {
				t.Fatal(err)
			}
			values, err := archive.ReadFloat32("values")
			if values != nil || !errors.Is(err, test.want) || !strings.Contains(err.Error(), `tensor "values"`) {
				t.Fatalf("got %d values, %v, want tensor read error %v", len(values), err, test.want)
			}
		})
	}
}
