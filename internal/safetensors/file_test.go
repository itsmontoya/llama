package safetensors

import (
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func writeArchive(t *testing.T, header string, payload []byte) string {
	t.Helper()
	data := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	data = append(data, header...)
	data = append(data, payload...)
	return writeBytes(t, data)
}

func writeBytes(t *testing.T, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.safetensors")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func openArchive(t *testing.T, path string) *File {
	t.Helper()
	archive, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := archive.Close(); err != nil {
			t.Error(err)
		}
	})
	return archive
}

func TestOpenInventory(t *testing.T) {
	path := writeArchive(t, `{
		"b": {"dtype":"F32","shape":[],"data_offsets":[2,6]},
		"a": {"dtype":"BF16","shape":[1],"data_offsets":[0,2]},
		"empty": {"dtype":"BF16","shape":[0,8],"data_offsets":[2,2]},
		"also_empty": {"dtype":"F32","shape":[0],"data_offsets":[2,2]},
		"__metadata__": {"format":"pt"}
	}   `, make([]byte, 6))
	archive := openArchive(t, path)
	got := archive.Tensors()
	want := map[string]Tensor{
		"a":          {DType: BF16, Shape: []int64{1}, Offsets: [2]int64{0, 2}},
		"b":          {DType: F32, Shape: []int64{}, Offsets: [2]int64{2, 6}},
		"empty":      {DType: BF16, Shape: []int64{0, 8}, Offsets: [2]int64{2, 2}},
		"also_empty": {DType: F32, Shape: []int64{0}, Offsets: [2]int64{2, 2}},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d descriptors, want %d", len(got), len(want))
	}
	for name, descriptor := range want {
		actual := got[name]
		if actual.DType != descriptor.DType || actual.Offsets != descriptor.Offsets || !slices.Equal(actual.Shape, descriptor.Shape) {
			t.Fatalf("tensor %q: got %+v, want %+v", name, actual, descriptor)
		}
	}
	got["a"].Shape[0] = 9
	delete(got, "b")
	if actual := archive.Tensors(); actual["a"].Shape[0] != 1 || len(actual) != len(want) {
		t.Fatalf("descriptor edits changed the archive: %+v", actual)
	}
	metadata := archive.Metadata()
	if len(metadata) != 1 || metadata["format"] != "pt" {
		t.Fatalf("unexpected metadata: %v", metadata)
	}
	metadata["format"] = "changed"
	if got := archive.Metadata()["format"]; got != "pt" {
		t.Fatalf("metadata edits changed the archive format: %q", got)
	}
}

func TestOpenRejectsInvalidHeaderLength(t *testing.T) {
	for _, test := range []struct {
		name string
		data []byte
		want string
	}{
		{"short prefix", []byte{0, 0, 0}, io.ErrUnexpectedEOF.Error()},
		{"zero", binary.LittleEndian.AppendUint64(nil, 0), "outside [1,100000000]"},
		{"limit", binary.LittleEndian.AppendUint64(nil, maxHeaderBytes+1), "outside [1,100000000]"},
		{"unsigned overflow", binary.LittleEndian.AppendUint64(nil, ^uint64(0)), "outside [1,100000000]"},
		{"truncated header", binary.LittleEndian.AppendUint64(nil, 2), "exceeds file size"},
	} {
		t.Run(test.name, func(t *testing.T) {
			assertOpenError(t, writeBytes(t, test.data), test.want)
		})
	}
}

func TestOpenRejectsDirectory(t *testing.T) {
	assertOpenError(t, t.TempDir(), "expected a regular file")
}

func TestOpenRejectsMalformedHeader(t *testing.T) {
	for _, test := range []struct {
		name    string
		header  string
		payload int
		want    string
	}{
		{"invalid JSON", `{"x":]}`, 0, `tensor "x"`},
		{"trailing object", `{} {}`, 0, "unexpected data after the header object"},
		{"invalid start", ` {}`, 0, "starting with '{'"},
		{"invalid UTF-8", "{\"\xff\":{}}", 0, "UTF-8"},
		{"duplicate tensor", `{"x":{"dtype":"BF16","shape":[0],"data_offsets":[0,0]},"\u0078":{}}`, 0, `duplicate key "x"`},
		{"duplicate metadata", `{"__metadata__":{},"__metadata__":{}}`, 0, `duplicate key "__metadata__"`},
		{"duplicate metadata key", `{"__metadata__":{"format":"pt","format":"other"}}`, 0, `duplicate key "format"`},
		{"metadata type", `{"__metadata__":[]}`, 0, "__metadata__"},
		{"metadata value type", `{"__metadata__":{"format":null}}`, 0, "expected a JSON string"},
		{"duplicate tensor field", `{"x":{"dtype":"BF16","dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 4, `tensor "x": duplicate key "dtype"`},
		{"missing dtype", `{"x":{"shape":[1],"data_offsets":[0,2]}}`, 2, "required fields are dtype"},
		{"missing shape", `{"x":{"dtype":"BF16","data_offsets":[0,2]}}`, 2, "required fields are dtype, shape"},
		{"missing offsets", `{"x":{"dtype":"BF16","shape":[1]}}`, 2, "two data_offsets"},
		{"three offsets", `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2,2]}}`, 2, "two data_offsets"},
		{"null dimension", `{"x":{"dtype":"BF16","shape":[null],"data_offsets":[0,0]}}`, 0, "shape: expected a nonnegative integer"},
		{"negative dimension", `{"x":{"dtype":"BF16","shape":[-1],"data_offsets":[0,0]}}`, 0, "shape: expected a nonnegative int64"},
		{"quoted offset", `{"x":{"dtype":"BF16","shape":[1],"data_offsets":["0",2]}}`, 2, "data_offsets: expected a nonnegative integer"},
		{"overflowing offset", `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,9223372036854775808]}}`, 2, "data_offsets: expected a nonnegative int64"},
		{"reversed range", `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[2,0]}}`, 2, "outside the 2-byte payload or reversed"},
		{"end beyond file", `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}`, 1, "outside the 1-byte payload"},
		{"unsupported dtype", `{"x":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}`, 2, `tensor "x": unsupported dtype "F16"`},
		{"element count overflow", `{"x":{"dtype":"BF16","shape":[9223372036854775807,2],"data_offsets":[0,0]}}`, 0, "element count overflows"},
		{"source byte count overflow", `{"x":{"dtype":"BF16","shape":[9223372036854775807],"data_offsets":[0,0]}}`, 0, "BF16 byte size"},
		{"FP32 byte count overflow", `{"x":{"dtype":"BF16","shape":[2305843009213693952],"data_offsets":[0,0]}}`, 0, "FP32 byte size"},
		{"wrong span", `{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,2]}}`, 2, "span 2 bytes, want 4"},
		{"initial hole", `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}`, 4, "hole from 0 to 2"},
		{"internal hole", `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"y":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}}`, 6, "hole from 2 to 4"},
		{"overlap", `{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"y":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}`, 4, `tensor "y" starts at 2 and overlaps`},
		{"uncovered end", `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}`, 4, "tensors cover 2 bytes of the 4-byte payload"},
	} {
		t.Run(test.name, func(t *testing.T) {
			assertOpenError(t, writeArchive(t, test.header, make([]byte, test.payload)), test.want)
		})
	}
}

func assertOpenError(t *testing.T, path, want string) {
	t.Helper()
	archive, err := Open(path)
	if archive != nil {
		archive.Close()
	}
	if archive != nil || err == nil || !strings.Contains(err.Error(), want) || !strings.Contains(err.Error(), path) {
		t.Fatalf("Open(%q) returned %v, %v, want a file error containing %q", path, archive, err, want)
	}
}
