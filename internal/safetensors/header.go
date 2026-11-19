package safetensors

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"unicode/utf8"
)

func parseHeader(header []byte, payloadSize int64) (map[string]Tensor, map[string]string, error) {
	if len(header) == 0 || header[0] != '{' || !utf8.Valid(header) {
		return nil, nil, fmt.Errorf("expected a UTF-8 JSON header starting with '{'")
	}
	decoder := json.NewDecoder(bytes.NewReader(header))
	decoder.UseNumber()
	tensors := make(map[string]Tensor)
	var metadata map[string]string
	err := readObject(decoder, func(name string) error {
		if name == "__metadata__" {
			metadata = make(map[string]string)
			err := readObject(decoder, func(key string) error {
				value, err := readString(decoder)
				if err != nil {
					return fmt.Errorf("%q: %w", key, err)
				}
				metadata[key] = value
				return nil
			})
			if err != nil {
				return fmt.Errorf("__metadata__: %w", err)
			}
			return nil
		}
		descriptor, err := readTensor(decoder, payloadSize)
		if err != nil {
			return fmt.Errorf("tensor %q: %w", name, err)
		}
		tensors[name] = descriptor
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	if len(bytes.Trim(header[decoder.InputOffset():], " ")) != 0 {
		return nil, nil, fmt.Errorf("unexpected data after the header object")
	}
	return tensors, metadata, nil
}

func readTensor(decoder *json.Decoder, payloadSize int64) (Tensor, error) {
	var dtype DType
	var shape, offsets []int64
	err := readObject(decoder, func(field string) error {
		var err error
		switch field {
		case "dtype":
			var name string
			name, err = readString(decoder)
			dtype = DType(name)
		case "shape":
			shape, err = readIntArray(decoder)
		case "data_offsets":
			offsets, err = readIntArray(decoder)
		default:
			return fmt.Errorf("unknown field %q", field)
		}
		if err != nil {
			return fmt.Errorf("%s: %w", field, err)
		}
		return nil
	})
	if err != nil {
		return Tensor{}, err
	}
	if dtype == "" || shape == nil || len(offsets) != 2 {
		return Tensor{}, fmt.Errorf("required fields are dtype, shape, and two data_offsets")
	}
	if offsets[0] > offsets[1] || offsets[1] > payloadSize {
		return Tensor{}, fmt.Errorf("data_offsets %v are outside the %d-byte payload or reversed", offsets, payloadSize)
	}
	width := int64(dtype.bytes())
	if width == 0 {
		return Tensor{}, fmt.Errorf("unsupported dtype %q", dtype)
	}
	elements := int64(1)
	for _, dim := range shape {
		if dim != 0 && elements > math.MaxInt64/dim {
			return Tensor{}, fmt.Errorf("shape %v element count overflows int64", shape)
		}
		elements *= dim
	}
	if elements > math.MaxInt64/width {
		return Tensor{}, fmt.Errorf("%s byte size for shape %v overflows int64", dtype, shape)
	}
	if elements > math.MaxInt/4 {
		return Tensor{}, fmt.Errorf("FP32 byte size for shape %v overflows int", shape)
	}
	if span, expected := offsets[1]-offsets[0], elements*width; span != expected {
		return Tensor{}, fmt.Errorf("data_offsets span %d bytes, want %d for %s shape %v", span, expected, dtype, shape)
	}
	return Tensor{DType: dtype, Shape: shape, Offsets: [2]int64{offsets[0], offsets[1]}}, nil
}

func readObject(decoder *json.Decoder, readField func(string) error) error {
	if err := readDelimiter(decoder, '{'); err != nil {
		return err
	}
	seen := make(map[string]struct{})
	for decoder.More() {
		key, err := readString(decoder)
		if err != nil {
			return err
		}
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("duplicate key %q", key)
		}
		seen[key] = struct{}{}
		if err := readField(key); err != nil {
			return err
		}
	}
	return readDelimiter(decoder, '}')
}

func readIntArray(decoder *json.Decoder) ([]int64, error) {
	if err := readDelimiter(decoder, '['); err != nil {
		return nil, err
	}
	values := []int64{}
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		number, ok := token.(json.Number)
		if !ok {
			return nil, fmt.Errorf("expected a nonnegative integer at index %d", len(values))
		}
		value, err := number.Int64()
		if err != nil || value < 0 {
			return nil, fmt.Errorf("expected a nonnegative int64 at index %d, got %s", len(values), number)
		}
		values = append(values, value)
	}
	if err := readDelimiter(decoder, ']'); err != nil {
		return nil, err
	}
	return values, nil
}

func readString(decoder *json.Decoder) (string, error) {
	token, err := decoder.Token()
	if err != nil {
		return "", err
	}
	value, ok := token.(string)
	if !ok {
		return "", fmt.Errorf("expected a JSON string")
	}
	return value, nil
}

func readDelimiter(decoder *json.Decoder, delimiter json.Delim) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	if token != delimiter {
		return fmt.Errorf("expected %q", delimiter)
	}
	return nil
}
