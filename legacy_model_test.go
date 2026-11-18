package llama

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func legacyTestModel(t *testing.T, tied bool) modelFixture {
	t.Helper()
	model, _ := readFixture(t, "tiny_gqa")
	model.Config.NormEpsilon = 1e-5
	model.Config.RopeTheta = 10000
	model.Config.TieEmbeddings = tied
	if !tied {
		embedding := model.Weights.TokenEmbedding
		model.Weights.Output = make([]float32, len(embedding))
		for i := range embedding {
			model.Weights.Output[i] = -embedding[len(embedding)-1-i]
		}
	}
	return model
}

func packLegacyTestModel(t *testing.T, model modelFixture) []byte {
	t.Helper()
	var checkpoint bytes.Buffer
	write := func(values any) {
		t.Helper()
		if err := binary.Write(&checkpoint, binary.LittleEndian, values); err != nil {
			t.Fatal(err)
		}
	}
	config, weights := model.Config, model.Weights
	vocab := int32(config.VocabSize)
	if !config.TieEmbeddings {
		vocab = -vocab
	}
	write([7]int32{int32(config.Dim), int32(config.FFNDim), int32(config.Layers), int32(config.Heads), int32(config.KVHeads), vocab, int32(config.MaxContext)})
	write(weights.TokenEmbedding)
	for group := range 9 {
		for _, layer := range weights.Layers {
			tensors := [...][]float32{layer.AttentionNorm, layer.WQ, layer.WK, layer.WV, layer.WO, layer.FFNNorm, layer.WGate, layer.WDown, layer.WUp}
			write(tensors[group])
		}
	}
	write(weights.FinalNorm)
	// Deliberately distinct values ensure the file's rotary tables are ignored.
	for i := range config.MaxContext * (config.Dim / config.Heads) {
		write(float32(i) + 10)
	}
	if !config.TieEmbeddings {
		write(weights.Output)
	}
	return checkpoint.Bytes()
}

func legacyTestPath(t *testing.T, payload []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "checkpoint.bin")
	if err := os.WriteFile(path, payload, 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadLegacyModel(t *testing.T) {
	for _, test := range []struct {
		name string
		tied bool
	}{
		{"tied", true},
		{"untied", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := legacyTestModel(t, test.tied)
			loaded, err := LoadLegacyModel(legacyTestPath(t, packLegacyTestModel(t, model)))
			if err != nil {
				t.Fatal(err)
			}
			if got := loaded.Config(); got != model.Config {
				t.Fatalf("config = %+v, want %+v", got, model.Config)
			}
			direct, err := NewDecoder(model.Config, model.Weights)
			if err != nil {
				t.Fatal(err)
			}
			tokens := []int64{1, 3, 2, 1}
			got, err := loaded.Forward(tokens)
			if err != nil {
				t.Fatal(err)
			}
			want, err := direct.Forward(tokens)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.Shape(), want.Shape()) || !slices.Equal(tensorValues(t, got), tensorValues(t, want)) {
				t.Fatal("loaded model logits differ from direct construction")
			}
		})
	}
}

func TestLoadLegacyModelRejectsMalformedFiles(t *testing.T) {
	const headerSize = 7 * 4
	valid := packLegacyTestModel(t, legacyTestModel(t, true))
	setWord := func(payload []byte, index int, value uint32) []byte {
		modified := bytes.Clone(payload)
		binary.LittleEndian.PutUint32(modified[index*4:], value)
		return modified
	}
	packHeader := func(header [7]int32) []byte {
		payload, err := binary.Append(nil, binary.LittleEndian, header)
		if err != nil {
			t.Fatal(err)
		}
		return payload
	}
	for _, test := range []struct {
		name    string
		payload []byte
		wantErr string
	}{
		{"short header", valid[:headerSize-1], "read header"},
		{"zero layers", setWord(valid, 2, 0), "invalid header"},
		{"invalid head division", setWord(valid, 4, 3), "invalid header"},
		{"zero vocabulary", setWord(valid, 5, 0), "invalid header"},
		{"odd head width", packHeader([7]int32{6, 8, 1, 2, 1, 7, 8}), "invalid header"},
		{"group size overflow", packHeader([7]int32{1 << 20, 7, 1 << 22, 1, 1, 7, 8}), "size"},
		{"total size overflow", packHeader([7]int32{1 << 20, 7, 1 << 19, 1, 1, 7, 8}), "size"},
		{"minimum signed vocabulary", packHeader([7]int32{2, 1, 1, 1, 1, math.MinInt32, 1}), "size"},
		{"truncated payload", valid[:len(valid)-4], "file size"},
		{"trailing byte", append(bytes.Clone(valid), 0), "file size"},
		{"nonfinite weight", setWord(valid, 7, math.Float32bits(float32(math.Inf(1)))), "nonfinite"},
	} {
		t.Run(test.name, func(t *testing.T) {
			decoder, err := LoadLegacyModel(legacyTestPath(t, test.payload))
			if decoder != nil || err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("got decoder = %v, error = %v, want nil decoder and error containing %q", decoder, err, test.wantErr)
			}
		})
	}
}
