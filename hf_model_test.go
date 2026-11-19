package llama

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/itsmontoya/llama/internal/safetensors"
)

type hfFixtureTensor struct {
	name   string
	shape  []int64
	values []float32
}

func makeHFCheckpointFixture(t *testing.T, dir string, tied bool) (Config, Weights, []hfFixtureTensor) {
	t.Helper()
	config := Config{
		Dim: 24, FFNDim: 19, Layers: 2, Heads: 4, KVHeads: 2, VocabSize: 11, MaxContext: 8,
		NormEpsilon: 1e-5, RopeTheta: 10000, TieEmbeddings: tied,
	}
	writeHFTestJSON(t, dir, "config.json", map[string]any{
		"model_type": "llama", "hidden_size": config.Dim, "intermediate_size": config.FFNDim,
		"num_hidden_layers": config.Layers, "num_attention_heads": config.Heads, "num_key_value_heads": config.KVHeads,
		"vocab_size": config.VocabSize, "max_position_embeddings": 16, "rms_norm_eps": config.NormEpsilon,
		"tie_word_embeddings": tied,
	})
	writeHFTestJSON(t, dir, "generation_config.json", map[string]any{"bos_token_id": 1, "eos_token_id": []int{2, 3}})
	var entries []hfFixtureTensor
	add := func(name string, shape []int64, splitRotary bool) []float32 {
		count := int64(1)
		for _, dim := range shape {
			count *= dim
		}
		logical := make([]float32, count)
		// Dyadic weights remain exact in either fixture dtype. Each role, row,
		// column, and layer has a distinct pattern within its matrix.
		for i := range logical {
			logical[i] = float32((i*19+len(entries)*11)%103-51) / 128
			if len(shape) == 1 {
				logical[i] += 1
			}
		}
		onDisk := slices.Clone(logical)
		if splitRotary {
			// For a six-wide head the HF converter writes even rows, then odd
			// rows. This permutation is not its own inverse.
			for head := int64(0); head < shape[0]/6; head++ {
				for index, logicalRow := range []int64{0, 2, 4, 1, 3, 5} {
					start := (head*6 + logicalRow) * shape[1]
					destination := (head*6 + int64(index)) * shape[1]
					copy(onDisk[destination:destination+shape[1]], logical[start:start+shape[1]])
				}
			}
		}
		entries = append(entries, hfFixtureTensor{name, shape, onDisk})
		return logical
	}
	dim, ffn := config.Dim, config.FFNDim
	norm := []int64{dim}
	embed := []int64{config.VocabSize, dim}
	weights := Weights{
		TokenEmbedding: add("model.embed_tokens.weight", embed, false),
		FinalNorm:      add("model.norm.weight", norm, false),
		Layers:         make([]LayerWeights, config.Layers),
	}
	if !tied {
		weights.Output = add("lm_head.weight", embed, false)
	}
	for layer := range weights.Layers {
		prefix := "model.layers." + strconv.Itoa(layer) + "."
		weights.Layers[layer] = LayerWeights{
			AttentionNorm: add(prefix+"input_layernorm.weight", norm, false),
			WQ:            add(prefix+"self_attn.q_proj.weight", []int64{24, dim}, true),
			WK:            add(prefix+"self_attn.k_proj.weight", []int64{12, dim}, true),
			WV:            add(prefix+"self_attn.v_proj.weight", []int64{12, dim}, false),
			WO:            add(prefix+"self_attn.o_proj.weight", []int64{dim, 24}, false),
			FFNNorm:       add(prefix+"post_attention_layernorm.weight", norm, false),
			WGate:         add(prefix+"mlp.gate_proj.weight", []int64{ffn, dim}, false),
			WUp:           add(prefix+"mlp.up_proj.weight", []int64{ffn, dim}, false),
			WDown:         add(prefix+"mlp.down_proj.weight", []int64{dim, ffn}, false),
		}
	}
	return config, weights, entries
}

func writeHFFixtureTensors(t *testing.T, dir string, dtype safetensors.DType, entries []hfFixtureTensor) string {
	t.Helper()
	header := make(map[string]map[string]any, len(entries))
	var payload []byte
	for _, entry := range entries {
		start := len(payload)
		for _, value := range entry.values {
			bits := math.Float32bits(value)
			if dtype == safetensors.BF16 {
				payload = binary.LittleEndian.AppendUint16(payload, uint16(bits>>16))
			} else {
				payload = binary.LittleEndian.AppendUint32(payload, bits)
			}
		}
		header[entry.name] = map[string]any{
			"dtype": dtype, "shape": entry.shape, "data_offsets": []int{start, len(payload)},
		}
	}
	encoded, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	contents := binary.LittleEndian.AppendUint64(nil, uint64(len(encoded)))
	contents = append(contents, encoded...)
	contents = append(contents, payload...)
	path := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(path, contents, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadHFModelMultilayerGQA(t *testing.T) {
	for _, test := range []struct {
		name  string
		dtype safetensors.DType
		tied  bool
	}{
		{"tied BF16", safetensors.BF16, true},
		{"untied F32", safetensors.F32, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			config, weights, entries := makeHFCheckpointFixture(t, dir, test.tied)
			writeHFFixtureTensors(t, dir, test.dtype, entries)
			got, generation, err := LoadHFModel(dir, config.MaxContext)
			if err != nil {
				t.Fatal(err)
			}
			if got.Config() != config || generation.BOSID != 1 || !slices.Equal(generation.StopIDs, []int64{2, 3}) {
				t.Fatalf("unexpected config %+v and generation settings %+v", got.Config(), generation)
			}
			if (got.output == got.embedding) != test.tied {
				t.Fatal("checkpoint classifier ownership disagrees with tie_word_embeddings")
			}
			for layer, block := range got.blocks {
				want := weights.Layers[layer]
				if !slices.Equal(tensorValues(t, block.wq), want.WQ) || !slices.Equal(tensorValues(t, block.wk), want.WK) {
					t.Fatalf("layer %d: incorrect Q/K rows after rotary layout conversion", layer)
				}
			}
			want, err := NewDecoder(config, weights)
			if err != nil {
				t.Fatal(err)
			}
			prefix := []int64{1, 10, 4, 7}
			gotLogits, err := got.Forward(prefix)
			if err != nil {
				t.Fatal(err)
			}
			wantLogits, err := want.Forward(prefix)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(tensorValues(t, gotLogits), tensorValues(t, wantLogits)) {
				t.Fatal("checkpoint logits differ from direct construction")
			}
		})
	}
}

func TestLoadHFModelRejectsInvalidInventory(t *testing.T) {
	for _, test := range []struct {
		name   string
		tied   bool
		modify func([]hfFixtureTensor) []hfFixtureTensor
		want   string
	}{
		{"missing block tensor", true, func(entries []hfFixtureTensor) []hfFixtureTensor {
			return entries[:len(entries)-1]
		}, `missing tensor "model.layers.1.mlp.down_proj.weight"`},
		{"missing classifier", false, func(entries []hfFixtureTensor) []hfFixtureTensor {
			return slices.Delete(entries, 2, 3)
		}, `missing tensor "lm_head.weight"`},
		{"wrong matrix shape", true, func(entries []hfFixtureTensor) []hfFixtureTensor {
			entries[0].shape = []int64{24, 11}
			return entries
		}, `"model.embed_tokens.weight" has shape [24 11], want [11 24]`},
		{"unsupported tensor", true, func(entries []hfFixtureTensor) []hfFixtureTensor {
			return append(entries, hfFixtureTensor{"model.layers.0.self_attn.q_proj.bias", []int64{1}, []float32{1}})
		}, `unsupported checkpoint tensors: ["model.layers.0.self_attn.q_proj.bias"]`},
		{"redundant tied classifier", true, func(entries []hfFixtureTensor) []hfFixtureTensor {
			classifier := entries[0]
			classifier.name = "lm_head.weight"
			return append(entries, classifier)
		}, "tie_word_embeddings requires lm_head.weight to be omitted"},
		{"nonfinite block", true, func(entries []hfFixtureTensor) []hfFixtureTensor {
			entries[len(entries)-1].values[0] = float32(math.Inf(1))
			return entries
		}, "model.layers.1.mlp.down_proj.weight contains nonfinite value at 0"},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			config, _, entries := makeHFCheckpointFixture(t, dir, test.tied)
			writeHFFixtureTensors(t, dir, safetensors.F32, test.modify(entries))
			decoder, generation, err := LoadHFModel(dir, config.MaxContext)
			if decoder != nil || generation.StopIDs != nil || err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("got decoder=%p, generation=%+v, err=%v; want %q", decoder, generation, err, test.want)
			}
		})
	}
}

func TestLoadHFModelRejectsShardOnlyDirectory(t *testing.T) {
	for _, filename := range []string{"model.safetensors.index.json", "model-00001-of-00002.safetensors"} {
		t.Run(filename, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, filename), nil, 0o600); err != nil {
				t.Fatal(err)
			}
			const want = "sharded safetensors checkpoints are unsupported (expected model.safetensors)"
			if decoder, _, err := LoadHFModel(dir, 8); decoder != nil || err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("got decoder=%p, err=%v; want %q", decoder, err, want)
			}
		})
	}
}

func TestLoadHFModelBoundsLayerAllocation(t *testing.T) {
	dir := t.TempDir()
	config, _, entries := makeHFCheckpointFixture(t, dir, true)
	writeHFFixtureTensors(t, dir, safetensors.BF16, entries)
	var model map[string]json.RawMessage
	if err := readHFJSON(filepath.Join(dir, "config.json"), &model); err != nil {
		t.Fatal(err)
	}
	model["num_hidden_layers"] = json.RawMessage(strconv.FormatInt(math.MaxInt64, 10))
	writeHFTestJSON(t, dir, "config.json", model)
	if decoder, _, err := LoadHFModel(dir, config.MaxContext); decoder != nil || err == nil || !strings.Contains(err.Error(), "only 20 tensor descriptors") {
		t.Fatalf("got decoder=%p, err=%v; want layer count error", decoder, err)
	}
}

func TestLoadHFCheckpointReadFailures(t *testing.T) {
	for _, test := range []struct {
		name        string
		truncateAll bool
		tensorName  string
	}{
		{"early", true, "model.embed_tokens.weight"},
		{"late", false, "model.layers.1.mlp.down_proj.weight"},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			config, _, entries := makeHFCheckpointFixture(t, dir, true)
			path := writeHFFixtureTensors(t, dir, safetensors.BF16, entries)
			checkpoint, err := safetensors.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer checkpoint.Close()
			info, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			size := info.Size() - 1
			if test.truncateAll {
				size = 0
			}
			if err := os.Truncate(path, size); err != nil {
				t.Fatal(err)
			}
			if decoder, err := loadHFCheckpoint(config, checkpoint); decoder != nil || err == nil || !strings.Contains(err.Error(), "read tensor "+strconv.Quote(test.tensorName)) {
				t.Fatalf("got decoder=%p, err=%v; want read error for %s", decoder, err, test.tensorName)
			}
		})
	}
}
