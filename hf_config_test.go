package llama

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func syntheticHFConfig() map[string]any {
	return map[string]any{
		"model_type": "llama", "architectures": []string{"LlamaForCausalLM"},
		"hidden_size": 8, "intermediate_size": 13, "num_hidden_layers": 2,
		"num_attention_heads": 2, "vocab_size": 16, "max_position_embeddings": 96,
	}
}

func writeHFTestJSON(t *testing.T, dir, name string, data any) {
	t.Helper()
	encoded, err := json.Marshal(data)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestReadHFConfigDefaults(t *testing.T) {
	dir := t.TempDir()
	writeHFTestJSON(t, dir, "config.json", syntheticHFConfig())
	got, err := readHFConfig(dir, 24)
	if err != nil {
		t.Fatal(err)
	}
	want := Config{
		Dim: 8, FFNDim: 13, Layers: 2, Heads: 2, KVHeads: 2,
		VocabSize: 16, MaxContext: 24, NormEpsilon: 1e-6, RopeTheta: 10000,
	}
	if got.decoder != want || got.maxPositionEmbeddings != 96 || got.decoder.Dim/got.decoder.Heads != 4 {
		t.Fatalf("got config %+v, want %+v with head width 4 and position limit 96", got, want)
	}
	if got.generation.BOSID != 1 || !slices.Equal(got.generation.StopIDs, []int64{2}) {
		t.Fatalf("got generation %+v, want default BOS 1 and EOS 2", got.generation)
	}
}

func TestReadHFConfigGenerationPrecedence(t *testing.T) {
	dir := t.TempDir()
	config := syntheticHFConfig()
	config["bos_token_id"] = 3
	config["eos_token_id"] = 4
	config["head_dim"] = 4
	config["num_key_value_heads"] = 1
	config["tie_word_embeddings"] = true
	writeHFTestJSON(t, dir, "config.json", config)
	writeHFTestJSON(t, dir, "generation_config.json", map[string]any{
		"bos_token_id": 0, "eos_token_id": []int64{15, 5}, "_from_model_config": true,
	})
	got, err := readHFConfig(dir, 96)
	if err != nil {
		t.Fatal(err)
	}
	if got.generation.BOSID != 0 || !slices.Equal(got.generation.StopIDs, []int64{15, 5}) {
		t.Fatalf("got generation %+v, want BOS 0 and ordered EOS IDs [15 5] from generation_config.json", got.generation)
	}
	if got.decoder.KVHeads != 1 || !got.decoder.TieEmbeddings || got.decoder.MaxContext != 96 {
		t.Fatalf("unexpected GQA config %+v", got.decoder)
	}
	if err := os.Remove(filepath.Join(dir, "generation_config.json")); err != nil {
		t.Fatal(err)
	}
	got, err = readHFConfig(dir, 96)
	if err != nil {
		t.Fatal(err)
	}
	if got.generation.BOSID != 3 || !slices.Equal(got.generation.StopIDs, []int64{4}) {
		t.Fatalf("got generation %+v, want BOS 3 and EOS 4 from config.json", got.generation)
	}
}

func TestReadHFConfigRejectsIncompatibleModels(t *testing.T) {
	for _, test := range []struct {
		field string
		value any
		want  string
	}{
		{"model_type", "mistral", "model_type"},
		{"architectures", []string{"LlamaModel"}, "architectures"},
		{"hidden_act", "gelu", "hidden_act"},
		{"hidden_act", nil, "hidden_act"},
		{"attention_bias", true, "attention_bias"},
		{"mlp_bias", true, "mlp_bias"},
		{"pretraining_tp", 2, "pretraining_tp"},
		{"partial_rotary_factor", 0.5, "partial_rotary_factor"},
		{"head_dim", 6, "head_dim"},
		{"hidden_size", 7, "divisible"},
		{"hidden_size", 6, "even head width"},
		{"num_key_value_heads", 3, "divisible"},
		{"num_attention_heads", 0, "dimensions"},
		{"max_position_embeddings", 0, "max_position_embeddings"},
		{"rms_norm_eps", 0, "epsilon"},
		{"rms_norm_eps", nil, "rms_norm_eps"},
		{"rope_theta", -1, "theta"},
		{"rope_theta", nil, "rope_theta"},
		{"rope_scaling", map[string]any{"rope_type": "linear", "factor": 2}, "rope_type"},
		{"rope_scaling", map[string]any{"rope_type": "llama3", "factor": 0, "low_freq_factor": 1,
			"high_freq_factor": 4, "original_max_position_embeddings": 8}, "scaling factors"},
		{"rope_scaling", map[string]any{"rope_type": "llama3", "factor": 4, "low_freq_factor": 4,
			"high_freq_factor": 1, "original_max_position_embeddings": 8}, "high_freq_factor"},
		{"rope_scaling", map[string]any{"rope_type": "llama3", "factor": 4, "low_freq_factor": 1,
			"high_freq_factor": 4}, "original context"},
	} {
		t.Run(test.field, func(t *testing.T) {
			dir := t.TempDir()
			config := syntheticHFConfig()
			config[test.field] = test.value
			writeHFTestJSON(t, dir, "config.json", config)
			if _, err := readHFConfig(dir, 24); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("got error %v, want %q", err, test.want)
			}
		})
	}
}

func TestReadHFConfigRejectsInvalidContext(t *testing.T) {
	dir := t.TempDir()
	writeHFTestJSON(t, dir, "config.json", syntheticHFConfig())
	for _, cap := range []int64{-1, 0, 97} {
		if _, err := readHFConfig(dir, cap); err == nil || !strings.Contains(err.Error(), "runtime context") {
			t.Fatalf("cap %d: got error %v, want runtime context error", cap, err)
		}
	}
}

func TestReadHFConfigRejectsInvalidGenerationIDs(t *testing.T) {
	dir := t.TempDir()
	writeHFTestJSON(t, dir, "config.json", syntheticHFConfig())
	for _, test := range []struct {
		name       string
		generation string
		want       string
	}{
		{"absent BOS", `{"eos_token_id":2}`, "bos_token_id"},
		{"negative BOS", `{"bos_token_id":-1,"eos_token_id":2}`, "bos_token_id"},
		{"noninteger BOS", `{"bos_token_id":1.5,"eos_token_id":2}`, "bos_token_id"},
		{"out of range BOS", `{"bos_token_id":16,"eos_token_id":2}`, "bos_token_id"},
		{"absent EOS", `{"bos_token_id":1}`, "eos_token_id"},
		{"null EOS", `{"bos_token_id":1,"eos_token_id":null}`, "eos_token_id"},
		{"empty EOS", `{"bos_token_id":1,"eos_token_id":[]}`, "eos_token_id"},
		{"null EOS element", `{"bos_token_id":1,"eos_token_id":[2,null]}`, "eos_token_id"},
		{"noninteger EOS element", `{"bos_token_id":1,"eos_token_id":[2,2.5]}`, "eos_token_id"},
		{"negative EOS", `{"bos_token_id":1,"eos_token_id":-1}`, "eos_token_id"},
		{"out of range EOS element", `{"bos_token_id":1,"eos_token_id":[2,16]}`, "eos_token_id"},
		{"malformed JSON", `{"bos_token_id":1,"eos_token_id":2} {}`, "invalid character"},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(dir, "generation_config.json")
			if err := os.WriteFile(path, []byte(test.generation), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := readHFConfig(dir, 24); err == nil || !strings.Contains(err.Error(), "generation_config.json") || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("got error %v, want generation_config.json error containing %q", err, test.want)
			}
		})
	}
}

func TestLlama32Config(t *testing.T) {
	dir := os.Getenv("LLAMA_32_1B_MODEL_DIR")
	if dir == "" {
		t.Skip("set LLAMA_32_1B_MODEL_DIR to the pinned Meta snapshot")
	}
	var manifest struct {
		Architecture struct {
			Dim           int64       `json:"hidden_size"`
			FFNDim        int64       `json:"intermediate_size"`
			Layers        int64       `json:"num_hidden_layers"`
			Heads         int64       `json:"num_attention_heads"`
			KVHeads       int64       `json:"num_key_value_heads"`
			HeadWidth     int64       `json:"head_dim"`
			VocabSize     int64       `json:"vocab_size"`
			MaxPosition   int64       `json:"max_position_embeddings"`
			NormEpsilon   float32     `json:"rms_norm_eps"`
			RopeTheta     float64     `json:"rope_theta"`
			RopeScaling   RopeScaling `json:"rope_scaling"`
			TieEmbeddings bool        `json:"tie_word_embeddings"`
		} `json:"architecture"`
		Generation struct {
			BOSID   int64   `json:"bos_token_id"`
			StopIDs []int64 `json:"stop_token_ids"`
		} `json:"generation_policy"`
		Files map[string]struct {
			Size   int    `json:"size"`
			SHA256 string `json:"sha256"`
		} `json:"files"`
	}
	if err := readHFJSON("testdata/llama32_manifest.json", &manifest); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"config.json", "generation_config.json"} {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		file := manifest.Files[name]
		if len(data) != file.Size {
			t.Fatalf("%s: got %d bytes, want %d", name, len(data), file.Size)
		}
		assertHash(t, data, file.SHA256)
	}
	const contextCap = 512
	got, err := readHFConfig(dir, contextCap)
	if err != nil {
		t.Fatal(err)
	}
	arch := manifest.Architecture
	want := Config{
		Dim: arch.Dim, FFNDim: arch.FFNDim, Layers: arch.Layers,
		Heads: arch.Heads, KVHeads: arch.KVHeads, VocabSize: arch.VocabSize, MaxContext: contextCap,
		NormEpsilon: arch.NormEpsilon, RopeTheta: arch.RopeTheta, RopeScaling: arch.RopeScaling, TieEmbeddings: arch.TieEmbeddings,
	}
	if got.decoder != want || got.maxPositionEmbeddings != arch.MaxPosition || got.decoder.Dim/got.decoder.Heads != arch.HeadWidth {
		t.Fatalf("got config %+v, want %+v with head width %d and position limit %d", got, want, arch.HeadWidth, arch.MaxPosition)
	}
	if got.generation.BOSID != manifest.Generation.BOSID || !slices.Equal(got.generation.StopIDs, manifest.Generation.StopIDs) {
		t.Fatalf("got generation %+v, want %+v", got.generation, manifest.Generation)
	}
	t.Logf("resolved %+v with head width %d and position limit %d; generation %+v", got.decoder, arch.HeadWidth, got.maxPositionEmbeddings, got.generation)
}
