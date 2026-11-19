package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

const (
	hfFixtureBOS   = 256
	hfFixtureEOS   = 257
	hfFixtureVocab = 258
)

func TestGenerateHFSynthetic(t *testing.T) {
	dir := syntheticHFArtifacts(t)
	for _, test := range []struct {
		name      string
		args      []string
		promptIDs []int64
		suffixIDs []int64
		stdout    string
	}{
		{"default context and EOS stop", nil, []int64{hfFixtureBOS}, []int64{'!'}, "!\n"},
		{"first prediction is EOS", []string{"-prompt", "!"}, []int64{hfFixtureBOS, '!'}, nil, "\n"},
		{"zero budget", []string{"-max-new-tokens", "0"}, []int64{hfFixtureBOS}, nil, "\n"},
		{"exhausted context", []string{"-context", "1"}, []int64{hfFixtureBOS}, nil, "\n"},
	} {
		t.Run(test.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			args := append([]string{"-hf-model-dir", dir}, test.args...)
			got, status := run(args, &stdout, &stderr)
			if status != 0 || stderr.Len() != 0 || stdout.String() != test.stdout {
				t.Fatalf("exit = %d, stdout = %q, stderr = %q, want success with stdout %q", status, stdout.String(), stderr.String(), test.stdout)
			}
			if !slices.Equal(got.promptIDs, test.promptIDs) || !slices.Equal(got.suffixIDs, test.suffixIDs) {
				t.Fatalf("prompt IDs = %v, suffix IDs = %v, want %v and %v", got.promptIDs, got.suffixIDs, test.promptIDs, test.suffixIDs)
			}
		})
	}
}

func TestGenerateHFFailures(t *testing.T) {
	dir := syntheticHFArtifacts(t)
	for _, test := range []struct {
		name string
		args []string
		want string
	}{
		{"context exceeds model limit", []string{"-context", "513"}, "runtime context must be in [1,512]"},
		{"prompt exceeds context", []string{"-context", "1", "-prompt", "!"}, "prefix length"},
	} {
		t.Run(test.name, func(t *testing.T) {
			assertHFGenerationError(t, dir, test.args, test.want)
		})
	}
	t.Run("model and tokenizer disagree", func(t *testing.T) {
		dir := syntheticHFArtifacts(t)
		writeGenerationJSON(t, dir, "generation_config.json", map[string]int{
			"bos_token_id": hfFixtureEOS, "eos_token_id": hfFixtureEOS,
		})
		assertHFGenerationError(t, dir, nil, "bos_token disagrees with checkpoint BOS")
	})
	t.Run("unsupported configuration", func(t *testing.T) {
		dir := syntheticHFArtifacts(t)
		writeGenerationJSON(t, dir, "config.json", map[string]string{"model_type": "unsupported"})
		assertHFGenerationError(t, dir, nil, "unsupported model_type")
	})
}

func assertHFGenerationError(t *testing.T, dir string, extraArgs []string, want string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	args := append([]string{"-hf-model-dir", dir}, extraArgs...)
	_, status := run(args, &stdout, &stderr)
	if status != 1 || stdout.Len() != 0 || !strings.Contains(stderr.String(), want) {
		t.Fatalf("exit = %d, stdout = %q, stderr = %q, want exit 1 mentioning %q", status, stdout.String(), stderr.String(), want)
	}
}

func syntheticHFArtifacts(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeGenerationJSON(t, dir, "config.json", map[string]any{
		"model_type": "llama", "hidden_size": 2, "intermediate_size": 2, "num_hidden_layers": 1,
		"num_attention_heads": 1, "vocab_size": hfFixtureVocab, "max_position_embeddings": 512,
	})
	writeGenerationJSON(t, dir, "generation_config.json", map[string]int{
		"bos_token_id": hfFixtureBOS, "eos_token_id": hfFixtureEOS,
	})
	writeSyntheticHFWeights(t, dir)
	writeSyntheticHFTokenizer(t, dir)
	return dir
}

func writeSyntheticHFWeights(t *testing.T, dir string) {
	t.Helper()
	const dim = 2
	embedding := make([]float32, hfFixtureVocab*dim)
	classifier := make([]float32, hfFixtureVocab*dim)
	// Zero block projections preserve the input direction. BOS predicts '!'
	// and '!' predicts EOS. If generation continued through EOS, it would emit 'x'.
	embedding[hfFixtureBOS*dim] = 1
	embedding['!'*dim+1] = 1
	embedding[hfFixtureEOS*dim] = -1
	classifier['!'*dim] = 1
	classifier[hfFixtureEOS*dim+1] = 1
	classifier['x'*dim] = -1

	type descriptor struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	header := make(map[string]descriptor)
	var payload []byte
	add := func(name string, shape []int, values []float32) {
		start := len(payload)
		for _, value := range values {
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
		}
		header[name] = descriptor{DType: "F32", Shape: shape, Offsets: [2]int{start, len(payload)}}
	}
	add("model.embed_tokens.weight", []int{hfFixtureVocab, dim}, embedding)
	add("lm_head.weight", []int{hfFixtureVocab, dim}, classifier)
	add("model.norm.weight", []int{dim}, []float32{1, 1})
	for _, name := range []string{"input_layernorm.weight", "post_attention_layernorm.weight"} {
		add("model.layers.0."+name, []int{dim}, []float32{1, 1})
	}
	for _, name := range []string{
		"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
		"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
	} {
		add("model.layers.0."+name, []int{dim, dim}, make([]float32, dim*dim))
	}
	encoded, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := binary.LittleEndian.AppendUint64(nil, uint64(len(encoded)))
	checkpoint = append(checkpoint, encoded...)
	checkpoint = append(checkpoint, payload...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), checkpoint, 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeSyntheticHFTokenizer(t *testing.T, dir string) {
	t.Helper()
	// Keep the audited byte alphabet and processing stages while using a tiny,
	// merge-free vocabulary with a distinct BOS and EOS.
	var manifest struct {
		Tokenizer struct {
			ByteToUnicode []string `json:"byte_to_unicode"`
			Backend       struct {
				PreTokenizer json.RawMessage `json:"pre_tokenizer"`
				Decoder      json.RawMessage `json:"decoder"`
			} `json:"backend"`
		} `json:"tokenizer"`
	}
	readGenerationFixture(t, "llama32_manifest.json", &manifest)
	vocab := make(map[string]int, 256)
	for id, piece := range manifest.Tokenizer.ByteToUnicode {
		vocab[piece] = id
	}
	postProcessor := json.RawMessage(`{
		"type":"Sequence","processors":[
			{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true},
			{"type":"TemplateProcessing",
			 "single":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
			 "pair":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},
			         {"SpecialToken":{"id":"<bos>","type_id":1}},{"Sequence":{"id":"B","type_id":1}}],
			 "special_tokens":{"<bos>":{"id":"<bos>","ids":[256],"tokens":["<bos>"]}}}
		]}`)
	writeGenerationJSON(t, dir, "tokenizer.json", map[string]any{
		"version": "1.0", "pre_tokenizer": manifest.Tokenizer.Backend.PreTokenizer,
		"post_processor": postProcessor, "decoder": manifest.Tokenizer.Backend.Decoder,
		"added_tokens": []map[string]any{
			{"content": "<bos>", "id": hfFixtureBOS, "special": true},
			{"content": "<eos>", "id": hfFixtureEOS, "special": true},
		},
		"model": map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}, "ignore_merges": true},
	})
	writeGenerationJSON(t, dir, "tokenizer_config.json", map[string]any{
		"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<bos>", "eos_token": "<eos>",
		"added_tokens_decoder": map[int]map[string]any{
			hfFixtureBOS: {"content": "<bos>", "special": true},
			hfFixtureEOS: {"content": "<eos>", "special": true},
		},
	})
	writeGenerationJSON(t, dir, "special_tokens_map.json", map[string]map[string]string{
		"bos_token": {"content": "<bos>"}, "eos_token": {"content": "<eos>"},
	})
}

func writeGenerationJSON(t *testing.T, dir, name string, value any) {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
}

func readGenerationFixture(t *testing.T, name string, value any) {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", name)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, value); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}
