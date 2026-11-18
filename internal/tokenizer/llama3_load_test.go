package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

const (
	fixtureLlama3VocabSize = 259
	fixtureLlama3BOSID     = 257
	fixtureLlama3EOSID     = 258
	fixtureLlama3BOS       = "<bos>"
	fixtureLlama3EOS       = "<eos>"
)

func llama3Fixture(t *testing.T) (file, settings, special map[string]any) {
	t.Helper()
	vocab := make(map[string]int)
	for id, piece := range llama3ByteAlphabet() {
		vocab[string(piece)] = id
	}
	vocab["ab"] = 256
	file = map[string]any{
		"version": "1.0", "normalizer": nil, "truncation": nil, "padding": nil,
		"pre_tokenizer":  json.RawMessage(llama3PreTokenizer()),
		"post_processor": json.RawMessage(llama3PostProcessor(fixtureLlama3BOS, fixtureLlama3BOSID)),
		"decoder":        json.RawMessage(llama3DecoderJSON),
		"model":          map[string]any{"type": "BPE", "ignore_merges": true, "vocab": vocab, "merges": []string{"a b"}},
		"added_tokens": []map[string]any{
			{"id": fixtureLlama3BOSID, "content": fixtureLlama3BOS, "special": true},
			{"id": fixtureLlama3EOSID, "content": fixtureLlama3EOS, "special": true},
		},
	}
	settings = map[string]any{
		"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": fixtureLlama3BOS, "eos_token": fixtureLlama3EOS,
		"clean_up_tokenization_spaces": true,
		"added_tokens_decoder": map[string]llama3TokenData{
			"257": {Content: fixtureLlama3BOS, Special: true}, "258": {Content: fixtureLlama3EOS, Special: true},
		},
	}
	special = map[string]any{
		"bos_token": llama3TokenData{Content: fixtureLlama3BOS},
		"eos_token": llama3TokenData{Content: fixtureLlama3EOS},
	}
	return file, settings, special
}

func writeLlama3Fixture(t *testing.T, file, settings, special map[string]any) string {
	t.Helper()
	dir := t.TempDir()
	for name, value := range map[string]map[string]any{
		"tokenizer.json": file, "tokenizer_config.json": settings, "special_tokens_map.json": special,
	} {
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestLoadLlama3(t *testing.T) {
	file, settings, special := llama3Fixture(t)
	dir := writeLlama3Fixture(t, file, settings, special)
	stops := []int64{fixtureLlama3EOSID}
	vocab, err := LoadLlama3(dir, fixtureLlama3VocabSize, fixtureLlama3BOSID, stops)
	if err != nil {
		t.Fatal(err)
	}
	stops[0] = -1
	ids, err := vocab.Encode("ab<eos>á!! .")
	want := []int64{fixtureLlama3BOSID, 256, fixtureLlama3EOSID, 0xC3, 0xA1, '!', '!', ' ', '.'}
	if err != nil || !slices.Equal(ids, want) {
		t.Fatalf("Encode: got %v, %v, want %v", ids, err, want)
	}
	if text, err := vocab.Decode(ids); err != nil || text != "abá!! ." {
		t.Fatalf("Decode: got %q, %v, want %q", text, err, "abá!! .")
	}
	// ByteLevel maps space to U+0120 and newline to U+010A.
	if alphabet := llama3ByteAlphabet(); alphabet[' '] != 'Ġ' || alphabet['\n'] != 'Ċ' || alphabet[0] != 'Ā' {
		t.Fatal("incorrect ByteLevel alphabet")
	}
}

func TestLoadLlama3RejectsBadArtifacts(t *testing.T) {
	for _, test := range []struct {
		name, want string
		change     func(file, settings, special map[string]any)
	}{
		{"unrecognized regex", "pre_tokenizer", func(file, _, _ map[string]any) {
			file["pre_tokenizer"] = json.RawMessage(strings.Replace(llama3PreTokenizer(), "Isolated", "MergedWithPrevious", 1))
		}},
		{"normalizer", "normalizer", func(file, _, _ map[string]any) {
			file["normalizer"] = map[string]string{"type": "NFC"}
		}},
		{"wrong decoder", "decoder", func(file, _, _ map[string]any) {
			file["decoder"] = map[string]string{"type": "ByteFallback"}
		}},
		{"missing BOS template", "post_processor", func(file, _, _ map[string]any) {
			file["post_processor"] = nil
		}},
		{"mismatched BOS template", "post_processor", func(file, _, _ map[string]any) {
			file["post_processor"] = json.RawMessage(llama3PostProcessor(fixtureLlama3BOS, fixtureLlama3EOSID))
		}},
		{"wrong BPE options", "BPE model options", func(file, _, _ map[string]any) {
			file["model"].(map[string]any)["ignore_merges"] = false
		}},
		{"duplicate base ID", "duplicate base vocabulary ID", func(file, _, _ map[string]any) {
			file["model"].(map[string]any)["vocab"].(map[string]int)["ab"] = 0
		}},
		{"out of range base ID", "invalid base vocabulary ID", func(file, _, _ map[string]any) {
			file["model"].(map[string]any)["vocab"].(map[string]int)["ab"] = fixtureLlama3VocabSize
		}},
		{"invalid ByteLevel piece", "outside the ByteLevel alphabet", func(file, _, _ map[string]any) {
			vocab := file["model"].(map[string]any)["vocab"].(map[string]int)
			delete(vocab, "ab")
			vocab["🤍"] = 256
		}},
		{"missing byte mapping", "missing base token for byte 0x61", func(file, _, _ map[string]any) {
			vocab := file["model"].(map[string]any)["vocab"].(map[string]int)
			delete(vocab, "a")
			vocab["aa"] = 'a'
		}},
		{"invalid merge", "not a pair", func(file, _, _ map[string]any) {
			file["model"].(map[string]any)["merges"] = []string{"ab"}
		}},
		{"bogus merge", "missing vocabulary piece", func(file, _, _ map[string]any) {
			file["model"].(map[string]any)["merges"] = []string{"a c"}
		}},
		{"duplicate merge", "duplicate merge", func(file, _, _ map[string]any) {
			file["model"].(map[string]any)["merges"] = []string{"a b", "a b"}
		}},
		{"duplicate special ID", "duplicate added token ID", func(file, _, _ map[string]any) {
			file["added_tokens"].([]map[string]any)[1]["id"] = fixtureLlama3BOSID
		}},
		{"duplicate special content", "duplicate added token content", func(file, _, _ map[string]any) {
			file["added_tokens"].([]map[string]any)[1]["content"] = fixtureLlama3BOS
		}},
		{"unsupported special flags", "unsupported added token flags", func(file, _, _ map[string]any) {
			file["added_tokens"].([]map[string]any)[1]["lstrip"] = true
		}},
		{"conflicting decoder metadata", "added_tokens_decoder disagrees", func(_, settings, _ map[string]any) {
			settings["added_tokens_decoder"].(map[string]llama3TokenData)["257"] = llama3TokenData{Content: fixtureLlama3BOS}
		}},
		{"conflicting special map", "special tokens disagree", func(_, _, special map[string]any) {
			special["bos_token"] = llama3TokenData{Content: fixtureLlama3EOS}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			file, settings, special := llama3Fixture(t)
			test.change(file, settings, special)
			dir := writeLlama3Fixture(t, file, settings, special)
			vocab, err := LoadLlama3(dir, fixtureLlama3VocabSize, fixtureLlama3BOSID, []int64{fixtureLlama3EOSID})
			if vocab != nil || err == nil || !strings.Contains(err.Error(), test.want) {
				t.Errorf("LoadLlama3: got %p, %v, want error containing %q", vocab, err, test.want)
			}
		})
	}
}

func TestLoadLlama3RejectsBadModelMetadata(t *testing.T) {
	file, settings, special := llama3Fixture(t)
	dir := writeLlama3Fixture(t, file, settings, special)
	for _, test := range []struct {
		vocabSize, bosID int64
		stopIDs          []int64
		want             string
	}{
		{1 << 50, fixtureLlama3BOSID, []int64{fixtureLlama3EOSID}, "checkpoint vocabulary size"},
		{fixtureLlama3VocabSize, fixtureLlama3EOSID, []int64{fixtureLlama3EOSID}, "checkpoint BOS ID"},
		{fixtureLlama3VocabSize, fixtureLlama3BOSID, []int64{0, fixtureLlama3EOSID}, "stop ID 0"},
		{fixtureLlama3VocabSize, fixtureLlama3BOSID, []int64{fixtureLlama3BOSID}, "eos_token does not match"},
		{fixtureLlama3VocabSize, fixtureLlama3BOSID, nil, "eos_token does not match"},
	} {
		vocab, err := LoadLlama3(dir, test.vocabSize, test.bosID, test.stopIDs)
		if vocab != nil || err == nil || !strings.Contains(err.Error(), test.want) {
			t.Errorf("LoadLlama3: got %p, %v, want error containing %q", vocab, err, test.want)
		}
	}
}

func TestLoadLlama3RejectsMalformedJSON(t *testing.T) {
	for _, test := range []struct{ name, data, want string }{
		{"truncated", `{"version":"1.0",`, "end of JSON input"},
		{"trailing JSON", `{} {}`, "unexpected trailing JSON"},
		{"invalid UTF-8", "{\"\xff\":0}", "invalid UTF-8"},
		{"duplicate root field", `{"version":"1.0","version":"1.0"}`, "duplicate JSON key"},
		{"escaped duplicate piece", `{"model":{"vocab":{"a":0,"\u0061":1}}}`, "duplicate JSON key"},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(test.data), 0o600); err != nil {
				t.Fatal(err)
			}
			vocab, err := LoadLlama3(dir, fixtureLlama3VocabSize, fixtureLlama3BOSID, []int64{fixtureLlama3EOSID})
			if vocab != nil || err == nil || !strings.Contains(err.Error(), "tokenizer.json") || !strings.Contains(err.Error(), test.want) {
				t.Errorf("LoadLlama3: got %p, %v, want tokenizer.json error containing %q", vocab, err, test.want)
			}
		})
	}
}
