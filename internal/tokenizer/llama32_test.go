package tokenizer

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestLlama32TokenizerParity(t *testing.T) {
	dir := os.Getenv("LLAMA_32_1B_MODEL_DIR")
	if dir == "" {
		t.Skip("set LLAMA_32_1B_MODEL_DIR to the pinned Meta snapshot")
	}
	const revision = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
	fixtureDir := filepath.Join("..", "..", "testdata")
	var manifest struct {
		Revision string `json:"revision"`
		Files    map[string]struct {
			Size   int    `json:"size"`
			SHA256 string `json:"sha256"`
		} `json:"files"`
		Architecture struct {
			VocabularySize int64 `json:"vocab_size"`
		} `json:"architecture"`
		Generation struct {
			BOSID   int64   `json:"bos_token_id"`
			StopIDs []int64 `json:"stop_token_ids"`
		} `json:"generation_policy"`
	}
	if err := readTokenizerFixture(filepath.Join(fixtureDir, "llama32_manifest.json"), &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Revision != revision {
		t.Fatalf("snapshot manifest revision %q, want %q", manifest.Revision, revision)
	}
	for _, name := range []string{"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"} {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		want := manifest.Files[name]
		if len(data) != want.Size || hex.EncodeToString(sum[:]) != want.SHA256 {
			t.Fatalf("%s: got %d bytes and SHA-256 %x, want %d bytes and SHA-256 %s", name, len(data), sum, want.Size, want.SHA256)
		}
	}
	var reference struct {
		Revision string `json:"revision"`
		Schema   int    `json:"schema_version"`
		Cases    []struct {
			Name       string  `json:"name"`
			Text       string  `json:"input_text"`
			IDs        []int64 `json:"prompt_ids"`
			DecodedHex string  `json:"decoded_bytes_hex"`
		} `json:"tokenizer_cases"`
		Generation struct {
			Text       string  `json:"input_text"`
			PromptIDs  []int64 `json:"prompt_ids"`
			SuffixIDs  []int64 `json:"suffix_ids"`
			DecodedHex string  `json:"decoded_bytes_hex"`
		} `json:"generation"`
	}
	if err := readTokenizerFixture(filepath.Join(fixtureDir, "llama32_reference.json"), &reference); err != nil {
		t.Fatal(err)
	}
	if reference.Revision != revision || reference.Schema != 1 || len(reference.Cases) < 6 || len(reference.Generation.SuffixIDs) == 0 {
		t.Fatal("invalid or incomplete pinned Llama 3.2 tokenizer reference")
	}
	vocab, err := LoadLlama3(dir, manifest.Architecture.VocabularySize, manifest.Generation.BOSID, manifest.Generation.StopIDs)
	if err != nil {
		t.Fatal(err)
	}
	compareIDs := func(t *testing.T, text string, want []int64) {
		t.Helper()
		if got, err := vocab.Encode(text); err != nil || !slices.Equal(got, want) {
			t.Fatalf("Encode %q: got %v, %v, want %v", text, got, err, want)
		}
	}
	compareBytes := func(t *testing.T, ids []int64, want string) {
		t.Helper()
		decoded, err := vocab.Decode(ids)
		if err != nil || hex.EncodeToString([]byte(decoded)) != want {
			t.Fatalf("Decode %v: got %x, %v, want %s", ids, decoded, err, want)
		}
	}
	for _, test := range reference.Cases {
		t.Run(test.Name, func(t *testing.T) {
			compareIDs(t, test.Text, test.IDs)
			compareBytes(t, test.IDs, test.DecodedHex)
		})
	}
	compareIDs(t, reference.Generation.Text, reference.Generation.PromptIDs)
	compareBytes(t, reference.Generation.SuffixIDs, reference.Generation.DecodedHex)
	t.Logf("matched %d tokenizer cases, generation prompt IDs %v and decoded suffix IDs %v", len(reference.Cases), reference.Generation.PromptIDs, reference.Generation.SuffixIDs)
}

func readTokenizerFixture(path string, out any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}
