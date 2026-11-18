package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
)

const llama32Revision = "4e20de362430cd3b72f300e6b0f18e50e7166e08"

func TestLlama32Generation(t *testing.T) {
	dir := os.Getenv("LLAMA_32_1B_MODEL_DIR")
	if dir == "" {
		t.Skip("set LLAMA_32_1B_MODEL_DIR to the pinned Meta snapshot")
	}
	verifyGenerationSnapshot(t, dir)
	var fixture struct {
		Revision   string `json:"revision"`
		Schema     int    `json:"schema_version"`
		Generation struct {
			Prompt       string  `json:"input_text"`
			MaxNewTokens int     `json:"max_new_tokens"`
			PromptIDs    []int64 `json:"prompt_ids"`
			SuffixIDs    []int64 `json:"suffix_ids"`
			DecodedHex   string  `json:"decoded_bytes_hex"`
		} `json:"generation"`
	}
	readGenerationFixture(t, "llama32_reference.json", &fixture)
	want := fixture.Generation
	if fixture.Revision != llama32Revision || fixture.Schema != 1 || want.Prompt == "" || want.MaxNewTokens != 8 ||
		len(want.PromptIDs) == 0 || len(want.SuffixIDs) == 0 || want.DecodedHex == "" {
		t.Fatal("invalid or incomplete pinned Llama 3.2 generation reference")
	}
	decoded, err := hex.DecodeString(want.DecodedHex)
	if err != nil {
		t.Fatal(err)
	}
	args := []string{"-hf-model-dir", dir, "-context", "512", "-prompt", want.Prompt, "-max-new-tokens", strconv.Itoa(want.MaxNewTokens)}
	for _, mode := range []struct {
		name  string
		extra []string
	}{
		{name: "cached"},
		{name: "uncached", extra: []string{"-uncached"}},
	} {
		t.Run(mode.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			got, status := run(append(slices.Clone(args), mode.extra...), &stdout, &stderr)
			if status != 0 || stderr.Len() != 0 {
				t.Fatalf("exit = %d, stdout = %q, stderr = %q, want success", status, stdout.String(), stderr.String())
			}
			if !slices.Equal(got.promptIDs, want.PromptIDs) {
				t.Errorf("prompt IDs = %v, want %v", got.promptIDs, want.PromptIDs)
			}
			if !slices.Equal(got.suffixIDs, want.SuffixIDs) {
				t.Errorf("suffix IDs = %v, want %v", got.suffixIDs, want.SuffixIDs)
			}
			if wantStdout := string(decoded) + "\n"; stdout.String() != wantStdout {
				t.Errorf("stdout = %q, want %q", stdout.String(), wantStdout)
			}
			t.Logf("prompt IDs: %v, suffix IDs: %v, stdout: %q", got.promptIDs, got.suffixIDs, stdout.String())
		})
	}
}

func verifyGenerationSnapshot(t *testing.T, dir string) {
	t.Helper()
	var manifest struct {
		Revision string `json:"revision"`
		Files    map[string]struct {
			Size   int64  `json:"size"`
			SHA256 string `json:"sha256"`
		} `json:"files"`
	}
	readGenerationFixture(t, "llama32_manifest.json", &manifest)
	names := []string{"config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"}
	if manifest.Revision != llama32Revision || len(manifest.Files) != len(names) {
		t.Fatal("invalid pinned Llama 3.2 snapshot manifest")
	}
	for _, name := range names {
		want := manifest.Files[name]
		if want.Size <= 0 || len(want.SHA256) != sha256.Size*2 {
			t.Fatalf("missing size or SHA-256 for %s in snapshot manifest", name)
		}
		verifyGenerationArtifact(t, filepath.Join(dir, name), want.Size, want.SHA256)
	}
}

func verifyGenerationArtifact(t *testing.T, path string, size int64, checksum string) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !info.Mode().IsRegular() || info.Size() != size {
		t.Fatalf("%s: expected a regular snapshot file with %d bytes, got mode %v and %d bytes", path, size, info.Mode(), info.Size())
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	digest := sha256.New()
	n, err := io.Copy(digest, file)
	if err != nil {
		t.Fatalf("hash %s: %v", path, err)
	}
	if got := hex.EncodeToString(digest.Sum(nil)); n != size || got != checksum {
		t.Fatalf("%s: got %d bytes and SHA-256 %s, want %d bytes and SHA-256 %s", path, n, got, size, checksum)
	}
}
