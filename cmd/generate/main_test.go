package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/itsmontoya/llama/internal/tokenizer"
)

func TestGenerateArgumentErrors(t *testing.T) {
	for _, test := range []struct {
		name string
		args []string
		want string
	}{
		{"missing model", []string{"-tokenizer", "tokens.bin"}, "-model and -tokenizer are required"},
		{"missing tokenizer", []string{"-model", "model.bin"}, "-model and -tokenizer are required"},
		{"positional argument", []string{"stray"}, "unexpected positional arguments"},
		{"unknown flag", []string{"-unknown"}, "flag provided but not defined"},
		{"invalid limit", []string{"-max-new-tokens", "three"}, "invalid value"},
		{"negative limit", []string{"-model", "model.bin", "-tokenizer", "tokens.bin", "-max-new-tokens", "-1"}, "must be nonnegative"},
		{"HF with model", []string{"-hf-model-dir", "snapshot", "-model", "model.bin"}, "cannot be combined"},
		{"HF with tokenizer", []string{"-hf-model-dir", "snapshot", "-tokenizer="}, "cannot be combined"},
		{"empty HF path", []string{"-hf-model-dir="}, "requires a nonempty directory path"},
		{"context with legacy", []string{"-model", "model.bin", "-tokenizer", "tokens.bin", "-context", "512"}, "-context requires -hf-model-dir"},
		{"FFN backend with legacy", []string{"-model", "model.bin", "-tokenizer", "tokens.bin", "-ffn-backend", "metal"}, "-ffn-backend requires -hf-model-dir"},
		{"unknown FFN backend", []string{"-hf-model-dir", "snapshot", "-ffn-backend", "other"}, "-ffn-backend must be cpu or metal"},
		{"attention backend with legacy", []string{"-model", "model.bin", "-tokenizer", "tokens.bin", "-attention-backend", "metal"}, "-attention-backend requires -hf-model-dir"},
		{"unknown attention backend", []string{"-hf-model-dir", "snapshot", "-attention-backend", "other"}, "-attention-backend must be cpu or metal"},
		{"Metal attention requires Metal FFN", []string{"-hf-model-dir", "snapshot", "-attention-backend", "metal"}, "-attention-backend metal requires -ffn-backend metal"},
		{"uncached with legacy", []string{"-model", "model.bin", "-tokenizer", "tokens.bin", "-uncached"}, "-uncached requires -hf-model-dir"},
		{"invalid context", []string{"-hf-model-dir", "snapshot", "-context", "three"}, "invalid value"},
		{"zero context", []string{"-hf-model-dir", "snapshot", "-context", "0"}, "-context must be positive"},
		{"negative context", []string{"-hf-model-dir", "snapshot", "-context", "-1"}, "-context must be positive"},
		{"HF negative limit", []string{"-hf-model-dir", "snapshot", "-max-new-tokens", "-1"}, "must be nonnegative"},
	} {
		t.Run(test.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			_, status := run(test.args, &stdout, &stderr)
			if status != 2 || stdout.Len() != 0 || !strings.Contains(stderr.String(), test.want) {
				t.Fatalf("exit = %d, stdout = %q, stderr = %q, want exit 2 mentioning %q", status, stdout.String(), stderr.String(), test.want)
			}
		})
	}
}

func TestGenerateSynthetic(t *testing.T) {
	modelPath, tokenizerPath := syntheticArtifacts(t)
	for _, test := range []struct {
		name string
		args []string
		want string
	}{
		{"empty prompt and BOS stop", nil, "there\n"},
		{"default budget and EOS decoded", []string{"-prompt", "!"}, strings.Repeat("</s>!", 8) + "\n"},
		{"explicit budget", []string{"-prompt", "!", "-max-new-tokens", "1"}, "</s>\n"},
		{"remaining context", []string{"-prompt", "!", "-max-new-tokens", "99"}, strings.Repeat("</s>!", 8) + "</s>\n"},
		{"zero budget", []string{"-max-new-tokens", "0"}, "\n"},
	} {
		t.Run(test.name, func(t *testing.T) {
			args := append([]string{"-model", modelPath, "-tokenizer", tokenizerPath}, test.args...)
			var stdout, stderr bytes.Buffer
			_, status := run(args, &stdout, &stderr)
			if status != 0 || stderr.Len() != 0 || stdout.String() != test.want {
				t.Fatalf("exit = %d, stdout = %q, stderr = %q, want success with stdout %q", status, stdout.String(), stderr.String(), test.want)
			}
		})
	}

	t.Run("prompt exceeds context", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		args := []string{"-model", modelPath, "-tokenizer", tokenizerPath, "-prompt", strings.Repeat("!", 20)}
		_, status := run(args, &stdout, &stderr)
		if status == 0 || stdout.Len() != 0 || !strings.Contains(stderr.String(), "prefix length") {
			t.Fatalf("exit = %d, stdout = %q, stderr = %q, want a context failure", status, stdout.String(), stderr.String())
		}
	})
}

func syntheticArtifacts(t *testing.T) (modelPath, tokenizerPath string) {
	t.Helper()
	const (
		dim        = 2
		maxContext = 20
		thereID    = 260
		exclaimID  = 261
		vocabSize  = 262
	)
	embedding := make([]float32, vocabSize*dim)
	output := make([]float32, vocabSize*dim)
	// Zero block projections preserve the embedding direction for each token.
	// BOS predicts " there", which predicts BOS. "!" and EOS predict each other.
	embedding[tokenizer.BOS*dim] = 1
	embedding[thereID*dim+1] = 1
	embedding[exclaimID*dim] = -1
	embedding[tokenizer.EOS*dim+1] = -1
	output[thereID*dim] = 1
	output[tokenizer.BOS*dim+1] = 1
	output[tokenizer.EOS*dim] = -1
	output[exclaimID*dim+1] = -1
	var model bytes.Buffer
	write := func(values any) {
		if err := binary.Write(&model, binary.LittleEndian, values); err != nil {
			t.Fatal(err)
		}
	}
	write([7]int32{dim, dim, 1, 1, 1, -vocabSize, maxContext})
	norm := []float32{1, 1}
	zero := make([]float32, dim*dim)
	write(embedding)
	write(norm)
	for range 4 { // Attention projections.
		write(zero)
	}
	write(norm)
	for range 3 { // FFN projections.
		write(zero)
	}
	write(norm)
	write(make([]float32, maxContext*dim)) // Legacy rotary tables.
	write(output)

	pieces := []string{"<unk>", "<s>", "</s>"}
	for b := range 256 {
		pieces = append(pieces, fmt.Sprintf("<0x%02X>", b))
	}
	pieces = append(pieces, " ", " there", "!")
	var maxLength int
	for _, piece := range pieces {
		maxLength = max(maxLength, len(piece))
	}
	vocabulary := binary.LittleEndian.AppendUint32(nil, uint32(maxLength))
	for _, piece := range pieces {
		vocabulary = binary.LittleEndian.AppendUint32(vocabulary, 0) // Score.
		vocabulary = binary.LittleEndian.AppendUint32(vocabulary, uint32(len(piece)))
		vocabulary = append(vocabulary, piece...)
	}
	dir := t.TempDir()
	modelPath = filepath.Join(dir, "model.bin")
	tokenizerPath = filepath.Join(dir, "tokens.bin")
	if err := os.WriteFile(modelPath, model.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(tokenizerPath, vocabulary, 0o600); err != nil {
		t.Fatal(err)
	}
	return modelPath, tokenizerPath
}

func TestStories260KGeneration(t *testing.T) {
	modelPath := os.Getenv("LLAMA_STORIES260K_MODEL")
	tokenizerPath := os.Getenv("LLAMA_STORIES260K_TOKENIZER")
	if modelPath == "" || tokenizerPath == "" {
		t.Skip("set LLAMA_STORIES260K_MODEL and LLAMA_STORIES260K_TOKENIZER to the pinned artifacts")
	}
	for _, artifact := range []struct{ path, sha256 string }{
		{modelPath, "b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696"},
		{tokenizerPath, "037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312"},
	} {
		data, err := os.ReadFile(artifact.path)
		if err != nil {
			t.Fatal(err)
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != artifact.sha256 {
			t.Fatalf("SHA-256 of %s = %s, want %s", artifact.path, got, artifact.sha256)
		}
	}
	data, err := os.ReadFile(filepath.Join("..", "..", "testdata", "stories260k_generation.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Prompt       string  `json:"prompt"`
		MaxNewTokens int     `json:"max_new_tokens"`
		PromptIDs    []int64 `json:"prompt_ids"`
		GeneratedIDs []int64 `json:"generated_ids"`
		DecodedHex   string  `json:"decoded_hex"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Prompt != "Once upon a time" || fixture.MaxNewTokens != 16 || len(fixture.GeneratedIDs) == 0 || fixture.DecodedHex == "" {
		t.Fatal("expected a nonempty reference continuation with a 16-token budget")
	}
	want, err := hex.DecodeString(fixture.DecodedHex)
	if err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	args := []string{"-model", modelPath, "-tokenizer", tokenizerPath, "-prompt", fixture.Prompt}
	got, status := run(args, &stdout, &stderr)
	wantStdout := string(want) + "\n"
	if status != 0 || stderr.Len() != 0 || stdout.String() != wantStdout {
		t.Fatalf("exit = %d, stdout = %q, stderr = %q, want success with stdout %q", status, stdout.String(), stderr.String(), wantStdout)
	}
	if !slices.Equal(got.promptIDs, fixture.PromptIDs) {
		t.Errorf("prompt IDs = %v, want %v", got.promptIDs, fixture.PromptIDs)
	}
	if !slices.Equal(got.suffixIDs, fixture.GeneratedIDs) {
		t.Errorf("generated IDs = %v, want %v", got.suffixIDs, fixture.GeneratedIDs)
	}
	t.Logf("prompt IDs: %v, generated IDs: %v, stdout: %q", got.promptIDs, got.suffixIDs, stdout.String())
}
