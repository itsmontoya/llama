package safetensors

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestLlama32SafetensorsInventory(t *testing.T) {
	dir := os.Getenv("LLAMA_32_1B_MODEL_DIR")
	if dir == "" {
		t.Skip("set LLAMA_32_1B_MODEL_DIR to the pinned Meta snapshot")
	}
	data, err := os.ReadFile(filepath.Join("..", "..", "testdata", "llama32_manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		Files map[string]struct {
			SHA256 string `json:"sha256"`
			Size   int64  `json:"size"`
		} `json:"files"`
		Safetensors struct {
			Metadata          map[string]string `json:"metadata"`
			TensorCount       int               `json:"tensor_count"`
			ParameterCount    int64             `json:"parameter_count"`
			ClassifierPresent bool              `json:"classifier_present"`
			Tensors           []struct {
				Name  string  `json:"name"`
				File  string  `json:"file"`
				DType DType   `json:"dtype"`
				Shape []int64 `json:"shape"`
			} `json:"tensors"`
		} `json:"safetensors"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	const filename = "model.safetensors"
	path := filepath.Join(dir, filename)
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	expectedFile := manifest.Files[filename]
	info, err := file.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() != expectedFile.Size {
		t.Fatalf("%s size: got %d, want %d", path, info.Size(), expectedFile.Size)
	}
	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		t.Fatalf("hash %s: %v", path, err)
	}
	if sum := hex.EncodeToString(digest.Sum(nil)); sum != expectedFile.SHA256 {
		t.Fatalf("%s SHA-256: got %s, want %s", path, sum, expectedFile.SHA256)
	}

	archive := openArchive(t, path)
	tensors := archive.Tensors()
	want := manifest.Safetensors
	if len(tensors) != want.TensorCount || len(tensors) != len(want.Tensors) {
		t.Fatalf("got %d tensors, want %d", len(tensors), want.TensorCount)
	}
	if got := archive.Metadata(); !maps.Equal(got, want.Metadata) {
		t.Fatalf("got metadata %v, want %v", got, want.Metadata)
	}
	if _, present := tensors["lm_head.weight"]; present != want.ClassifierPresent {
		t.Fatalf("classifier present = %t, want %t", present, want.ClassifierPresent)
	}
	var parameters int64
	for _, expected := range want.Tensors {
		tensor, found := tensors[expected.Name]
		if !found || expected.File != filename || tensor.DType != expected.DType || !slices.Equal(tensor.Shape, expected.Shape) {
			t.Fatalf("tensor %q: got %+v, want %+v", expected.Name, tensor, expected)
		}
		parameters += (tensor.Offsets[1] - tensor.Offsets[0]) / int64(tensor.DType.bytes())
		delete(tensors, expected.Name)
	}
	if parameters != want.ParameterCount {
		t.Fatalf("got %d parameters, want %d", parameters, want.ParameterCount)
	}
	t.Logf("validated %s (%d bytes, %d tensors, %d parameters)", filename, info.Size(), want.TensorCount, parameters)
}
