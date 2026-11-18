package llama

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

const referenceTolerance = 2e-5

var fixtureNames = []string{"tiny_gqa", "tiny_mha", "tiny_mqa"}

type modelFixture struct {
	Config  Config  `json:"config"`
	Weights Weights `json:"weights"`
}

type referenceTensor struct {
	Shape  []int64   `json:"shape"`
	Values []float64 `json:"values"`
}

type referenceCase struct {
	Name       string                     `json:"name"`
	Tokens     []int64                    `json:"tokens"`
	TokensHash string                     `json:"tokens_sha256"`
	Trace      map[string]referenceTensor `json:"trace"`
}

type referenceFixture struct {
	Metadata struct {
		Schema        int     `json:"schema"`
		ModelHash     string  `json:"model_sha256"`
		GeneratorHash string  `json:"generator_sha256"`
		InputDtype    string  `json:"input_dtype"`
		ReferenceType string  `json:"reference_dtype"`
		ATol          float64 `json:"atol"`
		RTol          float64 `json:"rtol"`
	} `json:"metadata"`
	Cases []referenceCase `json:"cases"`
}

func readFixture(t *testing.T, name string) (modelFixture, referenceFixture) {
	t.Helper()
	var model modelFixture
	var reference referenceFixture
	modelBytes, err := os.ReadFile(filepath.Join("testdata", name+".model.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(modelBytes, &model); err != nil {
		t.Fatal(err)
	}
	referenceBytes, err := os.ReadFile(filepath.Join("testdata", name+".reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(referenceBytes, &reference); err != nil {
		t.Fatal(err)
	}
	generator, err := os.ReadFile(filepath.Join("testdata", "generate_reference.py"))
	if err != nil {
		t.Fatal(err)
	}
	if reference.Metadata.Schema != 1 || reference.Metadata.InputDtype != "float32" || reference.Metadata.ReferenceType != "float64" {
		t.Fatal("unexpected reference schema or arithmetic")
	}
	if reference.Metadata.ATol != referenceTolerance || reference.Metadata.RTol != referenceTolerance {
		t.Fatal("fixture tolerances differ from the conformance contract")
	}
	assertHash(t, modelBytes, reference.Metadata.ModelHash)
	assertHash(t, generator, reference.Metadata.GeneratorHash)
	if len(reference.Cases) != 2 {
		t.Fatal("expected prefix and single-token reference cases")
	}
	for _, test := range reference.Cases {
		encoded := make([]byte, len(test.Tokens)*8)
		for index, token := range test.Tokens {
			binary.LittleEndian.PutUint64(encoded[index*8:], uint64(token))
		}
		assertHash(t, encoded, test.TokensHash)
	}
	return model, reference
}

func assertHash(t *testing.T, payload []byte, want string) {
	t.Helper()
	sum := sha256.Sum256(payload)
	if hex.EncodeToString(sum[:]) != want {
		t.Fatalf("fixture provenance hash mismatch: got %x, want %s", sum, want)
	}
}

func tensorValues(t *testing.T, value *tensor.Tensor) []float32 {
	t.Helper()
	values, err := value.Float32Values()
	if err != nil {
		t.Fatal(err)
	}
	return values
}

func assertClose(t *testing.T, label string, got []float32, want []float64) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d values, want %d", label, len(got), len(want))
	}
	var maximum float64
	for index, value := range got {
		difference := math.Abs(float64(value) - want[index])
		if !finite(float64(value)) || !finite(want[index]) || difference > referenceTolerance+referenceTolerance*math.Abs(want[index]) {
			t.Fatalf("%s[%d]: got %.9g, want %.17g, absolute error %.9g", label, index, value, want[index], difference)
		}
		maximum = max(maximum, difference)
	}
	return maximum
}

func asFloat64(values []float32) []float64 {
	result := make([]float64, len(values))
	for index, value := range values {
		result[index] = float64(value)
	}
	return result
}

func TestDecoderReference(t *testing.T) {
	for _, name := range fixtureNames {
		t.Run(name, func(t *testing.T) {
			model, fixture := readFixture(t, name)
			decoder, err := NewDecoder(model.Config, model.Weights)
			if err != nil {
				t.Fatal(err)
			}
			for _, test := range fixture.Cases {
				t.Run(test.Name, func(t *testing.T) {
					before := slices.Clone(test.Tokens)
					logits, trace, err := decoder.ForwardWithTrace(test.Tokens)
					if err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(before, test.Tokens) || trace["logits"] != logits {
						t.Fatal("token ownership or logits trace contract violated")
					}
					if len(trace) != len(test.Trace) {
						t.Fatalf("got %d trace tensors, want %d", len(trace), len(test.Trace))
					}
					var maximum float64
					for key, expected := range test.Trace {
						actual := trace[key]
						if actual == nil || !slices.Equal(actual.Shape(), expected.Shape) {
							t.Fatalf("%s: missing tensor or incorrect shape; want %v", key, expected.Shape)
						}
						maximum = max(maximum, assertClose(t, key, tensorValues(t, actual), expected.Values))
					}
					plain, err := decoder.Forward(test.Tokens)
					if err != nil {
						t.Fatal(err)
					}
					assertClose(t, "untraced logits", tensorValues(t, plain), test.Trace["logits"].Values)
					assertClose(t, "retained traced logits", tensorValues(t, logits), test.Trace["logits"].Values)
					if plain == logits {
						t.Fatal("Forward reused an earlier output")
					}
					t.Logf("%d intermediate/logit tensors; maximum absolute error %.9g", len(trace), maximum)
				})
			}
		})
	}
}
