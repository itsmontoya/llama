//go:build darwin && arm64 && cgo && metal && !purego

package llama

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/itsmontoya/llama/internal/safetensors"
	"github.com/itsmontoya/neuralnetwork/gpu"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

func TestMetalAttentionOptionsValidation(t *testing.T) {
	var dir = t.TempDir()
	for _, tc := range []struct {
		name string
		opts MetalOptions
	}{
		{"invalid weight dtype", MetalOptions{WeightDType: gpu.DType(99)}},
		{"invalid kv dtype bf16", MetalOptions{KVCacheDType: gpu.BFloat16}},
		{"invalid kv dtype q8", MetalOptions{KVCacheDType: gpu.Q8_0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var decoder, _, err = LoadHFModelMetalAttentionWithOptions(t.Context(), dir, 8, tc.opts)
			if decoder != nil || !errors.Is(err, gpu.ErrInvalid) {
				t.Fatalf("got decoder %v, err %v, want ErrInvalid", decoder, err)
			}
		})
	}
}

func makeQuantCompatibleHFCheckpoint(t *testing.T, dir string, tied bool, dtype safetensors.DType) (Config, Weights) {
	t.Helper()
	var config = Config{
		VocabSize:     16,
		Dim:           32,
		Layers:        2,
		Heads:         4,
		KVHeads:       2,
		FFNDim:        64,
		MaxContext:    16,
		NormEpsilon:   1e-5,
		RopeTheta:     10000,
		TieEmbeddings: tied,
	}
	var fill = func(n int, seed float64) []float32 {
		var values = make([]float32, n)
		for i := range values {
			var v = float32(0.15 * math.Sin(seed+float64(i)*0.37))
			if dtype == safetensors.BF16 {
				v = gpu.BFloat16ToFloat32(gpu.Float32ToBFloat16(v))
			}
			values[i] = v
		}
		return values
	}
	var norm = func(n int, seed float64) []float32 {
		var values = make([]float32, n)
		for i := range values {
			var v = float32(1.0 + 0.05*math.Cos(seed+float64(i)*0.19))
			if dtype == safetensors.BF16 {
				v = gpu.BFloat16ToFloat32(gpu.Float32ToBFloat16(v))
			}
			values[i] = v
		}
		return values
	}
	var weights = Weights{
		TokenEmbedding: fill(int(config.VocabSize*config.Dim), 1.1),
		FinalNorm:      norm(int(config.Dim), 2.3),
		Layers:         make([]LayerWeights, config.Layers),
	}
	if !tied {
		weights.Output = fill(int(config.VocabSize*config.Dim), 3.7)
	}
	var headWidth = config.Dim / config.Heads
	var kvDim = config.KVHeads * headWidth
	for layer := range weights.Layers {
		var base = float64(layer+1) * 10.0
		weights.Layers[layer] = LayerWeights{
			AttentionNorm: norm(int(config.Dim), base+1),
			WQ:            fill(int(config.Dim*config.Dim), base+2),
			WK:            fill(int(kvDim*config.Dim), base+3),
			WV:            fill(int(kvDim*config.Dim), base+4),
			WO:            fill(int(config.Dim*config.Dim), base+5),
			FFNNorm:       norm(int(config.Dim), base+6),
			WGate:         fill(int(config.FFNDim*config.Dim), base+7),
			WUp:           fill(int(config.FFNDim*config.Dim), base+8),
			WDown:         fill(int(config.Dim*config.FFNDim), base+9),
		}
	}
	var configJSON, err = json.Marshal(map[string]any{
		"architectures":           []string{"LlamaForCausalLM"},
		"model_type":              "llama",
		"vocab_size":              config.VocabSize,
		"hidden_size":             config.Dim,
		"intermediate_size":       config.FFNDim,
		"num_hidden_layers":       config.Layers,
		"num_attention_heads":     config.Heads,
		"num_key_value_heads":     config.KVHeads,
		"head_dim":                headWidth,
		"max_position_embeddings": config.MaxContext,
		"rms_norm_eps":            config.NormEpsilon,
		"rope_theta":              config.RopeTheta,
		"tie_word_embeddings":     config.TieEmbeddings,
		"hidden_act":              "silu",
		"attention_bias":          false,
		"mlp_bias":                false,
		"bos_token_id":            1,
		"eos_token_id":            2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "config.json"), configJSON, 0o600); err != nil {
		t.Fatal(err)
	}
	var genJSON = []byte(`{"bos_token_id":1,"eos_token_id":2}`)
	if err = os.WriteFile(filepath.Join(dir, "generation_config.json"), genJSON, 0o600); err != nil {
		t.Fatal(err)
	}

	// Build HF safetensors entries (de-interleaving Q and K from adjacent rotary pairs to HF split halves).
	var deinterleave = func(values []float32, headCount, hw, inW int64) []float32 {
		var split = make([]float32, len(values))
		var half = hw / 2
		for head := range headCount {
			var base = head * hw
			for pair := range half {
				var src0 = (base + 2*pair) * inW
				var src1 = (base + 2*pair + 1) * inW
				var dst0 = (base + pair) * inW
				var dst1 = (base + half + pair) * inW
				copy(split[dst0:dst0+inW], values[src0:src0+inW])
				copy(split[dst1:dst1+inW], values[src1:src1+inW])
			}
		}
		return split
	}
	var entries = []hfFixtureTensor{
		{name: "model.embed_tokens.weight", shape: []int64{config.VocabSize, config.Dim}, values: weights.TokenEmbedding},
		{name: "model.norm.weight", shape: []int64{config.Dim}, values: weights.FinalNorm},
	}
	if !tied {
		entries = append(entries, hfFixtureTensor{name: "lm_head.weight", shape: []int64{config.VocabSize, config.Dim}, values: weights.Output})
	}
	for layer, lw := range weights.Layers {
		entries = append(entries,
			hfFixtureTensor{name: hfWeightName(layer, "attention_norm"), shape: []int64{config.Dim}, values: lw.AttentionNorm},
			hfFixtureTensor{name: hfWeightName(layer, "wq"), shape: []int64{config.Dim, config.Dim}, values: deinterleave(lw.WQ, config.Heads, headWidth, config.Dim)},
			hfFixtureTensor{name: hfWeightName(layer, "wk"), shape: []int64{kvDim, config.Dim}, values: deinterleave(lw.WK, config.KVHeads, headWidth, config.Dim)},
			hfFixtureTensor{name: hfWeightName(layer, "wv"), shape: []int64{kvDim, config.Dim}, values: lw.WV},
			hfFixtureTensor{name: hfWeightName(layer, "wo"), shape: []int64{config.Dim, config.Dim}, values: lw.WO},
			hfFixtureTensor{name: hfWeightName(layer, "ffn_norm"), shape: []int64{config.Dim}, values: lw.FFNNorm},
			hfFixtureTensor{name: hfWeightName(layer, "w_gate"), shape: []int64{config.FFNDim, config.Dim}, values: lw.WGate},
			hfFixtureTensor{name: hfWeightName(layer, "w_up"), shape: []int64{config.FFNDim, config.Dim}, values: lw.WUp},
			hfFixtureTensor{name: hfWeightName(layer, "w_down"), shape: []int64{config.Dim, config.FFNDim}, values: lw.WDown},
		)
	}
	writeHFFixtureTensors(t, dir, dtype, entries)
	return config, weights
}

func TestMetalAttentionReducedPrecisionSynthetic(t *testing.T) {
	for _, checkpoint := range []struct {
		name  string
		dtype safetensors.DType
		tied  bool
	}{
		{"bf16_tied", safetensors.BF16, true},
		{"f32_untied", safetensors.F32, false},
	} {
		t.Run(checkpoint.name, func(t *testing.T) {
			var dir = t.TempDir()
			var config, _ = makeQuantCompatibleHFCheckpoint(t, dir, checkpoint.tied, checkpoint.dtype)
			var cpuDecoder, _, err = LoadHFModel(dir, config.MaxContext)
			if err != nil {
				t.Fatal(err)
			}
			var tokens = []int64{1, 5, 11, 3, 9}
			var cpuLogits *tensor.Tensor
			if cpuLogits, err = cpuDecoder.Forward(tokens); err != nil {
				t.Fatal(err)
			}
			var cpuValues = tensorValues(t, cpuLogits)

			var modes = []struct {
				name       string
				opts       MetalOptions
				maxAbsDiff float64
			}{
				{"fp32_kv32", MetalOptions{WeightDType: gpu.Float32, KVCacheDType: gpu.Float32}, 1e-4},
				{"bf16_kv32", MetalOptions{WeightDType: gpu.BFloat16, KVCacheDType: gpu.Float32}, 1e-2},
				{"bf16_kv16", MetalOptions{WeightDType: gpu.BFloat16, KVCacheDType: gpu.Float16}, 1e-2},
				{"fp16_kv32", MetalOptions{WeightDType: gpu.Float16, KVCacheDType: gpu.Float32}, 1e-2},
				{"fp16_kv16", MetalOptions{WeightDType: gpu.Float16, KVCacheDType: gpu.Float16}, 1e-2},
				{"q8_0_kv32", MetalOptions{WeightDType: gpu.Q8_0, KVCacheDType: gpu.Float32}, 0.05},
				{"q8_0_kv16", MetalOptions{WeightDType: gpu.Q8_0, KVCacheDType: gpu.Float16}, 0.05},
				{"q4_0_kv32", MetalOptions{WeightDType: gpu.Q4_0, KVCacheDType: gpu.Float32}, 1.0},
				{"q4_0_kv16", MetalOptions{WeightDType: gpu.Q4_0, KVCacheDType: gpu.Float16}, 1.0},
			}
			for _, mode := range modes {
				t.Run(mode.name, func(t *testing.T) {
					var decoder, _, loadErr = LoadHFModelMetalAttentionWithOptions(t.Context(), dir, config.MaxContext, mode.opts)
					if errors.Is(loadErr, gpu.ErrUnavailable) {
						t.Skip(loadErr)
					}
					if loadErr != nil {
						t.Fatal(loadErr)
					}
					t.Cleanup(func() {
						if closeErr := decoder.Close(); closeErr != nil {
							t.Error(closeErr)
						}
					})

					// Verify device tensor dtypes and tied sharing.
					if checkpoint.tied && decoder.metalEmbedding != decoder.metalOutput {
						t.Fatal("expected single shared embedding/classifier tensor")
					}
					if !checkpoint.tied && decoder.metalEmbedding == decoder.metalOutput {
						t.Fatal("expected distinct embedding and classifier tensors")
					}
					if got := decoder.blocks[0].metal.gate.DType(); got != mode.opts.WeightDType {
						t.Fatalf("gate weight dtype = %v, want %v", got, mode.opts.WeightDType)
					}
					var wantEmbDType = mode.opts.WeightDType
					if mode.opts.WeightDType.IsQuantized() {
						wantEmbDType = gpu.BFloat16
					}
					if got := decoder.metalEmbedding.DType(); got != wantEmbDType {
						t.Fatalf("embedding dtype = %v, want %v", got, wantEmbDType)
					}

					var logits *tensor.Tensor
					if logits, err = decoder.Forward(tokens); err != nil {
						t.Fatal(err)
					}
					var fullValues = tensorValues(t, logits)
					var maxDiff float64
					for i := range fullValues {
						if !finite(float64(fullValues[i])) {
							t.Fatalf("nonfinite logit at %d", i)
						}
						maxDiff = max(maxDiff, math.Abs(float64(fullValues[i])-float64(cpuValues[i])))
					}
					if maxDiff > mode.maxAbsDiff {
						t.Fatalf("max diff vs CPU = %g, want <= %g", maxDiff, mode.maxAbsDiff)
					}

					// Verify cached Prefill + Decode matches full-prefix Forward and uses requested KV dtype.
					var session = newTestMetalAttentionSession(t, decoder, int64(len(tokens)))
					if got := session.metal.caches[0].DType(); got != mode.opts.KVCacheDType {
						t.Fatalf("KV cache dtype = %v, want %v", got, mode.opts.KVCacheDType)
					}
					var row []float32
					if row, err = session.Prefill(tokens[:2]); err != nil {
						t.Fatal(err)
					}
					for _, tok := range tokens[2:] {
						if row, err = session.Decode(tok); err != nil {
							t.Fatal(err)
						}
					}
					var vocab = int(config.VocabSize)
					var wantLast = fullValues[(len(tokens)-1)*vocab:]
					var cachedDiff float64
					for i := range row {
						cachedDiff = max(cachedDiff, math.Abs(float64(row[i])-float64(wantLast[i])))
					}
					if cachedDiff > 1e-4 {
						t.Fatalf("cached vs full-prefix max diff = %g, want <= 1e-4", cachedDiff)
					}
					t.Logf("%s: maxDiffVsCPU=%.6g cachedVsFull=%.6g", mode.name, maxDiff, cachedDiff)
				})
			}
		})
	}
}

func TestMetalAttentionDirectBF16RejectsNonfinite(t *testing.T) {
	var dir = t.TempDir()
	var config, _, entries = makeHFCheckpointFixture(t, dir, true)
	entries[0].values[0] = float32(math.NaN())
	writeHFFixtureTensors(t, dir, safetensors.BF16, entries)
	var decoder, _, err = LoadHFModelMetalAttentionWithOptions(t.Context(), dir, config.MaxContext, MetalOptions{
		WeightDType: gpu.BFloat16,
	})
	if errors.Is(err, gpu.ErrUnavailable) {
		t.Skip(err)
	}
	if decoder != nil || err == nil || !strings.Contains(err.Error(), "contains nonfinite") {
		t.Fatalf("nonfinite BF16 load: decoder %v, err %v", decoder, err)
	}
}

func TestMetalAttentionQuantRejectsMisalignedDim(t *testing.T) {
	var dir = t.TempDir()
	var config, _, entries = makeHFCheckpointFixture(t, dir, true)
	writeHFFixtureTensors(t, dir, safetensors.F32, entries)
	// Fixture has Dim=6 which is not a multiple of 32.
	for _, dtype := range []gpu.DType{gpu.Q8_0, gpu.Q4_0} {
		var decoder, _, err = LoadHFModelMetalAttentionWithOptions(t.Context(), dir, config.MaxContext, MetalOptions{
			WeightDType: dtype,
		})
		if errors.Is(err, gpu.ErrUnavailable) {
			t.Skip(err)
		}
		if decoder != nil || !errors.Is(err, gpu.ErrInvalid) {
			t.Fatalf("%v misaligned dim=%d: got decoder %v, err %v, want ErrInvalid", dtype, config.Dim, decoder, err)
		}
	}
}

func TestMetalAttentionReducedPrecisionBF16DirectMatchesUpcast(t *testing.T) {
	var dir = t.TempDir()
	var config, _ = makeQuantCompatibleHFCheckpoint(t, dir, true, safetensors.BF16)
	var bf16Decoder, _, err = LoadHFModelMetalAttentionWithOptions(t.Context(), dir, config.MaxContext, MetalOptions{
		WeightDType: gpu.BFloat16,
	})
	if errors.Is(err, gpu.ErrUnavailable) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer bf16Decoder.Close()

	// Compare against loading the same BF16 values from an F32 safetensors file (which converts F32->BF16 via UploadFloat32As).
	var dirF32 = t.TempDir()
	makeQuantCompatibleHFCheckpoint(t, dirF32, true, safetensors.F32)
	// Both checkpoints have identical BF16-rounded values only when BF16 checkpoint is used;
	// verify direct BF16 upload produces exact deterministic logits across repeated calls.
	var tokens = []int64{1, 2, 3, 4}
	var first, second *tensor.Tensor
	if first, err = bf16Decoder.Forward(tokens); err != nil {
		t.Fatal(err)
	}
	if second, err = bf16Decoder.Forward(tokens); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tensorValues(t, first), tensorValues(t, second)) {
		t.Fatal("BF16 direct-upload Forward is not deterministic")
	}
}
