package llama

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/itsmontoya/llama/internal/safetensors"
	"github.com/itsmontoya/neuralnetwork/gpu"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

const hfCheckpointName = "model.safetensors"

// LoadHFModel loads a dense Llama model from one safetensors checkpoint and
// returns its completion settings. A tied model must omit lm_head.weight.
func LoadHFModel(dir string, maxContext int64) (*Decoder, HFGenerationConfig, error) {
	return loadHFModel(context.Background(), dir, maxContext, nil)
}

// LoadHFModelMetal loads a Hugging Face model with its FFN weights resident
// on the system's default Metal device. The remaining decoder uses the
// regular tensor path. Call Decoder.Close to release Metal resources.
// The context controls loading, including cleanup after a canceled upload
// boundary. Unavailable builds or devices return an error wrapping
// gpu.ErrUnavailable.
func LoadHFModelMetal(ctx context.Context, dir string, maxContext int64) (*Decoder, HFGenerationConfig, error) {
	return loadHFModelMetal(ctx, dir, maxContext, false, MetalOptions{WeightDType: gpu.Float32, KVCacheDType: gpu.Float32})
}

// LoadHFModelMetalAttention loads a Hugging Face model for resident FP32
// inference on one Metal device, from embedding lookup through classification.
// Each session keeps its own device KV cache. Forward returns every logit row.
// Prefill and Decode download only the final row. Call Decoder.Close to release
// all model and session resources. The context controls loading. Unavailable
// builds or devices return an error wrapping gpu.ErrUnavailable.
func LoadHFModelMetalAttention(ctx context.Context, dir string, maxContext int64) (decoder *Decoder, generation HFGenerationConfig, err error) {
	return LoadHFModelMetalAttentionWithOptions(ctx, dir, maxContext, MetalOptions{})
}

// LoadHFModelMetalAttentionWithOptions loads a Hugging Face model for resident
// Metal inference using the requested weight and KV cache storage dtypes.
// Supported WeightDType values are 0 (Float32), Float32, Float16, BFloat16,
// Q8_0, and Q4_0. Supported KVCacheDType values are 0 (Float32), Float32, and
// Float16. Native BF16 checkpoints upload directly to BFloat16 device tensors
// without FP32 upcast. In quantized weight modes (Q8_0/Q4_0), transformer
// projections and embeddings (including tied embedding/classifier tables) use
// the quantized dtype while norms remain Float32. Unavailable builds or devices
// return an error wrapping gpu.ErrUnavailable; invalid options return an error
// wrapping gpu.ErrInvalid.
func LoadHFModelMetalAttentionWithOptions(ctx context.Context, dir string, maxContext int64, options MetalOptions) (decoder *Decoder, generation HFGenerationConfig, err error) {
	var resolved MetalOptions
	if resolved, err = options.resolve(); err != nil {
		return nil, HFGenerationConfig{}, err
	}
	return loadHFModelMetal(ctx, dir, maxContext, true, resolved)
}

func loadHFModelMetal(ctx context.Context, dir string, maxContext int64, attention bool, options MetalOptions) (decoder *Decoder, generation HFGenerationConfig, err error) {
	if err = ctx.Err(); err != nil {
		return nil, HFGenerationConfig{}, err
	}
	var device *gpu.Device
	if device, err = gpu.OpenDefault(); err != nil {
		var operation = "FFN"
		if attention {
			operation = "attention"
		}
		return nil, HFGenerationConfig{}, fmt.Errorf("llama: Metal %s requires a Metal device on darwin/arm64 and a cgo build with -tags=metal (without purego): %w", operation, err)
	}
	var model = metalModel{
		device:       device,
		attention:    attention,
		weightDType:  options.WeightDType,
		kvCacheDType: options.KVCacheDType,
	}
	decoder, generation, err = loadHFModel(ctx, dir, maxContext, &model)
	if err != nil {
		if closeErr := model.close(); closeErr != nil {
			err = errors.Join(err, closeErr)
		}
		return nil, HFGenerationConfig{}, err
	}
	return decoder, generation, nil
}

func loadHFModel(ctx context.Context, dir string, maxContext int64, metal *metalModel) (decoder *Decoder, generation HFGenerationConfig, err error) {
	checkpoint, err := openHFCheckpoint(dir)
	if err != nil {
		return nil, HFGenerationConfig{}, err
	}
	defer checkpoint.Close()

	model, err := readHFConfig(dir, maxContext)
	if err != nil {
		return nil, HFGenerationConfig{}, err
	}
	decoder, err = loadHFCheckpoint(ctx, model.decoder, checkpoint, metal)
	if err != nil {
		return nil, HFGenerationConfig{}, fmt.Errorf("llama: checkpoint %q: %w", filepath.Join(dir, hfCheckpointName), err)
	}
	return decoder, model.generation, nil
}

func openHFCheckpoint(dir string) (*safetensors.File, error) {
	checkpoint, err := safetensors.Open(filepath.Join(dir, hfCheckpointName))
	if !errors.Is(err, os.ErrNotExist) {
		return checkpoint, err
	}
	entries, readErr := os.ReadDir(dir)
	if readErr != nil {
		return nil, err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if name == "model.safetensors.index.json" ||
			(strings.HasPrefix(name, "model-") && strings.Contains(name, "-of-") && strings.HasSuffix(name, ".safetensors")) {
			return nil, fmt.Errorf("llama: %q: sharded safetensors checkpoints are unsupported (expected model.safetensors)", dir)
		}
	}
	return nil, err
}

func loadHFCheckpoint(ctx context.Context, config Config, checkpoint *safetensors.File, metal *metalModel) (out *Decoder, err error) {
	inventory := checkpoint.Tensors()
	// Bound the layout allocation by the available descriptors before walking
	// dimensions derived from the configuration.
	if config.Layers > int64(len(inventory)) {
		return nil, fmt.Errorf("%d layers cannot be represented by only %d tensor descriptors", config.Layers, len(inventory))
	}
	decoder := &Decoder{config: config, blocks: make([]blockWeights, config.Layers), metal: metal}
	var descriptors = checkpoint.Tensors()
	if err = decoder.validateHFInventory(inventory); err != nil {
		return nil, err
	}
	headWidth := config.Dim / config.Heads
	if metal != nil && metal.attention && metal.weightDType != 0 && metal.weightDType != gpu.Float32 {
		if err = decoder.loadHFMetalWeights(ctx, checkpoint, descriptors, headWidth); err != nil {
			return nil, err
		}
	} else if err = decoder.loadWeights(ctx, Weights{}, func(layer int, weight decoderWeight) (string, []float32, error) {
		name := hfWeightName(layer, weight.name)
		values, readErr := checkpoint.ReadFloat32(name)
		if readErr != nil {
			return name, nil, readErr
		}
		switch weight.name {
		case "wq":
			values = interleaveRotaryRows(values, config.Heads, headWidth, config.Dim)
		case "wk":
			values = interleaveRotaryRows(values, config.KVHeads, headWidth, config.Dim)
		}
		return name, values, nil
	}); err != nil {
		return nil, err
	}
	if metal != nil && metal.attention {
		if err = metal.loadRotary(ctx, config); err != nil {
			return nil, err
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return decoder, nil
}

// validateHFInventory consumes a private copy of the descriptors so it can
// report missing, mismatched, and extra tensors before reading any payload.
func (d *Decoder) validateHFInventory(inventory map[string]safetensors.Tensor) error {
	if _, exists := inventory["lm_head.weight"]; d.config.TieEmbeddings && exists {
		return fmt.Errorf("tie_word_embeddings requires lm_head.weight to be omitted")
	}
	var payloadSize int64
	if err := d.visitWeights(Weights{}, func(layer int, weight decoderWeight) error {
		name := hfWeightName(layer, weight.name)
		descriptor, ok := inventory[name]
		if !ok {
			return fmt.Errorf("missing tensor %q", name)
		}
		if !slices.Equal(descriptor.Shape, weight.shape) {
			return fmt.Errorf("tensor %q has shape %v, want %v", name, descriptor.Shape, weight.shape)
		}
		_, size, err := tensor.ValidateShape(weight.shape, tensor.Float32)
		if err != nil {
			return fmt.Errorf("tensor %q FP32 size: %w", name, err)
		}
		if size > math.MaxInt64-payloadSize {
			return fmt.Errorf("total FP32 weight payload overflows int64 at tensor %q", name)
		}
		payloadSize += size
		delete(inventory, name)
		return nil
	}); err != nil {
		return err
	}
	if len(inventory) > 0 {
		return fmt.Errorf("unsupported checkpoint tensors: %q", slices.Sorted(maps.Keys(inventory)))
	}
	return nil
}

var hfWeightNames = map[string]string{
	"token_embedding": "model.embed_tokens.weight",
	"final_norm":      "model.norm.weight",
	"output":          "lm_head.weight",
	"attention_norm":  "input_layernorm.weight",
	"wq":              "self_attn.q_proj.weight",
	"wk":              "self_attn.k_proj.weight",
	"wv":              "self_attn.v_proj.weight",
	"wo":              "self_attn.o_proj.weight",
	"ffn_norm":        "post_attention_layernorm.weight",
	"w_gate":          "mlp.gate_proj.weight",
	"w_up":            "mlp.up_proj.weight",
	"w_down":          "mlp.down_proj.weight",
}

func hfWeightName(layer int, weight string) string {
	name := hfWeightNames[weight]
	if layer >= 0 {
		name = fmt.Sprintf("model.layers.%d.%s", layer, name)
	}
	return name
}

// interleaveRotaryRows converts the two split halves of each Q or K head to
// adjacent rows. Its output has the same [headCount*headWidth, inputWidth]
// shape. The validated checkpoint supplies the exact input length.
func interleaveRotaryRows(values []float32, headCount, headWidth, inputWidth int64) []float32 {
	adjacent := make([]float32, len(values))
	half := headWidth / 2
	for head := range headCount {
		base := head * headWidth
		for pair := range half {
			first := (base + pair) * inputWidth
			second := (base + half + pair) * inputWidth
			target := (base + 2*pair) * inputWidth
			copy(adjacent[target:target+inputWidth], values[first:first+inputWidth])
			copy(adjacent[target+inputWidth:target+2*inputWidth], values[second:second+inputWidth])
		}
	}
	return adjacent
}

func interleaveRotaryRowsUint16(values []uint16, headCount, headWidth, inputWidth int64) []uint16 {
	adjacent := make([]uint16, len(values))
	half := headWidth / 2
	for head := range headCount {
		base := head * headWidth
		for pair := range half {
			first := (base + pair) * inputWidth
			second := (base + half + pair) * inputWidth
			target := (base + 2*pair) * inputWidth
			copy(adjacent[target:target+inputWidth], values[first:first+inputWidth])
			copy(adjacent[target+inputWidth:target+2*inputWidth], values[second:second+inputWidth])
		}
	}
	return adjacent
}

func (d *Decoder) loadHFMetalWeights(ctx context.Context, checkpoint *safetensors.File, descriptors map[string]safetensors.Tensor, headWidth int64) (err error) {
	if err = d.visitWeights(Weights{}, func(layer int, weight decoderWeight) (readErr error) {
		if readErr = ctx.Err(); readErr != nil {
			return readErr
		}
		var (
			name        = hfWeightName(layer, weight.name)
			targetDType = d.metal.weightDTypeFor(weight.name)
			descriptor  = descriptors[name]
		)
		if targetDType == gpu.BFloat16 && descriptor.DType == safetensors.BF16 {
			var bits []uint16
			if bits, readErr = checkpoint.ReadBFloat16Bits(name); readErr != nil {
				return readErr
			}
			switch weight.name {
			case "wq":
				bits = interleaveRotaryRowsUint16(bits, d.config.Heads, headWidth, d.config.Dim)
			case "wk":
				bits = interleaveRotaryRowsUint16(bits, d.config.KVHeads, headWidth, d.config.Dim)
			}
			return weight.storeBFloat16Bits(name, bits, d.metal)
		}
		var values []float32
		if values, readErr = checkpoint.ReadFloat32(name); readErr != nil {
			return readErr
		}
		switch weight.name {
		case "wq":
			values = interleaveRotaryRows(values, d.config.Heads, headWidth, d.config.Dim)
		case "wk":
			values = interleaveRotaryRows(values, d.config.KVHeads, headWidth, d.config.Dim)
		}
		return weight.store(name, values, d.metal)
	}); err != nil {
		return err
	}
	if d.config.TieEmbeddings {
		d.metalOutput = d.metalEmbedding
	}
	return nil
}
