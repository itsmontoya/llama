package llama

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// HFGenerationConfig contains the token IDs used for completion.
// StopIDs is owned by the caller.
type HFGenerationConfig struct {
	BOSID   int64
	StopIDs []int64
}

type hfModelConfig struct {
	decoder               Config
	maxPositionEmbeddings int64
	generation            HFGenerationConfig
}

type hfConfigFile struct {
	ModelType           string   `json:"model_type"`
	Architectures       []string `json:"architectures"`
	HiddenSize          int64    `json:"hidden_size"`
	IntermediateSize    int64    `json:"intermediate_size"`
	Layers              int64    `json:"num_hidden_layers"`
	Heads               int64    `json:"num_attention_heads"`
	KVHeads             *int64   `json:"num_key_value_heads"`
	HeadWidth           *int64   `json:"head_dim"`
	VocabSize           int64    `json:"vocab_size"`
	MaxPosition         int64    `json:"max_position_embeddings"`
	NormEpsilon         *float32 `json:"rms_norm_eps"`
	HiddenAct           *string  `json:"hidden_act"`
	AttentionBias       bool     `json:"attention_bias"`
	MLPBias             bool     `json:"mlp_bias"`
	TieEmbeddings       bool     `json:"tie_word_embeddings"`
	PretrainingTP       int64    `json:"pretraining_tp"`
	PartialRotaryFactor float64  `json:"partial_rotary_factor"`
	RopeTheta           *float64 `json:"rope_theta"`
	RopeScaling         struct {
		RopeScaling
		LegacyType string `json:"type"`
	} `json:"rope_scaling"`
	hfGenerationFields
}

type hfGenerationFields struct {
	BOSID  *int64   `json:"bos_token_id"`
	EOSIDs hfEOSIDs `json:"eos_token_id"`
}

type hfEOSIDs []int64

func (ids *hfEOSIDs) UnmarshalJSON(data []byte) error {
	if data[0] == '[' {
		var values []*int64
		if err := json.Unmarshal(data, &values); err != nil {
			return fmt.Errorf("eos_token_id: %w", err)
		}
		parsed := make(hfEOSIDs, len(values))
		for i, id := range values {
			if id == nil {
				return fmt.Errorf("eos_token_id[%d] must be an integer", i)
			}
			parsed[i] = *id
		}
		*ids = parsed
		return nil
	}
	var id *int64
	if err := json.Unmarshal(data, &id); err != nil {
		return fmt.Errorf("eos_token_id: %w", err)
	}
	*ids = nil
	if id != nil {
		*ids = hfEOSIDs{*id}
	}
	return nil
}

func readHFConfig(dir string, maxContext int64) (hfModelConfig, error) {
	// The dimensions required by this reader come from the checkpoint. These
	// remaining defaults match the pinned Transformers LlamaConfig.
	defaultBOS := int64(1)
	defaultEpsilon := float32(1e-6)
	defaultTheta := 10000.0
	defaultActivation := "silu"
	file := hfConfigFile{
		NormEpsilon: &defaultEpsilon, HiddenAct: &defaultActivation, RopeTheta: &defaultTheta,
		PretrainingTP: 1, PartialRotaryFactor: 1,
		hfGenerationFields: hfGenerationFields{BOSID: &defaultBOS, EOSIDs: hfEOSIDs{2}},
	}
	configPath := filepath.Join(dir, "config.json")
	if err := readHFJSON(configPath, &file); err != nil {
		return hfModelConfig{}, err
	}
	decoder, err := file.decoderConfig(maxContext)
	if err != nil {
		return hfModelConfig{}, fmt.Errorf("%s: %w", configPath, err)
	}

	// Transformers uses the entire generation configuration when available.
	// Only an absent file falls back to the model's token settings.
	generationPath := filepath.Join(dir, "generation_config.json")
	var fields hfGenerationFields
	if err := readHFJSON(generationPath, &fields); err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return hfModelConfig{}, err
		}
		generationPath = configPath
		fields = file.hfGenerationFields
	}
	generation, err := fields.resolve(decoder.VocabSize)
	if err != nil {
		return hfModelConfig{}, fmt.Errorf("%s: %w", generationPath, err)
	}
	return hfModelConfig{decoder: decoder, maxPositionEmbeddings: file.MaxPosition, generation: generation}, nil
}

func (f hfConfigFile) decoderConfig(maxContext int64) (Config, error) {
	if f.ModelType != "llama" {
		return Config{}, fmt.Errorf("unsupported model_type %q, want llama", f.ModelType)
	}
	if len(f.Architectures) > 0 && (len(f.Architectures) != 1 || f.Architectures[0] != "LlamaForCausalLM") {
		return Config{}, fmt.Errorf("unsupported architectures %q, want LlamaForCausalLM", f.Architectures)
	}
	if f.HiddenAct == nil || *f.HiddenAct != "silu" {
		return Config{}, fmt.Errorf("hidden_act must be silu")
	}
	if f.AttentionBias || f.MLPBias {
		return Config{}, fmt.Errorf("attention_bias and mlp_bias must both be false")
	}
	if f.PretrainingTP != 1 || f.PartialRotaryFactor != 1 {
		return Config{}, fmt.Errorf("pretraining_tp and partial_rotary_factor must both be 1")
	}
	if f.MaxPosition <= 0 {
		return Config{}, fmt.Errorf("max_position_embeddings must be positive, got %d", f.MaxPosition)
	}
	if maxContext <= 0 || maxContext > f.MaxPosition {
		return Config{}, fmt.Errorf("runtime context must be in [1,%d], got %d", f.MaxPosition, maxContext)
	}
	if f.NormEpsilon == nil || f.RopeTheta == nil {
		return Config{}, fmt.Errorf("rms_norm_eps and rope_theta must be numbers")
	}
	kvHeads := f.Heads
	if f.KVHeads != nil {
		kvHeads = *f.KVHeads
	}
	scaling := f.RopeScaling.RopeScaling
	if f.RopeScaling.LegacyType != "" {
		scaling.Type = f.RopeScaling.LegacyType
	}
	config := Config{
		Dim: f.HiddenSize, FFNDim: f.IntermediateSize, Layers: f.Layers,
		Heads: f.Heads, KVHeads: kvHeads, VocabSize: f.VocabSize, MaxContext: maxContext,
		NormEpsilon: *f.NormEpsilon, RopeTheta: *f.RopeTheta, RopeScaling: scaling, TieEmbeddings: f.TieEmbeddings,
	}
	if err := config.validate(); err != nil {
		return Config{}, err
	}
	if width := config.Dim / config.Heads; f.HeadWidth != nil && *f.HeadWidth != width {
		return Config{}, fmt.Errorf("head_dim is %d, want hidden_size / num_attention_heads = %d", *f.HeadWidth, width)
	}
	return config, nil
}

func (fields hfGenerationFields) resolve(vocabSize int64) (HFGenerationConfig, error) {
	if fields.BOSID == nil {
		return HFGenerationConfig{}, fmt.Errorf("bos_token_id is required")
	}
	if id := *fields.BOSID; id < 0 || id >= vocabSize {
		return HFGenerationConfig{}, fmt.Errorf("bos_token_id %d is outside vocabulary [0,%d)", id, vocabSize)
	}
	if len(fields.EOSIDs) == 0 {
		return HFGenerationConfig{}, fmt.Errorf("at least one eos_token_id is required")
	}
	for _, id := range fields.EOSIDs {
		if id < 0 || id >= vocabSize {
			return HFGenerationConfig{}, fmt.Errorf("eos_token_id %d is outside vocabulary [0,%d)", id, vocabSize)
		}
	}
	return HFGenerationConfig{BOSID: *fields.BOSID, StopIDs: fields.EOSIDs}, nil
}

func readHFJSON(path string, target any) error {
	contents, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(contents, target); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}
