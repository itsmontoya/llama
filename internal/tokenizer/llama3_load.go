package tokenizer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"unicode/utf8"
)

type llama3ModelFile struct {
	Type             string            `json:"type"`
	Dropout          *float64          `json:"dropout"`
	UnknownToken     *string           `json:"unk_token"`
	ContinuingPrefix *string           `json:"continuing_subword_prefix"`
	EndSuffix        *string           `json:"end_of_word_suffix"`
	FuseUnknown      bool              `json:"fuse_unk"`
	ByteFallback     bool              `json:"byte_fallback"`
	IgnoreMerges     bool              `json:"ignore_merges"`
	Vocabulary       map[string]*int64 `json:"vocab"`
	Merges           []string          `json:"merges"`
}

type llama3TokenData struct {
	Content    string `json:"content"`
	LeftStrip  bool   `json:"lstrip"`
	RightStrip bool   `json:"rstrip"`
	Normalized bool   `json:"normalized"`
	SingleWord bool   `json:"single_word"`
	Special    bool   `json:"special"`
}

type llama3AddedToken struct {
	ID *int64 `json:"id"`
	llama3TokenData
}

type llama3TokenizerFile struct {
	Version       string             `json:"version"`
	Truncation    json.RawMessage    `json:"truncation"`
	Padding       json.RawMessage    `json:"padding"`
	AddedTokens   []llama3AddedToken `json:"added_tokens"`
	Normalizer    json.RawMessage    `json:"normalizer"`
	PreTokenizer  json.RawMessage    `json:"pre_tokenizer"`
	PostProcessor json.RawMessage    `json:"post_processor"`
	Decoder       json.RawMessage    `json:"decoder"`
	Model         llama3ModelFile    `json:"model"`
}

type llama3TokenizerSettings struct {
	Class string                     `json:"tokenizer_class"`
	BOS   string                     `json:"bos_token"`
	EOS   string                     `json:"eos_token"`
	Added map[string]llama3TokenData `json:"added_tokens_decoder"`
	// The reference supplies its own context and decoding options. Decode
	// these metadata fields to validate the artifact types.
	MaxLength     int64    `json:"model_max_length"`
	ModelInputs   []string `json:"model_input_names"`
	CleanupSpaces bool     `json:"clean_up_tokenization_spaces"`
}

// LoadLlama3 loads the audited Hugging Face Llama 3 tokenizer components from
// dir and verifies vocabulary and special-token agreement with the decoder.
func LoadLlama3(dir string, vocabSize, bosID int64, stopIDs []int64) (*Llama3Tokenizer, error) {
	var file llama3TokenizerFile
	path := filepath.Join(dir, "tokenizer.json")
	if err := readLlama3JSON(path, &file); err != nil {
		return nil, err
	}
	if err := validateLlama3Tokenizer(file); err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	baseSize := int64(len(file.Model.Vocabulary))
	if vocabSize < baseSize || vocabSize-baseSize != int64(len(file.AddedTokens)) {
		return nil, fmt.Errorf("tokenizer: %s: checkpoint vocabulary size %d disagrees with %d base and %d added tokens", path, vocabSize, baseSize, len(file.AddedTokens))
	}
	tokenizer := &Llama3Tokenizer{pieces: make([]string, vocabSize), bosID: bosID}
	if err := tokenizer.loadBPE(file.Model); err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	added, err := tokenizer.loadAdded(file.AddedTokens, baseSize)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	if err := validateLlama3Settings(dir, added, bosID, stopIDs); err != nil {
		return nil, err
	}
	postProcessor := llama3PostProcessor(added[bosID].Content, bosID)
	if !llama3JSONEqual(file.PostProcessor, postProcessor) {
		return nil, fmt.Errorf("tokenizer: %s: unsupported post_processor, want the Llama 3 BOS template", path)
	}
	return tokenizer, nil
}

func validateLlama3Tokenizer(file llama3TokenizerFile) error {
	if file.Version != "1.0" {
		return fmt.Errorf("unsupported version %q", file.Version)
	}
	for _, part := range []struct {
		name string
		data json.RawMessage
	}{
		{"normalizer", file.Normalizer}, {"truncation", file.Truncation}, {"padding", file.Padding},
	} {
		if len(part.data) != 0 && !bytes.Equal(part.data, []byte("null")) {
			return fmt.Errorf("unsupported %s", part.name)
		}
	}
	if !llama3JSONEqual(file.PreTokenizer, llama3PreTokenizer()) {
		return fmt.Errorf("unsupported pre_tokenizer, want the pinned Llama 3 Split and ByteLevel stages")
	}
	if !llama3JSONEqual(file.Decoder, llama3DecoderJSON) {
		return fmt.Errorf("unsupported decoder, want ByteLevel")
	}
	model := file.Model
	if model.Type != "BPE" || !model.IgnoreMerges || model.ByteFallback || model.FuseUnknown || model.Dropout != nil ||
		model.UnknownToken != nil || model.ContinuingPrefix != nil || model.EndSuffix != nil || model.Merges == nil {
		return fmt.Errorf("unsupported BPE model options")
	}
	return nil
}

const llama3DecoderJSON = `{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}`

func llama3PreTokenizer() string {
	// JSON quoting preserves the regex backslashes in the expected Split stage.
	pattern, _ := json.Marshal(llama3Pattern)
	return fmt.Sprintf(`{"type":"Sequence","pretokenizers":[
		{"type":"Split","behavior":"Isolated","invert":false,"pattern":{"Regex":%s}},
		{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
	]}`, pattern)
}

func llama3PostProcessor(bos string, id int64) string {
	quotedBOS, _ := json.Marshal(bos)
	return fmt.Sprintf(`{"type":"Sequence","processors":[
		{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true},
		{"type":"TemplateProcessing",
		 "single":[{"SpecialToken":{"id":%[1]s,"type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
		 "pair":[{"SpecialToken":{"id":%[1]s,"type_id":0}},{"Sequence":{"id":"A","type_id":0}},
		         {"SpecialToken":{"id":%[1]s,"type_id":1}},{"Sequence":{"id":"B","type_id":1}}],
		 "special_tokens":{%[1]s:{"id":%[1]s,"ids":[%[2]d],"tokens":[%[1]s]}}}
	]}`, quotedBOS, id)
}

func llama3JSONEqual(got json.RawMessage, want string) bool {
	var left, right any
	actual := json.NewDecoder(bytes.NewReader(got))
	expected := json.NewDecoder(strings.NewReader(want))
	actual.UseNumber()
	expected.UseNumber()
	return actual.Decode(&left) == nil && expected.Decode(&right) == nil && reflect.DeepEqual(left, right)
}

func llama3ByteAlphabet() [256]rune {
	var alphabet [256]rune
	next := rune(256)
	for i := range alphabet {
		value := rune(i)
		if '!' <= value && value <= '~' || 0xA1 <= value && value <= 0xAC || 0xAE <= value {
			alphabet[i] = value
			continue
		}
		alphabet[i] = next
		next++
	}
	return alphabet
}

func (t *Llama3Tokenizer) loadBPE(model llama3ModelFile) error {
	alphabet := llama3ByteAlphabet()
	reverse := make(map[rune]byte, len(alphabet))
	for value, letter := range alphabet {
		reverse[letter] = byte(value)
	}
	t.baseIDs = make(map[string]int64, len(model.Vocabulary))
	for piece, id := range model.Vocabulary {
		if piece == "" || id == nil || *id < 0 || *id >= int64(len(model.Vocabulary)) {
			return fmt.Errorf("empty piece or invalid base vocabulary ID for %q", piece)
		}
		if t.pieces[*id] != "" {
			return fmt.Errorf("duplicate base vocabulary ID %d for %q", *id, piece)
		}
		var raw strings.Builder
		for _, letter := range piece {
			value, ok := reverse[letter]
			if !ok {
				return fmt.Errorf("base piece %q contains %U outside the ByteLevel alphabet", piece, letter)
			}
			raw.WriteByte(value)
		}
		t.pieces[*id] = raw.String()
		t.baseIDs[raw.String()] = *id
	}
	for value := range t.byteIDs {
		id, ok := t.baseIDs[string([]byte{byte(value)})]
		if !ok {
			return fmt.Errorf("missing base token for byte 0x%02X", value)
		}
		t.byteIDs[value] = id
	}
	t.merges = make(map[llama3Pair]llama3Merge, len(model.Merges))
	for rank, line := range model.Merges {
		left, right, ok := strings.Cut(line, " ")
		if !ok || left == "" || right == "" {
			return fmt.Errorf("merge %d is not a pair: %q", rank, line)
		}
		leftID, rightID, mergedID := model.Vocabulary[left], model.Vocabulary[right], model.Vocabulary[left+right]
		if leftID == nil || rightID == nil || mergedID == nil {
			return fmt.Errorf("merge %d references a missing vocabulary piece: %q", rank, line)
		}
		pair := llama3Pair{*leftID, *rightID}
		if _, exists := t.merges[pair]; exists {
			return fmt.Errorf("duplicate merge %d: %q", rank, line)
		}
		t.merges[pair] = llama3Merge{id: *mergedID, rank: rank}
	}
	return nil
}

func (t *Llama3Tokenizer) loadAdded(tokens []llama3AddedToken, baseSize int64) (map[int64]llama3TokenData, error) {
	added := make(map[int64]llama3TokenData, len(tokens))
	for _, token := range tokens {
		if token.ID == nil || *token.ID < baseSize || *token.ID >= int64(len(t.pieces)) || token.Content == "" {
			return nil, fmt.Errorf("empty content or invalid added token ID for %q", token.Content)
		}
		if !token.Special || token.Normalized || token.LeftStrip || token.RightStrip || token.SingleWord {
			return nil, fmt.Errorf("unsupported added token flags for %q", token.Content)
		}
		if _, exists := added[*token.ID]; exists {
			return nil, fmt.Errorf("duplicate added token ID %d", *token.ID)
		}
		_, basePiece := t.baseIDs[token.Content]
		_, addedLength := t.added.match(token.Content)
		if basePiece || addedLength == len(token.Content) {
			return nil, fmt.Errorf("duplicate added token content %q", token.Content)
		}
		added[*token.ID] = token.llama3TokenData
		t.added.add(token.Content, *token.ID)
	}
	return added, nil
}

func validateLlama3Settings(dir string, added map[int64]llama3TokenData, bosID int64, stopIDs []int64) error {
	var settings llama3TokenizerSettings
	path := filepath.Join(dir, "tokenizer_config.json")
	if err := readLlama3JSON(path, &settings); err != nil {
		return err
	}
	if settings.Class != "PreTrainedTokenizerFast" || len(settings.Added) != len(added) {
		return fmt.Errorf("tokenizer: %s: class or added_tokens_decoder disagrees with the pinned pipeline", path)
	}
	for id, token := range added {
		if settings.Added[strconv.FormatInt(id, 10)] != token {
			return fmt.Errorf("tokenizer: %s: added_tokens_decoder disagrees for ID %d", path, id)
		}
	}
	if settings.BOS == "" || added[bosID].Content != settings.BOS {
		return fmt.Errorf("tokenizer: %s: bos_token disagrees with checkpoint BOS ID %d", path, bosID)
	}
	var eosIsStop bool
	for _, id := range stopIDs {
		token, ok := added[id]
		if !ok {
			return fmt.Errorf("tokenizer: %s: checkpoint stop ID %d is not an added special token", path, id)
		}
		eosIsStop = eosIsStop || token.Content == settings.EOS
	}
	if !eosIsStop {
		return fmt.Errorf("tokenizer: %s: eos_token does not match any checkpoint stop ID", path)
	}
	path = filepath.Join(dir, "special_tokens_map.json")
	var special map[string]llama3TokenData
	if err := readLlama3JSON(path, &special); err != nil {
		return err
	}
	if len(special) != 2 || special["bos_token"] != (llama3TokenData{Content: settings.BOS}) ||
		special["eos_token"] != (llama3TokenData{Content: settings.EOS}) {
		return fmt.Errorf("tokenizer: %s: special tokens disagree with tokenizer_config.json", path)
	}
	return nil
}

func readLlama3JSON(path string, out any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	if !utf8.Valid(data) {
		return fmt.Errorf("tokenizer: %s: invalid UTF-8", path)
	}
	// Detect duplicates before decoding into Go maps or structs, which would
	// otherwise silently replace earlier values.
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := readUniqueLlama3Value(decoder); err != nil {
		return fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	if _, err := decoder.Token(); err != io.EOF {
		return fmt.Errorf("tokenizer: %s: unexpected trailing JSON", path)
	}
	decoder = json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(out); err != nil {
		return fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	return nil
}

func readUniqueLlama3Value(decoder *json.Decoder) error {
	value, err := decoder.Token()
	if err != nil {
		return err
	}
	switch value {
	case json.Delim('{'):
		seen := make(map[string]struct{})
		for decoder.More() {
			key, err := decoder.Token()
			if err != nil {
				return err
			}
			name := key.(string) // Token validates object key syntax.
			if _, exists := seen[name]; exists {
				return fmt.Errorf("duplicate JSON key %q", name)
			}
			seen[name] = struct{}{}
			if err := readUniqueLlama3Value(decoder); err != nil {
				return err
			}
		}
	case json.Delim('['):
		for decoder.More() {
			if err := readUniqueLlama3Value(decoder); err != nil {
				return err
			}
		}
	default:
		return nil
	}
	_, err = decoder.Token()
	return err
}
