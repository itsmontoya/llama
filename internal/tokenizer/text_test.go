package tokenizer

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestEncode(t *testing.T) {
	const (
		space = byteFallbackStart + byteFallbackCount + iota
		a
		b
		c
		z
		eAcute
		spaceA
		twoSpaces
		ab
		bc
		aa
		aaaa
	)
	entries := append(testEntries()[:space],
		entry{piece: " "},
		entry{piece: "a"},
		entry{piece: "b"},
		entry{piece: "c"},
		entry{piece: "z"},
		entry{piece: "é"},
		entry{piece: " a", score: 1},
		entry{piece: "  ", score: 4},
		entry{piece: "ab", score: 2},
		entry{piece: "bc", score: 2},
		entry{piece: "aa", score: 5},
		entry{piece: "aaaa", score: 6},
		entry{piece: "zz", score: -1e10},
		entry{piece: "ac", score: -1e11},
	)
	vocab, err := Load(writeVocabulary(t, packVocabulary(entries)), int64(len(entries)))
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name   string
		prompt string
		want   []int64
	}{
		{"empty", "", []int64{BOS}},
		{"dummy space", "a", []int64{BOS, spaceA}},
		{"leading space", " a", []int64{BOS, twoSpaces, a}},
		{"equal scores favor the first pair", "abc", []int64{BOS, space, ab, c}},
		{"another pair with the same score", "bc", []int64{BOS, space, bc}},
		{"highest score wins", "aab", []int64{BOS, space, aa, b}},
		{"repeated merges", "aaaa", []int64{BOS, space, aaaa}},
		{"score at threshold", "zz", []int64{BOS, space, z, z}},
		{"score below threshold", "ac", []int64{BOS, spaceA, c}},
		{"exact code point and byte fallback", "é🦙", []int64{
			BOS, space, eAcute, byteFallbackStart + 0xf0, byteFallbackStart + 0x9f,
			byteFallbackStart + 0xa6, byteFallbackStart + 0x99,
		}},
		{"valid replacement rune", "\ufffd", []int64{
			BOS, space, byteFallbackStart + 0xef, byteFallbackStart + 0xbf, byteFallbackStart + 0xbd,
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := vocab.Encode(test.prompt)
			if err != nil || !slices.Equal(got, test.want) {
				t.Fatalf("Encode(%q) = (%v, %v), want (%v, nil)", test.prompt, got, err, test.want)
			}
		})
	}
	for _, prompt := range []string{"a\xff", "\xf0\x9f", "a\x00b"} {
		if ids, err := vocab.Encode(prompt); ids != nil || err == nil {
			t.Errorf("Encode(%q) = (%v, %v), want nil and an error", prompt, ids, err)
		}
	}
}

func TestDecode(t *testing.T) {
	entries := testEntries()
	vocab, err := Load(writeVocabulary(t, packVocabulary(entries)), int64(len(entries)))
	if err != nil {
		t.Fatal(err)
	}
	const (
		space     = byteFallbackStart + byteFallbackCount
		spaceA    = space + 1
		twoSpaces = space + 2
	)
	for _, test := range []struct {
		name string
		prev int64
		ids  []int64
		want string
	}{
		{"empty", BOS, nil, ""},
		{"strip only after BOS", BOS, []int64{spaceA, spaceA}, "a a"},
		{"keep space after another token", EOS, []int64{spaceA}, " a"},
		{"strip only one space", BOS, []int64{twoSpaces}, " "},
		{"update previous ID after empty piece", BOS, []int64{space, spaceA}, " a"},
		{"decode special IDs and update BOS", EOS, []int64{BOS, spaceA, EOS}, "\n<s>\na\n</s>\n"},
		{"assemble raw bytes", BOS, []int64{
			byteFallbackStart + ' ', byteFallbackStart, byteFallbackStart + 0xf0,
			byteFallbackStart + 0x9f, byteFallbackStart + 0xa6, byteFallbackStart + 0x99,
			byteFallbackStart + 0xff,
		}, " \x00🦙\xff"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := vocab.Decode(test.prev, test.ids)
			if err != nil || got != test.want {
				t.Fatalf("Decode(%d, %v) = (% x, %v), want (% x, nil)", test.prev, test.ids, got, err, test.want)
			}
		})
	}
	for _, id := range []int64{-1, int64(len(entries))} {
		if got, err := vocab.Decode(id, nil); got != "" || err == nil {
			t.Errorf("Decode(%d, nil) = (%q, %v), want an empty string and an error", id, got, err)
		}
		ids := []int64{spaceA, id}
		if got, err := vocab.Decode(BOS, ids); got != "" || err == nil {
			t.Errorf("Decode(%d, %v) = (%q, %v), want an empty string and an error", BOS, ids, got, err)
		}
	}
}

func TestStories260KTokenizerParity(t *testing.T) {
	vocab := loadStories260KTokenizer(t)
	path := filepath.Join("..", "..", "testdata", "stories260k_tokenizer.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name       string  `json:"name"`
		PromptHex  string  `json:"prompt_hex"`
		PromptIDs  []int64 `json:"prompt_ids"`
		DecodedHex string  `json:"decoded_hex"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) != 4 {
		t.Fatalf("loaded %d tokenizer reference cases, want 4", len(cases))
	}
	for _, test := range cases {
		t.Run(test.Name, func(t *testing.T) {
			prompt, err := hex.DecodeString(test.PromptHex)
			if err != nil {
				t.Fatal(err)
			}
			if len(test.PromptIDs) == 0 || test.PromptIDs[0] != BOS {
				t.Fatalf("reference IDs %v must begin with BOS", test.PromptIDs)
			}
			ids, err := vocab.Encode(string(prompt))
			if err != nil || !slices.Equal(ids, test.PromptIDs) {
				t.Errorf("Encode(%q) = (%v, %v), want (%v, nil)", prompt, ids, err, test.PromptIDs)
			}
			decoded, err := vocab.Decode(test.PromptIDs[0], test.PromptIDs[1:])
			if err != nil || hex.EncodeToString([]byte(decoded)) != test.DecodedHex {
				t.Errorf("Decode(%d, %v) = (%x, %v), want (%s, nil)", test.PromptIDs[0], test.PromptIDs[1:], decoded, err, test.DecodedHex)
			}
		})
	}
}
