package tokenizer

import (
	"slices"
	"testing"
)

func TestLlama3MergeRankAndTie(t *testing.T) {
	var tokenizer Llama3Tokenizer
	tokenizer.byteIDs['a'], tokenizer.byteIDs['b'], tokenizer.byteIDs['c'] = 1, 2, 3
	tokenizer.baseIDs = map[string]int64{"known": 9}
	tokenizer.merges = map[llama3Pair]llama3Merge{
		{2, 3}: {4, 0}, // bc is preferred to ab.
		{1, 2}: {5, 1},
		{1, 1}: {6, 2},
		{1, 4}: {7, 3},
	}
	for _, test := range []struct {
		text string
		want []int64
	}{
		{"abc", []int64{7}},
		{"aaa", []int64{6, 1}}, // The two occurrences of (a, a) overlap.
		{"aaab", []int64{6, 5}},
		{"aaaa", []int64{6, 6}},
		{"known", []int64{9}},
	} {
		if got := tokenizer.appendBPE(nil, test.text); !slices.Equal(got, test.want) {
			t.Errorf("BPE for %q: got %v, want %v", test.text, got, test.want)
		}
	}
}

func TestLlama3AddedTokenMatching(t *testing.T) {
	var tokenizer Llama3Tokenizer
	tokenizer.bosID = 256
	tokenizer.added.add("<special>", 257)
	tokenizer.added.add("<special>long", 258)
	for i := range tokenizer.byteIDs {
		tokenizer.byteIDs[i] = int64(i)
	}
	ids, err := tokenizer.Encode("a<special>long<special>b")
	if want := []int64{256, 'a', 258, 257, 'b'}; err != nil || !slices.Equal(ids, want) {
		t.Fatalf("Encode: got %v, %v, want %v", ids, err, want)
	}
	if _, err := tokenizer.Encode("\xFF"); err == nil {
		t.Fatal("Encode accepted an invalid UTF-8 prompt")
	}
	ids[0] = 0
	if empty, err := tokenizer.Encode(""); err != nil || !slices.Equal(empty, []int64{256}) {
		t.Fatalf("Encode empty: got %v, %v, want BOS", empty, err)
	}
}

func TestLlama3DecodeBytes(t *testing.T) {
	tokenizer := Llama3Tokenizer{pieces: []string{" ", "\xE2", "\x82", "\xAC", "", "\xFF", "!", "\xF0\x9F", "\x91", "\xA9", "\xED\xA0\x80", " ."}}
	for _, test := range []struct {
		ids  []int64
		want string
	}{
		{[]int64{0, 1, 4, 2, 3, 6, 11}, " €! ."},
		{[]int64{7, 4, 8, 9}, "👩"},
		{[]int64{1, 2, 1, 6, 7, 8}, "��!�"},
		{[]int64{5, 5, 10}, "�����"},
		{[]int64{4}, ""},
	} {
		if got, err := tokenizer.Decode(test.ids); err != nil || got != test.want {
			t.Errorf("Decode %v: got %q, %v, want %q", test.ids, got, err, test.want)
		}
	}
	for _, invalid := range [][]int64{{-1}, {0, int64(len(tokenizer.pieces))}} {
		if got, err := tokenizer.Decode(invalid); err == nil || got != "" {
			t.Errorf("Decode accepted invalid IDs %v: %q, %v", invalid, got, err)
		}
	}
}
