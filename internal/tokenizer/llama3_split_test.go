package tokenizer

import (
	"slices"
	"strings"
	"testing"
)

func TestLlama3PreTokenization(t *testing.T) {
	// The expected boundaries come from the pinned backend pre_tokenize_str,
	// reversing its ByteLevel mapping to show the original segments.
	for _, want := range [][]string{
		{"I", "'m", "ore", " YOU", "'RE", " she", "'ll", " wouldn", "'t"},
		{"I", "'ſ", "aid", " I", "'ﬅI"},
		{"a", "   ", "\t", "42²", "١３4", "5"},
		{"x", "   \n\r \t\n", " ", " y"},
		{"\t\n", "\t ", " !", "\u00a0\u0085"},
		{"\u0085", "\u0085x", "!\r\n", "\x00a", " 😀😀", "x"},
		{"Café", " Cafe", "\u0301", " 北京", " مرحبا"},
		{"A", "\U00011F02A", " \U00010D40", "1"},
	} {
		input := strings.Join(want, "")
		rest := input
		var got []string
		for rest != "" {
			end := llama3SegmentEnd(rest)
			if end <= 0 || end > len(rest) {
				t.Fatalf("pre-tokenization of %q did not advance at %q", input, rest)
			}
			got = append(got, rest[:end])
			rest = rest[end:]
		}
		if !slices.Equal(got, want) {
			t.Errorf("pre-tokenization of %q: got %q, want %q", input, got, want)
		}
	}
}
