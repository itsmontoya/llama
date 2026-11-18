package tokenizer

import (
	"slices"
	"unicode"
	"unicode/utf8"
)

// llama3Pattern is the Split regex in the pinned tokenizer.json. This scanner
// implements its ordered alternatives, including the whitespace lookahead.
const llama3Pattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

type llama3RuneRange struct{ first, last rune }

func llama3InRanges(r rune, ranges []llama3RuneRange) bool {
	_, found := slices.BinarySearchFunc(ranges, r, func(interval llama3RuneRange, target rune) int {
		if interval.last < target {
			return -1
		}
		if interval.first > target {
			return 1
		}
		return 0
	})
	return found
}

func llama3Letter(r rune) bool {
	if r < utf8.RuneSelf {
		return 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z'
	}
	return llama3InRanges(r, llama3Letters)
}

func llama3Number(r rune) bool {
	if r < utf8.RuneSelf {
		return '0' <= r && r <= '9'
	}
	return llama3InRanges(r, llama3Numbers)
}

func llama3Punctuation(r rune) bool {
	return !unicode.IsSpace(r) && !llama3Letter(r) && !llama3Number(r)
}

func llama3RunEnd(text string, include func(rune) bool) int {
	for offset, r := range text {
		if !include(r) {
			return offset
		}
	}
	return len(text)
}

func llama3ContractionEnd(text string) int {
	// The audited case-insensitive suffixes contain only ASCII letters. Long s
	// is the additional Unicode case-fold match for the suffix 's.
	suffix, width := utf8.DecodeRuneInString(text[1:])
	switch suffix {
	case 's', 'S', 'ſ', 't', 'T', 'm', 'M', 'd', 'D':
		return 1 + width
	case 'r', 'R', 'v', 'V':
		if len(text) >= 3 && (text[2] == 'e' || text[2] == 'E') {
			return 3
		}
	case 'l', 'L':
		if len(text) >= 3 && (text[2] == 'l' || text[2] == 'L') {
			return 3
		}
	}
	return 0
}

// llama3SegmentEnd returns the first pre-token's byte length. The caller
// supplies nonempty, valid UTF-8 after splitting out added tokens.
func llama3SegmentEnd(text string) int {
	if text[0] == '\'' {
		if end := llama3ContractionEnd(text); end != 0 {
			return end
		}
	}
	first, width := utf8.DecodeRuneInString(text)
	if llama3Letter(first) {
		return width + llama3RunEnd(text[width:], llama3Letter)
	}
	if llama3Number(first) {
		end := width
		for count := 1; count < 3 && end < len(text); count++ {
			r, size := utf8.DecodeRuneInString(text[end:])
			if !llama3Number(r) {
				break
			}
			end += size
		}
		return end
	}
	if first != '\r' && first != '\n' && width < len(text) {
		next, _ := utf8.DecodeRuneInString(text[width:])
		if llama3Letter(next) {
			return width + llama3RunEnd(text[width:], llama3Letter)
		}
	}
	start := 0
	if first == ' ' {
		start = 1
	}
	if end := start + llama3RunEnd(text[start:], llama3Punctuation); end > start {
		for end < len(text) && (text[end] == '\r' || text[end] == '\n') {
			end++
		}
		return end
	}
	return llama3WhitespaceEnd(text)
}

func llama3WhitespaceEnd(text string) int {
	end, lastStart, newlineEnd := 0, 0, 0
	for end < len(text) {
		r, size := utf8.DecodeRuneInString(text[end:])
		if !unicode.IsSpace(r) {
			break
		}
		lastStart = end
		end += size
		if r == '\r' || r == '\n' {
			newlineEnd = end
		}
	}
	if newlineEnd != 0 {
		return newlineEnd
	}
	// Before a non-space, the negative lookahead backtracks by one rune.
	if end < len(text) && lastStart != 0 {
		return lastStart
	}
	return end
}
