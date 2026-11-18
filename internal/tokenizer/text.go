package tokenizer

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// Encode converts a UTF-8 prompt to token IDs with a BOS token and, for
// nonempty prompts, the model's dummy space prefix.
func (t *Tokenizer) Encode(prompt string) ([]int64, error) {
	if !utf8.ValidString(prompt) {
		return nil, fmt.Errorf("tokenizer: prompt is not valid UTF-8")
	}
	if strings.IndexByte(prompt, 0) >= 0 {
		return nil, fmt.Errorf("tokenizer: prompt contains NUL")
	}
	if prompt == "" {
		return []int64{BOS}, nil
	}

	ids := make([]int64, 0, len(prompt)+2)
	ids = append(ids, BOS, t.pieceIDs[" "])
	for len(prompt) > 0 {
		_, width := utf8.DecodeRuneInString(prompt)
		piece := prompt[:width]
		prompt = prompt[width:]
		if id, ok := t.pieceIDs[piece]; ok {
			ids = append(ids, id)
			continue
		}
		for i := range len(piece) {
			ids = append(ids, int64(piece[i])+byteFallbackStart)
		}
	}

	for {
		bestScore := float32(-1e10)
		bestIndex := -1
		var bestID int64
		for i := range len(ids) - 1 {
			pair := t.entries[ids[i]].piece + t.entries[ids[i+1]].piece
			if id, ok := t.pieceIDs[pair]; ok && t.entries[id].score > bestScore {
				bestScore = t.entries[id].score
				bestIndex = i
				bestID = id
			}
		}
		if bestIndex < 0 {
			return ids, nil
		}
		ids[bestIndex] = bestID
		copy(ids[bestIndex+1:], ids[bestIndex+2:])
		ids = ids[:len(ids)-1]
	}
}

// Decode returns the raw bytes of ids as a string. prevID is the token
// immediately before ids and determines whether to strip a leading space.
func (t *Tokenizer) Decode(prevID int64, ids []int64) (string, error) {
	vocabSize := int64(len(t.entries))
	if prevID < 0 || prevID >= vocabSize {
		return "", fmt.Errorf("tokenizer: previous token ID %d is outside vocabulary of size %d", prevID, vocabSize)
	}
	var decoded strings.Builder
	for i, id := range ids {
		if id < 0 || id >= vocabSize {
			return "", fmt.Errorf("tokenizer: token ID %d at index %d is outside vocabulary of size %d", id, i, vocabSize)
		}
		piece := t.entries[id].piece
		if prevID == BOS {
			piece = strings.TrimPrefix(piece, " ")
		}
		prevID = id

		// Load verifies the canonical byte markers occupy a contiguous ID range.
		if byteID, ok := t.pieceIDs[piece]; ok && byteID >= byteFallbackStart && byteID < byteFallbackStart+byteFallbackCount {
			decoded.WriteByte(byte(byteID - byteFallbackStart))
		} else {
			decoded.WriteString(piece)
		}
	}
	return decoded.String(), nil
}
