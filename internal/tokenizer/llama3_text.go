package tokenizer

import (
	"container/heap"
	"fmt"
	"strings"
	"unicode/utf8"
)

// Llama3Tokenizer holds the immutable vocabulary and rules for Llama 3
// completion prompts. Its Encode and Decode methods may be used concurrently.
type Llama3Tokenizer struct {
	pieces  []string
	baseIDs map[string]int64
	byteIDs [256]int64
	merges  map[llama3Pair]llama3Merge
	added   llama3AddedNode
	bosID   int64
}

type llama3AddedNode struct {
	next map[byte]*llama3AddedNode
	id   int64 // Added IDs follow the positive-sized byte vocabulary. Zero means no match.
}

func (n *llama3AddedNode) add(piece string, id int64) {
	for i := range len(piece) {
		if n.next == nil {
			n.next = make(map[byte]*llama3AddedNode)
		}
		if n.next[piece[i]] == nil {
			n.next[piece[i]] = &llama3AddedNode{}
		}
		n = n.next[piece[i]]
	}
	n.id = id
}

func (n *llama3AddedNode) match(text string) (id int64, length int) {
	for i := range len(text) {
		n = n.next[text[i]]
		if n == nil {
			break
		}
		if n.id != 0 {
			id, length = n.id, i+1
		}
	}
	return id, length
}

// Encode returns a caller-owned sequence for the prompt. It prepends BOS,
// preserves literal added tokens as IDs, and does not append EOS.
func (t *Llama3Tokenizer) Encode(prompt string) ([]int64, error) {
	if !utf8.ValidString(prompt) {
		return nil, fmt.Errorf("tokenizer: prompt is not valid UTF-8")
	}
	ids := []int64{t.bosID}
	// The reference extracts non-normalized added tokens before splitting
	// ordinary text. Match the longest literal at each earliest byte offset.
	start := 0
	for offset := 0; offset < len(prompt); {
		id, length := t.added.match(prompt[offset:])
		if length == 0 {
			offset++
			continue
		}
		ids = t.appendOrdinary(ids, prompt[start:offset])
		ids = append(ids, id)
		offset += length
		start = offset
	}
	return t.appendOrdinary(ids, prompt[start:]), nil
}

func (t *Llama3Tokenizer) appendOrdinary(ids []int64, text string) []int64 {
	for text != "" {
		end := llama3SegmentEnd(text)
		ids = t.appendBPE(ids, text[:end])
		text = text[end:]
	}
	return ids
}

type llama3Pair struct{ left, right int64 }

type llama3Merge struct {
	id   int64
	rank int
}

type llama3Symbol struct {
	id         int64
	prev, next int
}

type llama3Candidate struct {
	left  int
	pair  llama3Pair
	merge llama3Merge
}

type llama3Candidates []llama3Candidate

func (q llama3Candidates) Len() int      { return len(q) }
func (q llama3Candidates) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q llama3Candidates) Less(i, j int) bool {
	if q[i].merge.rank != q[j].merge.rank {
		return q[i].merge.rank < q[j].merge.rank
	}
	return q[i].left < q[j].left
}
func (q *llama3Candidates) Push(value any) {
	*q = append(*q, value.(llama3Candidate))
}
func (q *llama3Candidates) Pop() any {
	last := len(*q) - 1
	value := (*q)[last]
	*q = (*q)[:last]
	return value
}

func (t *Llama3Tokenizer) appendBPE(ids []int64, segment string) []int64 {
	// The audited BPE's ignore_merges option uses a complete pre-token's
	// vocabulary ID directly when it exists.
	if id, ok := t.baseIDs[segment]; ok {
		return append(ids, id)
	}
	symbols := make([]llama3Symbol, len(segment))
	for i := range symbols {
		symbols[i] = llama3Symbol{id: t.byteIDs[segment[i]], prev: i - 1, next: i + 1}
	}
	var candidates llama3Candidates
	queue := func(left int) {
		if left < 0 || symbols[left].next >= len(symbols) {
			return
		}
		pair := llama3Pair{symbols[left].id, symbols[symbols[left].next].id}
		if merge, ok := t.merges[pair]; ok {
			heap.Push(&candidates, llama3Candidate{left: left, pair: pair, merge: merge})
		}
	}
	for i := range symbols {
		queue(i)
	}
	for len(candidates) != 0 {
		match := heap.Pop(&candidates).(llama3Candidate)
		left := &symbols[match.left]
		if left.next >= len(symbols) || left.id != match.pair.left || symbols[left.next].id != match.pair.right {
			continue
		}
		// Keep the left symbol's index for tie-breaking. Invalidate candidates
		// at the removed index and queue the newly adjacent pairs.
		right := &symbols[left.next]
		left.id, left.next = match.merge.id, right.next
		right.next = len(symbols)
		if left.next < len(symbols) {
			symbols[left.next].prev = match.left
		}
		queue(left.prev)
		queue(match.left)
	}
	for i := 0; i < len(symbols); i = symbols[i].next {
		ids = append(ids, symbols[i].id)
	}
	return ids
}

// Decode reconstructs bytes across pieces, skips added special tokens, and
// replaces invalid UTF-8 with U+FFFD. It preserves whitespace and punctuation.
func (t *Llama3Tokenizer) Decode(ids []int64) (string, error) {
	var decoded strings.Builder
	for i, id := range ids {
		if id < 0 || id >= int64(len(t.pieces)) {
			return "", fmt.Errorf("tokenizer: token ID %d at index %d is outside vocabulary of size %d", id, i, len(t.pieces))
		}
		// Pieces at added special token IDs are empty.
		decoded.WriteString(t.pieces[id])
	}
	return llama3LossyUTF8(decoded.String()), nil
}

func llama3LossyUTF8(bytes string) string {
	if utf8.ValidString(bytes) {
		return bytes
	}
	var decoded strings.Builder
	for len(bytes) != 0 {
		r, width := utf8.DecodeRuneInString(bytes)
		if r == utf8.RuneError && width == 1 {
			decoded.WriteRune(utf8.RuneError)
			bytes = bytes[llama3InvalidUTF8Width(bytes):]
			continue
		}
		decoded.WriteString(bytes[:width])
		bytes = bytes[width:]
	}
	return decoded.String()
}

// Match String::from_utf8_lossy: replace a partially valid multi-byte
// prefix once, and resume at the first invalid or restricted continuation.
func llama3InvalidUTF8Width(bytes string) int {
	first := bytes[0]
	if first < 0xC2 || first > 0xF4 || len(bytes) < 2 {
		return 1
	}
	second := bytes[1]
	if second < 0x80 || second > 0xBF ||
		first == 0xE0 && second < 0xA0 || first == 0xED && second > 0x9F ||
		first == 0xF0 && second < 0x90 || first == 0xF4 && second > 0x8F {
		return 1
	}
	// A valid two-byte sequence, or a valid three-byte sequence with a valid
	// third byte, would already have been accepted by utf8.DecodeRuneInString.
	if first >= 0xF0 && len(bytes) >= 3 && bytes[2] >= 0x80 && bytes[2] <= 0xBF {
		return 3
	}
	return 2
}
