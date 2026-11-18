package tokenizer

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	// BOS is the beginning-of-sequence token ID.
	BOS int64 = 1
	// EOS is the end-of-sequence token ID.
	EOS int64 = 2
)

const (
	byteFallbackStart = 3
	byteFallbackCount = 256
	recordHeaderSize  = 8
	minRecordSize     = recordHeaderSize + 1
)

// Tokenizer holds an immutable vocabulary in token ID order.
type Tokenizer struct {
	entries  []entry
	pieceIDs map[string]int64
}

type entry struct {
	piece string
	score float32
}

// Load reads a little-endian llama2.c tokenizer vocabulary with the supplied
// token count from its matching checkpoint.
func Load(path string, vocabSize int64) (*Tokenizer, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: open vocabulary: %w", err)
	}
	defer file.Close()

	vocab, err := readVocabulary(file, vocabSize)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: vocabulary %q: %w", path, err)
	}
	return vocab, nil
}

func readVocabulary(file *os.File, vocabSize int64) (*Tokenizer, error) {
	const minVocabSize = byteFallbackStart + byteFallbackCount + 1 // Includes the space piece.
	if vocabSize < minVocabSize || vocabSize > math.MaxInt {
		return nil, fmt.Errorf("invalid vocabulary size %d", vocabSize)
	}
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	reader := bufio.NewReader(file)
	var header [4]byte
	if _, err := io.ReadFull(reader, header[:]); err != nil {
		return nil, fmt.Errorf("read maximum piece length: %w", err)
	}
	remaining := info.Size() - int64(len(header))
	if vocabSize > remaining/minRecordSize {
		return nil, fmt.Errorf("vocabulary size %d exceeds the file size", vocabSize)
	}
	maxPieceLength := int64(binary.LittleEndian.Uint32(header[:]))
	// Reserve one byte for every other piece and a header for every record.
	maxPossibleLength := remaining - vocabSize*minRecordSize + 1
	if maxPieceLength < 1 || maxPieceLength > math.MaxInt || maxPieceLength > maxPossibleLength {
		return nil, fmt.Errorf("invalid maximum piece length %d", maxPieceLength)
	}

	vocab := &Tokenizer{
		entries:  make([]entry, vocabSize),
		pieceIDs: make(map[string]int64, vocabSize),
	}
	var longestPiece int64
	var record [recordHeaderSize]byte
	for id := range vocabSize {
		if _, err := io.ReadFull(reader, record[:]); err != nil {
			return nil, fmt.Errorf("read token %d record: %w", id, err)
		}
		remaining -= recordHeaderSize
		score := math.Float32frombits(binary.LittleEndian.Uint32(record[:4]))
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
			return nil, fmt.Errorf("token %d has a nonfinite score", id)
		}
		length := int64(binary.LittleEndian.Uint32(record[4:]))
		available := remaining - (vocabSize-id-1)*minRecordSize
		if length < 1 || length > maxPieceLength || length > available {
			return nil, fmt.Errorf("token %d has invalid piece length %d", id, length)
		}
		pieceBytes := make([]byte, length)
		if _, err := io.ReadFull(reader, pieceBytes); err != nil {
			return nil, fmt.Errorf("read token %d piece: %w", id, err)
		}
		remaining -= length
		longestPiece = max(longestPiece, length)
		piece := string(pieceBytes)
		if previous, exists := vocab.pieceIDs[piece]; exists {
			return nil, fmt.Errorf("token %d duplicates piece %q from token %d", id, piece, previous)
		}
		vocab.entries[id] = entry{piece: piece, score: score}
		vocab.pieceIDs[piece] = id
	}
	if _, err := reader.ReadByte(); err != io.EOF {
		if err != nil {
			return nil, fmt.Errorf("check vocabulary end: %w", err)
		}
		return nil, fmt.Errorf("unexpected trailing data after %d tokens", vocabSize)
	}
	if longestPiece != maxPieceLength {
		return nil, fmt.Errorf("maximum piece length %d does not match longest piece %d", maxPieceLength, longestPiece)
	}
	for value := range byteFallbackCount {
		id := byteFallbackStart + value
		want := fmt.Sprintf("<0x%02X>", value)
		if vocab.entries[id].piece != want {
			return nil, fmt.Errorf("token %d must be byte fallback piece %q", id, want)
		}
	}
	if _, exists := vocab.pieceIDs[" "]; !exists {
		return nil, fmt.Errorf("missing space piece")
	}
	return vocab, nil
}
