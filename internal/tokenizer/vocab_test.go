package tokenizer

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoad(t *testing.T) {
	entries := testEntries()
	vocab, err := Load(writeVocabulary(t, packVocabulary(entries)), int64(len(entries)))
	if err != nil {
		t.Fatal(err)
	}
	if BOS != 1 || EOS != 2 {
		t.Fatalf("special IDs = BOS %d and EOS %d, want 1 and 2", BOS, EOS)
	}
	if len(vocab.entries) != len(entries) || len(vocab.pieceIDs) != len(entries) {
		t.Fatalf("loaded %d entries and %d lookup IDs, want %d of each", len(vocab.entries), len(vocab.pieceIDs), len(entries))
	}
	for id, want := range entries {
		got := vocab.entries[id]
		if got.piece != want.piece || math.Float32bits(got.score) != math.Float32bits(want.score) {
			t.Errorf("token %d = %q with score %v, want %q with score %v", id, got.piece, got.score, want.piece, want.score)
		}
		if gotID, ok := vocab.pieceIDs[want.piece]; !ok || gotID != int64(id) {
			t.Errorf("lookup of %q = %d, %t, want %d, true", want.piece, gotID, ok, id)
		}
	}
}

func TestLoadRejectsMalformedVocabulary(t *testing.T) {
	entries := testEntries()
	data := packVocabulary(entries)
	count := int64(len(entries))
	const finalPieceLength = len("final")
	finalRecord := len(data) - finalPieceLength - 8
	withEntry := func(id int, replacement entry) []byte {
		modified := append([]entry(nil), entries...)
		modified[id] = replacement
		return packVocabulary(modified)
	}
	withUint32 := func(offset int, value uint32) []byte {
		modified := append([]byte(nil), data...)
		binary.LittleEndian.PutUint32(modified[offset:], value)
		return modified
	}

	for _, test := range []struct {
		name  string
		data  []byte
		count int64
		want  string
	}{
		{"short header", data[:3], count, "read maximum piece length"},
		{"zero maximum", withUint32(0, 0), count, "invalid maximum piece length"},
		{"maximum exceeds file", withUint32(0, math.MaxUint32), count, "invalid maximum piece length"},
		{"incorrect maximum", withUint32(0, 7), count, "does not match longest piece"},
		{"negative count", data, -1, "invalid vocabulary size"},
		{"zero count", data, 0, "invalid vocabulary size"},
		{"count without room for space", data, 259, "invalid vocabulary size"},
		{"too few records", data, count + 1, "invalid piece length"},
		{"too many records", data, count - 1, "unexpected trailing data"},
		{"count exceeds file", data, math.MaxInt64, "vocabulary size"},
		{"empty piece", withUint32(8, 0), count, "invalid piece length"},
		{"piece exceeds maximum", withUint32(8, 7), count, "invalid piece length"},
		{"piece exceeds file", withUint32(finalRecord+4, 6), count, "invalid piece length"},
		{"NaN score", withUint32(4, 0x7fc00000), count, "nonfinite score"},
		{"positive infinity score", withUint32(4, 0x7f800000), count, "nonfinite score"},
		{"negative infinity score", withUint32(4, 0xff800000), count, "nonfinite score"},
		{"duplicate piece", withEntry(264, entries[259]), count, "duplicates piece"},
		{"missing space", withEntry(259, entry{piece: "\t"}), count, "missing space piece"},
		{"missing first byte fallback", withEntry(3, entry{piece: "<0x00"}), count, "token 3 must be byte fallback piece"},
		{"missing last byte fallback", withEntry(258, entry{piece: "<0xff>"}), count, "token 258 must be byte fallback piece"},
		{"incomplete record", data[:finalRecord+4], count, "invalid piece length"},
		{"incomplete piece", data[:len(data)-1], count, "invalid piece length"},
		{"trailing byte", append(append([]byte(nil), data...), 0), count, "unexpected trailing data"},
	} {
		t.Run(test.name, func(t *testing.T) {
			vocab, err := Load(writeVocabulary(t, test.data), test.count)
			if vocab != nil || err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load() = (%v, %v), want nil and an error containing %q", vocab, err, test.want)
			}
		})
	}
}

func TestStories260KVocabulary(t *testing.T) {
	vocab := loadStories260KTokenizer(t)
	if len(vocab.entries) != stories260KVocabSize {
		t.Fatalf("loaded %d entries, want %d", len(vocab.entries), stories260KVocabSize)
	}
	for _, test := range []struct {
		id    int64
		entry entry
	}{
		{BOS, entry{piece: "\n<s>\n"}},
		{EOS, entry{piece: "\n</s>\n"}},
		{3, entry{piece: "<0x00>"}},
		{258, entry{piece: "<0xFF>"}},
		{410, entry{piece: " ", score: -151}},
		{511, entry{piece: "\u200a", score: -252}},
	} {
		if got := vocab.entries[test.id]; got != test.entry {
			t.Errorf("token %d = %q with score %v, want %q with score %v", test.id, got.piece, got.score, test.entry.piece, test.entry.score)
		}
	}
}

const stories260KVocabSize = 512 // The vocabulary size in the pinned Stories260K checkpoint.

func loadStories260KTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	path := os.Getenv("LLAMA_STORIES260K_TOKENIZER")
	if path == "" {
		t.Skip("set LLAMA_STORIES260K_TOKENIZER to the pinned tok512.bin")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	const wantHash = "037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312"
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != wantHash {
		t.Fatalf("tokenizer SHA-256 = %s, want %s", got, wantHash)
	}
	vocab, err := Load(path, stories260KVocabSize)
	if err != nil {
		t.Fatal(err)
	}
	return vocab
}

func testEntries() []entry {
	entries := []entry{{piece: "<unk>"}, {piece: "\n<s>\n"}, {piece: "\n</s>\n"}}
	for value := range 256 {
		entries = append(entries, entry{piece: fmt.Sprintf("<0x%02X>", value)})
	}
	return append(entries,
		entry{piece: " ", score: -2.5},
		entry{piece: " a", score: 3.25},
		entry{piece: "  ", score: math.Float32frombits(0x80000000)},
		entry{piece: "é", score: -7},
		entry{piece: "\xff", score: 0.125},
		entry{piece: "final", score: -9},
	)
}

func packVocabulary(entries []entry) []byte {
	var longest int
	for _, token := range entries {
		longest = max(longest, len(token.piece))
	}
	data := binary.LittleEndian.AppendUint32(nil, uint32(longest))
	for _, token := range entries {
		data = binary.LittleEndian.AppendUint32(data, math.Float32bits(token.score))
		data = binary.LittleEndian.AppendUint32(data, uint32(len(token.piece)))
		data = append(data, token.piece...)
	}
	return data
}

func writeVocabulary(t *testing.T, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokenizer.bin")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}
