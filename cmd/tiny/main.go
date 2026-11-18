// Command tiny runs the synthetic FP32 reference decoder on token IDs.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/itsmontoya/llama"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "testdata/tiny_gqa.model.json", "synthetic model JSON path")
	tokenList := flag.String("tokens", "1,4,2,4,0", "comma-separated token IDs")
	flag.Parse()
	if flag.NArg() != 0 {
		return fmt.Errorf("tiny: unexpected positional arguments")
	}
	file, err := os.Open(*modelPath)
	if err != nil {
		return err
	}
	defer file.Close()
	var model struct {
		Config  llama.Config  `json:"config"`
		Weights llama.Weights `json:"weights"`
	}
	artifactDecoder := json.NewDecoder(file)
	artifactDecoder.DisallowUnknownFields()
	if err = artifactDecoder.Decode(&model); err != nil {
		return fmt.Errorf("tiny: synthetic model: %w", err)
	}
	if err = artifactDecoder.Decode(new(any)); err != io.EOF {
		return fmt.Errorf("tiny: expected exactly one model JSON object")
	}
	decoder, err := llama.NewDecoder(model.Config, model.Weights)
	if err != nil {
		return err
	}
	parts := strings.Split(*tokenList, ",")
	tokens := make([]int64, len(parts))
	for index, part := range parts {
		if tokens[index], err = strconv.ParseInt(strings.TrimSpace(part), 10, 64); err != nil {
			return fmt.Errorf("tiny: token %d: %w", index, err)
		}
	}
	logits, err := decoder.Forward(tokens)
	if err != nil {
		return err
	}
	values, err := logits.Float32Values()
	if err != nil {
		return err
	}
	last := values[int64(len(tokens)-1)*model.Config.VocabSize:]
	best := 0
	for index, value := range last {
		if value > last[best] {
			best = index
		}
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(struct {
		Tokens    []int64   `json:"tokens"`
		Logits    []float32 `json:"last_logits"`
		NextToken int       `json:"greedy_next_token"`
	}{tokens, last, best})
}
