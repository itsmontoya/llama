// Command generate prints a greedy text continuation from a legacy FP32 model.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/itsmontoya/llama"
	"github.com/itsmontoya/llama/internal/tokenizer"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	flags := flag.NewFlagSet("generate", flag.ContinueOnError)
	flags.SetOutput(stderr)
	modelPath := flags.String("model", "", "legacy FP32 checkpoint path")
	tokenizerPath := flags.String("tokenizer", "", "matching tokenizer vocabulary path")
	prompt := flags.String("prompt", "", "text prompt")
	maxNewTokens := flags.Int("max-new-tokens", 16, "maximum new tokens")
	if err := flags.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	switch {
	case flags.NArg() != 0:
		fmt.Fprintln(stderr, "generate: unexpected positional arguments")
		return 2
	case *modelPath == "" || *tokenizerPath == "":
		fmt.Fprintln(stderr, "generate: -model and -tokenizer are required")
		return 2
	case *maxNewTokens < 0:
		fmt.Fprintln(stderr, "generate: -max-new-tokens must be nonnegative")
		return 2
	}

	continuation, err := generate(*modelPath, *tokenizerPath, *prompt, *maxNewTokens)
	if err == nil {
		_, err = fmt.Fprintln(stdout, continuation)
	}
	if err != nil {
		fmt.Fprintln(stderr, "generate:", err)
		return 1
	}
	return 0
}

func generate(modelPath, tokenizerPath, prompt string, maxNewTokens int) (string, error) {
	decoder, err := llama.LoadLegacyModel(modelPath)
	if err != nil {
		return "", err
	}
	vocab, err := tokenizer.Load(tokenizerPath, decoder.Config().VocabSize)
	if err != nil {
		return "", err
	}
	promptIDs, err := vocab.Encode(prompt)
	if err != nil {
		return "", err
	}
	generatedIDs, err := decoder.GenerateGreedy(promptIDs, maxNewTokens, tokenizer.BOS)
	if err != nil {
		return "", err
	}
	return vocab.Decode(promptIDs[len(promptIDs)-1], generatedIDs)
}
