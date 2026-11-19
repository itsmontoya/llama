// Command generate prints a greedy text continuation from a Llama checkpoint.
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
	_, status := run(os.Args[1:], os.Stdout, os.Stderr)
	os.Exit(status)
}

type generationResult struct {
	promptIDs []int64
	suffixIDs []int64
	text      string
}

func run(args []string, stdout, stderr io.Writer) (generationResult, int) {
	flags := flag.NewFlagSet("generate", flag.ContinueOnError)
	flags.SetOutput(stderr)
	modelPath := flags.String("model", "", "legacy FP32 checkpoint path")
	tokenizerPath := flags.String("tokenizer", "", "matching legacy tokenizer vocabulary path")
	hfModelDir := flags.String("hf-model-dir", "", "Hugging Face model and tokenizer directory")
	context := flags.Int64("context", 512, "Hugging Face runtime context cap in tokens")
	uncached := flags.Bool("uncached", false, "recompute full prefixes in Hugging Face mode")
	prompt := flags.String("prompt", "", "text prompt")
	maxNewTokens := flags.Int("max-new-tokens", 16, "maximum new tokens")
	if err := flags.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return generationResult{}, 0
		}
		return generationResult{}, 2
	}
	provided := make(map[string]bool)
	flags.Visit(func(f *flag.Flag) { provided[f.Name] = true })
	useHF := provided["hf-model-dir"]
	var usageError string
	switch {
	case flags.NArg() != 0:
		usageError = "unexpected positional arguments"
	case useHF && (provided["model"] || provided["tokenizer"]):
		usageError = "-hf-model-dir cannot be combined with -model or -tokenizer"
	case useHF && *hfModelDir == "":
		usageError = "-hf-model-dir requires a nonempty directory path"
	case !useHF && provided["context"]:
		usageError = "-context requires -hf-model-dir"
	case !useHF && provided["uncached"]:
		usageError = "-uncached requires -hf-model-dir"
	case !useHF && (*modelPath == "" || *tokenizerPath == ""):
		usageError = "-model and -tokenizer are required for legacy mode"
	case *context <= 0:
		usageError = "-context must be positive"
	case *maxNewTokens < 0:
		usageError = "-max-new-tokens must be nonnegative"
	}
	if usageError != "" {
		fmt.Fprintln(stderr, "generate:", usageError)
		return generationResult{}, 2
	}

	var result generationResult
	var err error
	if useHF {
		result, err = generateHF(*hfModelDir, *context, *prompt, *maxNewTokens, *uncached)
	} else {
		result, err = generateLegacy(*modelPath, *tokenizerPath, *prompt, *maxNewTokens)
	}
	if err == nil {
		_, err = fmt.Fprintln(stdout, result.text)
	}
	if err != nil {
		fmt.Fprintln(stderr, "generate:", err)
		return generationResult{}, 1
	}
	return result, 0
}

func generateLegacy(modelPath, tokenizerPath, prompt string, maxNewTokens int) (generationResult, error) {
	decoder, err := llama.LoadLegacyModel(modelPath)
	if err != nil {
		return generationResult{}, err
	}
	vocab, err := tokenizer.Load(tokenizerPath, decoder.Config().VocabSize)
	if err != nil {
		return generationResult{}, err
	}
	promptIDs, err := vocab.Encode(prompt)
	if err != nil {
		return generationResult{}, err
	}
	suffixIDs, err := decoder.GenerateGreedy(promptIDs, maxNewTokens, tokenizer.BOS)
	if err != nil {
		return generationResult{}, err
	}
	text, err := vocab.Decode(promptIDs[len(promptIDs)-1], suffixIDs)
	return generationResult{promptIDs: promptIDs, suffixIDs: suffixIDs, text: text}, err
}

func generateHF(dir string, context int64, prompt string, maxNewTokens int, uncached bool) (generationResult, error) {
	decoder, generation, err := llama.LoadHFModel(dir, context)
	if err != nil {
		return generationResult{}, err
	}
	vocab, err := tokenizer.LoadLlama3(dir, decoder.Config().VocabSize, generation.BOSID, generation.StopIDs)
	if err != nil {
		return generationResult{}, err
	}
	promptIDs, err := vocab.Encode(prompt)
	if err != nil {
		return generationResult{}, err
	}
	generate := decoder.GenerateGreedyCached
	if uncached {
		generate = decoder.GenerateGreedy
	}
	suffixIDs, err := generate(promptIDs, maxNewTokens, generation.StopIDs...)
	if err != nil {
		return generationResult{}, err
	}
	text, err := vocab.Decode(suffixIDs)
	return generationResult{promptIDs: promptIDs, suffixIDs: suffixIDs, text: text}, err
}
