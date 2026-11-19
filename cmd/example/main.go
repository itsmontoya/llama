// Command example answers questions using a local Llama 3.2 checkpoint.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/itsmontoya/llama"
	"github.com/itsmontoya/llama/internal/tokenizer"
)

func main() {
	info, err := os.Stdin.Stat()
	interactive := err == nil && info.Mode()&os.ModeCharDevice != 0
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr, interactive))
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer, interactive bool) int {
	flags := flag.NewFlagSet("example", flag.ContinueOnError)
	flags.SetOutput(stderr)
	modelDir := flags.String("hf-model-dir", os.Getenv("LLAMA_32_1B_MODEL_DIR"), "local model directory (defaults to LLAMA_32_1B_MODEL_DIR)")
	context := flags.Int64("context", 512, "runtime context cap in tokens")
	maxNewTokens := flags.Int("max-new-tokens", 16, "maximum new tokens per answer")
	if err := flags.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	var usageError string
	switch {
	case flags.NArg() != 0:
		usageError = "enter questions on stdin"
	case *modelDir == "":
		usageError = "set -hf-model-dir or LLAMA_32_1B_MODEL_DIR"
	case *context <= 0:
		usageError = "-context must be positive"
	case *maxNewTokens <= 0:
		usageError = "-max-new-tokens must be positive"
	}
	if usageError != "" {
		fmt.Fprintln(stderr, "example:", usageError)
		return 2
	}

	decoder, generation, err := llama.LoadHFModel(*modelDir, *context)
	if err != nil {
		fmt.Fprintln(stderr, "example:", err)
		return 1
	}
	vocab, err := tokenizer.LoadLlama3(*modelDir, decoder.Config().VocabSize, generation.BOSID, generation.StopIDs)
	if err != nil {
		fmt.Fprintln(stderr, "example:", err)
		return 1
	}
	return answerQuestions(stdin, stdout, stderr, interactive, func(question string) (string, error) {
		prompt, err := vocab.Encode("Question: " + question + "\nAnswer:")
		if err != nil {
			return "", err
		}
		if int64(len(prompt)) >= *context {
			return "", fmt.Errorf("question uses %d tokens and leaves no room to answer within context %d", len(prompt), *context)
		}
		suffix, err := decoder.GenerateGreedyCached(prompt, *maxNewTokens, generation.StopIDs...)
		if err != nil {
			return "", err
		}
		text, err := vocab.Decode(suffix)
		if err != nil {
			return "", err
		}
		answer, _, _ := strings.Cut(text, "\nQuestion:")
		answer = strings.TrimSpace(answer)
		if answer == "" {
			return "", fmt.Errorf("model returned an empty answer")
		}
		return answer, nil
	})
}

func answerQuestions(stdin io.Reader, stdout, stderr io.Writer, interactive bool, answer func(string) (string, error)) int {
	if interactive {
		fmt.Fprintln(stderr, "Ask one question per line. Press Ctrl-D to exit.")
	}
	questions := bufio.NewScanner(stdin)
	status := 0
	for {
		if interactive {
			fmt.Fprint(stderr, "> ")
		}
		if !questions.Scan() {
			if interactive {
				fmt.Fprintln(stderr)
			}
			break
		}
		question := strings.TrimSpace(questions.Text())
		if question == "" {
			continue
		}
		text, err := answer(question)
		if err != nil {
			fmt.Fprintln(stderr, "example:", err)
			status = 1
			continue
		}
		if _, err := fmt.Fprintln(stdout, text); err != nil {
			fmt.Fprintln(stderr, "example:", err)
			return 1
		}
	}
	if err := questions.Err(); err != nil {
		fmt.Fprintln(stderr, "example: read question:", err)
		return 1
	}
	return status
}
