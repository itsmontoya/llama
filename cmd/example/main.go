// Command example answers questions using a local Llama 3.2 checkpoint.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/itsmontoya/llama"
	"github.com/itsmontoya/llama/internal/tokenizer"
	"github.com/itsmontoya/neuralnetwork/gpu"
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
	contextCap := flags.Int64("context", 512, "runtime context cap in tokens")
	maxNewTokens := flags.Int("max-new-tokens", 16, "maximum new tokens per answer")
	ffnBackend := flags.String("ffn-backend", "cpu", "Hugging Face FFN backend (cpu or metal)")
	attentionBackend := flags.String("attention-backend", "cpu", "Hugging Face attention backend (cpu or metal with -ffn-backend metal)")
	metalWeightDType := flags.String("metal-weight-dtype", "float32", "Metal attention weight dtype (float32, float16, bfloat16, q8_0, or q4_0)")
	metalKVDType := flags.String("metal-kv-dtype", "float32", "Metal attention KV cache dtype (float32 or float16)")
	if err := flags.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	provided := make(map[string]bool)
	flags.Visit(func(f *flag.Flag) { provided[f.Name] = true })
	var usageError string
	switch {
	case flags.NArg() != 0:
		usageError = "enter questions on stdin"
	case *modelDir == "":
		usageError = "set -hf-model-dir or LLAMA_32_1B_MODEL_DIR"
	case *contextCap <= 0:
		usageError = "-context must be positive"
	case *maxNewTokens <= 0:
		usageError = "-max-new-tokens must be positive"
	case *ffnBackend != "cpu" && *ffnBackend != "metal":
		usageError = "-ffn-backend must be cpu or metal"
	case *attentionBackend != "cpu" && *attentionBackend != "metal":
		usageError = "-attention-backend must be cpu or metal"
	case *attentionBackend == "metal" && *ffnBackend != "metal":
		usageError = "-attention-backend metal requires -ffn-backend metal"
	case provided["metal-weight-dtype"] && *attentionBackend != "metal":
		usageError = "-metal-weight-dtype requires -attention-backend metal"
	case provided["metal-kv-dtype"] && *attentionBackend != "metal":
		usageError = "-metal-kv-dtype requires -attention-backend metal"
	}

	var (
		weightDType gpu.DType
		kvDType     gpu.DType
		parseErr    error
	)

	if usageError == "" && *attentionBackend == "metal" {
		if weightDType, parseErr = parseMetalWeightDType(*metalWeightDType); parseErr != nil {
			usageError = parseErr.Error()
		} else if kvDType, parseErr = parseMetalKVDType(*metalKVDType); parseErr != nil {
			usageError = parseErr.Error()
		}
	}

	if usageError != "" {
		fmt.Fprintln(stderr, "example:", usageError)
		return 2
	}

	decoder, generation, err := loadDecoder(*modelDir, *contextCap, *ffnBackend == "metal", *attentionBackend == "metal", llama.MetalOptions{
		WeightDType:  weightDType,
		KVCacheDType: kvDType,
	})

	if err != nil {
		fmt.Fprintln(stderr, "example:", err)
		return 1
	}

	defer func() {
		if closeErr := decoder.Close(); closeErr != nil {
			fmt.Fprintln(stderr, "example:", closeErr)
		}
	}()

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
		if int64(len(prompt)) >= *contextCap {
			return "", fmt.Errorf("question uses %d tokens and leaves no room to answer within context %d", len(prompt), *contextCap)
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

func loadDecoder(dir string, maxContext int64, metalFFN, metalAttention bool, metalOptions llama.MetalOptions) (*llama.Decoder, llama.HFGenerationConfig, error) {
	switch {
	case metalAttention:
		return llama.LoadHFModelMetalAttentionWithOptions(context.Background(), dir, maxContext, metalOptions)
	case metalFFN:
		return llama.LoadHFModelMetal(context.Background(), dir, maxContext)
	default:
		return llama.LoadHFModel(dir, maxContext)
	}
}

func parseMetalWeightDType(value string) (gpu.DType, error) {
	switch value {
	case "float32":
		return gpu.Float32, nil
	case "float16":
		return gpu.Float16, nil
	case "bfloat16":
		return gpu.BFloat16, nil
	case "q8_0":
		return gpu.Q8_0, nil
	case "q4_0":
		return gpu.Q4_0, nil
	default:
		return 0, fmt.Errorf("-metal-weight-dtype must be float32, float16, bfloat16, q8_0, or q4_0")
	}
}

func parseMetalKVDType(value string) (gpu.DType, error) {
	switch value {
	case "float32":
		return gpu.Float32, nil
	case "float16":
		return gpu.Float16, nil
	default:
		return 0, fmt.Errorf("-metal-kv-dtype must be float32 or float16")
	}
}
