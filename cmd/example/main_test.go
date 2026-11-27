package main

import (
	"bytes"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/gpu"
)

func TestAnswerQuestions(t *testing.T) {
	var stdout, stderr bytes.Buffer
	var asked []string
	stdin := strings.NewReader(" \n  First?  \nToo long?\nLast?")
	status := answerQuestions(stdin, &stdout, &stderr, false, func(question string) (string, error) {
		asked = append(asked, question)
		if question == "Too long?" {
			return "", fmt.Errorf("question exceeds context")
		}
		return "Reply to " + question, nil
	})
	if !slices.Equal(asked, []string{"First?", "Too long?", "Last?"}) {
		t.Fatalf("questions = %q", asked)
	}
	if status != 1 || stdout.String() != "Reply to First?\nReply to Last?\n" || stderr.String() != "example: question exceeds context\n" {
		t.Fatalf("exit = %d, stdout = %q, stderr = %q", status, stdout.String(), stderr.String())
	}
}

func TestRunMetalFlagValidation(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want string
	}{
		{"unknown FFN backend", []string{"-hf-model-dir", "snapshot", "-ffn-backend", "other"}, "-ffn-backend must be cpu or metal"},
		{"unknown attention backend", []string{"-hf-model-dir", "snapshot", "-attention-backend", "other"}, "-attention-backend must be cpu or metal"},
		{"Metal attention requires Metal FFN", []string{"-hf-model-dir", "snapshot", "-attention-backend", "metal"}, "-attention-backend metal requires -ffn-backend metal"},
		{"metal weight dtype requires metal attention", []string{"-hf-model-dir", "snapshot", "-metal-weight-dtype", "bfloat16"}, "-metal-weight-dtype requires -attention-backend metal"},
		{"metal kv dtype requires metal attention", []string{"-hf-model-dir", "snapshot", "-metal-kv-dtype", "float16"}, "-metal-kv-dtype requires -attention-backend metal"},
		{"unknown metal weight dtype", []string{"-hf-model-dir", "snapshot", "-ffn-backend", "metal", "-attention-backend", "metal", "-metal-weight-dtype", "int4"}, "-metal-weight-dtype must be"},
		{"unknown metal kv dtype", []string{"-hf-model-dir", "snapshot", "-ffn-backend", "metal", "-attention-backend", "metal", "-metal-kv-dtype", "bfloat16"}, "-metal-kv-dtype must be"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			status := run(tc.args, strings.NewReader(""), &stdout, &stderr, false)
			if status != 2 || !strings.Contains(stderr.String(), tc.want) {
				t.Fatalf("status = %d, stderr = %q, want substring %q", status, stderr.String(), tc.want)
			}
		})
	}
}

func TestParseMetalWeightDType(t *testing.T) {
	for _, tc := range []struct {
		arg  string
		want gpu.DType
	}{
		{"float32", gpu.Float32},
		{"float16", gpu.Float16},
		{"bfloat16", gpu.BFloat16},
		{"q8_0", gpu.Q8_0},
		{"q4_0", gpu.Q4_0},
		{"q4_1", gpu.Q4_1},
	} {
		got, err := parseMetalWeightDType(tc.arg)
		if err != nil || got != tc.want {
			t.Fatalf("parseMetalWeightDType(%q) = %v, %v, want %v", tc.arg, got, err, tc.want)
		}
	}
}
