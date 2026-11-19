package main

import (
	"bytes"
	"fmt"
	"slices"
	"strings"
	"testing"
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
