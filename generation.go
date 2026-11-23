package llama

import (
	"fmt"
	"slices"
)

// GenerateGreedy returns an owned continuation bounded by maxNewTokens and
// the remaining context. Each step recomputes the prefix and selects the token
// with the highest final-row logit, breaking ties toward the lowest ID.
// A selected stop ID ends generation before it is appended. An error returns
// a nil continuation.
// Callers must synchronize mutation of the input slices during the call.
func (d *Decoder) GenerateGreedy(prompt []int64, maxNewTokens int, stopIDs ...int64) ([]int64, error) {
	budget, err := d.generationBudget(prompt, maxNewTokens, stopIDs)
	if err != nil {
		return nil, err
	}
	if budget == 0 {
		return []int64{}, nil
	}
	prefix := slices.Clone(prompt)
	for step := range budget {
		logits, err := d.Forward(prefix)
		if err != nil {
			return nil, fmt.Errorf("llama: generate token %d: %w", step+1, err)
		}
		values, err := logits.Float32Values()
		if err != nil {
			return nil, fmt.Errorf("llama: read logits for generated token %d: %w", step+1, err)
		}
		last := values[int64(len(values))-d.config.VocabSize:]
		next, err := greedyToken(last)
		if err != nil {
			return nil, fmt.Errorf("llama: generate token %d: %w", step+1, err)
		}
		if slices.Contains(stopIDs, next) {
			break
		}
		prefix = append(prefix, next)
	}
	return prefix[len(prompt):len(prefix):len(prefix)], nil
}

// GenerateGreedyCached follows the same selection and limits as GenerateGreedy.
// It prefills the prompt once and reuses the session for later predictions.
// The continuation belongs to the caller. An error returns a nil continuation.
// Callers must synchronize mutation of the input slices during the call.
func (d *Decoder) GenerateGreedyCached(prompt []int64, maxNewTokens int, stopIDs ...int64) ([]int64, error) {
	budget, err := d.generationBudget(prompt, maxNewTokens, stopIDs)
	if err != nil {
		return nil, err
	}
	if budget == 0 {
		return []int64{}, nil
	}
	// The final selected token needs no cache row because it is never fed back.
	session, err := d.NewSession(int64(len(prompt)) + budget - 1)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	row, err := session.Prefill(prompt)
	if err != nil {
		return nil, fmt.Errorf("llama: generate token 1: %w", err)
	}
	suffix := []int64{}
	for step := int64(1); step <= budget; step++ {
		next, err := greedyToken(row)
		if err != nil {
			return nil, fmt.Errorf("llama: generate token %d: %w", step, err)
		}
		if slices.Contains(stopIDs, next) {
			break
		}
		suffix = append(suffix, next)
		if step == budget {
			break
		}
		row, err = session.Decode(next)
		if err != nil {
			return nil, fmt.Errorf("llama: generate token %d: %w", step+1, err)
		}
	}
	return suffix, nil
}

func (d *Decoder) generationBudget(prompt []int64, maxNewTokens int, stopIDs []int64) (int64, error) {
	if err := d.validatePrefix(prompt); err != nil {
		return 0, err
	}
	if maxNewTokens < 0 {
		return 0, fmt.Errorf("llama: max new tokens must be nonnegative: %d", maxNewTokens)
	}
	for _, id := range stopIDs {
		if id < 0 || id >= d.config.VocabSize {
			return 0, fmt.Errorf("llama: stop token is out of range: %d", id)
		}
	}
	return min(int64(maxNewTokens), d.config.MaxContext-int64(len(prompt))), nil
}

func greedyToken(row []float32) (int64, error) {
	best := 0
	for id, value := range row {
		if !finite(float64(value)) {
			return 0, fmt.Errorf("nonfinite logit for ID %d", id)
		}
		if value > row[best] {
			best = id
		}
	}
	return int64(best), nil
}
