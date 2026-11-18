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
	if err := d.validatePrefix(prompt); err != nil {
		return nil, err
	}
	if maxNewTokens < 0 {
		return nil, fmt.Errorf("llama: max new tokens must be nonnegative: %d", maxNewTokens)
	}
	for _, id := range stopIDs {
		if id < 0 || id >= d.config.VocabSize {
			return nil, fmt.Errorf("llama: stop token is out of range: %d", id)
		}
	}

	budget := min(int64(maxNewTokens), d.config.MaxContext-int64(len(prompt)))
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
		best := 0
		for id, value := range last {
			if !finite(float64(value)) {
				return nil, fmt.Errorf("llama: generate token %d: nonfinite logit for ID %d", step+1, id)
			}
			if value > last[best] {
				best = id
			}
		}
		next := int64(best)
		if slices.Contains(stopIDs, next) {
			break
		}
		prefix = append(prefix, next)
	}
	return prefix[len(prompt):len(prefix):len(prefix)], nil
}
