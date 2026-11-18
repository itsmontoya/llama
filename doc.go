// Package llama implements a forward-only FP32 reference decoder using
// neuralnetwork tensors. It evaluates a complete, nonempty token prefix at
// batch size one. It is a numerical baseline for later cached inference.
package llama
