package llama

import "github.com/itsmontoya/neuralnetwork/tensor"

func rmsNorm(input, weight *tensor.Tensor, epsilon float32) (*tensor.Tensor, error) {
	squared, err := elementwise(input, input, tensor.MulInto)
	if err != nil {
		return nil, err
	}
	shape, err := tensor.ReductionShape(input.Shape(), []int64{-1}, true)
	if err != nil {
		return nil, err
	}
	mean, err := tensor.New(tensor.Float32, shape)
	if err != nil {
		return nil, err
	}
	if err = tensor.ReduceMeanInto(squared, mean, []int64{-1}, true); err != nil {
		return nil, err
	}
	eps, err := scalar(epsilon)
	if err != nil {
		return nil, err
	}
	shifted, err := elementwise(mean, eps, tensor.AddInto)
	if err != nil {
		return nil, err
	}
	denominator, err := unary(shifted, tensor.SqrtInto)
	if err != nil {
		return nil, err
	}
	normalized, err := elementwise(input, denominator, tensor.DivInto)
	if err != nil {
		return nil, err
	}
	return elementwise(normalized, weight, tensor.MulInto)
}

func swiGLU(gate, up *tensor.Tensor) (*tensor.Tensor, error) {
	sigmoid, err := unary(gate, tensor.SigmoidInto)
	if err != nil {
		return nil, err
	}
	silu, err := elementwise(gate, sigmoid, tensor.MulInto)
	if err != nil {
		return nil, err
	}
	return elementwise(silu, up, tensor.MulInto)
}
