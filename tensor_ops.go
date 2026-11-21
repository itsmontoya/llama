package llama

import "github.com/itsmontoya/neuralnetwork/tensor"

// All helpers allocate disjoint outputs and preserve tensor's ownership rules.
func elementwise(left, right *tensor.Tensor, operation func(*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) error) (*tensor.Tensor, error) {
	shape, err := tensor.BroadcastShape(left.Shape(), right.Shape())
	if err != nil {
		return nil, err
	}
	out, err := tensor.New(tensor.Float32, shape)
	if err != nil {
		return nil, err
	}
	if err = operation(left, right, out); err != nil {
		return nil, err
	}
	return out, nil
}

func unary(input *tensor.Tensor, operation func(*tensor.Tensor, *tensor.Tensor) error) (*tensor.Tensor, error) {
	out, err := tensor.New(tensor.Float32, input.Shape())
	if err != nil {
		return nil, err
	}
	if err = operation(input, out); err != nil {
		return nil, err
	}
	return out, nil
}

func scalar(value float32) (*tensor.Tensor, error) {
	return tensor.FromFloat32(nil, []float32{value})
}

func reshape(input *tensor.Tensor, shape []int64) (*tensor.Tensor, error) {
	out, err := tensor.New(tensor.Float32, shape)
	if err != nil {
		return nil, err
	}
	if err = tensor.ReshapeInto(input, out); err != nil {
		return nil, err
	}
	return out, nil
}

func transpose(input *tensor.Tensor, permutation []int64) (*tensor.Tensor, error) {
	shape, err := tensor.TransposeShape(input.Shape(), permutation)
	if err != nil {
		return nil, err
	}
	out, err := tensor.New(tensor.Float32, shape)
	if err != nil {
		return nil, err
	}
	if err = tensor.TransposeInto(input, out, permutation); err != nil {
		return nil, err
	}
	return out, nil
}

func gather(input *tensor.Tensor, indices []int64, axis int64) (*tensor.Tensor, error) {
	indexTensor, err := tensor.FromInt64([]int64{int64(len(indices))}, indices)
	if err != nil {
		return nil, err
	}
	shape, err := tensor.GatherShape(input.Shape(), indexTensor.Shape(), axis)
	if err != nil {
		return nil, err
	}
	out, err := tensor.New(tensor.Float32, shape)
	if err != nil {
		return nil, err
	}
	if err = tensor.GatherInto(input, indexTensor, out, axis); err != nil {
		return nil, err
	}
	return out, nil
}

func project(input, weight *tensor.Tensor) (*tensor.Tensor, error) {
	config := tensor.GemmConfig{TransposeB: true, Alpha: 1}
	out, err := projectionOutput(input, weight.Shape(), config)
	if err != nil {
		return nil, err
	}
	if err = tensor.GemmInto(input, weight, nil, out, config); err != nil {
		return nil, err
	}
	return out, nil
}

func projectPrepared(input *tensor.Tensor, weight *tensor.PreparedRightTranspose) (*tensor.Tensor, error) {
	config := tensor.GemmConfig{TransposeB: true, Alpha: 1}
	out, err := projectionOutput(input, weight.Shape(), config)
	if err != nil {
		return nil, err
	}
	if err = weight.GemmInto(input, nil, out, config); err != nil {
		return nil, err
	}
	return out, nil
}

func projectionOutput(input *tensor.Tensor, weightShape []int64, config tensor.GemmConfig) (*tensor.Tensor, error) {
	shape, err := tensor.GemmShape(input.Shape(), weightShape, config)
	if err != nil {
		return nil, err
	}
	return tensor.New(tensor.Float32, shape)
}

func matmul(left, right *tensor.Tensor) (*tensor.Tensor, error) {
	shape, err := tensor.MatMulShape(left.Shape(), right.Shape())
	if err != nil {
		return nil, err
	}
	out, err := tensor.New(tensor.Float32, shape)
	if err != nil {
		return nil, err
	}
	if err = tensor.MatMulInto(left, right, out); err != nil {
		return nil, err
	}
	return out, nil
}
