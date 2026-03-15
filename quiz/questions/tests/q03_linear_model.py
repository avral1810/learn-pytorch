from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_close


def get_visible_tests():
    return [
        ("computes x @ weight + bias", test_basic_formula),
        ("returns shape (batch, 1)", test_output_shape),
    ]


def get_hidden_tests():
    return [
        ("bias broadcasts correctly", test_bias_broadcast),
        ("works for multiple rows", test_multiple_rows),
        ("returns floating values", test_float_output),
    ]


def test_basic_formula(solution_module):
    x = torch.tensor([[2.0], [3.0]])
    weight = torch.tensor([[4.0]])
    bias = torch.tensor([1.0])
    expected = torch.tensor([[9.0], [13.0]])
    assert_tensor_close(solution_module.linear_model(x, weight, bias), expected)


def test_output_shape(solution_module):
    x = torch.randn(5, 1)
    weight = torch.randn(1, 1)
    bias = torch.randn(1)
    actual = solution_module.linear_model(x, weight, bias)
    assert_shape(actual, (5, 1))


def test_bias_broadcast(solution_module):
    x = torch.tensor([[1.0], [1.0], [1.0]])
    weight = torch.tensor([[2.0]])
    bias = torch.tensor([3.0])
    expected = torch.tensor([[5.0], [5.0], [5.0]])
    assert_tensor_close(solution_module.linear_model(x, weight, bias), expected)


def test_multiple_rows(solution_module):
    x = torch.tensor([[0.0], [1.0], [2.0]])
    weight = torch.tensor([[1.5]])
    bias = torch.tensor([0.5])
    expected = torch.tensor([[0.5], [2.0], [3.5]])
    assert_tensor_close(solution_module.linear_model(x, weight, bias), expected)


def test_float_output(solution_module):
    x = torch.tensor([[1.0]])
    weight = torch.tensor([[2.0]])
    bias = torch.tensor([0.0])
    actual = solution_module.linear_model(x, weight, bias)
    assert actual.dtype.is_floating_point, "Expected a floating point output tensor"
