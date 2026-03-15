from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_tensor_close


def get_visible_tests():
    return [
        ("creates a float32 tensor", test_dtype_is_float32),
        ("keeps the original numeric values", test_values_are_preserved),
    ]


def get_hidden_tests():
    return [
        ("works on integer inputs", test_integer_input),
        ("works on decimal inputs", test_decimal_input),
        ("returns a tensor object", test_returns_tensor),
    ]


def test_dtype_is_float32(solution_module):
    actual = solution_module.make_float_tensor([1, 2, 3])
    assert actual.dtype == torch.float32, f"Expected torch.float32, got {actual.dtype}"


def test_values_are_preserved(solution_module):
    actual = solution_module.make_float_tensor([1, 2, 3])
    expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    assert_tensor_close(actual, expected)


def test_integer_input(solution_module):
    actual = solution_module.make_float_tensor([4, 5])
    assert actual.dtype == torch.float32, f"Expected torch.float32, got {actual.dtype}"


def test_decimal_input(solution_module):
    actual = solution_module.make_float_tensor([1.25, 2.5])
    expected = torch.tensor([1.25, 2.5], dtype=torch.float32)
    assert_tensor_close(actual, expected)


def test_returns_tensor(solution_module):
    actual = solution_module.make_float_tensor([1])
    assert isinstance(actual, torch.Tensor), f"Expected a torch.Tensor, got {type(actual)}"
