from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_equal


def get_visible_tests():
    return [
        ("creates the requested shape", test_creates_requested_shape),
        ("fills the tensor with zeros", test_all_values_are_zero),
    ]


def get_hidden_tests():
    return [
        ("works for square shapes", test_square_shape),
        ("works for one row", test_one_row),
        ("returns a tensor object", test_returns_tensor),
    ]


def test_creates_requested_shape(solution_module):
    actual = solution_module.make_zeros(2, 3)
    assert_shape(actual, (2, 3))


def test_all_values_are_zero(solution_module):
    actual = solution_module.make_zeros(2, 2)
    expected = torch.zeros(2, 2)
    assert_tensor_equal(actual, expected)


def test_square_shape(solution_module):
    actual = solution_module.make_zeros(3, 3)
    assert_tensor_equal(actual, torch.zeros(3, 3))


def test_one_row(solution_module):
    actual = solution_module.make_zeros(1, 4)
    assert_tensor_equal(actual, torch.zeros(1, 4))


def test_returns_tensor(solution_module):
    actual = solution_module.make_zeros(1, 1)
    assert isinstance(actual, torch.Tensor), f"Expected a torch.Tensor, got {type(actual)}"
