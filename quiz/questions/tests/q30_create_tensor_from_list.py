from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_equal


def get_visible_tests():
    return [
        ("creates a vector from a list", test_creates_vector),
        ("keeps the original order", test_keeps_order),
    ]


def get_hidden_tests():
    return [
        ("works on float values", test_float_values),
        ("works on tuples too", test_tuple_input),
        ("returns a torch tensor", test_returns_tensor),
    ]


def test_creates_vector(solution_module):
    actual = solution_module.make_tensor([1, 2, 3])
    expected = torch.tensor([1, 2, 3])
    assert_tensor_equal(actual, expected)


def test_keeps_order(solution_module):
    actual = solution_module.make_tensor([5, 1, 9, 2])
    assert_shape(actual, (4,))
    assert_tensor_equal(actual, torch.tensor([5, 1, 9, 2]))


def test_float_values(solution_module):
    actual = solution_module.make_tensor([1.5, 2.5])
    assert_tensor_equal(actual, torch.tensor([1.5, 2.5]))


def test_tuple_input(solution_module):
    actual = solution_module.make_tensor((7, 8))
    assert_tensor_equal(actual, torch.tensor([7, 8]))


def test_returns_tensor(solution_module):
    actual = solution_module.make_tensor([1])
    assert isinstance(actual, torch.Tensor), f"Expected a torch.Tensor, got {type(actual)}"
