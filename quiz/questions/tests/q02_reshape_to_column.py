from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_equal


def get_visible_tests():
    return [
        ("turns a list into shape (n, 1)", test_list_input),
        ("keeps value order", test_value_order),
    ]


def get_hidden_tests():
    return [
        ("accepts a tensor input", test_tensor_input),
        ("works for length 1", test_single_value),
        ("returns a tensor", test_returns_tensor),
    ]


def test_list_input(solution_module):
    actual = solution_module.reshape_to_column([1, 2, 3])
    assert_shape(actual, (3, 1))


def test_value_order(solution_module):
    actual = solution_module.reshape_to_column([4, 5, 6])
    expected = torch.tensor([[4], [5], [6]])
    assert_tensor_equal(actual, expected)


def test_tensor_input(solution_module):
    actual = solution_module.reshape_to_column(torch.tensor([7, 8]))
    expected = torch.tensor([[7], [8]])
    assert_tensor_equal(actual, expected)


def test_single_value(solution_module):
    actual = solution_module.reshape_to_column([9])
    assert_shape(actual, (1, 1))


def test_returns_tensor(solution_module):
    actual = solution_module.reshape_to_column([1, 2])
    assert isinstance(actual, torch.Tensor), "Expected a torch.Tensor"
