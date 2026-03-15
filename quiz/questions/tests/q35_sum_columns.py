from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_equal


def get_visible_tests():
    return [
        ("sums each column", test_sums_columns),
        ("returns a 1d tensor", test_returns_vector),
    ]


def get_hidden_tests():
    return [
        ("works for float matrices", test_float_matrix),
        ("works for negative values", test_negative_values),
        ("keeps one value per column", test_column_count),
    ]


def test_sums_columns(solution_module):
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    expected = torch.tensor([5, 7, 9])
    actual = solution_module.sum_columns(x)
    assert_tensor_equal(actual, expected)


def test_returns_vector(solution_module):
    x = torch.tensor([[1, 2], [3, 4]])
    actual = solution_module.sum_columns(x)
    assert_shape(actual, (2,))


def test_float_matrix(solution_module):
    x = torch.tensor([[1.5, 2.0], [0.5, 3.0]])
    expected = torch.tensor([2.0, 5.0])
    actual = solution_module.sum_columns(x)
    assert_tensor_equal(actual, expected)


def test_negative_values(solution_module):
    x = torch.tensor([[2, -1], [3, -4]])
    expected = torch.tensor([5, -5])
    actual = solution_module.sum_columns(x)
    assert_tensor_equal(actual, expected)


def test_column_count(solution_module):
    x = torch.zeros(4, 3)
    actual = solution_module.sum_columns(x)
    assert_shape(actual, (3,))
