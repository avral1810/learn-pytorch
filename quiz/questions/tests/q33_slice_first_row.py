from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_equal


def get_visible_tests():
    return [
        ("returns the first row values", test_returns_first_row_values),
        ("returns a 1d row tensor", test_returns_1d_row),
    ]


def get_hidden_tests():
    return [
        ("works for float matrices", test_float_matrix),
        ("works for wider matrices", test_wider_matrix),
        ("keeps row order exactly", test_keeps_order),
    ]


def test_returns_first_row_values(solution_module):
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    expected = torch.tensor([1, 2, 3])
    actual = solution_module.first_row(x)
    assert_tensor_equal(actual, expected)


def test_returns_1d_row(solution_module):
    x = torch.tensor([[1, 2], [3, 4]])
    actual = solution_module.first_row(x)
    assert_shape(actual, (2,))


def test_float_matrix(solution_module):
    x = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    expected = torch.tensor([1.5, 2.5])
    actual = solution_module.first_row(x)
    assert_tensor_equal(actual, expected)


def test_wider_matrix(solution_module):
    x = torch.tensor([[9, 8, 7, 6], [1, 2, 3, 4]])
    expected = torch.tensor([9, 8, 7, 6])
    actual = solution_module.first_row(x)
    assert_tensor_equal(actual, expected)


def test_keeps_order(solution_module):
    x = torch.tensor([[5, 1, 4], [9, 2, 8]])
    expected = torch.tensor([5, 1, 4])
    actual = solution_module.first_row(x)
    assert_tensor_equal(actual, expected)
