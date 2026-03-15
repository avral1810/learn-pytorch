from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_equal


def get_visible_tests():
    return [("adds bias across rows", test_basic), ("keeps matrix shape", test_shape)]


def get_hidden_tests():
    return [("works with negatives", test_negative), ("works with floats", test_float)]


def test_basic(solution_module):
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    bias = torch.tensor([10, 20, 30])
    expected = torch.tensor([[11, 22, 33], [14, 25, 36]])
    assert_tensor_equal(solution_module.add_bias(matrix, bias), expected)


def test_shape(solution_module):
    assert_shape(solution_module.add_bias(torch.ones(4, 3), torch.zeros(3)), (4, 3))


def test_negative(solution_module):
    matrix = torch.tensor([[1, -1]])
    bias = torch.tensor([2, 3])
    assert_tensor_equal(solution_module.add_bias(matrix, bias), torch.tensor([[3, 2]]))


def test_float(solution_module):
    matrix = torch.tensor([[1.5, 2.5]])
    bias = torch.tensor([0.5, 1.5])
    assert_tensor_equal(solution_module.add_bias(matrix, bias), torch.tensor([[2.0, 4.0]]))
