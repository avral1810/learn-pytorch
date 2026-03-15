from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_equal


def get_visible_tests():
    return [("multiplies 2x2 matrices", test_basic), ("keeps matrix shape", test_shape)]


def get_hidden_tests():
    return [("works on identity", test_identity), ("handles vectors in matrix form", test_rectangular)]


def test_basic(solution_module):
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    assert_tensor_equal(solution_module.matrix_multiply(a, b), torch.tensor([[19, 22], [43, 50]]))


def test_shape(solution_module):
    actual = solution_module.matrix_multiply(torch.ones(2, 3), torch.ones(3, 4))
    assert_shape(actual, (2, 4))


def test_identity(solution_module):
    a = torch.tensor([[2, 1], [0, 3]])
    eye = torch.eye(2, dtype=torch.int64)
    assert_tensor_equal(solution_module.matrix_multiply(a, eye), a)


def test_rectangular(solution_module):
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[1.0], [0.0], [1.0]])
    assert_tensor_equal(solution_module.matrix_multiply(a, b), torch.tensor([[4.0]]))
