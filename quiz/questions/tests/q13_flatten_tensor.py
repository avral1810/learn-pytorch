from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_equal


def get_visible_tests():
    return [("flattens matrix", test_shape), ("keeps row-major order", test_values)]


def get_hidden_tests():
    return [("works on 3d tensor", test_three_dim), ("returns 1d", test_is_1d)]


def test_shape(solution_module):
    assert_shape(solution_module.flatten_tensor(torch.ones(2, 3)), (6,))


def test_values(solution_module):
    actual = solution_module.flatten_tensor(torch.tensor([[1, 2], [3, 4]]))
    assert_tensor_equal(actual, torch.tensor([1, 2, 3, 4]))


def test_three_dim(solution_module):
    assert_shape(solution_module.flatten_tensor(torch.ones(2, 2, 2)), (8,))


def test_is_1d(solution_module):
    assert solution_module.flatten_tensor(torch.ones(1, 4)).ndim == 1
