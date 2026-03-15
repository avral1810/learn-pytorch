from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_equal


def get_visible_tests():
    return [("removes column singleton", test_shape), ("keeps values", test_values)]


def get_hidden_tests():
    return [("works for one row", test_single), ("returns 1d tensor", test_vector_result)]


def test_shape(solution_module):
    assert_shape(solution_module.remove_singleton_dim(torch.tensor([[1], [2], [3]])), (3,))


def test_values(solution_module):
    actual = solution_module.remove_singleton_dim(torch.tensor([[4], [5]]))
    assert_tensor_equal(actual, torch.tensor([4, 5]))


def test_single(solution_module):
    assert_shape(solution_module.remove_singleton_dim(torch.tensor([[9]])), (1,))


def test_vector_result(solution_module):
    actual = solution_module.remove_singleton_dim(torch.tensor([[1.0], [2.0]]))
    assert actual.ndim == 1
