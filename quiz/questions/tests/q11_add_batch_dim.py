from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_equal


def get_visible_tests():
    return [("adds one front dimension", test_shape), ("keeps values", test_values)]


def get_hidden_tests():
    return [("works on 1d tensors", test_vector), ("works on 3d tensors", test_three_dim)]


def test_shape(solution_module):
    assert_shape(solution_module.add_batch_dim(torch.ones(3, 4)), (1, 3, 4))


def test_values(solution_module):
    actual = solution_module.add_batch_dim(torch.tensor([[1, 2], [3, 4]]))
    expected = torch.tensor([[[1, 2], [3, 4]]])
    assert_tensor_equal(actual, expected)


def test_vector(solution_module):
    assert_shape(solution_module.add_batch_dim(torch.tensor([1, 2, 3])), (1, 3))


def test_three_dim(solution_module):
    assert_shape(solution_module.add_batch_dim(torch.ones(2, 3, 4)), (1, 2, 3, 4))
