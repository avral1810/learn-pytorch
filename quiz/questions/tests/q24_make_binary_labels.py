from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_equal


def get_visible_tests():
    return [("positive scores become 1", test_values), ("returns shape (n, 1)", test_shape)]


def get_hidden_tests():
    return [("zero is not positive", test_zero), ("returns float tensor", test_dtype)]


def test_values(solution_module):
    scores = torch.tensor([-1.0, 2.0, 3.0])
    expected = torch.tensor([[0.0], [1.0], [1.0]])
    assert_tensor_equal(solution_module.make_binary_labels(scores), expected)


def test_shape(solution_module):
    assert_shape(solution_module.make_binary_labels(torch.tensor([1.0, -1.0])), (2, 1))


def test_zero(solution_module):
    expected = torch.tensor([[0.0], [1.0]])
    assert_tensor_equal(solution_module.make_binary_labels(torch.tensor([0.0, 0.1])), expected)


def test_dtype(solution_module):
    assert solution_module.make_binary_labels(torch.tensor([1.0])).dtype.is_floating_point
