from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_equal


def get_visible_tests():
    return [("computes preds-targets", test_basic), ("keeps original shape", test_shape)]


def get_hidden_tests():
    return [("handles negative residuals", test_negative), ("handles column tensors", test_column)]


def test_basic(solution_module):
    assert_tensor_equal(solution_module.compute_residuals(torch.tensor([3, 5]), torch.tensor([1, 2])), torch.tensor([2, 3]))


def test_shape(solution_module):
    actual = solution_module.compute_residuals(torch.ones(4, 1), torch.zeros(4, 1))
    assert_shape(actual, (4, 1))


def test_negative(solution_module):
    assert_tensor_equal(solution_module.compute_residuals(torch.tensor([1, 2]), torch.tensor([3, 1])), torch.tensor([-2, 1]))


def test_column(solution_module):
    actual = solution_module.compute_residuals(torch.tensor([[1.0], [2.0]]), torch.tensor([[0.5], [3.0]]))
    assert_tensor_equal(actual, torch.tensor([[0.5], [-1.0]]))
