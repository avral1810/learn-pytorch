from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_tensor_equal


def get_visible_tests():
    return [("thresholds at 0.5", test_basic), ("returns floats", test_float)]


def get_hidden_tests():
    return [("keeps shape", test_shape), ("handles edge case 0.5", test_edge)]


def test_basic(solution_module):
    probs = torch.tensor([[0.2], [0.7]])
    assert_tensor_equal(solution_module.threshold_predictions(probs), torch.tensor([[0.0], [1.0]]))


def test_float(solution_module):
    actual = solution_module.threshold_predictions(torch.tensor([0.3, 0.9]))
    assert actual.dtype.is_floating_point


def test_shape(solution_module):
    actual = solution_module.threshold_predictions(torch.tensor([[0.1], [0.2], [0.9]]))
    assert actual.shape == (3, 1)


def test_edge(solution_module):
    actual = solution_module.threshold_predictions(torch.tensor([0.5]))
    assert_tensor_equal(actual, torch.tensor([1.0]))
