from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_tensor_equal


def get_visible_tests():
    return [("thresholds logits via sigmoid", test_basic), ("returns float predictions", test_float)]


def get_hidden_tests():
    return [("keeps shape", test_shape), ("handles zero logit", test_zero)]


def test_basic(solution_module):
    logits = torch.tensor([[-2.0], [2.0]])
    expected = torch.tensor([[0.0], [1.0]])
    assert_tensor_equal(solution_module.predict_binary_classes(logits), expected)


def test_float(solution_module):
    assert solution_module.predict_binary_classes(torch.tensor([0.0])).dtype.is_floating_point


def test_shape(solution_module):
    actual = solution_module.predict_binary_classes(torch.tensor([[0.0], [1.0], [-1.0]]))
    assert actual.shape == (3, 1)


def test_zero(solution_module):
    assert_tensor_equal(solution_module.predict_binary_classes(torch.tensor([0.0])), torch.tensor([1.0]))
