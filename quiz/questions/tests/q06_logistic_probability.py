from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_between_zero_and_one, assert_shape, assert_tensor_close


def get_visible_tests():
    return [
        ("returns values between 0 and 1", test_probability_range),
        ("returns shape (batch, 1)", test_output_shape),
    ]


def get_hidden_tests():
    return [
        ("matches sigmoid(logits)", test_matches_sigmoid_formula),
        ("handles negative logits", test_negative_logits),
        ("works on multiple rows", test_multiple_rows),
    ]


def test_probability_range(solution_module):
    x = torch.tensor([[1.0, 2.0]])
    weight = torch.tensor([[1.0], [1.0]])
    bias = torch.tensor([0.0])
    actual = solution_module.logistic_probability(x, weight, bias)
    assert_between_zero_and_one(actual)


def test_output_shape(solution_module):
    x = torch.randn(5, 2)
    weight = torch.randn(2, 1)
    bias = torch.randn(1)
    actual = solution_module.logistic_probability(x, weight, bias)
    assert_shape(actual, (5, 1))


def test_matches_sigmoid_formula(solution_module):
    x = torch.tensor([[1.0, 0.0]])
    weight = torch.tensor([[2.0], [0.0]])
    bias = torch.tensor([-1.0])
    expected = torch.sigmoid(x @ weight + bias)
    assert_tensor_close(solution_module.logistic_probability(x, weight, bias), expected)


def test_negative_logits(solution_module):
    x = torch.tensor([[0.0, 1.0]])
    weight = torch.tensor([[0.0], [-4.0]])
    bias = torch.tensor([0.0])
    actual = solution_module.logistic_probability(x, weight, bias)
    assert_between_zero_and_one(actual)


def test_multiple_rows(solution_module):
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weight = torch.tensor([[0.1], [0.2]])
    bias = torch.tensor([0.3])
    expected = torch.sigmoid(x @ weight + bias)
    assert_tensor_close(solution_module.logistic_probability(x, weight, bias), expected)
