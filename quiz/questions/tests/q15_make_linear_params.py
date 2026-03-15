from __future__ import annotations

from quiz.runner.test_utils import assert_shape


def get_visible_tests():
    return [("weight has shape (1, 1)", test_weight_shape), ("bias has shape (1,)", test_bias_shape)]


def get_hidden_tests():
    return [("weight requires grad", test_weight_grad), ("bias requires grad", test_bias_grad)]


def test_weight_shape(solution_module):
    weight, _ = solution_module.make_linear_params()
    assert_shape(weight, (1, 1))


def test_bias_shape(solution_module):
    _, bias = solution_module.make_linear_params()
    assert_shape(bias, (1,))


def test_weight_grad(solution_module):
    weight, _ = solution_module.make_linear_params()
    assert weight.requires_grad


def test_bias_grad(solution_module):
    _, bias = solution_module.make_linear_params()
    assert bias.requires_grad
