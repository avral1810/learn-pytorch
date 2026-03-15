from __future__ import annotations

from torch import nn


def get_visible_tests():
    return [("returns nn.Linear", test_type), ("has 1 in and 1 out feature", test_features)]


def get_hidden_tests():
    return [("weight shape is correct", test_weight_shape)]


def test_type(solution_module):
    assert isinstance(solution_module.build_linear_layer(), nn.Linear)


def test_features(solution_module):
    layer = solution_module.build_linear_layer()
    assert layer.in_features == 1 and layer.out_features == 1


def test_weight_shape(solution_module):
    layer = solution_module.build_linear_layer()
    assert tuple(layer.weight.shape) == (1, 1)
