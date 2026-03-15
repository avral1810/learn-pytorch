from __future__ import annotations

from torch import nn


def get_visible_tests():
    return [("returns nn.Linear", test_type), ("uses 2 inputs and 1 output", test_features)]


def get_hidden_tests():
    return [("weight shape is (1, 2)", test_weight_shape)]


def test_type(solution_module):
    assert isinstance(solution_module.build_logistic_layer(), nn.Linear)


def test_features(solution_module):
    layer = solution_module.build_logistic_layer()
    assert layer.in_features == 2 and layer.out_features == 1


def test_weight_shape(solution_module):
    layer = solution_module.build_logistic_layer()
    assert tuple(layer.weight.shape) == (1, 2)
