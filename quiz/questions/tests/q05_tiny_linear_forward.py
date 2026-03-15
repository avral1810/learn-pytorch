from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_close


def get_visible_tests():
    return [
        ("returns shape (batch, 1)", test_output_shape),
        ("uses the stored linear layer", test_uses_linear_layer),
    ]


def get_hidden_tests():
    return [
        ("works with three rows", test_three_rows),
        ("returns raw layer output", test_exact_output),
        ("keeps module structure", test_has_linear_attribute),
    ]


def build_model(solution_module):
    return solution_module.TinyLinearModel()


def test_output_shape(solution_module):
    model = build_model(solution_module)
    x = torch.randn(4, 1)
    actual = model(x)
    assert_shape(actual, (4, 1))


def test_uses_linear_layer(solution_module):
    model = build_model(solution_module)
    with torch.no_grad():
        model.linear.weight.fill_(2.0)
        model.linear.bias.fill_(1.0)
    x = torch.tensor([[3.0]])
    actual = model(x)
    expected = torch.tensor([[7.0]])
    assert_tensor_close(actual, expected)


def test_three_rows(solution_module):
    model = build_model(solution_module)
    x = torch.tensor([[1.0], [2.0], [3.0]])
    actual = model(x)
    assert_shape(actual, (3, 1))


def test_exact_output(solution_module):
    model = build_model(solution_module)
    with torch.no_grad():
        model.linear.weight.fill_(1.5)
        model.linear.bias.fill_(0.5)
    x = torch.tensor([[2.0], [4.0]])
    expected = torch.tensor([[3.5], [6.5]])
    assert_tensor_close(model(x), expected)


def test_has_linear_attribute(solution_module):
    model = build_model(solution_module)
    assert hasattr(model, "linear"), "Expected the module to keep self.linear"
