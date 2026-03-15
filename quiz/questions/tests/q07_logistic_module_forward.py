from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_shape, assert_tensor_close


def get_visible_tests():
    return [
        ("returns shape (batch, 1)", test_output_shape),
        ("does not squash output to 0..1", test_returns_logits),
    ]


def get_hidden_tests():
    return [
        ("uses self.linear(x)", test_exact_linear_output),
        ("keeps the linear attribute", test_has_linear_layer),
        ("works for three examples", test_three_examples),
    ]


def build_model(solution_module):
    return solution_module.LogisticRegressionModel()


def test_output_shape(solution_module):
    model = build_model(solution_module)
    x = torch.randn(4, 2)
    actual = model(x)
    assert_shape(actual, (4, 1))


def test_returns_logits(solution_module):
    model = build_model(solution_module)
    with torch.no_grad():
        model.linear.weight.fill_(10.0)
        model.linear.bias.fill_(5.0)
    x = torch.tensor([[1.0, 1.0]])
    actual = model(x)
    assert actual.item() > 1.0, "Expected raw logits, not sigmoid probabilities"


def test_exact_linear_output(solution_module):
    model = build_model(solution_module)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
        model.linear.bias.fill_(0.5)
    x = torch.tensor([[3.0, 4.0]])
    expected = torch.tensor([[2.5]])
    assert_tensor_close(model(x), expected)


def test_has_linear_layer(solution_module):
    model = build_model(solution_module)
    assert hasattr(model, "linear"), "Expected the module to keep self.linear"


def test_three_examples(solution_module):
    model = build_model(solution_module)
    x = torch.randn(3, 2)
    actual = model(x)
    assert_shape(actual, (3, 1))
