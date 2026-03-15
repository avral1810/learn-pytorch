from __future__ import annotations

import torch


def get_visible_tests():
    return [
        ("returns a scalar tensor", test_scalar_tensor),
        ("enables requires_grad", test_requires_grad),
    ]


def get_hidden_tests():
    return [
        ("keeps the numeric value", test_keeps_value),
        ("supports backward", test_backward_works),
        ("returns floating type", test_float_tensor),
    ]


def test_scalar_tensor(solution_module):
    result = solution_module.build_trainable_scalar(2.5)
    assert isinstance(result, torch.Tensor), "Expected a torch.Tensor"
    assert result.ndim == 0, f"Expected a scalar tensor, got ndim={result.ndim}"


def test_requires_grad(solution_module):
    result = solution_module.build_trainable_scalar(1.0)
    assert result.requires_grad, "Expected requires_grad=True"


def test_keeps_value(solution_module):
    result = solution_module.build_trainable_scalar(7.0)
    assert torch.isclose(result, torch.tensor(7.0)), f"Expected value 7.0, got {result}"


def test_backward_works(solution_module):
    result = solution_module.build_trainable_scalar(3.0)
    loss = (result * 2 - 5) ** 2
    loss.backward()
    assert result.grad is not None, "Expected backward() to populate grad"


def test_float_tensor(solution_module):
    result = solution_module.build_trainable_scalar(4)
    assert result.dtype.is_floating_point, "Expected a floating point tensor"
