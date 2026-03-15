from __future__ import annotations

import torch


def get_visible_tests():
    return [
        ("returns zero for identical tensors", test_zero_error),
        ("returns a scalar tensor", test_scalar_output),
    ]


def get_hidden_tests():
    return [
        ("matches manual mse computation", test_manual_computation),
        ("handles multiple examples", test_multiple_examples),
        ("penalizes larger errors more", test_larger_errors),
    ]


def test_zero_error(solution_module):
    preds = torch.tensor([1.0, 2.0])
    targets = torch.tensor([1.0, 2.0])
    loss = solution_module.mean_squared_error(preds, targets)
    assert torch.isclose(loss, torch.tensor(0.0)), f"Expected 0.0, got {loss}"


def test_scalar_output(solution_module):
    loss = solution_module.mean_squared_error(torch.tensor([1.0]), torch.tensor([2.0]))
    assert isinstance(loss, torch.Tensor), "Expected a tensor result"
    assert loss.ndim == 0, f"Expected a scalar tensor, got ndim={loss.ndim}"


def test_manual_computation(solution_module):
    preds = torch.tensor([2.0, 4.0])
    targets = torch.tensor([1.0, 3.0])
    expected = torch.tensor(1.0)
    assert torch.isclose(solution_module.mean_squared_error(preds, targets), expected)


def test_multiple_examples(solution_module):
    preds = torch.tensor([[1.0], [2.0], [3.0]])
    targets = torch.tensor([[0.0], [2.0], [4.0]])
    expected = torch.tensor((1.0 + 0.0 + 1.0) / 3.0)
    assert torch.isclose(solution_module.mean_squared_error(preds, targets), expected)


def test_larger_errors(solution_module):
    small = solution_module.mean_squared_error(torch.tensor([2.0]), torch.tensor([1.0]))
    large = solution_module.mean_squared_error(torch.tensor([5.0]), torch.tensor([1.0]))
    assert large > small, "Expected larger mistakes to produce a larger MSE"
