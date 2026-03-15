from __future__ import annotations

import torch


def get_visible_tests():
    return [("returns scalar loss", test_scalar), ("matches simple expected loss", test_basic)]


def get_hidden_tests():
    return [("zero when equal", test_zero), ("works on column tensors", test_column)]


def test_scalar(solution_module):
    loss = solution_module.compute_mse_loss(torch.tensor([1.0]), torch.tensor([2.0]))
    assert loss.ndim == 0


def test_basic(solution_module):
    loss = solution_module.compute_mse_loss(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 3.0]))
    assert torch.isclose(loss, torch.tensor(1.0))


def test_zero(solution_module):
    loss = solution_module.compute_mse_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert torch.isclose(loss, torch.tensor(0.0))


def test_column(solution_module):
    loss = solution_module.compute_mse_loss(torch.tensor([[1.0], [3.0]]), torch.tensor([[1.0], [1.0]]))
    assert torch.isclose(loss, torch.tensor(2.0))
