from __future__ import annotations

import torch
from torch import nn


def get_visible_tests():
    return [("returns a loss tensor", test_returns_loss), ("updates parameters", test_updates_params)]


def get_hidden_tests():
    return [("loss is scalar", test_scalar), ("handles batch input", test_batch)]


def build_case():
    model = nn.Linear(1, 1)
    batch_x = torch.tensor([[1.0], [2.0], [3.0]])
    batch_y = torch.tensor([[2.0], [4.0], [6.0]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, batch_x, batch_y, optimizer


def test_returns_loss(solution_module):
    model, batch_x, batch_y, optimizer = build_case()
    loss = solution_module.run_training_step(model, batch_x, batch_y, optimizer)
    assert isinstance(loss, torch.Tensor)


def test_updates_params(solution_module):
    model, batch_x, batch_y, optimizer = build_case()
    before = model.weight.detach().clone()
    solution_module.run_training_step(model, batch_x, batch_y, optimizer)
    after = model.weight.detach().clone()
    assert not torch.equal(before, after), "Expected model parameters to change after optimizer.step()"


def test_scalar(solution_module):
    model, batch_x, batch_y, optimizer = build_case()
    loss = solution_module.run_training_step(model, batch_x, batch_y, optimizer)
    assert loss.ndim == 0


def test_batch(solution_module):
    model, batch_x, batch_y, optimizer = build_case()
    solution_module.run_training_step(model, batch_x, batch_y, optimizer)
