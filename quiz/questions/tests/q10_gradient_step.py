from __future__ import annotations

import torch


def get_visible_tests():
    return [("moves weight closer to 3", test_moves_toward_target), ("returns tensor", test_returns_tensor)]


def get_hidden_tests():
    return [("keeps requires_grad", test_requires_grad), ("works from another start", test_other_start)]


def test_moves_toward_target(solution_module):
    weight = torch.tensor(2.0, requires_grad=True)
    updated = solution_module.gradient_step(weight, 0.1)
    assert updated.item() > 2.0, "Expected weight to move upward toward 3.0"


def test_returns_tensor(solution_module):
    updated = solution_module.gradient_step(torch.tensor(2.0, requires_grad=True), 0.1)
    assert isinstance(updated, torch.Tensor), "Expected a tensor result"


def test_requires_grad(solution_module):
    updated = solution_module.gradient_step(torch.tensor(2.5, requires_grad=True), 0.1)
    assert updated.requires_grad, "Expected updated tensor to keep requires_grad"


def test_other_start(solution_module):
    updated = solution_module.gradient_step(torch.tensor(4.0, requires_grad=True), 0.1)
    assert updated.item() < 4.0, "Expected weight to move downward toward 3.0"
