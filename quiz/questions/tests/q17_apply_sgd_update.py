from __future__ import annotations

import torch


def get_visible_tests():
    return [("updates weight using grad", test_weight_update), ("updates bias using grad", test_bias_update)]


def get_hidden_tests():
    return [("keeps requires_grad", test_requires_grad), ("uses learning rate scale", test_lr_scale)]


def test_weight_update(solution_module):
    weight = torch.tensor([[2.0]], requires_grad=True)
    bias = torch.tensor([1.0], requires_grad=True)
    weight.grad = torch.tensor([[0.5]])
    bias.grad = torch.tensor([0.2])
    new_weight, _ = solution_module.apply_sgd_update(weight, bias, 0.1)
    assert torch.allclose(new_weight, torch.tensor([[1.95]]))


def test_bias_update(solution_module):
    weight = torch.tensor([[2.0]], requires_grad=True)
    bias = torch.tensor([1.0], requires_grad=True)
    weight.grad = torch.tensor([[0.5]])
    bias.grad = torch.tensor([0.2])
    _, new_bias = solution_module.apply_sgd_update(weight, bias, 0.1)
    assert torch.allclose(new_bias, torch.tensor([0.98]))


def test_requires_grad(solution_module):
    weight = torch.tensor([[2.0]], requires_grad=True)
    bias = torch.tensor([1.0], requires_grad=True)
    weight.grad = torch.tensor([[1.0]])
    bias.grad = torch.tensor([1.0])
    new_weight, new_bias = solution_module.apply_sgd_update(weight, bias, 0.01)
    assert new_weight.requires_grad and new_bias.requires_grad


def test_lr_scale(solution_module):
    weight = torch.tensor([[1.0]], requires_grad=True)
    bias = torch.tensor([0.0], requires_grad=True)
    weight.grad = torch.tensor([[2.0]])
    bias.grad = torch.tensor([0.0])
    new_weight, _ = solution_module.apply_sgd_update(weight, bias, 0.5)
    assert torch.allclose(new_weight, torch.tensor([[0.0]]))
