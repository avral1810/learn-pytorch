from __future__ import annotations

import torch


def get_visible_tests():
    return [("returns scalar loss", test_scalar), ("loss is low for good predictions", test_good_preds)]


def get_hidden_tests():
    return [("loss is higher for bad predictions", test_bad_preds), ("works on multiple examples", test_multi)]


def test_scalar(solution_module):
    loss = solution_module.binary_cross_entropy(torch.tensor([[0.8]]), torch.tensor([[1.0]]))
    assert loss.ndim == 0


def test_good_preds(solution_module):
    loss = solution_module.binary_cross_entropy(torch.tensor([[0.9], [0.1]]), torch.tensor([[1.0], [0.0]]))
    assert loss.item() < 0.2


def test_bad_preds(solution_module):
    good = solution_module.binary_cross_entropy(torch.tensor([[0.9]]), torch.tensor([[1.0]]))
    bad = solution_module.binary_cross_entropy(torch.tensor([[0.1]]), torch.tensor([[1.0]]))
    assert bad > good


def test_multi(solution_module):
    loss = solution_module.binary_cross_entropy(torch.tensor([[0.7], [0.3]]), torch.tensor([[1.0], [0.0]]))
    assert loss.ndim == 0
