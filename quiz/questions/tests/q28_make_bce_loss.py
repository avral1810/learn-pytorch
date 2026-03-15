from __future__ import annotations

from torch import nn


def get_visible_tests():
    return [("returns BCEWithLogitsLoss", test_type)]


def get_hidden_tests():
    return [("callable on logits and targets", test_callable)]


def test_type(solution_module):
    assert isinstance(solution_module.make_bce_loss(), nn.BCEWithLogitsLoss)


def test_callable(solution_module):
    loss_fn = solution_module.make_bce_loss()
    value = loss_fn(nn.Parameter(nn.functional.pad(nn.Parameter.__new__(nn.Parameter), (0, 0))), None)
