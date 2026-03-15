from __future__ import annotations

import torch


def get_visible_tests():
    return [("outputs lie in [0,1]", test_range), ("keeps same shape", test_shape)]


def get_hidden_tests():
    return [("matches torch.sigmoid", test_matches), ("works on scalar", test_scalar)]


def test_range(solution_module):
    actual = solution_module.logits_to_probs(torch.tensor([-2.0, 0.0, 2.0]))
    assert torch.all((actual >= 0) & (actual <= 1))


def test_shape(solution_module):
    logits = torch.randn(4, 1)
    assert solution_module.logits_to_probs(logits).shape == (4, 1)


def test_matches(solution_module):
    logits = torch.tensor([-1.0, 1.0])
    assert torch.allclose(solution_module.logits_to_probs(logits), torch.sigmoid(logits))


def test_scalar(solution_module):
    actual = solution_module.logits_to_probs(torch.tensor(0.0))
    assert torch.isclose(actual, torch.tensor(0.5))
