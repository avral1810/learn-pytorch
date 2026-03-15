from __future__ import annotations

import torch


def get_visible_tests():
    return [("sigmoid(0) is 0.5", test_zero), ("outputs stay in range", test_range)]


def get_hidden_tests():
    return [("matches torch.sigmoid", test_matches_torch), ("works on vectors", test_vector)]


def test_zero(solution_module):
    actual = solution_module.sigmoid(torch.tensor(0.0))
    assert torch.isclose(actual, torch.tensor(0.5))


def test_range(solution_module):
    actual = solution_module.sigmoid(torch.tensor([-10.0, 0.0, 10.0]))
    assert torch.all((actual >= 0) & (actual <= 1))


def test_matches_torch(solution_module):
    z = torch.tensor([-2.0, 1.0, 3.0])
    assert torch.allclose(solution_module.sigmoid(z), torch.sigmoid(z))


def test_vector(solution_module):
    assert solution_module.sigmoid(torch.tensor([1.0, 2.0])).shape == (2,)
