from __future__ import annotations

import torch


def get_visible_tests():
    return [("mean of simple vector", test_vector_mean), ("returns scalar", test_scalar)]


def get_hidden_tests():
    return [("works on matrix", test_matrix), ("handles floats", test_float_values)]


def test_vector_mean(solution_module):
    actual = solution_module.tensor_mean(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.isclose(actual, torch.tensor(2.0))


def test_scalar(solution_module):
    actual = solution_module.tensor_mean(torch.tensor([5.0, 7.0]))
    assert actual.ndim == 0, f"Expected scalar tensor, got ndim={actual.ndim}"


def test_matrix(solution_module):
    actual = solution_module.tensor_mean(torch.tensor([[1.0, 3.0], [5.0, 7.0]]))
    assert torch.isclose(actual, torch.tensor(4.0))


def test_float_values(solution_module):
    actual = solution_module.tensor_mean(torch.tensor([1.5, 2.5]))
    assert torch.isclose(actual, torch.tensor(2.0))
