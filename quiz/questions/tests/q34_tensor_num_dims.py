from __future__ import annotations

import torch


def get_visible_tests():
    return [
        ("vector has 1 dimension", test_vector_dim),
        ("matrix has 2 dimensions", test_matrix_dim),
    ]


def get_hidden_tests():
    return [
        ("scalar has 0 dimensions", test_scalar_dim),
        ("3d tensor has 3 dimensions", test_three_dim_tensor),
        ("returns a Python int", test_returns_int),
    ]


def test_vector_dim(solution_module):
    x = torch.tensor([1, 2, 3])
    actual = solution_module.num_dims(x)
    assert actual == 1, f"Expected 1, got {actual}"


def test_matrix_dim(solution_module):
    x = torch.tensor([[1, 2], [3, 4]])
    actual = solution_module.num_dims(x)
    assert actual == 2, f"Expected 2, got {actual}"


def test_scalar_dim(solution_module):
    x = torch.tensor(5)
    actual = solution_module.num_dims(x)
    assert actual == 0, f"Expected 0, got {actual}"


def test_three_dim_tensor(solution_module):
    x = torch.zeros(2, 3, 4)
    actual = solution_module.num_dims(x)
    assert actual == 3, f"Expected 3, got {actual}"


def test_returns_int(solution_module):
    x = torch.zeros(1)
    actual = solution_module.num_dims(x)
    assert isinstance(actual, int), f"Expected int, got {type(actual)}"
