from __future__ import annotations

from typing import Iterable

import torch


def assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual, expected), f"Expected {expected}, got {actual}"


def assert_tensor_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6) -> None:
    assert torch.allclose(actual, expected, atol=atol), f"Expected {expected}, got {actual}"


def assert_shape(actual: torch.Tensor, expected_shape: Iterable[int]) -> None:
    assert tuple(actual.shape) == tuple(expected_shape), (
        f"Expected shape {tuple(expected_shape)}, got {tuple(actual.shape)}"
    )


def assert_between_zero_and_one(actual: torch.Tensor) -> None:
    assert torch.all((actual >= 0) & (actual <= 1)), "Expected all values to be in [0, 1]"
