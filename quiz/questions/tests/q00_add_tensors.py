from __future__ import annotations

import torch

from quiz.runner.test_utils import assert_tensor_equal, assert_shape


def get_visible_tests():
    return [
        ("adds a 2x2 matrix", test_adds_2x2),
        ("keeps the correct shape", test_shape_is_preserved),
    ]


def get_hidden_tests():
    return [
        ("adds negative values too", test_negative_values),
        ("adds float tensors", test_float_values),
        ("works on 1d tensors", test_vector_addition),
    ]


def test_adds_2x2(solution_module):
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    expected = torch.tensor([[6, 8], [10, 12]])
    assert_tensor_equal(solution_module.add_tensors(a, b), expected)


def test_shape_is_preserved(solution_module):
    actual = solution_module.add_tensors(torch.ones(3, 2), torch.zeros(3, 2))
    assert_shape(actual, (3, 2))


def test_negative_values(solution_module):
    actual = solution_module.add_tensors(torch.tensor([-1, 4]), torch.tensor([3, -2]))
    expected = torch.tensor([2, 2])
    assert_tensor_equal(actual, expected)


def test_float_values(solution_module):
    actual = solution_module.add_tensors(torch.tensor([1.5, 2.5]), torch.tensor([0.5, 1.5]))
    expected = torch.tensor([2.0, 4.0])
    assert_tensor_equal(actual, expected)


def test_vector_addition(solution_module):
    actual = solution_module.add_tensors(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
    expected = torch.tensor([5, 7, 9])
    assert_tensor_equal(actual, expected)
