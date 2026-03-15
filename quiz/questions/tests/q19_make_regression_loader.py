from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def get_visible_tests():
    return [("returns a DataLoader", test_type), ("uses batch size 32", test_batch_size)]


def get_hidden_tests():
    return [("shuffles data", test_shuffle), ("pairs x and y data", test_pairs)]


def test_type(solution_module):
    loader = solution_module.make_regression_loader(torch.randn(64, 1), torch.randn(64, 1))
    assert isinstance(loader, DataLoader)


def test_batch_size(solution_module):
    loader = solution_module.make_regression_loader(torch.randn(64, 1), torch.randn(64, 1))
    assert loader.batch_size == 32


def test_shuffle(solution_module):
    loader = solution_module.make_regression_loader(torch.randn(64, 1), torch.randn(64, 1))
    assert loader.sampler.__class__.__name__ == "RandomSampler"


def test_pairs(solution_module):
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    loader = solution_module.make_regression_loader(x, y)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape[1] == 1 and batch_y.shape[1] == 1
