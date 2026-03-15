from __future__ import annotations

import json
import zlib
from pathlib import Path

import torch

from quiz.runner.test_utils import assert_between_zero_and_one, assert_shape, assert_tensor_close, assert_tensor_equal

HIDDEN_CASES_PATH = Path(__file__).with_name("hidden_cases.bin")


def load_hidden_case_map() -> dict[str, list[dict[str, object]]]:
    if not HIDDEN_CASES_PATH.exists():
        return {}
    raw = HIDDEN_CASES_PATH.read_bytes()
    payload = zlib.decompress(raw).decode("utf-8")
    return json.loads(payload)


def _decode_value(value):
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if isinstance(value, dict):
        if value.get("__kind__") == "tensor":
            dtype_name = value.get("dtype")
            dtype = getattr(torch, dtype_name, None) if dtype_name else None
            tensor = torch.tensor(value["value"], dtype=dtype)
            if value.get("requires_grad"):
                tensor.requires_grad_()
            grad_value = value.get("grad")
            if grad_value is not None:
                tensor.grad = _decode_value(grad_value)
            return tensor
        return {key: _decode_value(item) for key, item in value.items()}
    return value


def _assert_type_name(actual, expected: str) -> None:
    actual_name = f"{type(actual).__module__}.{type(actual).__name__}"
    assert actual_name == expected, f"Expected type {expected}, got {actual_name}"


def _assert_expr(actual, expression: str) -> None:
    ok = eval(expression, {"torch": torch}, {"actual": actual})
    assert ok, f"Expression failed: {expression}"


def _apply_assertion(actual, assertion: dict[str, object]) -> object:
    kind = assertion["kind"]
    expected = _decode_value(assertion.get("expected"))
    if kind == "tensor_equal":
        assert_tensor_equal(actual, expected)
    elif kind == "tensor_close":
        assert_tensor_close(actual, expected, atol=assertion.get("atol", 1e-6))
    elif kind == "shape":
        assert_shape(actual, expected)
    elif kind == "equals":
        assert actual == expected, f"Expected {expected}, got {actual}"
    elif kind == "between_zero_and_one":
        assert_between_zero_and_one(actual)
    elif kind == "type_name":
        _assert_type_name(actual, expected)
    elif kind == "has_attr":
        assert hasattr(actual, expected), f"Expected attribute {expected}"
    elif kind == "attr_equals":
        attr_name = assertion["attr"]
        actual_value = getattr(actual, attr_name)
        assert actual_value == expected, f"Expected {attr_name}={expected}, got {actual_value}"
    elif kind == "attr_shape":
        attr_name = assertion["attr"]
        actual_value = getattr(actual, attr_name)
        assert tuple(actual_value.shape) == tuple(expected), f"Expected {attr_name}.shape={tuple(expected)}, got {tuple(actual_value.shape)}"
    elif kind == "callable":
        assert callable(actual), "Expected a callable object"
    elif kind == "sequence_length":
        assert len(actual) == expected, f"Expected length {expected}, got {len(actual)}"
    elif kind == "requires_grad":
        actual_value = bool(getattr(actual, "requires_grad", False))
        assert actual_value is expected, f"Expected requires_grad={expected}, got {actual_value}"
    elif kind == "item_compare":
        actual_item = actual.item() if hasattr(actual, "item") else actual
        operator = assertion["op"]
        if operator == "gt":
            assert actual_item > expected, f"Expected {actual_item} > {expected}"
        elif operator == "lt":
            assert actual_item < expected, f"Expected {actual_item} < {expected}"
        elif operator == "eq":
            assert actual_item == expected, f"Expected {actual_item} == {expected}"
        else:
            raise ValueError(f"Unknown comparison operator: {operator}")
    elif kind == "expr":
        _assert_expr(actual, str(assertion["expression"]))
    else:
        raise ValueError(f"Unknown assertion kind: {kind}")
    return expected


def build_hidden_tests(question_id: str, symbol_name: str):
    hidden_case_map = load_hidden_case_map()
    cases = hidden_case_map.get(question_id, [])
    tests = []

    for case in cases:
        name = case["name"]
        hint = case.get("hint", "Compare your result carefully to the expected behavior for this test.")
        args = _decode_value(case.get("args", []))
        kwargs = _decode_value(case.get("kwargs", {}))
        assertions = case.get("assertions", [])

        def make_test(case_args=args, case_kwargs=kwargs, case_assertions=assertions, symbol=symbol_name):
            def test_fn(solution_module):
                target = getattr(solution_module, symbol)
                inputs = case_args
                kwargs = case_kwargs
                actual = target(*inputs, **kwargs)
                expected = None
                for assertion in case_assertions:
                    expected = _apply_assertion(actual, assertion)
                return actual, expected

            return test_fn

        tests.append((name, make_test(), {"hint": hint}))

    return tests
