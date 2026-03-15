from __future__ import annotations

import importlib
import importlib.util
import io
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from quiz.questions.data import QUESTIONS_BY_ID


def load_solution_module(solution_path: Path):
    spec = importlib.util.spec_from_file_location("learner_solution", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load solution module from {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_tests(test_module_name: str, solution_path: Path, mode: str) -> dict[str, object]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        solution_module = load_solution_module(solution_path)
        test_module = importlib.import_module(test_module_name)

        if mode == "run":
            tests = test_module.get_visible_tests()
        else:
            tests = test_module.get_hidden_tests()

        results: list[dict[str, object]] = []
        passed = 0
        for name, test_fn in tests:
            try:
                test_fn(solution_module)
                results.append({"name": name, "passed": True})
                passed += 1
            except AssertionError as exc:
                results.append({"name": name, "passed": False, "message": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive path
                results.append(
                    {
                        "name": name,
                        "passed": False,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )

    return {
        "ok": passed == len(tests),
        "mode": mode,
        "passed": passed,
        "total": len(tests),
        "results": results,
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
    }


def main() -> int:
    if len(sys.argv) != 4:
        print(json.dumps({"ok": False, "error": "Usage: question_id mode solution_path"}))
        return 1

    question_id, mode, solution_path_text = sys.argv[1:]
    question = QUESTIONS_BY_ID.get(question_id)
    if question is None:
        print(json.dumps({"ok": False, "error": f"Unknown question id: {question_id}"}))
        return 1

    solution_path = Path(solution_path_text)

    try:
        payload = run_tests(question["test_module"], solution_path, mode)
        print(json.dumps(payload))
        return 0 if payload["ok"] else 0
    except Exception as exc:
        payload = {
            "ok": False,
            "mode": mode,
            "passed": 0,
            "total": 0,
            "results": [],
            "stdout": "",
            "stderr": traceback.format_exc(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(payload))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
