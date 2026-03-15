from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory

from quiz.content.data import get_chapter, public_chapter_summaries
from quiz.questions.data import QUESTIONS_BY_ID, get_public_chapter_detail


ROOT_DIR = Path(__file__).resolve().parent.parent
QUIZ_DIR = ROOT_DIR / "quiz"
CONTENT_ASSET_DIR = QUIZ_DIR / "content" / "assets" / "chapters"
RUNNER_MODULE = "quiz.runner.run_submission"

app = Flask(__name__, template_folder=str(QUIZ_DIR / "templates"), static_folder=str(QUIZ_DIR / "static"))


def runner_python() -> str:
    local_python = QUIZ_DIR / ".venv" / "bin" / "python"
    if local_python.exists():
        return str(local_python)
    return sys.executable


def execute_submission(chapter_id: str, question_id: str, code: str, mode: str) -> tuple[dict[str, object], int]:
    chapter = get_chapter(chapter_id)
    question = QUESTIONS_BY_ID.get(question_id)

    if chapter is None:
        return {"ok": False, "error": f"Unknown chapter id: {chapter_id}"}, 404
    if question is None:
        return {"ok": False, "error": f"Unknown question id: {question_id}"}, 404
    if question.chapter_id != chapter_id or question_id not in chapter["question_ids"]:
        return {"ok": False, "error": "Question does not belong to the requested chapter."}, 400

    with tempfile.TemporaryDirectory(prefix="quiz-run-") as temp_dir:
        solution_path = Path(temp_dir) / "solution.py"
        solution_path.write_text(code, encoding="utf-8")
        command = [runner_python(), "-m", RUNNER_MODULE, question_id, mode, str(solution_path)]
        try:
            completed = subprocess.run(
                command,
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "mode": mode,
                "passed": 0,
                "total": 0,
                "results": [],
                "stdout": "",
                "stderr": "",
                "error": "Execution timed out after 8 seconds.",
            }, 200

    stdout = completed.stdout.strip()
    if not stdout:
        return {
            "ok": False,
            "mode": mode,
            "passed": 0,
            "total": 0,
            "results": [],
            "stdout": "",
            "stderr": completed.stderr,
            "error": "Runner returned no JSON output.",
        }, 200

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "mode": mode,
            "passed": 0,
            "total": 0,
            "results": [],
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "error": "Runner output could not be parsed as JSON.",
        }, 200

    return payload, 200


def execute_playground(code: str) -> tuple[dict[str, object], int]:
    with tempfile.TemporaryDirectory(prefix="quiz-playground-") as temp_dir:
        script_path = Path(temp_dir) / "playground.py"
        script_path.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [runner_python(), str(script_path)],
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": "", "error": "Playground execution timed out after 8 seconds."}, 200

    return {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "error": "" if completed.returncode == 0 else f"Exited with status {completed.returncode}",
    }, 200


@app.get("/")
def index():
    return render_template("home.html", chapters=public_chapter_summaries())


@app.get("/chapters/<chapter_id>")
def chapter_page(chapter_id: str):
    chapter = get_public_chapter_detail(chapter_id)
    if chapter is None:
        abort(404)
    return render_template("chapter.html", chapter=chapter, chapters=public_chapter_summaries())


@app.get("/api/chapters")
def chapters():
    return jsonify({"chapters": public_chapter_summaries()})


@app.get("/api/chapters/<chapter_id>")
def chapter_detail(chapter_id: str):
    chapter = get_public_chapter_detail(chapter_id)
    if chapter is None:
        return jsonify({"ok": False, "error": f"Unknown chapter id: {chapter_id}"}), 404
    return jsonify(chapter)


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "python": runner_python()})


@app.get("/chapter-assets/<path:filename>")
def chapter_assets(filename: str):
    return send_from_directory(CONTENT_ASSET_DIR, filename)


@app.get("/resources/<path:resource_path>")
def resource_file(resource_path: str):
    target = (ROOT_DIR / resource_path).resolve()
    root = ROOT_DIR.resolve()

    if not str(target).startswith(str(root)) or not target.exists() or not target.is_file():
        abort(404)

    return send_file(target)


@app.post("/api/playground")
def run_playground():
    payload = request.get_json(force=True)
    result, status = execute_playground(payload["code"])
    return jsonify(result), status


@app.post("/api/run")
def run_code():
    payload = request.get_json(force=True)
    result, status = execute_submission(payload["chapter_id"], payload["question_id"], payload["code"], "run")
    return jsonify(result), status


@app.post("/api/submit")
def submit_code():
    payload = request.get_json(force=True)
    result, status = execute_submission(payload["chapter_id"], payload["question_id"], payload["code"], "submit")
    return jsonify(result), status


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
