#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUIZ_DIR="$ROOT_DIR/quiz"
VENV_DIR="$QUIZ_DIR/.venv"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Missing quiz virtual environment at $VENV_DIR" >&2
  echo "Run: bash quiz/setup.sh" >&2
  exit 1
fi

cd "$ROOT_DIR"
"$VENV_DIR/bin/python" quiz/app.py
