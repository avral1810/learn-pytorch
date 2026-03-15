#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUIZ_DIR="$ROOT_DIR/quiz"
VENV_DIR="$QUIZ_DIR/.venv"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install --upgrade -r "$QUIZ_DIR/requirements.txt"

echo "Quiz environment created at $VENV_DIR"
echo "Start the app with: bash quiz/run.sh"
