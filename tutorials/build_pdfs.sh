#!/bin/bash

set -euo pipefail

venv_python="/Users/aviral/Desktop/MachineLearning/learn-pytorch/.pdf-venv/bin/python"

if [ ! -x "$venv_python" ]; then
  echo "Missing PDF virtualenv at $venv_python" >&2
  echo "Create it with: python3 -m venv .pdf-venv && .pdf-venv/bin/python -m pip install reportlab pillow" >&2
  exit 1
fi

"$venv_python" tutorials/build_pdfs.py
