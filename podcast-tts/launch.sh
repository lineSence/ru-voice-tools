#!/usr/bin/env bash
# RU Podcast TTS - launcher for Linux/macOS
set -e
cd "$(dirname "$0")"

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python

if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv .venv
fi

if [ ! -f .venv/.installed ]; then
  echo "Installing dependencies, first run takes 5-15 minutes..."
  .venv/bin/python -m pip install --upgrade pip --disable-pip-version-check
  .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check
  .venv/bin/python -m pip install -r requirements.txt --disable-pip-version-check
  touch .venv/.installed
fi

echo "Starting GUI... A browser tab will open at http://127.0.0.1:7860"
exec .venv/bin/python app.py