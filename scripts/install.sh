#!/usr/bin/env bash
set -e

PYTHON_VERSION="3.13"
ENV_PATH="${HOME}/venv_janusx"
PACKAGE="janusx"

echo "[1/4] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"

echo "[2/4] Installing Python via uv..."
uv python install $PYTHON_VERSION >/dev/null 2>&1

echo "[3/4] Creating virtual environment at $ENV_PATH..."
uv venv "$ENV_PATH" --python $PYTHON_VERSION --clear

echo "[4/4] Installing janusx..."
uv pip install --python "$ENV_PATH/bin/python" $PACKAGE >/dev/null 2>&1

echo ""
echo "Done."
echo "Script:"
echo "$ENV_PATH/bin/jx"