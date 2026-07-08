#!/usr/bin/env bash
set -e

PYTHON_VERSION="3.13"
ENV_PATH="${HOME}/venv_janusx"
PACKAGE="janusx"

if command -v uv >/dev/null 2>&1; then
  echo "[1/4] uv already installed: $(uv --version)"
else
  echo "[1/4] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"

echo "[2/4] Installing Python via uv..."
uv python install $PYTHON_VERSION >/dev/null 2>&1

echo "[3/4] Creating virtual environment at $ENV_PATH..."
uv venv "$ENV_PATH" --python $PYTHON_VERSION --clear

echo "[4/4] Installing janusx..."
DEV="${DEV:-0}"
if [ "$DEV" = "1" ]; then
  echo "Installing janusx from TestPyPI..."
    uv pip install --python "$ENV_PATH/bin/python" \
    --prerelease allow \
    --index-strategy unsafe-best-match \
    $PACKAGE \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/
else
  echo "Installing janusx from PyPI..."
  uv pip install --python "$ENV_PATH/bin/python" $PACKAGE
fi
$ENV_PATH/bin/jx -v
echo ""
echo "Done."
echo "Script:"
echo "$ENV_PATH/bin/jx"
