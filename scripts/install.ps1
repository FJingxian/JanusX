# =========================
# config
# =========================
$PythonVersion = "3.13"
$EnvPath = "$HOME\venv_janusx"
$Package = "janusx"

Write-Host "[1/4] Installing uv..."

irm https://astral.sh/uv/install.ps1 | iex

$env:Path += ";$env:USERPROFILE\.local\bin"

Write-Host "[2/4] Installing Python via uv..."
uv python install $PythonVersion

Write-Host "[3/4] Creating venv..."
uv venv $EnvPath --python $PythonVersion

Write-Host "[4/4] Installing janusx..."
uv pip install --python "$EnvPath\Scripts\python.exe" $Package

Write-Host ""
Write-Host "Done."
Write-Host "Activate with:"
Write-Host "$EnvPath\Scripts\activate"