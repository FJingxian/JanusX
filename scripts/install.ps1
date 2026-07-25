# =========================
# config
# =========================
$PythonVersion = "3.13"
$EnvPath = "$HOME\venv_janusx"
$Package = "janusx"

$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "[1/4] uv already installed: $(uv --version)"
} 
else {
    Write-Host "[1/4] Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex *>$null
}

Write-Host "[2/4] Installing Python via uv..."
uv python install $PythonVersion

Write-Host "[3/4] Creating venv..."
uv venv $EnvPath --python $PythonVersion --clear

Write-Host "[4/4] Installing janusx..."
$Dev = if ([string]::IsNullOrEmpty($env:DEV)) { "0" } else { $env:DEV }
if ($Dev -eq "1") {
    Write-Host "Installing janusx from TestPyPI..."
    $InstallArgs = @(
        "pip", "install",
        "--python", "$EnvPath\Scripts\python.exe",
        "--prerelease", "allow",
        "--index-strategy", "unsafe-best-match",
        $Package,
        "--index-url", "https://test.pypi.org/simple/",
        "--extra-index-url", "https://pypi.org/simple/"
    )
    uv @InstallArgs
}
else {
    Write-Host "Installing janusx from PyPI..."
    uv pip install --python "$EnvPath\Scripts\python.exe" $Package
}

$JxPath = Join-Path $EnvPath "Scripts\jx.exe"
& $JxPath -v
Write-Host ""
Write-Host "Done."
Write-Host "Run with:"
Write-Host "$JxPath -h"
