# JanusX Rust Launcher (`jx`)

This launcher is a standalone Rust binary that can replace the Python console entrypoint.

## What it does

- Creates and reuses a dedicated runtime venv
- Auto-installs `janusx` from PyPI when missing
- Supports update from PyPI/GitHub
- Forwards all module commands to:
  - `python -m janusx.script.JanusX ...`

## Build

```bash
cargo build --release --manifest-path launcher/Cargo.toml
```

Binary output:

- `launcher/target/release/jx` (Linux/macOS)
- `launcher/target/release/jx.exe` (Windows)

## Usage

```bash
jx --update
jx --update latest
jx --update latest --verbose
jx gwas -h
```

## Runtime location

Default runtime venv:

- Linux/macOS: `~/.janusx/venv`
- Windows: `%LOCALAPPDATA%/JanusX/venv`

Optional env vars:

- `JX_HOME`: custom runtime home
- `JX_PYTHON`: system Python used to create the venv
