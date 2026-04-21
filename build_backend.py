from __future__ import annotations

import base64
import csv
import hashlib
import importlib
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

_MATURIN = importlib.import_module("maturin")


def _env_flag(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _build_kmc_extension_for_wheel() -> Path | None:
    if str(os.environ.get("JANUSX_PREBUILD_KMC_BIND", "")).strip() == "0":
        return None
    py_src = _project_root() / "python"
    if not py_src.is_dir():
        return None
    if str(py_src) not in sys.path:
        sys.path.insert(0, str(py_src))
    try:
        from janusx.kmc_bind import build_kmc_bind_module  # type: ignore

        kmc_src_raw = str(os.environ.get("JANUSX_KMC_SRC", "")).strip()
        ext_path = build_kmc_bind_module(
            kmc_src=kmc_src_raw or None,
            rebuild=_env_flag("JANUSX_KMC_REBUILD"),
            verbose=_env_flag("JANUSX_BUILD_VERBOSE"),
        )
        return Path(ext_path)
    except Exception as exc:
        if _env_flag("JANUSX_STRICT_KMC_BIND"):
            raise
        print(
            f"[build-backend] warning: prebuild _kmc_count failed ({exc}); "
            "wheel will fallback to runtime compilation path.",
            flush=True,
        )
        return None


def _sha256_record(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    digest = base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode("ascii")
    return f"sha256={digest}", int(size)


def _rewrite_record(extract_root: Path) -> None:
    dist_infos = [p for p in extract_root.glob("*.dist-info") if p.is_dir()]
    if len(dist_infos) != 1:
        raise RuntimeError(f"Expected one *.dist-info directory, got {len(dist_infos)}")
    dist = dist_infos[0]
    record = dist / "RECORD"

    rows: list[list[str]] = []
    files = sorted([p for p in extract_root.rglob("*") if p.is_file()])
    for p in files:
        rel = p.relative_to(extract_root).as_posix()
        if rel == f"{dist.name}/RECORD":
            continue
        h, sz = _sha256_record(p)
        rows.append([rel, h, str(sz)])
    rows.append([f"{dist.name}/RECORD", "", ""])

    with record.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)


def _inject_extension_into_wheel(wheel_path: Path, ext_path: Path) -> None:
    if not ext_path.is_file():
        return

    with tempfile.TemporaryDirectory(prefix="janusx_wheel_edit_") as td:
        root = Path(td) / "wheel"
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheel_path, "r") as zf:
            zf.extractall(root)

        target = root / "janusx" / ext_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ext_path, target)

        _rewrite_record(root)

        rebuilt = wheel_path.with_suffix(".tmp.whl")
        with zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    arc = p.relative_to(root).as_posix()
                    zf.write(p, arcname=arc)
        rebuilt.replace(wheel_path)


def get_requires_for_build_wheel(config_settings: dict[str, Any] | None = None) -> list[str]:
    return _MATURIN.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings: dict[str, Any] | None = None) -> list[str]:
    return _MATURIN.get_requires_for_build_sdist(config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return _MATURIN.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def get_requires_for_build_editable(config_settings: dict[str, Any] | None = None) -> list[str]:
    fn = getattr(_MATURIN, "get_requires_for_build_editable", None)
    if fn is None:
        return get_requires_for_build_wheel(config_settings)
    return fn(config_settings)


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    fn = getattr(_MATURIN, "prepare_metadata_for_build_editable", None)
    if fn is None:
        return prepare_metadata_for_build_wheel(metadata_directory, config_settings)
    return fn(metadata_directory, config_settings)


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return _MATURIN.build_sdist(sdist_directory, config_settings)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    fn = getattr(_MATURIN, "build_editable", None)
    if fn is None:
        return build_wheel(wheel_directory, config_settings, metadata_directory)
    return fn(wheel_directory, config_settings, metadata_directory)


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    ext_path = _build_kmc_extension_for_wheel()

    prev = os.environ.get("JANUSX_PREBUILD_KMC_BIND")
    os.environ["JANUSX_PREBUILD_KMC_BIND"] = "0"
    try:
        wheel_name = _MATURIN.build_wheel(wheel_directory, config_settings, metadata_directory)
    finally:
        if prev is None:
            os.environ.pop("JANUSX_PREBUILD_KMC_BIND", None)
        else:
            os.environ["JANUSX_PREBUILD_KMC_BIND"] = prev

    if ext_path is not None:
        wheel_path = Path(wheel_directory).resolve() / wheel_name
        _inject_extension_into_wheel(wheel_path, ext_path)
        print(f"[build-backend] injected extension: janusx/{ext_path.name}", flush=True)
    return wheel_name
