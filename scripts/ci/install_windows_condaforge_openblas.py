#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import re
import tarfile
import tempfile
import urllib.request
import zipfile


def _parse_ver_tuple(v: str) -> tuple[int, ...]:
    nums = []
    for tok in str(v).split("."):
        m = re.match(r"^(\d+)", tok)
        if m is None:
            nums.append(0)
        else:
            nums.append(int(m.group(1)))
    return tuple(nums)


def _iter_repodata_records(repodata: dict) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for sec in ("packages.conda", "packages"):
        sec_map = repodata.get(sec, {})
        if isinstance(sec_map, dict):
            for fn, meta in sec_map.items():
                if isinstance(meta, dict):
                    rows.append((str(fn), meta))
    return rows


def _pick_openblas_package(repodata: dict, flavor: str) -> tuple[str, dict]:
    cands: list[tuple[tuple[int, ...], int, str, dict]] = []
    for fn, meta in _iter_repodata_records(repodata):
        if meta.get("name") != "openblas":
            continue
        build = str(meta.get("build", ""))
        if flavor and (flavor not in build):
            continue
        ver = _parse_ver_tuple(str(meta.get("version", "0")))
        bn = int(meta.get("build_number", 0))
        cands.append((ver, bn, fn, meta))
    if not cands:
        raise RuntimeError(f"No conda-forge openblas package found for flavor={flavor!r}.")
    cands.sort()
    _, _, fn, meta = cands[-1]
    return fn, meta


def _pick_matching_libopenblas(
    repodata: dict,
    *,
    version: str,
    flavor: str,
    dep_hint: str | None,
) -> tuple[str, dict]:
    dep_build_hint = None
    if dep_hint:
        # dependency format usually: "libopenblas 0.3.33 pthreads_h877e47f_0"
        parts = dep_hint.strip().split()
        if len(parts) >= 3 and parts[0] == "libopenblas":
            dep_build_hint = parts[2]
    cands: list[tuple[tuple[int, ...], int, str, dict]] = []
    for fn, meta in _iter_repodata_records(repodata):
        if meta.get("name") != "libopenblas":
            continue
        if str(meta.get("version", "")) != str(version):
            continue
        build = str(meta.get("build", ""))
        if flavor and (flavor not in build):
            continue
        if dep_build_hint and (build != dep_build_hint):
            continue
        ver = _parse_ver_tuple(str(meta.get("version", "0")))
        bn = int(meta.get("build_number", 0))
        cands.append((ver, bn, fn, meta))
    if not cands:
        hint = f", dep_hint={dep_hint!r}" if dep_hint else ""
        raise RuntimeError(
            "No matching conda-forge libopenblas package found "
            f"for version={version!r}, flavor={flavor!r}{hint}."
        )
    cands.sort()
    _, _, fn, meta = cands[-1]
    return fn, meta


def _download(url: str, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    out_path.write_bytes(data)


def _extract_conda_or_tar(pkg_path: pathlib.Path, dest_dir: pathlib.Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    lower = pkg_path.name.lower()
    if lower.endswith(".conda"):
        try:
            import zstandard as zstd
        except Exception as ex:  # pragma: no cover
            raise RuntimeError(
                "zstandard is required to extract .conda archives. "
                "Install it first: python -m pip install zstandard"
            ) from ex
        with zipfile.ZipFile(pkg_path, "r") as zf:
            members = [n for n in zf.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst")]
            if len(members) == 0:
                raise RuntimeError(f"No pkg-*.tar.zst found in {pkg_path}")
            member = members[0]
            zst_bytes = zf.read(member)
        dctx = zstd.ZstdDecompressor()
        tar_bytes = dctx.decompress(zst_bytes)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
            tf.extractall(dest_dir)
        return

    with tarfile.open(pkg_path, mode="r:*") as tf:
        tf.extractall(dest_dir)


def _find_openblas_dependency_hint(openblas_meta: dict) -> str | None:
    deps = openblas_meta.get("depends", [])
    if not isinstance(deps, list):
        return None
    for d in deps:
        s = str(d).strip()
        if s.startswith("libopenblas "):
            return s
    return None


def _verify_layout(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    lib_dir = root / "Library" / "lib"
    bin_dir = root / "Library" / "bin"
    include_dir = root / "Library" / "include"
    openblas_lib = lib_dir / "openblas.lib"
    openblas_dll = bin_dir / "openblas.dll"
    cblas_h = include_dir / "openblas" / "cblas.h"
    if not openblas_lib.is_file():
        raise RuntimeError(f"missing import library: {openblas_lib}")
    if not openblas_dll.is_file():
        raise RuntimeError(f"missing runtime DLL: {openblas_dll}")
    if not cblas_h.is_file():
        raise RuntimeError(f"missing header: {cblas_h}")

    blob = openblas_dll.read_bytes()
    required_tokens = [b"openblas_set_num_threads", b"openblas_get_num_threads", b"dsyevd_", b"dsyevr_"]
    missing = [tok.decode("ascii", "ignore") for tok in required_tokens if tok not in blob]
    if missing:
        raise RuntimeError(
            "openblas.dll missing required symbols/tokens: " + ", ".join(missing)
        )
    return lib_dir, bin_dir, include_dir


def _append_github_env(path: str, items: dict[str, str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for k, v in items.items():
            f.write(f"{k}={v}\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Install threaded OpenBLAS for Windows wheels from conda-forge archives."
    )
    ap.add_argument("--subdir", default="win-64", help="Conda subdir (default: win-64)")
    ap.add_argument(
        "--flavor",
        default="pthreads",
        choices=["pthreads", "openmp"],
        help="OpenBLAS flavor (default: pthreads)",
    )
    ap.add_argument(
        "--dest",
        required=True,
        help="Destination root to extract conda archives into (e.g. $RUNNER_TEMP/cf-openblas).",
    )
    ap.add_argument(
        "--github-env",
        default="",
        help="Path to GITHUB_ENV file; when set, exports OPENBLAS_* and JANUSX_FORCE_OPENBLAS_LAPACK.",
    )
    ap.add_argument("--verbose", action="store_true", default=False)
    args = ap.parse_args()

    subdir = str(args.subdir).strip()
    flavor = str(args.flavor).strip()
    dest_root = pathlib.Path(str(args.dest)).resolve()
    dl_dir = pathlib.Path(tempfile.mkdtemp(prefix="janusx_cf_openblas_dl_"))

    repodata_url = f"https://conda.anaconda.org/conda-forge/{subdir}/repodata.json"
    with urllib.request.urlopen(repodata_url, timeout=120) as r:
        repodata = json.load(r)

    openblas_fn, openblas_meta = _pick_openblas_package(repodata, flavor)
    dep_hint = _find_openblas_dependency_hint(openblas_meta)
    libopenblas_fn, _libopenblas_meta = _pick_matching_libopenblas(
        repodata,
        version=str(openblas_meta.get("version", "")),
        flavor=flavor,
        dep_hint=dep_hint,
    )

    for fn in (openblas_fn, libopenblas_fn):
        url = f"https://conda.anaconda.org/conda-forge/{subdir}/{fn}"
        out = dl_dir / fn
        _download(url, out)
        if args.verbose:
            print(f"[condaforge-openblas] downloaded: {url}")
        _extract_conda_or_tar(out, dest_root)
        if args.verbose:
            print(f"[condaforge-openblas] extracted: {out.name}")

    lib_dir, bin_dir, include_dir = _verify_layout(dest_root)

    env_items = {
        "OPENBLAS_LIB_DIR": str(lib_dir),
        "OPENBLAS_BIN_DIR": str(bin_dir),
        "OPENBLAS_INCLUDE_DIR": str(include_dir),
        # conda-forge libopenblas bundles LAPACK symbols (dsyevd_/dsyevr_),
        # but build.rs cannot detect this from lib names on Windows.
        "JANUSX_FORCE_OPENBLAS_LAPACK": "1",
    }
    if args.github_env:
        _append_github_env(str(args.github_env), env_items)

    print("[condaforge-openblas] selected:")
    print(f"  openblas     : {openblas_fn}")
    print(f"  libopenblas  : {libopenblas_fn}")
    print("[condaforge-openblas] exports:")
    for k, v in env_items.items():
        print(f"  {k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
