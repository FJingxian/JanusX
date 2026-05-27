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
import fnmatch
import subprocess
import sysconfig
from pathlib import Path
from typing import Any

# We intentionally use a custom PEP517 backend (this file) to wrap maturin
# and inject optional native artifacts into wheels. Silence maturin's
# informational warning about non-`maturin` build-backend to reduce noise.
os.environ.setdefault("MATURIN_NO_MISSING_BUILD_BACKEND_WARNING", "1")

_MATURIN = importlib.import_module("maturin")


def _prepend_path_entry(path_entry: str) -> None:
    entry = str(path_entry).strip()
    if not entry:
        return
    current = str(os.environ.get("PATH", ""))
    parts = current.split(os.pathsep) if current else []
    norm_entry = os.path.normcase(os.path.normpath(entry))
    for p in parts:
        if os.path.normcase(os.path.normpath(p)) == norm_entry:
            return
    os.environ["PATH"] = entry + (os.pathsep + current if current else "")


def _ensure_maturin_on_path() -> None:
    # Build frontends may invoke this backend with a sanitized PATH that does
    # not include the active interpreter's scripts directory. Maturin's own
    # PEP517 shim shells out to `maturin ...`, so ensure both candidate
    # locations are visible.
    try:
        _prepend_path_entry(str(Path(sys.executable).resolve().parent))
    except Exception:
        pass
    try:
        scripts_dir = sysconfig.get_path("scripts")
    except Exception:
        scripts_dir = None
    if scripts_dir:
        _prepend_path_entry(str(scripts_dir))


def _build_wheel_via_python_module(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    options: list[str] = []
    add_args_fn = getattr(_MATURIN, "_additional_pep517_args", None)
    if callable(add_args_fn):
        try:
            options.extend(list(add_args_fn()))
        except Exception:
            pass
    get_args_fn = getattr(_MATURIN, "get_maturin_pep517_args", None)
    if callable(get_args_fn):
        try:
            options.extend(list(get_args_fn(config_settings)))
        except Exception:
            pass

    if "--compatibility" not in options and "--manylinux" not in options:
        options = ["--compatibility", "off", *options]

    command = [
        sys.executable,
        "-m",
        "maturin",
        "pep517",
        "build-wheel",
        "-i",
        sys.executable,
        *options,
    ]
    print(
        "[build-backend] fallback: running `" + " ".join(command) + "`",
        flush=True,
    )
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    output_bytes = result.stdout or b""
    if output_bytes:
        sys.stdout.buffer.write(output_bytes)
        sys.stdout.flush()
    if result.returncode != 0:
        raise RuntimeError(
            f"`python -m maturin` fallback failed with exit status {result.returncode}"
        )
    output = output_bytes.decode(errors="replace")
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    if len(lines) == 0:
        raise RuntimeError("maturin fallback produced empty output; cannot locate wheel path.")
    wheel_path = lines[-1]
    filename = os.path.basename(wheel_path)
    shutil.copy2(wheel_path, os.path.join(wheel_directory, filename))
    return filename


def _env_flag(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}

def _env_flag_optional(name: str) -> bool | None:
    raw = str(os.environ.get(name, "")).strip().lower()
    if raw == "":
        return None
    return raw in {"1", "true", "yes", "on"}

def _strict_kmc_bind_mode() -> bool:
    explicit = _env_flag_optional("JANUSX_STRICT_KMC_BIND")
    if explicit is not None:
        return bool(explicit)
    # Default strict on Linux/macOS wheel builds to avoid publishing wheels
    # that require runtime C++ compilation for `jx kmer`.
    return sys.platform.startswith("linux") or (sys.platform == "darwin")


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parent


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
        if _strict_kmc_bind_mode():
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


def _iter_unique_existing_dirs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp).lower()
        if key in seen:
            continue
        seen.add(key)
        if rp.exists() and rp.is_dir():
            out.append(rp)
    return out


def _windows_runtime_dirs() -> list[Path]:
    dirs: list[Path] = []
    for var in ("OPENBLAS_BIN_DIR", "OPENBLAS_LIB_DIR", "LIBRARY_BIN", "LIBRARY_LIB"):
        raw = str(os.environ.get(var, "")).strip()
        if raw:
            dirs.append(Path(raw))

    conda_prefix = str(os.environ.get("CONDA_PREFIX", "")).strip()
    if conda_prefix:
        cp = Path(conda_prefix)
        dirs.extend(
            [
                cp / "Library" / "bin",
                cp / "Library" / "lib",
                cp / "DLLs",
                cp / "bin",
            ]
        )
    return _iter_unique_existing_dirs(dirs)


def _collect_windows_runtime_dlls() -> list[Path]:
    if not sys.platform.startswith("win"):
        return []

    # Keep this list tight to avoid wheel bloat while still covering common
    # OpenBLAS + MinGW runtime dependencies in conda environments.
    patterns = [
        "libopenblas*.dll",
        "openblas*.dll",
        "liblapack*.dll",
        "lapack*.dll",
        "libblas*.dll",
        "blas*.dll",
        "libgfortran*.dll",
        "libquadmath*.dll",
        "libgcc_s*.dll",
        "libwinpthread*.dll",
        "libstdc++*.dll",
        "libgomp*.dll",
        "libomp*.dll",
        "libiomp*.dll",
        "vcomp*.dll",
    ]
    picked: dict[str, Path] = {}
    for d in _windows_runtime_dirs():
        try:
            names = sorted(os.listdir(d))
        except Exception:
            continue
        for name in names:
            lname = name.lower()
            if not lname.endswith(".dll"):
                continue
            if not any(fnmatch.fnmatch(lname, pat) for pat in patterns):
                continue
            full = d / name
            if not full.is_file():
                continue
            key = name.lower()
            # Prefer the first match by search priority.
            picked.setdefault(key, full)
    return sorted(picked.values(), key=lambda p: p.name.lower())


def _macos_runtime_dirs() -> list[Path]:
    dirs: list[Path] = []

    for var in ("OPENBLAS_LIB_DIR", "OPENBLAS_INCLUDE_DIR"):
        raw = str(os.environ.get(var, "")).strip()
        if not raw:
            continue
        p = Path(raw)
        # OPENBLAS_INCLUDE_DIR may point to include/, so also try sibling lib/.
        dirs.append(p)
        if p.name.lower() == "include":
            dirs.append(p.parent / "lib")
        else:
            dirs.append(p.parent)

    conda_prefix = str(os.environ.get("CONDA_PREFIX", "")).strip()
    if conda_prefix:
        cp = Path(conda_prefix)
        dirs.extend(
            [
                cp / "lib",
                cp / "Library" / "lib",
            ]
        )

    # Common Homebrew prefixes.
    dirs.extend(
        [
            Path("/opt/homebrew/opt/openblas/lib"),
            Path("/usr/local/opt/openblas/lib"),
            Path("/opt/homebrew/opt/gcc/lib/gcc/current"),
            Path("/usr/local/opt/gcc/lib/gcc/current"),
            Path("/opt/homebrew/opt/libomp/lib"),
            Path("/usr/local/opt/libomp/lib"),
        ]
    )
    return _iter_unique_existing_dirs(dirs)


def _otool_libraries(path: Path) -> list[str]:
    try:
        out = subprocess.check_output(
            ["otool", "-L", str(path)],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return []

    lines = [ln.rstrip() for ln in out.splitlines()]
    if len(lines) <= 1:
        return []

    deps: list[str] = []
    for ln in lines[1:]:
        s = ln.strip()
        if not s:
            continue
        dep = s.split(" (", 1)[0].strip()
        if dep:
            deps.append(dep)
    return deps


def _is_macos_system_lib(dep: str) -> bool:
    return dep.startswith("/usr/lib/") or dep.startswith("/System/Library/")


def _resolve_macos_otool_dep(
    dep: str,
    *,
    origin_dir: Path,
    extra_dirs: list[Path],
) -> Path | None:
    s = str(dep).strip()
    if s == "":
        return None
    if _is_macos_system_lib(s):
        return None

    # Absolute dependency recorded by install name.
    if s.startswith("/"):
        p = Path(s)
        return p if p.is_file() else None

    dep_name = Path(s).name
    search_dirs: list[Path] = [origin_dir, *extra_dirs]
    seen: set[str] = set()
    for d in search_dirs:
        try:
            rd = d.resolve()
        except Exception:
            rd = d
        key = str(rd)
        if key in seen:
            continue
        seen.add(key)
        cand = rd / dep_name
        if cand.is_file():
            return cand.resolve()
    return None


def _collect_macos_openblas_dylibs() -> list[Path]:
    if sys.platform != "darwin":
        return []

    if str(os.environ.get("JANUSX_BUNDLE_OPENBLAS_DYLIB", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return []

    strict = str(os.environ.get("JANUSX_STRICT_OPENBLAS_BUNDLE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    prefer = [
        "libopenblas.dylib",
        "libopenblas.0.dylib",
    ]
    primary: Path | None = None
    source_dir: Path | None = None
    for d in _macos_runtime_dirs():
        picked: list[Path] = []
        for name in prefer:
            p = d / name
            if p.is_file():
                picked.append(p)
        if len(picked) == 0:
            picked.extend(sorted(p for p in d.glob("libopenblas*.dylib") if p.is_file()))
        if len(picked) > 0:
            primary = picked[0]
            source_dir = d
            break

    if primary is None or source_dir is None:
        if strict:
            raise RuntimeError(
                "JANUSX_STRICT_OPENBLAS_BUNDLE=1 but no macOS OpenBLAS dylib was found. "
                "Set OPENBLAS_LIB_DIR to a directory containing libopenblas*.dylib."
            )
        return []

    # Keep all OpenBLAS aliases from the selected source directory so loader
    # can resolve both libopenblas.dylib and libopenblas.0.dylib references.
    seed: list[Path] = []
    seen_seed: set[str] = set()
    for p in sorted(source_dir.glob("libopenblas*.dylib")):
        if not p.is_file():
            continue
        key = p.name.lower()
        if key in seen_seed:
            continue
        seen_seed.add(key)
        seed.append(p.resolve())
    if len(seed) == 0:
        seed.append(primary.resolve())

    # Recursive dependency closure for non-system absolute dylibs.
    runtime_dirs = _macos_runtime_dirs()
    search_dirs: list[Path] = [source_dir, *runtime_dirs]
    picked: dict[str, Path] = {}
    queue: list[Path] = list(seed)
    while len(queue) > 0:
        cur = queue.pop(0)
        key = cur.name.lower()
        if key in picked:
            continue
        picked[key] = cur
        for dep in _otool_libraries(cur):
            dep_path = _resolve_macos_otool_dep(
                dep,
                origin_dir=cur.parent,
                extra_dirs=search_dirs,
            )
            if dep_path is None:
                if strict and (not _is_macos_system_lib(dep)):
                    raise RuntimeError(
                        "JANUSX_STRICT_OPENBLAS_BUNDLE=1 but failed to resolve "
                        f"macOS dylib dependency {dep!r} for {cur.name}."
                    )
                continue
            if dep_path.name.lower() not in picked:
                queue.append(dep_path)
    return sorted(picked.values(), key=lambda p: p.name.lower())


def _collect_fasttree_artifacts() -> list[Path]:
    raw = str(os.environ.get("JANUSX_FASTTREE_BIN", "")).strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            root = p.resolve().parent
            return sorted([x.resolve() for x in root.iterdir() if x.is_file()], key=lambda x: x.name.lower())
        raise RuntimeError(f"JANUSX_FASTTREE_BIN does not exist: {raw}")

    root = _project_root() / "target" / "janusx-artifacts"
    if not root.is_dir():
        return []

    exe_names = ["FastTreeMP.exe", "FastTree.exe"] if sys.platform.startswith("win") else ["FastTree"]
    candidates: list[tuple[float, int, Path]] = []
    for priority, exe_name in enumerate(exe_names):
        for p in root.rglob(exe_name):
            if not (p.is_file() and any(part.lower() == "fasttree" for part in p.parts)):
                continue
            try:
                mtime = p.stat().st_mtime
            except Exception:
                mtime = 0.0
            candidates.append((mtime, len(exe_names) - priority, p))
    if len(candidates) == 0:
        return []
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    picked = candidates[0][2].resolve()
    return sorted([x.resolve() for x in picked.parent.iterdir() if x.is_file()], key=lambda x: x.name.lower())


def _rewrite_macos_dylib_install_names(extract_root: Path, rel_paths: list[Path]) -> None:
    if sys.platform != "darwin" or len(rel_paths) == 0:
        return

    dylibs: list[Path] = []
    by_name: dict[str, Path] = {}
    for rel in rel_paths:
        p = extract_root / rel
        if not p.is_file():
            continue
        if p.suffix.lower() != ".dylib":
            continue
        dylibs.append(p)
        by_name[p.name] = p

    if len(dylibs) == 0:
        return

    def _is_omp_family(name: str) -> bool:
        low = name.lower()
        return low.startswith("libomp") or low.startswith("libgomp")

    omp_canonical_name: str | None = None
    if "libomp.dylib" in by_name:
        omp_canonical_name = "libomp.dylib"
    else:
        omp_candidates = sorted(n for n in by_name.keys() if _is_omp_family(n))
        if len(omp_candidates) > 0:
            omp_canonical_name = omp_candidates[0]

    def _run_install_name_tool(args: list[str]) -> None:
        try:
            subprocess.run(
                ["install_name_tool", *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            out = str(exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(
                f"install_name_tool failed: {' '.join(args)}; {out}"
            ) from exc

    def _run_codesign(path: Path) -> None:
        # install_name_tool mutates Mach-O load commands and invalidates the
        # existing signature; re-sign ad-hoc so dlopen() will not be killed.
        cmd = ["/usr/bin/codesign", "--force", "--sign", "-", str(path)]
        if not Path(cmd[0]).is_file():
            cmd[0] = "codesign"
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            out = str(exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(
                f"codesign failed for {path.name}: {out}"
            ) from exc

    # Normalize dylib id so dependencies can resolve from wheel-local directory.
    for lib in dylibs:
        id_name = lib.name
        # Keep one logical OpenMP runtime id across libomp/libgomp aliases.
        # This prevents accidental double-loading of the same runtime family.
        if omp_canonical_name is not None and _is_omp_family(lib.name):
            id_name = omp_canonical_name
        _run_install_name_tool(["-id", f"@loader_path/{id_name}", str(lib)])

    # Rewrite intra-bundle references to @loader_path/<name>.
    for lib in dylibs:
        for dep in _otool_libraries(lib):
            dep_name = Path(dep).name
            if omp_canonical_name is not None and _is_omp_family(dep_name):
                replacement = f"@loader_path/{omp_canonical_name}"
                if dep != replacement:
                    _run_install_name_tool(["-change", dep, replacement, str(lib)])
                continue
            if dep_name not in by_name:
                continue
            replacement = f"@loader_path/{dep_name}"
            if dep == replacement:
                continue
            _run_install_name_tool(["-change", dep, replacement, str(lib)])

    for lib in dylibs:
        _run_codesign(lib)


def _inject_artifacts_into_wheel(
    wheel_path: Path,
    artifacts: list[tuple[Path, Path]],
) -> None:
    if len(artifacts) == 0:
        return

    with tempfile.TemporaryDirectory(prefix="janusx_wheel_edit_") as td:
        root = Path(td) / "wheel"
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheel_path, "r") as zf:
            zf.extractall(root)

        for src, rel in artifacts:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)

        # For macOS wheel-local dylibs, fix install names to avoid dependence
        # on host-specific Homebrew/Conda absolute paths.
        mac_rel = [
            rel for _src, rel in artifacts
            if rel.as_posix().startswith("janusx.libs/")
        ]
        _rewrite_macos_dylib_install_names(root, mac_rel)

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
    win_dlls = _collect_windows_runtime_dlls()
    mac_dylibs = _collect_macos_openblas_dylibs()
    fasttree_files = _collect_fasttree_artifacts()
    if sys.platform.startswith("win") and _env_flag("JANUSX_REQUIRE_OPENBLAS"):
        has_openblas_dll = any("openblas" in p.name.lower() for p in win_dlls)
        if not has_openblas_dll:
            raise RuntimeError(
                "JANUSX_REQUIRE_OPENBLAS=1 but no OpenBLAS runtime DLL was found for wheel bundling. "
                "Set OPENBLAS_BIN_DIR/OPENBLAS_LIB_DIR or use a conda env with OpenBLAS in Library/bin."
            )
    if len(fasttree_files) == 0:
        raise RuntimeError(
            "FastTree executable was not produced under target/janusx-artifacts; "
            "build.rs did not finish the FastTree bundle step."
        )

    prev = os.environ.get("JANUSX_PREBUILD_KMC_BIND")
    os.environ["JANUSX_PREBUILD_KMC_BIND"] = "0"
    try:
        _ensure_maturin_on_path()
        try:
            wheel_name = _MATURIN.build_wheel(
                wheel_directory,
                config_settings,
                metadata_directory,
            )
        except FileNotFoundError as exc:
            missing = str(getattr(exc, "filename", "")).strip().lower()
            if missing != "maturin":
                raise
            print(
                "[build-backend] warning: `maturin` executable not found on PATH; "
                "falling back to `python -m maturin`.",
                flush=True,
            )
            wheel_name = _build_wheel_via_python_module(
                wheel_directory,
                config_settings,
            )
    finally:
        if prev is None:
            os.environ.pop("JANUSX_PREBUILD_KMC_BIND", None)
        else:
            os.environ["JANUSX_PREBUILD_KMC_BIND"] = prev

    artifacts: list[tuple[Path, Path]] = []
    if ext_path is not None:
        artifacts.append((ext_path, Path("janusx") / ext_path.name))
    for fasttree_file in fasttree_files:
        artifacts.append((fasttree_file, Path("janusx") / "bin" / fasttree_file.name))
    for dll in win_dlls:
        artifacts.append((dll, Path("janusx") / dll.name))
    for dylib in mac_dylibs:
        artifacts.append((dylib, Path("janusx.libs") / dylib.name))

    if len(artifacts) > 0:
        wheel_path = Path(wheel_directory).resolve() / wheel_name
        _inject_artifacts_into_wheel(wheel_path, artifacts)
        if ext_path is not None:
            print(f"[build-backend] injected extension: janusx/{ext_path.name}", flush=True)
        joined_fasttree = ", ".join(f"janusx/bin/{p.name}" for p in fasttree_files)
        print(f"[build-backend] injected FastTree artifacts: {joined_fasttree}", flush=True)
        if win_dlls:
            joined = ", ".join(sorted(d.name for d in win_dlls))
            print(f"[build-backend] injected runtime DLLs: {joined}", flush=True)
        if mac_dylibs:
            joined = ", ".join(sorted(d.name for d in mac_dylibs))
            print(f"[build-backend] injected macOS dylibs: {joined}", flush=True)
    return wheel_name
