from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional


def _default_kmc_source_candidates() -> tuple[str, ...]:
    here = Path(__file__).resolve().parent
    packaged = here / "native" / "vendor" / "kmc"
    repo_third_party = here.parent.parent / "third_party" / "kmc"
    return (
        os.environ.get("JANUSX_KMC_SRC", "").strip(),
        str(packaged),
        str(repo_third_party),
    )


_DEFAULT_KMC_SOURCE_CANDIDATES = _default_kmc_source_candidates()
_DEFAULT_KMC_CACHE_ENV = "JANUSX_KMC_BIND_CACHE"
_MODULE_BASENAME = "_kmc_count"
_PREBUILT_BLOB = f"{_MODULE_BASENAME}.prebuilt.bin"
_LOADED: dict[str, ModuleType] = {}


def _norm_path(path: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def resolve_kmc_source(kmc_src: str | None = None) -> Path:
    candidates: list[str] = []
    if kmc_src:
        candidates.append(str(kmc_src))
    candidates.extend([x for x in _DEFAULT_KMC_SOURCE_CANDIDATES if x])
    for raw in candidates:
        p = _norm_path(raw)
        if (p / "kmc_core" / "kmc_runner.h").is_file() and (p / "kmc_api" / "kmc_file.h").is_file():
            return p
    raise FileNotFoundError(
        "KMC source not found. Please pass --kmc-src or set JANUSX_KMC_SRC. "
        "Default search order: janusx/native/vendor/kmc, ../../third_party/kmc."
    )


def _parse_version_tuple(text: str | None) -> tuple[int, int, int]:
    if not text:
        return (0, 0, 0)
    m = re.match(r"\s*(\d+)\.(\d+)(?:\.(\d+))?", str(text).strip())
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def _required_pybind_version() -> tuple[int, int, int]:
    py = sys.version_info
    # Conservative constraints for Python C-API compatibility.
    if py >= (3, 13):
        return (2, 13, 0)
    if py >= (3, 12):
        return (2, 11, 0)
    return (2, 3, 0)


def _read_pybind_header_version(include_dir: Path) -> tuple[int, int, int]:
    common_h = include_dir / "pybind11" / "detail" / "common.h"
    if not common_h.is_file():
        return (0, 0, 0)
    txt = common_h.read_text(encoding="utf-8", errors="ignore")
    m_major = re.search(r"#define\s+PYBIND11_VERSION_MAJOR\s+(\d+)", txt)
    m_minor = re.search(r"#define\s+PYBIND11_VERSION_MINOR\s+(\d+)", txt)
    m_patch = re.search(r"#define\s+PYBIND11_VERSION_PATCH\s+(\d+)", txt)
    major = int(m_major.group(1)) if m_major else 0
    minor = int(m_minor.group(1)) if m_minor else 0
    patch = int(m_patch.group(1)) if m_patch else 0
    return (major, minor, patch)


def _warn_pybind_version_if_old(version: tuple[int, int, int], source: str) -> None:
    req = _required_pybind_version()
    if version < req:
        print(
            f"! Warning: pybind11 from {source} may be too old for Python "
            f"{sys.version_info.major}.{sys.version_info.minor} "
            f"(found {version[0]}.{version[1]}.{version[2]}, "
            f"recommended >= {req[0]}.{req[1]}.{req[2]}). "
            "Build will continue; if compile fails, install newer pybind11 "
            "via `python -m pip install -U pybind11`.",
            file=sys.stderr,
            flush=True,
        )


def _resolve_pybind_include(kmc_src_dir: Path) -> Path:
    import_err: Optional[Exception] = None
    try:
        import pybind11  # type: ignore

        _warn_pybind_version_if_old(
            _parse_version_tuple(getattr(pybind11, "__version__", "")),
            "installed pybind11 package",
        )
        inc = Path(pybind11.get_include())
        if inc.is_dir():
            return inc
    except ModuleNotFoundError:
        pass
    except Exception as e:
        # Keep original import failure details; do not mask it as a version issue.
        import_err = e

    vendored = kmc_src_dir / "py_kmc_api" / "libs" / "pybind11" / "include"
    if vendored.is_dir():
        _warn_pybind_version_if_old(
            _read_pybind_header_version(vendored),
            "vendored KMC pybind11 headers",
        )
        return vendored

    if import_err is not None:
        raise RuntimeError(
            "Failed to import installed pybind11 headers and no vendored fallback was found. "
            f"Original error: {repr(import_err)}"
        ) from import_err
    raise FileNotFoundError(
        "pybind11 headers not found. Install pybind11 or provide KMC source with py_kmc_api/libs/pybind11."
    )


def _resolve_python_include() -> Path:
    inc = sysconfig.get_paths().get("include", "")
    p = Path(inc)
    if p.is_dir():
        return p
    raise RuntimeError("Python include directory not found in sysconfig.")


def _resolve_ext_suffix() -> str:
    ext = str(sysconfig.get_config_var("EXT_SUFFIX") or "").strip()
    if ext:
        return ext
    if platform.system() == "Windows":
        return ".pyd"
    if platform.system() == "Darwin":
        return ".so"
    return ".so"


def _pick_cxx() -> str:
    pref = str(os.environ.get("CXX", "")).strip()
    candidates = [pref, "clang++", "g++", "c++"] if pref else ["clang++", "g++", "c++"]
    for cand in candidates:
        if not cand:
            continue
        exe = shutil.which(cand)
        if exe:
            return exe
    raise RuntimeError("No C++ compiler found. Please install clang++/g++ and ensure it is in PATH.")


def _build_cache_root() -> Path:
    custom = str(os.environ.get(_DEFAULT_KMC_CACHE_ENV, "")).strip()
    if custom:
        p = _norm_path(custom)
    else:
        p = _norm_path("~/.cache/janusx/kmc_bind")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_gnuext_compat_header(compat_root: Path) -> Path:
    """
    KMC includes <ext/algorithm> and expects __gnu_cxx::copy_n.
    Apple clang/libc++ lacks this header; provide a tiny compatibility shim.
    """
    ext_dir = compat_root / "ext"
    ext_dir.mkdir(parents=True, exist_ok=True)
    hdr = ext_dir / "algorithm"
    if not hdr.is_file():
        hdr.write_text(
            "#pragma once\n"
            "#include <algorithm>\n"
            "namespace __gnu_cxx { using std::copy_n; }\n",
            encoding="utf-8",
        )
    return compat_root


def _ensure_cloudflare_zlib_header(kmc_src_dir: Path) -> None:
    """
    Some KMC source snapshots are missing initialized 3rd_party/cloudflare submodule.
    KMC headers include ../3rd_party/cloudflare/zlib.h, so provide a shim to system zlib.
    """
    hdr = kmc_src_dir / "3rd_party" / "cloudflare" / "zlib.h"
    if hdr.is_file():
        try:
            txt = hdr.read_text(encoding="utf-8", errors="ignore")
            # If this looks like a real bundled zlib header (not our shim), keep it.
            if "JANUSX_KMC_ZLIB_SHIM" not in txt:
                return
        except Exception:
            return
    hdr.parent.mkdir(parents=True, exist_ok=True)
    hdr.write_text(
        "/* JANUSX_KMC_ZLIB_SHIM */\n"
        "#pragma once\n"
        "#if defined(__has_include_next)\n"
        "#  if __has_include_next(<zlib.h>)\n"
        "#    include_next <zlib.h>\n"
        "#  else\n"
        "#    include <zlib.h>\n"
        "#  endif\n"
        "#else\n"
        "#  include <zlib.h>\n"
        "#endif\n",
        encoding="utf-8",
    )


def _kmc_sources(kmc_src_dir: Path) -> list[Path]:
    core = [
        "kmc_core/mem_disk_file.cpp",
        "kmc_core/rev_byte.cpp",
        "kmc_core/bkb_writer.cpp",
        "kmc_core/cpu_info.cpp",
        "kmc_core/bkb_reader.cpp",
        "kmc_core/fastq_reader.cpp",
        "kmc_core/timer.cpp",
        "kmc_core/develop.cpp",
        "kmc_core/kb_completer.cpp",
        "kmc_core/kb_storer.cpp",
        "kmc_core/kmer.cpp",
        "kmc_core/splitter.cpp",
        "kmc_core/kb_collector.cpp",
        "kmc_core/kmc_runner.cpp",
        "kmc_core/kff_writer.cpp",
    ]
    api = [
        "kmc_api/mmer.cpp",
        "kmc_api/kmc_file.cpp",
        "kmc_api/kmer_api.cpp",
    ]
    arch = platform.machine().lower()
    if arch in {"arm64", "aarch64"}:
        raduls = ["kmc_core/raduls_neon.cpp"]
    else:
        raduls = [
            "kmc_core/raduls_sse2.cpp",
            "kmc_core/raduls_sse41.cpp",
            "kmc_core/raduls_avx.cpp",
            "kmc_core/raduls_avx2.cpp",
        ]
    return [kmc_src_dir / x for x in (core + api + raduls)]


def _run(cmd: list[str], *, cwd: Path | None = None, verbose: bool = False) -> None:
    if verbose:
        print("[kmc-bind]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        msg = (
            f"Command failed (exit={proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        raise RuntimeError(msg)


def _source_fingerprint(kmc_src_dir: Path, wrapper_cpp: Path, kmc_cpp: list[Path]) -> str:
    h = hashlib.sha1()
    files: list[Path] = list(kmc_cpp) + [wrapper_cpp, kmc_src_dir / "3rd_party" / "cloudflare" / "zlib.h"]
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in files:
        rp = p.resolve()
        if rp in seen or (not p.is_file()):
            continue
        seen.add(rp)
        uniq.append(p)
    uniq.sort(key=lambda x: str(x))
    for p in uniq:
        try:
            rel = p.relative_to(kmc_src_dir)
            tag = f"kmc:{rel.as_posix()}"
        except Exception:
            tag = f"local:{p.name}"
        h.update(tag.encode("utf-8"))
        h.update(b"\0")
        with p.open("rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()[:16]


def _compile_objects(
    *,
    cxx: str,
    sources: Iterable[Path],
    obj_dir: Path,
    include_dirs: list[Path],
    verbose: bool,
) -> list[Path]:
    objs: list[Path] = []
    common = ["-O3", "-DNDEBUG", "-std=c++14", "-fPIC", "-Wall"]
    inc_flags = [f"-I{d}" for d in include_dirs]
    for src in sources:
        if not src.is_file():
            raise FileNotFoundError(f"KMC source file not found: {src}")
        obj = obj_dir / f"{src.stem}.o"
        extra: list[str] = []
        name = src.name
        if name == "raduls_sse2.cpp":
            extra = ["-msse2"]
        elif name == "raduls_sse41.cpp":
            extra = ["-msse4.1"]
        elif name == "raduls_avx.cpp":
            extra = ["-mavx"]
        elif name == "raduls_avx2.cpp":
            extra = ["-mavx2"]
        cmd = [cxx, *common, *extra, *inc_flags, "-c", str(src), "-o", str(obj)]
        _run(cmd, verbose=verbose)
        objs.append(obj)
    return objs


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def build_kmc_bind_module(
    *,
    kmc_src: str | None = None,
    rebuild: bool = False,
    verbose: bool = False,
) -> Path:
    if platform.system() == "Windows":
        raise RuntimeError("KMC pybind builder currently supports Linux/macOS only.")

    kmc_src_dir = resolve_kmc_source(kmc_src)
    wrapper_cpp = _norm_path(str(Path(__file__).resolve().parent / "native" / "kmc_count_bind.cpp"))
    if not wrapper_cpp.is_file():
        raise FileNotFoundError(f"Wrapper source not found: {wrapper_cpp}")

    ext_suffix = _resolve_ext_suffix()
    kmc_cpp = _kmc_sources(kmc_src_dir)
    src_fp = _source_fingerprint(kmc_src_dir, wrapper_cpp, kmc_cpp)
    sig = hashlib.sha1(
        f"{src_fp}|{platform.system()}|{platform.machine()}|{sys.version_info[:2]}".encode("utf-8")
    ).hexdigest()[:12]
    build_root = _build_cache_root() / f"build_{sig}"
    obj_dir = build_root / "obj"
    obj_dir.mkdir(parents=True, exist_ok=True)
    compat_dir = _ensure_gnuext_compat_header(build_root / "compat_include")
    out = build_root / f"{_MODULE_BASENAME}{ext_suffix}"
    if out.is_file() and not rebuild:
        return out

    pybind_include = _resolve_pybind_include(kmc_src_dir)
    py_include = _resolve_python_include()
    cxx = _pick_cxx()
    _ensure_cloudflare_zlib_header(kmc_src_dir)

    include_dirs = [
        compat_dir,
        pybind_include,
        py_include,
        kmc_src_dir,
        kmc_src_dir / "kmc_core",
        kmc_src_dir / "kmc_api",
    ]
    kmc_objs = _compile_objects(
        cxx=cxx,
        sources=kmc_cpp,
        obj_dir=obj_dir,
        include_dirs=include_dirs,
        verbose=verbose,
    )
    wrapper_obj = _compile_objects(
        cxx=cxx,
        sources=[wrapper_cpp],
        obj_dir=obj_dir,
        include_dirs=include_dirs,
        verbose=verbose,
    )

    link_cmd = [cxx, "-shared", "-o", str(out), *(str(x) for x in (kmc_objs + wrapper_obj)), "-lpthread", "-lz", "-lm"]
    if platform.system() == "Darwin":
        link_cmd.extend(["-undefined", "dynamic_lookup"])
    _run(link_cmd, verbose=verbose)
    return out


def build_packaged_kmc_bind(
    *,
    kmc_src: str | None = None,
    rebuild: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Build _kmc_count and place it under janusx package directory so wheel can include it.
    """
    built = build_kmc_bind_module(kmc_src=kmc_src, rebuild=rebuild, verbose=verbose)
    ext_suffix = _resolve_ext_suffix()
    out = _package_dir() / f"{_MODULE_BASENAME}{ext_suffix}"
    blob = _package_dir() / _PREBUILT_BLOB
    if rebuild or (not out.is_file()) or (out.stat().st_mtime < built.stat().st_mtime):
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, out)
    if rebuild or (not blob.is_file()) or (blob.stat().st_mtime < built.stat().st_mtime):
        shutil.copy2(built, blob)
    return out


def _try_load_packaged_module() -> ModuleType | None:
    try:
        return importlib.import_module(f"janusx.{_MODULE_BASENAME}")
    except Exception:
        pass

    ext_suffix = _resolve_ext_suffix()
    so_path = _package_dir() / f"{_MODULE_BASENAME}{ext_suffix}"
    if not so_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"janusx.{_MODULE_BASENAME}", so_path)
    if spec is None or spec.loader is None:
        so_path = None
    if so_path is not None:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # Fallback: load from prebuilt blob shipped in wheel/sdist.
    blob = _package_dir() / _PREBUILT_BLOB
    if not blob.is_file():
        return None
    staged = _build_cache_root() / "prebuilt"
    staged.mkdir(parents=True, exist_ok=True)
    staged_so = staged / f"{_MODULE_BASENAME}{ext_suffix}"
    if (not staged_so.is_file()) or (staged_so.stat().st_size != blob.stat().st_size):
        shutil.copy2(blob, staged_so)
    spec2 = importlib.util.spec_from_file_location(f"janusx.{_MODULE_BASENAME}", staged_so)
    if spec2 is None or spec2.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod)
    return mod


def load_kmc_bind_module(
    *,
    kmc_src: str | None = None,
    rebuild: bool = False,
    verbose: bool = False,
) -> ModuleType:
    kmc_src_dir = str(resolve_kmc_source(kmc_src))
    cache_key = f"{kmc_src_dir}|{int(rebuild)}"
    if cache_key in _LOADED:
        return _LOADED[cache_key]
    if not rebuild:
        packaged = _try_load_packaged_module()
        if packaged is not None:
            _LOADED[cache_key] = packaged
            return packaged
    so_path = build_kmc_bind_module(kmc_src=kmc_src, rebuild=rebuild, verbose=verbose)
    spec = importlib.util.spec_from_file_location(_MODULE_BASENAME, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load extension module from: {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LOADED[cache_key] = mod
    return mod


def run_kmc_count(
    *,
    input_files: list[str],
    output_prefix: str,
    tmp_dir: str,
    kmer_len: int,
    threads: int,
    max_ram_gb: int,
    cutoff_min: int,
    cutoff_max: int,
    counter_max: int,
    canonical: bool,
    input_type: str,
    kmc_src: str | None = None,
    rebuild_bind: bool = False,
    verbose_build: bool = False,
) -> dict:
    mod = load_kmc_bind_module(kmc_src=kmc_src, rebuild=rebuild_bind, verbose=verbose_build)
    return dict(
        mod.kmc_count(
            input_files=input_files,
            output_prefix=output_prefix,
            tmp_dir=tmp_dir,
            kmer_len=int(kmer_len),
            threads=int(threads),
            max_ram_gb=int(max_ram_gb),
            cutoff_min=int(cutoff_min),
            cutoff_max=int(cutoff_max),
            counter_max=int(counter_max),
            canonical=bool(canonical),
            input_type=str(input_type),
        )
    )


def run_kmc_db_info(
    *,
    kmc_prefix: str,
    kmc_src: str | None = None,
    rebuild_bind: bool = False,
    verbose_build: bool = False,
) -> dict:
    mod = load_kmc_bind_module(kmc_src=kmc_src, rebuild=rebuild_bind, verbose=verbose_build)
    return dict(mod.kmc_db_info(kmc_prefix=str(kmc_prefix)))


def run_kmc_export_janusx_single(
    *,
    kmc_prefix: str,
    out_prefix: str,
    sample_id: str,
    kmc_src: str | None = None,
    rebuild_bind: bool = False,
    verbose_build: bool = False,
) -> dict:
    mod = load_kmc_bind_module(kmc_src=kmc_src, rebuild=rebuild_bind, verbose=verbose_build)
    return dict(
        mod.kmc_export_janusx_single(
            kmc_prefix=str(kmc_prefix),
            out_prefix=str(out_prefix),
            sample_id=str(sample_id),
        )
    )


def run_kmc_export_bin_single(
    *,
    kmc_prefix: str,
    out_prefix: str,
    sample_id: str,
    progress_callback=None,
    progress_every: int = 200000,
    threads: int = 0,
    kmc_src: str | None = None,
    rebuild_bind: bool = False,
    verbose_build: bool = False,
) -> dict:
    mod = load_kmc_bind_module(kmc_src=kmc_src, rebuild=rebuild_bind, verbose=verbose_build)
    if not hasattr(mod, "kmc_export_bin_single"):
        raise RuntimeError(
            "Current _kmc_count binding does not provide kmc_export_bin_single(). "
            "Please rebuild JanusX KMC extension."
        )
    return dict(
        mod.kmc_export_bin_single(
            kmc_prefix=str(kmc_prefix),
            out_prefix=str(out_prefix),
            sample_id=str(sample_id),
            progress_callback=progress_callback,
            progress_every=int(progress_every),
            threads=int(threads),
        )
    )


def run_kmc_export_bin_multi(
    *,
    kmc_prefixes: list[str],
    out_prefix: str,
    sample_ids: list[str],
    max_kmers: int = 0,
    kmerf: float = 0.2,
    progress_callback=None,
    progress_every: int = 200000,
    benchmark_callback=None,
    benchmark_progress_every: int = 5000,
    benchmark_fraction: float = 0.01,
    threads: int = 0,
    kmc_src: str | None = None,
    rebuild_bind: bool = False,
    verbose_build: bool = False,
) -> dict:
    mod = load_kmc_bind_module(kmc_src=kmc_src, rebuild=rebuild_bind, verbose=verbose_build)
    if not hasattr(mod, "kmc_export_bin_multi"):
        if not bool(rebuild_bind):
            mod = load_kmc_bind_module(kmc_src=kmc_src, rebuild=True, verbose=verbose_build)
    if not hasattr(mod, "kmc_export_bin_multi"):
        raise RuntimeError(
            "Current _kmc_count binding does not provide kmc_export_bin_multi(). "
            "Please rebuild JanusX KMC extension."
        )
    return dict(
        mod.kmc_export_bin_multi(
            kmc_prefixes=[str(x) for x in kmc_prefixes],
            out_prefix=str(out_prefix),
            sample_ids=[str(x) for x in sample_ids],
            max_kmers=int(max_kmers),
            kmerf=float(kmerf),
            progress_callback=progress_callback,
            progress_every=int(progress_every),
            benchmark_callback=benchmark_callback,
            benchmark_progress_every=int(benchmark_progress_every),
            benchmark_fraction=float(benchmark_fraction),
            threads=int(threads),
        )
    )


def run_kmc_dump_pairs(
    *,
    kmc_prefix: str,
    max_kmers: int = 0,
    kmc_src: str | None = None,
    rebuild_bind: bool = False,
    verbose_build: bool = False,
) -> dict:
    mod = load_kmc_bind_module(kmc_src=kmc_src, rebuild=rebuild_bind, verbose=verbose_build)
    return dict(
        mod.kmc_dump_pairs(
            kmc_prefix=str(kmc_prefix),
            max_kmers=int(max_kmers),
        )
    )
