from __future__ import annotations

import fnmatch
import os
import site
import struct
import sys
import tempfile
import time
from pathlib import Path


_DLL_DIR_HANDLES: list[object] = []
_NATIVE_EXT_PATTERNS = ("janusx*.so", "janusx*.pyd")
_RUNTIME_CACHE_ROOT_ENV = "JANUSX_RUNTIME_CACHE_DIR"
_ALLOW_EXTERNAL_RUNTIME_ENV = "JANUSX_ALLOW_EXTERNAL_NUMERICAL_LIBRARIES"
_LINUX_NUMERICAL_RUNTIME_PATTERNS = (
    "libopenblas*.so*",
    "libblas*.so*",
    "liblapack*.so*",
    "libomp*.so*",
    "libgomp*.so*",
    "libiomp*.so*",
    "libmkl*.so*",
)
_WINDOWS_NUMERICAL_RUNTIME_PATTERNS = (
    "libopenblas*.dll",
    "openblas*.dll",
    "libblas*.dll",
    "blas*.dll",
    "liblapack*.dll",
    "lapack*.dll",
    "libomp*.dll",
    "libgomp*.dll",
    "libiomp*.dll",
    "mkl*.dll",
    "vcomp*.dll",
)


def _matching_conda_prefix() -> Path | None:
    raw = str(os.environ.get("CONDA_PREFIX", "")).strip()
    if not raw:
        return None
    try:
        conda_prefix = Path(raw).resolve()
        runtime_prefixes = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
        return conda_prefix if conda_prefix in runtime_prefixes else None
    except OSError:
        return None


def _runtime_cache_root() -> Path:
    explicit = str(os.environ.get(_RUNTIME_CACHE_ROOT_ENV, "")).strip()
    if explicit != "":
        return Path(explicit)
    try:
        uid = str(int(os.getuid()))
    except Exception:
        uid = "nouid"
    return Path(tempfile.gettempdir()) / f"janusx-runtime-{uid}"


def _path_writable_or_creatable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    probe = path / ".janusx-write-probe"
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("1")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _default_xdg_cache_home() -> Path:
    try:
        home = Path.home()
    except Exception:
        return _runtime_cache_root() / "xdg-cache"
    return home / ".cache"


def _default_mplconfigdir() -> Path:
    try:
        home = Path.home()
    except Exception:
        return _runtime_cache_root() / "mplconfig"
    return home / ".matplotlib"


def _ensure_runtime_cache_dirs() -> None:
    """
    Keep plotting/cache-heavy imports away from unwritable home directories.

    This avoids slow matplotlib/fontconfig fallback behavior in shared or
    locked-down environments (for example conda base, HPC login nodes, or
    read-only home mounts) before GWAS/GS modules import matplotlib.
    """
    cache_root = _runtime_cache_root()

    raw_xdg = str(os.environ.get("XDG_CACHE_HOME", "")).strip()
    xdg_path = Path(raw_xdg) if raw_xdg != "" else _default_xdg_cache_home()
    if not _path_writable_or_creatable(xdg_path):
        xdg_path = cache_root / "xdg-cache"
        if _path_writable_or_creatable(xdg_path):
            os.environ["XDG_CACHE_HOME"] = str(xdg_path)
    try:
        (xdg_path / "fontconfig").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    raw_mpl = str(os.environ.get("MPLCONFIGDIR", "")).strip()
    mpl_path = Path(raw_mpl) if raw_mpl != "" else _default_mplconfigdir()
    if not _path_writable_or_creatable(mpl_path):
        mpl_path = xdg_path / "matplotlib"
        if _path_writable_or_creatable(mpl_path):
            os.environ["MPLCONFIGDIR"] = str(mpl_path)


_ensure_runtime_cache_dirs()


def _package_dir_has_native_extension(pkg_dir: Path) -> bool:
    try:
        if not pkg_dir.is_dir():
            return False
    except Exception:
        return False
    for pattern in _NATIVE_EXT_PATTERNS:
        try:
            for hit in pkg_dir.glob(pattern):
                try:
                    if hit.is_file():
                        return True
                except Exception:
                    continue
        except Exception:
            continue
    return False


def _iter_janusx_package_dirs() -> list[Path]:
    pkg_dir = Path(__file__).resolve().parent
    pkg_name = pkg_dir.name
    roots: list[str] = []
    roots.extend(str(x) for x in sys.path)
    try:
        roots.extend(str(x) for x in site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            roots.append(user_site)
        else:
            roots.extend(str(x) for x in user_site)
    except Exception:
        pass

    out: list[Path] = []
    seen: set[str] = set()
    for raw in roots:
        text = str(raw).strip()
        if text == "":
            text = os.getcwd()
        root = Path(text)
        candidates = [root / pkg_name]
        try:
            if root.name == pkg_name and root.is_dir():
                candidates.insert(0, root)
        except Exception:
            pass
        for cand in candidates:
            try:
                resolved = cand.resolve()
            except Exception:
                resolved = cand
            key = str(resolved)
            if key in seen or resolved == pkg_dir:
                continue
            seen.add(key)
            try:
                if resolved.is_dir():
                    out.append(resolved)
            except Exception:
                continue
    return out


def _extend_package_path_for_native_extension() -> None:
    pkg_dir = Path(__file__).resolve().parent
    if _package_dir_has_native_extension(pkg_dir):
        return
    pkg_path = globals().get("__path__")
    if pkg_path is None:
        return
    for cand in _iter_janusx_package_dirs():
        if not _package_dir_has_native_extension(cand):
            continue
        cand_s = str(cand)
        try:
            present = {str(x) for x in pkg_path}
        except Exception:
            present = set()
        if cand_s not in present:
            pkg_path.append(cand_s)
        return


_extend_package_path_for_native_extension()


def _runtime_library_dirs() -> list[Path]:
    pkg_dirs = [Path(__file__).resolve().parent, *_iter_janusx_package_dirs()]
    dirs = [Path(sys.prefix) / "lib", Path(sys.base_prefix) / "lib"]
    for pkg_dir in pkg_dirs:
        dirs.extend([pkg_dir, pkg_dir / ".libs", pkg_dir.parent / "janusx.libs"])
    out: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        try:
            resolved = directory.resolve()
        except OSError:
            resolved = directory
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _is_inside_any(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _matches_numerical_runtime(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name.lower(), pattern.lower()) for pattern in patterns)


def _numerical_runtime_candidates(directory: Path) -> list[Path]:
    candidates: dict[str, Path] = {}
    for pattern in _LINUX_NUMERICAL_RUNTIME_PATTERNS:
        for candidate in directory.glob(pattern):
            if candidate.is_file():
                candidates.setdefault(candidate.name.lower(), candidate)
    return list(candidates.values())


def _is_trusted_runtime_library(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.resolve()
        parent = path.parent.resolve()
    except (OSError, RuntimeError):
        resolved = path
        parent = path.parent
    return _is_inside_any(resolved, roots) or _is_inside_any(parent, roots)


def _windows_trusted_library_dirs() -> list[Path]:
    dirs: list[Path] = []
    for pkg_dir in _iter_janusx_package_dirs():
        dirs.extend([pkg_dir, pkg_dir / ".libs", pkg_dir.parent / "janusx.libs"])
    for prefix in (Path(sys.prefix), Path(sys.base_prefix)):
        dirs.extend(
            [prefix / "Library" / "bin", prefix / "Library" / "lib", prefix / "DLLs", prefix / "bin"]
        )
    out: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        try:
            resolved = directory.resolve()
        except OSError:
            resolved = directory
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _elf_dynamic_info_from_sections(path: Path) -> tuple[set[str], str | None] | None:
    """Return (DT_NEEDED SONAMEs, DT_SONAME) from ELF section headers, or None."""
    try:
        with path.open("rb") as fh:
            e_ident = fh.read(16)
            if len(e_ident) < 16 or e_ident[:4] != b"\x7fELF":
                return None
            ei_class = e_ident[4]
            ei_data = e_ident[5]
            if ei_class not in (1, 2) or ei_data not in (1, 2):
                return None
            endian = "<" if ei_data == 1 else ">"
            if ei_class == 2:
                hdr = fh.read(48)
                if len(hdr) < 48:
                    return None
                e_shoff = struct.unpack_from(f"{endian}Q", hdr, 24)[0]
                e_shentsize, e_shnum = struct.unpack_from(f"{endian}HH", hdr, 42)
                min_shdr = 64
                dyn_fmt = f"{endian}qQ"
                dyn_size_step = 16
            else:
                hdr = fh.read(36)
                if len(hdr) < 36:
                    return None
                e_shoff = struct.unpack_from(f"{endian}I", hdr, 16)[0]
                e_shentsize, e_shnum = struct.unpack_from(f"{endian}HH", hdr, 30)
                min_shdr = 40
                dyn_fmt = f"{endian}iI"
                dyn_size_step = 8
            if e_shoff == 0 or e_shnum == 0 or e_shentsize < min_shdr:
                return None
            fh.seek(e_shoff)
            sh_table = fh.read(e_shnum * e_shentsize)
            if len(sh_table) < e_shnum * e_shentsize:
                return None

            dyn_section: tuple[int, int, int] | None = None
            for idx in range(e_shnum):
                base = idx * e_shentsize
                sh_type = struct.unpack_from(f"{endian}I", sh_table, base + 4)[0]
                if sh_type == 6:  # SHT_DYNAMIC
                    if ei_class == 2:
                        sh_offset, sh_size, sh_link = struct.unpack_from(
                            f"{endian}QQI", sh_table, base + 24
                        )
                    else:
                        sh_offset, sh_size, sh_link = struct.unpack_from(
                            f"{endian}III", sh_table, base + 16
                        )
                    dyn_section = (sh_offset, sh_size, sh_link)
                    break
            if dyn_section is None:
                return None
            dyn_offset, dyn_size, strtab_idx = dyn_section
            if strtab_idx >= e_shnum:
                return None
            str_base = strtab_idx * e_shentsize
            if ei_class == 2:
                str_offset, str_size = struct.unpack_from(f"{endian}QQ", sh_table, str_base + 24)
            else:
                str_offset, str_size = struct.unpack_from(f"{endian}II", sh_table, str_base + 16)

            fh.seek(str_offset)
            dynstr = fh.read(str_size)
            fh.seek(dyn_offset)
            dyn_bytes = fh.read(dyn_size)

            needed: set[str] = set()
            soname: str | None = None
            for pos in range(0, len(dyn_bytes) - dyn_size_step + 1, dyn_size_step):
                d_tag, d_val = struct.unpack_from(dyn_fmt, dyn_bytes, pos)
                if d_tag == 0:  # DT_NULL
                    break
                if d_tag in (1, 14) and d_val < len(dynstr):  # DT_NEEDED / DT_SONAME
                    end = dynstr.find(b"\x00", d_val)
                    raw_name = dynstr[d_val:] if end == -1 else dynstr[d_val:end]
                    if raw_name:
                        decoded = raw_name.decode("utf-8", errors="replace")
                        if d_tag == 1:
                            needed.add(decoded)
                        else:
                            soname = decoded
            return needed, soname
    except (OSError, struct.error, ValueError):
        return None


def _elf_dynamic_info_from_program_headers(path: Path) -> tuple[set[str], str | None] | None:
    """Read (DT_NEEDED SONAMEs, DT_SONAME) from PT_DYNAMIC when section headers are stripped."""
    try:
        with path.open("rb") as fh:
            e_ident = fh.read(16)
            if len(e_ident) < 16 or e_ident[:4] != b"\x7fELF":
                return None
            ei_class, ei_data = e_ident[4], e_ident[5]
            if ei_class not in (1, 2) or ei_data not in (1, 2):
                return None
            endian = "<" if ei_data == 1 else ">"
            if ei_class == 2:
                hdr = fh.read(48)
                if len(hdr) < 48:
                    return None
                phoff = struct.unpack_from(f"{endian}Q", hdr, 16)[0]
                phentsize, phnum = struct.unpack_from(f"{endian}HH", hdr, 38)
                ph_fmt, ph_size = f"{endian}IIQQQQQQ", 56
                dyn_fmt, dyn_step = f"{endian}qQ", 16
            else:
                hdr = fh.read(36)
                if len(hdr) < 36:
                    return None
                phoff = struct.unpack_from(f"{endian}I", hdr, 12)[0]
                phentsize, phnum = struct.unpack_from(f"{endian}HH", hdr, 26)
                ph_fmt, ph_size = f"{endian}IIIIIIII", 32
                dyn_fmt, dyn_step = f"{endian}iI", 8
            if phoff == 0 or phentsize < ph_size or phnum == 0:
                return None

            fh.seek(phoff)
            ph_table = fh.read(phnum * phentsize)
            if len(ph_table) < phnum * phentsize:
                return None

            loads: list[tuple[int, int, int, int]] = []
            dynamic: tuple[int, int] | None = None
            for idx in range(phnum):
                base = idx * phentsize
                fields = struct.unpack_from(ph_fmt, ph_table, base)
                if ei_class == 2:
                    p_type, _, p_offset, p_vaddr, _, p_filesz, p_memsz, _ = fields
                else:
                    p_type, p_offset, p_vaddr, _, p_filesz, p_memsz, _, _ = fields
                if p_type == 1:  # PT_LOAD
                    loads.append((p_vaddr, p_memsz, p_offset, p_filesz))
                elif p_type == 2:  # PT_DYNAMIC
                    dynamic = (p_offset, p_filesz)
            if dynamic is None:
                return None

            dyn_offset, dyn_size = dynamic
            fh.seek(dyn_offset)
            dyn_bytes = fh.read(dyn_size)

            needed_offsets: list[int] = []
            soname_offset: int | None = None
            str_vaddr: int | None = None
            str_size: int | None = None
            for pos in range(0, len(dyn_bytes) - dyn_step + 1, dyn_step):
                tag, value = struct.unpack_from(dyn_fmt, dyn_bytes, pos)
                if tag == 0:  # DT_NULL
                    break
                if tag == 1:  # DT_NEEDED
                    needed_offsets.append(value)
                elif tag == 5:  # DT_STRTAB
                    str_vaddr = value
                elif tag == 10:  # DT_STRSZ
                    str_size = value
                elif tag == 14:  # DT_SONAME
                    soname_offset = value
            if str_vaddr is None or str_size is None:
                return None

            str_offset: int | None = None
            for load_vaddr, _load_memsz, load_offset, load_filesz in loads:
                if load_vaddr <= str_vaddr < load_vaddr + load_filesz:
                    delta = str_vaddr - load_vaddr
                    if delta + str_size <= load_filesz:
                        str_offset = load_offset + delta
                    break
            if str_offset is None:
                return None

            fh.seek(str_offset)
            dynstr = fh.read(str_size)
            if len(dynstr) < str_size:
                return None

            def _read_cstr(offset: int) -> str | None:
                if offset >= len(dynstr):
                    return None
                end = dynstr.find(b"\x00", offset)
                raw_name = dynstr[offset:] if end == -1 else dynstr[offset:end]
                return raw_name.decode("utf-8", errors="replace") if raw_name else None

            needed: set[str] = set()
            for off in needed_offsets:
                name = _read_cstr(off)
                if name:
                    needed.add(name)
            soname = _read_cstr(soname_offset) if soname_offset is not None else None
            return needed, soname
    except (OSError, struct.error, ValueError):
        return None


def _elf_dynamic_info(path: Path) -> tuple[set[str], str | None] | None:
    info = _elf_dynamic_info_from_sections(path)
    if info is not None:
        return info
    return _elf_dynamic_info_from_program_headers(path)


def _elf_needed_sonames(path: Path) -> set[str] | None:
    info = _elf_dynamic_info(path)
    return info[0] if info is not None else None


def _elf_soname(path: Path) -> str | None:
    info = _elf_dynamic_info(path)
    return info[1] if info is not None else None


def _find_linux_native_extensions() -> list[Path]:
    pkg_dirs = [Path(__file__).resolve().parent, *_iter_janusx_package_dirs()]
    out: list[Path] = []
    seen: set[Path] = set()
    for pkg_dir in pkg_dirs:
        try:
            if not pkg_dir.is_dir():
                continue
            candidates = sorted(pkg_dir.glob("janusx*.so"))
        except OSError:
            continue
        for cand in candidates:
            try:
                if not cand.is_file():
                    continue
                resolved = cand.resolve()
            except OSError:
                continue
            if resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
    return out


def _collect_linux_numerical_needed_sonames(
    trusted: list[Path],
    extension_paths: list[Path] | None = None,
) -> set[str] | None:
    """Collect numerical runtime SONAMEs actually required by JanusX ELF binaries."""
    roots = list(extension_paths) if extension_paths is not None else _find_linux_native_extensions()
    if not roots:
        return None

    queue: list[Path] = list(roots)
    visited: set[Path] = set()
    active_sonames: set[str] = set()
    parsed_any = False
    while queue:
        current = queue.pop(0)
        try:
            resolved = current.resolve()
        except (OSError, RuntimeError):
            resolved = current
        if resolved in visited:
            continue
        visited.add(resolved)
        needed = _elf_needed_sonames(resolved)
        if needed is None:
            continue
        parsed_any = True
        for soname in needed:
            low = soname.lower()
            if _matches_numerical_runtime(low, _LINUX_NUMERICAL_RUNTIME_PATTERNS):
                active_sonames.add(low)
            for directory in trusted:
                candidate = directory / soname
                try:
                    if candidate.is_file():
                        cand_resolved = candidate.resolve()
                        if cand_resolved not in visited:
                            queue.append(cand_resolved)
                        break
                except OSError:
                    continue
    return active_sonames if parsed_any else None


def _validate_linux_library_environment(
    library_dirs: list[Path] | None = None,
    *,
    extension_paths: list[Path] | None = None,
) -> None:
    """Reject foreign BLAS/OpenMP runtimes that would override JanusX's active dependencies."""
    if sys.platform != "linux":
        return
    if str(os.environ.get(_ALLOW_EXTERNAL_RUNTIME_ENV, "")).strip().lower() in (
        "1", "true", "yes", "y", "on"
    ):
        return

    trusted: list[Path] = []
    for path in library_dirs or _runtime_library_dirs():
        try:
            trusted.append(path.resolve())
        except (OSError, RuntimeError):
            trusted.append(path)

    # Only flag an LD_LIBRARY_PATH entry when it wins the loader's search for
    # a SONAME that the active environment or wheel provides AND that JanusX
    # actually links (transitively via ELF DT_NEEDED). This prevents false
    # positives on HPC clusters where module-loaded GCC exposes libgomp.so.1
    # even though JanusX wheels bundle pthreads OpenBLAS with hashed SONAMEs.
    trusted_names: set[str] = set()
    for directory in trusted:
        for candidate in _numerical_runtime_candidates(directory):
            trusted_names.add(candidate.name.lower())

    active_sonames: set[str] | None = None
    if library_dirs is None or extension_paths is not None:
        active_sonames = _collect_linux_numerical_needed_sonames(
            trusted,
            extension_paths=extension_paths,
        )
    checked_names = (
        trusted_names & active_sonames if active_sonames is not None else trusted_names
    )

    def _matches_active_name(name: str, active: set[str]) -> bool:
        return name in active or any(name.startswith(f"{item}.") for item in active)

    library_path_dirs: list[Path] = []
    first_library_by_name: dict[str, Path] = {}
    raw_library_path = os.environ.get("LD_LIBRARY_PATH")
    library_path_entries = (
        raw_library_path.split(os.pathsep) if raw_library_path is not None else []
    )
    for raw in library_path_entries:
        # glibc treats an empty path component as the current directory.
        directory = Path(raw or os.curdir).expanduser()
        try:
            directory = directory.resolve()
        except (OSError, RuntimeError):
            pass
        library_path_dirs.append(directory)
        for candidate in _numerical_runtime_candidates(directory):
            first_library_by_name.setdefault(candidate.name.lower(), candidate)

    for name, candidate in first_library_by_name.items():
        if name not in checked_names or _is_trusted_runtime_library(candidate, trusted):
            continue
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            resolved = candidate
        raise RuntimeError(
            f"LD_LIBRARY_PATH would override the active BLAS/OpenMP runtime with {resolved}. "
            f"Keep the active environment's library directory first, remove the conflicting "
            f"directory, or set {_ALLOW_EXTERNAL_RUNTIME_ENV}=1 only when the external runtime "
            "is intentional."
        )

    preload = os.environ.get("LD_PRELOAD", "")
    for token in preload.replace(":", " ").split():
        candidate = Path(token).expanduser()
        if candidate.is_absolute() or os.sep in token:
            try:
                resolved = candidate.resolve()
            except (OSError, RuntimeError):
                resolved = candidate
            preload_names = {resolved.name.lower()}
            preload_soname = _elf_soname(resolved)
            if preload_soname:
                preload_names.add(preload_soname.lower())
            if not any(
                _matches_numerical_runtime(item, _LINUX_NUMERICAL_RUNTIME_PATTERNS)
                for item in preload_names
            ):
                continue
            if active_sonames is not None and not any(
                _matches_active_name(item, active_sonames) for item in preload_names
            ):
                continue
            if not _is_trusted_runtime_library(resolved, trusted):
                raise RuntimeError(
                    f"LD_PRELOAD injects an external BLAS/OpenMP runtime: {resolved}. "
                    f"Remove it or set {_ALLOW_EXTERNAL_RUNTIME_ENV}=1 only when intentional."
                )
            continue

        matches = [
            directory / token
            for directory in library_path_dirs
            if (directory / token).is_file()
        ]
        preload_names = {candidate.name.lower()}
        if matches:
            preload_soname = _elf_soname(matches[0])
            if preload_soname:
                preload_names.add(preload_soname.lower())
        if not any(
            _matches_numerical_runtime(item, _LINUX_NUMERICAL_RUNTIME_PATTERNS)
            for item in preload_names
        ):
            continue
        if active_sonames is not None and not any(
            _matches_active_name(item, active_sonames) for item in preload_names
        ):
            continue
        # A bare LD_PRELOAD name is resolved through the loader cache when it
        # is absent from LD_LIBRARY_PATH. We cannot prove which copy that
        # cache selects, so only accept it when the active copy is explicit in
        # LD_LIBRARY_PATH; otherwise fail closed.
        if not matches or not _is_trusted_runtime_library(matches[0], trusted):
            try:
                resolved_label = str(matches[0].resolve()) if matches else token
            except (OSError, RuntimeError):
                resolved_label = str(matches[0])
            raise RuntimeError(
                f"LD_PRELOAD names an external or unresolved BLAS/OpenMP runtime: {resolved_label}. "
                f"Use a library from the active Python environment or set "
                f"{_ALLOW_EXTERNAL_RUNTIME_ENV}=1 only when intentional."
            )


def _validate_windows_library_environment(library_dirs: list[Path] | None = None) -> None:
    """Reject stale explicit Windows DLL hints unless the caller opts in."""
    if str(os.environ.get(_ALLOW_EXTERNAL_RUNTIME_ENV, "")).strip().lower() in (
        "1", "true", "yes", "y", "on"
    ):
        return

    trusted = [path.resolve() for path in (library_dirs or _windows_trusted_library_dirs())]
    for variable in ("OPENBLAS_BIN_DIR", "OPENBLAS_LIB_DIR", "LIBRARY_BIN", "LIBRARY_LIB"):
        raw = str(os.environ.get(variable, "")).strip()
        if not raw:
            continue
        directory = Path(raw).expanduser().resolve()
        for pattern in _WINDOWS_NUMERICAL_RUNTIME_PATTERNS:
            for candidate in directory.glob(pattern):
                if candidate.is_file() and not _is_inside_any(candidate.resolve(), trusted):
                    raise RuntimeError(
                        f"{variable} points to an external BLAS/OpenMP DLL: {candidate.resolve()}. "
                        f"Remove the stale override or set {_ALLOW_EXTERNAL_RUNTIME_ENV}=1 only "
                        "when this external runtime is intentional."
                    )


def _windows_dll_search_dirs() -> list[Path]:
    pkg_dirs: list[Path] = []
    for raw in globals().get("__path__", [Path(__file__).resolve().parent]):
        try:
            pkg_dirs.append(Path(str(raw)))
        except Exception:
            continue
    if not pkg_dirs:
        pkg_dirs = [Path(__file__).resolve().parent]

    candidates: list[Path] = []
    for pkg_dir in pkg_dirs:
        candidates.extend([pkg_dir, pkg_dir / ".libs", pkg_dir.parent / "janusx.libs"])

    # Keep explicit development overrides, then add the actual interpreter's
    # DLL directories even when CONDA_PREFIX is absent or stale.
    for variable in ("OPENBLAS_BIN_DIR", "OPENBLAS_LIB_DIR", "LIBRARY_BIN", "LIBRARY_LIB"):
        raw = str(os.environ.get(variable, "")).strip()
        if raw:
            candidates.append(Path(raw))

    prefixes: list[Path] = []
    for prefix in (Path(sys.prefix), Path(sys.base_prefix), _matching_conda_prefix()):
        if prefix is None:
            continue
        try:
            resolved = prefix.resolve()
        except OSError:
            resolved = prefix
        if resolved not in prefixes:
            prefixes.append(resolved)
    for prefix in prefixes:
        candidates.extend(
            [prefix / "Library" / "bin", prefix / "Library" / "lib", prefix / "DLLs", prefix / "bin"]
        )

    out: list[Path] = []
    seen: set[str] = set()
    for directory in candidates:
        try:
            resolved = directory.resolve()
        except OSError:
            resolved = directory
        key = str(resolved).lower()
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        out.append(resolved)
    return out


def _init_windows_dll_search_path() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    _validate_windows_library_environment()
    for directory in _windows_dll_search_dirs():
        try:
            h = os.add_dll_directory(str(directory))
            _DLL_DIR_HANDLES.append(h)
        except Exception:
            pass


_init_windows_dll_search_path()


def _validate_macos_library_environment(library_dirs: list[Path]) -> None:
    """Reject loader overrides that can replace our BLAS/OpenMP runtime."""
    trusted = [directory.resolve() for directory in library_dirs]
    for variable in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        for raw in os.environ.get(variable, "").split(os.pathsep):
            if not raw:
                continue
            directory = Path(raw).expanduser().resolve()
            for pattern in (
                "libomp*.dylib", "libgomp*.dylib", "libiomp*.dylib",
                "libopenblas*.dylib", "libblas*.dylib", "liblapack*.dylib",
            ):
                for candidate in directory.glob(pattern):
                    if candidate.is_file() and not any(
                        candidate.resolve().is_relative_to(root) for root in trusted
                    ):
                        raise RuntimeError(
                            f"{variable} introduces an external BLAS/OpenMP runtime: {candidate}. "
                            f"Remove this directory from {variable} and restart Python/jx."
                        )
    for raw in os.environ.get("DYLD_INSERT_LIBRARIES", "").split(os.pathsep):
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if candidate.name.lower().startswith(("libomp", "libgomp", "libiomp", "libopenblas")) and not any(
            candidate.is_relative_to(root) for root in trusted
        ):
            raise RuntimeError(
                f"DYLD_INSERT_LIBRARIES preloads an external numerical runtime: {candidate}. "
                "Remove this entry and restart Python/jx."
            )


def _validate_macos_openblas_override(raw: str, library_dirs: list[Path]) -> None:
    # Inspect dependencies without dlopen(): a conflicting libomp can crash
    # before an exception handler gets control. Only explicit overrides need
    # this probe; ordinary environment/wheel selection has no subprocess cost.
    import subprocess

    path = Path(raw).expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("libopenblas*.dylib"))
        if not candidates:
            raise RuntimeError(f"No OpenBLAS dylib found in override directory: {path}")
    elif path.is_file():
        candidates = [path]
    else:
        raise RuntimeError(f"OpenBLAS override does not exist: {path}")
    trusted = [directory.resolve() for directory in library_dirs]
    if all(any(candidate.resolve().is_relative_to(root) for root in trusted) for candidate in candidates):
        # Environment and wheel runtimes must also work on machines without
        # Xcode command-line tools. Inspect only external overrides.
        return
    visited: set[Path] = set()
    queue = list(candidates)
    while queue:
        current = queue.pop().resolve()
        if current in visited:
            continue
        visited.add(current)
        try:
            result = subprocess.run(
                ["/usr/bin/otool", "-L", str(current)],
                capture_output=True, text=True, check=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Cannot inspect OpenBLAS override {current}: {exc}") from exc
        for line in result.stdout.splitlines()[1:]:
            dep = line.strip().split(" (", 1)[0]
            if not dep or dep.startswith(("/usr/lib/", "/System/Library/")):
                continue
            if dep.startswith("@loader_path/"):
                resolved = current.parent / dep[len("@loader_path/"):]
            elif dep.startswith("@rpath/"):
                # An external rpath dependency cannot be assumed to resolve
                # to the active environment's runtime.
                resolved = current.parent / dep[len("@rpath/"):]
            else:
                resolved = Path(dep)
            is_openmp = resolved.name.lower().startswith(("libomp", "libgomp", "libiomp"))
            if is_openmp and not any(resolved.resolve().is_relative_to(root) for root in trusted):
                raise RuntimeError(
                    f"OpenBLAS override {path} depends on an external OpenMP runtime ({dep}). "
                    "Unset JX_OPENBLAS_LIB_PATH/OPENBLAS_LIB_PATH to use the current environment "
                    "or wheel runtime, then restart Python/jx."
                )
            if resolved.is_file() and resolved.resolve() not in visited:
                queue.append(resolved)


def _init_macos_openblas_path() -> None:
    if sys.platform != "darwin":
        return

    def _preferred_candidate(cands: list[Path]) -> Path | None:
        if len(cands) == 0:
            return None
        preferred_order = [
            "libopenblas.0.dylib",
            "libopenblas.dylib",
        ]
        for name in preferred_order:
            for cand in cands:
                if cand.name == name:
                    return cand
        return cands[0]

    pkg_dirs: list[Path] = []
    for raw in globals().get("__path__", [Path(__file__).resolve().parent]):
        try:
            pkg_dirs.append(Path(str(raw)))
        except Exception:
            continue
    if len(pkg_dirs) == 0:
        pkg_dirs = [Path(__file__).resolve().parent]

    site_roots: list[Path] = []
    for raw in [*site.getsitepackages(), site.getusersitepackages()]:
        try:
            site_roots.append(Path(str(raw)))
        except Exception:
            continue

    library_dirs = [Path(sys.prefix) / "lib", Path(sys.base_prefix) / "lib"]
    for site_root in site_roots:
        library_dirs.append(site_root / "janusx.libs")
    for pkg_dir in pkg_dirs:
        library_dirs.extend([pkg_dir / ".dylibs", pkg_dir / ".libs", pkg_dir.parent / "janusx.libs"])
    _validate_macos_library_environment(library_dirs)
    backend = str(os.environ.get(
        "JX_RUST_EIGH_LAPACK_BACKEND", os.environ.get("JX_RUST_LAPACK_BACKEND", "auto")
    )).strip().lower()
    if backend in ("accelerate", "veclib"):
        return
    for key in ("JX_OPENBLAS_LIB_PATH", "OPENBLAS_LIB_PATH"):
        explicit = str(os.environ.get(key, "")).strip()
        if explicit:
            _validate_macos_openblas_override(explicit, library_dirs)
            # Pin the validated override so Rust's two aliases agree.
            os.environ["JX_OPENBLAS_LIB_PATH"] = str(Path(explicit).expanduser().resolve())
            return

    # Prefer wheel-local janusx.libs pthreads OpenBLAS over environment-level
    # OpenBLAS so a user's conda OpenMP OpenBLAS cannot pull a conflicting
    # libomp.dylib into the process ahead of XGBoost/scikit-learn.
    candidate_groups: list[list[Path]] = []
    for site_root in site_roots:
        candidate_groups.append(
            sorted((site_root / "janusx.libs").glob("libopenblas*.dylib"))
        )
    wheel_local_cands: list[Path] = []
    for pkg_dir in pkg_dirs:
        wheel_local_cands.extend(sorted((pkg_dir / ".dylibs").glob("libopenblas*.dylib")))
        wheel_local_cands.extend(sorted((pkg_dir.parent / "janusx.libs").glob("libopenblas*.dylib")))
        wheel_local_cands.extend(sorted((pkg_dir / ".libs").glob("libopenblas*.dylib")))
    candidate_groups.append(wheel_local_cands)

    # Fallback to the active interpreter's own prefix (never an unrelated
    # shell CONDA_PREFIX) when no wheel-bundled OpenBLAS is present.
    runtime_prefixes = [sys.prefix, sys.base_prefix]
    conda_prefix = _matching_conda_prefix()
    if conda_prefix is not None:
        runtime_prefixes.append(conda_prefix)
    seen_prefixes: set[Path] = set()
    for prefix in runtime_prefixes:
        lib_dir = Path(prefix).resolve() / "lib"
        if lib_dir in seen_prefixes:
            continue
        seen_prefixes.add(lib_dir)
        candidate_groups.append(sorted(lib_dir.glob("libopenblas*.dylib")))

    picked: Path | None = None
    for group in candidate_groups:
        picked = _preferred_candidate(group)
        if picked is not None:
            break
    if picked is None:
        if backend == "openblas":
            raise RuntimeError(
                "OpenBLAS LAPACK was requested, but no runtime exists in the current Python "
                "environment or wheel. Install a compatible OpenBLAS runtime, set a validated "
                "JX_OPENBLAS_LIB_PATH, or use JX_RUST_EIGH_LAPACK_BACKEND=accelerate."
            )
        return

    os.environ.setdefault("JX_OPENBLAS_LIB_PATH", str(picked))


_init_macos_openblas_path()
_validate_linux_library_environment()


def _should_prewarm_macos_native_eigh_backend() -> bool:
    if sys.platform != "darwin":
        return False
    raw_pref = str(
        os.environ.get(
            "JX_RUST_EIGH_LAPACK_BACKEND",
            os.environ.get("JX_RUST_LAPACK_BACKEND", "auto"),
        )
    ).strip().lower()
    # Respect explicit Accelerate requests.
    if raw_pref in ("accelerate", "veclib"):
        return False
    return True


def _prewarm_macos_native_eigh_backend() -> str:
    """
    Best-effort early native-LAPACK prewarm on macOS.

    Import the JanusX extension and probe the Rust eigh backend before heavy
    optional dependencies (SciPy/sklearn/xgboost/matplotlib) can pull a
    foreign OpenMP runtime into the process and force auto-selection back to
    Accelerate.
    """
    if not _should_prewarm_macos_native_eigh_backend():
        return ""
    try:
        from . import janusx as _jx
    except Exception:
        return ""
    try:
        probe = getattr(_jx, "rust_eigh_lapack_backend", None)
        if not callable(probe):
            return ""
        return str(probe()).strip().lower()
    except Exception:
        return ""


_ = _prewarm_macos_native_eigh_backend()

from . import linalg as linalg


def _attach_native_linalg_alias() -> None:
    """
    Expose linear algebra convenience API on native extension namespace:
        import janusx.janusx as jx
        jx.linalg.eigh(...)
    """
    try:
        from . import janusx as _jx
    except Exception:
        return
    try:
        if not hasattr(_jx, "linalg"):
            setattr(_jx, "linalg", linalg)
    except Exception:
        pass


_attach_native_linalg_alias()


_EIGH_WARN_ONCE_ENV = "JANUSX_EIGH_FALLBACK_WARNED"
_EIGH_WARN_SENTINEL_TTL_SEC = 300.0


def _env_truthy(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _stderr_is_interactive() -> bool:
    try:
        return bool(getattr(sys.stderr, "isatty", lambda: False)())
    except Exception:
        return False


def _clear_current_stdout_line() -> None:
    try:
        if not bool(getattr(sys.stdout, "isatty", lambda: False)()):
            return
    except Exception:
        return
    try:
        cols = int(getattr(os, "get_terminal_size", lambda *_: os.terminal_size((80, 20)))().columns)
    except Exception:
        cols = 80
    cols = max(1, int(cols))
    try:
        sys.stdout.write("\r" + (" " * cols) + "\r")
        sys.stdout.flush()
    except Exception:
        pass


def _eigh_warn_sentinel_file() -> str:
    try:
        uid = str(int(os.getuid()))
    except Exception:
        uid = "nouid"
    return os.path.join(tempfile.gettempdir(), f".janusx_eigh_fallback_warned_{uid}")


def _is_macos_eigh_fallback_backend(backend_label: str) -> bool:
    b = str(backend_label).strip().lower()
    if b == "accelerate":
        return True
    return "accelerate" in b and b.startswith("lapack_")


def _should_check_macos_eigh_hint() -> bool:
    if sys.platform != "darwin":
        return False
    raw_pref = str(
        os.environ.get(
            "JX_RUST_EIGH_LAPACK_BACKEND",
            os.environ.get("JX_RUST_LAPACK_BACKEND", "auto"),
        )
    ).strip().lower()
    # Explicit Accelerate LAPACK request is intentional.
    if raw_pref in ("accelerate", "veclib"):
        return False
    return True


def _probe_macos_eigh_backend() -> str:
    try:
        from . import janusx as _jx
    except Exception:
        return ""
    try:
        probe_sgemm = getattr(_jx, "rust_sgemm_backend", None)
        probe_eigh = getattr(_jx, "rust_eigh_lapack_backend", None)
        if not callable(probe_sgemm):
            return ""
        sgemm_backend = str(probe_sgemm()).strip().lower()
        # OpenBLAS BLAS builds should not trigger fallback hints.
        if sgemm_backend != "accelerate":
            return ""
        if not callable(probe_eigh):
            return ""
        return str(probe_eigh()).strip().lower()
    except Exception:
        return ""


def maybe_emit_macos_eigh_fallback_hint(
    *,
    evd_backend: str | None = None,
    logger=None,
) -> bool:
    """
    Emit a gentle one-time hint when macOS eigendecomposition falls back to
    Accelerate LAPACK in auto/openblas mode.

    Returns True when the hint was emitted in this call.
    """
    if not _should_check_macos_eigh_hint():
        return False
    if sys.platform != "darwin":
        return False
    # Process-tree once-only sentinel.
    if _env_truthy(_EIGH_WARN_ONCE_ENV):
        return False
    # Cross-process once-only sentinel (pipeline/subprocess chains) is only
    # applied in non-interactive mode. Interactive CLI invocations should still
    # surface the warning once per run.
    use_cross_process_sentinel = not _stderr_is_interactive()
    sentinel_path = _eigh_warn_sentinel_file() if use_cross_process_sentinel else ""
    if use_cross_process_sentinel and os.path.exists(sentinel_path):
        try:
            age = max(0.0, float(time.time()) - float(os.path.getmtime(sentinel_path)))
            if age <= float(_EIGH_WARN_SENTINEL_TTL_SEC):
                return False
        except Exception:
            return False

    backend = (
        str(evd_backend).strip().lower()
        if evd_backend is not None and str(evd_backend).strip() != ""
        else _probe_macos_eigh_backend()
    )
    if not _is_macos_eigh_fallback_backend(backend):
        return False

    os.environ[_EIGH_WARN_ONCE_ENV] = "1"
    if use_cross_process_sentinel:
        try:
            with open(sentinel_path, "w", encoding="utf-8") as fh:
                fh.write("1\n")
        except Exception:
            pass

    backend_line = "  backend: Accelerate LAPACK"
    warning_body = (
        "OpenBLAS LAPACK was unavailable or skipped due to a runtime conflict; "
        "JanusX used Accelerate instead.\n"
        "  Set JX_RUST_EIGH_LAPACK_BACKEND=accelerate to silence this fallback."
    )
    _clear_current_stdout_line()
    if logger is not None and hasattr(logger, "info"):
        logger.info(backend_line)
        if hasattr(logger, "warning"):
            logger.warning(warning_body)
        else:
            logger.info(f"Warning: {warning_body}")
    else:
        sys.stderr.write(backend_line + "\n")
        sys.stderr.write("Warning: " + warning_body + "\n")
        sys.stderr.flush()
    return True


__all__ = ["linalg", "maybe_emit_macos_eigh_fallback_hint"]
