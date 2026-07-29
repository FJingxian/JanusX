from __future__ import annotations

import os
import sys
from typing import Any, Optional, Sequence

from .pathcheck import safe_expanduser


def option_present(argv: Optional[Sequence[str]], *flags: str) -> bool:
    tokens = list(sys.argv[1:] if argv is None else argv)
    flag_set = {str(f).strip() for f in flags if str(f).strip()}
    for tok in tokens:
        t = str(tok).strip()
        if t in flag_set:
            return True
        for f in flag_set:
            if t.startswith(f + "="):
                return True
    return False


def looks_like_output_directory_hint(path_value: str) -> bool:
    raw = str(path_value).strip()
    if raw == "":
        return False
    if raw.endswith(("/", "\\")):
        return True
    norm = os.path.normpath(raw)
    tail = os.path.basename(norm)
    if tail in {"", ".", ".."}:
        return True
    try:
        expanded = safe_expanduser(raw)
        return bool(expanded.exists() and expanded.is_dir())
    except Exception:
        return False


def resolve_output_prefix(
    out_value: Optional[str],
    legacy_prefix: Optional[str],
    auto_prefix: str,
    *,
    out_was_explicit: bool,
    fallback_prefix: str = "JanusX",
) -> tuple[str, str, str]:
    auto_prefix_text = str(auto_prefix).strip() or str(fallback_prefix).strip() or "JanusX"
    legacy_prefix_text = str(legacy_prefix).strip() if legacy_prefix is not None else ""
    out_text = str(out_value).strip() if out_value is not None else ""

    if legacy_prefix_text != "":
        if out_was_explicit and out_text != "":
            raw_prefix = os.path.join(out_text, legacy_prefix_text)
        else:
            raw_prefix = legacy_prefix_text
    elif out_was_explicit and out_text != "":
        if looks_like_output_directory_hint(out_text):
            raw_prefix = os.path.join(out_text, auto_prefix_text)
        else:
            raw_prefix = out_text
    else:
        raw_prefix = auto_prefix_text

    outprefix = os.path.normpath(str(safe_expanduser(raw_prefix)))
    outdir = os.path.dirname(outprefix)
    if str(outdir).strip() == "":
        outdir = "."
    outstem = os.path.basename(outprefix)
    if str(outstem).strip() == "":
        outstem = auto_prefix_text
        outprefix = os.path.join(outdir, outstem)
    return str(outdir), str(outprefix), str(outstem)


def apply_output_prefix_compat(
    args: Any,
    auto_prefix: str,
    *,
    argv: Optional[Sequence[str]] = None,
    fallback_prefix: str = "JanusX",
) -> tuple[str, str, str]:
    out_was_explicit = bool(option_present(argv, "-o", "--out"))
    prefix_was_explicit = bool(option_present(argv, "-prefix", "--prefix"))
    legacy_prefix = getattr(args, "prefix", None) if prefix_was_explicit else None
    outdir, outprefix, outstem = resolve_output_prefix(
        getattr(args, "out", None),
        legacy_prefix,
        auto_prefix,
        out_was_explicit=out_was_explicit,
        fallback_prefix=fallback_prefix,
    )
    setattr(args, "_out_was_explicit", out_was_explicit)
    setattr(args, "_prefix_was_explicit", prefix_was_explicit)
    setattr(args, "out", str(outdir))
    setattr(args, "out_dir", str(outdir))
    setattr(args, "outprefix", str(outprefix))
    setattr(args, "out_stem", str(outstem))
    setattr(args, "prefix", str(outstem))
    return str(outdir), str(outprefix), str(outstem)
