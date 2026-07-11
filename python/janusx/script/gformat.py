# -*- coding: utf-8 -*-
"""
JanusX: Genotype format converter (gformat)

Supported inputs
----------------
- `-vcf/--vcf`    : VCF / VCF.GZ genotype file
- `-hmp/--hmp`    : HMP / HMP.GZ genotype file
- `-bfile/--bfile`: PLINK prefix (.bed/.bim/.fam)
- `-file/--file`  : text / npy genotype matrix or prefix

Supported outputs
-----------------
- `-fmt plink` : PLINK bed/bim/fam
- `-fmt vcf`   : bgzipped VCF
- `-fmt hmp`   : HapMap text (.hmp)
- `-fmt txt`   : `prefix.txt` + `prefix.id` + `prefix.site`
- `-fmt npy`   : `prefix.npy` + `prefix.id` + `prefix.site`

Default output format is PLINK (`-fmt plink`) when `-fmt` is omitted.

Notes
-----
- `-file` inputs must carry sibling `prefix.id`.
- `-file -> vcf/hmp/plink` requires real site metadata via `prefix.bsite`/`prefix.site` or `prefix.bim`.
- `txt/npy` outputs always write headerless `prefix.site` as four columns:
  `CHR POS REF ALT`.
"""

from __future__ import annotations

import argparse
import os
import socket
import time
import re
import gzip
import shutil
import subprocess
import itertools
from threading import Lock
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from janusx.gfreader import (
    convert_genotypes,
    inspect_genotype_file,
    load_bed_2bit_packed,
    load_genotype_chunks,
    prepare_bed_logic_keep_mask,
    save_genotype_streaming,
    SiteInfo,
)
from janusx.janusx import (
    VcfChunkReader,
    HmpChunkReader,
    TxtChunkReader,
    count_vcf_snps,
    count_hmp_snps,
)
from janusx.gfreader.gfreader import _resolve_input

try:
    from janusx.janusx import bed_packed_ld_prune_maf_priority as _bed_packed_ld_prune_maf_priority
except Exception:
    _bed_packed_ld_prune_maf_priority = None

try:
    from janusx.janusx import bed_prune_to_plink_rust as _bed_prune_to_plink_rust
except Exception:
    _bed_prune_to_plink_rust = None

try:
    from janusx.janusx import bed_prune_mask_to_plink_rust as _bed_prune_mask_to_plink_rust
except Exception:
    _bed_prune_mask_to_plink_rust = None

try:
    from janusx.janusx import bed_filter_to_plink_rust as _bed_filter_to_plink_rust
except Exception:
    _bed_filter_to_plink_rust = None

try:
    from janusx.janusx import packed_prune_kernel_stats as _packed_prune_kernel_stats
except Exception:
    _packed_prune_kernel_stats = None

from ._common.config_render import emit_cli_configuration
from ._common.cli_args import (
    add_common_genotype_source_args,
    add_common_out_arg,
    add_common_prefix_arg,
    add_common_thread_arg,
)
from ._common.genoio import (
    GENOTYPE_TEXT_SUFFIXES,
    build_prefix_candidates,
    determine_genotype_source_from_args as determine_genotype_source,
    discover_id_sidecar_path,
    discover_site_path,
    write_gfd_output,
    write_npy_output,
    write_text_output,
)
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    format_output_display,
    format_path_for_display,
    ensure_plink_prefix_exists,
)
from ._common.progress import (
    CliStatus,
    ProgressAdapter,
    _emit_to_file_handlers,
    build_rich_progress,
    log_success,
    stdout_is_tty,
)
from ._common.genocache import configure_genotype_cache_from_out
from ._common.threads import (
    apply_blas_thread_env,
    detect_effective_threads,
    format_requested_thread_usage,
)


def _normalize_chr_key(chrom: str) -> str:
    s = str(chrom).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    s = s.strip().upper()
    if s == "M":
        return "MT"
    return s


def _split_tokens(line: str) -> list[str]:
    return [x for x in re.split(r"[,\s]+", line.strip()) if x]


_SIMPLE_SNP_ALLELES = frozenset({"A", "C", "G", "T"})


def _normalize_site_allele(value: object) -> str:
    return str(value).strip().upper()


def _normalize_snp_name_template(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        raise ValueError("--snp-name cannot be empty.")
    if any(ch.isspace() for ch in text):
        raise ValueError("--snp-name cannot contain whitespace.")
    has_brace_chr = "{chr}" in text
    has_brace_pos = "{pos}" in text
    if has_brace_chr or has_brace_pos:
        if not (has_brace_chr and has_brace_pos):
            raise ValueError("--snp-name must contain both {chr} and {pos}.")
        return text
    if "chr" not in text or "pos" not in text:
        raise ValueError("--snp-name must contain both chr and pos placeholders.")
    return text


def _format_snp_name(template: str, chrom: object, pos: object) -> str:
    chrom_s = str(chrom).strip()
    try:
        pos_s = str(int(pos))
    except Exception:
        pos_s = str(pos).strip()
    if "{chr}" in template or "{pos}" in template:
        return template.replace("{chr}", chrom_s).replace("{pos}", pos_s)
    return template.replace("chr", chrom_s).replace("pos", pos_s)


def _rename_sites_for_snp_name(
    sites: Sequence[SiteInfo],
    snp_name_template: str | None,
) -> list[SiteInfo]:
    if snp_name_template is None:
        return list(sites)
    out: list[SiteInfo] = []
    for site in sites:
        try:
            pos_i = int(site.pos)
        except Exception:
            pos_i = int(float(site.pos))
        out.append(
            SiteInfo(
                str(site.chrom),
                pos_i,
                _format_snp_name(snp_name_template, site.chrom, pos_i),
                str(site.ref_allele),
                str(site.alt_allele),
            )
        )
    return out


def _iter_snp_named_chunks(
    chunks: Iterator[tuple[np.ndarray, list[SiteInfo]]],
    snp_name_template: str | None,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    if snp_name_template is None:
        yield from chunks
        return
    for geno, sites in chunks:
        yield geno, _rename_sites_for_snp_name(sites, snp_name_template)


def _rewrite_plink_bim_snp_names(prefix: str, snp_name_template: str | None) -> None:
    if snp_name_template is None:
        return
    bim_path = f"{str(prefix)}.bim"
    tmp_path = f"{bim_path}.jxsnprename.tmp"
    with open(bim_path, "r", encoding="utf-8", errors="replace") as fr, open(
        tmp_path, "w", encoding="utf-8"
    ) as fw:
        for line_no, line in enumerate(fr, start=1):
            tok = _split_tokens(line)
            if len(tok) < 6:
                raise ValueError(
                    f"Invalid BIM row at {bim_path}:{line_no} while applying --snp-name."
                )
            chrom = str(tok[0])
            pos = int(float(tok[3]))
            tok[1] = _format_snp_name(snp_name_template, chrom, pos)
            fw.write("\t".join(tok) + "\n")
    os.replace(tmp_path, bim_path)


def _simple_snp_only_requested(*, snps_only: bool, biallelic_only: bool) -> bool:
    # `--biallelic-only` is a historical flag name. The intended behavior here is
    # to keep only simple single-base A/C/G/T sites.
    return bool(snps_only) or bool(biallelic_only)


def _site_is_biallelic(site: SiteInfo) -> bool:
    ref_allele = _normalize_site_allele(site.ref_allele)
    alt_allele = _normalize_site_allele(site.alt_allele)
    if (
        not ref_allele
        or not alt_allele
        or ref_allele == "."
        or alt_allele == "."
        or "," in ref_allele
        or "," in alt_allele
    ):
        return False
    return ref_allele != alt_allele


def _site_is_simple_snp(site: SiteInfo) -> bool:
    if not _site_is_biallelic(site):
        return False
    ref_allele = _normalize_site_allele(site.ref_allele)
    alt_allele = _normalize_site_allele(site.alt_allele)
    return (
        len(ref_allele) == 1
        and len(alt_allele) == 1
        and ref_allele in _SIMPLE_SNP_ALLELES
        and alt_allele in _SIMPLE_SNP_ALLELES
    )


def _apply_site_type_filters(
    geno: np.ndarray,
    sites: Sequence[SiteInfo],
    *,
    snps_only: bool,
    biallelic_only: bool,
) -> tuple[np.ndarray, list[SiteInfo]]:
    if not _simple_snp_only_requested(
        snps_only=bool(snps_only),
        biallelic_only=bool(biallelic_only),
    ):
        return np.asarray(geno), list(sites)

    keep_idx: list[int] = []
    keep_sites: list[SiteInfo] = []
    for i, site in enumerate(sites):
        if _site_is_simple_snp(site):
            keep_idx.append(i)
            keep_sites.append(site)

    arr = np.asarray(geno)
    if len(keep_idx) == len(sites):
        return arr, list(sites)
    if len(keep_idx) == 0:
        if arr.ndim == 2:
            return np.empty((0, arr.shape[1]), dtype=arr.dtype), []
        return np.empty((0, 0), dtype=np.float32), []
    return np.ascontiguousarray(arr[keep_idx]), keep_sites


def _log_status_file_only(logger, message: str) -> None:
    if not _emit_to_file_handlers(logger, 20, str(message)):
        logger.info(str(message))


class _ChromPruneProgress:
    def __init__(self, max_visible: int) -> None:
        self.max_visible = int(max(1, min(5, int(max_visible))))
        self._lock = Lock()
        self._progress = None
        self._states: dict[int, tuple[str, int, int, bool]] = {}
        self._visible: dict[int, object] = {}
        self._pending: list[int] = []
        if stdout_is_tty():
            self._progress = build_rich_progress(
                description_template="[green]{task.description}",
                show_spinner=False,
                show_bar=True,
                show_percentage=True,
                show_elapsed=True,
                show_remaining=True,
                field_templates=["{task.fields[postfix]}"],
                transient=True,
            )
            if self._progress is not None:
                self._progress.start()

    @staticmethod
    def _label(chrom: str) -> str:
        c = str(chrom)
        if c.lower().startswith("chr"):
            return f"LD prune {c}"
        return f"LD prune chr{c}"

    def _remember_pending(self, block_idx: int) -> None:
        if block_idx not in self._pending:
            self._pending.append(block_idx)

    def _show_locked(self, block_idx: int) -> None:
        if self._progress is None or block_idx in self._visible:
            return
        chrom, done, total, finished = self._states.get(block_idx, ("?", 0, 1, False))
        if finished:
            return
        task_id = self._progress.add_task(
            self._label(chrom),
            total=max(1, int(total)),
            completed=max(0, min(int(done), max(1, int(total)))),
            postfix=f"{int(done)}/{int(total)}",
        )
        self._visible[block_idx] = task_id

    def _promote_locked(self) -> None:
        while len(self._visible) < self.max_visible and self._pending:
            idx = self._pending.pop(0)
            state = self._states.get(idx)
            if state is None or state[3] or idx in self._visible:
                continue
            self._show_locked(idx)

    def update(self, block_idx: int, chrom: str, done: int, total: int, finished: bool) -> None:
        if self._progress is None:
            return
        idx = int(block_idx)
        total_i = int(max(1, int(total)))
        done_i = int(max(0, min(int(done), total_i)))
        finished_b = bool(finished)
        with self._lock:
            self._states[idx] = (str(chrom), done_i, total_i, finished_b)
            if idx not in self._visible and not finished_b:
                self._remember_pending(idx)
            if idx in self._visible:
                task_id = self._visible[idx]
                self._progress.update(
                    task_id,
                    total=total_i,
                    completed=done_i,
                    postfix=f"{done_i}/{total_i}",
                )
            elif not finished_b and len(self._visible) < self.max_visible:
                if idx in self._pending:
                    self._pending.remove(idx)
                self._show_locked(idx)

            if finished_b:
                if idx in self._pending:
                    self._pending.remove(idx)
                task_id = self._visible.pop(idx, None)
                if task_id is not None:
                    try:
                        self._progress.update(
                            task_id,
                            total=total_i,
                            completed=total_i,
                            postfix=f"{total_i}/{total_i}",
                        )
                        self._progress.remove_task(task_id)
                    except Exception:
                        pass
                self._promote_locked()

    def close(self) -> None:
        with self._lock:
            progress = self._progress
            self._progress = None
            self._visible.clear()
            self._pending.clear()
        if progress is not None:
            progress.stop()


def _parse_int_token(token: str, name: str) -> int:
    t = str(token).strip()
    if t == "":
        raise ValueError(f"Empty integer token for {name}.")
    try:
        return int(float(t))
    except Exception as e:
        raise ValueError(f"Invalid integer token for {name}: {token}") from e


def _parse_site_token(token: str) -> tuple[str, int]:
    t = str(token).strip()
    if t == "":
        raise ValueError("Empty site token in --extract file.")
    if ":" in t:
        c, p = t.split(":", 1)
        return _normalize_chr_key(c), _parse_int_token(p, "site position")
    if "_" in t:
        c, p = t.split("_", 1)
        return _normalize_chr_key(c), _parse_int_token(p, "site position")
    raise ValueError(
        f"Unsupported site token '{token}'. Use CHR:POS / CHR_POS, or two columns: CHR POS."
    )


def _read_keep_samples(path: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = _split_tokens(s)
            if not tok:
                continue
            sid = str(tok[0]).strip()
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)
    if not out:
        raise ValueError(f"--keep file is empty or invalid: {path}")
    return out


def _select_keep_samples_from_fam(prefix: str, keep_file: str) -> list[str]:
    """
    Build keep-sample IDs in FAM order (PLINK-compatible ordering).
    """
    keep_list = _read_keep_samples(str(keep_file))
    keep_set = set(keep_list)
    out: list[str] = []
    fam_path = f"{str(prefix)}.fam"
    with open(fam_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()
            if len(tok) < 2:
                continue
            sid = str(tok[1]).strip()
            if sid and sid in keep_set:
                out.append(sid)
    if len(out) == 0:
        miss_preview = ", ".join(keep_list[:10])
        raise ValueError(f"--keep selected 0 samples. Missing examples: {miss_preview}")
    return out


def _read_extract_sites(path: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = _split_tokens(s)
            if not tok:
                continue
            if len(tok) == 1:
                key = _parse_site_token(tok[0])
            else:
                key = (_normalize_chr_key(tok[0]), _parse_int_token(tok[1], "site position"))
            if key not in seen:
                seen.add(key)
                out.append(key)
    if not out:
        raise ValueError(f"--extract file is empty or invalid: {path}")
    return out


def _read_extract_ranges(path: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = _split_tokens(s)
            if not tok:
                continue
            if len(tok) == 1:
                c, p = _parse_site_token(tok[0])
                out.append((c, p, p))
                continue
            if len(tok) == 2:
                c = _normalize_chr_key(tok[0])
                p = _parse_int_token(tok[1], "range position")
                out.append((c, p, p))
                continue
            c = _normalize_chr_key(tok[0])
            a = _parse_int_token(tok[1], "range start")
            b = _parse_int_token(tok[2], "range end")
            start, end = (a, b) if a <= b else (b, a)
            out.append((c, start, end))
    if not out:
        raise ValueError(f"--extract range file is empty or invalid: {path}")
    return out


def _parse_extract_arg(values: list[str] | None) -> tuple[str | None, str | None]:
    if not values:
        return None, None
    parts = [str(v).strip() for v in values if str(v).strip()]
    if len(parts) == 1:
        return "sites", parts[0]
    if len(parts) == 2 and parts[0].lower() == "range":
        return "range", parts[1]
    raise ValueError("Invalid --extract usage. Use '--extract <file>' or '--extract range <file>'.")


def _expand_chr_selector(tokens: list[str] | None) -> set[str]:
    if not tokens:
        return set()
    out: set[str] = set()
    for tok in tokens:
        for part in str(tok).split(","):
            p = part.strip()
            if not p:
                continue
            if "-" in p:
                left, right = p.split("-", 1)
                left = left.strip()
                right = right.strip()
                if left.isdigit() and right.isdigit():
                    a = int(left)
                    b = int(right)
                    if a > b:
                        raise ValueError(f"Invalid --chr range: {p}")
                    for k in range(a, b + 1):
                        out.add(_normalize_chr_key(str(k)))
                    continue
            out.add(_normalize_chr_key(p))
    return out


@dataclass
class SiteFilterSpec:
    chr_keys: set[str] | None = None
    bp_min: int | None = None
    bp_max: int | None = None
    exact_sites: set[tuple[str, int]] | None = None
    ranges: list[tuple[str, int, int]] | None = None

    def active(self) -> bool:
        return bool(self.chr_keys) or self.bp_min is not None or self.bp_max is not None or bool(self.exact_sites) or bool(self.ranges)


@dataclass
class PruneSpec:
    window_variants: int | None
    window_bp: int | None
    step_variants: int
    r2_threshold: float

    def label(self) -> str:
        if self.window_bp is not None:
            return (
                f"window={int(self.window_bp)}bp, "
                f"step={int(self.step_variants)}, r2={float(self.r2_threshold):.4f}"
            )
        return (
            f"window={int(self.window_variants or 0)} variants, "
            f"step={int(self.step_variants)}, r2={float(self.r2_threshold):.4f}"
        )


def _parse_prune_window(token: str) -> tuple[int | None, int | None]:
    t = str(token).strip().lower()
    if t == "":
        raise ValueError("Empty prune window token.")
    # PLINK-compatible syntax: a bare number is a variant-count window.
    # Use explicit kb/bp suffixes for physical-position windows.
    if t.endswith("kb"):
        v = float(t[:-2].strip())
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"Invalid prune window (kb): {token}")
        return None, int(max(1, round(v * 1000.0)))
    if t.endswith("bp"):
        v = float(t[:-2].strip())
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"Invalid prune window (bp): {token}")
        return None, int(max(1, round(v)))
    v = float(t)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"Invalid prune window: {token}")
    if abs(v - round(v)) > 1e-9:
        raise ValueError(
            f"Invalid prune window: {token}. Use an integer variant count, "
            "or add kb/bp suffix for a physical window."
        )
    return int(max(1, round(v))), None


def _parse_prune_args(values: list[str] | None) -> PruneSpec | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError(
            "Invalid --prune usage. Expected 3 values: "
            "--prune <window size['kb']> <step size (variant ct)> <r^2 threshold>"
        )
    w_var, w_bp = _parse_prune_window(values[0])
    step = int(float(str(values[1]).strip()))
    if step <= 0:
        raise ValueError(f"--prune step must be > 0, got {values[1]!r}")
    r2 = float(str(values[2]).strip())
    if (not np.isfinite(r2)) or (r2 <= 0.0) or (r2 > 1.0):
        raise ValueError(f"--prune r^2 threshold must be in (0, 1], got {values[2]!r}")
    return PruneSpec(
        window_variants=w_var,
        window_bp=w_bp,
        step_variants=int(step),
        r2_threshold=float(r2),
    )


def _env_truthy(name: str, default: str = "0") -> bool:
    v = str(os.getenv(name, default)).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _rust_convert_fastpath_enabled() -> bool:
    raw = str(os.getenv("JANUSX_GFORMAT_CONVERT_DIRECT", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _resolve_max_output_sites_env() -> int | None:
    raw = os.getenv("JX_GFORMAT_MAX_OUTPUT_SITES")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"", "none", "off", "false", "disable", "disabled"}:
        return None
    try:
        value = int(text)
    except Exception as e:
        raise ValueError(
            "JX_GFORMAT_MAX_OUTPUT_SITES must be a non-negative integer "
            "(0 disables variant-row output)."
        ) from e
    if value < 0:
        raise ValueError("JX_GFORMAT_MAX_OUTPUT_SITES must be >= 0.")
    return int(value)


def _resolve_prune_mask_mmap_window_mb() -> int | None:
    raw = os.getenv("JX_GFORMAT_PRUNE_MASK_MMAP_WINDOW_MB")
    if raw is None:
        return 128
    text = str(raw).strip().lower()
    if text in {"", "none", "off", "false", "disable", "disabled", "full"}:
        return None
    try:
        value = int(text)
    except Exception as e:
        raise ValueError(
            "JX_GFORMAT_PRUNE_MASK_MMAP_WINDOW_MB must be a positive integer "
            "or one of none/off/full."
        ) from e
    if value <= 0:
        raise ValueError("JX_GFORMAT_PRUNE_MASK_MMAP_WINDOW_MB must be > 0 when set.")
    return int(value)


def _resolve_ld_prune_backend() -> str:
    raw = str(os.getenv("JX_LD_PRUNE_BACKEND", "")).strip().lower()
    if raw in {"", "auto", "default", "rust"}:
        return "rust"
    if raw in {"plink", "plink-external", "external-plink"}:
        return "plink"
    return "rust"


def _format_prune_window_for_plink(spec: PruneSpec) -> str:
    if spec.window_variants is not None:
        return str(int(spec.window_variants))
    if spec.window_bp is None:
        raise ValueError("Invalid prune window for PLINK backend.")
    kb = float(int(spec.window_bp)) / 1000.0
    tok = f"{kb:.6f}".rstrip("0").rstrip(".")
    if tok == "":
        tok = "0"
    return f"{tok}kb"


def _count_nonempty_text_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and (not s.startswith("#")):
                n += 1
    return int(n)


def _inspect_plink_prefix_light(prefix: str) -> tuple[list[str], int]:
    p = str(prefix)
    if p.lower().endswith((".bed", ".bim", ".fam")):
        p = str(Path(p).with_suffix(""))
    fam_path = f"{p}.fam"
    bim_path = f"{p}.bim"

    sample_ids: list[str] = []
    with open(fam_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            tok = _split_tokens(s)
            if not tok:
                continue
            sample_ids.append(str(tok[1] if len(tok) >= 2 else tok[0]))
    if not sample_ids:
        raise ValueError(f"No samples found in FAM: {fam_path}")

    n_sites = 0
    last = b""
    with open(bim_path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            n_sites += chunk.count(b"\n")
            last = chunk[-1:]
    if last and last != b"\n":
        n_sites += 1
    if n_sites <= 0:
        raise ValueError(f"No variants found in BIM: {bim_path}")
    return sample_ids, int(n_sites)


def _run_external_command(
    cmd: list[str],
    *,
    logger,
) -> None:
    logger.info("Running external command: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = str(proc.stdout or "").strip()
    if out:
        logger.info(out)
    if int(proc.returncode) != 0:
        tail = "\n".join(out.splitlines()[-20:]) if out else "(no output)"
        raise RuntimeError(
            "External command failed "
            f"(exit={int(proc.returncode)}): {' '.join(cmd)}\n{tail}"
        )


def _site_passes(site: SiteInfo, flt: SiteFilterSpec) -> bool:
    c = _normalize_chr_key(str(site.chrom))
    p = int(site.pos)
    if flt.chr_keys and c not in flt.chr_keys:
        return False
    if flt.bp_min is not None and p < int(flt.bp_min):
        return False
    if flt.bp_max is not None and p > int(flt.bp_max):
        return False
    if flt.exact_sites and (c, p) not in flt.exact_sites:
        return False
    if flt.ranges:
        matched = False
        for rc, rs, re in flt.ranges:
            if c == rc and int(rs) <= p <= int(re):
                matched = True
                break
        if not matched:
            return False
    return True


def _prepare_direct_filter_expr(
    *,
    snp_sites: list[tuple[str, int]] | None,
    bim_range: tuple[str, int, int] | None,
    chr_keys: set[str] | None,
    bp_min: int | None,
    bp_max: int | None,
    ranges: list[tuple[str, int, int]] | None,
) -> tuple[
    set[tuple[str, int]] | None,
    tuple[str, int, int] | None,
    set[str] | None,
    int | None,
    int | None,
    list[tuple[str, int, int]] | None,
]:
    site_set: set[tuple[str, int]] | None = None
    if snp_sites:
        site_set = set((_normalize_chr_key(c), int(p)) for c, p in snp_sites)

    bim_norm: tuple[str, int, int] | None = None
    if bim_range is not None:
        bim_norm = (_normalize_chr_key(str(bim_range[0])), int(bim_range[1]), int(bim_range[2]))

    chr_set: set[str] | None = None
    if chr_keys:
        chr_set = set(_normalize_chr_key(c) for c in chr_keys)

    ranges_norm: list[tuple[str, int, int]] | None = None
    if ranges:
        ranges_norm = [(_normalize_chr_key(c), int(s), int(e)) for c, s, e in ranges]

    return site_set, bim_norm, chr_set, bp_min, bp_max, ranges_norm


def _site_passes_direct_expr(
    site: SiteInfo,
    *,
    site_set: set[tuple[str, int]] | None,
    bim_range: tuple[str, int, int] | None,
    chr_set: set[str] | None,
    bp_min: int | None,
    bp_max: int | None,
    ranges: list[tuple[str, int, int]] | None,
) -> bool:
    c = _normalize_chr_key(str(site.chrom))
    p = int(site.pos)
    if site_set and (c, p) not in site_set:
        return False
    if bim_range is not None:
        bc, bs, be = bim_range
        if not (c == bc and bs <= p <= be):
            return False
    if chr_set and c not in chr_set:
        return False
    if bp_min is not None and p < int(bp_min):
        return False
    if bp_max is not None and p > int(bp_max):
        return False
    if ranges:
        hit = False
        for rc, rs, re in ranges:
            if c == rc and rs <= p <= re:
                hit = True
                break
        if not hit:
            return False
    return True


def _refine_plink_source_keep_mask(
    prefix: str,
    keep_mask: np.ndarray,
    *,
    snp_sites: list[tuple[str, int]] | None,
    bim_range: tuple[str, int, int] | None,
    chr_keys: list[str] | set[str] | None,
    bp_min: int | None,
    bp_max: int | None,
    ranges: list[tuple[str, int, int]] | None,
    require_simple_snp: bool = False,
) -> np.ndarray:
    keep = np.ascontiguousarray(
        np.asarray(keep_mask, dtype=np.bool_).reshape(-1),
        dtype=np.bool_,
    )
    n_sites = int(keep.shape[0])
    if n_sites <= 0:
        return np.zeros((0,), dtype=np.bool_)
    if (
        not snp_sites
        and bim_range is None
        and not chr_keys
        and bp_min is None
        and bp_max is None
        and not ranges
        and (not require_simple_snp)
    ):
        return keep

    site_set, bim_norm, chr_set, bp_lo, bp_hi, ranges_norm = _prepare_direct_filter_expr(
        snp_sites=snp_sites,
        bim_range=bim_range,
        chr_keys=(set(chr_keys) if chr_keys is not None else None),
        bp_min=bp_min,
        bp_max=bp_max,
        ranges=ranges,
    )
    bim_path = f"{str(prefix)}.bim"
    resolved_rows = 0
    with open(bim_path, "r", encoding="utf-8", errors="replace") as f:
        for row_idx, line in enumerate(f):
            if row_idx >= n_sites:
                break
            resolved_rows = row_idx + 1
            if not bool(keep[row_idx]):
                continue
            s = line.strip()
            tok = _split_tokens(s)
            if len(tok) < 6:
                raise ValueError(
                    f"Invalid BIM row at {bim_path}:{row_idx + 1} (need >=6 columns)."
                )
            chrom = str(tok[0])
            pos = int(float(tok[3]))
            snp = str(tok[1]).strip()
            if snp == "" or snp == ".":
                snp = f"{chrom}_{pos}"
            site = SiteInfo(chrom, int(pos), snp, str(tok[4]), str(tok[5]))
            ok = _site_passes_direct_expr(
                site,
                site_set=site_set,
                bim_range=bim_norm,
                chr_set=chr_set,
                bp_min=bp_lo,
                bp_max=bp_hi,
                ranges=ranges_norm,
            )
            if ok and require_simple_snp:
                ok = _site_is_simple_snp(site)
            if not ok:
                keep[row_idx] = False
    if resolved_rows != n_sites:
        raise ValueError(
            f"BIM row count mismatch while refining source keep mask: resolved={resolved_rows}, expected={n_sites}"
        )
    return keep


def _filter_chunk_by_site(
    geno: np.ndarray,
    sites: list[SiteInfo],
    flt: SiteFilterSpec,
) -> tuple[np.ndarray, list[SiteInfo]]:
    if not flt.active():
        return geno, sites
    keep_idx: list[int] = []
    keep_sites: list[SiteInfo] = []
    for i, s in enumerate(sites):
        if _site_passes(s, flt):
            keep_idx.append(i)
            keep_sites.append(s)
    if len(keep_idx) == len(sites):
        return geno, sites
    if not keep_idx:
        return np.empty((0, geno.shape[1]), dtype=np.float32), []
    return np.asarray(geno[keep_idx, :], dtype=np.float32), keep_sites


def _heter_keep_mask(geno: np.ndarray, het_threshold: float) -> np.ndarray:
    """
    Keep SNP rows whose heterozygosity rate is at most `het`.

    - Heterozygous call is defined as genotype == 1 (within tolerance).
    - Missing calls (NaN/Inf/<0) are ignored in the denominator.
    - Rows with no observed calls are dropped.
    """
    g = np.asarray(geno, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError(f"Invalid genotype matrix shape for het filter: {g.shape}")
    m = int(g.shape[0])
    keep = np.ones(m, dtype=bool)
    h = float(het_threshold)
    if m == 0 or h <= 0.0:
        return keep

    valid = np.isfinite(g) & (g >= 0.0)
    non_missing = np.sum(valid, axis=1, dtype=np.int64)
    has_obs = non_missing > 0

    het_count = np.sum(np.isclose(g, 1.0, atol=1e-6) & valid, axis=1, dtype=np.int64)
    het_rate = np.zeros(m, dtype=np.float64)
    het_rate[has_obs] = (
        het_count[has_obs].astype(np.float64) / non_missing[has_obs].astype(np.float64)
    )

    max_het = float(h)
    keep &= has_obs & (het_rate <= max_het)
    return keep


def _has_real_file_sites(file_arg: str) -> bool:
    raw_prefix, _ = build_prefix_candidates(file_arg, text_suffixes=GENOTYPE_TEXT_SUFFIXES)
    prefixes = [raw_prefix]
    return discover_site_path(prefixes) is not None


def _resolve_output_target(args) -> tuple[str, str, str]:
    _, prefix = determine_genotype_source(args)
    outdir = os.path.normpath(str(args.out or "."))
    fmt = str(args.format).lower()
    base = os.path.join(outdir, prefix)
    if fmt == "vcf":
        return fmt, base, f"{base}.vcf.gz"
    if fmt == "hmp":
        return fmt, base, f"{base}.hmp"
    if fmt == "txt":
        return fmt, base, f"{base}.txt"
    if fmt == "npy":
        return fmt, base, f"{base}.npy"
    if fmt == "gfd":
        return fmt, base, f"{base}.bin"
    if fmt == "plink":
        return fmt, base, base
    raise ValueError("Unsupported output format. Use one of: plink, vcf, hmp, txt, npy, gfd.")

def _iter_filtered_chunks(
    chunks: Iterator[tuple[np.ndarray, list[SiteInfo]]],
    flt: SiteFilterSpec,
    *,
    het_threshold: float = 0.0,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    for geno, sites in chunks:
        g2 = np.asarray(geno, dtype=np.float32)
        s2 = list(sites)

        if flt.active():
            g2, s2 = _filter_chunk_by_site(g2, s2, flt)
        if len(s2) == 0:
            continue

        if float(het_threshold) > 0.0:
            keep_mask = _heter_keep_mask(g2, float(het_threshold))
            if not np.any(keep_mask):
                continue
            if not np.all(keep_mask):
                keep_idx = np.flatnonzero(keep_mask).astype(int).tolist()
                g2 = np.asarray(g2[keep_idx, :], dtype=np.float32)
                s2 = [s2[i] for i in keep_idx]

        yield g2, s2


def _count_sites_from_chunks(
    chunks: Iterator[tuple[np.ndarray, list[SiteInfo]]],
) -> int:
    cnt = 0
    for block, _sites in chunks:
        cnt += int(np.asarray(block).shape[0])
    return int(cnt)


def _peek_chunk_iterator(
    chunks: Iterator[tuple[np.ndarray, list[SiteInfo]]],
) -> tuple[
    tuple[np.ndarray, list[SiteInfo]] | None,
    Iterator[tuple[np.ndarray, list[SiteInfo]]],
]:
    it = iter(chunks)
    first = next(it, None)
    if first is None:
        return None, iter(())
    return first, itertools.chain([first], it)


def _iter_counted_chunks(
    chunks: Iterator[tuple[np.ndarray, list[SiteInfo]]],
    counter: list[int],
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    for block, sites in chunks:
        counter[0] += int(np.asarray(block).shape[0])
        yield block, sites


def _mode_impute_012(geno: np.ndarray) -> np.ndarray:
    x = np.asarray(geno, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Genotype block must be 2D, got {x.shape}")
    if x.shape[0] == 0 or x.shape[1] == 0:
        return np.empty_like(x, dtype=np.int8)

    g = np.full(x.shape, -9, dtype=np.int8)
    valid = np.isfinite(x) & (x >= 0.0)
    if np.any(valid):
        vals = np.rint(x[valid]).astype(np.int16, copy=False)
        vals = np.clip(vals, 0, 2).astype(np.int8, copy=False)
        g[valid] = vals

    c0 = np.sum(g == 0, axis=1, dtype=np.int64)
    c1 = np.sum(g == 1, axis=1, dtype=np.int64)
    c2 = np.sum(g == 2, axis=1, dtype=np.int64)
    modes = np.argmax(np.stack([c0, c1, c2], axis=1), axis=1).astype(np.int8, copy=False)

    miss = g < 0
    if np.any(miss):
        rows, cols = np.where(miss)
        g[rows, cols] = modes[rows]
    return g


def _transform_gfd_block(
    geno: np.ndarray,
    sites: Sequence[SiteInfo],
) -> tuple[np.ndarray, list[SiteInfo]]:
    x = np.asarray(geno, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Genotype block must be 2D, got {x.shape}")
    m, n = x.shape
    if m != len(sites):
        raise ValueError(f"Genotype/sites mismatch: rows={m}, sites={len(sites)}")
    if m == 0:
        return np.empty((0, n), dtype=np.uint8), []

    g = _mode_impute_012(x)
    left = (g != 0).astype(np.uint8, copy=False)
    right = (g != 2).astype(np.uint8, copy=False)

    out = np.empty((m * 2, n), dtype=np.uint8)
    out[0::2, :] = left
    out[1::2, :] = right

    out_sites: list[SiteInfo] = []
    for s in sites:
        chrom0 = str(s.chrom)
        try:
            pos0 = int(s.pos)
        except Exception:
            pos0 = 0
        out_sites.append(SiteInfo(f"{chrom0}_1", pos0, "0", "1"))
        out_sites.append(SiteInfo(f"{chrom0}_2", pos0, "0", "1"))
    return out, out_sites


def _iter_gfd_chunks(
    chunks: Iterator[tuple[np.ndarray, list[SiteInfo]]],
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    for geno, sites in chunks:
        arr, gfd_sites = _transform_gfd_block(np.asarray(geno, dtype=np.float32), list(sites))
        if arr.shape[0] == 0:
            continue
        yield arr, gfd_sites


def _stack_chunks(
    chunks: Iterator[tuple[np.ndarray, list[SiteInfo]]],
) -> tuple[np.ndarray, list[SiteInfo]]:
    mats: list[np.ndarray] = []
    sites_all: list[SiteInfo] = []
    for geno, sites in chunks:
        gg = np.asarray(geno, dtype=np.float32)
        if gg.ndim != 2:
            raise ValueError(f"Invalid genotype block shape: {gg.shape}")
        if gg.shape[0] != len(sites):
            raise ValueError(
                f"Genotype/sites length mismatch: rows={gg.shape[0]}, sites={len(sites)}"
            )
        if gg.shape[0] == 0:
            continue
        mats.append(gg)
        sites_all.extend(list(sites))
    if len(mats) == 0:
        return np.empty((0, 0), dtype=np.float32), []
    mat = np.concatenate(mats, axis=0).astype(np.float32, copy=False)
    return np.ascontiguousarray(mat, dtype=np.float32), sites_all


def _read_plink_bim_sites(prefix: str) -> list[SiteInfo]:
    p = str(prefix)
    bim_path = p if p.lower().endswith(".bim") else f"{p}.bim"
    if not Path(bim_path).is_file():
        raise ValueError(f"Cannot find BIM file for PLINK prefix: {prefix}")
    out: list[SiteInfo] = []
    with open(bim_path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = _split_tokens(s)
            if len(tok) < 6:
                raise ValueError(
                    f"Invalid BIM row at {bim_path}:{line_no} (need >=6 columns)."
                )
            chrom = str(tok[0])
            pos = int(float(tok[3]))
            snp = str(tok[1]).strip()
            if snp == "" or snp == ".":
                snp = f"{chrom}_{pos}"
            ref = str(tok[4])
            alt = str(tok[5])
            out.append(SiteInfo(chrom, int(pos), snp, ref, alt))
    if len(out) == 0:
        raise ValueError(f"No variants found in BIM: {bim_path}")
    return out


def _ld_prune_with_maf_priority_packed_plink(
    prefix: str,
    spec: PruneSpec,
    *,
    threads: int = 0,
) -> tuple[np.ndarray, list[SiteInfo], np.ndarray, int]:
    if _bed_packed_ld_prune_maf_priority is None:
        raise RuntimeError(
            "Rust packed LD prune kernel is unavailable. Rebuild JanusX extension."
        )
    packed_raw, _miss_raw, _maf_raw, _std_raw, n_samples = load_bed_2bit_packed(str(prefix))
    packed = np.ascontiguousarray(packed_raw, dtype=np.uint8)
    if packed.ndim != 2:
        raise ValueError(f"Invalid packed BED matrix shape: {packed.shape}")
    sites_all = _read_plink_bim_sites(str(prefix))
    if int(packed.shape[0]) != len(sites_all):
        raise ValueError(
            "BED/BIM row count mismatch for prune: "
            f"packed_rows={packed.shape[0]}, bim_rows={len(sites_all)}"
        )

    keep = _ld_prune_with_maf_priority_packed_sites(
        packed=packed,
        sites=sites_all,
        n_samples=int(n_samples),
        spec=spec,
        threads=int(threads),
    )
    return keep, sites_all, packed, int(n_samples)


def _ld_prune_with_maf_priority_packed_sites(
    *,
    packed: np.ndarray,
    sites: Sequence[SiteInfo],
    n_samples: int,
    spec: PruneSpec,
    threads: int = 0,
) -> np.ndarray:
    if _bed_packed_ld_prune_maf_priority is None:
        raise RuntimeError(
            "Rust packed LD prune kernel is unavailable. Rebuild JanusX extension."
        )
    packed_arr = np.ascontiguousarray(np.asarray(packed, dtype=np.uint8))
    if packed_arr.ndim != 2:
        raise ValueError(f"Invalid packed BED matrix shape: {packed_arr.shape}")
    site_list = list(sites)
    if int(packed_arr.shape[0]) != len(site_list):
        raise ValueError(
            "Packed prune input mismatch: "
            f"packed_rows={packed_arr.shape[0]}, sites={len(site_list)}"
        )

    chrom_code_map: dict[str, int] = {}
    chrom_codes = np.zeros((len(site_list),), dtype=np.int32)
    positions = np.zeros((len(site_list),), dtype=np.int64)
    next_code = 0
    for i, s in enumerate(site_list):
        c = str(s.chrom)
        if c not in chrom_code_map:
            chrom_code_map[c] = int(next_code)
            next_code += 1
        chrom_codes[i] = int(chrom_code_map[c])
        positions[i] = int(s.pos)

    keep = np.asarray(
        _bed_packed_ld_prune_maf_priority(
            packed=packed_arr,
            n_samples=int(n_samples),
            chrom_codes=chrom_codes,
            positions=positions,
            window_bp=(
                int(spec.window_bp)
                if spec.window_bp is not None
                else None
            ),
            window_variants=(
                int(spec.window_variants)
                if spec.window_variants is not None
                else None
            ),
            step_variants=int(spec.step_variants),
            r2_threshold=float(spec.r2_threshold),
            threads=int(max(0, int(threads))),
        ),
        dtype=bool,
    ).reshape(-1)
    if int(keep.size) != len(site_list):
        raise ValueError(
            f"Rust prune keep-mask mismatch: got {keep.size}, expected {len(site_list)}"
        )
    return keep


def _write_plink_subset_from_packed(
    *,
    src_prefix: str,
    out_prefix: str,
    packed: np.ndarray,
    keep_mask: np.ndarray,
    source_row_idx: np.ndarray | None = None,
    max_output_sites: int | None = None,
) -> None:
    src = str(src_prefix)
    out = str(out_prefix)
    src_bed = f"{src}.bed"
    src_bim = f"{src}.bim"
    src_fam = f"{src}.fam"
    out_bed = f"{out}.bed"
    out_bim = f"{out}.bim"
    out_fam = f"{out}.fam"

    if not Path(src_bed).is_file() or not Path(src_bim).is_file() or not Path(src_fam).is_file():
        raise ValueError(f"PLINK source prefix is incomplete: {src_prefix}")

    keep = np.asarray(keep_mask, dtype=bool).reshape(-1)
    x = np.ascontiguousarray(np.asarray(packed, dtype=np.uint8))
    if x.ndim != 2:
        raise ValueError(f"Packed BED matrix must be 2D, got {x.shape}")
    if int(x.shape[0]) != int(keep.size):
        raise ValueError(
            f"Packed rows/keep mask mismatch: rows={x.shape[0]}, keep={keep.size}"
        )
    if source_row_idx is None:
        selected_source_idx = np.flatnonzero(keep).astype(np.int64, copy=False)
    else:
        source_idx = np.ascontiguousarray(
            np.asarray(source_row_idx, dtype=np.int64).reshape(-1),
            dtype=np.int64,
        )
        if int(source_idx.shape[0]) != int(keep.size):
            raise ValueError(
                "Packed rows/source row index mismatch: "
                f"rows={x.shape[0]}, source_idx={source_idx.shape[0]}"
            )
        selected_source_idx = source_idx[np.flatnonzero(keep).astype(np.int64, copy=False)]
        if selected_source_idx.size > 1 and bool(
            np.any(selected_source_idx[1:] < selected_source_idx[:-1])
        ):
            raise ValueError("Selected PLINK source rows must remain in ascending order.")
    if max_output_sites is not None:
        selected_source_idx = selected_source_idx[: max(0, int(max_output_sites))]
    keep_idx = np.flatnonzero(keep).astype(np.int64, copy=False)
    if max_output_sites is not None:
        keep_idx = keep_idx[: max(0, int(max_output_sites))]

    Path(out_bed).parent.mkdir(parents=True, exist_ok=True)

    # Write SNP-major BED directly from packed rows.
    with open(out_bed, "wb") as fw:
        fw.write(bytes([0x6C, 0x1B, 0x01]))
        for i in keep_idx.tolist():
            fw.write(memoryview(x[int(i)]))

    # Keep original BIM lines for kept variants.
    with open(src_bim, "r", encoding="utf-8", errors="replace") as fr, open(
        out_bim, "w", encoding="utf-8"
    ) as fw:
        row_ptr = 0
        n_selected = int(selected_source_idx.shape[0])
        for i, line in enumerate(fr):
            if row_ptr >= n_selected:
                break
            if int(selected_source_idx[row_ptr]) == int(i):
                fw.write(line)
                row_ptr += 1

    # FAM is unchanged when sample filtering is absent.
    shutil.copyfile(src_fam, out_fam)


def _ld_prune_with_mask_first_streaming_pipeline(
    *,
    src_prefix: str,
    out_prefix: str,
    spec: PruneSpec,
    threads: int,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    snps_only: bool,
    biallelic_only: bool,
    snp_sites: list[tuple[str, int]] | None,
    bim_range: tuple[str, int, int] | None,
    chr_keys: list[str] | None,
    bp_min: int | None,
    bp_max: int | None,
    ranges: list[tuple[str, int, int]] | None,
    max_output_sites: int | None = None,
) -> tuple[int, int, float]:
    if _bed_prune_mask_to_plink_rust is None:
        raise RuntimeError(
            "Rust mask-stream prune backend is unavailable. Rebuild JanusX extension."
        )
    t0 = time.time()
    mmap_window_mb = _resolve_prune_mask_mmap_window_mb()
    site_keep_raw, _n_samples, total_sites = prepare_bed_logic_keep_mask(
        str(src_prefix),
        sample_indices=None,
        maf_threshold=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
        het_threshold=float(het_threshold),
        snps_only=_simple_snp_only_requested(
            snps_only=bool(snps_only),
            biallelic_only=bool(biallelic_only),
        ),
        mmap_window_mb=mmap_window_mb,
        threads=max(1, int(threads)),
    )
    site_keep = np.ascontiguousarray(
        np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1),
        dtype=np.bool_,
    )
    total_sites_i = int(total_sites)
    if int(site_keep.shape[0]) != total_sites_i:
        raise ValueError(
            "BED logic site-keep mask mismatch: "
            f"mask={site_keep.shape[0]}, total_sites={total_sites_i}"
        )
    site_keep = _refine_plink_source_keep_mask(
        str(src_prefix),
        site_keep,
        snp_sites=snp_sites,
        bim_range=bim_range,
        chr_keys=chr_keys,
        bp_min=bp_min,
        bp_max=bp_max,
        ranges=ranges,
        require_simple_snp=False,
    )
    pre_n = int(np.sum(site_keep, dtype=np.int64))
    if pre_n <= 0:
        raise ValueError(
            "All variants were filtered out before LD prune "
            "(-extract/-chr/-from-bp/-to-bp/-maf/-geno/-het/--snps-only/--biallelic-only)."
        )
    keep_n, total_n = _bed_prune_mask_to_plink_rust(
        src_prefix=str(src_prefix),
        out_prefix=str(out_prefix),
        keep_mask=np.ascontiguousarray(site_keep, dtype=np.bool_),
        window_bp=(int(spec.window_bp) if spec.window_bp is not None else None),
        window_variants=(
            int(spec.window_variants) if spec.window_variants is not None else None
        ),
        step_variants=int(spec.step_variants),
        r2_threshold=float(spec.r2_threshold),
        threads=int(threads),
        max_output_sites=max_output_sites,
    )
    keep_n = int(keep_n)
    total_n = int(total_n)
    if total_n != pre_n:
        raise ValueError(
            f"Rust mask-stream prune pre-count mismatch: got {total_n}, expected {pre_n}"
        )
    if keep_n <= 0:
        raise ValueError(
            "All variants were filtered out by LD prune; "
            "try a looser --prune r^2 threshold or larger window."
        )
    return keep_n, pre_n, float(time.time() - t0)


def _ld_prune_with_external_plink(
    *,
    src_prefix: str,
    out_prefix: str,
    spec: PruneSpec,
    threads: int,
    logger,
) -> tuple[int, int, float]:
    plink_bin_raw = str(os.getenv("JX_PLINK_BIN", "plink")).strip() or "plink"
    if os.path.sep in plink_bin_raw or plink_bin_raw.startswith("."):
        if os.path.isfile(plink_bin_raw) and os.access(plink_bin_raw, os.X_OK):
            plink_bin = plink_bin_raw
        else:
            plink_bin = ""
    else:
        found = shutil.which(plink_bin_raw)
        plink_bin = str(found) if found else ""
    if plink_bin == "":
        raise RuntimeError(
            "External PLINK backend requested but PLINK binary was not found. "
            "Please install PLINK 1.9 and/or set JX_PLINK_BIN=/path/to/plink."
        )

    window_tok = _format_prune_window_for_plink(spec)
    step_tok = str(int(spec.step_variants))
    r2_tok = f"{float(spec.r2_threshold):.12g}"
    thread_tok = str(max(1, int(threads)))

    tmp_prefix = (
        f"{str(out_prefix)}.jxprune_tmp_{int(os.getpid())}_{int(time.time() * 1000)}"
    )
    prune_in = f"{tmp_prefix}.prune.in"
    t0 = time.time()
    keep_tmp = _env_truthy("JX_KEEP_PLINK_TMP", "0")
    tmp_paths = [
        f"{tmp_prefix}.prune.in",
        f"{tmp_prefix}.prune.out",
        f"{tmp_prefix}.log",
        f"{tmp_prefix}.nosex",
    ]
    try:
        _run_external_command(
            [
                str(plink_bin),
                "--bfile",
                str(src_prefix),
                "--indep-pairwise",
                str(window_tok),
                str(step_tok),
                str(r2_tok),
                "--threads",
                str(thread_tok),
                "--out",
                str(tmp_prefix),
            ],
            logger=logger,
        )
        if not Path(prune_in).is_file():
            raise RuntimeError(f"PLINK did not produce prune list: {prune_in}")

        _run_external_command(
            [
                str(plink_bin),
                "--bfile",
                str(src_prefix),
                "--extract",
                str(prune_in),
                "--make-bed",
                "--threads",
                str(thread_tok),
                "--out",
                str(out_prefix),
            ],
            logger=logger,
        )

        total_n = _count_nonempty_text_lines(f"{str(src_prefix)}.bim")
        keep_n = _count_nonempty_text_lines(f"{str(out_prefix)}.bim")
        return int(keep_n), int(total_n), float(time.time() - t0)
    finally:
        if not keep_tmp:
            for p in tmp_paths:
                try:
                    if Path(p).exists():
                        Path(p).unlink()
                except Exception:
                    pass


def _ld_prune_with_rust_filtered_pipeline(
    *,
    src_prefix: str,
    out_prefix: str,
    spec: PruneSpec,
    threads: int,
    maf_threshold: float,
    max_missing_rate: float,
    model: str,
    het_threshold: float,
    sample_ids: list[str] | None,
    snp_sites: list[tuple[str, int]] | None,
    bim_range: tuple[str, int, int] | None,
    chr_keys: list[str] | None,
    bp_min: int | None,
    bp_max: int | None,
    ranges: list[tuple[str, int, int]] | None,
    logger,
    filter_progress_callback=None,
    filter_progress_every: int = 0,
    prune_progress_callback=None,
    prune_progress_every: int = 0,
    max_output_sites: int | None = None,
) -> tuple[int, int, int, float, int]:
    if _bed_filter_to_plink_rust is None or _bed_prune_to_plink_rust is None:
        raise RuntimeError(
            "Rust prune/filter backend is unavailable. Rebuild JanusX extension."
        )

    tmp_prefix = (
        f"{str(out_prefix)}.jxpruneflt_tmp_{int(os.getpid())}_{int(time.time() * 1000)}"
    )
    keep_tmp = _env_truthy("JX_KEEP_RUST_TMP", "0")
    tmp_paths = [
        f"{tmp_prefix}.bed",
        f"{tmp_prefix}.bim",
        f"{tmp_prefix}.fam",
    ]
    t0 = time.time()
    try:
        filter_args = dict(
            src_prefix=str(src_prefix),
            out_prefix=str(tmp_prefix),
            maf_threshold=float(maf_threshold),
            max_missing_rate=float(max_missing_rate),
            fill_missing=False,
            model=str(model),
            het_threshold=float(het_threshold),
            sample_ids=sample_ids,
            snp_sites=snp_sites,
            bim_range=bim_range,
            chr_keys=chr_keys,
            bp_min=bp_min,
            bp_max=bp_max,
            ranges=ranges,
        )
        if filter_progress_callback is not None:
            try:
                filt_keep, _filt_scanned, out_n = _bed_filter_to_plink_rust(
                    **filter_args,
                    progress_callback=filter_progress_callback,
                    progress_every=max(0, int(filter_progress_every)),
                )
            except TypeError as ex:
                emsg = str(ex).lower()
                if ("keyword" not in emsg) and ("argument" not in emsg) and ("positional" not in emsg):
                    raise
                filt_keep, _filt_scanned, out_n = _bed_filter_to_plink_rust(**filter_args)
        else:
            filt_keep, _filt_scanned, out_n = _bed_filter_to_plink_rust(**filter_args)
        filt_keep = int(filt_keep)
        out_n = int(out_n)
        if filt_keep <= 0:
            raise ValueError(
                "All variants were filtered out before LD prune "
                "(-extract/-chr/-from-bp/-to-bp/-maf/-geno/-het)."
            )

        prune_args = dict(
            src_prefix=str(tmp_prefix),
            out_prefix=str(out_prefix),
            window_bp=(
                int(spec.window_bp)
                if spec.window_bp is not None
                else None
            ),
            window_variants=(
                int(spec.window_variants)
                if spec.window_variants is not None
                else None
            ),
            step_variants=int(spec.step_variants),
            r2_threshold=float(spec.r2_threshold),
            threads=int(threads),
            max_output_sites=(
                int(max_output_sites) if max_output_sites is not None else None
            ),
        )
        if prune_progress_callback is not None:
            try:
                keep_n, total_n = _bed_prune_to_plink_rust(
                    **prune_args,
                    progress_callback=prune_progress_callback,
                    progress_every=max(0, int(prune_progress_every)),
                )
            except TypeError as ex:
                emsg = str(ex).lower()
                if ("keyword" not in emsg) and ("argument" not in emsg) and ("positional" not in emsg):
                    raise
                legacy_prune_args = dict(prune_args)
                legacy_prune_args.pop("max_output_sites", None)
                keep_n, total_n = _bed_prune_to_plink_rust(**legacy_prune_args)
        else:
            try:
                keep_n, total_n = _bed_prune_to_plink_rust(**prune_args)
            except TypeError as ex:
                emsg = str(ex).lower()
                if ("keyword" not in emsg) and ("argument" not in emsg) and ("positional" not in emsg):
                    raise
                legacy_prune_args = dict(prune_args)
                legacy_prune_args.pop("max_output_sites", None)
                keep_n, total_n = _bed_prune_to_plink_rust(**legacy_prune_args)
        keep_n = int(keep_n)
        total_n = int(total_n)
        if total_n != filt_keep:
            logger.warning(
                "Rust prune total mismatch after filtered stage: "
                f"filter_kept={filt_keep}, prune_total={total_n}"
            )
        return keep_n, total_n, filt_keep, float(time.time() - t0), out_n
    finally:
        if not keep_tmp:
            for p in tmp_paths:
                try:
                    if Path(p).exists():
                        Path(p).unlink()
                except Exception:
                    pass


def _compute_maf_and_z(geno: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(geno, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Genotype matrix must be 2D, got {x.shape}")
    m, n = x.shape
    valid = np.isfinite(x) & (x >= 0.0)
    cnt = valid.sum(axis=1).astype(np.float64, copy=False)
    x_safe = np.where(valid, x, 0.0).astype(np.float64, copy=False)
    s = x_safe.sum(axis=1)
    mean = np.divide(s, cnt, out=np.zeros_like(s), where=cnt > 0)
    af = np.divide(s, 2.0 * cnt, out=np.zeros_like(s), where=cnt > 0)
    maf = np.minimum(af, 1.0 - af).astype(np.float64, copy=False)

    x_imp = np.where(valid, x.astype(np.float64, copy=False), mean[:, None])
    xc = x_imp - mean[:, None]
    denom = max(1, int(n - 1))
    var = np.sum(xc * xc, axis=1) / float(denom)
    std = np.sqrt(np.maximum(var, 1e-12))
    z = (xc / std[:, None]).astype(np.float32, copy=False)
    return np.asarray(maf, dtype=np.float64), np.ascontiguousarray(z, dtype=np.float32)


def _window_end_by_bp(pos: np.ndarray, start_idx: int, win_bp: int) -> int:
    if pos.size <= 0:
        return 0
    s = int(start_idx)
    if s >= int(pos.size):
        return int(pos.size)
    p0 = int(pos[s])
    target = p0 + int(win_bp)
    if np.all(pos[1:] >= pos[:-1]):
        e = int(np.searchsorted(pos, target, side="right"))
        return max(s + 1, e)
    e2 = s + 1
    n = int(pos.size)
    while e2 < n and int(pos[e2]) <= target:
        e2 += 1
    return e2


def _ld_prune_with_maf_priority(
    geno: np.ndarray,
    sites: list[SiteInfo],
    spec: PruneSpec,
) -> np.ndarray:
    """
    Greedy LD prune (windowed) with MAF-priority conflict rule:
    1) Current site i compares with sites j in current window.
    2) If any correlated j has higher MAF than i, drop i.
    3) Otherwise drop correlated j with lower MAF than i.
    """
    x = np.asarray(geno, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Invalid genotype matrix shape: {x.shape}")
    m, n = x.shape
    if m == 0:
        return np.zeros((0,), dtype=bool)
    if len(sites) != int(m):
        raise ValueError(f"sites length mismatch: rows={m}, sites={len(sites)}")

    maf, z = _compute_maf_and_z(x)
    keep = np.ones((m,), dtype=bool)
    denom = float(max(1, int(n - 1)))
    eps = 1e-12
    by_chr: dict[str, list[int]] = {}
    for i, s in enumerate(sites):
        c = _normalize_chr_key(str(s.chrom))
        by_chr.setdefault(c, []).append(int(i))

    for _chrom, idx_list in by_chr.items():
        if len(idx_list) <= 1:
            continue
        idx_arr = np.asarray(idx_list, dtype=np.int64)
        pos = np.asarray([int(sites[int(i)].pos) for i in idx_arr], dtype=np.int64)
        dropped = np.zeros((idx_arr.size,), dtype=bool)
        l = int(idx_arr.size)
        start = 0
        while start < l:
            if spec.window_bp is not None:
                end = _window_end_by_bp(pos, start, int(spec.window_bp))
            else:
                end = min(l, start + int(spec.window_variants or 1))
            if end <= start:
                end = start + 1
            win = np.arange(start, end, dtype=np.int64)
            for li in win.tolist():
                if dropped[int(li)]:
                    continue
                right = win[win > int(li)]
                if right.size <= 0:
                    continue
                right = right[~dropped[right]]
                if right.size <= 0:
                    continue

                gi = int(idx_arr[int(li)])
                gj = idx_arr[right].astype(np.int64, copy=False)
                zi = np.asarray(z[gi, :], dtype=np.float32)
                zj = np.asarray(z[gj, :], dtype=np.float32)
                corr = (zj @ zi) / denom
                r2 = np.asarray(corr * corr, dtype=np.float64)
                hit = np.asarray(r2 >= float(spec.r2_threshold), dtype=bool)
                if not np.any(hit):
                    continue
                hit_local = right[hit]
                hit_global = idx_arr[hit_local].astype(np.int64, copy=False)
                maf_i = float(maf[gi])
                maf_j = np.asarray(maf[hit_global], dtype=np.float64)

                if np.any(maf_j > (maf_i + eps)):
                    dropped[int(li)] = True
                    continue

                drop_low = hit_local[maf_j < (maf_i - eps)]
                if drop_low.size > 0:
                    dropped[drop_low] = True

            start += int(spec.step_variants)

        keep[idx_arr[dropped]] = False
    return keep


def _iter_chunks_from_matrix(
    geno: np.ndarray,
    sites: list[SiteInfo],
    *,
    chunk_size: int = 50000,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    x = np.asarray(geno, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Invalid genotype matrix shape: {x.shape}")
    if int(x.shape[0]) != len(sites):
        raise ValueError(
            f"Genotype/sites length mismatch: rows={x.shape[0]}, sites={len(sites)}"
        )
    m = int(x.shape[0])
    step = max(1000, int(chunk_size))
    for s in range(0, m, step):
        e = min(m, s + step)
        yield np.asarray(x[s:e, :], dtype=np.float32), list(sites[s:e])


def _open_text_auto(path: str):
    low = str(path).lower()
    if low.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _read_id_sidecar(id_path: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    with open(id_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = _split_tokens(s)
            if not tok:
                continue
            sid = str(tok[0]).strip()
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)
    if len(out) == 0:
        raise ValueError(f"ID sidecar is empty or invalid: {id_path}")
    return out


def _count_text_matrix_rows(matrix_path: str) -> int:
    n = 0
    with _open_text_auto(matrix_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            n += 1
    return int(n)


def _resolve_txt_direct_paths(file_arg: str) -> tuple[str, str, str | None]:
    raw_prefix, _ = build_prefix_candidates(file_arg, text_suffixes=GENOTYPE_TEXT_SUFFIXES)
    p = str(file_arg)
    low = p.lower()

    matrix_path = None
    if low.endswith((".txt", ".tsv", ".csv")) and Path(p).is_file():
        matrix_path = p
    else:
        for ext in (".txt", ".tsv", ".csv"):
            cand = f"{raw_prefix}{ext}"
            if Path(cand).is_file():
                matrix_path = cand
                break
    if matrix_path is None:
        raise ValueError(
            f"Cannot find FILE text matrix for direct conversion: {file_arg}"
        )

    id_path = discover_id_sidecar_path(matrix_path, [raw_prefix])
    if id_path is None:
        raise ValueError(
            f"Cannot find ID sidecar for text matrix: {matrix_path} (expected {raw_prefix}.id or {matrix_path}.id)"
        )
    site_path = discover_site_path([raw_prefix])
    return matrix_path, id_path, site_path


def _iter_vcf_direct_chunks(
    vcf_path: str,
    *,
    chunk_size: int,
    sample_ids: list[str] | None,
    maf_threshold: float,
    max_missing_rate: float,
    model: str = "add",
    het_threshold: float = 0.02,
    snp_sites: list[tuple[str, int]] | None = None,
    bim_range: tuple[str, int, int] | None = None,
    chr_keys: set[str] | None = None,
    bp_min: int | None = None,
    bp_max: int | None = None,
    ranges: list[tuple[str, int, int]] | None = None,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    model_key = str(model).strip().lower()
    if model_key not in {"add", "dom", "rec", "het"}:
        raise ValueError("model must be one of: add, dom, rec, het")
    het_val = float(het_threshold)
    if not (0.0 <= het_val <= 1.0):
        raise ValueError("het_threshold must be within [0, 1.0]")

    chr_list = sorted(list(chr_keys)) if chr_keys else None
    try:
        reader = VcfChunkReader(
            str(vcf_path),
            float(maf_threshold),
            float(max_missing_rate),
            False,
            sample_ids,
            None,
            model_key,
            het_val,
            snp_sites=snp_sites,
            bim_range=bim_range,
            chr_keys=chr_list,
            bp_min=bp_min,
            bp_max=bp_max,
            ranges=ranges,
        )
    except TypeError:
        raise RuntimeError(
            "VcfChunkReader extension does not support full Rust-side filter arguments "
            "(model/het/snp_sites/bim_range/chr/bp/ranges). "
            "Please rebuild/reinstall janusx Rust extension."
        )
    while True:
        out = reader.next_chunk(int(chunk_size))
        if out is None:
            break
        geno, sites = out
        yield np.asarray(geno, dtype=np.float32), list(sites)


def _iter_hmp_direct_chunks(
    hmp_path: str,
    *,
    chunk_size: int,
    sample_ids: list[str] | None,
    maf_threshold: float,
    max_missing_rate: float,
    model: str = "add",
    het_threshold: float = 0.02,
    snp_sites: list[tuple[str, int]] | None = None,
    bim_range: tuple[str, int, int] | None = None,
    chr_keys: set[str] | None = None,
    bp_min: int | None = None,
    bp_max: int | None = None,
    ranges: list[tuple[str, int, int]] | None = None,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    model_key = str(model).strip().lower()
    if model_key not in {"add", "dom", "rec", "het"}:
        raise ValueError("model must be one of: add, dom, rec, het")
    het_val = float(het_threshold)
    if not (0.0 <= het_val <= 1.0):
        raise ValueError("het_threshold must be within [0, 1.0]")

    chr_list = sorted(list(chr_keys)) if chr_keys else None
    try:
        reader = HmpChunkReader(
            str(hmp_path),
            float(maf_threshold),
            float(max_missing_rate),
            False,
            sample_ids,
            None,
            model_key,
            het_val,
            snp_sites=snp_sites,
            bim_range=bim_range,
            chr_keys=chr_list,
            bp_min=bp_min,
            bp_max=bp_max,
            ranges=ranges,
        )
    except TypeError:
        raise RuntimeError(
            "HmpChunkReader extension does not support full Rust-side filter arguments "
            "(model/het/snp_sites/bim_range/chr/bp/ranges). "
            "Please rebuild/reinstall janusx Rust extension."
        )
    while True:
        out = reader.next_chunk(int(chunk_size))
        if out is None:
            break
        geno, sites = out
        yield np.asarray(geno, dtype=np.float32), list(sites)


def _iter_txt_direct_chunks(
    matrix_path: str,
    *,
    id_path: str,
    site_path: str | None,
    chunk_size: int,
    sample_ids: list[str] | None,
    maf_threshold: float,
    max_missing_rate: float,
    model: str = "add",
    het_threshold: float = 0.02,
    snp_sites: list[tuple[str, int]] | None = None,
    bim_range: tuple[str, int, int] | None = None,
    chr_keys: set[str] | None = None,
    bp_min: int | None = None,
    bp_max: int | None = None,
    ranges: list[tuple[str, int, int]] | None = None,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    model_key = str(model).strip().lower()
    if model_key not in {"add", "dom", "rec", "het"}:
        raise ValueError("model must be one of: add, dom, rec, het")
    het_val = float(het_threshold)
    if not (0.0 <= het_val <= 1.0):
        raise ValueError("het_threshold must be within [0, 1.0]")

    # Keep explicit sidecar resolution in caller for early validation,
    # but sink per-site filtering into the unified Rust kernel.
    _ = (id_path, site_path)
    chr_list = sorted(list(chr_keys)) if chr_keys else None
    try:
        reader = TxtChunkReader(
            str(matrix_path),
            None,
            None,
            None,
            bim_range,
            snp_sites,
            sample_ids,
            None,
            float(maf_threshold),
            float(max_missing_rate),
            False,
            model_key,
            het_val,
            chr_keys=chr_list,
            bp_min=bp_min,
            bp_max=bp_max,
            ranges=ranges,
        )
    except TypeError:
        raise RuntimeError(
            "TxtChunkReader extension does not support full Rust-side filter arguments "
            "(model/het/snp_sites/bim_range/chr/bp/ranges). "
            "Please rebuild/reinstall janusx Rust extension."
        )
    while True:
        out = reader.next_chunk(int(chunk_size))
        if out is None:
            break
        geno, sites = out
        yield np.asarray(geno, dtype=np.float32), list(sites)


def _count_selected_sites(
    gfile: str,
    *,
    sample_ids: list[str] | None,
    snp_sites: list[tuple[str, int]] | None,
    bim_range: tuple[str, int, int] | None,
    site_filter: SiteFilterSpec,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
) -> int:
    cnt = 0
    reader_model = "het" if float(het_threshold) > 0.0 else "add"
    reader_het = float(het_threshold) if float(het_threshold) > 0.0 else 0.02
    chunks = load_genotype_chunks(
        gfile,
        chunk_size=50_000,
        maf=float(maf_threshold),
        missing_rate=float(max_missing_rate),
        impute=False,
        model=reader_model,
        het=reader_het,
        sample_ids=sample_ids,
        snp_sites=snp_sites,
        bim_range=bim_range,
        chr_keys=(
            sorted(list(site_filter.chr_keys))
            if site_filter.chr_keys
            else None
        ),
        bp_min=(int(site_filter.bp_min) if site_filter.bp_min is not None else None),
        bp_max=(int(site_filter.bp_max) if site_filter.bp_max is not None else None),
        ranges=(
            [(str(c), int(s), int(e)) for c, s, e in site_filter.ranges]
            if site_filter.ranges
            else None
        ),
    )
    for block, _sites in chunks:
        cnt += int(np.asarray(block).shape[0])
    return int(cnt)


def build_parser() -> CliArgumentParser:
    parser = CliArgumentParser(
        prog="jx gformat",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx gformat -vcf geno.vcf.gz",
                "jx gformat -hmp geno.hmp.gz -fmt plink",
                "jx gformat -bfile geno_prefix -fmt txt -o outdir -prefix panel",
                "jx gformat -file geno_prefix -fmt vcf",
                "jx gformat -bfile geno_prefix --prune 500 50 0.2",
                "jx gformat -bfile geno_prefix --prune 500kb 10 0.2",
            ]
        ),
    )

    req = parser.add_argument_group("Input Arguments")
    src = req.add_mutually_exclusive_group(required=True)
    add_common_genotype_source_args(src, help_profile="gformat")

    opt = parser.add_argument_group("Optional Arguments")
    add_common_thread_arg(
        opt,
        default_threads=detect_effective_threads(),
        help_profile="packed_rust",
    )
    opt.add_argument("--threads", dest="thread", type=int, help=argparse.SUPPRESS)
    opt.add_argument(
        "-fmt",
        "--fmt",
        dest="format",
        choices=["plink", "vcf", "hmp", "txt", "npy", "gfd"],
        default="plink",
        help="Output genotype format: plink, vcf, hmp, txt, npy, gfd (default: plink).",
    )
    add_common_out_arg(opt, default=".", help_profile="converted_results")
    add_common_prefix_arg(opt, default=None, help_profile="input_basename")
    opt.add_argument(
        "-maf",
        "--maf",
        type=float,
        default=0.0,
        help=(
            "Exclude variants with minor allele frequency lower than a threshold "
            "(default: %(default)s; no filtering)."
        ),
    )
    opt.add_argument(
        "-geno",
        "--geno",
        type=float,
        default=1.0,
        help=(
            "Exclude variants with missing call frequencies greater than a threshold "
            "(default: %(default)s; no filtering)."
        ),
    )
    opt.add_argument(
        "-het",
        "--het",
        type=float,
        default=0.0,
        help=(
            "Filter variants by maximum heterozygosity rate. Drop variants with het rate "
            "greater than this threshold (default: %(default)s; no filtering)."
        ),
    )
    opt.add_argument(
        "-keep",
        "--keep",
        type=str,
        default=None,
        help="Keep only samples listed in file (one sample ID per line, no header).",
    )
    opt.add_argument(
        "-extract",
        "--extract",
        nargs="+",
        default=None,
        metavar=("MODE_OR_FILE", "FILE"),
        help=(
            "Keep only listed variants. Use '--extract <file>' for site list "
            "(CHR POS or CHR:POS/CHR_POS), or '--extract range <file>' "
            "for range list (CHR START END). No header."
        ),
    )
    opt.add_argument(
        "-snps-only",
        "--snps-only",
        action="store_true",
        default=False,
        help=(
            "Keep only simple A/C/G/T single-base SNP sites. "
            "Retained as a compatibility alias of --biallelic-only."
        ),
    )
    opt.add_argument(
        "-biallelic-only",
        "--biallelic-only",
        action="store_true",
        default=False,
        help=(
            "Keep only simple A/C/G/T single-base SNP sites. "
            "Despite the historical flag name, indels and other non-ATCG alleles are removed."
        ),
    )
    opt.add_argument(
        "-snp-name",
        "--snp-name",
        type=str,
        default=None,
        help=(
            "Override output SNP names with a CHR/POS template. "
            "Examples: 'chr_pos', 'chr*pos', '{chr}_{pos}', 'chr{chr}_{pos}'. "
            "Omit to inherit source names."
        ),
    )
    opt.add_argument(
        "-prune",
        "--prune",
        nargs=3,
        default=None,
        metavar=("WINDOW", "STEP", "R2"),
        help=(
            "LD prune with MAF-priority rule. Usage: "
            "--prune <window size[kb|bp]> <step size (variant ct)> <r^2 threshold>. "
            "Built-in backend uses strict sliding-window + greedy semantics. "
            "Set env JX_LD_PRUNE_BACKEND=plink to run external PLINK backend "
            "(PLINK input/output prune-only path). "
            "Bare numeric windows are variant counts; use kb/bp suffixes for physical windows. "
            "Examples: --prune 500 50 0.2, --prune 500kb 50 0.2, --prune 100bp 5 0.2."
        ),
    )
    opt.add_argument(
        "-chr",
        "--chr",
        dest="chr_filter",
        nargs="+",
        default=None,
        help=(
            "Keep only variants on selected chromosome(s). Supports spaces/commas "
            "and numeric ranges, e.g. '--chr 1-4,22,XY'."
        ),
    )
    bp_help = (
        "Physical position range filter. Must be used with a single chromosome in --chr. "
        "Both --from-bp and --to-bp are inclusive."
    )
    opt.add_argument("-from-bp", "--from-bp", type=int, default=None, help=bp_help)
    opt.add_argument("-to-bp", "--to-bp", type=int, default=None, help=bp_help)
    return parser


def main() -> None:
    parser = build_parser()
    args, extras = parser.parse_known_args()
    if extras:
        parser.error("unrecognized arguments: " + " ".join(extras))
    if not (0.0 <= float(args.maf) <= 0.5):
        parser.error("-maf must be within [0, 0.5].")
    if not (0.0 <= float(args.geno) <= 1.0):
        parser.error("-geno must be within [0, 1.0].")
    if not (0.0 <= float(args.het) <= 0.5):
        parser.error("-het must be within [0, 0.5].")
    detected_threads = int(detect_effective_threads())
    requested_threads = int(args.thread)
    if requested_threads <= 0:
        args.thread = int(detected_threads)
        thread_capped = False
    else:
        args.thread = int(min(requested_threads, detected_threads))
        thread_capped = bool(requested_threads > detected_threads)
    apply_blas_thread_env(int(args.thread))
    try:
        prune_spec = _parse_prune_args(args.prune)
    except Exception as e:
        parser.error(str(e))
    prune_backend_raw = str(os.getenv("JX_LD_PRUNE_BACKEND", "")).strip().lower()
    prune_backend = _resolve_ld_prune_backend()
    try:
        output_site_limit = _resolve_max_output_sites_env()
    except ValueError as e:
        parser.error(str(e))
    try:
        snp_name_template = _normalize_snp_name_template(args.snp_name)
    except ValueError as e:
        parser.error(str(e))
    prune_streaming_io = _env_truthy("JX_GFORMAT_PRUNE_STREAMING_IO", "1")
    print_prune_kernel_stats = _env_truthy("JX_GFORMAT_PRINT_PRUNE_KERNEL_STATS", "0")

    gfile, default_prefix = determine_genotype_source(args)
    if args.prefix is None:
        args.prefix = default_prefix
    out_fmt, out_prefix, out_path = _resolve_output_target(args)
    output_display = format_output_display(out_fmt, out_prefix, out_path)

    os.makedirs(str(args.out), exist_ok=True, mode=0o755)
    configure_genotype_cache_from_out(str(args.out))
    log_path = f"{out_prefix}.gformat.log"
    logger = setup_logging(log_path)
    if thread_capped:
        logger.warning(
            f"Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(args.thread)}."
        )
    status_enabled = stdout_is_tty()
    if prune_backend_raw not in {"", "auto", "default", "rust", "plink", "plink-external", "external-plink"}:
        logger.warning(
            "Unknown JX_LD_PRUNE_BACKEND=%r; fallback to Rust backend.",
            prune_backend_raw,
        )

    extract_mode, extract_file = _parse_extract_arg(args.extract)
    chr_keys = _expand_chr_selector(args.chr_filter)
    if args.from_bp is not None and args.to_bp is not None and int(args.from_bp) > int(args.to_bp):
        raise ValueError("--from-bp cannot be greater than --to-bp.")
    if (args.from_bp is not None or args.to_bp is not None) and len(chr_keys) != 1:
        raise ValueError("--from-bp/--to-bp requires a single chromosome in --chr.")

    emit_cli_configuration(
        logger,
        app_title="JanusX - gformat",
        config_title="GFORMAT CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "General",
                [
                    ("Genotype file", gfile),
                    ("Output format", out_fmt),
                    ("Keep samples", str(args.keep or "None")),
                    ("Extract", f"{extract_mode or 'None'}:{extract_file or 'None'}"),
                    ("Prune", (prune_spec.label() if prune_spec is not None else "None")),
                    (
                        "Output site limit",
                        (
                            str(output_site_limit)
                            if output_site_limit is not None
                            else "None"
                        ),
                    ),
                    (
                        "Prune backend",
                        (
                            ("PLINK external (env)")
                            if (prune_spec is not None and prune_backend == "plink")
                            else "Rust (default)"
                        ),
                    ),
                    (
                        "Prune PLINK I/O",
                        (
                            "streaming-low-memory"
                            if prune_streaming_io
                            else "packed-speed"
                        ),
                    ),
                    (
                        "Threads",
                        format_requested_thread_usage(
                            requested_threads=int(requested_threads),
                            using_threads=int(args.thread),
                            detected_threads=int(detected_threads),
                        ),
                    ),
                    ("Chr filter", ",".join(sorted(chr_keys)) if chr_keys else "None"),
                    ("BP range", f"{args.from_bp if args.from_bp is not None else '-'}..{args.to_bp if args.to_bp is not None else '-'}"),
                    ("SNP name template", snp_name_template or "inherit"),
                    ("ATCG SNP only (--snps-only)", bool(args.snps_only)),
                    ("ATCG SNP only alias (--biallelic-only)", bool(args.biallelic_only)),
                    ("MAF threshold", float(args.maf)),
                    ("Miss threshold", float(args.geno)),
                    ("Het threshold", float(args.het)),
                ],
            )
        ],
        footer_rows=[
            ("Output", output_display),
        ],
        line_max_chars=60,
    )

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    if args.keep:
        checks.append(ensure_file_exists(logger, str(args.keep), "--keep file"))
    if extract_file:
        checks.append(ensure_file_exists(logger, str(extract_file), "--extract file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    if args.file and out_fmt in {"vcf", "hmp", "plink"} and (not _has_real_file_sites(gfile)):
        raise ValueError(
            f"{out_fmt.upper()} output from -file input requires real site metadata. "
            "Please provide a matching prefix.bsite/prefix.site or prefix.bim sidecar."
        )
    if snp_name_template is not None and out_fmt in {"txt", "npy", "gfd"}:
        logger.warning(
            "--snp-name only affects outputs carrying SNP IDs (PLINK/VCF/HMP); "
            "it has no effect for %s output.",
            out_fmt.upper(),
        )

    source_kind, _source_prefix, source_path = _resolve_input(gfile)
    use_direct_no_cache = bool(source_kind in {"vcf", "hmp", "txt"})
    txt_matrix_path: str | None = None
    txt_id_path: str | None = None
    txt_site_path: str | None = None

    extract_sites: list[tuple[str, int]] | None = None
    extract_ranges: list[tuple[str, int, int]] | None = None
    if extract_mode == "sites":
        extract_sites = _read_extract_sites(str(extract_file))
    elif extract_mode == "range":
        extract_ranges = _read_extract_ranges(str(extract_file))

    # Reader-side pushdown filters when possible.
    push_snp_sites: list[tuple[str, int]] | None = None
    push_bim_range: tuple[str, int, int] | None = None
    post_filter = SiteFilterSpec()

    if chr_keys:
        post_filter.chr_keys = set(chr_keys)
    if args.from_bp is not None:
        post_filter.bp_min = int(args.from_bp)
    if args.to_bp is not None:
        post_filter.bp_max = int(args.to_bp)

    if extract_sites:
        push_snp_sites = extract_sites
    if extract_ranges:
        if len(extract_ranges) == 1 and not post_filter.active():
            push_bim_range = extract_ranges[0]
        else:
            post_filter.ranges = extract_ranges

    # Optimize simple --chr + --from-bp/--to-bp into a direct bim_range pushdown.
    if (
        push_snp_sites is None
        and push_bim_range is None
        and extract_ranges is None
        and post_filter.chr_keys
        and len(post_filter.chr_keys) == 1
        and (post_filter.bp_min is not None or post_filter.bp_max is not None)
    ):
        c = next(iter(post_filter.chr_keys))
        start = int(post_filter.bp_min) if post_filter.bp_min is not None else 0
        end = int(post_filter.bp_max) if post_filter.bp_max is not None else 2_147_483_647
        push_bim_range = (c, start, end)
        post_filter = SiteFilterSpec()

    reader_model = "het" if float(args.het) > 0.0 else "add"
    reader_het = float(args.het) if float(args.het) > 0.0 else 0.02
    site_type_snps_only = _simple_snp_only_requested(
        snps_only=bool(args.snps_only),
        biallelic_only=bool(args.biallelic_only),
    )

    # Direct Rust packed path for PLINK -> PLINK conversion with all non-prune filters.
    # This keeps Python-side object construction minimal and sinks filtering/writing into Rust.
    if (
        prune_spec is None
        and source_kind == "plink"
        and out_fmt == "plink"
        and (not bool(args.snps_only))
        and (not bool(args.biallelic_only))
        and _bed_filter_to_plink_rust is not None
    ):
        direct_keep_sample_ids: list[str] | None = None
        if args.keep:
            direct_keep_sample_ids = _select_keep_samples_from_fam(
                str(gfile),
                str(args.keep),
            )

        filter_desc = "Applying filters/slices..."
        if not status_enabled:
            _log_status_file_only(logger, filter_desc)
        call_args = dict(
            src_prefix=str(gfile),
            out_prefix=str(out_prefix),
            maf_threshold=float(args.maf),
            max_missing_rate=float(args.geno),
            fill_missing=False,
            model=reader_model,
            het_threshold=reader_het,
            sample_ids=direct_keep_sample_ids,
            snp_sites=push_snp_sites,
            bim_range=push_bim_range,
            chr_keys=(
                sorted(list(post_filter.chr_keys))
                if post_filter.chr_keys
                else None
            ),
            bp_min=(int(post_filter.bp_min) if post_filter.bp_min is not None else None),
            bp_max=(int(post_filter.bp_max) if post_filter.bp_max is not None else None),
            ranges=post_filter.ranges,
            max_output_sites=output_site_limit,
        )
        if status_enabled:
            pbar = ProgressAdapter(
                total=1,
                desc="Applying filters/slices",
                force_animate=True,
                keep_display=False,
                show_remaining=True,
                emit_done=False,
            )
            direct_done = 0
            direct_total = 1

            def _on_direct_filter_progress(done: int, total: int) -> None:
                nonlocal direct_done, direct_total
                try:
                    d = int(done)
                    t = int(total)
                except Exception:
                    return
                if t > 0 and t != direct_total and direct_done == 0:
                    direct_total = t
                    pbar.set_total(int(max(1, direct_total)))
                d = int(max(0, d))
                d = int(min(d, max(1, direct_total)))
                if d > direct_done:
                    pbar.update(d - direct_done)
                    direct_done = d

            try:
                t_direct = time.time()
                try:
                    keep_n, scanned_n, out_n = _bed_filter_to_plink_rust(
                        **call_args,
                        progress_callback=_on_direct_filter_progress,
                        progress_every=0,
                    )
                except TypeError as ex:
                    emsg = str(ex).lower()
                    if ("keyword" not in emsg) and ("argument" not in emsg) and ("positional" not in emsg):
                        raise
                    legacy_call_args = dict(call_args)
                    legacy_call_args.pop("max_output_sites", None)
                    keep_n, scanned_n, out_n = _bed_filter_to_plink_rust(**legacy_call_args)
                scanned_n = int(scanned_n)
                if scanned_n > 0 and direct_done < scanned_n:
                    pbar.set_total(scanned_n)
                    pbar.update(scanned_n - direct_done)
                    direct_done = scanned_n
                if scanned_n > 0:
                    pbar.finish()
            finally:
                pbar.close()
        else:
            t_direct = time.time()
            try:
                keep_n, scanned_n, out_n = _bed_filter_to_plink_rust(**call_args)
            except TypeError as ex:
                emsg = str(ex).lower()
                if ("keyword" not in emsg) and ("argument" not in emsg) and ("positional" not in emsg):
                    raise
                legacy_call_args = dict(call_args)
                legacy_call_args.pop("max_output_sites", None)
                keep_n, scanned_n, out_n = _bed_filter_to_plink_rust(**legacy_call_args)
        rust_filter_direct_wall = float(time.time() - t_direct)
        selected_n_sites = int(keep_n)
        dropped_n = int(max(0, int(scanned_n) - int(keep_n)))
        if not status_enabled:
            _log_status_file_only(
                logger,
                "Applying filters/slices ...Finished "
                f"(backend=rust-packed-io, kept={int(keep_n)}, dropped={dropped_n})",
            )
        if selected_n_sites <= 0:
            raise ValueError(
                "All variants were filtered out by filters (-extract/-chr/-from-bp/-to-bp/-maf/-geno/-het)."
            )
        _rewrite_plink_bim_snp_names(str(out_prefix), snp_name_template)
        print(f"Genotype source: {format_path_for_display(gfile)}")
        print(f"Samples: {int(out_n)}, sites: {selected_n_sites}")
        log_success(logger, f"Format conversion completed in {rust_filter_direct_wall:.2f} s")
        log_success(logger, f"Output written: {output_display}")
        return

    inspect_desc = "Inspecting genotype metadata..."
    if not status_enabled:
        _log_status_file_only(logger, inspect_desc)
    with CliStatus(
        inspect_desc,
        enabled=status_enabled,
    ) as task:
        try:
            if use_direct_no_cache and source_kind == "vcf":
                if source_path is None:
                    raise ValueError(f"VCF source path not found: {gfile}")
                vcf_reader = VcfChunkReader(
                    str(source_path),
                    0.0,
                    1.0,
                    False,
                    None,
                    None,
                    "add",
                    0.02,
                )
                sample_ids = [str(x) for x in vcf_reader.sample_ids]
                n_sites = int(count_vcf_snps(str(source_path)))
            elif use_direct_no_cache and source_kind == "hmp":
                if source_path is None:
                    raise ValueError(f"HMP source path not found: {gfile}")
                hmp_reader = HmpChunkReader(
                    str(source_path),
                    0.0,
                    1.0,
                    False,
                    None,
                    None,
                    "add",
                    0.02,
                )
                sample_ids = [str(x) for x in hmp_reader.sample_ids]
                n_sites = int(count_hmp_snps(str(source_path)))
            elif use_direct_no_cache and source_kind == "txt":
                txt_arg = str(args.file or gfile)
                txt_matrix_path, txt_id_path, txt_site_path = _resolve_txt_direct_paths(txt_arg)
                # Keep explicit direct TXT paths for streaming later, but reuse
                # the same reader stack used by conversion to avoid inspect/write
                # count divergence on non-dosage numeric matrices.
                _ = (txt_matrix_path, txt_id_path, txt_site_path)
                sample_ids, n_sites = inspect_genotype_file(gfile)
            elif source_kind == "plink":
                sample_ids, n_sites = _inspect_plink_prefix_light(str(gfile))
            else:
                sample_ids, n_sites = inspect_genotype_file(gfile)
        except Exception:
            task.fail("Inspecting genotype metadata ...Failed")
            raise
        task.complete(
            f"Inspecting genotype metadata ...Finished (n={len(sample_ids)}, nSNP={int(n_sites)})"
        )
    if not status_enabled:
        _log_status_file_only(
            logger,
            f"Inspecting genotype metadata ...Finished (n={len(sample_ids)}, nSNP={int(n_sites)})",
        )
    keep_sample_ids: list[str] | None = None
    if args.keep:
        keep_list = _read_keep_samples(str(args.keep))
        keep_set = set(keep_list)
        keep_sample_ids = [str(s) for s in sample_ids if str(s) in keep_set]
        if len(keep_sample_ids) == 0:
            miss_preview = ", ".join(keep_list[:10])
            raise ValueError(f"--keep selected 0 samples. Missing examples: {miss_preview}")

    use_rust_convert_direct = bool(
        _rust_convert_fastpath_enabled()
        and prune_spec is None
        and source_kind in {"plink", "vcf", "hmp"}
        and out_fmt in {"plink", "vcf", "hmp"}
        and push_snp_sites is None
        and push_bim_range is None
        and (not post_filter.active())
        and output_site_limit is None
    )
    if use_rust_convert_direct:
        direct_input = str(gfile if source_kind == "plink" else (source_path or gfile))
        direct_output = str(out_prefix if out_fmt == "plink" else out_path)
        out_n = int(len(keep_sample_ids) if keep_sample_ids is not None else len(sample_ids))
        progress_every = max(1000, int(max(1, int(n_sites)) // 200))
        convert_stats = None
        backend = "rust-direct-io"

        if status_enabled:
            pbar = ProgressAdapter(
                total=max(1, int(n_sites)),
                desc="Converting formats",
                force_animate=True,
                keep_display=False,
                show_remaining=True,
                emit_done=False,
            )
            convert_done = 0
            convert_total = max(1, int(n_sites))

            def _on_convert_progress(done: int, total: int) -> None:
                nonlocal convert_done, convert_total
                try:
                    d = int(done)
                    t = int(total)
                except Exception:
                    return
                if t > 0 and t != convert_total and convert_done == 0:
                    convert_total = t
                    pbar.set_total(int(max(1, convert_total)))
                d = int(max(0, d))
                d = int(min(d, max(1, convert_total)))
                if d > convert_done:
                    pbar.update(d - convert_done)
                    convert_done = d

            try:
                t_direct = time.time()
                convert_stats = convert_genotypes(
                    direct_input,
                    direct_output,
                    out_fmt=str(out_fmt),
                    progress_callback=_on_convert_progress,
                    progress_every=int(progress_every),
                    threads=int(args.thread),
                    snps_only=bool(site_type_snps_only),
                    maf=float(args.maf),
                    geno=float(args.geno),
                    impute=False,
                    model=reader_model,
                    het=reader_het,
                    sample_ids=keep_sample_ids,
                    snp_name_template=snp_name_template,
                )
                seen_n = int(getattr(convert_stats, "n_sites_seen", 0))
                if seen_n > 0 and convert_done < seen_n:
                    pbar.set_total(int(max(1, max(convert_total, seen_n))))
                    pbar.update(seen_n - convert_done)
                    convert_done = seen_n
                if seen_n > 0:
                    pbar.finish()
            finally:
                pbar.close()
        else:
            t_direct = time.time()
            convert_stats = convert_genotypes(
                direct_input,
                direct_output,
                out_fmt=str(out_fmt),
                threads=int(args.thread),
                snps_only=bool(site_type_snps_only),
                maf=float(args.maf),
                geno=float(args.geno),
                impute=False,
                model=reader_model,
                het=reader_het,
                sample_ids=keep_sample_ids,
                snp_name_template=snp_name_template,
            )

        kept_n = int(getattr(convert_stats, "n_sites_written", 0))
        seen_n = int(getattr(convert_stats, "n_sites_seen", 0))
        dropped_n = int(max(0, seen_n - kept_n))
        if not status_enabled:
            _log_status_file_only(
                logger,
                "Converting formats ...Finished "
                f"(backend={backend}, kept={kept_n}, dropped={dropped_n})",
            )
        print(f"Genotype source: {format_path_for_display(gfile)}")
        print(f"Samples: {out_n}, sites: {kept_n}")
        log_success(logger, f"Format conversion completed in {time.time() - t_direct:.2f} s")
        log_success(logger, f"Output written: {output_display}")
        return

    def _make_chunks() -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
        if use_direct_no_cache and source_kind == "vcf":
            if source_path is None:
                raise ValueError(f"VCF source path not found: {gfile}")
            c = _iter_vcf_direct_chunks(
                str(source_path),
                chunk_size=50_000,
                sample_ids=keep_sample_ids,
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
                model=reader_model,
                het_threshold=reader_het,
                snp_sites=push_snp_sites,
                bim_range=push_bim_range,
                chr_keys=post_filter.chr_keys,
                bp_min=post_filter.bp_min,
                bp_max=post_filter.bp_max,
                ranges=post_filter.ranges,
            )
        elif use_direct_no_cache and source_kind == "hmp":
            if source_path is None:
                raise ValueError(f"HMP source path not found: {gfile}")
            c = _iter_hmp_direct_chunks(
                str(source_path),
                chunk_size=50_000,
                sample_ids=keep_sample_ids,
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
                model=reader_model,
                het_threshold=reader_het,
                snp_sites=push_snp_sites,
                bim_range=push_bim_range,
                chr_keys=post_filter.chr_keys,
                bp_min=post_filter.bp_min,
                bp_max=post_filter.bp_max,
                ranges=post_filter.ranges,
            )
        elif use_direct_no_cache and source_kind == "txt":
            if txt_matrix_path is None or txt_id_path is None:
                raise ValueError("Internal error: text input paths are unresolved.")
            c = _iter_txt_direct_chunks(
                str(txt_matrix_path),
                id_path=str(txt_id_path),
                site_path=txt_site_path,
                chunk_size=50_000,
                sample_ids=keep_sample_ids,
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
                model=reader_model,
                het_threshold=reader_het,
                snp_sites=push_snp_sites,
                bim_range=push_bim_range,
                chr_keys=post_filter.chr_keys,
                bp_min=post_filter.bp_min,
                bp_max=post_filter.bp_max,
                ranges=post_filter.ranges,
            )
        else:
            c = load_genotype_chunks(
                gfile,
                chunk_size=50_000,
                maf=float(args.maf),
                missing_rate=float(args.geno),
                impute=False,
                model=reader_model,
                het=reader_het,
                snps_only=bool(site_type_snps_only),
                sample_ids=keep_sample_ids,
                snp_sites=push_snp_sites,
                bim_range=push_bim_range,
                chr_keys=(
                    sorted(list(post_filter.chr_keys))
                    if post_filter.chr_keys
                    else None
                ),
                bp_min=(int(post_filter.bp_min) if post_filter.bp_min is not None else None),
                bp_max=(int(post_filter.bp_max) if post_filter.bp_max is not None else None),
                ranges=(
                    [(str(c), int(s), int(e)) for c, s, e in post_filter.ranges]
                    if post_filter.ranges
                    else None
                ),
            )
        for geno, sites in c:
            filtered_geno, filtered_sites = _apply_site_type_filters(
                geno,
                sites,
                snps_only=bool(args.snps_only),
                biallelic_only=bool(args.biallelic_only),
            )
            if len(filtered_sites) == 0:
                continue
            yield filtered_geno, filtered_sites

    need_selected_count = (
        bool(push_snp_sites)
        or (push_bim_range is not None)
        or post_filter.active()
        or bool(args.snps_only)
        or bool(args.biallelic_only)
        or (float(args.maf) > 0.0)
        or (float(args.geno) < 1.0)
        or (float(args.het) > 0.0)
        or (use_direct_no_cache and out_fmt in {"npy", "txt"})
        or (prune_spec is not None)
    )
    selected_n_sites: int | None = int(n_sites)
    pruned_geno: np.ndarray | None = None
    pruned_sites: list[SiteInfo] | None = None
    packed_prune_keep_mask: np.ndarray | None = None
    packed_prune_packed: np.ndarray | None = None
    packed_prune_source_prefix: str | None = None
    rust_prune_direct_done = False
    rust_prune_direct_wall = 0.0
    prune_direct_backend: str | None = None

    if prune_spec is not None:
        prune_desc = "Applying LD prune..."
        prune_pre_sites = 0
        has_prune_prefilters = bool(
            keep_sample_ids is not None
            or push_snp_sites is not None
            or push_bim_range is not None
            or post_filter.active()
            or bool(site_type_snps_only)
            or float(args.maf) > 0.0
            or float(args.geno) < 1.0
            or float(args.het) > 0.0
        )
        use_external_plink_prune = bool(
            prune_backend == "plink"
            and source_kind == "plink"
            and out_fmt == "plink"
            and keep_sample_ids is None
            and push_snp_sites is None
            and push_bim_range is None
            and (not post_filter.active())
            and (not bool(site_type_snps_only))
            and float(args.maf) <= 0.0
            and float(args.geno) >= 1.0
            and float(args.het) <= 0.0
        )
        use_packed_prune_fastpath = bool(
            source_kind == "plink"
            and _bed_packed_ld_prune_maf_priority is not None
            and keep_sample_ids is None
            and push_snp_sites is None
            and push_bim_range is None
            and (not post_filter.active())
            and (not bool(site_type_snps_only))
            and float(args.maf) <= 0.0
            and float(args.geno) >= 1.0
            and float(args.het) <= 0.0
        )
        use_masked_stream_prune_direct = bool(
            source_kind == "plink"
            and out_fmt == "plink"
            and keep_sample_ids is None
            and has_prune_prefilters
            and _bed_prune_mask_to_plink_rust is not None
            and prune_streaming_io
        )
        if (
            use_packed_prune_fastpath
            and (not use_external_plink_prune)
            and _packed_prune_kernel_stats is not None
        ):
            try:
                _packed_prune_kernel_stats(reset=True)
            except Exception:
                pass
        use_rust_prune_direct = bool(
            use_packed_prune_fastpath
            and out_fmt == "plink"
            and keep_sample_ids is None
            and _bed_prune_to_plink_rust is not None
            and prune_streaming_io
        )
        use_rust_prune_filtered_direct = bool(
            source_kind == "plink"
            and out_fmt == "plink"
            and has_prune_prefilters
            and (not bool(site_type_snps_only))
            and (
                keep_sample_ids is not None
                or (not use_masked_stream_prune_direct)
            )
            and _bed_filter_to_plink_rust is not None
            and _bed_prune_to_plink_rust is not None
        )
        if prune_backend == "plink" and (not use_external_plink_prune):
            logger.warning(
                "JX_LD_PRUNE_BACKEND=plink requires PLINK input/output prune without "
                "extra filters; falling back to built-in backend."
            )
        if not status_enabled:
            _log_status_file_only(logger, prune_desc)
        if use_external_plink_prune:
            with CliStatus(prune_desc, enabled=status_enabled) as task:
                try:
                    keep_n, total_n, wall = _ld_prune_with_external_plink(
                        src_prefix=str(gfile),
                        out_prefix=str(out_prefix),
                        spec=prune_spec,
                        threads=int(args.thread),
                        logger=logger,
                    )
                    rust_prune_direct_wall = float(wall)
                    prune_pre_sites = int(total_n)
                    selected_n_sites = int(keep_n)
                    rust_prune_direct_done = True
                    prune_direct_backend = "plink-external"
                    if keep_n <= 0:
                        raise ValueError(
                            "All variants were filtered out by LD prune; "
                            "try a looser --prune r^2 threshold or larger window."
                        )
                    task.complete(
                        "Applying LD prune ...Finished "
                        f"(backend={prune_direct_backend}, kept={keep_n}, dropped={int(prune_pre_sites - keep_n)})"
                    )
                except Exception:
                    task.fail("Applying LD prune ...Failed")
                    raise
            if not status_enabled:
                _log_status_file_only(
                    logger,
                    "Applying LD prune ...Finished "
                    f"(backend={prune_direct_backend}, kept={selected_n_sites}, dropped={int(prune_pre_sites - selected_n_sites)})",
                )
        elif use_rust_prune_direct and status_enabled:
            chrom_progress = _ChromPruneProgress(max_visible=min(5, int(args.thread)))
            prune_success_msg: str | None = None

            def _on_chrom_prune_progress(
                block_idx: int,
                chrom: str,
                done: int,
                total: int,
                finished: bool,
            ) -> None:
                chrom_progress.update(block_idx, chrom, done, total, finished)

            try:
                t_direct = time.time()
                call_args = dict(
                    src_prefix=str(gfile),
                    out_prefix=str(out_prefix),
                    window_bp=(
                        int(prune_spec.window_bp)
                        if prune_spec.window_bp is not None
                        else None
                    ),
                    window_variants=(
                        int(prune_spec.window_variants)
                        if prune_spec.window_variants is not None
                        else None
                    ),
                    step_variants=int(prune_spec.step_variants),
                    r2_threshold=float(prune_spec.r2_threshold),
                    threads=int(args.thread),
                    max_output_sites=output_site_limit,
                )
                try:
                    keep_n, total_n = _bed_prune_to_plink_rust(
                        **call_args,
                        chrom_progress_callback=_on_chrom_prune_progress,
                        progress_every=0,
                    )
                except TypeError as ex:
                    emsg = str(ex).lower()
                    if ("keyword" not in emsg) and ("argument" not in emsg) and ("positional" not in emsg):
                        raise
                    # Backward-compatible fallback for older extension without chromosome callback args.
                    legacy_call_args = dict(call_args)
                    legacy_call_args.pop("max_output_sites", None)
                    keep_n, total_n = _bed_prune_to_plink_rust(**legacy_call_args)
                total_n = int(total_n)
                keep_n = int(keep_n)
                rust_prune_direct_wall = float(time.time() - t_direct)
                prune_pre_sites = int(total_n)
                selected_n_sites = int(keep_n)
                rust_prune_direct_done = True
                prune_direct_backend = "rust-stream-io"
                if keep_n <= 0:
                    raise ValueError(
                        "All variants were filtered out by LD prune; "
                        "try a looser --prune r^2 threshold or larger window."
                    )
                prune_success_msg = (
                    "Applying LD prune ...Finished "
                    f"(backend={prune_direct_backend}, kept={keep_n}, dropped={int(prune_pre_sites - keep_n)})"
                )
            finally:
                chrom_progress.close()
            if prune_success_msg is not None:
                log_success(logger, prune_success_msg)
        elif use_rust_prune_filtered_direct and status_enabled:
            pbar = ProgressAdapter(
                total=max(1, int(n_sites)),
                desc="Applying LD prune",
                force_animate=True,
                keep_display=False,
                show_remaining=True,
                emit_done=False,
            )
            filter_done = 0
            filter_total = max(1, int(n_sites))
            prune_done = 0
            prune_total = 0
            phase_offset = 0
            prune_success_msg: str | None = None

            def _on_filter_progress(done: int, total: int) -> None:
                nonlocal filter_done, filter_total
                try:
                    d = int(done)
                    t = int(total)
                except Exception:
                    return
                t = int(max(1, t))
                if filter_done == 0 and t != filter_total:
                    filter_total = t
                    pbar.set_total(filter_total)
                d = int(max(0, min(d, filter_total)))
                if d > filter_done:
                    pbar.update(d - filter_done)
                    filter_done = d

            def _on_prune_progress(done: int, total: int) -> None:
                nonlocal filter_done, prune_done, prune_total, phase_offset
                try:
                    d = int(done)
                    t = int(total)
                except Exception:
                    return
                if phase_offset <= 0:
                    if filter_done < filter_total:
                        pbar.update(filter_total - filter_done)
                        filter_done = filter_total
                    phase_offset = int(max(filter_done, filter_total))
                prune_total = int(max(1, t))
                pbar.set_total(int(max(1, phase_offset + prune_total)))
                d = int(max(0, min(d, prune_total)))
                combined_done = phase_offset + d
                current_done = phase_offset + prune_done
                if combined_done > current_done:
                    pbar.update(combined_done - current_done)
                    prune_done = d

            try:
                keep_n, total_n, pre_n, wall, out_n = _ld_prune_with_rust_filtered_pipeline(
                    src_prefix=str(gfile),
                    out_prefix=str(out_prefix),
                    spec=prune_spec,
                    threads=int(args.thread),
                    maf_threshold=float(args.maf),
                    max_missing_rate=float(args.geno),
                    model=reader_model,
                    het_threshold=reader_het,
                    sample_ids=keep_sample_ids,
                    snp_sites=push_snp_sites,
                    bim_range=push_bim_range,
                    chr_keys=(
                        sorted(list(post_filter.chr_keys))
                        if post_filter.chr_keys
                        else None
                    ),
                    bp_min=(int(post_filter.bp_min) if post_filter.bp_min is not None else None),
                    bp_max=(int(post_filter.bp_max) if post_filter.bp_max is not None else None),
                    ranges=post_filter.ranges,
                    logger=logger,
                    filter_progress_callback=_on_filter_progress,
                    filter_progress_every=0,
                    prune_progress_callback=_on_prune_progress,
                    prune_progress_every=max(1000, int(max(1, int(n_sites) // 200))),
                    max_output_sites=output_site_limit,
                )
                rust_prune_direct_wall = float(wall)
                prune_pre_sites = int(pre_n)
                selected_n_sites = int(keep_n)
                rust_prune_direct_done = True
                prune_direct_backend = "rust-filter+stream-io"
                if keep_sample_ids is not None and int(out_n) != int(len(keep_sample_ids)):
                    logger.warning(
                        "Rust filtered-prune sample count mismatch: "
                        f"writer={int(out_n)}, expected={int(len(keep_sample_ids))}"
                    )
                if filter_done < filter_total:
                    pbar.update(filter_total - filter_done)
                    filter_done = filter_total
                phase_offset = int(max(filter_done, filter_total))
                prune_total = int(max(1, int(total_n)))
                pbar.set_total(int(max(1, phase_offset + prune_total)))
                combined_done = phase_offset + int(total_n)
                current_done = phase_offset + prune_done
                if combined_done > current_done:
                    pbar.update(combined_done - current_done)
                    prune_done = int(total_n)
                pbar.finish()
                if keep_n <= 0:
                    raise ValueError(
                        "All variants were filtered out by LD prune; "
                        "try a looser --prune r^2 threshold or larger window."
                    )
                prune_success_msg = (
                    "Applying LD prune ...Finished "
                    f"(backend={prune_direct_backend}, kept={keep_n}, dropped={int(prune_pre_sites - keep_n)})"
                )
            finally:
                pbar.close()
            if prune_success_msg is not None:
                log_success(logger, prune_success_msg)
        else:
            with CliStatus(prune_desc, enabled=status_enabled) as task:
                try:
                    if use_masked_stream_prune_direct:
                        keep_n, pre_n, wall = _ld_prune_with_mask_first_streaming_pipeline(
                            src_prefix=str(gfile),
                            out_prefix=str(out_prefix),
                            spec=prune_spec,
                            threads=int(args.thread),
                            maf_threshold=float(args.maf),
                            max_missing_rate=float(args.geno),
                            het_threshold=float(args.het),
                            snps_only=bool(args.snps_only),
                            biallelic_only=bool(args.biallelic_only),
                            snp_sites=push_snp_sites,
                            bim_range=push_bim_range,
                            chr_keys=(
                                sorted(list(post_filter.chr_keys))
                                if post_filter.chr_keys
                                else None
                            ),
                            bp_min=(int(post_filter.bp_min) if post_filter.bp_min is not None else None),
                            bp_max=(int(post_filter.bp_max) if post_filter.bp_max is not None else None),
                            ranges=post_filter.ranges,
                            max_output_sites=output_site_limit,
                        )
                        rust_prune_direct_wall = float(wall)
                        prune_pre_sites = int(pre_n)
                        selected_n_sites = int(keep_n)
                        rust_prune_direct_done = True
                        prune_direct_backend = "rust-mask+stream-io"
                    elif use_rust_prune_filtered_direct:
                        keep_n, total_n, pre_n, wall, out_n = _ld_prune_with_rust_filtered_pipeline(
                            src_prefix=str(gfile),
                            out_prefix=str(out_prefix),
                            spec=prune_spec,
                            threads=int(args.thread),
                            maf_threshold=float(args.maf),
                            max_missing_rate=float(args.geno),
                            model=reader_model,
                            het_threshold=reader_het,
                            sample_ids=keep_sample_ids,
                            snp_sites=push_snp_sites,
                            bim_range=push_bim_range,
                            chr_keys=(
                                sorted(list(post_filter.chr_keys))
                                if post_filter.chr_keys
                                else None
                            ),
                            bp_min=(int(post_filter.bp_min) if post_filter.bp_min is not None else None),
                            bp_max=(int(post_filter.bp_max) if post_filter.bp_max is not None else None),
                            ranges=post_filter.ranges,
                            logger=logger,
                            max_output_sites=output_site_limit,
                        )
                        rust_prune_direct_wall = float(wall)
                        prune_pre_sites = int(pre_n)
                        selected_n_sites = int(keep_n)
                        rust_prune_direct_done = True
                        prune_direct_backend = "rust-filter+temp-stream-io"
                        if keep_sample_ids is not None and int(out_n) != int(len(keep_sample_ids)):
                            logger.warning(
                                "Rust filtered-prune sample count mismatch: "
                                f"writer={int(out_n)}, expected={int(len(keep_sample_ids))}"
                            )
                    elif use_rust_prune_direct:
                        t_direct = time.time()
                        call_args = dict(
                            src_prefix=str(gfile),
                            out_prefix=str(out_prefix),
                            window_bp=(
                                int(prune_spec.window_bp)
                                if prune_spec.window_bp is not None
                                else None
                            ),
                            window_variants=(
                                int(prune_spec.window_variants)
                                if prune_spec.window_variants is not None
                                else None
                            ),
                            step_variants=int(prune_spec.step_variants),
                            r2_threshold=float(prune_spec.r2_threshold),
                            threads=int(args.thread),
                            max_output_sites=output_site_limit,
                        )
                        try:
                            keep_n, total_n = _bed_prune_to_plink_rust(**call_args)
                        except TypeError as ex:
                            emsg = str(ex).lower()
                            if ("keyword" not in emsg) and ("argument" not in emsg) and ("positional" not in emsg):
                                raise
                            legacy_call_args = dict(call_args)
                            legacy_call_args.pop("max_output_sites", None)
                            keep_n, total_n = _bed_prune_to_plink_rust(**legacy_call_args)
                        rust_prune_direct_wall = float(time.time() - t_direct)
                        prune_pre_sites = int(total_n)
                        selected_n_sites = int(keep_n)
                        rust_prune_direct_done = True
                        prune_direct_backend = "rust-stream-io"
                    elif use_packed_prune_fastpath:
                        keep_mask, sites_all, packed_raw, _packed_n = _ld_prune_with_maf_priority_packed_plink(
                            gfile,
                            prune_spec,
                            threads=int(args.thread),
                        )
                        packed_prune_keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)
                        packed_prune_packed = np.ascontiguousarray(
                            np.asarray(packed_raw, dtype=np.uint8)
                        )
                        packed_prune_source_prefix = str(gfile)
                    else:
                        geno_all, sites_all = _stack_chunks(_make_chunks())
                        if geno_all.shape[0] <= 0 or len(sites_all) == 0:
                            raise ValueError(
                                "All variants were filtered out before LD prune "
                                "(-extract/-chr/-from-bp/-to-bp/-maf/-geno/-het)."
                            )
                        keep_mask = _ld_prune_with_maf_priority(
                            geno_all,
                            sites_all,
                            prune_spec,
                        )
                    if not rust_prune_direct_done:
                        prune_pre_sites = int(len(sites_all))
                        keep_n = int(np.sum(keep_mask))
                    else:
                        keep_n = int(selected_n_sites)
                    if keep_n <= 0:
                        raise ValueError(
                            "All variants were filtered out by LD prune; "
                            "try a looser --prune r^2 threshold or larger window."
                        )
                    if rust_prune_direct_done:
                        pass
                    elif use_packed_prune_fastpath:
                        kept_sites = [sites_all[i] for i in np.flatnonzero(keep_mask)]
                        push_snp_sites = [(str(s.chrom), int(s.pos)) for s in kept_sites]
                        pruned_geno = None
                        pruned_sites = None
                    else:
                        pruned_geno = np.asarray(geno_all[keep_mask, :], dtype=np.float32)
                        pruned_sites = [sites_all[i] for i in np.flatnonzero(keep_mask)]
                    selected_n_sites = int(keep_n)
                    if rust_prune_direct_done:
                        backend = str(prune_direct_backend or "rust-stream-io")
                    else:
                        backend = "rust-packed" if use_packed_prune_fastpath else "python-dense"
                    task.complete(
                        "Applying LD prune ...Finished "
                        f"(backend={backend}, kept={keep_n}, dropped={int(prune_pre_sites - keep_n)})"
                    )
                except Exception:
                    task.fail("Applying LD prune ...Failed")
                    raise
            if not status_enabled:
                if rust_prune_direct_done:
                    backend = str(prune_direct_backend or "rust-stream-io")
                else:
                    backend = "rust-packed" if use_packed_prune_fastpath else "python-dense"
                _log_status_file_only(
                    logger,
                    "Applying LD prune ...Finished "
                    f"(backend={backend}, kept={selected_n_sites}, dropped={int(prune_pre_sites - selected_n_sites)})",
                )
        if (
            use_packed_prune_fastpath
            and (not use_external_plink_prune)
            and _packed_prune_kernel_stats is not None
        ):
            try:
                (
                    k_backend,
                    avx2_available,
                    neon_available,
                    force_scalar,
                    total_calls,
                    avx2_calls,
                    neon_calls,
                    scalar_calls,
                    avx2_hit_rate,
                ) = _packed_prune_kernel_stats(reset=False)
                stats_msg = (
                    "Prune kernel stats: "
                    f"backend={k_backend}, "
                    f"avx2_available={int(bool(avx2_available))}, "
                    f"neon_available={int(bool(neon_available))}, "
                    f"force_scalar={int(bool(force_scalar))}, "
                    f"dot_calls={int(total_calls)}, "
                    f"avx2_calls={int(avx2_calls)}, "
                    f"neon_calls={int(neon_calls)}, "
                    f"scalar_calls={int(scalar_calls)}, "
                    f"avx2_hit_rate={float(avx2_hit_rate):.4f}"
                )
                if print_prune_kernel_stats:
                    logger.info(stats_msg)
                else:
                    _log_status_file_only(logger, stats_msg)
            except Exception:
                pass
    elif need_selected_count and out_fmt == "npy":
        # NPY memmap requires known target shape; keep pre-count only for this path.
        selected_n_sites = _count_sites_from_chunks(_make_chunks())
        if selected_n_sites <= 0:
            raise ValueError("All variants were filtered out by filters (-extract/-chr/-from-bp/-to-bp/-maf/-geno/-het).")
    elif need_selected_count:
        # Single-pass mode: write and count simultaneously, then backfill site count.
        selected_n_sites = None

    out_sample_ids = keep_sample_ids if keep_sample_ids is not None else [str(s) for s in sample_ids]
    print(f"Genotype source: {format_path_for_display(gfile)}")
    if selected_n_sites is not None:
        print(f"Samples: {len(out_sample_ids)}, sites: {int(selected_n_sites)}")
    else:
        print(f"Samples: {len(out_sample_ids)}, sites: calculating...")

    def _make_output_chunks() -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
        if pruned_geno is not None and pruned_sites is not None:
            return _iter_snp_named_chunks(
                _iter_chunks_from_matrix(pruned_geno, pruned_sites, chunk_size=50_000),
                snp_name_template,
            )
        return _iter_snp_named_chunks(_make_chunks(), snp_name_template)

    if rust_prune_direct_done:
        _rewrite_plink_bim_snp_names(str(out_prefix), snp_name_template)
        log_success(logger, f"Format conversion completed in {rust_prune_direct_wall:.2f} s")
        log_success(logger, f"Output written: {output_display}")
        return

    # Fast path for PLINK->PLINK when prune was done on packed BED:
    # avoid decode/re-encode by writing BED subset directly.
    if (
        out_fmt == "plink"
        and packed_prune_keep_mask is not None
        and packed_prune_packed is not None
        and packed_prune_source_prefix is not None
        and keep_sample_ids is None
    ):
        t0 = time.time()
        _write_plink_subset_from_packed(
            src_prefix=str(packed_prune_source_prefix),
            out_prefix=str(out_prefix),
            packed=np.asarray(packed_prune_packed, dtype=np.uint8),
            keep_mask=np.asarray(packed_prune_keep_mask, dtype=bool),
            max_output_sites=output_site_limit,
        )
        _rewrite_plink_bim_snp_names(str(out_prefix), snp_name_template)
        log_success(logger, f"Format conversion completed in {time.time() - t0:.2f} s")
        log_success(logger, f"Output written: {output_display}")
        return

    write_counted_sites = [0]
    output_chunks: Iterator[tuple[np.ndarray, list[SiteInfo]]] = _make_output_chunks()
    if selected_n_sites is None:
        first_chunk, output_chunks = _peek_chunk_iterator(output_chunks)
        if first_chunk is None:
            raise ValueError("All variants were filtered out by filters (-extract/-chr/-from-bp/-to-bp/-maf/-geno/-het).")
        output_chunks = _iter_counted_chunks(output_chunks, write_counted_sites)

    t0 = time.time()
    if out_fmt == "vcf":
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_genotype_streaming(
            out_path,
            out_sample_ids,
            output_chunks,
            fmt="vcf",
            total_snps=(int(selected_n_sites) if selected_n_sites is not None else None),
            desc="Writing VCF",
        )
    elif out_fmt == "hmp":
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_genotype_streaming(
            out_path,
            out_sample_ids,
            output_chunks,
            fmt="hmp",
            total_snps=(int(selected_n_sites) if selected_n_sites is not None else None),
            desc="Writing HMP",
        )
    elif out_fmt == "plink":
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
        save_genotype_streaming(
            out_prefix,
            out_sample_ids,
            output_chunks,
            fmt="plink",
            total_snps=(int(selected_n_sites) if selected_n_sites is not None else None),
            desc="Writing PLINK",
        )
    elif out_fmt == "txt":
        write_text_output(
            out_path,
            out_sample_ids,
            output_chunks,
            total_sites=(int(selected_n_sites) if selected_n_sites is not None else None),
        )
    elif out_fmt == "gfd":
        write_gfd_output(
            out_path,
            out_sample_ids,
            output_chunks,
            total_sites=(int(selected_n_sites) * 2 if selected_n_sites is not None else None),
            source_is_dosage012=True,
        )
    else:
        if selected_n_sites is None:
            raise RuntimeError("Internal error: selected_n_sites must be known for NPY output.")
        write_npy_output(
            out_path,
            out_sample_ids,
            output_chunks,
            total_sites=int(selected_n_sites),
        )

    if selected_n_sites is None:
        selected_n_sites = int(write_counted_sites[0])
        if selected_n_sites <= 0:
            raise ValueError("All variants were filtered out by filters (-extract/-chr/-from-bp/-to-bp/-maf/-geno/-het).")
        print(f"Samples: {len(out_sample_ids)}, sites: {int(selected_n_sites)}")

    log_success(logger, f"Format conversion completed in {time.time() - t0:.2f} s")
    log_success(logger, f"Output written: {output_display}")


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
