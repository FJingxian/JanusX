#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JanusX WebUI (Post-GWAS first)

A lightweight built-in web server for JanusX.
Current scope: submit and monitor postgwas jobs, and browse GWAS history.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
import numpy as np
from email import policy
from email.parser import BytesParser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..script._common.helptext import cli_help_formatter, minimal_help_epilog
from ..script._common.gwas_history import (
    get_gwas_history_row,
    get_postgwas_run,
    get_gwas_run,
    list_annotation_registry,
    list_gwas_history_rows,
    record_gwas_run,
    resolve_db_path,
    resolve_jx_home,
    upsert_postgwas_run,
)
from .render import (
    render_single_svg,
    render_merged_manhattan_svg,
    build_sig_table,
    build_merged_sig_table,
    annotate_sig_rows_with_genes,
    activate_task_cache,
)


META_FILE = "metadata.json"
STDOUT_FILE = "stdout.log"
STDERR_FILE = "stderr.log"


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_job_name(raw: str) -> str:
    txt = str(raw or "").strip()
    if txt == "":
        return "postgwas"
    out = []
    for ch in txt:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch in (" ", "."):
            out.append("_")
    s = "".join(out).strip("_")
    return s or "postgwas"


def _new_job_id(name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{_safe_job_name(name)}-{uuid.uuid4().hex[:8]}"


def _tail_text(path: Path, max_bytes: int = 32_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _looks_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}


def _resolve_existing_path(raw_path: str, base_dirs: list[Path] | None = None) -> str:
    txt = str(raw_path or "").strip()
    if txt == "":
        return ""
    p = Path(txt).expanduser()
    if p.exists():
        return str(p.resolve())
    if p.is_absolute():
        return ""
    for bd in (base_dirs or []):
        try:
            cand = (bd / p).expanduser()
        except Exception:
            continue
        if cand.exists():
            return str(cand.resolve())
    return ""


_CHR_COL_CANDIDATES = {
    "chrom",
    "chr",
    "chromosome",
    "chromosomeid",
    "chromid",
    "chrid",
    "chromname",
    "seqid",
    "seqname",
    "contig",
    "scaffold",
    "chromosome_id",
    "chrom_id",
}
_POS_COL_CANDIDATES = {
    "pos",
    "position",
    "bp",
    "ps",
    "basepair",
    "basepairposition",
    "physicalposition",
    "physicalpos",
    "coordinate",
    "coord",
    "location",
    "loc",
}
_P_COL_CANDIDATES = {
    "p",
    "pvalue",
    "pval",
    "pwald",
    "pwaldvalue",
    "waldp",
    "waldpvalue",
    "plrt",
    "lrtp",
    "prob",
}


def _guess_delimiter(header_line: str) -> str | None:
    text = str(header_line or "")
    if "\t" in text:
        return "\t"
    if "," in text:
        return ","
    if ";" in text:
        return ";"
    return None


def _split_fields(line: str, delim: str | None) -> list[str]:
    txt = str(line or "").rstrip("\n\r")
    if delim is None:
        return re.split(r"\s+", txt.strip())
    return txt.split(delim)


def _parse_multipart_form_data(content_type: str, raw: bytes) -> dict[str, Any]:
    ctype = str(content_type or "").strip()
    if "multipart/form-data" not in ctype.lower():
        raise ValueError("Content-Type must be multipart/form-data.")
    if not isinstance(raw, (bytes, bytearray)) or len(raw) == 0:
        return {}
    head = f"Content-Type: {ctype}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8", errors="replace")
    msg = BytesParser(policy=policy.default).parsebytes(head + bytes(raw))
    if not msg.is_multipart():
        raise ValueError("invalid multipart payload.")

    out: dict[str, Any] = {}

    def _put(name: str, value: Any) -> None:
        if name not in out:
            out[name] = value
            return
        cur = out.get(name)
        if isinstance(cur, list):
            cur.append(value)
            out[name] = cur
        else:
            out[name] = [cur, value]

    for part in msg.iter_parts():
        name = str(part.get_param("name", header="content-disposition") or "").strip()
        if name == "":
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename is not None:
            _put(
                name,
                {
                    "filename": str(filename),
                    "content_type": str(part.get_content_type() or ""),
                    "data": bytes(payload),
                },
            )
        else:
            charset = str(part.get_content_charset() or "utf-8")
            try:
                txt = bytes(payload).decode(charset, errors="replace")
            except Exception:
                txt = bytes(payload).decode("utf-8", errors="replace")
            _put(name, txt)
    return out


def _col_key(v: object) -> str:
    s = str(v or "").strip().lower()
    if s.startswith("#"):
        s = s[1:]
    return re.sub(r"[^a-z0-9]+", "", s)


def _chrom_sort_key(value: object) -> tuple[int, object]:
    s = str(value or "").strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    if s.isdigit():
        return (0, int(s))
    up = s.upper()
    if up == "X":
        return (1, 23)
    if up == "Y":
        return (1, 24)
    if up in {"M", "MT"}:
        return (1, 25)
    parts = re.split(r"(\d+)", s)
    out: list[tuple[int, object]] = []
    for p in parts:
        if not p:
            continue
        if p.isdigit():
            out.append((0, int(p)))
        else:
            out.append((1, p.lower()))
    return (2, tuple(out))


def _extract_chromosomes_from_gwas(path: Path, *, max_lines: int | None = None) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        if str(path.name).lower().endswith(".gz"):
            import gzip
            fh = gzip.open(path, "rt", encoding="utf-8", errors="replace")
        else:
            fh = path.open("r", encoding="utf-8", errors="replace")
        with fh as f:
            header = f.readline()
            if not header:
                return []
            delim = _guess_delimiter(header)
            cols = [c.strip() for c in _split_fields(header, delim)]
            if len(cols) == 0:
                return []
            chrom_idx = 0
            for i, c in enumerate(cols):
                ck = _col_key(c)
                if ck in _CHR_COL_CANDIDATES or ck.startswith("chr") or ("chrom" in ck):
                    chrom_idx = i
                    break
            out: list[str] = []
            seen: set[str] = set()
            for n, line in enumerate(f, start=1):
                if max_lines is not None and int(max_lines) > 0 and n > int(max_lines):
                    break
                if not line.strip():
                    continue
                arr = _split_fields(line, delim)
                if chrom_idx >= len(arr):
                    continue
                val = str(arr[chrom_idx]).strip()
                if val == "":
                    continue
                if val not in seen:
                    seen.add(val)
                    out.append(val)
            return sorted(out, key=_chrom_sort_key)
    except Exception:
        return []


def _pick_column_index(cols: list[str], *, kind: str) -> int:
    best_idx = -1
    best_score = -1
    for i, c in enumerate(cols):
        ck = _col_key(c)
        score = -1
        if kind == "chr":
            if ck in _CHR_COL_CANDIDATES:
                score = 100
            elif ck.startswith("chr"):
                score = 95
            elif ("chrom" in ck) or ("seq" in ck) or ("contig" in ck) or ("scaffold" in ck):
                score = 90
        elif kind == "pos":
            if ck in _POS_COL_CANDIDATES:
                score = 100
            elif ck.startswith("pos") or ck.endswith("bp"):
                score = 95
            elif ("position" in ck) or ("coord" in ck) or ("location" in ck):
                score = 90
        elif kind == "p":
            if ck in _P_COL_CANDIDATES:
                score = 100
            elif ck.startswith("pvalue") or ck.startswith("pval"):
                score = 95
            elif ("pwald" in ck) or ("wald" in ck and ck.startswith("p")):
                score = 92
            elif ("lrt" in ck and ck.startswith("p")):
                score = 90
        if score > best_score:
            best_score = score
            best_idx = int(i)
    return int(best_idx) if best_score >= 90 else -1


def _safe_float(text: object) -> float | None:
    try:
        v = float(str(text).strip())
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _detect_gwas_columns_fast(path: Path, *, max_scan_lines: int = 200_000) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise ValueError(f"uploaded file not found: {path}")
    try:
        if str(path.name).lower().endswith(".gz"):
            import gzip
            fh = gzip.open(path, "rt", encoding="utf-8", errors="replace")
        else:
            fh = path.open("r", encoding="utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(f"failed to open file: {exc}") from exc

    with fh as f:
        header = ""
        for line in f:
            t = str(line).strip()
            if t == "":
                continue
            if t.startswith("##"):
                continue
            header = str(line)
            break
        if header == "":
            raise RuntimeError("empty GWAS file")

        delim = _guess_delimiter(header)
        cols = [c.strip() for c in _split_fields(header, delim)]
        if len(cols) == 0:
            raise RuntimeError("failed to parse GWAS header")

        i_chr = _pick_column_index(cols, kind="chr")
        i_pos = _pick_column_index(cols, kind="pos")
        i_p = _pick_column_index(cols, kind="p")
        if i_chr < 0 or i_pos < 0 or i_p < 0:
            raise RuntimeError("Cannot detect required columns (chrom/pos/pvalue).")

        valid = 0
        chroms: set[str] = set()
        for n, line in enumerate(f, start=1):
            if n > int(max_scan_lines):
                break
            if not str(line).strip():
                continue
            arr = _split_fields(line, delim)
            if i_chr >= len(arr) or i_pos >= len(arr) or i_p >= len(arr):
                continue
            pos_v = _safe_float(arr[i_pos])
            p_v = _safe_float(arr[i_p])
            if pos_v is None or p_v is None:
                continue
            if p_v <= 0.0 or p_v > 1.0:
                continue
            valid += 1
            c = str(arr[i_chr]).strip()
            if c != "":
                chroms.add(c)

    if valid <= 0:
        raise RuntimeError("No valid SNP rows after filtering (pos/pvalue).")

    out_chroms = sorted(list(chroms), key=_chrom_sort_key)
    return {
        "chr": str(cols[i_chr]),
        "pos": str(cols[i_pos]),
        "pvalue": str(cols[i_p]),
        "n_rows": int(valid),
        "n_chrom": int(len(out_chroms)),
        "chroms": out_chroms,
    }


def _normalize_bimrange_items(raw: Any, *, max_items: int = 4) -> list[str]:
    items: list[str] = []
    if isinstance(raw, list):
        src = [str(x).strip() for x in raw]
    else:
        txt = str(raw or "")
        src = [x.strip() for x in re.split(r"[\n,;]+", txt)]
    for x in src:
        if x:
            items.append(x)
        if len(items) >= int(max_items):
            break
    return items


class WebUIState:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.jobs_dir = (self.root_dir / "jobs").resolve()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = resolve_db_path()
        self.anno_dir = self.db_path.parent / "anno"
        self.anno_dir.mkdir(parents=True, exist_ok=True)
        self.gwas_upload_dir = self.db_path.parent / "uploaded" / "gwas"
        self.gwas_upload_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._render_cache: dict[str, tuple[float, Any, dict[str, str]]] = {}
        self._chrom_cache: dict[str, tuple[float, list[str]]] = {}
        self._sig_cache: dict[str, dict[str, Any]] = {}
        self._history_error: str = ""
        self._startup_cleanup_report: dict[str, Any] = {
            "db_scanned": 0,
            "scanned": 0,
            "removed": 0,
            "pruned_runs": 0,
            "pruned_summary_rows": 0,
            "removed_missing_genotype": 0,
            "removed_missing_result": 0,
            "error": "",
        }
        try:
            self._startup_cleanup_report = self._cleanup_gwas_history_db_startup()
        except Exception as exc:
            self._startup_cleanup_report["error"] = str(exc)

    def startup_cleanup_report(self) -> dict[str, Any]:
        return dict(self._startup_cleanup_report)

    def _history_db_paths(self) -> list[Path]:
        out: list[Path] = []
        seen: set[str] = set()

        def _add(p: Path) -> None:
            try:
                rp = p.expanduser().resolve()
            except Exception:
                rp = p.expanduser()
            key = str(rp)
            if key in seen:
                return
            seen.add(key)
            out.append(rp)

        env_home = str(os.environ.get("JX_HOME", "")).strip()
        if env_home != "":
            _add(Path(env_home) / "janusx_tasks.db")
        _add(Path.home() / "JanusX" / ".janusx" / "janusx_tasks.db")
        _add(Path.home() / ".janusx" / "janusx_tasks.db")
        _add(self.db_path)
        return out

    def _run_base_dirs(self, run_row: dict[str, Any]) -> list[Path]:
        bases: list[Path] = []
        for p in [Path.cwd(), self.root_dir, self.db_path.parent]:
            if p not in bases:
                bases.append(p)
        for key in ("outprefix", "log_file", "phenofile", "genofile"):
            v = str(run_row.get(key, "")).strip()
            if v == "":
                continue
            try:
                parent = Path(v).expanduser().parent.resolve()
            except Exception:
                parent = Path(v).expanduser().parent
            if parent not in bases:
                bases.append(parent)
        return bases

    @staticmethod
    def _resolve_bfile_prefix(prefix: str, base_dirs: list[Path]) -> str:
        txt = str(prefix or "").strip()
        if txt == "":
            return ""
        base = Path(txt).expanduser()
        candidates: list[Path] = []
        if base.is_absolute():
            candidates.append(base)
        else:
            candidates.append(base)
            for bd in base_dirs:
                try:
                    candidates.append((bd / base).expanduser())
                except Exception:
                    continue
        for cand in candidates:
            bed = Path(f"{cand}.bed")
            bim = Path(f"{cand}.bim")
            fam = Path(f"{cand}.fam")
            if bed.exists() and bim.exists() and fam.exists():
                try:
                    return str(cand.resolve())
                except Exception:
                    return str(cand)
        return ""

    def _has_valid_genotype(self, run_row: dict[str, Any], base_dirs: list[Path]) -> bool:
        gfile = str(run_row.get("genofile", "")).strip()
        gkind = str(run_row.get("genofile_kind", "")).strip().lower()
        if gfile == "":
            return False
        if gkind == "bfile":
            return self._resolve_bfile_prefix(gfile, base_dirs) != ""
        return _resolve_existing_path(gfile, base_dirs) != ""

    def _has_valid_result_file(self, run_row: dict[str, Any], base_dirs: list[Path]) -> bool:
        summary_rows = run_row.get("summary_rows", [])
        if isinstance(summary_rows, list):
            for srow in summary_rows:
                if not isinstance(srow, dict):
                    continue
                rf = str(srow.get("result_file", "")).strip()
                if rf == "":
                    continue
                if _resolve_existing_path(rf, base_dirs) != "":
                    return True
        result_files = run_row.get("result_files", [])
        if isinstance(result_files, list):
            for rf in result_files:
                p = str(rf or "").strip()
                if p == "":
                    continue
                if _resolve_existing_path(p, base_dirs) != "":
                    return True
        return False

    def _cleanup_gwas_history_db_startup(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "db_scanned": 0,
            "scanned": 0,
            "removed": 0,
            "pruned_runs": 0,
            "pruned_summary_rows": 0,
            "removed_missing_genotype": 0,
            "removed_missing_result": 0,
            "error": "",
        }
        db_paths = self._history_db_paths()
        for db_path in db_paths:
            if not db_path.exists():
                continue
            report["db_scanned"] = int(report["db_scanned"]) + 1
            conn = sqlite3.connect(str(db_path), timeout=30)
            try:
                has_gwas = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='gwas_runs' LIMIT 1"
                ).fetchone()
                if has_gwas is None:
                    continue
                has_postgwas = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='postgwas_runs' LIMIT 1"
                ).fetchone() is not None
                rows = conn.execute(
                    """
                    SELECT run_id, genofile, genofile_kind, phenofile, outprefix, log_file,
                           result_files_json, summary_json, args_json
                    FROM gwas_runs
                    """
                ).fetchall()
                for row in rows:
                    run_id = str(row[0] or "").strip()
                    if run_id == "":
                        continue
                    run_row: dict[str, Any] = {
                        "run_id": run_id,
                        "genofile": str(row[1] or ""),
                        "genofile_kind": str(row[2] or ""),
                        "phenofile": str(row[3] or ""),
                        "outprefix": str(row[4] or ""),
                        "log_file": str(row[5] or ""),
                        "result_files": [],
                        "summary_rows": [],
                        "args": {},
                    }
                    try:
                        rf = json.loads(str(row[6] or "[]"))
                        if isinstance(rf, list):
                            run_row["result_files"] = rf
                    except Exception:
                        pass
                    try:
                        sr = json.loads(str(row[7] or "[]"))
                        if isinstance(sr, list):
                            run_row["summary_rows"] = sr
                    except Exception:
                        pass
                    try:
                        aj = json.loads(str(row[8] or "{}"))
                        if isinstance(aj, dict):
                            run_row["args"] = aj
                    except Exception:
                        pass
                    report["scanned"] = int(report["scanned"]) + 1
                    repaired_genofile = False
                    gfile0 = str(run_row.get("genofile", "")).strip()
                    gkind0 = str(run_row.get("genofile_kind", "")).strip().lower()
                    args0 = run_row.get("args", {})
                    src0 = ""
                    if isinstance(args0, dict):
                        src0 = str(args0.get("source", "")).strip().lower()
                    # Backward compatibility:
                    # old `jx -load gwas` records wrote empty genofile and were
                    # pruned by startup cleanup. Repair them in-place.
                    if gfile0 == "" and src0 == "jx-load-gwas":
                        cand = str(run_row.get("phenofile", "")).strip()
                        if cand != "":
                            run_row["genofile"] = cand
                            run_row["genofile_kind"] = "tsv" if gkind0 == "" else gkind0
                            repaired_genofile = True

                    base_dirs = self._run_base_dirs(run_row)
                    missing_geno = not self._has_valid_genotype(run_row, base_dirs)

                    orig_result_files = run_row.get("result_files", [])
                    orig_summary_rows = run_row.get("summary_rows", [])
                    repaired_summary_rows: list[dict[str, Any]] = []
                    repaired_pheno = False
                    if isinstance(orig_summary_rows, list):
                        for srow in orig_summary_rows:
                            if not isinstance(srow, dict):
                                continue
                            new_row = dict(srow)
                            old_pheno = str(new_row.get("phenotype", "")).strip()
                            rf = str(new_row.get("result_file", "")).strip()
                            if self._looks_temp_upload_phenotype(old_pheno):
                                src_name = Path(rf).name if rf != "" else ""
                                new_pheno = self._infer_phenotype_from_name(src_name)
                                if new_pheno != "" and new_pheno != old_pheno:
                                    new_row["phenotype"] = new_pheno
                                    repaired_pheno = True
                            repaired_summary_rows.append(new_row)
                    if repaired_pheno:
                        orig_summary_rows = repaired_summary_rows
                        run_row["summary_rows"] = repaired_summary_rows
                    valid_result_files: list[str] = []
                    if isinstance(orig_result_files, list):
                        for rf in orig_result_files:
                            p = str(rf or "").strip()
                            if p == "":
                                continue
                            if _resolve_existing_path(p, base_dirs) != "":
                                valid_result_files.append(p)
                    valid_summary_rows: list[dict[str, Any]] = []
                    if isinstance(orig_summary_rows, list):
                        for srow in orig_summary_rows:
                            if not isinstance(srow, dict):
                                continue
                            rf = str(srow.get("result_file", "")).strip()
                            if rf == "":
                                continue
                            if _resolve_existing_path(rf, base_dirs) != "":
                                valid_summary_rows.append(dict(srow))
                    removed_summary_n = max(
                        0,
                        (len(orig_summary_rows) if isinstance(orig_summary_rows, list) else 0)
                        - len(valid_summary_rows),
                    )
                    has_any_result = (len(valid_summary_rows) > 0) or (len(valid_result_files) > 0)
                    missing_result = not has_any_result

                    if missing_geno or missing_result:
                        conn.execute("DELETE FROM gwas_runs WHERE run_id = ?", (run_id,))
                        if has_postgwas:
                            conn.execute(
                                "DELETE FROM postgwas_runs WHERE run_id = ? OR history_id LIKE ?",
                                (run_id, f"{run_id}|%"),
                            )
                        report["removed"] = int(report["removed"]) + 1
                        if missing_geno:
                            report["removed_missing_genotype"] = int(report["removed_missing_genotype"]) + 1
                        if missing_result:
                            report["removed_missing_result"] = int(report["removed_missing_result"]) + 1
                        continue

                    pruned = False
                    if isinstance(orig_result_files, list) and len(valid_result_files) != len(orig_result_files):
                        pruned = True
                    if isinstance(orig_summary_rows, list) and len(valid_summary_rows) != len(orig_summary_rows):
                        pruned = True
                    if pruned:
                        if repaired_genofile:
                            conn.execute(
                                "UPDATE gwas_runs SET genofile = ?, genofile_kind = ?, result_files_json = ?, summary_json = ? WHERE run_id = ?",
                                (
                                    str(run_row.get("genofile", "")),
                                    str(run_row.get("genofile_kind", "")),
                                    json.dumps(valid_result_files, ensure_ascii=False),
                                    json.dumps(valid_summary_rows, ensure_ascii=False),
                                    run_id,
                                ),
                            )
                        else:
                            conn.execute(
                                "UPDATE gwas_runs SET result_files_json = ?, summary_json = ? WHERE run_id = ?",
                                (
                                    json.dumps(valid_result_files, ensure_ascii=False),
                                    json.dumps(valid_summary_rows, ensure_ascii=False),
                                    run_id,
                                ),
                            )
                        if has_postgwas:
                            # history_id is run_id|idx; summary-row reindex would stale old mappings.
                            conn.execute(
                                "DELETE FROM postgwas_runs WHERE run_id = ? OR history_id LIKE ?",
                                (run_id, f"{run_id}|%"),
                            )
                        report["pruned_runs"] = int(report["pruned_runs"]) + 1
                        report["pruned_summary_rows"] = int(report["pruned_summary_rows"]) + int(removed_summary_n)
                    elif repaired_pheno or repaired_genofile:
                        if repaired_pheno and repaired_genofile:
                            conn.execute(
                                "UPDATE gwas_runs SET genofile = ?, genofile_kind = ?, summary_json = ? WHERE run_id = ?",
                                (
                                    str(run_row.get("genofile", "")),
                                    str(run_row.get("genofile_kind", "")),
                                    json.dumps(valid_summary_rows, ensure_ascii=False),
                                    run_id,
                                ),
                            )
                        elif repaired_genofile:
                            conn.execute(
                                "UPDATE gwas_runs SET genofile = ?, genofile_kind = ? WHERE run_id = ?",
                                (
                                    str(run_row.get("genofile", "")),
                                    str(run_row.get("genofile_kind", "")),
                                    run_id,
                                ),
                            )
                        else:
                            conn.execute(
                                "UPDATE gwas_runs SET summary_json = ? WHERE run_id = ?",
                                (
                                    json.dumps(valid_summary_rows, ensure_ascii=False),
                                    run_id,
                                ),
                            )
                conn.commit()
            finally:
                conn.close()
        return report

    def list_anno_files(self) -> list[dict[str, Any]]:
        rows = list_annotation_registry(limit=500)
        out: list[dict[str, Any]] = []
        for r in rows:
            chroms = r.get("chroms", [])
            if not isinstance(chroms, list):
                chroms = []
            out.append(
                {
                    "alias": str(r.get("alias", "")),
                    "name": str(r.get("file_name", "")),
                    "path": str(r.get("path", "")),
                    "md5": str(r.get("md5", "")),
                    "chroms": chroms,
                    "mtime": str(r.get("imported_at", "")),
                }
            )
        return out

    @staticmethod
    def _safe_upload_filename(name: str) -> str:
        raw = str(name or "").strip().replace("\\", "/").split("/")[-1]
        out = []
        for ch in raw:
            if ch.isalnum() or ch in ("-", "_", "."):
                out.append(ch)
            elif ch in (" ",):
                out.append("_")
        s = "".join(out).strip("._")
        if s == "":
            return "uploaded_gwas.tsv"
        # Windows reserved device names
        p = Path(s)
        stem = p.stem.strip(" .")
        suffix = p.suffix
        reserved = {
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        }
        if stem.upper() in reserved:
            stem = f"_{stem}"
        # Keep filename reasonably short for portability.
        if len(stem) > 80:
            stem = stem[:80]
        s = f"{stem}{suffix}" if suffix else stem
        if s == "":
            return "uploaded_gwas.tsv"
        return s

    def _upload_storage_path(self, original_name: str, idx: int) -> Path:
        safe = self._safe_upload_filename(original_name)
        low = safe.lower()
        ext = ".tsv"
        if low.endswith(".vcf.gz"):
            ext = ".vcf.gz"
        elif low.endswith(".tsv.gz"):
            ext = ".tsv.gz"
        elif low.endswith(".txt.gz"):
            ext = ".txt.gz"
        else:
            p = Path(safe)
            if p.suffix:
                ext = p.suffix
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}-{int(idx):03d}-{uuid.uuid4().hex[:10]}{ext}"
        return self.gwas_upload_dir / name

    @staticmethod
    def _infer_model_from_name(name: str) -> str:
        low = str(name or "").lower()
        if low.endswith(".lmm.tsv") or ".lmm." in low:
            return "LMM"
        if low.endswith(".farmcpu.tsv") or ".farmcpu." in low:
            return "FarmCPU"
        if low.endswith(".lm.tsv") or ".lm." in low:
            return "LM"
        return "LMM"

    @staticmethod
    def _infer_phenotype_from_name(name: str) -> str:
        raw = str(name or "").strip()
        if raw == "":
            return "phenotype"
        base = Path(raw).name
        low = base.lower()
        for ext in (".tsv.gz", ".txt.gz", ".csv.gz", ".vcf.gz"):
            if low.endswith(ext):
                base = base[: -len(ext)]
                low = base.lower()
                break
        for ext in (".tsv", ".txt", ".csv", ".vcf", ".gz"):
            if low.endswith(ext):
                base = base[: -len(ext)]
                low = base.lower()
                break
        parts = [p for p in base.split(".") if str(p).strip() != ""]
        if len(parts) == 0:
            return "phenotype"
        assoc_tail = {"lmm", "lm", "farmcpu"}
        genetic_tail = {"add", "dom", "rec", "het"}
        while len(parts) > 0 and parts[-1].lower() in assoc_tail:
            parts.pop()
        while len(parts) > 0 and parts[-1].lower() in genetic_tail:
            parts.pop()
        if len(parts) == 0:
            return Path(raw).stem
        if len(parts) >= 3:
            cand = ".".join(parts[1:]).strip()
            if cand != "":
                return cand
        return ".".join(parts).strip() or "phenotype"

    @staticmethod
    def _looks_temp_upload_phenotype(name: str) -> bool:
        txt = str(name or "").strip()
        if txt == "":
            return False
        # Legacy upload filename pattern: YYYYMMDD-HHMMSS-<idx>-<hex>
        return re.fullmatch(r"\d{8}-\d{6}-\d{3}-[0-9a-fA-F]{8,16}", txt) is not None

    def _resolve_upload_genofile(self, genofile: str, genofile_kind: str, *, base: Path) -> tuple[str, str]:
        gtxt = str(genofile or "").strip()
        ktxt = str(genofile_kind or "").strip().lower()
        base_dirs = [Path.cwd(), self.root_dir, self.db_path.parent, base]
        if gtxt == "":
            try:
                return str(base.expanduser().absolute()), "tsv"
            except Exception:
                return str(base), "tsv"
        if ktxt == "bfile":
            rp = self._resolve_bfile_prefix(gtxt, base_dirs)
            if rp == "":
                raise ValueError(f"PLINK prefix not found or incomplete: {gtxt} (.bed/.bim/.fam required)")
            return rp, "bfile"
        rp = _resolve_existing_path(gtxt, base_dirs)
        if rp == "":
            raise ValueError(f"genofile not found: {gtxt}")
        low = str(rp).lower()
        if low.endswith(".vcf") or low.endswith(".vcf.gz"):
            return rp, "vcf"
        return rp, "tsv"

    def register_uploaded_gwas(
        self,
        *,
        file_path: Path,
        original_name: str = "",
        phenotype: str = "",
        model: str = "",
        genofile: str = "",
        genofile_kind: str = "",
    ) -> dict[str, Any]:
        p0 = Path(file_path).expanduser()
        try:
            p = p0.absolute()
        except Exception:
            p = p0
        if (not p.exists()) or (not p.is_file()):
            raise ValueError(f"uploaded file not found: {p}")
        try:
            det = _detect_gwas_columns_fast(p)
        except Exception as exc:
            raise RuntimeError(f"detect-columns failed for {p.name}: {exc}") from exc
        pheno = str(phenotype or "").strip()
        if pheno == "":
            pheno = self._infer_phenotype_from_name(str(original_name or p.name))
        mdl = str(model or "").strip()
        if mdl == "":
            mdl = self._infer_model_from_name(str(original_name or p.name))
        try:
            gpath, gkind = self._resolve_upload_genofile(genofile, genofile_kind, base=p.parent)
        except Exception as exc:
            raise RuntimeError(f"resolve-genofile failed: {exc}") from exc

        run_id = f"webui-upload-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        uploads_dir = self.root_dir / "uploads"
        try:
            uploads_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            outprefix = str((uploads_dir / p.stem).absolute())
        except Exception:
            outprefix = str(uploads_dir / p.stem)
        try:
            log_file = str((uploads_dir / f"{p.stem}.log").absolute())
        except Exception:
            log_file = str(uploads_dir / f"{p.stem}.log")
        summary_rows = [
            {
                "phenotype": pheno,
                "model": mdl,
                "pheno_col_idx": -1,
                "result_file": str(p),
                "nidv": 0,
                "eff_snp": int(det.get("n_rows", 0) or 0),
            }
        ]
        args_data = {
            "model": "upload",
            "source": "webui-upload",
            "detected_columns": {
                "chr": str(det.get("chr", "")),
                "pos": str(det.get("pos", "")),
                "pvalue": str(det.get("pvalue", "")),
            },
        }
        try:
            record_gwas_run(
                run_id=run_id,
                status="completed",
                genofile=str(gpath),
                genofile_kind=str(gkind),
                phenofile=str(p),
                outprefix=outprefix,
                log_file=log_file,
                result_files=[str(p)],
                summary_rows=summary_rows,
                args_data=args_data,
                error_text="",
                created_at=_now_str(),
            )
        except Exception as exc:
            raise RuntimeError(f"record-db failed: {exc}") from exc
        return {
            "run_id": run_id,
            "history_id": f"{run_id}|0",
            "path": str(p),
            "file_name": str(p.name),
            "detected": {
                "chr": str(det.get("chr", "")),
                "pos": str(det.get("pos", "")),
                "pvalue": str(det.get("pvalue", "")),
            },
            "n_rows": int(det.get("n_rows", 0) or 0),
            "n_chrom": int(det.get("n_chrom", 0) or 0),
        }

    def _job_dir(self, job_id: str) -> Path:
        return (self.jobs_dir / job_id).resolve()

    def _meta_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / META_FILE

    def _load_meta_unlocked(self, job_id: str) -> dict[str, Any] | None:
        p = self._meta_path(job_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_meta_unlocked(self, job_id: str, meta: dict[str, Any]) -> None:
        p = self._meta_path(job_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_jobs(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for d in self.jobs_dir.iterdir():
            if not d.is_dir():
                continue
            p = d / META_FILE
            if not p.exists():
                continue
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append(
                {
                    "job_id": meta.get("job_id", d.name),
                    "name": meta.get("name", ""),
                    "status": meta.get("status", "unknown"),
                    "created_at": meta.get("created_at", ""),
                    "started_at": meta.get("started_at", ""),
                    "finished_at": meta.get("finished_at", ""),
                    "return_code": meta.get("return_code", None),
                }
            )
        rows.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return rows

    def list_gwas_history(self, limit: int = 200) -> list[dict[str, Any]]:
        try:
            rows = list_gwas_history_rows(limit=limit)
            self._history_error = ""
        except Exception:
            self._history_error = "Failed to read GWAS history DB."
            return []
        return rows

    def get_gwas_history(self, run_id: str) -> dict[str, Any] | None:
        rid = str(run_id or "").strip()
        if rid == "":
            return None
        try:
            # Prefer row-level ID: "<run_id>|<idx>"
            if "|" in rid:
                return get_gwas_history_row(rid)
            return get_gwas_run(rid)
        except Exception:
            return None

    @staticmethod
    def _row_genotype_key(row: dict[str, Any]) -> str:
        gfile = str(row.get("genofile", "")).strip()
        gkind = str(row.get("genofile_kind", "")).strip().lower()
        if gfile != "":
            try:
                gp = str(Path(gfile).expanduser().resolve())
            except Exception:
                gp = gfile
            return f"{gkind}:{gp}"
        geno = str(row.get("genotype", "")).strip()
        return f"{gkind}:{geno}"

    def _resolve_merge_rows(self, history_ids: list[str]) -> list[dict[str, Any]]:
        ids = [str(x).strip() for x in (history_ids or []) if str(x).strip() != ""]
        if len(ids) == 0:
            raise ValueError("history_ids is required.")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hid in ids:
            if hid in seen:
                continue
            seen.add(hid)
            row = self.get_gwas_history(hid)
            if row is None:
                raise ValueError(f"GWAS history not found: {hid}")
            rows.append(row)
        if len(rows) < 2:
            raise ValueError("Select at least 2 GWAS tasks for merged Manhattan plot.")
        base_key = self._row_genotype_key(rows[0])
        for r in rows[1:]:
            if self._row_genotype_key(r) != base_key:
                raise ValueError("Selected GWAS tasks must have the same genotype file.")
        return rows

    def render_history_svg(
        self,
        history_id: str,
        *,
        bimrange: Any = "",
        anno: str = "",
        highlight_range: Any = None,
        full: bool = False,
        manh_ratio: Any = 2.0,
        manh_palette: str = "auto",
        manh_alpha: Any = 0.7,
        manh_marker: str = "o",
        manh_size: Any = 16.0,
        ld_color: str = "#4b5563",
        ld_p_threshold: Any = "auto",
        editable_svg: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        row = self.get_gwas_history(history_id)
        if row is None:
            raise ValueError(f"GWAS history not found: {history_id}")
        activate_task_cache(row, anno_file=str(anno or ""), preload=False)
        with self._lock:
            svg, info = render_single_svg(
                row,
                bimrange_text=bimrange,
                anno_file=str(anno or "").strip(),
                highlight_ranges=highlight_range,
                full_plot=bool(full),
                manh_ratio=manh_ratio,
                manh_palette=str(manh_palette or "auto"),
                manh_alpha=manh_alpha,
                manh_marker=str(manh_marker or "o"),
                manh_size=manh_size,
                ld_color=str(ld_color or "").strip(),
                ld_p_threshold=ld_p_threshold,
                editable_svg=bool(editable_svg),
                cache=self._render_cache,
            )
        return svg, info

    def render_merged_history_svg(
        self,
        history_ids: list[str],
        *,
        bimrange: Any = "",
        threshold: Any = "auto",
        include_all_points: bool = False,
        manh_ratio: Any = 2.0,
        manh_palette: str = "auto",
        manh_alpha: Any = 0.7,
        manh_marker: str = "o",
        manh_size: Any = 16.0,
        series_styles: Any = None,
        ld_color: str = "#4b5563",
        ld_p_threshold: Any = "auto",
        render_qq: bool = True,
        editable_svg: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        ids = [str(x).strip() for x in (history_ids or []) if str(x).strip() != ""]
        rows = self._resolve_merge_rows(ids)
        base_key = self._row_genotype_key(rows[0])

        with self._lock:
            svg, info = render_merged_manhattan_svg(
                rows,
                bimrange_text=bimrange,
                threshold=threshold,
                include_all_points=bool(include_all_points),
                manh_ratio=manh_ratio,
                manh_palette=str(manh_palette or "auto"),
                manh_alpha=manh_alpha,
                manh_marker=str(manh_marker or "o"),
                manh_size=manh_size,
                series_styles=series_styles,
                ld_color=str(ld_color or "").strip(),
                ld_p_threshold=ld_p_threshold,
                render_qq=bool(render_qq),
                editable_svg=bool(editable_svg),
                cache=self._render_cache,
            )
        info["history_ids"] = [str(x) for x in ids]
        info["genotype_key"] = base_key
        return svg, info

    def _resolve_history_result_path(self, row: dict[str, Any]) -> str:
        base_dirs = self._run_base_dirs(row)
        row_result = str(row.get("result_file", "")).strip()
        if row_result:
            rp = _resolve_existing_path(row_result, base_dirs)
            if rp:
                return rp
        result_files_raw = row.get("result_files", [])
        if isinstance(result_files_raw, list):
            for p in result_files_raw:
                rp = _resolve_existing_path(str(p).strip(), base_dirs)
                if rp:
                    return rp
        return ""

    def get_history_chromosomes(self, history_id: str) -> list[str]:
        row = self.get_gwas_history(history_id)
        if row is None:
            raise ValueError(f"GWAS history not found: {history_id}")
        activate_task_cache(row, anno_file=None)
        gwas_path = self._resolve_history_result_path(row)
        if gwas_path == "":
            return []
        p = Path(gwas_path)
        try:
            rp = str(p.resolve())
            mtime = float(p.stat().st_mtime)
        except Exception:
            rp = str(p)
            mtime = -1.0
        with self._lock:
            cached = self._chrom_cache.get(rp)
            if cached is not None and float(cached[0]) == mtime:
                return list(cached[1])
        chroms = _extract_chromosomes_from_gwas(p, max_lines=None)
        with self._lock:
            self._chrom_cache[rp] = (mtime, list(chroms))
        return chroms

    def get_history_sigsites(
        self,
        history_id: str,
        *,
        threshold: Any = "auto",
        anno_file: str = "",
        search: str = "",
        limit: int = 2000,
    ) -> dict[str, Any]:
        row = self.get_gwas_history(history_id)
        if row is None:
            raise ValueError(f"GWAS history not found: {history_id}")
        activate_task_cache(row, anno_file=str(anno_file or ""))

        gwas_path = self._resolve_history_result_path(row)
        if gwas_path == "":
            raise ValueError("No valid GWAS result file for selected history.")
        p = Path(gwas_path)
        try:
            rp = str(p.resolve())
            mtime = float(p.stat().st_mtime)
        except Exception:
            rp = str(p)
            mtime = -1.0

        try:
            thr = float(threshold)
        except Exception:
            thr = float("nan")
        if not np.isfinite(thr):
            thr = float("nan")

        anno_txt = str(anno_file or "").strip()
        anno_tag = ""
        if anno_txt != "":
            ap = Path(anno_txt).expanduser()
            try:
                arp = str(ap.resolve())
                am = float(ap.stat().st_mtime) if ap.exists() else -1.0
            except Exception:
                arp = str(ap)
                am = -1.0
            anno_tag = f"|anno={arp}|anno_m={am:.6f}"

        cache_key = (
            f"{history_id}|{rp}|{mtime:.6f}|thr={thr:.3e}{anno_tag}"
        )
        with self._lock:
            cached = self._sig_cache.get(cache_key)
        if cached is None:
            sig = build_sig_table(
                row,
                threshold=thr,
                anno_file=anno_txt,
                annotate_rows=False,
            )
            with self._lock:
                self._sig_cache[cache_key] = dict(sig)
            data = sig
        else:
            data = dict(cached)

        rows_raw = data.get("rows", [])
        if not isinstance(rows_raw, list):
            rows_raw = []
        rows = list(rows_raw)

        def _p_rank_key(r: dict[str, Any]) -> tuple[float, str, int]:
            try:
                p = float(r.get("p", np.nan))
            except Exception:
                p = np.nan
            if (not np.isfinite(p)) or p <= 0.0:
                p = float("inf")
            chrom = str(r.get("chrom", ""))
            try:
                pos = int(r.get("pos", 0))
            except Exception:
                pos = 0
            return (p, chrom, pos)

        q = str(search or "").strip().lower()
        if q:
            if anno_txt != "":
                annotate_sig_rows_with_genes(rows, anno_txt)

            def _hit(r: dict[str, Any]) -> bool:
                txt = " ".join(
                    [
                        str(r.get("chrom", "")),
                        str(r.get("pos", "")),
                        str(r.get("p", "")),
                        str(r.get("Gene", "")),
                    ]
                ).lower()
                return q in txt

            rows = [r for r in rows if _hit(r)]
        else:
            # Default UI view: only show top 100 significant sites.
            rows = sorted(rows, key=_p_rank_key)[:100]
            if anno_txt != "":
                annotate_sig_rows_with_genes(rows, anno_txt)
        return {
            "history_id": str(history_id),
            "threshold": float(data.get("threshold", thr)),
            "n_sig": int(data.get("n_sig", 0)),
            "n_leads": int(data.get("n_leads", 0)),
            "n_show": int(len(rows)),
            "rows": rows,
        }

    def get_merged_history_sigsites(
        self,
        history_ids: list[str],
        *,
        threshold: Any = "auto",
        anno_file: str = "",
        search: str = "",
        limit: int = 2000,
    ) -> dict[str, Any]:
        ids = [str(x).strip() for x in (history_ids or []) if str(x).strip() != ""]
        rows = self._resolve_merge_rows(ids)
        # Activate cache with first row (shared genotype); then preload each GWAS table.
        activate_task_cache(rows[0], anno_file=str(anno_file or ""))
        for r in rows:
            gwas_path = self._resolve_history_result_path(r)
            if gwas_path:
                try:
                    from .render import _load_sig_table_cached  # local import to avoid broad API change

                    _load_sig_table_cached(Path(gwas_path))
                except Exception:
                    pass

        try:
            thr = float(threshold)
        except Exception:
            thr = float("nan")
        if not np.isfinite(thr):
            thr = float("nan")

        # Cache key depends on all selected result files + params.
        path_tokens: list[str] = []
        for r in rows:
            p0 = self._resolve_history_result_path(r)
            if p0 == "":
                continue
            p = Path(p0)
            try:
                rp = str(p.resolve())
                mtime = float(p.stat().st_mtime)
            except Exception:
                rp = str(p)
                mtime = -1.0
            path_tokens.append(f"{rp}@{mtime:.6f}")
        path_tokens.sort()

        anno_txt = str(anno_file or "").strip()
        anno_tag = ""
        if anno_txt != "":
            ap = Path(anno_txt).expanduser()
            try:
                arp = str(ap.resolve())
                am = float(ap.stat().st_mtime) if ap.exists() else -1.0
            except Exception:
                arp = str(ap)
                am = -1.0
            anno_tag = f"|anno={arp}|anno_m={am:.6f}"

        cache_key = (
            "merge|"
            + "|".join(path_tokens)
            + f"|thr={thr:.3e}{anno_tag}"
        )
        with self._lock:
            cached = self._sig_cache.get(cache_key)
        if cached is None:
            sig = build_merged_sig_table(
                rows,
                threshold=thr,
                anno_file=anno_txt,
                annotate_rows=False,
            )
            with self._lock:
                self._sig_cache[cache_key] = dict(sig)
            data = sig
        else:
            data = dict(cached)

        rows_raw = data.get("rows", [])
        if not isinstance(rows_raw, list):
            rows_raw = []
        out_rows = list(rows_raw)

        def _p_rank_key(r: dict[str, Any]) -> tuple[float, str, int]:
            try:
                p = float(r.get("p", np.nan))
            except Exception:
                p = np.nan
            if (not np.isfinite(p)) or p <= 0.0:
                p = float("inf")
            chrom = str(r.get("chrom", ""))
            try:
                pos = int(r.get("pos", 0))
            except Exception:
                pos = 0
            return (p, chrom, pos)

        q = str(search or "").strip().lower()
        if q:
            if anno_txt != "":
                annotate_sig_rows_with_genes(out_rows, anno_txt)

            def _hit(r: dict[str, Any]) -> bool:
                txt = " ".join(
                    [
                        str(r.get("chrom", "")),
                        str(r.get("pos", "")),
                        str(r.get("p", "")),
                        str(r.get("Gene", "")),
                    ]
                ).lower()
                return q in txt

            out_rows = [r for r in out_rows if _hit(r)]
        else:
            # Default UI view: only show top 100 significant sites.
            out_rows = sorted(out_rows, key=_p_rank_key)[:100]
            if anno_txt != "":
                annotate_sig_rows_with_genes(out_rows, anno_txt)
        return {
            "history_ids": ids,
            "threshold": float(data.get("threshold", thr)),
            "n_sig": int(data.get("n_sig", 0)),
            "n_leads": int(data.get("n_leads", 0)),
            "n_show": int(len(out_rows)),
            "rows": out_rows,
        }

    def warmup_history(self, history_id: str, *, anno_file: str = "") -> dict[str, Any]:
        row = self.get_gwas_history(history_id)
        if row is None:
            raise ValueError(f"GWAS history not found: {history_id}")
        t0 = time.time()
        activate_task_cache(row, anno_file=str(anno_file or ""))
        gwas_path = self._resolve_history_result_path(row)
        dt = float(time.time() - t0)
        return {
            "history_id": str(history_id),
            "gwas_file": Path(gwas_path).name if gwas_path else "",
            "anno_file": Path(str(anno_file)).name if str(anno_file or "").strip() != "" else "",
            "elapsed_sec": round(dt, 3),
            "message": f"Loaded GWAS/annotation into memory in {dt:.2f}s",
        }

    def warmup_merged_history(self, history_ids: list[str], *, anno_file: str = "") -> dict[str, Any]:
        ids = [str(x).strip() for x in (history_ids or []) if str(x).strip() != ""]
        rows = self._resolve_merge_rows(ids)
        t0 = time.time()
        activate_task_cache(rows[0], anno_file=str(anno_file or ""))
        loaded: list[str] = []
        try:
            from .render import _load_sig_table_cached  # local import to avoid broad API change
        except Exception:
            _load_sig_table_cached = None  # type: ignore
        for r in rows:
            gwas_path = self._resolve_history_result_path(r)
            if gwas_path == "":
                continue
            loaded.append(Path(gwas_path).name)
            if _load_sig_table_cached is not None:
                try:
                    _load_sig_table_cached(Path(gwas_path))
                except Exception:
                    pass
        dt = float(time.time() - t0)
        return {
            "history_ids": ids,
            "n_tasks": int(len(rows)),
            "gwas_files": loaded,
            "anno_file": Path(str(anno_file)).name if str(anno_file or "").strip() != "" else "",
            "elapsed_sec": round(dt, 3),
            "message": f"Loaded {len(rows)} GWAS tables/annotation into memory in {dt:.2f}s",
        }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        if "/" in job_id or "\\" in job_id or ".." in job_id:
            return None
        with self._lock:
            meta = self._load_meta_unlocked(job_id)
        if meta is None:
            return None

        job_dir = self._job_dir(job_id)
        out_tail = _tail_text(job_dir / STDOUT_FILE)
        err_tail = _tail_text(job_dir / STDERR_FILE)
        files: list[dict[str, Any]] = []
        for p in sorted(job_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(job_dir).as_posix()
            if rel in {META_FILE, STDOUT_FILE, STDERR_FILE}:
                continue
            try:
                st = p.stat()
                files.append(
                    {
                        "name": rel,
                        "size": st.st_size,
                        "mtime": datetime.fromtimestamp(st.st_mtime).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "image": _looks_image(p),
                        "url": f"/api/jobs/{job_id}/files/{rel}",
                    }
                )
            except Exception:
                continue
        meta["stdout_tail"] = out_tail
        meta["stderr_tail"] = err_tail
        meta["files"] = files
        return meta

    def get_postgwas_job_by_history(self, history_id: str) -> dict[str, Any] | None:
        hid = str(history_id or "").strip()
        if hid == "":
            return None
        prow = get_postgwas_run(hid)
        if prow is None:
            return None
        job_id = str(prow.get("job_id", "")).strip()
        if job_id == "":
            return None
        meta = self.get_job(job_id)
        if meta is None:
            # Fallback when job metadata files were removed.
            return {
                "job_id": job_id,
                "name": f"postgwas-{hid}",
                "status": str(prow.get("status", "")),
                "created_at": str(prow.get("created_at", "")),
                "started_at": str(prow.get("started_at", "")),
                "finished_at": str(prow.get("finished_at", "")),
                "return_code": prow.get("return_code", None),
                "error": str(prow.get("error_text", "")),
                "stdout_tail": "",
                "stderr_tail": "",
                "files": [],
                "history_id": hid,
            }
        meta["history_id"] = hid
        return meta

    def _validate_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        errors: list[str] = []
        params: dict[str, Any] = {}
        params["source_history_id"] = str(payload.get("source_history_id", "")).strip()
        params["source_gwas_run_id"] = str(payload.get("source_gwas_run_id", "")).strip()
        params["source_phenotype"] = str(payload.get("source_phenotype", "")).strip()
        params["source_model"] = str(payload.get("source_model", "")).strip()
        params["source_genotype"] = str(payload.get("source_genotype", "")).strip()
        params["source_genotype_type"] = str(payload.get("source_genotype_type", "")).strip()

        raw_name = str(payload.get("name", "")).strip()
        params["name"] = raw_name or "postgwas"

        gwas_raw = payload.get("gwas_files", "")
        if isinstance(gwas_raw, list):
            gwas_files = [str(x).strip() for x in gwas_raw if str(x).strip()]
        else:
            gwas_files = [x.strip() for x in str(gwas_raw).splitlines() if x.strip()]
        if len(gwas_files) == 0:
            errors.append("gwas_files is required (one path per line).")
        else:
            bad = [p for p in gwas_files if not Path(p).exists()]
            if bad:
                errors.append(f"GWAS file not found: {bad[0]}")
        params["gwas_files"] = gwas_files

        params["chr"] = str(payload.get("chr", "chrom") or "chrom").strip()
        params["pos"] = str(payload.get("pos", "pos") or "pos").strip()
        params["pvalue"] = str(payload.get("pvalue", "pwald") or "pwald").strip()

        params["threshold"] = str(payload.get("threshold", "")).strip()

        params["manh"] = bool(payload.get("manh", True))
        params["manh_ratio"] = str(payload.get("manh_ratio", "2")).strip() or "2"
        params["qq"] = bool(payload.get("qq", True))

        fmt = str(payload.get("format", "png")).lower().strip()
        if fmt not in {"png", "pdf", "svg", "tif"}:
            errors.append("format must be one of: png, pdf, svg, tif.")
            fmt = "png"
        params["format"] = fmt

        prefix = str(payload.get("prefix", "JanusX")).strip() or "JanusX"
        params["prefix"] = prefix

        thread_raw = str(payload.get("thread", "-1")).strip()
        try:
            params["thread"] = int(thread_raw)
        except Exception:
            errors.append("thread must be an integer.")
            params["thread"] = -1

        anno = str(payload.get("anno", "")).strip()
        if anno:
            if not Path(anno).exists():
                errors.append(f"Annotation file not found: {anno}")
            params["anno"] = anno
        else:
            params["anno"] = ""

        ab = str(payload.get("annobroaden", "")).strip()
        if ab:
            try:
                float(ab)
                params["annobroaden"] = ab
            except Exception:
                errors.append("annobroaden must be numeric.")
                params["annobroaden"] = ""
        else:
            params["annobroaden"] = ""

        geno = str(payload.get("genofile", "")).strip()
        geno_kind = str(payload.get("genofile_kind", "bfile")).strip().lower()
        if geno_kind not in {"bfile", "vcf", "file", "tsv", "txt", "csv", "npy"}:
            geno_kind = "bfile"
        if geno:
            if geno_kind == "bfile":
                bed = Path(f"{geno}.bed")
                bim = Path(f"{geno}.bim")
                fam = Path(f"{geno}.fam")
                if not (bed.exists() and bim.exists() and fam.exists()):
                    errors.append(
                        f"PLINK prefix not found or incomplete: {geno} (.bed/.bim/.fam required)"
                    )
            elif not Path(geno).exists():
                errors.append(f"Genotype file not found: {geno}")
        params["genofile"] = geno
        params["genofile_kind"] = geno_kind

        bimranges = _normalize_bimrange_items(payload.get("bimrange", ""), max_items=4)
        params["bimranges"] = bimranges
        params["bimrange"] = bimranges[0] if len(bimranges) > 0 else ""

        return params, errors

    def create_job(
        self,
        payload: dict[str, Any],
        *,
        fixed_job_id: str | None = None,
        replace_existing: bool = False,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        params, errors = self._validate_payload(payload)
        if errors:
            return None, errors

        job_id = str(fixed_job_id or "").strip() or _new_job_id(str(params.get("name", "postgwas")))
        job_dir = self._job_dir(job_id)
        if job_dir.exists():
            if not replace_existing:
                return None, [f"Job already exists: {job_id}"]
            shutil.rmtree(job_dir, ignore_errors=True)
        job_dir.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [
            sys.executable,
            "-m",
            "janusx.script.postgwas",
            "-gwasfile",
            *params["gwas_files"],
            "-chr",
            params["chr"],
            "-pos",
            params["pos"],
            "-pvalue",
            params["pvalue"],
            "-format",
            params["format"],
            "-o",
            str(job_dir),
            "-prefix",
            params["prefix"],
            "-t",
            str(params["thread"]),
        ]

        if params["threshold"] != "":
            cmd.extend(["-thr", params["threshold"]])
        if params["manh"]:
            cmd.extend(["-manh", params["manh_ratio"]])
        if params["qq"]:
            cmd.append("-qq")
        if params["anno"] != "":
            cmd.extend(["-a", params["anno"]])
        if params["annobroaden"] != "":
            cmd.extend(["-ab", params["annobroaden"]])
        if params["genofile"] != "":
            flag = {"bfile": "-bfile", "vcf": "-vcf"}.get(params["genofile_kind"], "-file")
            cmd.extend([flag, params["genofile"]])
        for br in params.get("bimranges", []):
            if str(br).strip() != "":
                cmd.extend(["-bimrange", str(br).strip()])

        meta: dict[str, Any] = {
            "job_id": job_id,
            "name": params["name"],
            "kind": "postgwas",
            "status": "queued",
            "created_at": _now_str(),
            "started_at": "",
            "finished_at": "",
            "return_code": None,
            "command": cmd,
            "params": params,
            "job_dir": str(job_dir),
            "stdout_log": STDOUT_FILE,
            "stderr_log": STDERR_FILE,
            "error": "",
        }

        with self._lock:
            self._save_meta_unlocked(job_id, meta)

        src_hid = str(params.get("source_history_id", "")).strip()
        if src_hid:
            try:
                upsert_postgwas_run(
                    history_id=src_hid,
                    run_id=str(params.get("source_gwas_run_id", "")),
                    phenotype=str(params.get("source_phenotype", "")),
                    model=str(params.get("source_model", "")),
                    genotype=str(params.get("source_genotype", "")),
                    genotype_type=str(params.get("source_genotype_type", "")),
                    job_id=job_id,
                    status="queued",
                    created_at=meta["created_at"],
                    started_at="",
                    finished_at="",
                    return_code=None,
                    job_dir=str(job_dir),
                    error_text="",
                )
            except Exception:
                pass

        th = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        th.start()
        return meta, []

    def create_job_from_gwas_history(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, list[str]]:
        history_id = str(payload.get("history_id", "")).strip()
        if history_id == "":
            return None, ["history_id is required."]
        run_id = str(payload.get("run_id", "")).strip()
        lookup_id = history_id
        row = self.get_gwas_history(lookup_id)
        if row is None:
            return None, [f"GWAS history not found: {lookup_id}"]

        base_dirs: list[Path] = []
        for key, use_parent in (
            ("outprefix", True),
            ("log_file", True),
            ("phenofile", True),
            ("genofile", True),
        ):
            v = str(row.get(key, "")).strip()
            if v == "":
                continue
            p = Path(v).expanduser()
            bd = p.parent if use_parent else p
            if bd not in base_dirs:
                base_dirs.append(bd)
        cwd = Path.cwd().resolve()
        if cwd not in base_dirs:
            base_dirs.append(cwd)

        selected_result_file = str(payload.get("result_file", "")).strip()
        if selected_result_file:
            g = _resolve_existing_path(selected_result_file, base_dirs)
            gwas_files = [g] if g else []
        else:
            # Fallback to run-level result list.
            result_files_raw = row.get("result_files", [])
            if not isinstance(result_files_raw, list):
                result_files_raw = []
            gwas_files = []
            for p in result_files_raw:
                rp = _resolve_existing_path(str(p).strip(), base_dirs)
                if rp:
                    gwas_files.append(rp)
            # Row-level result_file fallback.
            row_result = str(row.get("result_file", "")).strip()
            if row_result:
                rr = _resolve_existing_path(row_result, base_dirs)
                if rr:
                    gwas_files = [rr]
        # Deduplicate while preserving order.
        dedup: list[str] = []
        seen: set[str] = set()
        for p in gwas_files:
            if p not in seen:
                seen.add(p)
                dedup.append(p)
        gwas_files = dedup
        if len(gwas_files) == 0:
            row_result = str(row.get("result_file", "")).strip()
            return None, [
                f"No valid GWAS result files found for: {lookup_id}. "
                f"history result_file={row_result or 'NA'}"
            ]

        anno = str(payload.get("anno", "")).strip()

        run_outprefix = str(row.get("outprefix", "")).strip()
        run_prefix = Path(run_outprefix).name if run_outprefix else "JanusX"
        prefix_default = f"{run_prefix}.postgwas"

        genofile = str(row.get("genofile", "")).strip()
        genofile_kind = str(row.get("genofile_kind", "bfile")).strip().lower()
        if genofile_kind not in {"bfile", "vcf", "file", "tsv", "txt", "csv", "npy"}:
            genofile_kind = "bfile"

        # Genotype is optional for normal postgwas plotting/annotation.
        # If history path is broken, continue without genotype instead of failing.
        if genofile != "":
            if genofile_kind == "bfile":
                if not (
                    Path(f"{genofile}.bed").exists()
                    and Path(f"{genofile}.bim").exists()
                    and Path(f"{genofile}.fam").exists()
                ):
                    genofile = ""
            elif not Path(genofile).exists():
                genofile = ""

        job_name = f"postgwas-{run_id or history_id}"
        new_payload: dict[str, Any] = {
            "name": job_name,
            "gwas_files": gwas_files,
            "chr": "chrom",
            "pos": "pos",
            "pvalue": "pwald",
            "threshold": "",
            "manh": True,
            "manh_ratio": "2",
            "qq": True,
            "format": "svg",
            "thread": "-1",
            "prefix": prefix_default,
            "anno": anno,
            "annobroaden": "50" if anno else "",
            "genofile": genofile,
            "genofile_kind": genofile_kind,
            "bimrange": _normalize_bimrange_items(payload.get("bimrange", ""), max_items=4),
            "source_history_id": str(history_id or ""),
            "source_gwas_run_id": str(row.get("run_id", run_id or history_id)),
            "source_phenotype": str(row.get("phenotype", "")),
            "source_model": str(row.get("model", "")),
            "source_genotype": str(row.get("genotype", "")),
            "source_genotype_type": str(row.get("genotype_type", "")),
        }
        hid_safe = _safe_job_name(str(history_id or run_id).replace("|", "-"))
        fixed_job_id = f"postgwas-{hid_safe}"
        return self.create_job(
            new_payload,
            fixed_job_id=fixed_job_id,
            replace_existing=True,
        )

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            meta = self._load_meta_unlocked(job_id)
            if meta is None:
                return
            meta["status"] = "running"
            meta["started_at"] = _now_str()
            self._save_meta_unlocked(job_id, meta)
        params0 = meta.get("params", {}) if isinstance(meta, dict) else {}
        src_hid0 = str(params0.get("source_history_id", "")).strip() if isinstance(params0, dict) else ""
        if src_hid0:
            try:
                upsert_postgwas_run(
                    history_id=src_hid0,
                    run_id=str(params0.get("source_gwas_run_id", "")),
                    phenotype=str(params0.get("source_phenotype", "")),
                    model=str(params0.get("source_model", "")),
                    genotype=str(params0.get("source_genotype", "")),
                    genotype_type=str(params0.get("source_genotype_type", "")),
                    job_id=job_id,
                    status="running",
                    created_at=str(meta.get("created_at", "")),
                    started_at=str(meta.get("started_at", "")),
                    finished_at="",
                    return_code=None,
                    job_dir=str(self._job_dir(job_id)),
                    error_text="",
                )
            except Exception:
                pass

        job_dir = self._job_dir(job_id)
        cmd = list(meta["command"])
        out_path = job_dir / STDOUT_FILE
        err_path = job_dir / STDERR_FILE
        rc = -1
        err_msg = ""
        try:
            with out_path.open("w", encoding="utf-8") as out_f, err_path.open(
                "w", encoding="utf-8"
            ) as err_f:
                proc = subprocess.run(
                    cmd,
                    stdout=out_f,
                    stderr=err_f,
                    cwd=str(Path.cwd()),
                    text=True,
                )
                rc = int(proc.returncode)
        except Exception as exc:
            err_msg = str(exc)
            rc = -1

        with self._lock:
            meta2 = self._load_meta_unlocked(job_id)
            if meta2 is None:
                return
            meta2["return_code"] = rc
            meta2["finished_at"] = _now_str()
            if rc == 0:
                meta2["status"] = "done"
                meta2["error"] = ""
            else:
                meta2["status"] = "failed"
                if err_msg:
                    meta2["error"] = err_msg
            self._save_meta_unlocked(job_id, meta2)
        params2 = meta2.get("params", {}) if isinstance(meta2, dict) else {}
        src_hid2 = str(params2.get("source_history_id", "")).strip() if isinstance(params2, dict) else ""
        if src_hid2:
            try:
                upsert_postgwas_run(
                    history_id=src_hid2,
                    run_id=str(params2.get("source_gwas_run_id", "")),
                    phenotype=str(params2.get("source_phenotype", "")),
                    model=str(params2.get("source_model", "")),
                    genotype=str(params2.get("source_genotype", "")),
                    genotype_type=str(params2.get("source_genotype_type", "")),
                    job_id=job_id,
                    status=str(meta2.get("status", "")),
                    created_at=str(meta2.get("created_at", "")),
                    started_at=str(meta2.get("started_at", "")),
                    finished_at=str(meta2.get("finished_at", "")),
                    return_code=None if meta2.get("return_code", None) is None else int(meta2["return_code"]),
                    job_dir=str(self._job_dir(job_id)),
                    error_text=str(meta2.get("error", "")),
                )
            except Exception:
                pass

    def resolve_job_file(self, job_id: str, rel: str) -> Path | None:
        if "/" in job_id or "\\" in job_id or ".." in job_id:
            return None
        rel_decoded = unquote(rel).lstrip("/")
        if rel_decoded == "":
            return None
        target = (self._job_dir(job_id) / rel_decoded).resolve()
        job_dir = self._job_dir(job_id)
        try:
            target.relative_to(job_dir)
        except ValueError:
            return None
        if not target.exists() or not target.is_file():
            return None
        return target


def _html_page() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>JanusX WebUI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { --bg:#f4f7fb; --card:#fff; --line:#dbe3ef; --text:#0f172a; --muted:#4b5563; --accent:#0ea5e9; --ok:#16a34a; --err:#dc2626; }
    * { box-sizing: border-box; }
    html, body { height:100%; overflow:hidden; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Ubuntu,sans-serif; background:var(--bg); color:var(--text); }
    .wrap { max-width: 1800px; margin: 0 auto; padding: 14px; height: 100%; display:flex; flex-direction:column; overflow:hidden; }
    .grid { display:grid; grid-template-columns: minmax(360px, 1fr) minmax(720px, 2fr); gap: 12px; align-items:stretch; flex:1 1 auto; min-height: 0; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding: 12px; height: 100%; min-height: 0; }
    .left-card { display:flex; flex-direction:column; min-height:0; }
    .left-top { flex:1 1 50%; min-height:0; display:flex; flex-direction:column; overflow:auto; }
    .left-bottom { flex:1 1 50%; min-height:0; display:flex; flex-direction:column; margin-top:10px; border-top:1px solid var(--line); padding-top:10px; overflow:hidden; }
    .right-card { display:flex; flex-direction:column; min-height:0; }
    .viz-pane, .ctrl-pane { min-height:0; display:flex; flex-direction:column; }
    .viz-pane { flex:1 1 auto; }
    h1 { margin:0 0 8px; font-size: 20px; }
    h2 { margin:10px 0 6px; font-size: 14px; }
    .sub { color: var(--muted); margin-bottom: 10px; font-size: 12px; }
    .toolbar { display:flex; gap:6px; align-items:center; margin-bottom: 8px; }
    .toolbar > * { min-width: 0; }
    .toolbar input, .toolbar select, .toolbar button { height: 34px; }
    input, select { width:100%; border:1px solid var(--line); border-radius:8px; padding:7px 9px; font-size:13px; }
    button { border:0; background:var(--accent); color:#fff; font-weight:600; border-radius:8px; padding:8px 10px; cursor:pointer; white-space:nowrap; }
    button:disabled { opacity:.65; cursor:not-allowed; }
    .muted { color:var(--muted); font-size:12px; }
    .status.done { color: var(--ok); font-weight:700; }
    .status.failed { color: var(--err); font-weight:700; }
    table { width:100%; border-collapse: collapse; font-size:12px; }
    th, td { border-bottom:1px solid var(--line); text-align:left; padding: 7px 6px; }
    tr.sel { background:#e0f2fe; }
    tr:hover { background:#f8fafc; }
    .composite-wrap { border:1px solid var(--line); border-radius:8px; background:#fff; padding:6px; overflow:hidden; flex:1 1 auto; min-height:0; position:relative; }
    .preview-canvas { width:100%; height:100%; transform-origin: 0 0; cursor: grab; user-select:none; }
    .preview-canvas.dragging { cursor: grabbing; }
    .preview-canvas svg { width:100%; height:auto; display:block; background:#fff; pointer-events:none; }
    .viz-toolbar { display:none; justify-content: space-between; gap:8px; margin: 0 0 8px; }
    .viz-toolbar .tools { display:flex; gap:6px; align-items:center; }
    .viz-toolbar button { height:30px; padding: 5px 10px; }
    #preview_meta { display:block; }
    .history-wrap { flex:1 1 auto; min-height:140px; overflow-x:auto; overflow-y:auto; margin-top:8px; border:1px solid var(--line); border-radius:8px; }
    .sig-wrap { flex:1 1 auto; min-height:0; overflow-x:auto; overflow-y:auto; margin-top:8px; border:1px solid var(--line); border-radius:8px; }
    .history-wrap table, .sig-wrap table { width:max-content; min-width:100%; table-layout:auto; }
    .history-wrap th, .history-wrap td, .sig-wrap th, .sig-wrap td { white-space:nowrap; }
    .sig-footer { margin-top:8px; margin-bottom:0; justify-content:flex-end; align-items:flex-end; gap:8px; }
    .sig-actions { margin-top:8px; margin-bottom:0; justify-content:flex-end; align-items:flex-end; gap:8px; }
    .sig-actions #sig_msg { flex:1 1 auto; }
    .anno-card { margin-top:10px; border:1px solid var(--line); border-radius:8px; padding:8px; background:#fff; }
    .anno-card h2 { margin:0 0 6px; font-size:13px; }
    .ctrl-actions { margin-top:auto; justify-content:flex-end; align-items:flex-end; gap:8px; margin-bottom:0; }
    .params-card {
      display:flex;
      flex-direction:column;
      flex:1 1 auto;
      min-height:0;
    }
    .left-bottom .params-card .style-scroll {
      flex:1 1 auto;
      min-height:140px;
      max-height:none;
    }
    .style-scroll {
      margin-top:2px;
      border:1px solid var(--line);
      border-radius:8px;
      padding:6px;
      max-height:160px;
      min-height:52px;
      overflow-y:auto;
      overflow-x:hidden;
      background:#fff;
    }
    .style-grid {
      display:grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap:6px;
    }
    .style-card {
      border:1px solid var(--line);
      border-radius:8px;
      padding:6px;
      background:#f8fafc;
      min-width:0;
      display:flex;
      flex-direction:column;
      gap:5px;
    }
    .style-head {
      color: var(--muted);
      font-size:12px;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
      flex:1 1 auto;
      min-width:0;
    }
    .style-head-row {
      display:flex;
      align-items:center;
      gap:6px;
      min-width:0;
    }
    .style-close {
      flex:0 0 auto;
      width:20px;
      height:20px;
      padding:0;
      border-radius:50%;
      border:1px solid #cbd5e1;
      background:#ffffff;
      color:#64748b;
      font-size:14px;
      line-height:18px;
      font-weight:700;
      cursor:pointer;
    }
    .style-close:hover {
      border-color:#94a3b8;
      color:#334155;
    }
    .style-ctrl {
      display:flex;
      align-items:center;
      gap:4px;
      min-width:0;
    }
    .style-ctrl input, .style-ctrl select { height:30px; }
    .action-btn { background:#334155 !important; color:#fff !important; border:0; width:auto; min-width:96px; font-weight:600; font-size:13px; letter-spacing:0; }
    .search-box { position:relative; flex:3 1 50%; width:50%; }
    .search-box .search-icon {
      position:absolute;
      left:9px;
      top:50%;
      transform:translateY(-50%);
      color:#64748b;
      font-size:13px;
      line-height:1;
      pointer-events:none;
    }
    .search-box input { width:100%; padding-left:26px; }
    .composite-svg { width:100%; height:auto; display:block; background:#fff; }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      .style-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>JanusX WebUI</h1>
    <div class="sub">PostGWAS preview</div>
    <div class="grid">
      <div class="card left-card">
        <div class="left-top">
          <h2>GWAS History</h2>
          <div class="toolbar" style="margin-bottom:6px;">
            <input id="gwas_upload_file" type="file" multiple style="flex:1 1 auto;" />
            <button id="gwas_upload_btn" class="action-btn" style="min-width:110px;">Upload GWAS</button>
          </div>
          <div id="upload_msg" class="muted" style="margin-bottom:6px;"></div>
          <input id="search_box" autocomplete="off" placeholder="fuzzy search genotype/phenotype/model/type/grm/qcov/cov/date"/>
          <div id="history_msg" class="muted" style="margin-top:6px;">history loading...</div>
          <div class="history-wrap">
            <table>
              <thead><tr><th>Sel</th><th>Genotype</th><th>Phenotype</th><th>Model</th><th>Type</th><th>GRM</th><th>QCov</th><th>Cov</th><th>Date</th></tr></thead>
              <tbody id="history_tbody"></tbody>
            </table>
          </div>
          <div id="left_msg" class="muted" style="margin-top:8px;"></div>
        </div>
        <div class="left-bottom">
          <div class="anno-card params-card" style="margin-top:0;">
            <h2>Realtime Parameters</h2>
            <select id="anno_sel" style="display:none;">
              <option value="">(No annotation)</option>
            </select>
            <span id="anno_msg" class="muted" style="display:none;"></span>
            <div class="toolbar" style="margin-top:2px; gap:4px; align-items:center; flex-wrap:wrap;">
              <span class="muted" style="flex:0 0 auto; margin-right:2px;">aspect ratio</span>
              <input
                id="manh_ratio"
                type="text"
                value="2"
                title="Manhattan width/height ratio"
                placeholder="2"
                style="flex:0 0 72px; width:72px; height:34px; padding:4px 6px; border:1px solid #334155; border-radius:6px; background:#0f172a; color:#e2e8f0;"
              />
              <span class="muted" style="flex:0 0 auto; margin-right:2px;">thr:</span>
              <input
                id="sig_thr"
                type="text"
                value="auto"
                title="Significance threshold for Manhattan line (P)"
                placeholder="auto"
                style="flex:0 0 68px; width:68px; height:34px; padding:4px 6px; border:1px solid #334155; border-radius:6px; background:#0f172a; color:#e2e8f0;"
              />
              <input
                id="ld_color"
                type="color"
                value="#4b5563"
                title="LD/Gene color"
                style="flex:0 0 34px; width:34px; height:34px; padding:0; border:1px solid #334155; border-radius:6px; background:#0f172a;"
              />
            </div>
            <div class="toolbar" style="gap:4px; align-items:center; flex-wrap:wrap;">
              <select id="bim_chr1" style="flex:1 1 8%;">
                <option value="1">1</option>
              </select>
              <input id="bim_pos1" style="flex:1 1 16%;" placeholder="pos:pos"/>
              <select id="bim_chr2" style="flex:1 1 8%;">
                <option value="1">1</option>
              </select>
              <input id="bim_pos2" style="flex:1 1 16%;" placeholder="pos:pos"/>
              <select id="bim_chr3" style="flex:1 1 8%;">
                <option value="1">1</option>
              </select>
              <input id="bim_pos3" style="flex:1 1 16%;" placeholder="pos:pos"/>
            </div>
            <div class="style-scroll">
              <div id="series_style_rows" class="style-grid"></div>
            </div>
            <div id="bim_msg" class="muted"></div>
            <div class="toolbar ctrl-actions">
              <button id="bim_apply_btn" class="action-btn">Visualizing</button>
              <button id="download_btn" class="action-btn">Download</button>
            </div>
          </div>
          <input id="sig_search" type="hidden" value=""/>
          <div id="sig_tbody" style="display:none;"></div>
          <span id="sig_msg" class="muted" style="display:none;"></span>
          <button id="sig_copy_btn" class="action-btn" style="display:none;" disabled>Copy</button>
        </div>
      </div>

      <div class="card right-card">
        <div class="viz-pane">
          <div id="preview_meta" class="muted" style="margin-bottom:8px;"></div>
          <div class="toolbar viz-toolbar">
            <div class="tools">
              <button id="zoom_out_btn" style="background:#334155;">-</button>
              <button id="zoom_in_btn" style="background:#334155;">+</button>
              <button id="zoom_reset_btn" style="background:#334155;">Reset</button>
              <button id="drag_toggle_btn" style="background:#334155;">Drag: On</button>
            </div>
            <span id="zoom_text" class="muted">100%</span>
          </div>
          <div id="img_preview" class="composite-wrap"><div class="muted">Waiting for selection...</div></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    function _setText(id, text) {
      const el = document.getElementById(id);
      if (el) el.textContent = String(text == null ? "" : text);
    }
    function _bindClick(id, fn) {
      const el = document.getElementById(id);
      if (el) el.onclick = fn;
    }
    function _bindInput(id, ev, fn) {
      const el = document.getElementById(id);
      if (el) el.addEventListener(ev, fn);
    }
    window.addEventListener("error", (ev) => {
      const msg = `UI error: ${String(ev && ev.message ? ev.message : "unknown")}`;
      _setText("history_msg", msg);
      _setText("left_msg", msg);
    });
    window.addEventListener("unhandledrejection", (ev) => {
      const reason = ev && ev.reason ? ev.reason : "unknown";
      const msg = `UI promise error: ${String(reason)}`;
      _setText("history_msg", msg);
      _setText("left_msg", msg);
    });

    let gwasRows = [];
    let selectedHistoryId = "";
    let selectedHistoryIds = [];
    let selectedRow = null;
    let selectedSigRange = null;
    let lastSigRows = [];
    let lastSigSummary = "";
    let pollTimer = null;
    let sigTimer = null;
    let styleRefreshTimer = null;
    let seriesStyleMap = {};
    const SPIN_FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"];
    const STYLE_DEFAULT_COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];
    const STYLE_MARKERS = ["o",".","s","^","v","D","x","+","*"];
    const viewer = { scale: 1.0, tx: 0.0, ty: 0.0, dragging: false, dragEnabled: true, x: 0.0, y: 0.0 };
    function _markStartInvalidated() {
      // no-op: start/warmup button was removed
    }
    function _normHistoryIds(ids) {
      const arr = Array.isArray(ids) ? ids : [];
      const uniq = Array.from(new Set(arr.map((x) => String(x || "").trim()).filter((x) => x !== "")));
      uniq.sort();
      return uniq;
    }
    function _isStartedForCurrent() {
      return _normHistoryIds(selectedHistoryIds).length > 0;
    }
    function _isMergeMode() {
      return Array.isArray(selectedHistoryIds) && selectedHistoryIds.length > 1;
    }
    function _rowGenotypeKey(row) {
      if (!row) return "";
      const g = String(row.genofile || "").trim();
      const k = String(row.genofile_kind || "").trim().toLowerCase();
      if (g) return `${k}:${g}`;
      const gg = String(row.genotype || "").trim();
      return `${k}:${gg}`;
    }
    function _syncPrimarySelection() {
      if (!Array.isArray(selectedHistoryIds)) selectedHistoryIds = [];
      selectedHistoryIds = selectedHistoryIds.filter((x) => String(x || "").trim() !== "");
      if (selectedHistoryIds.length === 0) {
        selectedHistoryId = "";
        selectedRow = null;
        return;
      }
      selectedHistoryId = String(selectedHistoryIds[0] || "");
      selectedRow = gwasRows.find((r) => String(r.history_id || "") === selectedHistoryId) || null;
    }
    function _toggleHistoryRow(row, forceChecked=null) {
      const hid = String((row && row.history_id) || "");
      if (!hid) return;
      const idx = selectedHistoryIds.indexOf(hid);
      let wantOn = idx < 0;
      if (forceChecked !== null) wantOn = Boolean(forceChecked);
      if (!wantOn && idx >= 0) {
        selectedHistoryIds.splice(idx, 1);
      } else if (wantOn && idx < 0) {
        // Merge mode requires same genotype source.
        if (selectedHistoryIds.length > 0) {
          const ref = gwasRows.find((x) => String(x.history_id || "") === String(selectedHistoryIds[0] || ""));
          if (ref && _rowGenotypeKey(ref) !== _rowGenotypeKey(row)) {
            _setText("left_msg", "Merge mode requires the same genotype file.");
            return;
          }
        }
        selectedHistoryIds.push(hid);
      }
      selectedSigRange = null;
      _markStartInvalidated();
      _syncPrimarySelection();
      renderHistoryTable();
      renderSeriesStyleRows();
      loadChromOptionsForSelection();
      const preview = document.getElementById("img_preview");
      if (_isMergeMode()) {
        if (preview) preview.innerHTML = '<div class="muted">Select tasks and click Visualizing.</div>';
        _setText("sig_msg", "Click Visualizing to load merged GWAS/annotation.");
        _setText("anno_msg", `${selectedHistoryIds.length} GWAS tasks selected.`);
      } else {
        if (preview) preview.innerHTML = '<div class="muted">Select task and click Visualizing.</div>';
        _setText("sig_msg", "Click Visualizing to load selected task.");
        _setText("anno_msg", "Ready.");
      }
      renderSigTable([]);
    }

    function _rowByHistoryId(hid) {
      const key = String(hid || "").trim();
      if (key === "") return null;
      return gwasRows.find((r) => String(r.history_id || "") === key) || null;
    }

    function _currentStyleHistoryIds() {
      if (_isMergeMode()) {
        return (Array.isArray(selectedHistoryIds) ? selectedHistoryIds : [])
          .map((x) => String(x || "").trim())
          .filter((x) => x !== "");
      }
      const hid = String(selectedHistoryId || "").trim();
      return hid ? [hid] : [];
    }

    function _defaultStyleByIndex(idx) {
      return {
        color: STYLE_DEFAULT_COLORS[idx % STYLE_DEFAULT_COLORS.length],
        alpha: 0.7,
        marker: "o",
        size: 16,
      };
    }

    function _ensureStyleFor(hid, idx) {
      const key = String(hid || "").trim();
      if (key === "") return _defaultStyleByIndex(idx);
      if (!seriesStyleMap[key]) {
        seriesStyleMap[key] = _defaultStyleByIndex(idx);
      }
      const st = seriesStyleMap[key] || {};
      const sizeNum = Number(st.size);
      return {
        color: String(st.color || _defaultStyleByIndex(idx).color),
        alpha: Number.isFinite(Number(st.alpha)) ? Math.max(0, Math.min(1, Number(st.alpha))) : 0.7,
        marker: String(st.marker || "o"),
        size: Number.isFinite(sizeNum) && sizeNum > 0 ? sizeNum : 16,
      };
    }

    function _collectSeriesStyles() {
      const ids = _currentStyleHistoryIds();
      return ids.map((hid, idx) => {
        const st = _ensureStyleFor(hid, idx);
        return {
          history_id: String(hid),
          index: Number(idx),
          color: String(st.color || ""),
          alpha: Number.isFinite(Number(st.alpha)) ? Math.max(0, Math.min(1, Number(st.alpha))) : 0.7,
          marker: String(st.marker || "o"),
          size: Number(st.size || 16),
        };
      });
    }

    function _readManhRatio() {
      const el = document.getElementById("manh_ratio");
      const raw = el ? String(el.value || "").trim() : "";
      const v = Number(raw);
      if (!Number.isFinite(v) || v <= 0) {
        if (el) el.value = "2";
        return "2";
      }
      const out = String(v);
      if (el) el.value = out;
      return out;
    }

    function _styleLabelFor(hid, idx) {
      const row = _rowByHistoryId(hid);
      const base = `gwas${idx + 1}`;
      if (!row) return base;
      const ph = String(row.phenotype || "").trim();
      return ph ? `${base}: ${ph}` : base;
    }

    function _triggerStyleRefresh() {
      if (!_isStartedForCurrent()) return;
      if (styleRefreshTimer) clearTimeout(styleRefreshTimer);
      styleRefreshTimer = setTimeout(() => {
        refreshAllByHistory(selectedHistoryId).catch((e) => {
          _setText("bim_msg", String(e));
        });
      }, 150);
    }

    function renderSeriesStyleRows() {
      const box = document.getElementById("series_style_rows");
      if (!box) return;
      const ids = _currentStyleHistoryIds();
      box.innerHTML = "";
      if (ids.length === 0) {
        box.innerHTML = '<div class="muted" style="grid-column:1 / -1;">Select GWAS task(s) to adjust style.</div>';
        return;
      }
      ids.forEach((hid, idx) => {
        const st = _ensureStyleFor(hid, idx);
        const card = document.createElement("div");
        card.className = "style-card";

        const head = document.createElement("div");
        head.className = "style-head-row";

        const lab = document.createElement("span");
        lab.className = "style-head";
        lab.textContent = _styleLabelFor(hid, idx);
        head.appendChild(lab);

        const closeBtn = document.createElement("button");
        closeBtn.type = "button";
        closeBtn.className = "style-close";
        closeBtn.title = "Remove this GWAS from current selection";
        closeBtn.textContent = "×";
        closeBtn.addEventListener("click", () => {
          const rowNow = _rowByHistoryId(hid);
          if (rowNow) {
            _toggleHistoryRow(rowNow, false);
            return;
          }
          selectedHistoryIds = (Array.isArray(selectedHistoryIds) ? selectedHistoryIds : [])
            .map((x) => String(x || "").trim())
            .filter((x) => x !== "" && x !== String(hid || "").trim());
          _syncPrimarySelection();
          renderHistoryTable();
          renderSeriesStyleRows();
          loadChromOptionsForSelection();
        });
        head.appendChild(closeBtn);
        card.appendChild(head);

        const ctrl = document.createElement("div");
        ctrl.className = "style-ctrl";

        const c = document.createElement("input");
        c.type = "color";
        c.value = String(st.color || "#1f77b4");
        c.title = "Color";
        c.style.flex = "0 0 34px";
        c.style.width = "34px";
        c.style.height = "34px";
        c.style.padding = "0";
        c.style.border = "1px solid #334155";
        c.style.borderRadius = "6px";
        c.style.background = "#0f172a";
        c.addEventListener("input", () => {
          const cur = _ensureStyleFor(hid, idx);
          cur.color = String(c.value || cur.color || "#1f77b4");
          seriesStyleMap[String(hid)] = cur;
          _triggerStyleRefresh();
        });
        ctrl.appendChild(c);

        const a = document.createElement("input");
        a.type = "text";
        a.value = String(
          Number.isFinite(Number(st.alpha)) ? Math.max(0, Math.min(1, Number(st.alpha))) : 0.7
        );
        a.placeholder = "alpha";
        a.title = "Alpha (0-1)";
        a.style.flex = "0 0 62px";
        a.style.width = "62px";
        a.style.height = "34px";
        a.style.padding = "4px 6px";
        a.style.border = "1px solid #334155";
        a.style.borderRadius = "6px";
        a.style.background = "#0f172a";
        a.style.color = "#e2e8f0";
        const _commitAlpha = () => {
          const cur = _ensureStyleFor(hid, idx);
          const v = Number(a.value);
          cur.alpha = (Number.isFinite(v)) ? Math.max(0, Math.min(1, v)) : 0.7;
          a.value = String(cur.alpha);
          seriesStyleMap[String(hid)] = cur;
          _triggerStyleRefresh();
        };
        a.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter") _commitAlpha();
        });
        a.addEventListener("blur", _commitAlpha);
        ctrl.appendChild(a);

        const mk = document.createElement("select");
        mk.title = "Marker";
        mk.style.flex = "0 0 60px";
        STYLE_MARKERS.forEach((m) => {
          const op = document.createElement("option");
          op.value = m;
          op.textContent = m;
          if (String(st.marker || "o") === m) op.selected = true;
          mk.appendChild(op);
        });
        mk.addEventListener("change", () => {
          const cur = _ensureStyleFor(hid, idx);
          cur.marker = String(mk.value || "o");
          seriesStyleMap[String(hid)] = cur;
          _triggerStyleRefresh();
        });
        ctrl.appendChild(mk);

        const sz = document.createElement("input");
        sz.type = "text";
        sz.value = String(st.size || 16);
        sz.placeholder = "size";
        sz.title = "Marker size";
        sz.style.flex = "0 0 58px";
        sz.style.width = "58px";
        sz.style.height = "34px";
        sz.style.padding = "4px 6px";
        sz.style.border = "1px solid #334155";
        sz.style.borderRadius = "6px";
        sz.style.background = "#0f172a";
        sz.style.color = "#e2e8f0";
        const _commitSize = () => {
          const cur = _ensureStyleFor(hid, idx);
          const v = Number(sz.value);
          cur.size = (Number.isFinite(v) && v > 0) ? v : 16;
          sz.value = String(cur.size);
          seriesStyleMap[String(hid)] = cur;
          _triggerStyleRefresh();
        };
        sz.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter") _commitSize();
        });
        sz.addEventListener("blur", _commitSize);
        ctrl.appendChild(sz);

        card.appendChild(ctrl);
        box.appendChild(card);
      });
    }

    function esc(s) {
      return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }
    async function api(path, method="GET", body=null, timeoutMs=0) {
      const init = { method, headers: {} };
      if (body !== null) {
        init.headers["Content-Type"] = "application/json";
        init.body = JSON.stringify(body);
      }
      let timer = null;
      let ctl = null;
      if (Number(timeoutMs) > 0 && typeof AbortController !== "undefined") {
        ctl = new AbortController();
        init.signal = ctl.signal;
        timer = setTimeout(() => {
          try { ctl.abort(); } catch (_e) {}
        }, Number(timeoutMs));
      }
      let r = null;
      try {
        r = await fetch(path, init);
      } finally {
        if (timer) clearTimeout(timer);
      }
      const t = await r.text();
      let j = {};
      try { j = JSON.parse(t || "{}"); } catch (_e) { j = {error: t}; }
      if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
      return j;
    }

    async function apiForm(path, formData, timeoutMs=0) {
      const init = { method: "POST", body: formData };
      let timer = null;
      let ctl = null;
      if (Number(timeoutMs) > 0 && typeof AbortController !== "undefined") {
        ctl = new AbortController();
        init.signal = ctl.signal;
        timer = setTimeout(() => {
          try { ctl.abort(); } catch (_e) {}
        }, Number(timeoutMs));
      }
      let r = null;
      try {
        r = await fetch(path, init);
      } finally {
        if (timer) clearTimeout(timer);
      }
      const t = await r.text();
      let j = {};
      try { j = JSON.parse(t || "{}"); } catch (_e) { j = {error: t}; }
      if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
      return j;
    }

    async function _withButtonSpinner(buttonId, busyLabel, idleLabel, workFn) {
      const btn = document.getElementById(String(buttonId || ""));
      let timer = null;
      let k = 0;
      const busy = String(busyLabel || "");
      const idle = String(idleLabel || "");
      if (btn) {
        btn.disabled = true;
        btn.textContent = busy ? `${SPIN_FRAMES[0]} ${busy}` : SPIN_FRAMES[0];
        timer = setInterval(() => {
          k = (k + 1) % SPIN_FRAMES.length;
          if (btn) {
            btn.textContent = busy ? `${SPIN_FRAMES[k]} ${busy}` : SPIN_FRAMES[k];
          }
        }, 90);
      }
      try {
        return await workFn();
      } finally {
        if (timer) clearInterval(timer);
        if (btn) {
          btn.disabled = false;
          btn.textContent = idle;
        }
      }
    }

    function fuzzyMatch(q, s) {
      const query = String(q || "").trim().toLowerCase();
      if (!query) return true;
      const text = String(s || "").toLowerCase();
      let j = 0;
      for (let i = 0; i < text.length && j < query.length; i++) {
        if (text[i] === query[j]) j++;
      }
      return j === query.length;
    }

    function statusClass(st) {
      if (st === "done") return "status done";
      if (st === "failed") return "status failed";
      return "status";
    }

    function _normChr(v) {
      const s = String(v || "").trim();
      if (s.toLowerCase().startsWith("chr")) return s.slice(3);
      return s;
    }

    function _toInt(v) {
      const n = Number(v);
      if (!Number.isFinite(n)) return null;
      return Math.trunc(n);
    }

    function _sameSigRange(a, b) {
      if (!a || !b) return false;
      const ac = _normChr(a.chrom);
      const bc = _normChr(b.chrom);
      const as = _toInt(a.start);
      const bs = _toInt(b.start);
      const ae = _toInt(a.end);
      const be = _toInt(b.end);
      return ac === bc && as === bs && ae === be;
    }

    function _safeNamePart(v) {
      return String(v || "")
        .trim()
        .replace(/[\\\\/:*?\"<>|\\s]+/g, "_")
        .replace(/^_+|_+$/g, "") || "plot";
    }

    function _getPreviewCanvas() {
      return document.querySelector("#img_preview .preview-canvas");
    }

    function _updateZoomText() {
      const el = document.getElementById("zoom_text");
      if (el) el.textContent = `${Math.round(viewer.scale * 100)}%`;
    }

    function _applyViewerTransform() {
      const canvas = _getPreviewCanvas();
      if (!canvas) return;
      canvas.style.transform = `translate(${viewer.tx}px, ${viewer.ty}px) scale(${viewer.scale})`;
      _updateZoomText();
    }

    function _setDragEnabled(flag) {
      viewer.dragEnabled = Boolean(flag);
      viewer.dragging = false;
      const btn = document.getElementById("drag_toggle_btn");
      if (btn) btn.textContent = viewer.dragEnabled ? "Drag: On" : "Drag: Off";
      const canvas = _getPreviewCanvas();
      if (canvas) canvas.style.cursor = viewer.dragEnabled ? "grab" : "default";
    }

    function _resetViewer() {
      viewer.scale = 1.0;
      viewer.tx = 0.0;
      viewer.ty = 0.0;
      viewer.dragging = false;
      _applyViewerTransform();
    }

    function _zoomBy(factor) {
      const wrap = document.getElementById("img_preview");
      const canvas = _getPreviewCanvas();
      if (!wrap || !canvas) return;
      const rect = wrap.getBoundingClientRect();
      const cx = rect.width * 0.5;
      const cy = rect.height * 0.5;
      const oldScale = viewer.scale;
      let newScale = oldScale * factor;
      newScale = Math.max(0.2, Math.min(10.0, newScale));
      if (!Number.isFinite(newScale) || newScale <= 0) return;
      const rx = (cx - viewer.tx) / oldScale;
      const ry = (cy - viewer.ty) / oldScale;
      viewer.scale = newScale;
      viewer.tx = cx - rx * newScale;
      viewer.ty = cy - ry * newScale;
      _applyViewerTransform();
    }

    function _bindViewerEvents() {
      const wrap = document.getElementById("img_preview");
      if (!wrap || wrap.dataset.boundViewer === "1") return;
      wrap.dataset.boundViewer = "1";

      wrap.addEventListener("wheel", (ev) => {
        ev.preventDefault();
        const factor = ev.deltaY < 0 ? 1.1 : 1 / 1.1;
        const oldScale = viewer.scale;
        let newScale = oldScale * factor;
        newScale = Math.max(0.2, Math.min(10.0, newScale));
        if (!Number.isFinite(newScale) || newScale <= 0) return;
        const rect = wrap.getBoundingClientRect();
        const cx = ev.clientX - rect.left;
        const cy = ev.clientY - rect.top;
        const rx = (cx - viewer.tx) / oldScale;
        const ry = (cy - viewer.ty) / oldScale;
        viewer.scale = newScale;
        viewer.tx = cx - rx * newScale;
        viewer.ty = cy - ry * newScale;
        _applyViewerTransform();
      }, { passive: false });

      wrap.addEventListener("mousedown", (ev) => {
        if (!viewer.dragEnabled) return;
        viewer.dragging = true;
        viewer.x = ev.clientX;
        viewer.y = ev.clientY;
        const canvas = _getPreviewCanvas();
        if (canvas) canvas.classList.add("dragging");
      });
      window.addEventListener("mousemove", (ev) => {
        if (!viewer.dragging) return;
        const dx = ev.clientX - viewer.x;
        const dy = ev.clientY - viewer.y;
        viewer.x = ev.clientX;
        viewer.y = ev.clientY;
        viewer.tx += dx;
        viewer.ty += dy;
        _applyViewerTransform();
      });
      const endDrag = () => {
        viewer.dragging = false;
        const canvas = _getPreviewCanvas();
        if (canvas) canvas.classList.remove("dragging");
      };
      window.addEventListener("mouseup", endDrag);
      wrap.addEventListener("mouseleave", endDrag);
    }

    function _setOneChromSelect(selId, chroms) {
      const sel = document.getElementById(selId);
      if (!sel) return;
      const prev = sel.value || "";
      const xs = Array.isArray(chroms) ? chroms : [];
      sel.innerHTML = "";
      if (xs.length === 0) {
        const op = document.createElement("option");
        op.value = "1";
        op.textContent = "1";
        sel.appendChild(op);
        sel.value = "1";
        return;
      }
      for (const c of xs) {
        const v = String(c || "").trim();
        if (!v) continue;
        const op = document.createElement("option");
        op.value = v;
        op.textContent = v;
        sel.appendChild(op);
      }
      const values = Array.from(sel.options).map((o) => String(o.value || ""));
      if (prev && values.includes(prev)) {
        sel.value = prev;
      } else if (sel.options.length > 0) {
        sel.selectedIndex = 0;
      }
    }

    function setChromOptions(chroms) {
      _setOneChromSelect("bim_chr1", chroms);
      _setOneChromSelect("bim_chr2", chroms);
      _setOneChromSelect("bim_chr3", chroms);
    }

    async function loadChromOptions(historyId) {
      if (!historyId) {
        setChromOptions(["1"]);
        return;
      }
      try {
        const res = await api(`/api/gwas-history/${encodeURIComponent(historyId)}/chroms`);
        setChromOptions(res.chroms || []);
      } catch (_e) {
        setChromOptions(["1"]);
      }
    }

    function _sortChromOptions(xs) {
      const arr = (Array.isArray(xs) ? xs : [])
        .map((x) => String(x || "").trim())
        .filter((x) => x !== "");
      arr.sort((a, b) => {
        const aa = String(a).toLowerCase().replace(/^chr/, "");
        const bb = String(b).toLowerCase().replace(/^chr/, "");
        const ai = Number(aa);
        const bi = Number(bb);
        const an = Number.isFinite(ai) && aa !== "";
        const bn = Number.isFinite(bi) && bb !== "";
        if (an && bn) return ai - bi;
        if (an) return -1;
        if (bn) return 1;
        return aa.localeCompare(bb, undefined, { numeric: true, sensitivity: "base" });
      });
      return arr;
    }

    async function loadChromOptionsForSelection() {
      const ids = _normHistoryIds(selectedHistoryIds);
      if (ids.length === 0) {
        setChromOptions(["1"]);
        return;
      }
      if (ids.length === 1) {
        await loadChromOptions(ids[0]);
        return;
      }
      try {
        const outs = await Promise.all(
          ids.map((hid) => api(`/api/gwas-history/${encodeURIComponent(hid)}/chroms`))
        );
        const merged = new Set();
        for (const out of outs) {
          const chroms = Array.isArray(out && out.chroms) ? out.chroms : [];
          for (const c of chroms) {
            const v = String(c || "").trim();
            if (v) merged.add(v);
          }
        }
        const sorted = _sortChromOptions(Array.from(merged));
        setChromOptions(sorted.length > 0 ? sorted : ["1"]);
      } catch (_e) {
        setChromOptions(["1"]);
      }
    }

    function buildBimranges() {
      const pairIds = [
        ["bim_chr1", "bim_pos1"],
        ["bim_chr2", "bim_pos2"],
        ["bim_chr3", "bim_pos3"],
      ];
      const out = [];
      const fullPattern = /^[^:]+:[^:-]+(?:-|:)[^:-]+$/;
      const rangeOnlyPattern = /^[^:-]+(?:-|:)[^:-]+$/;
      for (const [chrId, posId] of pairIds) {
        const chr = String((document.getElementById(chrId).value || "")).trim();
        let raw = String((document.getElementById(posId).value || "")).trim();
        if (!raw) continue;
        raw = raw
          .replace(/\\s+/g, "")
          .replace(/[：]/g, ":")
          .replace(/[－—–]/g, "-");
        if (fullPattern.test(raw)) {
          out.push(raw);
        } else if (rangeOnlyPattern.test(raw)) {
          if (chr) out.push(`${chr}:${raw}`);
        }
        if (out.length >= 4) break;
      }
      return out;
    }

    function _pickPanelFile(files, panel) {
      const isImage = (f) => Boolean(f && f.image);
      const low = (s) => String(s || "").toLowerCase();
      const fname = (f) => low(f.name || "");
      if (panel === "manh") return files.find((f) => isImage(f) && fname(f).includes(".manh.") && !fname(f).includes(".manhld.")) || null;
      if (panel === "qq") return files.find((f) => isImage(f) && fname(f).includes(".qq.")) || null;
      if (panel === "gene") return files.find((f) => isImage(f) && fname(f).includes(".gene.")) || null;
      if (panel === "ldblock") return files.find((f) => isImage(f) && fname(f).includes(".ldblock.")) || null;
      return null;
    }

    function _renderCompositeFigure(files) {
      const preview = document.getElementById("img_preview");
      preview.innerHTML = "";
      const manh = _pickPanelFile(files, "manh");
      const qq = _pickPanelFile(files, "qq");
      const gene = _pickPanelFile(files, "gene");
      const ld = _pickPanelFile(files, "ldblock");
      const hasQQ = Boolean(qq && qq.url);
      const SVG_NS = "http://www.w3.org/2000/svg";
      const XLINK_NS = "http://www.w3.org/1999/xlink";

      const build = (manhRatio) => {
        const ratio = (Number.isFinite(manhRatio) && manhRatio > 0) ? manhRatio : 2.0;
        const ldRatio = 2.0;
        const gap = 12;
        const leftW = 980;
        const rightW = hasQQ ? 980 : 0;
        const topH = leftW / ratio;
        const geneH = leftW / 20.0;
        const ldH = leftW / ldRatio;
        const fullW = hasQQ ? (leftW + gap + rightW) : leftW;
        const fullH = topH + gap + geneH + gap + ldH;
        const wrap = document.createElement("div");
        wrap.className = "composite-wrap";
        const svg = document.createElementNS(SVG_NS, "svg");
        svg.setAttribute("class", "composite-svg");
        svg.setAttribute("viewBox", `0 0 ${fullW} ${fullH}`);
        svg.setAttribute("preserveAspectRatio", "xMidYMin meet");

        const drawPanel = (x, y, w, h, title, fileObj) => {
          const frame = document.createElementNS(SVG_NS, "rect");
          frame.setAttribute("x", String(x));
          frame.setAttribute("y", String(y));
          frame.setAttribute("width", String(w));
          frame.setAttribute("height", String(h));
          frame.setAttribute("fill", "#fff");
          frame.setAttribute("stroke", "#dbe3ef");
          frame.setAttribute("stroke-width", "1");
          svg.appendChild(frame);
          const label = document.createElementNS(SVG_NS, "text");
          label.setAttribute("x", String(x + 7));
          label.setAttribute("y", String(y + 15));
          label.setAttribute("fill", "#334155");
          label.setAttribute("font-size", "12");
          label.setAttribute("font-family", "sans-serif");
          label.textContent = title;
          svg.appendChild(label);

          if (fileObj && fileObj.url) {
            const img = document.createElementNS(SVG_NS, "image");
            img.setAttributeNS(XLINK_NS, "href", fileObj.url);
            img.setAttribute("x", String(x + 2));
            img.setAttribute("y", String(y + 19));
            img.setAttribute("width", String(Math.max(1, w - 4)));
            img.setAttribute("height", String(Math.max(1, h - 21)));
            img.setAttribute("preserveAspectRatio", "xMidYMid meet");
            svg.appendChild(img);
          } else {
            const txt = document.createElementNS(SVG_NS, "text");
            txt.setAttribute("x", String(x + w * 0.5));
            txt.setAttribute("y", String(y + h * 0.52));
            txt.setAttribute("fill", "#64748b");
            txt.setAttribute("font-size", "14");
            txt.setAttribute("font-family", "sans-serif");
            txt.setAttribute("text-anchor", "middle");
            txt.textContent = "NA";
            svg.appendChild(txt);
          }
        };

        drawPanel(0, 0, leftW, topH, "manh", manh);
        drawPanel(0, topH + gap, leftW, geneH, "gene", gene);
        drawPanel(0, topH + gap + geneH + gap, leftW, ldH, "ldblock", ld);
        if (hasQQ) {
          drawPanel(leftW + gap, 0, rightW, topH, "qq", qq);
          drawPanel(leftW + gap, topH + gap, rightW, geneH, "NA", null);
          drawPanel(leftW + gap, topH + gap + geneH + gap, rightW, ldH, "NA", null);
        }
        wrap.appendChild(svg);
        preview.innerHTML = "";
        preview.appendChild(wrap);
      };

      build(2.0);
      if (manh && manh.url) {
        const probe = new Image();
        probe.onload = () => {
          if (probe.naturalWidth > 0 && probe.naturalHeight > 0) {
            build(probe.naturalWidth / probe.naturalHeight);
          }
        };
        probe.src = manh.url;
      }
    }

    function renderHistoryTable() {
      const tbody = document.getElementById("history_tbody");
      const hmsg = document.getElementById("history_msg");
      const q = document.getElementById("search_box").value || "";
      tbody.innerHTML = "";
      const filtered = gwasRows.filter((r) => fuzzyMatch(
        q,
        `${r.genotype} ${r.phenotype} ${r.model} ${r.genotype_type} ${r.grm || ""} ${r.qcov || ""} ${r.cov || ""} ${r.created_at}`
      ));
      const rows = (filtered.length > 0 || String(q).trim() === "") ? filtered : gwasRows;
      if (rows.length === 0) {
        const tr = document.createElement("tr");
        tr.innerHTML = '<td colspan="9" class="muted">No GWAS history rows</td>';
        tbody.appendChild(tr);
      }
      for (const r of rows) {
        const hid = String(r.history_id || "");
        const tr = document.createElement("tr");
        if (selectedHistoryIds.includes(hid)) tr.className = "sel";
        tr.style.cursor = "pointer";
        tr.onclick = () => _toggleHistoryRow(r, null);
        const checked = selectedHistoryIds.includes(hid) ? "checked" : "";
        tr.innerHTML = `<td><input type="checkbox" ${checked} onclick="event.stopPropagation();" /></td><td>${esc(r.genotype || "")}</td><td>${esc(r.phenotype || "")}</td><td>${esc(r.model || "")}</td><td>${esc(r.genotype_type || "")}</td><td>${esc(r.grm || "")}</td><td>${esc(r.qcov || "")}</td><td>${esc(r.cov || "")}</td><td>${esc(r.created_at || "")}</td>`;
        const cb = tr.querySelector('input[type="checkbox"]');
        if (cb) {
          cb.onchange = (ev) => {
            const on = Boolean(ev && ev.target && ev.target.checked);
            _toggleHistoryRow(r, on);
          };
        }
        tbody.appendChild(tr);
      }
      if (hmsg) hmsg.textContent = "";
      document.getElementById("left_msg").textContent = "";
    }

    function renderSigTable(rows) {
      const tbody = document.getElementById("sig_tbody");
      tbody.innerHTML = "";
      let foundSel = false;
      for (const r of (rows || [])) {
        const tr = document.createElement("tr");
        let logp = Number.NaN;
        if (Number.isFinite(Number(r.logp))) {
          logp = Number(r.logp);
        } else if (Number.isFinite(Number(r.p)) && Number(r.p) > 0) {
          logp = -Math.log10(Number(r.p));
        }
        const thisRange = {
          chrom: String(r.chrom || ""),
          start: _toInt(r.pos),
          end: _toInt(r.pos),
          logp: Number.isFinite(logp) ? Number(logp) : null,
        };
        if (_sameSigRange(selectedSigRange, thisRange)) {
          tr.className = "sel";
          foundSel = true;
        }
        tr.style.cursor = "pointer";
        const ptxt = Number.isFinite(logp) ? logp.toFixed(3) : "";
        tr.innerHTML = `<td>${esc(r.chrom || "")}</td><td>${esc(r.pos || "")}</td><td>${esc(ptxt)}</td><td>${esc(r.Gene || "")}</td>`;
        tr.onclick = () => {
          if (_sameSigRange(selectedSigRange, thisRange)) {
            selectedSigRange = null;
          } else {
            selectedSigRange = thisRange;
          }
          renderSigTable(rows || []);
          refreshCompositeByHistory(selectedHistoryId).catch((e) => {
            document.getElementById("bim_msg").textContent = String(e);
          });
        };
        tbody.appendChild(tr);
      }
      if (!foundSel) selectedSigRange = null;
    }

    async function loadSigSites(historyId) {
      const msg = document.getElementById("sig_msg");
      const copyBtn = document.getElementById("sig_copy_btn");
      const _sigThrEl = document.getElementById("sig_thr");
      const _sigSearchEl = document.getElementById("sig_search");
      const _annoSelEl = document.getElementById("anno_sel");
      const thr = String((_sigThrEl ? _sigThrEl.value : "") || "auto").trim();
      const q = String((_sigSearchEl ? _sigSearchEl.value : "") || "").trim();
      const annoPath = String((_annoSelEl ? _annoSelEl.value : "") || "").trim();
      if (_isMergeMode()) {
        msg.textContent = "Loading threshold SNPs...";
        try {
          const out = await api("/api/gwas-history/sigsites-merged", "POST", {
            history_ids: selectedHistoryIds,
            threshold: thr || "auto",
            anno: annoPath,
            search: q,
            limit: 3000,
          });
          renderSigTable(out.rows || []);
          lastSigRows = Array.isArray(out.rows) ? out.rows : [];
          lastSigSummary = `threshold=${out.threshold}, sig=${out.n_sig}, show=${out.n_show}`;
          if (copyBtn) copyBtn.disabled = lastSigRows.length === 0;
          msg.textContent = lastSigSummary;
        } catch (e) {
          renderSigTable([]);
          lastSigRows = [];
          lastSigSummary = "";
          if (copyBtn) copyBtn.disabled = true;
          msg.textContent = String(e);
        }
        return;
      }
      if (!historyId) {
        renderSigTable([]);
        lastSigRows = [];
        lastSigSummary = "";
        if (copyBtn) copyBtn.disabled = true;
        msg.textContent = "Select one GWAS history row.";
        return;
      }
      msg.textContent = "Loading threshold SNPs...";
      try {
        const out = await api(`/api/gwas-history/${encodeURIComponent(historyId)}/sigsites`, "POST", {
          threshold: thr || "auto",
          anno: annoPath,
          search: q,
          limit: 3000,
        });
        renderSigTable(out.rows || []);
        lastSigRows = Array.isArray(out.rows) ? out.rows : [];
        lastSigSummary = `threshold=${out.threshold}, sig=${out.n_sig}, show=${out.n_show}`;
        if (copyBtn) copyBtn.disabled = lastSigRows.length === 0;
        msg.textContent = lastSigSummary;
      } catch (e) {
        renderSigTable([]);
        lastSigRows = [];
        lastSigSummary = "";
        if (copyBtn) copyBtn.disabled = true;
        msg.textContent = String(e);
      }
    }

    async function loadAnnoFiles() {
      const res = await api("/api/anno-files");
      const sel = document.getElementById("anno_sel");
      const prev = sel.value;
      sel.innerHTML = '<option value="">(No annotation)</option>';
      for (const x of (res.files || [])) {
        const op = document.createElement("option");
        op.value = String(x.path || "");
        const alias = String(x.alias || "");
        const fname = String(x.name || "");
        const chroms = Array.isArray(x.chroms) ? x.chroms : [];
        const chrTxt = chroms.length > 0 ? `, chr=${chroms.length}` : "";
        op.textContent = alias ? `${alias} | ${fname}${chrTxt}` : `${fname}${chrTxt}`;
        sel.appendChild(op);
      }
      if (prev && (res.files || []).some((x) => String(x.path || "") === prev)) {
        sel.value = prev;
      }
    }

    async function loadHistory() {
      const hmsg = document.getElementById("history_msg");
      if (hmsg) hmsg.textContent = "history loading...";
      _setText("left_msg", "history loading...");
      let res = null;
      try {
        res = await api("/api/gwas-history", "GET", null, 12000);
      } catch (e1) {
        // one retry for transient fetch stall
        if (hmsg) hmsg.textContent = `history retrying... (${String(e1)})`;
        res = await api(`/api/gwas-history?_ts=${Date.now()}`, "GET", null, 12000);
      }
      gwasRows = Array.isArray(res.runs) ? res.runs : [];
      const valid = new Set(gwasRows.map((r) => String(r.history_id || "")));
      selectedHistoryIds = selectedHistoryIds.filter((x) => valid.has(String(x || "")));
      _syncPrimarySelection();
      await loadChromOptionsForSelection();
      if (hmsg) hmsg.textContent = `history received ${gwasRows.length} row(s)...`;
      try {
        renderHistoryTable();
        renderSeriesStyleRows();
      } catch (e2) {
        const msg = `history render failed: ${String(e2)}`;
        if (hmsg) hmsg.textContent = msg;
        _setText("left_msg", msg);
        return;
      }
      if (String(res.error || "").trim()) {
        if (hmsg) hmsg.textContent = String(res.error);
        _setText("left_msg", String(res.error));
      }
    }

    async function uploadGwas() {
      const fileEl = document.getElementById("gwas_upload_file");
      const btn = document.getElementById("gwas_upload_btn");
      const msg = document.getElementById("upload_msg");
      const files = (fileEl && fileEl.files) ? Array.from(fileEl.files) : [];
      if (files.length === 0) {
        if (msg) msg.textContent = "Select GWAS file(s) first.";
        return;
      }
      let timer = null;
      let k = 0;
      if (btn) {
        btn.disabled = true;
        btn.textContent = SPIN_FRAMES[0];
        timer = setInterval(() => {
          k = (k + 1) % SPIN_FRAMES.length;
          if (btn) btn.textContent = SPIN_FRAMES[k];
        }, 90);
      }
      if (msg) msg.textContent = `Uploading ${files.length} GWAS file(s)...`;
      try {
        const BATCH_SIZE = 4;
        let uploaded = [];
        let failed = [];
        let recvTotal = 0;
        for (let i = 0; i < files.length; i += BATCH_SIZE) {
          const batch = files.slice(i, i + BATCH_SIZE);
          if (msg) msg.textContent = `Uploading ${Math.min(i + 1, files.length)}-${Math.min(i + batch.length, files.length)}/${files.length} ...`;
          const fd = new FormData();
          for (const f of batch) fd.append("file", f);
          try {
            const out = await apiForm("/api/gwas-upload", fd, 1200000);
            const up = Array.isArray(out.uploaded) ? out.uploaded : [];
            const fl = Array.isArray(out.failed) ? out.failed : [];
            const recvN = Number(out.received_count || batch.length || 0);
            if (Number.isFinite(recvN) && recvN > 0) recvTotal += recvN;
            uploaded = uploaded.concat(up);
            failed = failed.concat(fl);
          } catch (eBatch) {
            const errTxt = String(eBatch || "batch upload failed");
            for (const f of batch) {
              failed.push({ file_name: String(f && f.name || ""), error: errTxt });
            }
          }
        }
        const upN = uploaded.length;
        const failN = failed.length;
        const first = upN > 0 ? uploaded[0] : null;
        const det = first ? (first.detected || {}) : {};
        let text = `Uploaded ${upN}/${files.length} file(s)`;
        if (Number.isFinite(recvTotal) && recvTotal > 0) text += `, received=${recvTotal}`;
        if (failN > 0) {
          const e0 = String((failed[0] && failed[0].error) ? failed[0].error : "").trim();
          text += `, failed ${failN}`;
          if (e0) text += ` (first error: ${e0})`;
        }
        if (first) {
          text += ` | chr=${String(det.chr || "")}, pos=${String(det.pos || "")}, p=${String(det.pvalue || "")}, rows=${Number(first.n_rows || 0)}`;
        }
        if (msg) msg.textContent = text;
        const hid = String((first && first.history_id) || "").trim();
        await loadHistory();
        if (hid) {
          const row = gwasRows.find((r) => String(r.history_id || "") === hid);
          if (row) await selectHistoryRow(row);
        }
        if (fileEl) fileEl.value = "";
      } catch (e) {
        if (msg) msg.textContent = String(e);
      } finally {
        if (timer) clearInterval(timer);
        if (btn) {
          btn.disabled = false;
          btn.textContent = "Upload GWAS";
        }
      }
    }

    async function refreshCompositeByHistory(historyId) {
      const preview = document.getElementById("img_preview");
      const selIds = Array.isArray(selectedHistoryIds) ? selectedHistoryIds.slice() : [];
      if (selIds.length === 0 && !historyId) {
        preview.innerHTML = '<div class="muted">Waiting for selection...</div>';
        return;
      }
      const _ldColorEl = document.getElementById("ld_color");
      const _sigThrEl = document.getElementById("sig_thr");
      const seriesStyles = _collectSeriesStyles();
      const style0 = (seriesStyles.length > 0) ? seriesStyles[0] : { color: "auto", alpha: 0.7, marker: "o", size: 16 };
      const manhRatio = _readManhRatio();
      const manhPalette = String(style0.color || "auto");
      const manhAlpha = String(style0.alpha == null ? 0.7 : style0.alpha);
      const manhMarker = String(style0.marker || "o");
      const manhSize = String(style0.size || "16");
      const ldColor = String((_ldColorEl ? _ldColorEl.value : "") || "#4b5563");
      const ldThr = "auto";
      const sigThr = String((_sigThrEl ? _sigThrEl.value : "") || "auto").trim();
      const annoPath = document.getElementById("anno_sel").value || "";
      const bimranges = buildBimranges();
      if (_isMergeMode()) {
        preview.innerHTML = '<div class="muted">Rendering merged Manhattan...</div>';
        try {
          const out = await api("/api/gwas-history/render-merged", "POST", {
            history_ids: selIds,
            bimrange: bimranges,
            manh_ratio: manhRatio,
            manh_palette: manhPalette,
            manh_alpha: manhAlpha,
            manh_marker: manhMarker,
            manh_size: manhSize,
            series_styles: seriesStyles,
            ld_color: ldColor,
            ld_p_threshold: ldThr,
            threshold: sigThr || "auto",
            render_qq: false,
            editable_svg: false,
          });
          const meta = out.meta || {};
          const loadSec = Number(meta.load_sec || 0);
          const hitRender = Number(meta.cache_hits_render || 0);
          const hitSig = Number(meta.cache_hits_sig || 0);
          const hitDisk = Number(meta.cache_hits_disk || 0);
          const missDisk = Number(meta.cache_miss_disk || 0);
          const renderSec = Number(meta.render_sec || 0);
          const rustTxt = Boolean(meta.rust_loader_available) ? "rust=on" : "rust=off";
          const loadTxt = Number.isFinite(loadSec) ? `, load=${loadSec.toFixed(2)}s` : "";
          const drawTxt = Number.isFinite(renderSec) ? `, render=${renderSec.toFixed(2)}s` : "";
          document.getElementById("preview_meta").textContent =
            `merged tasks=${meta.n_tasks || selIds.length}, style rows=${seriesStyles.length}${loadTxt}${drawTxt}, cache(render/sig/local/disk)=${hitRender}/${hitSig}/${hitDisk}/${missDisk}, ${rustTxt}`;
          preview.innerHTML = "";
          const canvas = document.createElement("div");
          canvas.className = "preview-canvas";
          canvas.innerHTML = String(out.svg || "");
          preview.appendChild(canvas);
          const svg = canvas.querySelector("svg");
          if (svg) {
            svg.style.width = "100%";
            svg.style.height = "auto";
            svg.style.display = "block";
          }
          _resetViewer();
        } catch (e) {
          document.getElementById("preview_meta").textContent = "Merged render failed";
          preview.innerHTML = `<div class="muted">${esc(String(e))}</div>`;
        }
        return;
      }
      if (!historyId) {
        preview.innerHTML = '<div class="muted">Waiting for selection...</div>';
        return;
      }
      function buildRenderPayload(forceFull) {
        return {
          anno: annoPath,
          bimrange: bimranges,
          highlight_range: selectedSigRange,
          full: Boolean(forceFull),
          editable_svg: false,
          manh_ratio: manhRatio,
          manh_palette: manhPalette,
          manh_alpha: manhAlpha,
          manh_marker: manhMarker,
          manh_size: manhSize,
          ld_color: ldColor,
          ld_p_threshold: ldThr,
        };
      }
      preview.innerHTML = '<div class="muted">Rendering...</div>';
      try {
        const out = await api(`/api/gwas-history/${encodeURIComponent(historyId)}/render`, "POST", buildRenderPayload(false));
        const meta = out.meta || {};
        const br = String(meta.bimrange || "");
        const pf = String(meta.p_filter || "");
        const loadSec = Number(meta.load_sec || 0);
        const renderSec = Number(meta.render_sec || 0);
        const cacheSrc = String(meta.cache_source || "");
        const rustTxt = Boolean(meta.rust_loader_available) ? "rust=on" : "rust=off";
        const loadTxt = Number.isFinite(loadSec) ? `, load=${loadSec.toFixed(2)}s` : "";
        const drawTxt = Number.isFinite(renderSec) ? `, render=${renderSec.toFixed(2)}s` : "";
        const srcTxt = cacheSrc ? `, cache=${cacheSrc}` : "";
        document.getElementById("preview_meta").textContent =
          `nTotal=${meta.n_total || 0}, nDraw=${meta.n_draw || 0}${pf ? `, draw=${pf}` : ""}${br ? `, bimrange=${br}` : ""}${loadTxt}${drawTxt}${srcTxt}, ${rustTxt}`;
        preview.innerHTML = "";
        const canvas = document.createElement("div");
        canvas.className = "preview-canvas";
        canvas.innerHTML = String(out.svg || "");
        preview.appendChild(canvas);
        const svg = canvas.querySelector("svg");
        if (svg) {
          svg.style.width = "100%";
          svg.style.height = "auto";
          svg.style.display = "block";
        }
        _resetViewer();
      } catch (e) {
        document.getElementById("preview_meta").textContent = "Render failed";
        preview.innerHTML = `<div class="muted">${esc(String(e))}</div>`;
      }
    }

    async function selectHistoryRow(row) {
      selectedHistoryIds = [];
      if (row && String(row.history_id || "").trim() !== "") {
        selectedHistoryIds = [String(row.history_id || "").trim()];
      }
      _syncPrimarySelection();
      selectedSigRange = null;
      _markStartInvalidated();
      renderHistoryTable();
      renderSeriesStyleRows();
      await loadChromOptionsForSelection();
      document.getElementById("img_preview").innerHTML = '<div class="muted">Select task and click Visualizing.</div>';
      renderSigTable([]);
      _setText("sig_msg", "Click Visualizing to load selected task.");
      _setText("anno_msg", "Ready.");
    }

    async function refreshAllByHistory(historyId) {
      await refreshCompositeByHistory(historyId);
    }

    async function runSelected() {
      const msg = document.getElementById("anno_msg");
      if (_isMergeMode()) {
        msg.textContent = "Run is disabled in merge mode.";
        return;
      }
      if (!selectedHistoryId) {
        msg.textContent = "Select one GWAS history row first";
        return;
      }
      const annoPath = document.getElementById("anno_sel").value || "";
      const bimranges = buildBimranges();
      const _ldColorEl = document.getElementById("ld_color");
      const ldColor = String((_ldColorEl ? _ldColorEl.value : "") || "#4b5563");
      const ldThr = "auto";
      msg.textContent = "Submitting...";
      await api("/api/jobs/from-gwas", "POST", {
        history_id: selectedHistoryId,
        run_id: selectedRow ? (selectedRow.run_id || "") : "",
        result_file: selectedRow ? (selectedRow.result_file || "") : "",
        anno: annoPath,
        bimrange: bimranges,
        ld_color: ldColor,
        ld_p_threshold: ldThr,
      });
      msg.textContent = "Submitted";
      await loadHistory();
      await refreshCompositeByHistory(selectedHistoryId);
    }

    async function copySigTable() {
      const msg = document.getElementById("sig_msg");
      const rows = Array.isArray(lastSigRows) ? lastSigRows : [];
      if (rows.length === 0) {
        if (msg) msg.textContent = "No rows to copy.";
        return;
      }
      const lines = [];
      if (String(lastSigSummary || "").trim()) lines.push(String(lastSigSummary));
      lines.push("Chr\tPos\t-log10P\tGene");
      for (const r of rows) {
        let logp = Number.NaN;
        if (Number.isFinite(Number(r.logp))) {
          logp = Number(r.logp);
        } else if (Number.isFinite(Number(r.p)) && Number(r.p) > 0) {
          logp = -Math.log10(Number(r.p));
        }
        const ptxt = Number.isFinite(logp) ? logp.toFixed(3) : "";
        lines.push(
          [
            String(r.chrom || ""),
            String(r.pos || ""),
            String(ptxt),
            String(r.Gene || ""),
          ].join("\t")
        );
      }
      const text = lines.join("\\n");
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
        } else {
          const ta = document.createElement("textarea");
          ta.value = text;
          ta.style.position = "fixed";
          ta.style.left = "-99999px";
          ta.style.top = "-99999px";
          document.body.appendChild(ta);
          ta.focus();
          ta.select();
          document.execCommand("copy");
          document.body.removeChild(ta);
        }
        if (msg) msg.textContent = `${lastSigSummary} | copied ${rows.length} row(s)`;
      } catch (e) {
        if (msg) msg.textContent = String(e);
      }
    }

    async function downloadCurrentSvg() {
      const msg = document.getElementById("bim_msg");
      const selIds = Array.isArray(selectedHistoryIds) ? selectedHistoryIds.slice() : [];
      if (selIds.length === 0 && !selectedHistoryId) {
        if (msg) msg.textContent = "Select one GWAS history row first.";
        return;
      }
      const annoPath = document.getElementById("anno_sel").value || "";
      const bimranges = buildBimranges();
      const _ldColorEl = document.getElementById("ld_color");
      const _sigThrEl = document.getElementById("sig_thr");
      const seriesStyles = _collectSeriesStyles();
      const style0 = (seriesStyles.length > 0) ? seriesStyles[0] : { color: "auto", alpha: 0.7, marker: "o", size: 16 };
      const manhRatio = _readManhRatio();
      const manhPalette = String(style0.color || "auto");
      const manhAlpha = String(style0.alpha == null ? 0.7 : style0.alpha);
      const manhMarker = String(style0.marker || "o");
      const manhSize = String(style0.size || "16");
      const ldColor = String((_ldColorEl ? _ldColorEl.value : "") || "#4b5563");
      const ldThr = "auto";
      const sigThr = String((_sigThrEl ? _sigThrEl.value : "") || "auto").trim();

      try {
        await _withButtonSpinner("download_btn", "Downloading", "Download", async () => {
          let out = null;
          if (_isMergeMode()) {
            out = await api("/api/gwas-history/render-merged", "POST", {
              history_ids: selIds,
              bimrange: bimranges,
              manh_ratio: manhRatio,
              manh_palette: manhPalette,
              manh_alpha: manhAlpha,
              manh_marker: manhMarker,
              manh_size: manhSize,
              series_styles: seriesStyles,
              ld_color: ldColor,
              ld_p_threshold: ldThr,
              // Download full figure: include all Manhattan points, but keep
              // threshold-based coloring (below-threshold points stay gray).
              threshold: sigThr || "auto",
              include_all_points: true,
              render_qq: true,
              editable_svg: true,
            });
          } else {
            out = await api(`/api/gwas-history/${encodeURIComponent(selectedHistoryId)}/render`, "POST", {
              anno: annoPath,
              bimrange: bimranges,
              highlight_range: selectedSigRange,
              full: true,
              editable_svg: true,
              manh_ratio: manhRatio,
              manh_palette: manhPalette,
              manh_alpha: manhAlpha,
              manh_marker: manhMarker,
              manh_size: manhSize,
              ld_color: ldColor,
              // Download full figure: include SNPs that were filtered by on-screen LD threshold.
              ld_p_threshold: 1.0,
            });
          }
          let xml = String(out && out.svg ? out.svg : "");
          if (xml.trim() === "") {
            throw new Error("No SVG returned.");
          }
          if (!String(xml).startsWith("<?xml")) {
            xml = '<?xml version="1.0" encoding="UTF-8"?>\\n' + xml;
          }
          const blob = new Blob([xml], { type: "image/svg+xml;charset=utf-8" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          let base = "postgwas";
          let mode = (Array.isArray(bimranges) && bimranges.length > 0) ? "manhld" : "manhqq";
          if (_isMergeMode()) {
            base = `merged_${selIds.length}_tasks`;
            mode = "manh";
          } else {
            const p = _safeNamePart((selectedRow && selectedRow.phenotype) ? selectedRow.phenotype : "postgwas");
            const m = _safeNamePart((selectedRow && selectedRow.model) ? selectedRow.model : "plot");
            const t = _safeNamePart((selectedRow && selectedRow.genotype_type) ? selectedRow.genotype_type : "");
            base = t ? `${p}.${t}.${m}` : `${p}.${m}`;
          }
          a.href = url;
          a.download = `${base}.full.${mode}.svg`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          setTimeout(() => URL.revokeObjectURL(url), 3000);
        });
      } catch (e) {
        if (msg) msg.textContent = String(e);
      }
    }

    _bindInput("search_box", "input", renderHistoryTable);
    _bindClick("zoom_in_btn", () => _zoomBy(1.1));
    _bindClick("zoom_out_btn", () => _zoomBy(1 / 1.1));
    _bindClick("zoom_reset_btn", () => _resetViewer());
    _bindClick("drag_toggle_btn", () => _setDragEnabled(!viewer.dragEnabled));
    _bindClick("gwas_upload_btn", () => uploadGwas());
    _bindClick("bim_apply_btn", () => _withButtonSpinner("bim_apply_btn", "Visualizing", "Visualizing", async () => {
      await refreshAllByHistory(selectedHistoryId);
    }).catch((e) => {
      _setText("bim_msg", String(e));
    }));
    _bindInput("anno_sel", "change", () => _markStartInvalidated());
    _bindClick("download_btn", () => downloadCurrentSvg());
    _bindClick("sig_copy_btn", () => copySigTable());
    _bindInput("manh_ratio", "keydown", (ev) => {
      if (ev.key === "Enter") {
        refreshAllByHistory(selectedHistoryId).catch((e) => {
          _setText("bim_msg", String(e));
        });
      }
    });
    _bindInput("manh_ratio", "blur", () => {
      if (_normHistoryIds(selectedHistoryIds).length === 0) return;
      refreshAllByHistory(selectedHistoryId).catch((e) => {
        _setText("bim_msg", String(e));
      });
    });
    _bindInput("sig_thr", "keydown", (ev) => {
      if (ev.key === "Enter") {
        refreshCompositeByHistory(selectedHistoryId).catch((e) => {
          _setText("bim_msg", String(e));
        });
      }
    });
    _bindInput("sig_search", "input", () => {
      if (sigTimer) clearTimeout(sigTimer);
      sigTimer = setTimeout(() => {
        loadSigSites(selectedHistoryId).catch((e) => {
          _setText("sig_msg", String(e));
        });
      }, 220);
    });
    for (const id of ["bim_pos1", "bim_pos2", "bim_pos3"]) {
      _bindInput(id, "keydown", (ev) => {
        if (ev.key === "Enter") {
          refreshAllByHistory(selectedHistoryId).catch((e) => {
            _setText("bim_msg", String(e));
          });
        }
      });
    }
    _bindViewerEvents();
    _setDragEnabled(true);
    renderSeriesStyleRows();
    const _searchEl = document.getElementById("search_box");
    if (_searchEl) _searchEl.value = "";
    loadAnnoFiles().catch((e) => {
      _setText("anno_msg", String(e));
    });
    loadHistory().catch((e) => {
      const hmsg = document.getElementById("history_msg");
      if (hmsg) hmsg.textContent = String(e);
      _setText("left_msg", String(e));
    });
  </script>
</body>
</html>
"""


def _make_handler(state: WebUIState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "JanusXWebUI/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep console clean.
            return

        def _safe_write_bytes(self, data: bytes) -> bool:
            try:
                self.wfile.write(data)
                return True
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return False
            except OSError as exc:
                # Common client-disconnect socket errors on different platforms.
                if getattr(exc, "errno", None) in {32, 54, 104, 10053, 10054}:
                    return False
                raise

        def _send_json(self, status: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self._safe_write_bytes(data)

        def _send_text(self, status: int, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
            data = text.encode("utf-8", errors="replace")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self._safe_write_bytes(data)

        def _read_json_body(self) -> dict[str, Any]:
            clen = self.headers.get("Content-Length", "0").strip()
            try:
                n = int(clen)
            except Exception:
                n = 0
            raw = self.rfile.read(max(0, n))
            if not raw:
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}

        def _read_multipart_body(self) -> dict[str, Any]:
            ctype = str(self.headers.get("Content-Type", "")).strip()
            if "multipart/form-data" not in ctype.lower():
                raise ValueError("Content-Type must be multipart/form-data.")
            try:
                n = int(str(self.headers.get("Content-Length", "0")).strip())
            except Exception:
                n = 0
            raw = self.rfile.read(max(0, n))
            if not raw:
                return {}
            return _parse_multipart_form_data(ctype, raw)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/":
                self._send_text(HTTPStatus.OK, _html_page(), "text/html; charset=utf-8")
                return

            if path == "/api/jobs":
                self._send_json(HTTPStatus.OK, {"jobs": state.list_jobs()})
                return

            if path == "/api/gwas-history":
                runs = state.list_gwas_history(limit=300)
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "db_path": str(state.db_path),
                        "runs": runs,
                        "error": str(state._history_error or ""),
                    },
                )
                return

            if path == "/api/anno-files":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "anno_dir": str(state.anno_dir),
                        "files": state.list_anno_files(),
                    },
                )
                return

            if path.startswith("/api/gwas-history/"):
                suffix = unquote(path[len("/api/gwas-history/") :].strip("/"))
                if suffix.endswith("/chroms"):
                    history_id = suffix[: -len("/chroms")].strip("/")
                    try:
                        chroms = state.get_history_chromosomes(history_id)
                    except Exception as exc:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    self._send_json(HTTPStatus.OK, {"history_id": history_id, "chroms": chroms})
                    return
                if suffix.endswith("/postgwas"):
                    history_id = suffix[: -len("/postgwas")].strip("/")
                    job = state.get_postgwas_job_by_history(history_id)
                    if job is None:
                        self._send_json(
                            HTTPStatus.NOT_FOUND,
                            {"error": "no postgwas run for this history row"},
                        )
                        return
                    self._send_json(HTTPStatus.OK, job)
                    return
                run_id = suffix
                row = state.get_gwas_history(run_id)
                if row is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "gwas history not found"})
                    return
                self._send_json(HTTPStatus.OK, row)
                return

            if path.startswith("/api/jobs/") and "/files/" in path:
                # /api/jobs/<job_id>/files/<relpath>
                prefix = "/api/jobs/"
                rest = path[len(prefix):]
                job_id, _, rel_part = rest.partition("/files/")
                target = state.resolve_job_file(job_id, rel_part)
                if target is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "file not found"})
                    return
                try:
                    data = target.read_bytes()
                except Exception:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "failed to read file"})
                    return
                ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self._safe_write_bytes(data)
                return

            if path.startswith("/api/jobs/"):
                job_id = path[len("/api/jobs/") :].strip("/")
                meta = state.get_job(job_id)
                if meta is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "job not found"})
                    return
                self._send_json(HTTPStatus.OK, meta)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/gwas-upload":
                try:
                    form = self._read_multipart_body()
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return

                def _first_text(v: Any) -> str:
                    x = v
                    if isinstance(x, list):
                        x = x[0] if len(x) > 0 else ""
                    if isinstance(x, dict):
                        return ""
                    return str(x or "").strip()

                def _as_file_items(v: Any) -> list[dict[str, Any]]:
                    if isinstance(v, dict):
                        return [v]
                    if isinstance(v, list):
                        out: list[dict[str, Any]] = []
                        for x in v:
                            if isinstance(x, dict):
                                out.append(x)
                        return out
                    return []

                file_items: list[dict[str, Any]] = []
                for key in ("file", "file[]", "files", "files[]"):
                    file_items.extend(_as_file_items(form.get(key)))
                if len(file_items) == 0:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "missing file field"})
                    return

                phenotype = _first_text(form.get("phenotype", ""))
                model = _first_text(form.get("model", ""))
                genofile = _first_text(form.get("genofile", ""))
                genofile_kind = _first_text(form.get("genofile_kind", "")).lower()

                uploaded: list[dict[str, Any]] = []
                failed: list[dict[str, Any]] = []
                for idx, item in enumerate(file_items, start=1):
                    file_bytes = item.get("data", b"")
                    if not isinstance(file_bytes, (bytes, bytearray)) or len(file_bytes) == 0:
                        failed.append(
                            {
                                "index": int(idx),
                                "file_name": str(item.get("filename", "") or ""),
                                "error": "invalid upload file",
                            }
                        )
                        continue
                    raw_name = str(item.get("filename", "") or f"uploaded_gwas_{idx}.tsv")
                    try:
                        dst = state._upload_storage_path(raw_name, idx)
                        while dst.exists():
                            dst = state._upload_storage_path(raw_name, idx)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        with dst.open("wb") as w:
                            w.write(bytes(file_bytes))
                    except Exception as exc:
                        failed.append(
                            {
                                "index": int(idx),
                                "file_name": raw_name,
                                "error": f"failed to save upload: {exc}",
                            }
                        )
                        continue
                    try:
                        out = state.register_uploaded_gwas(
                            file_path=dst,
                            original_name=raw_name,
                            phenotype=phenotype,
                            model=model,
                            genofile=genofile,
                            genofile_kind=genofile_kind,
                        )
                        uploaded.append(dict(out))
                    except Exception as exc:
                        try:
                            if dst.exists():
                                dst.unlink()
                        except Exception:
                            pass
                        failed.append(
                            {
                                "index": int(idx),
                                "file_name": raw_name,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

                if len(uploaded) == 0:
                    err0 = str(failed[0].get("error", "upload failed")) if len(failed) > 0 else "upload failed"
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {
                            "error": err0,
                            "uploaded_count": 0,
                            "failed_count": int(len(failed)),
                            "failed": failed,
                        },
                    )
                    return

                self._send_json(
                    HTTPStatus.OK,
                        {
                            "ok": len(failed) == 0,
                            "received_count": int(len(file_items)),
                            "uploaded_count": int(len(uploaded)),
                            "failed_count": int(len(failed)),
                            "uploaded": uploaded,
                        "failed": failed,
                        "history_id": str(uploaded[0].get("history_id", "")),
                    },
                )
                return

            if path.startswith("/api/gwas-history/") and path.endswith("/sigsites"):
                suffix = unquote(path[len("/api/gwas-history/") :].strip("/"))
                history_id = suffix[: -len("/sigsites")].strip("/")
                payload = self._read_json_body()
                threshold = payload.get("threshold", "auto")
                anno_file = str(payload.get("anno", "")).strip()
                search = str(payload.get("search", "")).strip()
                limit = payload.get("limit", 2000)
                try:
                    out = state.get_history_sigsites(
                        history_id,
                        threshold=threshold,
                        anno_file=anno_file,
                        search=search,
                        limit=limit,
                    )
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, out)
                return

            if path == "/api/gwas-history/sigsites-merged":
                payload = self._read_json_body()
                history_ids = payload.get("history_ids", [])
                if not isinstance(history_ids, list):
                    history_ids = []
                threshold = payload.get("threshold", "auto")
                anno_file = str(payload.get("anno", "")).strip()
                search = str(payload.get("search", "")).strip()
                limit = payload.get("limit", 2000)
                try:
                    out = state.get_merged_history_sigsites(
                        [str(x).strip() for x in history_ids if str(x).strip() != ""],
                        threshold=threshold,
                        anno_file=anno_file,
                        search=search,
                        limit=limit,
                    )
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, out)
                return

            if path.startswith("/api/gwas-history/") and path.endswith("/start"):
                suffix = unquote(path[len("/api/gwas-history/") :].strip("/"))
                history_id = suffix[: -len("/start")].strip("/")
                payload = self._read_json_body()
                anno_file = str(payload.get("anno", "")).strip()
                try:
                    out = state.warmup_history(history_id, anno_file=anno_file)
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, out)
                return

            if path == "/api/gwas-history/start-merged":
                payload = self._read_json_body()
                history_ids = payload.get("history_ids", [])
                if not isinstance(history_ids, list):
                    history_ids = []
                anno_file = str(payload.get("anno", "")).strip()
                try:
                    out = state.warmup_merged_history(
                        [str(x).strip() for x in history_ids if str(x).strip() != ""],
                        anno_file=anno_file,
                    )
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, out)
                return

            if path == "/api/gwas-history/render-merged":
                payload = self._read_json_body()
                history_ids = payload.get("history_ids", [])
                if not isinstance(history_ids, list):
                    history_ids = []
                bimranges = _normalize_bimrange_items(payload.get("bimrange", ""), max_items=4)
                manh_ratio = payload.get("manh_ratio", 2.0)
                manh_palette = str(payload.get("manh_palette", "auto"))
                manh_alpha = payload.get("manh_alpha", 0.7)
                manh_marker = str(payload.get("manh_marker", "o"))
                manh_size = payload.get("manh_size", 16.0)
                series_styles = payload.get("series_styles", None)
                ld_color = str(payload.get("ld_color", "#4b5563")).strip()
                ld_p_threshold = payload.get("ld_p_threshold", "auto")
                threshold = payload.get("threshold", "auto")
                include_all_points = bool(payload.get("include_all_points", False))
                render_qq = bool(payload.get("render_qq", True))
                editable_svg = bool(payload.get("editable_svg", False))
                try:
                    svg, meta = state.render_merged_history_svg(
                        [str(x).strip() for x in history_ids if str(x).strip() != ""],
                        bimrange=bimranges,
                        threshold=threshold,
                        include_all_points=include_all_points,
                        manh_ratio=manh_ratio,
                        manh_palette=manh_palette,
                        manh_alpha=manh_alpha,
                        manh_marker=manh_marker,
                        manh_size=manh_size,
                        series_styles=series_styles,
                        ld_color=ld_color,
                        ld_p_threshold=ld_p_threshold,
                        render_qq=render_qq,
                        editable_svg=editable_svg,
                    )
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "svg": svg,
                        "meta": meta,
                    },
                )
                return

            if path.startswith("/api/gwas-history/") and path.endswith("/render"):
                suffix = unquote(path[len("/api/gwas-history/") :].strip("/"))
                history_id = suffix[: -len("/render")].strip("/")
                payload = self._read_json_body()
                anno = str(payload.get("anno", "")).strip()
                bimranges = _normalize_bimrange_items(payload.get("bimrange", ""), max_items=4)
                full = bool(payload.get("full", False))
                manh_ratio = payload.get("manh_ratio", 2.0)
                manh_palette = str(payload.get("manh_palette", "auto"))
                manh_alpha = payload.get("manh_alpha", 0.7)
                manh_marker = str(payload.get("manh_marker", "o"))
                manh_size = payload.get("manh_size", 16.0)
                ld_color = str(payload.get("ld_color", "#4b5563")).strip()
                ld_p_threshold = payload.get("ld_p_threshold", "auto")
                editable_svg = bool(payload.get("editable_svg", False))
                highlight_range = payload.get("highlight_range", None)
                try:
                    svg, meta = state.render_history_svg(
                        history_id,
                        bimrange=bimranges,
                        anno=anno,
                        highlight_range=highlight_range,
                        full=full,
                        manh_ratio=manh_ratio,
                        manh_palette=manh_palette,
                        manh_alpha=manh_alpha,
                        manh_marker=manh_marker,
                        manh_size=manh_size,
                        ld_color=ld_color,
                        ld_p_threshold=ld_p_threshold,
                        editable_svg=editable_svg,
                    )
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "history_id": history_id,
                        "svg": svg,
                        "meta": meta,
                    },
                )
                return

            if path == "/api/anno-upload":
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": (
                            "Annotation upload is disabled in WebUI."
                        )
                    },
                )
                return

            if path == "/api/jobs":
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": (
                            "Manual postgwas submission is disabled in current WebUI. "
                            "Please use GWAS history + GFF/BED via /api/jobs/from-gwas."
                        )
                    },
                )
                return

            if path == "/api/jobs/from-gwas":
                payload = self._read_json_body()
                meta, errors = state.create_job_from_gwas_history(payload)
                if errors:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "; ".join(errors)})
                    return
                assert meta is not None
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "job_id": meta["job_id"],
                        "status": meta["status"],
                    },
                )
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    return Handler


def build_parser() -> argparse.ArgumentParser:
    default_root = str((resolve_jx_home() / "webui").resolve())
    parser = argparse.ArgumentParser(
        prog="jx webui",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx webui",
                "jx webui --host 0.0.0.0 --port 8765",
                "jx webui --root ~/JanusX/.janusx/webui",
            ]
        ),
        description="JanusX Web UI (Post-GWAS first implementation).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port (default: %(default)s).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=default_root,
        help="WebUI runtime directory (jobs/logs) (default: %(default)s).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="Do not auto-open browser.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = WebUIState(root)
    cleanup = state.startup_cleanup_report()

    handler = _make_handler(state)
    server = ThreadingHTTPServer((args.host, int(args.port)), handler)
    url_host = args.host
    if url_host in {"0.0.0.0", "::"}:
        url_host = "127.0.0.1"
    url = f"http://{url_host}:{args.port}/"

    print(f"JanusX WebUI root: {root}")
    if str(cleanup.get("error", "")).strip() != "":
        print(f"WebUI preflight cleanup warning: {cleanup['error']}")
    elif int(cleanup.get("removed", 0)) > 0:
        print(
            "WebUI preflight cleanup: removed "
            f"{int(cleanup.get('removed', 0))} invalid GWAS history run(s) "
            "(missing result/genotype files)."
        )
    elif int(cleanup.get("pruned_runs", 0)) > 0:
        print(
            "WebUI preflight cleanup: pruned "
            f"{int(cleanup.get('pruned_summary_rows', 0))} invalid history row(s) "
            f"across {int(cleanup.get('pruned_runs', 0))} run(s)."
        )
    print(f"Serving on: {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        try:
            webbrowser.open(url, new=2, autoraise=True)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
