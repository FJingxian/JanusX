from __future__ import annotations

import json
import os
import sqlite3
import hashlib
import gzip
import shutil
import subprocess
import sys
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any


_DB_NAME = "janusx_tasks.db"


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _new_load_id(used: set[str] | None = None) -> str:
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # 36^4 = 1,679,616
    for _ in range(2_000_000):
        lid = "".join(random.choice(chars) for _ in range(4))
        if used is None or lid not in used:
            return lid
    raise RuntimeError("failed to generate unique 4-char load id")


def _is_short_load_id(v: str) -> bool:
    s = str(v or "").strip()
    if len(s) != 4:
        return False
    for ch in s:
        if not (ch.isdigit() or ("A" <= ch <= "Z")):
            return False
    return True


def _normalize_genofile_kind(kind: str, genofile: str) -> str:
    k = str(kind or "").strip().lower()
    g = str(genofile or "").strip().lower()
    if k == "bfile":
        return "bfile"
    if k == "vcf":
        return "vcf"
    if k in {"tsv", "txt", "csv", "npy", "file"}:
        # Legacy "file" and file-like kinds are all treated as table/matrix input.
        if g.endswith(".vcf") or g.endswith(".vcf.gz"):
            return "vcf"
        return "tsv"
    if g.endswith(".vcf") or g.endswith(".vcf.gz"):
        return "vcf"
    return "tsv"


def resolve_jx_home() -> Path:
    """
    Resolve JanusX runtime home used for shared history DB.
    Priority:
      1) JX_HOME environment
      2) ~/JanusX/.janusx (launcher default)
      3) ~/.janusx (legacy fallback)
    """
    env_home = os.environ.get("JX_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()

    p1 = (Path.home() / "JanusX" / ".janusx").expanduser()
    p2 = (Path.home() / ".janusx").expanduser()

    # Prefer the home that already has a real history DB.
    # This avoids selecting an empty/new runtime dir while older history
    # actually exists in the other location.
    db_candidates: list[tuple[float, Path]] = []
    for p in (p1, p2):
        dbp = p / _DB_NAME
        try:
            if dbp.exists() and dbp.is_file() and dbp.stat().st_size > 0:
                db_candidates.append((float(dbp.stat().st_mtime), p))
        except Exception:
            continue
    if len(db_candidates) > 0:
        db_candidates.sort(key=lambda x: x[0], reverse=True)
        return db_candidates[0][1].resolve()

    if p1.exists():
        return p1.resolve()
    if p2.exists():
        return p2.resolve()
    return p1.resolve()


def resolve_db_path() -> Path:
    home = resolve_jx_home()
    home.mkdir(parents=True, exist_ok=True)
    return (home / _DB_NAME).resolve()


def _candidate_db_paths() -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    env_home = os.environ.get("JX_HOME", "").strip()
    homes = []
    if env_home:
        homes.append(Path(env_home).expanduser())
    homes.extend(
        [
            Path.home() / "JanusX" / ".janusx",
            Path.home() / ".janusx",
        ]
    )
    for h in homes:
        try:
            p = (h / _DB_NAME).expanduser().resolve()
        except Exception:
            continue
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def _connect(db_path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        uri = f"file:{str(db_path)}?mode=ro"
        return sqlite3.connect(uri, timeout=30, uri=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    # WAL/NORMAL are best-effort only. Reading old/shared DBs should not fail
    # just because PRAGMA cannot be applied in current environment.
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    return conn


def _init_db(conn: sqlite3.Connection, *, mutate: bool = True) -> None:
    if not mutate:
        return
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gwas_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            status TEXT NOT NULL,
            genofile TEXT NOT NULL,
            genofile_kind TEXT NOT NULL,
            phenofile TEXT NOT NULL,
            outprefix TEXT NOT NULL,
            log_file TEXT NOT NULL,
            result_files_json TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            args_json TEXT NOT NULL,
            error_text TEXT NOT NULL,
            genofile_md5 TEXT NOT NULL DEFAULT '',
            phenofile_md5 TEXT NOT NULL DEFAULT ''
        )
        """
    )
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(gwas_runs)").fetchall()}
    if "genofile_md5" not in cols:
        conn.execute(
            "ALTER TABLE gwas_runs ADD COLUMN genofile_md5 TEXT NOT NULL DEFAULT ''"
        )
    if "phenofile_md5" not in cols:
        conn.execute(
            "ALTER TABLE gwas_runs ADD COLUMN phenofile_md5 TEXT NOT NULL DEFAULT ''"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_gwas_runs_created_at "
        "ON gwas_runs(created_at DESC)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS postgwas_runs (
            history_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            phenotype TEXT NOT NULL,
            model TEXT NOT NULL,
            genotype TEXT NOT NULL,
            genotype_type TEXT NOT NULL,
            job_id TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            return_code INTEGER,
            job_dir TEXT NOT NULL,
            error_text TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_postgwas_runs_created_at "
        "ON postgwas_runs(created_at DESC)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS anno_registry (
            alias TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            md5 TEXT NOT NULL,
            chroms_json TEXT NOT NULL,
            imported_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_anno_registry_md5 "
        "ON anno_registry(md5)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_anno_registry_imported_at "
        "ON anno_registry(imported_at DESC)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS load_registry (
            load_id TEXT NOT NULL DEFAULT '',
            file_type TEXT NOT NULL,
            name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            file_name TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            md5 TEXT NOT NULL,
            source_mtime REAL NOT NULL DEFAULT 0,
            chroms_json TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            PRIMARY KEY(file_type, name)
        )
        """
    )
    load_cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(load_registry)").fetchall()}
    if "load_id" not in load_cols:
        conn.execute(
            "ALTER TABLE load_registry ADD COLUMN load_id TEXT NOT NULL DEFAULT ''"
        )
    # Backfill missing IDs for old rows before creating unique index.
    load_rows = conn.execute(
        "SELECT rowid, load_id FROM load_registry"
    ).fetchall()
    used_ids: set[str] = set()
    dup_or_empty_rowids: list[int] = []
    for rowid, lid in load_rows:
        sid = str(lid or "").strip().upper()
        if (sid == "") or (sid in used_ids) or (not _is_short_load_id(sid)):
            dup_or_empty_rowids.append(int(rowid))
            continue
        used_ids.add(sid)
    for rowid in dup_or_empty_rowids:
        nid = _new_load_id(used_ids)
        used_ids.add(nid)
        conn.execute(
            "UPDATE load_registry SET load_id = ? WHERE rowid = ?",
            (nid, rowid),
        )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_load_registry_load_id "
        "ON load_registry(load_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_load_registry_md5 "
        "ON load_registry(md5)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_load_registry_imported_at "
        "ON load_registry(imported_at DESC)"
    )
    conn.commit()


def _file_md5(path: str, chunk_size: int = 4 * 1024 * 1024) -> str:
    p = Path(str(path or "")).expanduser()
    if (not p.exists()) or (not p.is_file()):
        return ""
    h = hashlib.md5()
    try:
        with p.open("rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                h.update(data)
        return h.hexdigest()
    except Exception:
        return ""


def _genotype_input_md5(genofile: str, genofile_kind: str) -> str:
    kind = _normalize_genofile_kind(genofile_kind, genofile)
    if kind == "bfile":
        base = str(genofile or "").strip()
        bed = Path(f"{base}.bed").expanduser()
        bim = Path(f"{base}.bim").expanduser()
        fam = Path(f"{base}.fam").expanduser()
        files = [bed, bim, fam]
        if not all(p.exists() and p.is_file() for p in files):
            return ""
        h = hashlib.md5()
        try:
            for p in files:
                h.update(p.name.encode("utf-8", errors="ignore"))
                with p.open("rb") as f:
                    while True:
                        data = f.read(4 * 1024 * 1024)
                        if not data:
                            break
                        h.update(data)
            return h.hexdigest()
        except Exception:
            return ""
    return _file_md5(genofile)


def _file_stat_fingerprint(path: str) -> str:
    """
    Fast file fingerprint based on metadata only (size + mtime_ns).
    Avoids full-file MD5 scan for large inputs in GWAS run recording.
    """
    p = Path(str(path or "")).expanduser()
    if (not p.exists()) or (not p.is_file()):
        return ""
    try:
        st = p.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(float(st.st_mtime) * 1_000_000_000)))
        return f"stat:size={int(st.st_size)};mtime_ns={mtime_ns}"
    except Exception:
        return ""


def _genotype_input_stat_fingerprint(genofile: str, genofile_kind: str) -> str:
    """
    Fast genotype-input fingerprint:
      - bfile: combine (bed,bim,fam) basename + size + mtime_ns
      - others: file size + mtime_ns
    """
    kind = _normalize_genofile_kind(genofile_kind, genofile)
    if kind == "bfile":
        base = str(genofile or "").strip()
        files = [
            Path(f"{base}.bed").expanduser(),
            Path(f"{base}.bim").expanduser(),
            Path(f"{base}.fam").expanduser(),
        ]
        if not all(p.exists() and p.is_file() for p in files):
            return ""
        parts: list[str] = []
        try:
            for p in files:
                st = p.stat()
                mtime_ns = int(
                    getattr(st, "st_mtime_ns", int(float(st.st_mtime) * 1_000_000_000))
                )
                parts.append(
                    f"{p.name}:size={int(st.st_size)};mtime_ns={mtime_ns}"
                )
            return "bfile-stat:" + "|".join(parts)
        except Exception:
            return ""
    return _file_stat_fingerprint(genofile)


def upsert_postgwas_run(
    *,
    history_id: str,
    run_id: str,
    phenotype: str,
    model: str,
    genotype: str,
    genotype_type: str,
    job_id: str,
    status: str,
    created_at: str = "",
    started_at: str = "",
    finished_at: str = "",
    return_code: int | None = None,
    job_dir: str = "",
    error_text: str = "",
) -> Path:
    hid = str(history_id or "").strip()
    if hid == "":
        raise ValueError("history_id is required for upsert_postgwas_run")
    db_path = resolve_db_path()
    conn = _connect(db_path)
    try:
        _init_db(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO postgwas_runs (
                history_id, run_id, phenotype, model, genotype, genotype_type,
                job_id, status, created_at, started_at, finished_at,
                return_code, job_dir, error_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hid,
                str(run_id or ""),
                str(phenotype or ""),
                str(model or ""),
                str(genotype or ""),
                str(genotype_type or ""),
                str(job_id or ""),
                str(status or ""),
                str(created_at or _now_str()),
                str(started_at or ""),
                str(finished_at or ""),
                None if return_code is None else int(return_code),
                str(job_dir or ""),
                str(error_text or ""),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def get_postgwas_run(history_id: str) -> dict[str, Any] | None:
    hid = str(history_id or "").strip()
    if hid == "":
        return None
    row = None
    for db_path in _candidate_db_paths():
        if not db_path.exists():
            continue
        try:
            conn = _connect(db_path, readonly=True)
            try:
                row = conn.execute(
                    """
                    SELECT history_id, run_id, phenotype, model, genotype, genotype_type,
                           job_id, status, created_at, started_at, finished_at,
                           return_code, job_dir, error_text
                    FROM postgwas_runs
                    WHERE history_id = ?
                    """,
                    (hid,),
                ).fetchone()
            finally:
                conn.close()
            if row is not None:
                break
        except Exception:
            continue
    if row is None:
        return None
    return {
        "history_id": row[0],
        "run_id": row[1],
        "phenotype": row[2],
        "model": row[3],
        "genotype": row[4],
        "genotype_type": row[5],
        "job_id": row[6],
        "status": row[7],
        "created_at": row[8],
        "started_at": row[9],
        "finished_at": row[10],
        "return_code": row[11],
        "job_dir": row[12],
        "error_text": row[13],
    }


def _list_postgwas_runs_by_history_ids(history_ids: list[str]) -> dict[str, dict[str, Any]]:
    ids = [str(x).strip() for x in history_ids if str(x).strip()]
    if len(ids) == 0:
        return {}
    rows_all: list[Any] = []
    placeholders = ",".join(["?"] * len(ids))
    for db_path in _candidate_db_paths():
        if not db_path.exists():
            continue
        try:
            conn = _connect(db_path, readonly=True)
            try:
                rows = conn.execute(
                    f"""
                    SELECT history_id, run_id, phenotype, model, genotype, genotype_type,
                           job_id, status, created_at, started_at, finished_at,
                           return_code, job_dir, error_text
                    FROM postgwas_runs
                    WHERE history_id IN ({placeholders})
                    """,
                    tuple(ids),
                ).fetchall()
            finally:
                conn.close()
            if rows:
                rows_all.extend(rows)
        except Exception:
            continue

    out: dict[str, dict[str, Any]] = {}
    for row in rows_all:
        out[str(row[0])] = {
            "history_id": row[0],
            "run_id": row[1],
            "phenotype": row[2],
            "model": row[3],
            "genotype": row[4],
            "genotype_type": row[5],
            "job_id": row[6],
            "status": row[7],
            "created_at": row[8],
            "started_at": row[9],
            "finished_at": row[10],
            "return_code": row[11],
            "job_dir": row[12],
            "error_text": row[13],
        }
    return out


def record_gwas_run(
    *,
    run_id: str,
    status: str,
    genofile: str,
    genofile_kind: str,
    phenofile: str,
    outprefix: str,
    log_file: str,
    result_files: list[str],
    summary_rows: list[dict[str, Any]],
    args_data: dict[str, Any],
    error_text: str = "",
    created_at: str | None = None,
    genofile_md5: str | None = None,
    phenofile_md5: str | None = None,
) -> Path:
    """
    Record one GWAS CLI run into the shared JanusX DB.
    For speed, run-level input fingerprints use file metadata (size+mtime)
    instead of full-file MD5 scan.
    """
    kind_norm = _normalize_genofile_kind(genofile_kind, genofile)
    g_md5 = str(genofile_md5 or "").strip() or _genotype_input_stat_fingerprint(genofile, kind_norm)
    p_md5 = str(phenofile_md5 or "").strip() or _file_stat_fingerprint(phenofile)
    db_path = resolve_db_path()
    conn = _connect(db_path)
    try:
        _init_db(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO gwas_runs (
                run_id, created_at, finished_at, status,
                genofile, genofile_kind, phenofile, outprefix, log_file,
                result_files_json, summary_json, args_json, error_text,
                genofile_md5, phenofile_md5
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                str(created_at or _now_str()),
                _now_str(),
                str(status),
                str(genofile),
                str(kind_norm),
                str(phenofile),
                str(outprefix),
                str(log_file),
                json.dumps(list(result_files), ensure_ascii=False),
                json.dumps(list(summary_rows), ensure_ascii=False),
                json.dumps(dict(args_data), ensure_ascii=False),
                str(error_text or ""),
                str(g_md5),
                str(p_md5),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _decode_json_field(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return default


def _geno_display_name(genofile: str, genofile_kind: str) -> str:
    p = Path(str(genofile))
    name = p.name if p.name else str(genofile)
    # Keep genotype display as the original input basename.
    # - bfile: shows provided prefix name
    # - vcf/tsv/txt/csv/npy: shows filename with extension
    _ = _normalize_genofile_kind(genofile_kind, genofile)
    return name


def _basename_only(path: str) -> str:
    p = str(path).replace("\\", "/").rstrip("/")
    b = os.path.basename(p)
    return b if b else p


def _format_grm_display_from_args(args_data: dict[str, Any]) -> str:
    disp = str(args_data.get("grm_display", "")).strip()
    if disp:
        return disp
    raw = str(args_data.get("grm", "")).strip()
    if raw in {"1", "2"}:
        return raw
    if raw == "":
        return ""
    return _basename_only(raw)


def _format_qcov_display_from_args(args_data: dict[str, Any]) -> str:
    disp = str(args_data.get("qcov_display", "")).strip()
    if disp:
        return disp
    raw = str(args_data.get("qcov", "")).strip()
    if raw == "":
        return ""
    if raw.isdigit():
        return raw
    return _basename_only(raw)


def _format_cov_display_from_args(args_data: dict[str, Any]) -> str:
    disp = str(args_data.get("cov_display", "")).strip()
    if disp:
        return disp
    cov = args_data.get("cov", [])
    if not isinstance(cov, list):
        return ""
    parts: list[str] = []
    for x in cov:
        s = str(x).strip()
        if s == "":
            continue
        t = s.replace("：", ":")
        ps = [p.strip() for p in t.split(":")]
        if len(ps) in {2, 3} and ps[0] != "":
            try:
                pos = int(float(ps[1]))
                parts.append(f"{ps[0]}:{pos}")
                continue
            except Exception:
                pass
        parts.append(_basename_only(s))
    return ";".join(parts)


def _infer_model_from_result_file(path: str) -> str:
    low = str(path or "").lower()
    if low.endswith(".lmm.tsv"):
        return "LMM"
    if low.endswith(".lm.tsv"):
        return "LM"
    if low.endswith(".farmcpu.tsv"):
        return "FarmCPU"
    if ".lmm." in low:
        return "LMM"
    if ".farmcpu." in low:
        return "FarmCPU"
    if ".lm." in low:
        return "LM"
    return ""


def _safe_alias(alias: str) -> str:
    raw = str(alias or "").strip()
    if raw == "":
        return ""
    out = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        elif ch in (" ", "/"):
            out.append("_")
    return "".join(out).strip("_")


def _safe_load_type(file_type: str) -> str:
    t = str(file_type or "").strip().lower()
    if t in {"gff", "vcf", "bfile"}:
        return t
    return ""


def _safe_load_name(name: str) -> str:
    return _safe_alias(name)


def _safe_anno_filename(name: str) -> str:
    raw = str(name or "").strip().replace("\\", "/").split("/")[-1]
    out = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
    s = "".join(out).strip(".")
    low = s.lower()
    if not (
        low.endswith(".gff")
        or low.endswith(".gff3")
        or low.endswith(".bed")
        or low.endswith(".gff.gz")
        or low.endswith(".gff3.gz")
        or low.endswith(".bed.gz")
    ):
        return ""
    return s


def _safe_vcf_filename(name: str) -> str:
    raw = str(name or "").strip().replace("\\", "/").split("/")[-1]
    out = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
    s = "".join(out).strip(".")
    low = s.lower()
    if low.endswith(".vcf") or low.endswith(".vcf.gz"):
        return s
    return ""


def _safe_gwas_filename(name: str) -> str:
    raw = str(name or "").strip().replace("\\", "/").split("/")[-1]
    out = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
    s = "".join(out).strip(".")
    low = s.lower()
    if (
        low.endswith(".tsv")
        or low.endswith(".txt")
        or low.endswith(".csv")
        or low.endswith(".tsv.gz")
        or low.endswith(".txt.gz")
        or low.endswith(".csv.gz")
    ):
        return s
    return ""


def _split_auto_line(line: str) -> tuple[list[str], str]:
    t = str(line or "").rstrip("\n\r")
    if "\t" in t:
        return [x.strip() for x in t.split("\t")], "tab"
    if "," in t:
        return [x.strip() for x in t.split(",")], "csv"
    return t.strip().split(), "ws"


def _header_key(v: object) -> str:
    s = str(v or "").strip().lower()
    if s.startswith("#"):
        s = s[1:]
    return re.sub(r"[^a-z0-9]+", "", s)


_GWAS_CHR_CANDIDATES = [
    "#CHROM",
    "chrom",
    "chr",
    "chromosome",
    "chromosome_id",
    "chrom_id",
    "chrom_name",
    "chr_id",
    "seqid",
    "seqname",
    "contig",
    "scaffold",
]
_GWAS_POS_CANDIDATES = [
    "POS",
    "pos",
    "position",
    "bp",
    "ps",
    "base_pair",
    "basepair",
    "bp_position",
    "physical_position",
    "physical_pos",
    "coordinate",
    "coord",
    "location",
    "loc",
]
_GWAS_P_CANDIDATES = [
    "p",
    "pvalue",
    "p_value",
    "pval",
    "p_val",
    "pwald",
    "p_wald",
    "wald_p",
    "waldp",
    "p_lrt",
    "lrt_p",
    "prob",
]


def _pick_gwas_header_column(cols: list[str], candidates: list[str], *, kind: str = "") -> str | None:
    cmap = {_header_key(c): str(c) for c in cols}
    for c in candidates:
        hit = cmap.get(_header_key(c))
        if hit is not None:
            return hit
    best_col = None
    best_score = 0
    for c in cols:
        k = _header_key(c)
        if k == "":
            continue
        score = 0
        if kind == "chrom":
            if ("chromosome" in k) or k.startswith("chrom"):
                score = 100
            elif k.startswith("chr"):
                score = 95
            elif ("contig" in k) or ("scaffold" in k) or ("seqid" in k) or ("seqname" in k):
                score = 90
        elif kind == "pos":
            if k in {"pos", "bp", "ps"}:
                score = 100
            elif "position" in k:
                score = 98
            elif k.startswith("pos") or k.endswith("bp"):
                score = 95
            elif ("basepair" in k) or ("coordinate" in k) or ("location" in k):
                score = 90
        elif kind == "p":
            if k == "p":
                score = 100
            elif k in {"pvalue", "pval", "pwald", "pvaluewald", "waldp", "waldpvalue"}:
                score = 98
            elif k.startswith("pvalue") or k.startswith("pval"):
                score = 96
            elif ("pwald" in k) or ("wald" in k and k.startswith("p")):
                score = 94
            elif ("lrt" in k and k.startswith("p")):
                score = 90
        if score > best_score:
            best_score = score
            best_col = c
    if best_score >= 90:
        return str(best_col)
    return None


def _detect_gwas_columns(path: Path) -> tuple[str, str, str]:
    fh: Any = None
    try:
        low = path.name.lower()
        if low.endswith(".gz"):
            fh = gzip.open(path, "rt", encoding="utf-8", errors="replace")
        else:
            fh = path.open("r", encoding="utf-8", errors="replace")
        header_cols: list[str] = []
        for line in fh:
            t = line.strip()
            if t == "":
                continue
            if t.startswith("##"):
                continue
            header_cols, _ = _split_auto_line(t)
            break
        if len(header_cols) == 0:
            raise ValueError("GWAS file is empty.")
        c_chr = _pick_gwas_header_column(header_cols, _GWAS_CHR_CANDIDATES, kind="chrom")
        c_pos = _pick_gwas_header_column(header_cols, _GWAS_POS_CANDIDATES, kind="pos")
        c_p = _pick_gwas_header_column(header_cols, _GWAS_P_CANDIDATES, kind="p")
        if c_chr is None or c_pos is None or c_p is None:
            raise ValueError(
                "Cannot detect required columns (chrom/pos/pvalue)."
            )
        return str(c_chr), str(c_pos), str(c_p)
    finally:
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass


def _extract_chr_from_gwas_file(path: Path, *, max_lines: int = 300_000) -> list[str]:
    fh: Any = None
    try:
        low = path.name.lower()
        if low.endswith(".gz"):
            fh = gzip.open(path, "rt", encoding="utf-8", errors="replace")
        else:
            fh = path.open("r", encoding="utf-8", errors="replace")
        header_cols: list[str] = []
        mode = "tab"
        for line in fh:
            t = line.strip()
            if t == "":
                continue
            if t.startswith("##"):
                continue
            header_cols, mode = _split_auto_line(t)
            break
        if len(header_cols) == 0:
            return []
        c_chr = _pick_gwas_header_column(header_cols, _GWAS_CHR_CANDIDATES, kind="chrom")
        if c_chr is None:
            return []
        idx = -1
        for i, c in enumerate(header_cols):
            if str(c) == str(c_chr):
                idx = i
                break
        if idx < 0:
            return []
        chroms: set[str] = set()
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            t = line.strip()
            if t == "" or t.startswith("#"):
                continue
            if mode == "tab":
                parts = [x.strip() for x in t.split("\t")]
            elif mode == "csv":
                parts = [x.strip() for x in t.split(",")]
            else:
                parts = t.split()
            if idx >= len(parts):
                continue
            c = str(parts[idx]).strip()
            if c != "":
                chroms.add(c)
        out = list(chroms)
        out.sort(key=_chr_sort_key)
        return out
    except Exception:
        return []
    finally:
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass


def _strip_known_suffixes(file_name: str) -> str:
    s = str(file_name or "").strip()
    low = s.lower()
    for ext in (".tsv.gz", ".txt.gz", ".csv.gz", ".vcf.gz", ".gff3.gz", ".gff.gz", ".bed.gz"):
        if low.endswith(ext):
            return s[: -len(ext)]
    for ext in (".tsv", ".txt", ".csv", ".vcf", ".gff3", ".gff", ".bed", ".gz"):
        if low.endswith(ext):
            return s[: -len(ext)]
    return Path(s).stem


def _normalize_bfile_prefix(raw_path: str) -> Path:
    p = Path(str(raw_path or "").strip()).expanduser()
    s = str(p)
    low = s.lower()
    for ext in (".bed", ".bim", ".fam"):
        if low.endswith(ext):
            s = s[: -len(ext)]
            break
    return Path(s).expanduser().resolve()


def _source_exists_for_type(file_type: str, source_path: str) -> bool:
    t = _safe_load_type(file_type)
    if t == "bfile":
        base = str(source_path or "").strip()
        return all(Path(f"{base}{ext}").exists() for ext in (".bed", ".bim", ".fam"))
    p = Path(str(source_path or "")).expanduser()
    return p.exists() and p.is_file()


def _source_mtime_for_type(file_type: str, source_path: str) -> float:
    t = _safe_load_type(file_type)
    try:
        if t == "bfile":
            base = str(source_path or "").strip()
            vals = []
            for ext in (".bed", ".bim", ".fam"):
                p = Path(f"{base}{ext}")
                if not p.exists():
                    return 0.0
                vals.append(float(p.stat().st_mtime))
            return max(vals) if vals else 0.0
        p = Path(str(source_path or "")).expanduser()
        if not p.exists():
            return 0.0
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


def _source_md5_for_type(file_type: str, source_path: str) -> str:
    t = _safe_load_type(file_type)
    if t == "bfile":
        return _genotype_input_md5(source_path, "bfile")
    return _file_md5(source_path)


def _extract_chr_from_text_file(path: Path, *, max_lines: int = 300_000) -> list[str]:
    chroms: set[str] = set()
    try:
        low = path.name.lower()
        if low.endswith(".gz"):
            fh = gzip.open(path, "rt", encoding="utf-8", errors="replace")
        else:
            fh = path.open("r", encoding="utf-8", errors="replace")
        with fh as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                t = line.strip()
                if t == "":
                    continue
                if t.startswith("#") or t.lower().startswith("track ") or t.lower().startswith("browser "):
                    continue
                ps = t.split("\t")
                if len(ps) == 0:
                    continue
                c = ps[0].strip()
                if c != "":
                    chroms.add(c)
    except Exception:
        return []
    out = list(chroms)
    out.sort(key=_chr_sort_key)
    return out


def _extract_chr_from_bim_prefix(prefix: str, *, max_lines: int = 2_000_000) -> list[str]:
    p = Path(f"{prefix}.bim")
    chroms: set[str] = set()
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                t = line.strip()
                if t == "":
                    continue
                ps = t.split()
                if len(ps) == 0:
                    continue
                c = ps[0].strip()
                if c != "":
                    chroms.add(c)
    except Exception:
        return []
    out = list(chroms)
    out.sort(key=_chr_sort_key)
    return out


def _extract_chr_for_load(file_type: str, source_path: str) -> list[str]:
    t = _safe_load_type(file_type)
    if t == "gff":
        return _extract_anno_chroms(Path(source_path))
    if t == "vcf":
        return _extract_chr_from_text_file(Path(source_path))
    if t == "bfile":
        return _extract_chr_from_bim_prefix(source_path)
    return []


def _schedule_gzip_background(path: Path) -> bool:
    """
    Background-gzip the copied file and keep original.
    """
    p = path.resolve()
    if p.name.lower().endswith(".gz"):
        return False
    code = (
        "import gzip, shutil, sys, pathlib;"
        "p=pathlib.Path(sys.argv[1]);"
        "gz=p.with_suffix(p.suffix + '.gz');"
        "f_in=p.open('rb');"
        "f_out=gzip.open(gz,'wb');"
        "shutil.copyfileobj(f_in,f_out);"
        "f_out.close();"
        "f_in.close()"
    )
    kwargs: dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        creationflags = 0
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        kwargs["creationflags"] = creationflags
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen([sys.executable, "-c", code, str(p)], **kwargs)
        return True
    except Exception:
        return False


def register_loaded_file(file_type: str, name: str, source_path: str) -> dict[str, Any]:
    """
    Register input file metadata into DB.
    Syntax:
      jx --load <type> <name> <file>
    """
    t = _safe_load_type(file_type)
    if t == "":
        raise ValueError("type must be one of: gff, vcf, bfile")
    n = _safe_load_name(name)
    if n == "":
        raise ValueError("name is required (letters/numbers/_/-/.).")
    raw = str(source_path or "").strip()
    if raw == "":
        raise ValueError("file path is required.")

    if t == "bfile":
        src = _normalize_bfile_prefix(raw)
        file_name = src.name
        src_path = str(src)
    else:
        srcp = Path(raw).expanduser().resolve()
        src_path = str(srcp)
        file_name = srcp.name
        if t == "gff" and _safe_anno_filename(file_name) == "":
            raise ValueError("gff type requires .gff/.gff3/.bed (or .gz).")
        if t == "vcf" and _safe_vcf_filename(file_name) == "":
            raise ValueError("vcf type requires .vcf or .vcf.gz.")

    if not _source_exists_for_type(t, src_path):
        raise ValueError(f"source file not found or incomplete: {src_path}")

    src_md5 = _source_md5_for_type(t, src_path)
    src_mtime = _source_mtime_for_type(t, src_path)
    if src_md5 == "":
        raise ValueError("failed to compute source md5.")
    chroms = _extract_chr_for_load(t, src_path)
    chroms_json = json.dumps(chroms, ensure_ascii=False)
    imported_at = _now_str()

    jx_home = resolve_jx_home()
    load_dir = (jx_home / "loaded" / t / n).resolve()
    load_dir.mkdir(parents=True, exist_ok=True)

    if t in {"gff", "vcf"}:
        src_file = Path(src_path)
        copied = (load_dir / src_file.name).resolve()
        if not copied.exists():
            shutil.copy2(src_file, copied)
        stored_path = str(copied)
        gzip_started = bool(t in {"gff", "vcf"} and _schedule_gzip_background(copied))
    else:
        stored_path = src_path
        gzip_started = False

    db_path = resolve_db_path()
    conn = _connect(db_path)
    status = "imported"
    load_id = ""
    try:
        _init_db(conn)
        row = conn.execute(
            """
            SELECT source_path, load_id
            FROM load_registry
            WHERE file_type = ? AND name = ?
            """,
            (t, n),
        ).fetchone()
        if row is not None:
            old_source = str(row[0] or "")
            if old_source != src_path:
                raise ValueError(
                    f"{t}:{n} is already mapped to a different file: {old_source}"
                )
            load_id = str(row[1] or "").strip()
            status = "updated"
        if load_id == "":
            load_id = _new_load_id()
        conn.execute(
            """
            INSERT INTO load_registry(
                load_id,
                file_type, name, source_path, file_name, stored_path,
                md5, source_mtime, chroms_json, imported_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_type, name) DO UPDATE SET
                load_id=excluded.load_id,
                source_path=excluded.source_path,
                file_name=excluded.file_name,
                stored_path=excluded.stored_path,
                md5=excluded.md5,
                source_mtime=excluded.source_mtime,
                chroms_json=excluded.chroms_json,
                imported_at=excluded.imported_at
            """,
            (
                load_id,
                t,
                n,
                src_path,
                file_name,
                stored_path,
                src_md5,
                float(src_mtime),
                chroms_json,
                imported_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "id": load_id,
        "type": t,
        "name": n,
        "file_name": file_name,
        "source_path": src_path,
        "stored_path": stored_path,
        "md5": src_md5,
        "source_mtime": float(src_mtime),
        "chroms": chroms,
        "imported_at": imported_at,
        "status": status,
        "gzip_started": bool(gzip_started),
        "db_path": str(db_path),
    }


def register_loaded_gwas_result(source_path: str) -> dict[str, Any]:
    raw = str(source_path or "").strip()
    if raw == "":
        raise ValueError("gwas file path is required.")
    src = Path(raw).expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise ValueError(f"source file not found: {src}")
    if _safe_gwas_filename(src.name) == "":
        raise ValueError("gwas type requires .tsv/.txt/.csv (or .gz).")

    stem = _strip_known_suffixes(src.name)
    alias = _safe_load_name(stem)
    if alias == "":
        alias = f"gwas_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    load_meta = register_loaded_file("gwas", alias, str(src))
    c_chr, c_pos, c_p = _detect_gwas_columns(src)
    chroms = _extract_chr_from_gwas_file(src)
    model = _infer_model_from_result_file(str(src))
    if model == "":
        model = "LMM"
    run_id = f"load-gwas-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(0, 0xFFFFFF):06x}"
    pheno = _safe_alias(stem)
    if pheno == "":
        pheno = alias

    summary_rows = [
        {
            "phenotype": pheno,
            "model": model,
            "pheno_col_idx": -1,
            "result_file": str(src),
            "nidv": 0,
            "eff_snp": 0,
        }
    ]
    args_data = {
        "source": "jx-load-gwas",
        "model": "upload",
        "chr": c_chr,
        "pos": c_pos,
        "pvalue": c_p,
        "load_alias": alias,
    }
    created_at = _now_str()
    db_path = record_gwas_run(
        run_id=run_id,
        status="done",
        genofile=str(src),
        genofile_kind="tsv",
        phenofile=str(src),
        outprefix=str(src),
        log_file="",
        result_files=[str(src)],
        summary_rows=summary_rows,
        args_data=args_data,
        error_text="",
        created_at=created_at,
        genofile_md5="",
        phenofile_md5=_file_stat_fingerprint(str(src)),
    )
    return {
        "type": "gwas",
        "name": alias,
        "file_name": src.name,
        "source_path": str(src),
        "stored_path": str(load_meta.get("stored_path", "")),
        "id": str(load_meta.get("id", "")),
        "history_id": f"{run_id}|0",
        "run_id": run_id,
        "status": "imported",
        "model": model,
        "phenotype": pheno,
        "chr_col": c_chr,
        "pos_col": c_pos,
        "p_col": c_p,
        "chroms": chroms,
        "db_path": str(db_path),
    }


def list_loaded_files(limit: int = 1000, file_type: str = "") -> list[dict[str, Any]]:
    """
    status:
      0 -> normal
      1 -> source missing/incomplete
      2 -> source changed (mtime/md5 mismatch)
    """
    t = _safe_load_type(file_type) if str(file_type or "").strip() else ""
    db_path = resolve_db_path()
    if not db_path.exists():
        return []

    conn = None
    try:
        conn = _connect(db_path, readonly=False)
        _init_db(conn, mutate=True)
        if t:
            recs = conn.execute(
                """
                SELECT load_id, file_type, name, source_path, file_name, stored_path,
                       md5, source_mtime, chroms_json, imported_at
                FROM load_registry
                WHERE file_type = ?
                ORDER BY imported_at DESC
                LIMIT ?
                """,
                (t, int(max(1, limit))),
            ).fetchall()
        else:
            recs = conn.execute(
                """
                SELECT load_id, file_type, name, source_path, file_name, stored_path,
                       md5, source_mtime, chroms_json, imported_at
                FROM load_registry
                ORDER BY imported_at DESC
                LIMIT ?
                """,
                (int(max(1, limit)),),
            ).fetchall()
    except Exception:
        return []
    finally:
        if conn is not None:
            conn.close()

    out: list[dict[str, Any]] = []
    for r in recs:
        lid = str(r[0] or "")
        tp = str(r[1] or "")
        nm = str(r[2] or "")
        src = str(r[3] or "")
        fn = str(r[4] or "")
        stored = str(r[5] or "")
        md5 = str(r[6] or "")
        src_mtime = float(r[7] or 0.0)
        chroms = _decode_json_field(r[8], [])
        if not isinstance(chroms, list):
            chroms = []
        imported_at = str(r[9] or "")

        status = 0
        if not _source_exists_for_type(tp, src):
            status = 1
        else:
            now_mtime = _source_mtime_for_type(tp, src)
            if abs(float(now_mtime) - float(src_mtime)) > 1e-6:
                now_md5 = _source_md5_for_type(tp, src)
                if now_md5 != md5:
                    status = 2
        out.append(
            {
                "id": lid,
                "type": tp,
                "name": nm,
                "file_name": fn,
                "source_path": src,
                "stored_path": stored,
                "md5": md5,
                "source_mtime": src_mtime,
                "chroms": chroms,
                "imported_at": imported_at,
                "status": int(status),
            }
        )
    return out


def remove_loaded_file_by_id(load_id: str) -> dict[str, Any]:
    lid = str(load_id or "").strip()
    if lid == "":
        raise ValueError("load id is required")

    db_path = resolve_db_path()
    if not db_path.exists():
        raise ValueError("load registry DB not found")

    conn = _connect(db_path)
    try:
        _init_db(conn)
        row = conn.execute(
            """
            SELECT load_id, file_type, name, source_path, file_name, stored_path
            FROM load_registry
            WHERE load_id = ?
            """,
            (lid,),
        ).fetchone()
        if row is None:
            raise ValueError(f"load id not found: {lid}")
        conn.execute(
            "DELETE FROM load_registry WHERE load_id = ?",
            (lid,),
        )
        conn.commit()
    finally:
        conn.close()

    tp = str(row[1] or "")
    nm = str(row[2] or "")
    src = str(row[3] or "")
    fn = str(row[4] or "")
    stored = str(row[5] or "")

    removed_paths: list[str] = []
    failed_paths: list[str] = []
    load_dir = (resolve_jx_home() / "loaded" / tp / nm).resolve()
    if load_dir.exists():
        try:
            shutil.rmtree(load_dir)
            removed_paths.append(str(load_dir))
        except Exception as e:
            failed_paths.append(f"{load_dir}: {e}")

    return {
        "id": lid,
        "type": tp,
        "name": nm,
        "file_name": fn,
        "source_path": src,
        "stored_path": stored,
        "removed_paths": removed_paths,
        "failed_paths": failed_paths,
        "db_path": str(db_path),
    }


def _chr_sort_key(chrom: str) -> tuple[int, int, str]:
    s = str(chrom or "").strip()
    if s == "":
        return (2, 10**9, "")
    low = s.lower()
    if low.startswith("chr"):
        s2 = s[3:]
    else:
        s2 = s
    if s2.isdigit():
        return (0, int(s2), s)
    return (1, 10**9, s.lower())


def _extract_anno_chroms(path: Path) -> list[str]:
    chroms: set[str] = set()
    try:
        low = path.name.lower()
        if low.endswith(".gz"):
            fh = gzip.open(path, "rt", encoding="utf-8", errors="replace")
        else:
            fh = path.open("r", encoding="utf-8", errors="replace")
        with fh as f:
            for line in f:
                t = line.strip()
                if t == "":
                    continue
                if t.startswith("##sequence-region"):
                    ps = t.split()
                    if len(ps) >= 2:
                        c = ps[1].strip()
                        if c != "":
                            chroms.add(c)
                    continue
                low = t.lower()
                if t.startswith("#") or low.startswith("track ") or low.startswith("browser "):
                    continue
                ps = t.split("\t")
                if len(ps) == 0:
                    continue
                c = ps[0].strip()
                if c != "":
                    chroms.add(c)
    except Exception:
        return []
    out = list(chroms)
    out.sort(key=_chr_sort_key)
    return out


def register_annotation_file(alias: str, file_path: str) -> dict[str, Any]:
    out = register_loaded_file("gff", alias, file_path)
    return {
        "id": str(out.get("id", "")),
        "alias": str(out.get("name", "")),
        "file_name": str(out.get("file_name", "")),
        "stored_path": str(out.get("stored_path", "")),
        "md5": str(out.get("md5", "")),
        "chroms": list(out.get("chroms", [])),
        "imported_at": str(out.get("imported_at", "")),
        "status": str(out.get("status", "")),
        "db_path": str(out.get("db_path", "")),
    }


def list_annotation_registry(limit: int = 500) -> list[dict[str, Any]]:
    rows = list_loaded_files(limit=limit, file_type="gff")
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": str(r.get("id", "")),
                "alias": str(r.get("name", "")),
                "file_name": str(r.get("file_name", "")),
                "path": str(r.get("stored_path") or r.get("source_path") or ""),
                "md5": str(r.get("md5", "")),
                "chroms": list(r.get("chroms", [])),
                "imported_at": str(r.get("imported_at", "")),
                "status": int(r.get("status", 0) or 0),
            }
        )
    return out


def _build_fallback_summary_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    result_files = run.get("result_files", [])
    if not isinstance(result_files, list):
        result_files = []
    phenotype_fallback = Path(str(run.get("phenofile", "")).strip() or "phenotype").stem
    for i, rf in enumerate(result_files):
        p = str(rf or "").strip()
        if p == "" or not p.lower().endswith(".tsv"):
            continue
        model = _infer_model_from_result_file(p)
        if model == "":
            continue
        rows.append(
            {
                "phenotype": phenotype_fallback,
                "model": model,
                "pheno_col_idx": -1,
                "result_file": p,
                "nidv": 0,
                "eff_snp": 0,
            }
        )
    return rows


def list_gwas_runs(limit: int = 200) -> list[dict[str, Any]]:
    rows_all: list[Any] = []
    for db_path in _candidate_db_paths():
        if not db_path.exists():
            continue
        try:
            conn = _connect(db_path, readonly=True)
            try:
                rows = conn.execute(
                    """
                    SELECT run_id, created_at, finished_at, status, genofile, genofile_kind,
                           phenofile, outprefix, log_file, result_files_json, summary_json,
                           args_json, error_text, genofile_md5, phenofile_md5
                    FROM gwas_runs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (int(max(1, limit)),),
                ).fetchall()
            finally:
                conn.close()
            if rows:
                rows_all.extend(rows)
        except Exception:
            continue
    if len(rows_all) == 0:
        return []

    out: list[dict[str, Any]] = []
    for row in rows_all:
        result_files = _decode_json_field(row[9], [])
        summary_rows = _decode_json_field(row[10], [])
        args_data = _decode_json_field(row[11], {})
        out.append(
            {
                "run_id": row[0],
                "created_at": row[1],
                "finished_at": row[2],
                "status": row[3],
                "genofile": row[4],
                "genofile_kind": _normalize_genofile_kind(row[5], row[4]),
                "phenofile": row[6],
                "outprefix": row[7],
                "log_file": row[8],
                "result_files": result_files,
                "summary_rows": summary_rows,
                "args": args_data,
                "error_text": row[12],
                "genofile_md5": row[13] if len(row) > 13 else "",
                "phenofile_md5": row[14] if len(row) > 14 else "",
                "result_count": len(result_files) if isinstance(result_files, list) else 0,
            }
        )
    # Deduplicate same run_id from multiple DB candidates; keep latest created_at.
    by_run: dict[str, dict[str, Any]] = {}
    for r in out:
        rid = str(r.get("run_id", ""))
        if rid == "":
            continue
        old = by_run.get(rid)
        if old is None or str(r.get("created_at", "")) >= str(old.get("created_at", "")):
            by_run[rid] = r
    out2 = list(by_run.values()) if len(by_run) > 0 else out
    out2.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    if len(out2) > int(max(1, limit)):
        out2 = out2[: int(max(1, limit))]
    return out2


def list_gwas_history_rows(limit: int = 500) -> list[dict[str, Any]]:
    """
    Flatten GWAS run records into row-wise history:
      genotype | phenotype | model | genotype_type | date
    Each row maps to one (phenotype, model) summary item.
    """
    runs = list_gwas_runs(limit=limit)
    out: list[dict[str, Any]] = []
    ok_status = {"done", "success", "finished", "ok", "complete", "completed"}
    for run in runs:
        st = str(run.get("status", "")).strip().lower()
        if st not in ok_status:
            continue
        run_id = str(run.get("run_id", ""))
        created_at = str(run.get("created_at", ""))
        genofile = str(run.get("genofile", ""))
        genofile_kind = _normalize_genofile_kind(
            str(run.get("genofile_kind", "")),
            str(run.get("genofile", "")),
        )
        phenofile = str(run.get("phenofile", ""))
        genofile_md5 = str(run.get("genofile_md5", "")).strip()
        phenofile_md5 = str(run.get("phenofile_md5", "")).strip()
        args_data = run.get("args", {})
        if not isinstance(args_data, dict):
            args_data = {}
        genotype_type = ""
        genotype_type = str(args_data.get("model", "")).strip()
        grm_display = _format_grm_display_from_args(args_data)
        qcov_display = _format_qcov_display_from_args(args_data)
        cov_display = _format_cov_display_from_args(args_data)
        genotype = _geno_display_name(genofile, genofile_kind)
        summary_rows = run.get("summary_rows", [])
        if not isinstance(summary_rows, list):
            summary_rows = []
        if len(summary_rows) == 0:
            summary_rows = _build_fallback_summary_rows(run)
        for idx, srow in enumerate(summary_rows):
            if not isinstance(srow, dict):
                continue
            pheno = str(srow.get("phenotype", "")).strip()
            model = str(srow.get("model", "")).strip()
            pheno_col_idx = int(srow.get("pheno_col_idx", -1))
            result_file = str(srow.get("result_file", "")).strip()
            if pheno == "" or model == "":
                continue
            history_id = f"{run_id}|{idx}"
            out.append(
                {
                    "history_id": history_id,
                    "run_id": run_id,
                    "created_at": created_at,
                    "source": str(args_data.get("source", "")).strip(),
                    "genotype": genotype,
                    "phenotype": pheno,
                    "model": model,
                    "genotype_type": genotype_type,
                    "pheno_col_idx": pheno_col_idx,
                    "genofile": genofile,
                    "genofile_kind": genofile_kind,
                    "phenofile": phenofile,
                    "genofile_md5": genofile_md5,
                    "phenofile_md5": phenofile_md5,
                    "result_file": result_file,
                    "nidv": int(srow.get("nidv", 0) or 0),
                    "eff_snp": int(srow.get("eff_snp", 0) or 0),
                    "grm": grm_display,
                    "qcov": qcov_display,
                    "cov": cov_display,
                }
            )
    out.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    if len(out) > 0:
        dedup: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, ...]] = set()
        for row in out:
            src = str(row.get("source", "")).strip().lower()
            gm = str(row.get("genofile_md5", "")).strip()
            pm = str(row.get("phenofile_md5", "")).strip()
            pheno = str(row.get("phenotype", "")).strip()
            model = str(row.get("model", "")).strip()
            gtype = str(row.get("genotype_type", "")).strip()
            if src in {"webui-upload", "jx-load-gwas"}:
                key = (f"__history__{str(row.get('history_id', ''))}",)
            elif gm != "" and pm != "" and pheno != "" and model != "" and gtype != "":
                key = (gm, pm, pheno, model, gtype)
            else:
                key = (f"__history__{str(row.get('history_id', ''))}",)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            dedup.append(row)
        out = dedup

    return out


def get_gwas_history_row(history_id: str) -> dict[str, Any] | None:
    hid = str(history_id or "").strip()
    if hid == "":
        return None
    if "|" not in hid:
        return None
    run_id, idx_s = hid.rsplit("|", 1)
    try:
        idx = int(idx_s)
    except Exception:
        return None
    run = get_gwas_run(run_id)
    if run is None:
        return None
    if str(run.get("status", "")).strip().lower() not in {
        "done",
        "success",
        "finished",
        "ok",
        "complete",
        "completed",
    }:
        return None
    summary_rows = run.get("summary_rows", [])
    if not isinstance(summary_rows, list):
        return None
    if len(summary_rows) == 0:
        summary_rows = _build_fallback_summary_rows(run)
    if idx < 0 or idx >= len(summary_rows):
        return None
    srow = summary_rows[idx]
    if not isinstance(srow, dict):
        return None
    args_data = run.get("args", {})
    if not isinstance(args_data, dict):
        args_data = {}
    genotype_type = ""
    genotype_type = str(args_data.get("model", "")).strip()
    grm_display = _format_grm_display_from_args(args_data)
    qcov_display = _format_qcov_display_from_args(args_data)
    cov_display = _format_cov_display_from_args(args_data)
    out = {
        "history_id": hid,
        "run_id": str(run.get("run_id", "")),
        "created_at": str(run.get("created_at", "")),
        "outprefix": str(run.get("outprefix", "")),
        "log_file": str(run.get("log_file", "")),
        "genotype": _geno_display_name(
            str(run.get("genofile", "")),
            str(run.get("genofile_kind", "")),
        ),
        "phenotype": str(srow.get("phenotype", "")).strip(),
        "model": str(srow.get("model", "")).strip(),
        "genotype_type": genotype_type,
        "pheno_col_idx": int(srow.get("pheno_col_idx", -1)),
        "genofile": str(run.get("genofile", "")),
        "genofile_kind": _normalize_genofile_kind(
            str(run.get("genofile_kind", "")),
            str(run.get("genofile", "")),
        ),
        "phenofile": str(run.get("phenofile", "")),
        "genofile_md5": str(run.get("genofile_md5", "")),
        "phenofile_md5": str(run.get("phenofile_md5", "")),
        "result_file": str(srow.get("result_file", "")).strip(),
        "nidv": int(srow.get("nidv", 0) or 0),
        "eff_snp": int(srow.get("eff_snp", 0) or 0),
        "grm": grm_display,
        "qcov": qcov_display,
        "cov": cov_display,
    }
    return out


def get_gwas_run(run_id: str) -> dict[str, Any] | None:
    rid = str(run_id or "").strip()
    if rid == "":
        return None
    row = None
    for db_path in _candidate_db_paths():
        if not db_path.exists():
            continue
        try:
            conn = _connect(db_path, readonly=True)
            try:
                row = conn.execute(
                    """
                    SELECT run_id, created_at, finished_at, status, genofile, genofile_kind,
                           phenofile, outprefix, log_file, result_files_json, summary_json,
                           args_json, error_text, genofile_md5, phenofile_md5
                    FROM gwas_runs
                    WHERE run_id = ?
                    """,
                    (rid,),
                ).fetchone()
            finally:
                conn.close()
            if row is not None:
                break
        except Exception:
            continue
    if row is None:
        return None
    result_files = _decode_json_field(row[9], [])
    summary_rows = _decode_json_field(row[10], [])
    args_data = _decode_json_field(row[11], {})
    return {
        "run_id": row[0],
        "created_at": row[1],
        "finished_at": row[2],
        "status": row[3],
        "genofile": row[4],
        "genofile_kind": _normalize_genofile_kind(row[5], row[4]),
        "phenofile": row[6],
        "outprefix": row[7],
        "log_file": row[8],
        "result_files": result_files,
        "summary_rows": summary_rows,
        "args": args_data,
        "error_text": row[12],
        "genofile_md5": row[13] if len(row) > 13 else "",
        "phenofile_md5": row[14] if len(row) > 14 else "",
        "result_count": len(result_files) if isinstance(result_files, list) else 0,
    }
