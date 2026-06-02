#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency on HPC nodes
    psutil = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_ROW_RE = re.compile(
    r"^(?P<pheno>\S+)\s+FvLMM\s+(?P<nidv>\d+)\s+(?P<nsnp>\d+)\s+"
    r"(?P<pve>[-+0-9.eE]+)\s+(?P<mem_gb>[-+0-9.eE]+)\s+"
    r"(?P<ctime_s>[-+0-9.eE]+)\s+(?P<vtime_s>[-+0-9.eE]+)\s*$",
    flags=re.MULTILINE,
)
CPU_RSS_RE = re.compile(
    r"FvLMM:\s+avg CPU ~\s*(?P<cpu_pct>[-+0-9.eE]+)%\s+of\s+(?P<cpu_threads>\d+)\s+c,\s+peak RSS ~\s+(?P<peak_rss_gb>[-+0-9.eE]+)\s+G"
)
TOTAL_WALL_RE = re.compile(r"Finished\.\s+Total wall time:\s+(?P<wall_s>[-+0-9.eE]+)\s+seconds")


def _print(msg: str) -> None:
    print(msg, flush=True)
NULL_PVE_RE = re.compile(r"FvLMM:\s+PVE\(null\)\s+~\s+(?P<null_pve>[-+0-9.eE]+)")


@dataclass(frozen=True)
class Case:
    case_id: str
    phase: str
    threads: int
    proj_threads: int | None
    assoc_threads: int | None
    row_tile_mb: int | None
    col_block_mb: int | None
    scan_stage: str | None


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("empty integer list")
    return out


def _parse_token_list(text: str) -> list[str]:
    out = [part.strip().lower() for part in str(text).split(",") if part.strip()]
    if not out:
        raise ValueError("empty token list")
    return out


def _resolve_thread_token(token: str, threads: int) -> int | None:
    tok = str(token).strip().lower()
    if tok in ("", "auto", "none", "unset"):
        return None
    if tok in ("full", "t"):
        return max(1, int(threads))
    if tok == "half":
        return max(1, int(threads) // 2)
    if tok == "quarter":
        return max(1, int(threads) // 4)
    if tok == "eighth":
        return max(1, int(threads) // 8)
    return max(1, int(tok))


def _resolve_mb_token(token: str) -> int | None:
    tok = str(token).strip().lower()
    if tok in ("", "auto", "none", "unset"):
        return None
    return max(1, int(tok))


def _case_slug(case: Case) -> str:
    def fmt(v: Any) -> str:
        return "auto" if v is None else str(v)

    return (
        f"{case.case_id}__{case.phase}"
        f"__t{case.threads}"
        f"__proj{fmt(case.proj_threads)}"
        f"__assoc{fmt(case.assoc_threads)}"
        f"__rowmb{fmt(case.row_tile_mb)}"
        f"__colmb{fmt(case.col_block_mb)}"
        f"__scan{fmt(case.scan_stage)}"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _peak_rss_tree_bytes(proc: subprocess.Popen[str]) -> int:
    if psutil is None:
        return 0
    try:
        root = psutil.Process(proc.pid)
    except Exception:
        return 0
    total = 0
    try:
        total += int(root.memory_info().rss)
    except Exception:
        return 0
    try:
        for child in root.children(recursive=True):
            try:
                total += int(child.memory_info().rss)
            except Exception:
                continue
    except Exception:
        pass
    return total


def _tail_nonempty_lines(path: Path, max_lines: int) -> list[str]:
    if max_lines <= 0 or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    return lines[-max_lines:]


def _build_cases(args: argparse.Namespace) -> list[Case]:
    threads_list = _parse_int_list(args.threads_list)
    focus_threads = int(args.focus_threads or max(threads_list))
    proj_tokens = _parse_token_list(args.proj_threads_list)
    assoc_tokens = _parse_token_list(args.assoc_threads_list)
    row_tokens = _parse_token_list(args.row_tile_mb_list)
    col_tokens = _parse_token_list(args.col_block_mb_list)
    scan_tokens = _parse_token_list(args.scan_stage_list)

    cases: list[Case] = []
    seen: set[tuple[int, int | None, int | None, int | None, int | None, str | None]] = set()
    seq = 1

    def push(phase: str, threads: int, proj: str, assoc: str, row_mb: str, col_mb: str, scan: str) -> None:
        nonlocal seq
        proj_threads = _resolve_thread_token(proj, int(threads))
        assoc_threads = _resolve_thread_token(assoc, int(threads))
        row_tile_mb = _resolve_mb_token(row_mb)
        col_block_mb = _resolve_mb_token(col_mb)
        scan_stage = None if scan in ("", "auto", "none", "unset") else str(scan)
        key = (
            int(threads),
            proj_threads,
            assoc_threads,
            row_tile_mb,
            col_block_mb,
            scan_stage,
        )
        if key in seen:
            return
        seen.add(key)
        case = Case(
            case_id=f"run{seq:03d}",
            phase=phase,
            threads=int(threads),
            proj_threads=proj_threads,
            assoc_threads=assoc_threads,
            row_tile_mb=row_tile_mb,
            col_block_mb=col_block_mb,
            scan_stage=scan_stage,
        )
        cases.append(case)
        seq += 1

    for threads in threads_list:
        push("threads", threads, "auto", "auto", "auto", "auto", "auto")

    for proj in proj_tokens:
        for assoc in assoc_tokens:
            push("split", focus_threads, proj, assoc, "auto", "auto", "auto")

    tile_threads = int(args.tile_threads or focus_threads)
    for row_mb in row_tokens:
        for col_mb in col_tokens:
            push(
                "tiles",
                tile_threads,
                str(args.tile_proj_threads),
                str(args.tile_assoc_threads),
                row_mb,
                col_mb,
                "auto",
            )

    scan_threads = int(args.scan_stage_threads or focus_threads)
    for scan in scan_tokens:
        push(
            "scan_stage",
            scan_threads,
            str(args.scan_stage_proj_threads),
            str(args.scan_stage_assoc_threads),
            "auto",
            "auto",
            scan,
        )

    return cases


def _build_env(base_env: dict[str, str], case: Case, cache_root: Path) -> dict[str, str]:
    env = dict(base_env)
    env["MPLCONFIGDIR"] = str(cache_root / "mpl")
    env["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    for key in (
        "JX_FVLMM_PROJ_THREADS",
        "JX_FVLMM_ASSOC_THREADS",
        "JX_GWAS_ROTATE_ROW_TILE_MB",
        "JX_GWAS_ROTATE_COL_BLOCK_MB",
        "JX_FVLMM_SCAN_STAGE",
    ):
        env.pop(key, None)
    if case.proj_threads is not None:
        env["JX_FVLMM_PROJ_THREADS"] = str(case.proj_threads)
    if case.assoc_threads is not None:
        env["JX_FVLMM_ASSOC_THREADS"] = str(case.assoc_threads)
    if case.row_tile_mb is not None:
        env["JX_GWAS_ROTATE_ROW_TILE_MB"] = str(case.row_tile_mb)
    if case.col_block_mb is not None:
        env["JX_GWAS_ROTATE_COL_BLOCK_MB"] = str(case.col_block_mb)
    if case.scan_stage is not None:
        env["JX_FVLMM_SCAN_STAGE"] = str(case.scan_stage)
    return env


def _build_cmd(args: argparse.Namespace, out_dir: Path, threads: int) -> list[str]:
    cmd = [
        str(args.jx),
        "gwas",
        "-bfile",
        str(Path(args.bfile).resolve()),
        "-p",
        str(Path(args.pheno).resolve()),
        "-fvlmm",
        "-n",
        str(args.trait),
        "-o",
        str(out_dir),
        "-t",
        str(int(threads)),
        "-maf",
        str(float(args.maf)),
        "-geno",
        str(float(args.geno)),
        "-het",
        str(float(args.het)),
        "-q",
        str(args.qcov),
        "-model",
        str(args.model),
    ]
    grm = str(args.grm).strip()
    if grm:
        cmd.extend(["-k", str(Path(grm).resolve())])
    if bool(args.fast):
        cmd.append("-fast")
    return cmd


def _parse_log_metrics(log_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = SUMMARY_ROW_RE.search(log_text)
    if m:
        gd = m.groupdict()
        out.update(
            {
                "pheno": gd["pheno"],
                "nidv": int(gd["nidv"]),
                "nsnp": int(gd["nsnp"]),
                "pve": float(gd["pve"]),
                "summary_mem_gb": float(gd["mem_gb"]),
                "ctime_s": float(gd["ctime_s"]),
                "vtime_s": float(gd["vtime_s"]),
            }
        )
    cpu_m = CPU_RSS_RE.search(log_text)
    if cpu_m:
        out["avg_cpu_pct"] = float(cpu_m.group("cpu_pct"))
        out["cpu_threads_reported"] = int(cpu_m.group("cpu_threads"))
        out["log_peak_rss_gb"] = float(cpu_m.group("peak_rss_gb"))
    wall_m = TOTAL_WALL_RE.search(log_text)
    if wall_m:
        out["log_total_wall_s"] = float(wall_m.group("wall_s"))
    null_pve_m = NULL_PVE_RE.search(log_text)
    if null_pve_m:
        out["null_pve"] = float(null_pve_m.group("null_pve"))
    return out


def _run_case(args: argparse.Namespace, case: Case, out_root: Path) -> dict[str, Any]:
    case_dir = out_root / _case_slug(case)
    case_dir.mkdir(parents=True, exist_ok=True)
    cache_root = case_dir / ".runtime-cache"
    (cache_root / "mpl").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "stdout.log"
    cmd = _build_cmd(args, case_dir, case.threads)
    env = _build_env(os.environ.copy(), case, cache_root)
    t0 = time.perf_counter()
    peak_rss = 0
    next_heartbeat_s = float(max(0, int(args.heartbeat_seconds)))
    _print(f"[matrix] log={log_path}")
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while True:
            rc = proc.poll()
            peak_rss = max(peak_rss, _peak_rss_tree_bytes(proc))
            if rc is not None:
                break
            elapsed_s = time.perf_counter() - t0
            if next_heartbeat_s > 0.0 and elapsed_s >= next_heartbeat_s:
                tail = _tail_nonempty_lines(log_path, int(args.heartbeat_tail_lines))
                msg = (
                    f"[matrix] {case.case_id} elapsed={elapsed_s:.1f}s "
                    f"rss={peak_rss / (1024 ** 3):.2f}G"
                )
                if tail:
                    msg += f" tail={tail[-1]}"
                _print(msg)
                next_heartbeat_s += float(max(1, int(args.heartbeat_seconds)))
            time.sleep(0.2)
    wall_s = time.perf_counter() - t0
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    row: dict[str, Any] = asdict(case)
    row["status"] = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
    row["cmd"] = " ".join(shlex.quote(part) for part in cmd)
    row["env_overrides"] = json.dumps(
        {
            k: env[k]
            for k in (
                "JX_FVLMM_PROJ_THREADS",
                "JX_FVLMM_ASSOC_THREADS",
                "JX_GWAS_ROTATE_ROW_TILE_MB",
                "JX_GWAS_ROTATE_COL_BLOCK_MB",
                "JX_FVLMM_SCAN_STAGE",
            )
            if k in env
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    row["wall_s_measured"] = float(wall_s)
    row["peak_rss_gb_measured"] = float(peak_rss / (1024 ** 3)) if peak_rss > 0 else float("nan")
    row["log_path"] = str(log_path)
    row.update(_parse_log_metrics(log_text))
    if proc.returncode == 0:
        tsvs = sorted(case_dir.glob("*.fvlmm.tsv"))
        if len(tsvs) == 1:
            row["result_tsv"] = str(tsvs[0])
            row["result_sha256"] = _sha256(tsvs[0])
        else:
            row["result_tsv"] = ""
            row["result_sha256"] = ""
    else:
        row["result_tsv"] = ""
        row["result_sha256"] = ""
    return row


def _write_commands(cases: list[Case], args: argparse.Namespace, out_root: Path) -> None:
    commands_path = out_root / "commands.sh"
    with commands_path.open("w", encoding="utf-8") as fh:
        fh.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for case in cases:
            case_dir = out_root / _case_slug(case)
            env_bits: list[str] = []
            if case.proj_threads is not None:
                env_bits.append(f"JX_FVLMM_PROJ_THREADS={case.proj_threads}")
            if case.assoc_threads is not None:
                env_bits.append(f"JX_FVLMM_ASSOC_THREADS={case.assoc_threads}")
            if case.row_tile_mb is not None:
                env_bits.append(f"JX_GWAS_ROTATE_ROW_TILE_MB={case.row_tile_mb}")
            if case.col_block_mb is not None:
                env_bits.append(f"JX_GWAS_ROTATE_COL_BLOCK_MB={case.col_block_mb}")
            if case.scan_stage is not None:
                env_bits.append(f"JX_FVLMM_SCAN_STAGE={case.scan_stage}")
            env_bits.append(f"MPLCONFIGDIR={shlex.quote(str(case_dir / '.runtime-cache' / 'mpl'))}")
            env_bits.append(f"XDG_CACHE_HOME={shlex.quote(str(case_dir / '.runtime-cache' / 'xdg'))}")
            cmd = _build_cmd(args, case_dir, case.threads)
            line = ""
            if env_bits:
                line += "env " + " ".join(env_bits) + " "
            line += " ".join(shlex.quote(part) for part in cmd)
            fh.write(line + "\n")


def _write_summary(rows: list[dict[str, Any]], out_root: Path) -> None:
    if not rows:
        return
    first_hash = ""
    for row in rows:
        sha = str(row.get("result_sha256", "")).strip()
        if sha:
            first_hash = sha
            break
    for row in rows:
        sha = str(row.get("result_sha256", "")).strip()
        row["same_as_first"] = bool(first_hash and sha and sha == first_hash)

    summary_json = out_root / "summary.json"
    summary_tsv = out_root / "summary.tsv"
    summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run a staged HPC benchmark matrix for JX FvLMM. "
            "The matrix is split into thread sweep, proj/assoc split sweep, tile sweep, and scan-stage sweep."
        )
    )
    ap.add_argument("-bfile", "--bfile", required=True, type=str)
    ap.add_argument("-p", "--pheno", required=True, type=str)
    ap.add_argument("-n", "--trait", required=True, type=str)
    ap.add_argument("-o", "--out", required=True, type=str)
    ap.add_argument("-k", "--grm", type=str, default="")
    ap.add_argument("-q", "--qcov", type=str, default="0")
    ap.add_argument("-maf", "--maf", type=float, default=0.02)
    ap.add_argument("-geno", "--geno", type=float, default=0.05)
    ap.add_argument("-het", "--het", type=float, default=0.0)
    ap.add_argument("--model", type=str, default="add", choices=["add", "dom", "rec", "het"])
    ap.add_argument("--fast", action="store_true", default=False, help="Also pass -fast to jx gwas.")
    ap.add_argument("--jx", type=str, default="jx")
    ap.add_argument("--threads-list", type=str, default="12,24,48")
    ap.add_argument(
        "--proj-threads-list",
        type=str,
        default="auto,half,full",
        help="Split sweep. Tokens: auto, full, half, quarter, eighth, or integer.",
    )
    ap.add_argument(
        "--assoc-threads-list",
        type=str,
        default="auto,8,4,2,1",
        help="Split sweep. Tokens: auto, full, half, quarter, eighth, or integer.",
    )
    ap.add_argument(
        "--row-tile-mb-list",
        type=str,
        default="auto,4,6,8,12",
        help="Tile sweep. Tokens: auto or integer MB.",
    )
    ap.add_argument(
        "--col-block-mb-list",
        type=str,
        default="auto,8,12,16,24",
        help="Tile sweep. Tokens: auto or integer MB.",
    )
    ap.add_argument(
        "--scan-stage-list",
        type=str,
        default="auto,full,generic,blas_t_rayon_1",
        help="Scan-stage sweep. Tokens: auto, full, generic, blas_t_rayon_1.",
    )
    ap.add_argument("--focus-threads", type=int, default=0, help="Split sweep uses this thread count. Default=max(threads-list).")
    ap.add_argument("--tile-threads", type=int, default=0, help="Tile sweep uses this thread count. Default=focus-threads.")
    ap.add_argument(
        "--tile-proj-threads",
        type=str,
        default="auto",
        help="Tile sweep proj threads token.",
    )
    ap.add_argument(
        "--tile-assoc-threads",
        type=str,
        default="auto",
        help="Tile sweep assoc threads token.",
    )
    ap.add_argument("--scan-stage-threads", type=int, default=0, help="Scan-stage sweep uses this thread count. Default=focus-threads.")
    ap.add_argument("--scan-stage-proj-threads", type=str, default="auto")
    ap.add_argument("--scan-stage-assoc-threads", type=str, default="auto")
    ap.add_argument("--dry-run", action="store_true", default=False, help="Only write commands.sh and matrix.json without executing.")
    ap.add_argument("--keep-going", action="store_true", default=False, help="Continue after failed cases.")
    ap.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="Print a progress heartbeat while a case is running. Set 0 to disable.",
    )
    ap.add_argument(
        "--heartbeat-tail-lines",
        type=int,
        default=1,
        help="How many non-empty log lines to inspect for heartbeat status.",
    )
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cases = _build_cases(args)
    if not cases:
        raise RuntimeError("No benchmark cases were generated.")

    matrix_path = out_root / "matrix.json"
    matrix_path.write_text(json.dumps([asdict(case) for case in cases], indent=2), encoding="utf-8")
    _write_commands(cases, args, out_root)

    if args.dry_run:
        _print(f"[matrix] wrote {len(cases)} cases to {matrix_path}")
        _print(f"[matrix] commands: {out_root / 'commands.sh'}")
        return 0

    rows: list[dict[str, Any]] = []
    failures = 0
    for case in cases:
        _print(f"[matrix] {case.case_id} phase={case.phase} t={case.threads} proj={case.proj_threads} assoc={case.assoc_threads} row_mb={case.row_tile_mb} col_mb={case.col_block_mb} scan={case.scan_stage}")
        row = _run_case(args, case, out_root)
        rows.append(row)
        _write_summary(rows, out_root)
        if str(row.get("status")) != "ok":
            failures += 1
            print(f"[matrix] failed: {case.case_id} -> {row['status']}", file=sys.stderr, flush=True)
            if not args.keep_going:
                break

    _write_summary(rows, out_root)
    if failures > 0:
        print(f"[matrix] completed with {failures} failed case(s)", file=sys.stderr, flush=True)
        return 1
    _print(f"[matrix] completed {len(rows)} case(s)")
    _print(f"[matrix] summary: {out_root / 'summary.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
