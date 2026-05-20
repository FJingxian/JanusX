#!/usr/bin/env python3
from __future__ import annotations

import ast
import argparse
import glob
import hashlib
import importlib.util
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    TaskProgressColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    MofNCompleteColumn = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    TaskID = int  # type: ignore[misc,assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
DEFAULT_GETCHUNKS = Path("/Users/jingxianfu/script/progress/scripts/GetChunks.py")
MAF_PAIR_ORDER = ["LL", "LM", "LH", "MM", "MH", "HH"]
LD_PAIR_ORDER = ["LL", "LH", "HH"]
LD3_ORDER = ["low", "mid", "high"]
MAF3_LABELS = ["low", "mid", "high"]
LD2_LABELS = ["low", "high"]
LAYER_NAMES = {0: "left_third", 1: "middle_third", 2: "right_third"}
MAF3_SHORT = {"low": "L", "mid": "M", "high": "H"}
LD2_SHORT = {"low": "L", "high": "H"}
CHUNK_VIEW_LABELS = {"L1": "front", "L2": "mid", "L3": "back"}


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    chrom: str
    start_bp: int
    end_bp: int
    center_bp: int
    layer_index: int
    layer_name: str
    repeat_index: int
    chunk_prefix: str


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    family: str
    scenario: str
    replicate: int
    chunk_id: str | None
    chunk_range: tuple[str, int, int] | None
    causal_size: int
    logic_mode: str | None
    selected_sites: tuple[tuple[str, int], ...]
    selected_groups: tuple[str, ...]
    sim_input_mode: str
    subset_key: str | None
    notes: tuple[str, ...] = field(default_factory=tuple)


def _short_text(text: object, width: int = 60) -> str:
    raw = str(text).strip()
    if len(raw) <= width:
        return raw
    if width <= 3:
        return raw[:width]
    return raw[: width - 3] + "..."


def _slugify(text: object, *, default: str = "cmd", max_len: int = 72) -> str:
    raw = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text).strip()).strip("-_.").lower()
    if raw == "":
        raw = default
    return raw[:max_len]


class WorkflowUI:
    def __init__(self) -> None:
        self.console = Console(soft_wrap=True) if Console is not None else None
        self.progress = (
            Progress(
                SpinnerColumn(style="cyan", finished_text="[green]OK[/green]"),
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=28),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TextColumn("{task.fields[current]}", style="dim"),
                TimeElapsedColumn(),
                expand=True,
                console=self.console,
                transient=False,
            )
            if Progress is not None and self.console is not None
            else None
        )
        self._started = False

    def print_header(self, *, title: str, rows: Sequence[tuple[str, object]]) -> None:
        if self.console is None or Panel is None or Table is None:
            print(title)
            for key, value in rows:
                print(f"  {key}: {value}")
            return
        table = Table.grid(expand=False, padding=(0, 1))
        table.add_column(style="bold cyan", justify="right", no_wrap=True)
        table.add_column(style="white")
        for key, value in rows:
            table.add_row(str(key), _short_text(value, 72))
        self.console.print(Panel.fit(table, title=title, border_style="cyan"))

    def start(self) -> None:
        if self.progress is not None and not self._started:
            self.progress.start()
            self._started = True

    def stop(self) -> None:
        if self.progress is not None and self._started:
            self.progress.stop()
            self._started = False

    def add_stage(self, description: str, total: int, current: str = "queued") -> TaskID | None:
        if self.progress is None:
            if self.console is not None:
                self.console.print(f"[bold]{description}[/bold]")
            else:
                print(description)
            return None
        return self.progress.add_task(
            description,
            total=max(1, int(total)),
            current=_short_text(current, 54),
        )

    def update(
        self,
        task_id: TaskID | None,
        *,
        advance: int = 0,
        completed: int | None = None,
        current: str | None = None,
        total: int | None = None,
    ) -> None:
        if self.progress is None or task_id is None:
            return
        kwargs: dict[str, Any] = {}
        if advance:
            kwargs["advance"] = int(advance)
        if completed is not None:
            kwargs["completed"] = int(completed)
        if total is not None:
            kwargs["total"] = max(1, int(total))
        if current is not None:
            kwargs["current"] = _short_text(current, 54)
        self.progress.update(task_id, **kwargs)

    def complete(self, task_id: TaskID | None, *, current: str = "done") -> None:
        if self.progress is None or task_id is None:
            return
        task = self.progress.tasks[task_id]
        self.progress.update(
            task_id,
            completed=int(task.total),
            current=_short_text(current, 54),
        )

    def print_summary(self, *, rows: Sequence[tuple[str, object]], title: str = "Run Summary") -> None:
        if self.console is None or Panel is None or Table is None:
            print(title)
            for key, value in rows:
                print(f"  {key}: {value}")
            return
        table = Table.grid(expand=False, padding=(0, 1))
        table.add_column(style="bold green", justify="right", no_wrap=True)
        table.add_column(style="white")
        for key, value in rows:
            table.add_row(str(key), _short_text(value, 72))
        self.console.print(Panel.fit(table, title=title, border_style="green"))

    def print_warning(self, message: str) -> None:
        if self.console is not None:
            self.console.print(f"[yellow]{message}[/yellow]")
        else:
            print(message)


@dataclass
class WorkflowRuntime:
    logger: logging.Logger
    ui: WorkflowUI
    logs_dir: Path
    dry_run: bool
    cmd_counter: int = 0
    cmd_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        (self.logs_dir / "commands").mkdir(parents=True, exist_ok=True)

    def next_command_log_path(self, label: str) -> Path:
        with self.cmd_lock:
            self.cmd_counter += 1
            cmd_counter = self.cmd_counter
        return self.logs_dir / "commands" / f"{cmd_counter:04d}.{_slugify(label)}.log"


def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Tgarfield")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / "Tgarfield.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def _jx_env() -> dict[str, str]:
    env = os.environ.copy()
    py_path = str(PYTHON_ROOT)
    existing = str(env.get("PYTHONPATH", "")).strip()
    env["PYTHONPATH"] = py_path if existing == "" else py_path + os.pathsep + existing
    env.setdefault("JANUSX_ENTRYPOINT", "jxpy")
    tmp_root = os.path.join(tempfile.gettempdir(), "janusx_tgarfield_cache")
    env.setdefault("XDG_CACHE_HOME", tmp_root)
    env.setdefault("MPLCONFIGDIR", os.path.join(tmp_root, "mpl"))
    return env


def _thread_cap_env(threads: int) -> dict[str, str]:
    t = max(1, int(threads))
    text = str(t)
    return {
        "OMP_NUM_THREADS": text,
        "OPENBLAS_NUM_THREADS": text,
        "MKL_NUM_THREADS": text,
        "NUMEXPR_NUM_THREADS": text,
        "VECLIB_MAXIMUM_THREADS": text,
        "RAYON_NUM_THREADS": text,
    }


def _single_thread_env() -> dict[str, str]:
    return _thread_cap_env(1)


def _run_cmd(
    cmd: Sequence[str],
    *,
    runtime: WorkflowRuntime,
    desc: str,
    cwd: Path | None = None,
    log_label: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> Path:
    pretty = " ".join(str(x) for x in cmd)
    log_path = runtime.next_command_log_path(log_label or desc)
    runtime.logger.info("%s", desc)
    runtime.logger.info("CMD: %s", pretty)
    runtime.logger.info("LOG: %s", log_path)
    if runtime.dry_run:
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"# DRY RUN\n# DESC: {desc}\n# CWD: {cwd or REPO_ROOT}\n# CMD: {pretty}\n")
        return log_path

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# DESC: {desc}\n")
        fh.write(f"# CWD: {cwd or REPO_ROOT}\n")
        fh.write(f"# CMD: {pretty}\n\n")
        fh.flush()
        try:
            subprocess.run(
                [str(x) for x in cmd],
                check=True,
                cwd=str(cwd or REPO_ROOT),
                env={**_jx_env(), **(extra_env or {})},
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            runtime.logger.exception("Command failed: %s", desc)
            raise RuntimeError(f"{desc} failed; see log: {log_path}") from exc
    return log_path


def _jx_cmd(module: str, *args: object) -> list[str]:
    cmd = [sys.executable, "-m", "janusx.script.JanusX", str(module)]
    cmd.extend(str(x) for x in args)
    return cmd


def _normalize_prefix(path_or_prefix: str) -> str:
    p = str(path_or_prefix).strip()
    low = p.lower()
    if low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam"):
        return p[:-4]
    return p


def _plink_prefix_ready(prefix: str) -> bool:
    return all(Path(f"{prefix}.{ext}").is_file() for ext in ("bed", "bim", "fam"))


def _remove_plink_prefix(prefix: str) -> None:
    for ext in ("bed", "bim", "fam"):
        path = Path(f"{prefix}.{ext}")
        if path.exists():
            path.unlink()


def _read_fam_sample_ids(prefix: str) -> list[str]:
    sample_ids: list[str] = []
    with open(f"{prefix}.fam", "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            toks = line.split()
            if len(toks) < 2:
                raise ValueError(f"Malformed FAM at {prefix}.fam:{line_no}")
            sample_ids.append(str(toks[1]).strip())
    if not sample_ids:
        raise ValueError(f"No samples found in {prefix}.fam")
    return sample_ids


def _recommend_stage2_stream_chunk_size(n_samples: int, *, target_mb: int = 64) -> int:
    target_bytes = max(8, int(target_mb)) * 1024 * 1024
    bytes_per_snp = max(1, int(n_samples)) * 4
    chunk_size = max(256, target_bytes // bytes_per_snp)
    return int(min(8192, max(256, chunk_size)))


def _read_bim(prefix: str) -> pd.DataFrame:
    rows: list[tuple[str, str, float, int, str, str]] = []
    with open(f"{prefix}.bim", "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            toks = line.split()
            if len(toks) < 6:
                raise ValueError(f"Malformed BIM at {prefix}.bim:{line_no}")
            rows.append(
                (
                    str(toks[0]).strip(),
                    str(toks[1]).strip(),
                    float(toks[2]),
                    int(toks[3]),
                    str(toks[4]).strip(),
                    str(toks[5]).strip(),
                )
            )
    df = pd.DataFrame(rows, columns=["chrom", "snp", "cm", "pos", "a1", "a2"])
    if df.empty:
        raise ValueError(f"No variants found in {prefix}.bim")
    return df


def _chrom_order_from_bim(prefix: str) -> list[str]:
    bim = _read_bim(prefix)
    seen: set[str] = set()
    chroms: list[str] = []
    for chrom in bim["chrom"].astype(str).tolist():
        if chrom not in seen:
            seen.add(chrom)
            chroms.append(chrom)
    return chroms


def _parse_chr_list(raw: Sequence[str] | None, prefix: str) -> list[str]:
    if raw is None or len(raw) == 0:
        return _chrom_order_from_bim(prefix)
    out: list[str] = []
    for token in raw:
        for part in str(token).replace(",", " ").split():
            out.append(part.strip())
    return [x for x in out if x != ""]


def _parse_int_csv(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        val = part.strip()
        if val == "":
            continue
        out.append(int(val))
    if not out:
        raise ValueError(f"Empty integer CSV: {text!r}")
    return out


def _parse_logic_modes(text: str) -> list[str]:
    modes: list[str] = []
    for part in str(text).split(","):
        mode = part.strip().lower()
        if mode == "":
            continue
        if mode not in {"and", "or"}:
            raise ValueError(f"Unsupported logic mode: {mode}")
        modes.append(mode)
    if not modes:
        raise ValueError("At least one logic mode is required.")
    return modes


def _parse_ldsc_window(text: str) -> tuple[str, str]:
    raw = str(text).strip().lower().replace(" ", "")
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)([a-z]*)", raw)
    if m is None:
        raise ValueError(f"Invalid LD-score window: {text!r}")
    value = float(m.group(1))
    unit = str(m.group(2) or "")
    if unit in {"", "snp", "snps"}:
        if not float(value).is_integer():
            raise ValueError(f"SNP-count LD-score window must be integer: {text!r}")
        return ("variants", f"{int(round(value))}snp")
    if unit in {"b", "bp"}:
        return ("bp", f"{int(round(value))}b")
    if unit in {"kb", "mb", "cm"}:
        return ("bp" if unit in {"kb", "mb"} else "cm", raw)
    raise ValueError(f"Unsupported LD-score window unit: {text!r}")


def _locate_single_file(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected 1 file for pattern {pattern!r}, found {len(hits)}")
    return hits[0]


def _import_getchunks_module(path: Path, logger: logging.Logger):
    if not path.is_file():
        logger.warning("GetChunks script not found: %s", path)
        return None
    spec = importlib.util.spec_from_file_location("external_getchunks", str(path))
    if spec is None or spec.loader is None:
        logger.warning("Unable to load GetChunks module: %s", path)
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info("Loaded chunk sampling reference: %s", path)
    return module


def _read_chr_lengths(prefix: str, getchunks_module, logger: logging.Logger) -> dict[str, int]:
    if getchunks_module is not None and hasattr(getchunks_module, "read_chr_lengths_from_bim"):
        return dict(getchunks_module.read_chr_lengths_from_bim(prefix))
    logger.info("Falling back to local BIM length scan.")
    out: dict[str, int] = {}
    with open(f"{prefix}.bim", "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            toks = line.split()
            chrom = str(toks[0]).strip()
            pos = int(toks[3])
            prev = out.get(chrom, 0)
            if pos > prev:
                out[chrom] = pos
    return out


def _plan_chunks(
    prefix: str,
    chr_list: Sequence[str],
    *,
    repeats: int,
    chunk_len: int,
    out_dir: Path,
    prefix_tag: str,
    seed: int,
    getchunks_module,
    logger: logging.Logger,
) -> list[ChunkRecord]:
    chr_lengths = _read_chr_lengths(prefix, getchunks_module, logger)
    rng = random.Random(int(seed))
    half = chunk_len // 2
    records: list[ChunkRecord] = []
    for chrom in chr_list:
        if chrom not in chr_lengths:
            logger.warning("Chromosome %s not found in BIM; skipping chunk sampling.", chrom)
            continue
        chr_len = int(chr_lengths[chrom])
        layer_len = chr_len // 3
        for layer in range(3):
            start_region = layer * layer_len
            end_region = start_region + layer_len
            low = start_region + half
            high = end_region - half
            if high < low:
                center = max(1, chr_len // 2)
                centers = [center] * repeats
            else:
                centers = [rng.randint(low, high) for _ in range(repeats)]
            for rep_idx, center in enumerate(centers, start=1):
                start_bp = max(1, int(center) - half)
                end_bp = min(chr_len, int(center) + half)
                chunk_name = (
                    f"{prefix_tag}.chr{chrom}.L{layer + 1}.R{rep_idx}"
                    f".{start_bp}_{end_bp}"
                )
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_name,
                        chrom=str(chrom),
                        start_bp=int(start_bp),
                        end_bp=int(end_bp),
                        center_bp=int(center),
                        layer_index=int(layer),
                        layer_name=LAYER_NAMES[int(layer)],
                        repeat_index=int(rep_idx),
                        chunk_prefix=str(out_dir / chunk_name),
                    )
                )
    return records


def _extract_chunks(
    source_bfile: str,
    chunks: Sequence[ChunkRecord],
    *,
    out_dir: Path,
    threads: int,
    force: bool,
    runtime: WorkflowRuntime,
    task_id: TaskID | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chrom_groups: dict[str, list[ChunkRecord]] = {}
    pending = 0
    for chunk in chunks:
        runtime.ui.update(task_id, current=f"{chunk.chrom}:{chunk.start_bp}-{chunk.end_bp}")
        if (not force) and _plink_prefix_ready(chunk.chunk_prefix):
            runtime.logger.info("Reuse chunk: %s", chunk.chunk_prefix)
            runtime.ui.update(task_id, advance=1, current=f"reuse {Path(chunk.chunk_prefix).name}")
            continue
        chrom_groups.setdefault(str(chunk.chrom), []).append(chunk)
        pending += 1

    if pending == 0:
        return

    if runtime.dry_run:
        log_path = runtime.next_command_log_path("chunk.extract.gfreader")
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("# DRY RUN\n")
            fh.write("# DESC: Extract chunks with gfreader streaming backend\n")
            fh.write(f"# SOURCE: {source_bfile}\n")
            for chrom, chrom_chunks in sorted(chrom_groups.items()):
                fh.write(f"{chrom}\t{len(chrom_chunks)}\n")
        for chrom, chrom_chunks in sorted(chrom_groups.items()):
            runtime.ui.update(
                task_id,
                advance=len(chrom_chunks),
                current=f"dry-run chr{chrom} x {len(chrom_chunks)}",
            )
        return

    sample_ids = _read_fam_sample_ids(source_bfile)
    stream_chunk_size = _recommend_stage2_stream_chunk_size(len(sample_ids))
    max_workers = min(max(1, int(threads)), max(1, len(chrom_groups)))
    runtime.logger.info(
        "Stage 2 gfreader extraction: %d chunk(s), %d chromosome worker(s), stream_chunk_size=%d",
        pending,
        max_workers,
        stream_chunk_size,
    )

    future_map: dict[Future[dict[str, object]], tuple[str, list[ChunkRecord]]] = {}
    executor_name = "process"
    try:
        executor = ProcessPoolExecutor(max_workers=max_workers)
    except Exception as ex:
        runtime.logger.warning(
            "Stage 2 process-pool extraction unavailable (%s); falling back to thread pool.",
            ex,
        )
        executor = ThreadPoolExecutor(max_workers=max_workers)
        executor_name = "thread"

    with executor as pool:
        runtime.logger.info("Stage 2 worker backend: %s", executor_name)
        for chrom, chrom_chunks in sorted(chrom_groups.items()):
            payload = [
                (str(c.chunk_id), int(c.start_bp), int(c.end_bp), str(c.chunk_prefix))
                for c in sorted(chrom_chunks, key=lambda c: (int(c.start_bp), int(c.end_bp), str(c.chunk_id)))
            ]
            fut = pool.submit(
                _extract_chunks_for_chrom_worker,
                str(source_bfile),
                str(chrom),
                payload,
                list(sample_ids),
                str(PYTHON_ROOT),
                int(stream_chunk_size),
            )
            future_map[fut] = (str(chrom), chrom_chunks)

        try:
            for fut in future_map:
                chrom, chrom_chunks = future_map[fut]
                runtime.ui.update(task_id, current=f"chr{chrom} queued ({len(chrom_chunks)} chunk(s))")
            remaining = set(future_map.keys())
            while remaining:
                done, remaining = wait(remaining, return_when=FIRST_COMPLETED)
                for fut in done:
                    chrom, chrom_chunks = future_map[fut]
                    result = fut.result()
                    log_path = runtime.next_command_log_path(f"chunk.extract.chr{chrom}")
                    with open(log_path, "w", encoding="utf-8") as fh:
                        fh.write("# DESC: Stage 2 gfreader chromosome extraction\n")
                        fh.write(f"# SOURCE: {source_bfile}\n")
                        fh.write(f"# CHROM: {chrom}\n")
                        fh.write(f"# STREAM_CHUNK_SIZE: {stream_chunk_size}\n")
                        fh.write(f"# ELAPSED_SEC: {float(result.get('elapsed_sec', 0.0)):.6f}\n")
                        fh.write(f"# SCANNED_SITES: {int(result.get('scanned_sites', 0))}\n")
                        fh.write("chunk_id\tn_written\n")
                        for chunk_id, n_written in zip(
                            result.get("chunk_ids", []),
                            result.get("chunk_counts", []),
                        ):
                            fh.write(f"{chunk_id}\t{int(n_written)}\n")
                    runtime.logger.info(
                        "Extracted chr%s: %d chunk(s), scanned=%d, elapsed=%.2fs",
                        chrom,
                        len(chrom_chunks),
                        int(result.get("scanned_sites", 0)),
                        float(result.get("elapsed_sec", 0.0)),
                    )
                    runtime.ui.update(
                        task_id,
                        advance=len(chrom_chunks),
                        current=f"chr{chrom} done ({len(chrom_chunks)} chunk(s))",
                    )
        except Exception:
            for fut in future_map:
                fut.cancel()
            raise


def _extract_chunks_for_chrom_worker(
    source_bfile: str,
    chrom: str,
    chunk_payload: Sequence[tuple[str, int, int, str]],
    sample_ids: Sequence[str],
    python_root: str,
    stream_chunk_size: int,
) -> dict[str, object]:
    t0 = time.time()
    for key, value in _thread_cap_env(1).items():
        os.environ[key] = value
    py_root = str(python_root)
    if py_root not in sys.path:
        sys.path.insert(0, py_root)

    from janusx.gfreader import load_genotype_chunks
    from janusx.janusx import PlinkStreamWriter

    chunk_specs = sorted(
        [(str(cid), int(start), int(end), str(prefix)) for cid, start, end, prefix in chunk_payload],
        key=lambda x: (x[1], x[2], x[0]),
    )
    writers: dict[str, Any] = {}
    counts: dict[str, int] = {cid: 0 for cid, _s, _e, _p in chunk_specs}
    try:
        for chunk_id, _start, _end, chunk_prefix in chunk_specs:
            Path(chunk_prefix).parent.mkdir(parents=True, exist_ok=True)
            writers[chunk_id] = PlinkStreamWriter(str(chunk_prefix), list(sample_ids), None)

        scanned_sites = 0
        for geno, sites in load_genotype_chunks(
            str(source_bfile),
            chunk_size=max(256, int(stream_chunk_size)),
            maf=0.0,
            missing_rate=1.0,
            impute=False,
            model="add",
            het=0.02,
            chr_keys=[str(chrom)],
        ):
            sites_list = list(sites)
            if not sites_list:
                continue
            block = np.ascontiguousarray(np.asarray(geno, dtype=np.int8))
            positions = np.fromiter((int(s.pos) for s in sites_list), dtype=np.int64, count=len(sites_list))
            scanned_sites += int(len(sites_list))
            monotonic = bool(len(positions) <= 1 or np.all(positions[1:] >= positions[:-1]))

            for chunk_id, start_bp, end_bp, _chunk_prefix in chunk_specs:
                writer = writers[chunk_id]
                if monotonic:
                    left = int(np.searchsorted(positions, int(start_bp), side="left"))
                    right = int(np.searchsorted(positions, int(end_bp), side="right"))
                    if right <= left:
                        continue
                    sub_block = np.ascontiguousarray(block[left:right, :], dtype=np.int8)
                    writer.write_chunk(sub_block, sites_list[left:right])
                    counts[chunk_id] += int(right - left)
                else:
                    mask = (positions >= int(start_bp)) & (positions <= int(end_bp))
                    if not bool(np.any(mask)):
                        continue
                    idx = np.flatnonzero(mask).astype(int)
                    sub_block = np.ascontiguousarray(block[idx, :], dtype=np.int8)
                    writer.write_chunk(sub_block, [sites_list[int(i)] for i in idx.tolist()])
                    counts[chunk_id] += int(idx.size)

        for writer in writers.values():
            writer.close()

        empty = [chunk_id for chunk_id, n_written in counts.items() if int(n_written) <= 0]
        if empty:
            raise RuntimeError(
                f"Chromosome {chrom} produced empty extracted chunk(s): {', '.join(empty[:8])}"
            )

        chunk_ids = [cid for cid, _s, _e, _p in chunk_specs]
        chunk_counts = [int(counts[cid]) for cid in chunk_ids]
        return {
            "chrom": str(chrom),
            "chunk_ids": chunk_ids,
            "chunk_counts": chunk_counts,
            "scanned_sites": int(scanned_sites),
            "elapsed_sec": float(time.time() - t0),
        }
    except Exception:
        for _chunk_id, _start, _end, chunk_prefix in chunk_specs:
            _remove_plink_prefix(chunk_prefix)
        raise


def _filter_genotype(
    bfile: str,
    *,
    out_dir: Path,
    prefix: str,
    maf: float,
    geno: float,
    het: float,
    prune_window: str,
    prune_step: int,
    prune_r2: float,
    threads: int,
    force: bool,
    runtime: WorkflowRuntime,
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_prefix = str(out_dir / prefix)
    if (not force) and Path(f"{filtered_prefix}.bed").is_file():
        runtime.logger.info("Reuse filtered genotype: %s", filtered_prefix)
        return filtered_prefix
    cmd = _jx_cmd(
        "gformat",
        "-bfile",
        bfile,
        "-fmt",
        "plink",
        "-maf",
        maf,
        "-geno",
        geno,
        "-het",
        het,
        "-prune",
        prune_window,
        prune_step,
        prune_r2,
        "-o",
        out_dir,
        "-prefix",
        prefix,
        "-t",
        max(1, int(threads)),
    )
    _run_cmd(
        cmd,
        runtime=runtime,
        desc="Filter and LD-prune genotype with jx gformat",
        log_label="filter-genotype",
    )
    return filtered_prefix


def _run_chunk_gstats(
    chunk: ChunkRecord,
    *,
    out_dir: Path,
    ldsc_window: str,
    threads: int,
    force: bool,
    runtime: WorkflowRuntime,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outprefix = out_dir / chunk.chunk_id
    _, ldsc_label = _parse_ldsc_window(ldsc_window)
    freq_path = f"{outprefix}.freq"
    ldsc_glob = f"{outprefix}.*.{ldsc_label}.ldsc"
    have_freq = Path(freq_path).is_file()
    have_ld = len(glob.glob(ldsc_glob)) == 1
    need_run = bool(force or (not have_freq) or (not have_ld))
    if need_run:
        cmd = _jx_cmd(
            "gstats",
            "-bfile",
            chunk.chunk_prefix,
            "-freq",
            "-ldsc",
            ldsc_window,
            "-o",
            out_dir,
            "-prefix",
            chunk.chunk_id,
            "-t",
            max(1, int(threads)),
        )
        _run_cmd(
            cmd,
            runtime=runtime,
            desc=f"Run jx gstats on chunk {chunk.chunk_id}",
            log_label=f"gstats.{chunk.chunk_id}",
            extra_env=_thread_cap_env(threads),
        )
    else:
        runtime.logger.info("Reuse chunk gstats: %s", outprefix)
    if runtime.dry_run:
        return {"freq": freq_path, "ldsc": ldsc_glob}
    ld_hits = sorted(glob.glob(ldsc_glob))
    if len(ld_hits) != 1:
        raise FileNotFoundError(f"Expected exactly one LD-score output for {chunk.chunk_id}")
    return {"freq": freq_path, "ldsc": ld_hits[0]}


def _rank_qcut(values: pd.Series, q: int, labels: Sequence[str]) -> pd.Series:
    if len(values) < q:
        raise ValueError(f"Need at least {q} values for quantile binning, got {len(values)}")
    ranks = values.rank(method="first")
    bins = pd.qcut(ranks, q=q, labels=list(labels))
    return bins.astype(str)


def _build_chunk_site_stats(chunk: ChunkRecord, freq_path: str, ldsc_path: str) -> pd.DataFrame:
    freq = pd.read_csv(freq_path, sep="\t")
    ld = pd.read_csv(ldsc_path, sep="\t")
    freq["chrom"] = freq["chr"].astype(str)
    ld["chrom"] = ld["chr"].astype(str)
    merged = freq[["chrom", "pos", "freq"]].merge(
        ld[["chrom", "pos", "M", "ldsc"]],
        on=["chrom", "pos"],
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"No overlapping freq/ldsc rows for chunk {chunk.chunk_id}")
    merged["chunk_id"] = chunk.chunk_id
    merged["site_id"] = merged["chrom"].astype(str) + ":" + merged["pos"].astype(str)
    merged["site_key"] = merged["chrom"].astype(str) + "_" + merged["pos"].astype(str)
    merged["maf_bin3"] = _rank_qcut(merged["freq"], 3, MAF3_LABELS)
    merged["ld_bin2"] = _rank_qcut(merged["ldsc"], 2, LD2_LABELS)
    merged["ld_bin3"] = _rank_qcut(merged["ldsc"], 3, LD3_ORDER)
    maf_short = merged["maf_bin3"].map(MAF3_SHORT)
    ld_short = merged["ld_bin2"].map(LD2_SHORT)
    merged["combo_bin6"] = maf_short.astype(str) + ld_short.astype(str)
    return merged.sort_values(["chrom", "pos"]).reset_index(drop=True)


def _supported_pair_count(
    counts: pd.Series,
    *,
    labels: Sequence[str],
) -> int:
    total = 0
    for i, label_a in enumerate(labels):
        for label_b in labels[i:]:
            count_a = int(counts.get(label_a, 0))
            count_b = int(counts.get(label_b, 0))
            ok = count_a >= 2 if label_a == label_b else (count_a >= 1 and count_b >= 1)
            total += int(ok)
    return total


def _choose_anchor_chunk(
    chunk_stats: dict[str, pd.DataFrame],
    requested: str | None,
) -> str:
    if requested:
        for chunk_id in chunk_stats:
            if requested == chunk_id or requested in chunk_id:
                return chunk_id
        raise KeyError(f"Requested anchor chunk not found: {requested}")
    best_chunk = None
    best_score = None
    for chunk_id, df in chunk_stats.items():
        maf_counts = df["maf_bin3"].value_counts()
        ld2_counts = df["ld_bin2"].value_counts()
        ld3_counts = df["ld_bin3"].value_counts()
        maf_single_nonempty = sum(int(maf_counts.get(k, 0) > 0) for k in MAF3_LABELS)
        ld_single_nonempty = sum(int(ld3_counts.get(k, 0) > 0) for k in LD3_ORDER)
        maf_pair_nonempty = _supported_pair_count(maf_counts, labels=MAF3_LABELS)
        ld_pair_nonempty = _supported_pair_count(ld2_counts, labels=LD2_LABELS)
        score = (
            int(maf_pair_nonempty),
            int(ld_pair_nonempty),
            int(maf_single_nonempty),
            int(ld_single_nonempty),
            int(len(df)),
        )
        if best_score is None or score > best_score:
            best_chunk = chunk_id
            best_score = score
    if best_chunk is None:
        raise ValueError("No chunk stats available to choose an anchor chunk.")
    return best_chunk


def _sample_sites_from_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    group_value: str,
    n_sites: int,
    rng: np.random.Generator,
) -> list[tuple[str, int]]:
    pool = df.loc[df[group_col] == group_value, ["chrom", "pos"]]
    if len(pool) < n_sites:
        raise ValueError(
            f"Group {group_col}={group_value} has too few sites: need {n_sites}, got {len(pool)}"
        )
    pick = rng.choice(pool.index.to_numpy(dtype=int), size=n_sites, replace=False)
    out = pool.loc[pick].sort_values(["chrom", "pos"])
    return [(str(r.chrom), int(r.pos)) for r in out.itertuples(index=False)]


def _sample_pair_from_combo_bins(
    df: pd.DataFrame,
    group_a: str,
    group_b: str,
    rng: np.random.Generator,
) -> list[tuple[str, int]]:
    if group_a == group_b:
        return _sample_sites_from_group(df, group_col="combo_bin6", group_value=group_a, n_sites=2, rng=rng)
    site_a = _sample_sites_from_group(df, group_col="combo_bin6", group_value=group_a, n_sites=1, rng=rng)[0]
    site_b = _sample_sites_from_group(df, group_col="combo_bin6", group_value=group_b, n_sites=1, rng=rng)[0]
    if site_a == site_b:
        raise ValueError(f"Sampled duplicate sites for combo pair {group_a}/{group_b}")
    return [site_a, site_b]


def _sample_pair_from_group_bins(
    df: pd.DataFrame,
    *,
    group_col: str,
    group_a: str,
    group_b: str,
    rng: np.random.Generator,
) -> list[tuple[str, int]]:
    if group_a == group_b:
        return _sample_sites_from_group(df, group_col=group_col, group_value=group_a, n_sites=2, rng=rng)
    site_a = _sample_sites_from_group(df, group_col=group_col, group_value=group_a, n_sites=1, rng=rng)[0]
    site_b = _sample_sites_from_group(df, group_col=group_col, group_value=group_b, n_sites=1, rng=rng)[0]
    if site_a == site_b:
        raise ValueError(f"Sampled duplicate sites for pair {group_col}:{group_a}/{group_b}")
    return [site_a, site_b]


def _sample_random_sites(df: pd.DataFrame, n_sites: int, rng: np.random.Generator) -> list[tuple[str, int]]:
    if len(df) < n_sites:
        raise ValueError(f"Need {n_sites} sites, got {len(df)}")
    pick = rng.choice(df.index.to_numpy(dtype=int), size=n_sites, replace=False)
    out = df.loc[pick, ["chrom", "pos"]].sort_values(["chrom", "pos"])
    return [(str(r.chrom), int(r.pos)) for r in out.itertuples(index=False)]


def _collapse_logic_bin01_row(row: np.ndarray) -> np.ndarray | None:
    x = np.asarray(row, dtype=np.float32).reshape(-1)
    valid_mask = np.isfinite(x) & (x >= 0.0)
    if not bool(np.any(valid_mask)):
        return None
    valid = np.rint(x[valid_mask])
    g = np.where(valid <= 0.0, 0, np.where(valid >= 2.0, 2, 1)).astype(np.uint8, copy=False)
    c0 = int(np.sum(g == 0))
    c2 = int(np.sum(g == 2))
    mode02 = 2 if c2 > c0 else 0
    out = np.full(x.shape, mode02, dtype=np.uint8)
    out[valid_mask] = np.where(g == 1, mode02, g).astype(np.uint8, copy=False)
    return (out > 0).astype(np.uint8, copy=False)


def _load_chunk_logic_rows(chunk_prefix: str) -> dict[tuple[str, int], np.ndarray]:
    from janusx.gfreader import load_genotype_chunks

    out: dict[tuple[str, int], np.ndarray] = {}
    for geno, sites in load_genotype_chunks(
        chunk_prefix,
        chunk_size=4096,
        maf=0.0,
        missing_rate=1.0,
        impute=False,
        model="add",
        het=0.02,
    ):
        block = np.asarray(geno, dtype=np.float32)
        for i, site in enumerate(sites):
            key = (str(site.chrom), int(site.pos))
            if key in out:
                continue
            bin_row = _collapse_logic_bin01_row(block[i, :])
            if bin_row is not None:
                out[key] = np.asarray(bin_row, dtype=np.uint8)
    return out


def _logic_gate_hits(rows: Sequence[np.ndarray], mode: str) -> int:
    if not rows:
        return 0
    agg = np.asarray(rows[0], dtype=np.uint8).copy()
    mode_key = str(mode).strip().lower()
    if mode_key == "and":
        for row in rows[1:]:
            agg &= np.asarray(row, dtype=np.uint8)
    elif mode_key == "or":
        for row in rows[1:]:
            agg |= np.asarray(row, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported logic mode: {mode}")
    return int(np.sum(agg, dtype=np.int64))


def _resolve_valid_logic_mode_for_sites(
    sites: Sequence[tuple[str, int]],
    *,
    row_map: dict[tuple[str, int], np.ndarray],
    requested_mode: str,
) -> tuple[str | None, dict[str, int]]:
    rows: list[np.ndarray] = []
    for chrom, pos in sites:
        row = row_map.get((str(chrom), int(pos)))
        if row is None:
            return None, {}
        rows.append(row)
    if not rows:
        return None, {}
    n_samples = int(rows[0].shape[0])
    and_hits = _logic_gate_hits(rows, "and")
    or_hits = _logic_gate_hits(rows, "or")
    stats = {"and_hits": int(and_hits), "or_hits": int(or_hits)}
    and_ok = 0 < and_hits < n_samples
    or_ok = 0 < or_hits < n_samples
    mode_key = str(requested_mode).strip().lower()
    if mode_key == "and":
        return ("and" if and_ok else None), stats
    if mode_key == "or":
        return ("or" if or_ok else None), stats
    if mode_key == "and_or":
        if and_ok and or_ok:
            return "and_or", stats
        if and_ok:
            return "and", stats
        if or_ok:
            return "or", stats
        return None, stats
    raise ValueError(f"Unsupported logic mode: {requested_mode}")


def _sample_valid_logic_sites(
    site_sampler,
    *,
    row_map: dict[tuple[str, int], np.ndarray],
    requested_mode: str,
    rng: np.random.Generator | None = None,
    max_attempts: int = 128,
) -> tuple[tuple[tuple[str, int], ...], str, dict[str, int]]:
    last_stats: dict[str, int] = {}
    for _ in range(max(1, int(max_attempts))):
        sites = tuple(site_sampler())
        resolved_mode, stats = _resolve_valid_logic_mode_for_sites(
            sites,
            row_map=row_map,
            requested_mode=requested_mode,
        )
        last_stats = stats
        if resolved_mode is not None:
            if resolved_mode == "and_or":
                if rng is None:
                    resolved_mode = "and"
                else:
                    resolved_mode = "and" if int(rng.integers(0, 2)) == 0 else "or"
            return sites, resolved_mode, stats
    detail = ""
    if last_stats:
        detail = f" (last and_hits={last_stats.get('and_hits', -1)}, or_hits={last_stats.get('or_hits', -1)})"
    raise ValueError(
        f"Unable to sample a non-degenerate {requested_mode} logic gate after {max_attempts} attempts{detail}."
    )


def _unique_sorted_sizes(causal_sizes: Sequence[int], *, min_size: int = 1) -> list[int]:
    return sorted({int(x) for x in causal_sizes if int(x) >= int(min_size)})


def _subset_key(sites: Sequence[tuple[str, int]]) -> str:
    raw = "|".join(f"{chrom}:{pos}" for chrom, pos in sorted(sites))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"s{len(sites)}.{digest}"


def _build_chunk_random_experiments(
    *,
    chunks: Sequence[ChunkRecord],
    causal_sizes: Sequence[int],
    logic_modes: Sequence[str],
    include_chunk_random: bool,
    seed: int,
    max_experiments: int | None,
    logger: logging.Logger,
) -> tuple[list[ExperimentPlan], list[dict[str, object]]]:
    rng = np.random.default_rng(int(seed))
    plans: list[ExperimentPlan] = []
    skipped: list[dict[str, object]] = []
    tested_sizes = _unique_sorted_sizes(causal_sizes, min_size=1)
    interaction_sizes = _unique_sorted_sizes(causal_sizes, min_size=2)
    mode_set = {str(x).strip().lower() for x in logic_modes if str(x).strip() != ""}
    if mode_set == {"and", "or"}:
        chunk_logic_mode = "and_or"
    elif len(mode_set) == 1:
        chunk_logic_mode = next(iter(mode_set))
    elif len(mode_set) == 0:
        chunk_logic_mode = "and"
    else:
        raise ValueError(f"Unsupported chunk-random logic mode set: {sorted(mode_set)!r}")

    def _append(plan: ExperimentPlan) -> bool:
        if max_experiments is not None and len(plans) >= int(max_experiments):
            return False
        plans.append(plan)
        return True

    if not include_chunk_random:
        logger.info("Chunk-random phase disabled.")
        return plans, skipped

    for chunk in chunks:
        try:
            chunk_df = _read_bim(chunk.chunk_prefix)
        except Exception as exc:
            skipped.append(
                {
                    "phase": "chunk_random",
                    "family": "chunk_random",
                    "chunk_id": chunk.chunk_id,
                    "scenario": "read_bim",
                    "replicate": int(chunk.repeat_index),
                    "reason": str(exc),
                }
            )
            continue
        site_pool = chunk_df.loc[:, ["chrom", "pos"]].copy()
        if site_pool.empty:
            skipped.append(
                {
                    "phase": "chunk_random",
                    "family": "chunk_random",
                    "chunk_id": chunk.chunk_id,
                    "scenario": "empty_chunk",
                    "replicate": int(chunk.repeat_index),
                    "reason": "Chunk BIM contains no variants.",
                }
            )
            continue
        row_map: dict[tuple[str, int], np.ndarray] = {}
        if interaction_sizes:
            try:
                row_map = _load_chunk_logic_rows(chunk.chunk_prefix)
            except Exception as exc:
                skipped.append(
                    {
                        "phase": "chunk_random",
                        "family": "chunk_random",
                        "chunk_id": chunk.chunk_id,
                        "scenario": "load_logic_rows",
                        "replicate": int(chunk.repeat_index),
                        "reason": str(exc),
                    }
                )
                continue

        for causal_size in tested_sizes:
            try:
                if int(causal_size) == 1:
                    sites = tuple(_sample_random_sites(site_pool, int(causal_size), rng))
                    resolved_logic_mode = None
                else:
                    sites, resolved_logic_mode, _stats = _sample_valid_logic_sites(
                        lambda: _sample_random_sites(site_pool, int(causal_size), rng),
                        row_map=row_map,
                        requested_mode=chunk_logic_mode,
                        rng=rng,
                    )
            except Exception as exc:
                skipped.append(
                    {
                        "phase": "chunk_random",
                        "family": "chunk_random",
                        "chunk_id": chunk.chunk_id,
                        "scenario": f"size{int(causal_size)}",
                        "replicate": int(chunk.repeat_index),
                        "reason": str(exc),
                    }
                )
                continue
            subset_key = _subset_key(sites)
            if int(causal_size) == 1:
                if not _append(
                    ExperimentPlan(
                        experiment_id=f"{chunk.chunk_id}.single",
                        family="chunk_random",
                        scenario="chunk_random_single",
                        replicate=int(chunk.repeat_index),
                        chunk_id=chunk.chunk_id,
                        chunk_range=None,
                        causal_size=1,
                        logic_mode=None,
                        selected_sites=sites,
                        selected_groups=("random1",),
                        sim_input_mode="subset_exact",
                        subset_key=subset_key,
                        notes=("random_sites_within_chunk", "single_logic_mode_collapsed"),
                    )
                ):
                    logger.info("Chunk-random planning hit max_experiments=%d", int(max_experiments))
                    return plans, skipped
                continue

            if int(causal_size) in interaction_sizes:
                if not _append(
                    ExperimentPlan(
                        experiment_id=f"{chunk.chunk_id}.logic{int(causal_size)}.{resolved_logic_mode}",
                        family="chunk_random",
                        scenario=f"chunk_random_logic{int(causal_size)}",
                        replicate=int(chunk.repeat_index),
                        chunk_id=chunk.chunk_id,
                        chunk_range=None,
                        causal_size=int(causal_size),
                        logic_mode=str(resolved_logic_mode),
                        selected_sites=sites,
                        selected_groups=(f"random{int(causal_size)}",),
                        sim_input_mode="subset_exact",
                        subset_key=subset_key,
                        notes=("random_sites_within_chunk",),
                    )
                ):
                    logger.info("Chunk-random planning hit max_experiments=%d", int(max_experiments))
                    return plans, skipped

    logger.info("Planned chunk-random experiments: %d", len(plans))
    logger.info("Skipped chunk-random templates: %d", len(skipped))
    return plans, skipped


def _choose_stratified_chunks(
    chunks: Sequence[ChunkRecord],
    *,
    n_select: int,
    seed: int,
) -> list[ChunkRecord]:
    if n_select <= 0 or len(chunks) == 0:
        return []
    if n_select >= len(chunks):
        return list(chunks)
    rng = np.random.default_rng(int(seed))
    picked = np.sort(rng.choice(len(chunks), size=int(n_select), replace=False))
    return [chunks[int(i)] for i in picked]


def _build_stratified_experiments(
    *,
    chunks: Sequence[ChunkRecord],
    chunk_stats: dict[str, pd.DataFrame],
    logic_modes: Sequence[str],
    include_maf_single: bool,
    include_ld_single: bool,
    include_maf_pair: bool,
    include_ld_pair: bool,
    seed: int,
    max_experiments: int | None,
    logger: logging.Logger,
) -> tuple[list[ExperimentPlan], list[dict[str, object]]]:
    rng = np.random.default_rng(int(seed))
    plans: list[ExperimentPlan] = []
    skipped: list[dict[str, object]] = []

    def _append(plan: ExperimentPlan) -> bool:
        if max_experiments is not None and len(plans) >= int(max_experiments):
            return False
        plans.append(plan)
        return True

    maf_pair_bins = [
        (MAF3_LABELS[i], MAF3_LABELS[j])
        for i in range(len(MAF3_LABELS))
        for j in range(i, len(MAF3_LABELS))
    ]
    ld_pair_bins = [("low", "low"), ("low", "high"), ("high", "high")]

    for chunk in chunks:
        chunk_df = chunk_stats.get(chunk.chunk_id)
        if chunk_df is None or chunk_df.empty:
            skipped.append(
                {
                    "phase": "stratified",
                    "family": "chunk_stats",
                    "chunk_id": chunk.chunk_id,
                    "scenario": "missing_chunk_stats",
                    "replicate": int(chunk.repeat_index),
                    "reason": "No chunk stats were available for this chunk.",
                }
            )
            continue
        row_map: dict[tuple[str, int], np.ndarray] = {}
        if include_maf_pair or include_ld_pair:
            try:
                row_map = _load_chunk_logic_rows(chunk.chunk_prefix)
            except Exception as exc:
                skipped.append(
                    {
                        "phase": "stratified",
                        "family": "chunk_stats",
                        "chunk_id": chunk.chunk_id,
                        "scenario": "load_logic_rows",
                        "replicate": int(chunk.repeat_index),
                        "reason": str(exc),
                    }
                )
                continue

        if include_maf_single:
            for maf_bin in MAF3_LABELS:
                try:
                    sites = tuple(
                        _sample_sites_from_group(
                            chunk_df,
                            group_col="maf_bin3",
                            group_value=maf_bin,
                            n_sites=1,
                            rng=rng,
                        )
                    )
                except Exception as exc:
                    skipped.append(
                        {
                            "phase": "stratified",
                            "family": "maf_single",
                            "chunk_id": chunk.chunk_id,
                            "scenario": maf_bin,
                            "replicate": int(chunk.repeat_index),
                            "reason": str(exc),
                        }
                    )
                    continue
                if not _append(
                    ExperimentPlan(
                        experiment_id=f"{chunk.chunk_id}.maf_single.{maf_bin}",
                        family="maf_single",
                        scenario="maf_single",
                        replicate=int(chunk.repeat_index),
                        chunk_id=chunk.chunk_id,
                        chunk_range=None,
                        causal_size=1,
                        logic_mode=None,
                        selected_sites=sites,
                        selected_groups=(maf_bin,),
                        sim_input_mode="subset_exact",
                        subset_key=_subset_key(sites),
                    )
                ):
                    logger.info("Stratified planning hit max_experiments=%d", int(max_experiments))
                    return plans, skipped

        if include_maf_pair:
            for group_a, group_b in maf_pair_bins:
                pair_code = MAF3_SHORT[str(group_a)] + MAF3_SHORT[str(group_b)]
                for logic_mode in logic_modes:
                    try:
                        site_list, resolved_logic_mode, _stats = _sample_valid_logic_sites(
                            lambda ga=group_a, gb=group_b: _sample_pair_from_group_bins(
                                chunk_df,
                                group_col="maf_bin3",
                                group_a=ga,
                                group_b=gb,
                                rng=rng,
                            ),
                            row_map=row_map,
                            requested_mode=str(logic_mode),
                        )
                    except Exception as exc:
                        skipped.append(
                            {
                                "phase": "stratified",
                                "family": "maf_pair",
                                "chunk_id": chunk.chunk_id,
                                "scenario": f"{pair_code}.{logic_mode}",
                                "replicate": int(chunk.repeat_index),
                                "reason": str(exc),
                            }
                        )
                        continue
                    subset_key = _subset_key(site_list)
                    if not _append(
                        ExperimentPlan(
                            experiment_id=f"{chunk.chunk_id}.maf_pair.{pair_code}.{logic_mode}",
                            family="maf_pair",
                            scenario="maf_pair",
                            replicate=int(chunk.repeat_index),
                            chunk_id=chunk.chunk_id,
                            chunk_range=None,
                            causal_size=2,
                            logic_mode=str(resolved_logic_mode),
                            selected_sites=site_list,
                            selected_groups=(pair_code,),
                            sim_input_mode="subset_exact",
                            subset_key=subset_key,
                        )
                    ):
                        logger.info("Stratified planning hit max_experiments=%d", int(max_experiments))
                        return plans, skipped

        if include_ld_single:
            for ld_bin in LD2_LABELS:
                try:
                    sites = tuple(
                        _sample_sites_from_group(
                            chunk_df,
                            group_col="ld_bin2",
                            group_value=ld_bin,
                            n_sites=1,
                            rng=rng,
                        )
                    )
                except Exception as exc:
                    skipped.append(
                        {
                            "phase": "stratified",
                            "family": "ld_single",
                            "chunk_id": chunk.chunk_id,
                            "scenario": ld_bin,
                            "replicate": int(chunk.repeat_index),
                            "reason": str(exc),
                        }
                    )
                    continue
                if not _append(
                    ExperimentPlan(
                        experiment_id=f"{chunk.chunk_id}.ld_single.{ld_bin}",
                        family="ld_single",
                        scenario="ld_single",
                        replicate=int(chunk.repeat_index),
                        chunk_id=chunk.chunk_id,
                        chunk_range=None,
                        causal_size=1,
                        logic_mode=None,
                        selected_sites=sites,
                        selected_groups=(ld_bin,),
                        sim_input_mode="subset_exact",
                        subset_key=_subset_key(sites),
                    )
                ):
                    logger.info("Stratified planning hit max_experiments=%d", int(max_experiments))
                    return plans, skipped

        if include_ld_pair:
            for group_a, group_b in ld_pair_bins:
                pair_code = LD2_SHORT[str(group_a)] + LD2_SHORT[str(group_b)]
                for logic_mode in logic_modes:
                    try:
                        site_list, resolved_logic_mode, _stats = _sample_valid_logic_sites(
                            lambda ga=group_a, gb=group_b: _sample_pair_from_group_bins(
                                chunk_df,
                                group_col="ld_bin2",
                                group_a=ga,
                                group_b=gb,
                                rng=rng,
                            ),
                            row_map=row_map,
                            requested_mode=str(logic_mode),
                        )
                    except Exception as exc:
                        skipped.append(
                            {
                                "phase": "stratified",
                                "family": "ld_pair",
                                "chunk_id": chunk.chunk_id,
                                "scenario": f"{pair_code}.{logic_mode}",
                                "replicate": int(chunk.repeat_index),
                                "reason": str(exc),
                            }
                        )
                        continue
                    subset_key = _subset_key(site_list)
                    if not _append(
                        ExperimentPlan(
                            experiment_id=f"{chunk.chunk_id}.ld_pair.{pair_code}.{logic_mode}",
                            family="ld_pair",
                            scenario="ld_pair",
                            replicate=int(chunk.repeat_index),
                            chunk_id=chunk.chunk_id,
                            chunk_range=None,
                            causal_size=2,
                            logic_mode=str(resolved_logic_mode),
                            selected_sites=site_list,
                            selected_groups=(pair_code,),
                            sim_input_mode="subset_exact",
                            subset_key=subset_key,
                        )
                    ):
                        logger.info("Stratified planning hit max_experiments=%d", int(max_experiments))
                        return plans, skipped

    logger.info("Planned stratified experiments: %d", len(plans))
    logger.info("Skipped stratified templates: %d", len(skipped))
    return plans, skipped


def _write_table(
    path: Path,
    rows: Sequence[dict[str, object]],
    *,
    columns: Sequence[str] | None = None,
) -> None:
    if not rows:
        pd.DataFrame(columns=list(columns or [])).to_csv(path, sep="\t", index=False)
        return
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _ensure_exact_subset(
    full_bfile: str,
    sites: Sequence[tuple[str, int]],
    *,
    subset_dir: Path,
    subset_key: str,
    threads: int,
    force: bool,
    runtime: WorkflowRuntime,
) -> str:
    subset_dir.mkdir(parents=True, exist_ok=True)
    prefix_name = f"subset.{subset_key}"
    subset_prefix = str(subset_dir / prefix_name)
    if (not force) and Path(f"{subset_prefix}.bed").is_file():
        return subset_prefix
    extract_path = subset_dir / f"{prefix_name}.extract.tsv"
    with open(extract_path, "w", encoding="utf-8") as fw:
        for chrom, pos in sorted(sites):
            fw.write(f"{chrom}\t{pos}\n")
    cmd = _jx_cmd(
        "gformat",
        "-bfile",
        full_bfile,
        "-fmt",
        "plink",
        "-extract",
        str(extract_path),
        "-o",
        subset_dir,
        "-prefix",
        prefix_name,
        "-t",
        max(1, int(threads)),
    )
    _run_cmd(
        cmd,
        runtime=runtime,
        desc=f"Build exact-site subset {subset_key}",
        log_label=f"subset.{subset_key}",
        extra_env=_thread_cap_env(threads),
    )
    return subset_prefix


def _discover_garfield_rules(garfield_dir: Path, prefix_tag: str) -> str:
    pattern = f"{prefix_tag}.sim_bg*.garfield.rules.tsv"
    hits = sorted(str(p) for p in garfield_dir.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected 1 GARFIELD rules file for prefix {prefix_tag!r}, found {len(hits)}"
        )
    return hits[0]


def _discover_sim_fixed(sim_dir: Path, prefix_tag: str) -> str:
    path = sim_dir / f"{prefix_tag}.fixed.effects.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing fixed-effects TSV: {path}")
    return str(path)


def _run_simulation_job(
    plan: ExperimentPlan,
    *,
    sim_bfile: str,
    sim_dir: Path,
    sim_prefix: str,
    grm: str,
    grm_id: str | None,
    maf: float,
    geno: float,
    bg_pve: float,
    cs_pve: float,
    threads: int,
    seed: int,
    force: bool,
    runtime: WorkflowRuntime,
) -> str:
    sim_dir.mkdir(parents=True, exist_ok=True)
    pheno_path = sim_dir / f"{sim_prefix}.pheno.txt"
    if (not force) and pheno_path.is_file():
        runtime.logger.info("Reuse simulation: %s", sim_prefix)
        return str(pheno_path)
    cmd = _jx_cmd(
        "simulation",
        "-bfile",
        sim_bfile,
        "-o",
        sim_dir,
        "-prefix",
        sim_prefix,
        "-maf",
        maf,
        "-geno",
        geno,
        "-bg-pve",
        bg_pve,
        "-cs-pve",
        cs_pve,
        "-k",
        grm,
        "--seed",
        seed,
    )
    if grm_id:
        cmd.extend(["-kid", grm_id])
    if plan.sim_input_mode == "full_range":
        if plan.chunk_range is None:
            raise ValueError(f"Chunk-range simulation missing chunk_range: {plan.experiment_id}")
        chrom, start_bp, end_bp = plan.chunk_range
        cmd.extend(["-bimrange", f"{chrom}:{start_bp}:{end_bp}"])
    if plan.logic_mode is None:
        cmd.extend(["-causal", "1"])
    else:
        cmd.extend(["-causal", "1", "-logic-gate", str(plan.causal_size), str(plan.logic_mode)])
    _run_cmd(
        cmd,
        runtime=runtime,
        desc=f"Run jx simulation for {plan.experiment_id}",
        log_label=f"simulation.{plan.experiment_id}",
        extra_env=_thread_cap_env(threads),
    )
    return str(pheno_path)


def _run_garfield_job(
    plan: ExperimentPlan,
    *,
    scan_bfile: str,
    pheno_path: str,
    fixed_effects_path: str,
    garfield_dir: Path,
    garfield_prefix: str,
    grm: str,
    grm_id: str | None,
    maf: float,
    geno: float,
    het: float,
    garfield_raw: int | None,
    extension: int,
    threads: int,
    force: bool,
    runtime: WorkflowRuntime,
) -> str:
    garfield_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        hits = sorted(str(p) for p in garfield_dir.glob(f"{garfield_prefix}.sim_bg*.garfield.rules.tsv"))
        if len(hits) == 1:
            runtime.logger.info("Reuse GARFIELD output: %s", hits[0])
            return hits[0]
    logic_gate = str(plan.logic_mode) if plan.logic_mode is not None else "and_or"
    cmd = _jx_cmd(
        "garfield",
        "-bfile",
        scan_bfile,
        "-p",
        pheno_path,
        "-Window",
        "-o",
        garfield_dir,
        "-prefix",
        garfield_prefix,
        "-ext",
        extension,
        "-logic-gate",
        logic_gate,
        "-maf",
        maf,
        "-geno",
        geno,
        "-het",
        het,
        "-simbench",
        fixed_effects_path,
        "-k",
        grm,
        "-t",
        max(1, int(threads)),
    )
    if garfield_raw is not None:
        cmd.extend(["-raw", str(int(garfield_raw))])
    if grm_id:
        cmd.extend(["-kid", grm_id])
    _run_cmd(
        cmd,
        runtime=runtime,
        desc=f"Run jx garfield for {plan.experiment_id}",
        log_label=f"garfield.{plan.experiment_id}",
        extra_env=_thread_cap_env(threads),
    )
    return _discover_garfield_rules(garfield_dir, garfield_prefix)


def _resolve_plan_chunk_bfile(
    plan: ExperimentPlan,
    *,
    filtered_prefix: str,
    chunk_bfile_by_id: dict[str, str],
) -> str:
    if plan.chunk_id is None:
        return filtered_prefix
    plan_chunk_bfile = chunk_bfile_by_id.get(plan.chunk_id, "")
    if plan_chunk_bfile == "":
        raise KeyError(f"Missing chunk bfile for plan {plan.experiment_id}: {plan.chunk_id}")
    return plan_chunk_bfile


def _resolve_plan_sim_bfile(
    plan: ExperimentPlan,
    *,
    plan_chunk_bfile: str,
    subset_dir: Path,
    subset_threads: int,
    force: bool,
    runtime: WorkflowRuntime,
    subset_cache: dict[str, str],
    subset_inflight: dict[str, threading.Event],
    subset_lock: threading.Lock,
) -> str:
    if plan.sim_input_mode != "subset_exact":
        return plan_chunk_bfile
    if not plan.selected_sites or plan.subset_key is None:
        raise ValueError(f"Exact-subset experiment missing selected sites: {plan.experiment_id}")

    subset_key = plan.subset_key
    with subset_lock:
        cached = subset_cache.get(subset_key)
        if cached is not None:
            return cached
        event = subset_inflight.get(subset_key)
        if event is None:
            event = threading.Event()
            subset_inflight[subset_key] = event
            owner = True
        else:
            owner = False

    if owner:
        try:
            subset_bfile = _ensure_exact_subset(
                plan_chunk_bfile,
                plan.selected_sites,
                subset_dir=subset_dir,
                subset_key=subset_key,
                threads=max(1, int(subset_threads)),
                force=force,
                runtime=runtime,
            )
        except Exception:
            with subset_lock:
                subset_inflight.pop(subset_key, None)
                event.set()
            raise
        with subset_lock:
            subset_cache[subset_key] = subset_bfile
            subset_inflight.pop(subset_key, None)
            event.set()
        return subset_bfile

    event.wait()
    with subset_lock:
        cached = subset_cache.get(subset_key)
    if cached is None:
        raise RuntimeError(f"Subset build did not produce cached prefix: {subset_key}")
    return cached


def _run_plan_experiment(
    *,
    idx: int,
    plan: ExperimentPlan,
    filtered_prefix: str,
    chunk_bfile_by_id: dict[str, str],
    subset_dir: Path,
    sim_dir: Path,
    garfield_dir: Path,
    grm: str,
    grm_id: str | None,
    maf: float,
    geno: float,
    het: float,
    bg_pve: float,
    cs_pve: float,
    garfield_ext: int,
    garfield_raw: int | None,
    seed: int,
    stage5_inner_threads: int,
    force: bool,
    runtime: WorkflowRuntime,
    subset_cache: dict[str, str],
    subset_inflight: dict[str, threading.Event],
    subset_lock: threading.Lock,
) -> dict[str, object]:
    plan_chunk_bfile = _resolve_plan_chunk_bfile(
        plan,
        filtered_prefix=filtered_prefix,
        chunk_bfile_by_id=chunk_bfile_by_id,
    )
    sim_bfile = _resolve_plan_sim_bfile(
        plan,
        plan_chunk_bfile=plan_chunk_bfile,
        subset_dir=subset_dir,
        subset_threads=stage5_inner_threads,
        force=force,
        runtime=runtime,
        subset_cache=subset_cache,
        subset_inflight=subset_inflight,
        subset_lock=subset_lock,
    )

    sim_prefix = plan.experiment_id
    pheno_path = _run_simulation_job(
        plan,
        sim_bfile=sim_bfile,
        sim_dir=sim_dir,
        sim_prefix=sim_prefix,
        grm=grm,
        grm_id=grm_id,
        maf=maf,
        geno=geno,
        bg_pve=bg_pve,
        cs_pve=cs_pve,
        threads=stage5_inner_threads,
        seed=seed + idx,
        force=force,
        runtime=runtime,
    )
    fixed_effects = _discover_sim_fixed(sim_dir, sim_prefix)
    rules_path = _run_garfield_job(
        plan,
        scan_bfile=plan_chunk_bfile,
        pheno_path=pheno_path,
        fixed_effects_path=fixed_effects,
        garfield_dir=garfield_dir,
        garfield_prefix=plan.experiment_id,
        grm=grm,
        grm_id=grm_id,
        maf=maf,
        geno=geno,
        het=het,
        garfield_raw=garfield_raw,
        extension=garfield_ext,
        threads=stage5_inner_threads,
        force=force,
        runtime=runtime,
    )
    result_row = _summarize_rules(rules_path=rules_path, plan=plan)
    result_row["fixed_effects_path"] = fixed_effects
    result_row["pheno_path"] = pheno_path
    return result_row


def _run_plan_batch(
    *,
    task_title: str,
    plans: Sequence[ExperimentPlan],
    start_plan_idx: int,
    experiment_jobs: int,
    stage5_inner_threads: int,
    filtered_prefix: str,
    chunk_bfile_by_id: dict[str, str],
    subset_dir: Path,
    sim_dir: Path,
    garfield_dir: Path,
    grm: str,
    grm_id: str | None,
    maf: float,
    geno: float,
    het: float,
    bg_pve: float,
    cs_pve: float,
    garfield_ext: int,
    garfield_raw: int | None,
    seed: int,
    force: bool,
    runtime: WorkflowRuntime,
    subset_cache: dict[str, str],
    subset_inflight: dict[str, threading.Event],
    subset_lock: threading.Lock,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    if not plans:
        return results, failures, start_plan_idx

    task_id = runtime.ui.add_stage(
        task_title,
        max(1, len(plans)),
        current=f"jobs={experiment_jobs}, queued={len(plans)}",
    )
    max_workers = min(max(1, int(experiment_jobs)), len(plans))
    runtime.logger.info(
        "%s concurrency: jobs=%d, inner_threads=%d",
        task_title,
        max_workers,
        stage5_inner_threads,
    )
    pending: dict[Future[dict[str, object]], tuple[int, ExperimentPlan]] = {}
    next_submit_idx = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tgarfield-exp") as executor:
        while next_submit_idx < len(plans) and len(pending) < max_workers:
            plan = plans[next_submit_idx]
            plan_idx = start_plan_idx + next_submit_idx
            runtime.logger.info(
                "Queue experiment %d / %d: %s",
                plan_idx,
                start_plan_idx + len(plans) - 1,
                plan.experiment_id,
            )
            future = executor.submit(
                _run_plan_experiment,
                idx=plan_idx,
                plan=plan,
                filtered_prefix=filtered_prefix,
                chunk_bfile_by_id=chunk_bfile_by_id,
                subset_dir=subset_dir,
                sim_dir=sim_dir,
                garfield_dir=garfield_dir,
                grm=grm,
                grm_id=grm_id,
                maf=maf,
                geno=geno,
                het=het,
                bg_pve=bg_pve,
                cs_pve=cs_pve,
                garfield_ext=garfield_ext,
                garfield_raw=garfield_raw,
                seed=seed,
                stage5_inner_threads=stage5_inner_threads,
                force=force,
                runtime=runtime,
                subset_cache=subset_cache,
                subset_inflight=subset_inflight,
                subset_lock=subset_lock,
            )
            pending[future] = (plan_idx, plan)
            next_submit_idx += 1

        while pending:
            done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in done:
                plan_idx, plan = pending.pop(future)
                try:
                    result_row = future.result()
                    result_row["_plan_idx"] = plan_idx
                    results.append(result_row)
                except Exception as exc:
                    runtime.logger.exception("Experiment failed: %s", plan.experiment_id)
                    failures.append(
                        {
                            "_plan_idx": plan_idx,
                            "experiment_id": plan.experiment_id,
                            "family": plan.family,
                            "scenario": plan.scenario,
                            "error": str(exc),
                        }
                    )
                completed += 1

                while next_submit_idx < len(plans) and len(pending) < max_workers:
                    next_plan = plans[next_submit_idx]
                    next_plan_idx = start_plan_idx + next_submit_idx
                    runtime.logger.info(
                        "Queue experiment %d / %d: %s",
                        next_plan_idx,
                        start_plan_idx + len(plans) - 1,
                        next_plan.experiment_id,
                    )
                    next_future = executor.submit(
                        _run_plan_experiment,
                        idx=next_plan_idx,
                        plan=next_plan,
                        filtered_prefix=filtered_prefix,
                        chunk_bfile_by_id=chunk_bfile_by_id,
                        subset_dir=subset_dir,
                        sim_dir=sim_dir,
                        garfield_dir=garfield_dir,
                        grm=grm,
                        grm_id=grm_id,
                        maf=maf,
                        geno=geno,
                        het=het,
                        bg_pve=bg_pve,
                        cs_pve=cs_pve,
                        garfield_ext=garfield_ext,
                        garfield_raw=garfield_raw,
                        seed=seed,
                        stage5_inner_threads=stage5_inner_threads,
                        force=force,
                        runtime=runtime,
                        subset_cache=subset_cache,
                        subset_inflight=subset_inflight,
                        subset_lock=subset_lock,
                    )
                    pending[next_future] = (next_plan_idx, next_plan)
                    next_submit_idx += 1

                runtime.ui.update(
                    task_id,
                    advance=1,
                    current=(
                        f"{completed}/{len(plans)} done, "
                        f"running={len(pending)}, fail={len(failures)}, "
                        f"last={plan.experiment_id}"
                    ),
                )
    runtime.ui.complete(task_id, current=f"ok={len(results)} fail={len(failures)}")
    return results, failures, start_plan_idx + len(plans)


def _collect_chunk_stats(
    *,
    task_title: str,
    chunks: Sequence[ChunkRecord],
    out_dir: Path,
    summary_dir: Path,
    ldsc_window: str,
    stage3_jobs: int,
    stage3_inner_threads: int,
    force: bool,
    runtime: WorkflowRuntime,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    task_id = runtime.ui.add_stage(
        task_title,
        max(1, len(chunks)),
        current="jx gstats -freq -ldsc",
    )
    chunk_order = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
    chunk_stats: dict[str, pd.DataFrame] = {}
    chunk_stat_rows: list[dict[str, object]] = []
    stage3_workers = min(max(1, int(stage3_jobs)), len(chunks)) if chunks else 1
    runtime.logger.info(
        "%s concurrency: jobs=%d, inner_threads=%d",
        task_title,
        stage3_workers,
        stage3_inner_threads,
    )
    pending_gstats: dict[Future[dict[str, str]], tuple[int, ChunkRecord]] = {}
    next_chunk_idx = 1
    completed_gstats = 0

    if chunks:
        with ThreadPoolExecutor(max_workers=stage3_workers, thread_name_prefix="tgarfield-gstats") as executor:
            while next_chunk_idx <= len(chunks) and len(pending_gstats) < stage3_workers:
                chunk = chunks[next_chunk_idx - 1]
                future = executor.submit(
                    _run_chunk_gstats,
                    chunk,
                    out_dir=out_dir,
                    ldsc_window=str(ldsc_window),
                    threads=stage3_inner_threads,
                    force=bool(force),
                    runtime=runtime,
                )
                pending_gstats[future] = (next_chunk_idx, chunk)
                next_chunk_idx += 1

            while pending_gstats:
                done, _ = wait(set(pending_gstats), return_when=FIRST_COMPLETED)
                for future in done:
                    _, chunk = pending_gstats.pop(future)
                    outputs = future.result()
                    completed_gstats += 1
                    if not runtime.dry_run:
                        df = _build_chunk_site_stats(chunk, outputs["freq"], outputs["ldsc"])
                        chunk_stats[chunk.chunk_id] = df
                        maf_counts = df["maf_bin3"].value_counts().to_dict()
                        ld2_counts = df["ld_bin2"].value_counts().to_dict()
                        ld3_counts = df["ld_bin3"].value_counts().to_dict()
                        row = {
                            "chunk_id": chunk.chunk_id,
                            "n_sites": int(len(df)),
                            "n_maf3_nonempty": int(sum(int(maf_counts.get(k, 0) > 0) for k in MAF3_LABELS)),
                            "n_ld2_nonempty": int(sum(int(ld2_counts.get(k, 0) > 0) for k in LD2_LABELS)),
                            "n_ld3_nonempty": int(sum(int(ld3_counts.get(k, 0) > 0) for k in LD3_ORDER)),
                            "n_maf_pair_nonempty": int(_supported_pair_count(pd.Series(maf_counts), labels=MAF3_LABELS)),
                            "n_ld_pair_nonempty": int(_supported_pair_count(pd.Series(ld2_counts), labels=LD2_LABELS)),
                            "freq_path": outputs["freq"],
                            "ldsc_path": outputs["ldsc"],
                        }
                        for key in MAF3_LABELS:
                            row[f"maf_{key}"] = int(maf_counts.get(key, 0))
                        for key in LD2_LABELS:
                            row[f"ld2_{key}"] = int(ld2_counts.get(key, 0))
                        for key in LD3_ORDER:
                            row[f"ld3_{key}"] = int(ld3_counts.get(key, 0))
                        chunk_stat_rows.append(row)
                        df.to_csv(summary_dir / f"{chunk.chunk_id}.site_stats.tsv", sep="\t", index=False)
                    runtime.ui.update(
                        task_id,
                        advance=1,
                        current=(
                            f"{completed_gstats}/{len(chunks)} done, "
                            f"running={len(pending_gstats)}, last={chunk.chunk_id}"
                        ),
                    )
                    while next_chunk_idx <= len(chunks) and len(pending_gstats) < stage3_workers:
                        next_chunk = chunks[next_chunk_idx - 1]
                        next_future = executor.submit(
                            _run_chunk_gstats,
                            next_chunk,
                            out_dir=out_dir,
                            ldsc_window=str(ldsc_window),
                            threads=stage3_inner_threads,
                            force=bool(force),
                            runtime=runtime,
                        )
                        pending_gstats[next_future] = (next_chunk_idx, next_chunk)
                        next_chunk_idx += 1

    runtime.ui.complete(task_id, current=f"{len(chunks)} chunk(s)")
    if not runtime.dry_run:
        chunk_stats = {
            chunk.chunk_id: chunk_stats[chunk.chunk_id]
            for chunk in chunks
            if chunk.chunk_id in chunk_stats
        }
        chunk_stat_rows.sort(key=lambda row: chunk_order.get(str(row.get("chunk_id")), len(chunk_order)))
    return chunk_stats, chunk_stat_rows


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if pd.isna(value):
        return True
    return False


def _optional_text(value: object) -> str | None:
    if _is_missing_scalar(value):
        return None
    text = str(value).strip()
    return text if text != "" else None


def _literal_field(value: object, *, default: Any) -> Any:
    if _is_missing_scalar(value):
        return default
    text = str(value).strip()
    if text == "":
        return default
    return ast.literal_eval(text)


def _load_plans_from_tsv(path: Path) -> list[ExperimentPlan]:
    df = pd.read_csv(path, sep="\t")
    plans: list[ExperimentPlan] = []
    for row in df.itertuples(index=False):
        chunk_range_raw = _literal_field(row.chunk_range, default=None)
        chunk_range = None
        if chunk_range_raw is not None:
            chrom, start_bp, end_bp = chunk_range_raw
            chunk_range = (str(chrom), int(start_bp), int(end_bp))

        selected_sites_raw = _literal_field(row.selected_sites, default=())
        selected_sites = tuple((str(chrom), int(pos)) for chrom, pos in selected_sites_raw)
        selected_groups_raw = _literal_field(row.selected_groups, default=())
        selected_groups = tuple(str(x) for x in selected_groups_raw)
        notes_raw = _literal_field(row.notes, default=())
        notes = tuple(str(x) for x in notes_raw)

        plans.append(
            ExperimentPlan(
                experiment_id=str(row.experiment_id),
                family=str(row.family),
                scenario=str(row.scenario),
                replicate=int(row.replicate),
                chunk_id=_optional_text(row.chunk_id),
                chunk_range=chunk_range,
                causal_size=int(row.causal_size),
                logic_mode=_optional_text(row.logic_mode),
                selected_sites=selected_sites,
                selected_groups=selected_groups,
                sim_input_mode=str(row.sim_input_mode),
                subset_key=_optional_text(row.subset_key),
                notes=notes,
            )
        )
    return plans


def _parse_expr(expr: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    raw = str(expr)
    op = "SINGLE"
    if " AND " in raw:
        op = "AND"
    elif " OR " in raw:
        op = "OR"
    terms: list[str] = []
    site_only: list[str] = []
    for neg, site in re.findall(r"(NOT\s+)?BIN\(([^)]+)\)", raw):
        token = ("!" if str(neg).strip() != "" else "") + str(site).strip()
        terms.append(token)
        site_only.append(str(site).strip())
    terms.sort()
    site_only.sort()
    return op, tuple(terms), tuple(site_only)


def _canon_expr(expr: str) -> str:
    op, terms, _ = _parse_expr(expr)
    return op + ":" + "|".join(terms)


def _canon_site_logic(expr: str) -> str:
    op, _, sites = _parse_expr(expr)
    return op + ":" + "|".join(sites)


def _canon_site_only(expr: str) -> str:
    _, _, sites = _parse_expr(expr)
    return "|".join(sites)


def _expr_site_count(expr: str) -> int:
    _, _, sites = _parse_expr(expr)
    return len(sites)


def _parse_chunk_view_fields(chunk_id: object) -> tuple[str, str]:
    raw = str(chunk_id).strip()
    m = re.search(r"\.chr([^.]+)\.(L[123])\.", raw)
    if m is None:
        return "", ""
    chrom = f"chr{m.group(1)}"
    chunk_pos = CHUNK_VIEW_LABELS.get(str(m.group(2)), "")
    return chrom, chunk_pos


def _load_chunk_site_lookup(
    summary_dir: Path,
    chunk_id: object,
    cache: dict[str, dict[str, tuple[float, str]]],
) -> dict[str, tuple[float, str]]:
    key = str(chunk_id).strip()
    if key == "":
        return {}
    if key in cache:
        return cache[key]
    path = summary_dir / f"{key}.site_stats.tsv"
    lookup: dict[str, tuple[float, str]] = {}
    if path.is_file():
        df = pd.read_csv(path, sep="\t")
        for row in df[["site_key", "freq", "maf_bin3"]].itertuples(index=False):
            lookup[str(row.site_key)] = (float(row.freq), str(row.maf_bin3))
    cache[key] = lookup
    return lookup


def _format_pred_site_maf_fields(
    *,
    expr: object,
    chunk_id: object,
    summary_dir: Path,
    cache: dict[str, dict[str, tuple[float, str]]],
) -> tuple[str, str]:
    _, _, sites = _parse_expr(str(expr))
    if len(sites) == 0:
        return "", ""
    lookup = _load_chunk_site_lookup(summary_dir, chunk_id, cache)
    freq_parts: list[str] = []
    maf_parts: list[str] = []
    for site in sites:
        hit = lookup.get(str(site))
        if hit is None:
            freq_parts.append(f"{site}=NA")
            maf_parts.append(f"{site}=NA")
            continue
        freq, maf_bin = hit
        freq_parts.append(f"{site}={freq:.6g}")
        maf_parts.append(f"{site}={maf_bin}")
    return ";".join(freq_parts), ";".join(maf_parts)


def _import_packed_bed_loader() -> Any:
    py_path = str(PYTHON_ROOT)
    if py_path not in sys.path:
        sys.path.insert(0, py_path)
    from janusx.garfield.garfield2 import load_all_genotype_packed_bed

    return load_all_genotype_packed_bed


def _load_rules_window_snp_lookup(
    rules_path: object,
    cache: dict[str, dict[str, str]],
) -> dict[str, str]:
    key = str(rules_path).strip()
    if key == "":
        return {}
    if key in cache:
        return cache[key]
    df = pd.read_csv(key, sep="\t", usecols=["unit_kind", "expr", "snp_name"])
    win = df.loc[df["unit_kind"].astype(str) == "window", ["expr", "snp_name"]].copy()
    lookup: dict[str, str] = {}
    for row in win.itertuples(index=False):
        expr = str(row.expr)
        if expr not in lookup:
            lookup[expr] = str(row.snp_name)
    cache[key] = lookup
    return lookup


def _load_garfield_pseudo_maf_lookup(
    rules_path: object,
    cache: dict[str, dict[str, float]],
) -> dict[str, float]:
    key = str(rules_path).strip()
    if key == "":
        return {}
    if key in cache:
        return cache[key]
    if not key.endswith(".rules.tsv"):
        raise ValueError(f"Unexpected GARFIELD rules path: {key}")
    prefix = key[: -len(".rules.tsv")]
    load_all_genotype_packed_bed = _import_packed_bed_loader()
    packed_ctx, _sites = load_all_genotype_packed_bed(prefix, maf=0.0, missing_rate=1.0)
    maf_arr = np.asarray(packed_ctx["maf"], dtype=np.float64).reshape(-1)
    bim_names: list[str] = []
    with open(f"{prefix}.bim", "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line == "":
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid BIM row in {prefix}.bim: {line}")
            bim_names.append(str(parts[1]))
    if len(bim_names) != int(maf_arr.shape[0]):
        raise ValueError(
            f"Pseudo BED/BIM length mismatch for {prefix}: bim={len(bim_names)} maf={int(maf_arr.shape[0])}"
        )
    lookup = {str(name): float(maf_arr[idx]) for idx, name in enumerate(bim_names)}
    cache[key] = lookup
    return lookup


def _format_pred_pseudo_maf_field(
    *,
    expr: object,
    rules_path: object,
    expr_cache: dict[str, dict[str, str]],
    maf_cache: dict[str, dict[str, float]],
) -> str:
    raw_expr = str(expr).strip()
    if raw_expr == "":
        return ""
    expr_lookup = _load_rules_window_snp_lookup(rules_path, expr_cache)
    snp_name = expr_lookup.get(raw_expr, "")
    if snp_name == "":
        return "NA"
    maf_lookup = _load_garfield_pseudo_maf_lookup(rules_path, maf_cache)
    maf = maf_lookup.get(str(snp_name))
    if maf is None or not np.isfinite(float(maf)):
        return "NA"
    return f"{float(maf):.6g}"


def _demorgan_expr(expr: str) -> str | None:
    op, terms, _ = _parse_expr(expr)
    if op not in {"AND", "OR"} or len(terms) == 0:
        return None
    if any(term.startswith("!") for term in terms):
        return None
    target_op = "OR" if op == "AND" else "AND"
    neg_terms = sorted("!" + term for term in terms)
    return target_op + ":" + "|".join(neg_terms)


def _summarize_rules(
    *,
    rules_path: str,
    plan: ExperimentPlan,
) -> dict[str, object]:
    df = pd.read_csv(rules_path, sep="\t")
    sim = df.loc[df["unit_kind"].astype(str) == "simbench"].copy()
    win = df.loc[df["unit_kind"].astype(str) == "window"].copy()

    if not win.empty:
        best_window_row = win.sort_values("pwald", ascending=True).iloc[0]
        best_window_p = float(best_window_row["pwald"])
        best_window_expr = str(best_window_row["expr"])
        best_window_chisq = float(pd.to_numeric(win["chisq"], errors="coerce").max())
    else:
        best_window_p = float("nan")
        best_window_chisq = float("nan")
        best_window_expr = ""
    best_window_site_count = _expr_site_count(best_window_expr)

    win_expr_set = set(win["expr"].astype(str).map(_canon_expr).tolist())
    win_site_logic_set = set(win["expr"].astype(str).map(_canon_site_logic).tolist())
    win_site_set = set(win["expr"].astype(str).map(_canon_site_only).tolist())

    sim_exprs = sim["expr"].astype(str).tolist()
    if not sim.empty:
        best_sim_row = sim.sort_values("pwald", ascending=True).iloc[0]
        best_sim_expr = str(best_sim_row["expr"])
        sim_pwald = float(best_sim_row["pwald"])
        sim_chisq = float(pd.to_numeric(sim["chisq"], errors="coerce").max())
    else:
        best_sim_expr = ""
        sim_pwald = float("nan")
        sim_chisq = float("nan")
    exact_rule_match = any(_canon_expr(expr) in win_expr_set for expr in sim_exprs)
    exact_site_logic_match = any(_canon_site_logic(expr) in win_site_logic_set for expr in sim_exprs)
    exact_site_match = any(_canon_site_only(expr) in win_site_set for expr in sim_exprs)
    top_rule_match = bool(best_sim_expr != "" and _canon_expr(best_sim_expr) == _canon_expr(best_window_expr))
    top_site_logic_match = bool(
        best_sim_expr != "" and _canon_site_logic(best_sim_expr) == _canon_site_logic(best_window_expr)
    )
    top_site_match = bool(best_sim_expr != "" and _canon_site_only(best_sim_expr) == _canon_site_only(best_window_expr))
    demorgan_match = any(
        dm is not None and dm in win_expr_set
        for dm in (_demorgan_expr(expr) for expr in sim_exprs)
    )
    return {
        "experiment_id": plan.experiment_id,
        "family": plan.family,
        "scenario": plan.scenario,
        "chunk_id": plan.chunk_id,
        "replicate": int(plan.replicate),
        "causal_size": int(plan.causal_size),
        "logic_mode": "" if plan.logic_mode is None else str(plan.logic_mode),
        "selected_sites": ";".join(f"{c}:{p}" for c, p in plan.selected_sites),
        "selected_groups": ";".join(plan.selected_groups),
        "simbench_rows": int(len(sim)),
        "window_rows": int(len(win)),
        "simbench_best_pwald": sim_pwald,
        "simbench_best_expr": best_sim_expr,
        "best_window_pwald": best_window_p,
        "best_window_site_count": int(best_window_site_count),
        "simbench_pwald_le_best_window": (
            bool(np.isfinite(sim_pwald) and np.isfinite(best_window_p) and sim_pwald <= best_window_p)
        ),
        "simbench_pwald_ge_best_window": (
            bool(np.isfinite(sim_pwald) and np.isfinite(best_window_p) and sim_pwald >= best_window_p)
        ),
        "simbench_best_chisq": sim_chisq,
        "best_window_chisq": best_window_chisq,
        "simbench_chisq_ge_best_window": (
            bool(np.isfinite(sim_chisq) and np.isfinite(best_window_chisq) and sim_chisq >= best_window_chisq)
        ),
        "exact_rule_match": bool(exact_rule_match),
        "exact_site_logic_match": bool(exact_site_logic_match),
        "exact_site_match": bool(exact_site_match),
        "top_rule_match": bool(top_rule_match),
        "top_site_logic_match": bool(top_site_logic_match),
        "top_site_match": bool(top_site_match),
        "demorgan_equivalent_match": bool(demorgan_match),
        "best_window_expr": best_window_expr,
        "rules_path": rules_path,
    }


def _write_test_breakdown_tables(
    *,
    summary_dir: Path,
    results: Sequence[dict[str, object]],
) -> None:
    out_dir = summary_dir / "test_breakdown"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        _write_table(out_dir / "manifest.tsv", [], columns=["path", "family", "group_key", "rows"])
        return

    df = pd.DataFrame(results).copy()
    if "_plan_idx" in df.columns:
        df = df.drop(columns=["_plan_idx"])
    if "logic_mode" not in df.columns:
        df["logic_mode"] = ""
    if "selected_groups" not in df.columns:
        df["selected_groups"] = ""
    df["logic_mode"] = df["logic_mode"].fillna("").astype(str)
    df["selected_groups"] = df["selected_groups"].fillna("").astype(str)
    df["replicate"] = pd.to_numeric(df.get("replicate", 0), errors="coerce").fillna(0).astype(int)
    df["causal_size"] = pd.to_numeric(df.get("causal_size", 0), errors="coerce").fillna(0).astype(int)
    df["sim_pwald"] = pd.to_numeric(df.get("simbench_best_pwald", np.nan), errors="coerce")
    df["pred_pwald"] = pd.to_numeric(df.get("best_window_pwald", np.nan), errors="coerce")
    df["pred_site_count"] = pd.to_numeric(df.get("best_window_site_count", 0), errors="coerce").fillna(0).astype(int)
    df["exact_match"] = df.get("top_rule_match", False).astype(bool)
    site_lookup_cache: dict[str, dict[str, tuple[float, str]]] = {}
    expr_to_snp_cache: dict[str, dict[str, str]] = {}
    pseudo_maf_cache: dict[str, dict[str, float]] = {}
    pred_fields = [
        _format_pred_site_maf_fields(
            expr=row.best_window_expr,
            chunk_id=row.chunk_id,
            summary_dir=summary_dir,
            cache=site_lookup_cache,
        )
        for row in df[["best_window_expr", "chunk_id"]].itertuples(index=False)
    ]
    df["pred_site_freqs"] = [x[0] for x in pred_fields]
    df["pred_site_maf_bins"] = [x[1] for x in pred_fields]
    df["pred_pseudo_maf"] = [
        _format_pred_pseudo_maf_field(
            expr=row.best_window_expr,
            rules_path=row.rules_path,
            expr_cache=expr_to_snp_cache,
            maf_cache=pseudo_maf_cache,
        )
        for row in df[["best_window_expr", "rules_path"]].itertuples(index=False)
    ]

    manifest_rows: list[dict[str, object]] = []

    def _emit(
        rel_name: str,
        sub: pd.DataFrame,
        columns: Sequence[str],
        *,
        family: str,
        group_key: str,
        sort_cols: Sequence[str],
    ) -> None:
        keep = sub.loc[:, list(columns)].copy()
        if len(sort_cols) > 0:
            keep = keep.sort_values(list(sort_cols), kind="stable").reset_index(drop=True)
        keep.to_csv(out_dir / rel_name, sep="\t", index=False)
        manifest_rows.append(
            {
                "path": f"test_breakdown/{rel_name}",
                "family": family,
                "group_key": group_key,
                "rows": int(len(keep)),
            }
        )

    chunk_df = df.loc[df["family"].astype(str) == "chunk_random"].copy()
    if not chunk_df.empty:
        chunk_fields = chunk_df["chunk_id"].map(_parse_chunk_view_fields)
        chunk_df["chrom"] = chunk_fields.map(lambda x: x[0])
        chunk_df["chunk_position"] = chunk_fields.map(lambda x: x[1])
        for causal_size in (1, 2, 3):
            sub = chunk_df.loc[chunk_df["causal_size"] == causal_size].copy()
            if sub.empty:
                continue
            _emit(
                f"chunk_size{causal_size}.tsv",
                sub,
                [
                    "chrom",
                    "chunk_position",
                    "sim_pwald",
                    "pred_pwald",
                    "pred_site_count",
                    "exact_match",
                    "replicate",
                    "logic_mode",
                    "experiment_id",
                    "pred_site_freqs",
                    "pred_site_maf_bins",
                    "pred_pseudo_maf",
                ],
                family="chunk_random",
                group_key=f"size{causal_size}",
                sort_cols=["chrom", "chunk_position", "replicate", "logic_mode", "experiment_id"],
            )

    maf_single_df = df.loc[df["family"].astype(str) == "maf_single"].copy()
    if not maf_single_df.empty:
        for group_key in MAF3_LABELS:
            sub = maf_single_df.loc[maf_single_df["selected_groups"] == group_key].copy()
            if sub.empty:
                continue
            _emit(
                f"maf_single.{group_key}.tsv",
                sub,
                [
                    "chunk_id",
                    "sim_pwald",
                    "pred_pwald",
                    "pred_site_count",
                    "exact_match",
                    "replicate",
                    "experiment_id",
                    "pred_site_freqs",
                    "pred_site_maf_bins",
                    "pred_pseudo_maf",
                ],
                family="maf_single",
                group_key=group_key,
                sort_cols=["chunk_id", "replicate", "experiment_id"],
            )

    maf_pair_df = df.loc[df["family"].astype(str) == "maf_pair"].copy()
    if not maf_pair_df.empty:
        for group_key in MAF_PAIR_ORDER:
            sub = maf_pair_df.loc[maf_pair_df["selected_groups"] == group_key].copy()
            if sub.empty:
                continue
            _emit(
                f"maf_pair.{group_key}.tsv",
                sub,
                [
                    "chunk_id",
                    "sim_pwald",
                    "pred_pwald",
                    "pred_site_count",
                    "exact_match",
                    "replicate",
                    "logic_mode",
                    "experiment_id",
                    "pred_site_freqs",
                    "pred_site_maf_bins",
                    "pred_pseudo_maf",
                ],
                family="maf_pair",
                group_key=group_key,
                sort_cols=["chunk_id", "replicate", "logic_mode", "experiment_id"],
            )

    ld_single_df = df.loc[df["family"].astype(str) == "ld_single"].copy()
    if not ld_single_df.empty:
        for group_key in LD2_LABELS:
            sub = ld_single_df.loc[ld_single_df["selected_groups"] == group_key].copy()
            if sub.empty:
                continue
            _emit(
                f"ld_single.{group_key}.tsv",
                sub,
                [
                    "chunk_id",
                    "sim_pwald",
                    "pred_pwald",
                    "pred_site_count",
                    "exact_match",
                    "replicate",
                    "experiment_id",
                    "pred_site_freqs",
                    "pred_site_maf_bins",
                    "pred_pseudo_maf",
                ],
                family="ld_single",
                group_key=group_key,
                sort_cols=["chunk_id", "replicate", "experiment_id"],
            )

    ld_pair_df = df.loc[df["family"].astype(str) == "ld_pair"].copy()
    if not ld_pair_df.empty:
        for group_key in LD_PAIR_ORDER:
            sub = ld_pair_df.loc[ld_pair_df["selected_groups"] == group_key].copy()
            if sub.empty:
                continue
            _emit(
                f"ld_pair.{group_key}.tsv",
                sub,
                [
                    "chunk_id",
                    "sim_pwald",
                    "pred_pwald",
                    "pred_site_count",
                    "exact_match",
                    "replicate",
                    "logic_mode",
                    "experiment_id",
                    "pred_site_freqs",
                    "pred_site_maf_bins",
                    "pred_pseudo_maf",
                ],
                family="ld_pair",
                group_key=group_key,
                sort_cols=["chunk_id", "replicate", "logic_mode", "experiment_id"],
            )

    _write_table(out_dir / "manifest.tsv", manifest_rows, columns=["path", "family", "group_key", "rows"])


def _finalize_result_tables(
    *,
    summary_dir: Path,
    results: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> None:
    results.sort(key=lambda row: int(row.get("_plan_idx", 10**18)))
    for row in results:
        row.pop("_plan_idx", None)
    failures.sort(key=lambda row: int(row.get("_plan_idx", 10**18)))
    for row in failures:
        row.pop("_plan_idx", None)
    _write_table(summary_dir / "results.tsv", results)
    _write_table(
        summary_dir / "failures.tsv",
        failures,
        columns=["experiment_id", "family", "scenario", "error"],
    )
    _write_test_breakdown_tables(summary_dir=summary_dir, results=results)


def _rebuild_summary_from_existing_outputs(
    *,
    plans: Sequence[ExperimentPlan],
    sim_dir: Path,
    garfield_dir: Path,
    summary_dir: Path,
    runtime: WorkflowRuntime,
    task_id: TaskID | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for idx, plan in enumerate(plans, start=1):
        runtime.ui.update(task_id, current=plan.experiment_id)
        try:
            rules_path = _discover_garfield_rules(garfield_dir, plan.experiment_id)
            fixed_effects = _discover_sim_fixed(sim_dir, plan.experiment_id)
            pheno_path = sim_dir / f"{plan.experiment_id}.pheno.txt"
            if not pheno_path.is_file():
                raise FileNotFoundError(f"Missing phenotype file: {pheno_path}")
            result_row = _summarize_rules(rules_path=rules_path, plan=plan)
            result_row["fixed_effects_path"] = fixed_effects
            result_row["pheno_path"] = str(pheno_path)
            result_row["_plan_idx"] = idx
            results.append(result_row)
        except Exception as exc:
            runtime.logger.exception("Summary rebuild failed: %s", plan.experiment_id)
            failures.append(
                {
                    "_plan_idx": idx,
                    "experiment_id": plan.experiment_id,
                    "family": plan.family,
                    "scenario": plan.scenario,
                    "error": str(exc),
                }
            )
        finally:
            runtime.ui.update(
                task_id,
                advance=1,
                current=f"{idx}/{len(plans)} done, fail={len(failures)}",
            )
    _finalize_result_tables(summary_dir=summary_dir, results=results, failures=failures)
    return results, failures


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/Tgarfield.py",
        description=(
            "GARFIELD simulation-test workflow: filter genotype, sample chunks, "
            "run chunk-random simulation/GARFIELD tests on all chunks, then "
            "randomly select a subset of chunks for per-chunk MAF/LDscore-stratified "
            "simulation/GARFIELD tests, and summarize simbench hits."
        ),
    )
    p.add_argument("--bfile", default=None, help="Full-genome PLINK prefix.")
    p.add_argument("--grm", default=None, help="Global GRM path for simulation/garfield.")
    p.add_argument("--grm-id", default=None, help="Optional GRM ID file.")
    p.add_argument("-o", "--out", default="Tgarfield_out", help="Workflow output directory.")
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Reuse existing simulation/GARFIELD outputs under --out and only rebuild summary TSVs.",
    )
    p.add_argument("--prefix", default=None, help="Workflow prefix (default: infer from bfile).")
    p.add_argument("--getchunks", default=str(DEFAULT_GETCHUNKS), help="Reference GetChunks.py path.")
    p.add_argument("--chr", nargs="+", default=None, help="Chromosomes to sample. Default: all from BIM.")
    p.add_argument("--chunk-repeats", type=int, default=10, help="Repeats per chromosome-third chunk.")
    p.add_argument("--chunk-len", type=int, default=500_000, help="Chunk length in bp.")
    p.add_argument(
        "--stratified-chunks",
        type=int,
        default=30,
        help="Number of random chunks to keep for the MAF/LDscore-stratified phase.",
    )
    p.add_argument("--anchor-chunk", default=None, help=argparse.SUPPRESS)
    p.add_argument("--maf", type=float, default=0.02, help="Global MAF filter passed to jx gformat/simulation/garfield.")
    p.add_argument("--geno", type=float, default=0.05, help="Global missing-rate filter passed to jx gformat/simulation/garfield.")
    p.add_argument("--het", type=float, default=0.05, help="Global heterozygosity filter passed to jx gformat/garfield.")
    p.add_argument("--prune-window", default="100kb", help="LD-prune window passed to jx gformat -prune.")
    p.add_argument("--prune-step", type=int, default=5, help="LD-prune step passed to jx gformat -prune.")
    p.add_argument("--prune-r2", type=float, default=0.3, help="LD-prune r2 passed to jx gformat -prune.")
    p.add_argument("--ldsc-window", default="100kb", help="LD-score window for jx gstats.")
    p.add_argument("--bg-pve", type=float, default=0.6, help="Background PVE passed to jx simulation.")
    p.add_argument("--cs-pve", type=float, default=0.05, help="Causal/fixed-effect PVE passed to jx simulation.")
    p.add_argument("--garfield-ext", type=int, default=500_000, help="Window extension passed to jx garfield -ext.")
    p.add_argument(
        "--garfield-raw",
        type=int,
        default=None,
        help=(
            "Optional GARFIELD raw-ranking schedule. If set to N, rules with layer < N use raw "
            "and layer >= N use interaction-gain. If omitted, GARFIELD defaults are used "
            "(logic-gate=or => all raw; otherwise gain starts at layer 2)."
        ),
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Thread count for setup-stage jx subcommands (filter/chunks/gstats).",
    )
    p.add_argument(
        "--stage3-jobs",
        type=int,
        default=None,
        help=(
            "Concurrent chunk-level jx gstats jobs in the selected-chunk stats phase. "
            "Default: auto = min(4, --threads)."
        ),
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Concurrent simulation+GARFIELD experiments in stage 5 (default: use --threads).",
    )
    p.add_argument(
        "--stage5-inner-threads",
        type=int,
        default=None,
        help=(
            "Per-experiment inner thread cap for stage 5 simulation/GARFIELD jobs. "
            "Default: auto = max(1, floor(--threads / --jobs))."
        ),
    )
    p.add_argument("--seed", type=int, default=20260518, help="Workflow random seed.")
    p.add_argument(
        "--causal-sizes",
        default="1,2,3",
        help=(
            "Causal term sizes to test, comma-separated. "
            "1=single SNP, 2=pairwise/one-order interaction, 3+=higher-order random chunk interactions."
        ),
    )
    p.add_argument(
        "--logic-modes",
        default="and,or",
        help="Logic modes for interaction tests with causal size >= 2.",
    )
    p.add_argument(
        "--pair-matrix-mode",
        choices=["ordered", "unordered"],
        default="ordered",
        help="Reserved compatibility option; fixed MAF/LDscore pair tests are currently unordered.",
    )
    p.add_argument("--max-experiments", type=int, default=None, help="Optional cap for planned experiments.")
    p.add_argument(
        "--skip-chunk-random",
        action="store_true",
        help=(
            "Skip chunk-random simulation branch. "
            "This branch covers within-chunk random 1-site and k-site logic tests for all requested --causal-sizes."
        ),
    )
    p.add_argument(
        "--skip-maf-single",
        "--skip-combo-single",
        dest="skip_maf_single",
        action="store_true",
        help="Skip fixed-chunk MAF-tertile single-SNP branch.",
    )
    p.add_argument("--skip-ld-single", action="store_true", help="Skip fixed-chunk LDscore-tertile single-SNP branch.")
    p.add_argument(
        "--skip-maf-pair",
        "--skip-combo-pair",
        dest="skip_maf_pair",
        action="store_true",
        help="Skip fixed-chunk MAF pair branch (6 unordered pair types x and/or).",
    )
    p.add_argument(
        "--skip-ld-pair",
        action="store_true",
        help="Skip fixed-chunk LDscore pair branch (LL/LH/HH x and/or).",
    )
    p.add_argument(
        "--skip-anchor-random-high-order",
        action="store_true",
        help="Reserved compatibility option; anchor-chunk exact-site random high-order branch is currently disabled.",
    )
    p.add_argument("--force", action="store_true", help="Recompute outputs even if files already exist.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.summary_only:
        if not args.bfile or not args.grm:
            parser.error("--bfile and --grm are required unless --summary-only is used.")
    if args.garfield_raw is not None and int(args.garfield_raw) < 1:
        parser.error("--garfield-raw must be >= 1.")
    t0 = time.time()
    if args.stage3_jobs is None:
        stage3_jobs = min(4, max(1, int(args.threads)))
    else:
        stage3_jobs = max(1, int(args.stage3_jobs))
    stage3_inner_threads = max(1, int(args.threads) // stage3_jobs)
    args.stage3_jobs_effective = stage3_jobs
    args.stage3_inner_threads_effective = stage3_inner_threads
    experiment_jobs = max(1, int(args.jobs if args.jobs is not None else args.threads))
    if args.stage5_inner_threads is None:
        stage5_inner_threads = max(1, int(args.threads) // experiment_jobs)
    else:
        stage5_inner_threads = max(1, int(args.stage5_inner_threads))
    args.stage5_inner_threads_effective = stage5_inner_threads

    bfile = _normalize_prefix(str(args.bfile)) if args.bfile else ""
    prefix = str(args.prefix).strip() if args.prefix else (Path(bfile).name if bfile != "" else "")
    out_root = Path(args.out).resolve()
    logs_dir = out_root / "logs"
    logger = _setup_logger(logs_dir)
    ui = WorkflowUI()
    runtime = WorkflowRuntime(
        logger=logger,
        ui=ui,
        logs_dir=logs_dir,
        dry_run=bool(args.dry_run),
    )

    filtered_dir = out_root / "filtered"
    chunks_dir = out_root / "chunks"
    chunk_stats_dir = out_root / "chunk_stats"
    subset_dir = out_root / "subsets"
    sim_dir = out_root / "simulation"
    garfield_dir = out_root / "garfield"
    summary_dir = out_root / "summary"
    for d in (filtered_dir, chunks_dir, chunk_stats_dir, subset_dir, sim_dir, garfield_dir, summary_dir):
        d.mkdir(parents=True, exist_ok=True)

    existing_cfg: dict[str, object] = {}
    config_path = summary_dir / "workflow_config.json"
    if args.summary_only and config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as fh:
            existing_cfg = json.load(fh)

    display_chunk_repeats = int(existing_cfg.get("chunk_repeats", args.chunk_repeats)) if args.summary_only else int(args.chunk_repeats)
    display_chunk_len = int(existing_cfg.get("chunk_len", args.chunk_len)) if args.summary_only else int(args.chunk_len)
    display_stratified_chunks = int(existing_cfg.get("stratified_chunks", args.stratified_chunks)) if args.summary_only else int(args.stratified_chunks)
    display_causal_sizes = str(existing_cfg.get("causal_sizes", args.causal_sizes)) if args.summary_only else str(args.causal_sizes)
    display_logic_modes = str(existing_cfg.get("logic_modes", args.logic_modes)) if args.summary_only else str(args.logic_modes)
    display_ldsc_window = str(existing_cfg.get("ldsc_window", args.ldsc_window)) if args.summary_only else str(args.ldsc_window)
    display_garfield_raw = existing_cfg.get("garfield_raw", args.garfield_raw) if args.summary_only else args.garfield_raw
    display_bfile = bfile or str(existing_cfg.get("bfile", "(reuse existing)"))
    display_grm = str(args.grm or existing_cfg.get("grm", "(reuse existing)"))
    display_prefix = prefix or str(existing_cfg.get("prefix") or Path(display_bfile).name)
    display_mode = "summary-only" if args.summary_only else ("dry-run" if args.dry_run else "execute")

    ui.print_header(
        title="Tgarfield Workflow",
        rows=[
            ("Genotype", display_bfile),
            ("GRM", display_grm),
            ("Output", out_root),
            ("Prefix", display_prefix),
            ("Setup threads", int(args.threads)),
            ("Gstats jobs", stage3_jobs),
            ("Gstats inner threads", stage3_inner_threads),
            ("Experiment jobs", experiment_jobs),
            ("Stage5 inner threads", stage5_inner_threads),
            ("Chunk repeats", display_chunk_repeats),
            ("Chunk length", display_chunk_len),
            ("Stratified chunks", display_stratified_chunks),
            ("Causal sizes", display_causal_sizes),
            ("Logic modes", display_logic_modes),
            ("GARFIELD raw", display_garfield_raw),
            ("LDscore window", display_ldsc_window),
            ("Simulation", "jx simulation"),
            ("Command logs", logs_dir / "commands"),
            ("Mode", display_mode),
        ],
    )
    logger.info("Workflow prefix: %s", display_prefix)
    logger.info("Using simulation module: jx simulation")

    ui.start()
    try:
        if args.summary_only:
            planned_path = summary_dir / "planned_experiments.tsv"
            if not planned_path.is_file():
                raise FileNotFoundError(f"Missing planned experiments TSV: {planned_path}")
            plans = _load_plans_from_tsv(planned_path)
            rebuild_task = ui.add_stage(
                "1/1 Rebuild summaries",
                max(1, len(plans)),
                current=f"{len(plans)} planned experiment(s)",
            )
            results, failures = _rebuild_summary_from_existing_outputs(
                plans=plans,
                sim_dir=sim_dir,
                garfield_dir=garfield_dir,
                summary_dir=summary_dir,
                runtime=runtime,
                task_id=rebuild_task,
            )
            ui.complete(rebuild_task, current=f"ok={len(results)} fail={len(failures)}")
            logger.info("Rebuilt summary from existing outputs: %d results, %d failures", len(results), len(failures))
            logger.info("Elapsed: %.2f sec", time.time() - t0)
            ui.stop()
            ui.print_summary(
                title="Summary Rebuilt",
                rows=[
                    ("Experiments", len(plans)),
                    ("Completed", len(results)),
                    ("Failed", len(failures)),
                    ("Summary dir", summary_dir),
                    ("Elapsed", f"{time.time() - t0:.2f} sec"),
                ],
            )
            if failures:
                ui.print_warning(
                    f"{len(failures)} summary row(s) failed. See {summary_dir / 'failures.tsv'}."
                )
            return 0

        filter_task = ui.add_stage("1/6 Filter genotype", 1, current="jx gformat")
        filtered_prefix = _filter_genotype(
            bfile,
            out_dir=filtered_dir,
            prefix=f"{prefix}.maf{args.maf:.3f}.geno{args.geno:.3f}.het{args.het:.3f}.pruned",
            maf=float(args.maf),
            geno=float(args.geno),
            het=float(args.het),
            prune_window=str(args.prune_window),
            prune_step=int(args.prune_step),
            prune_r2=float(args.prune_r2),
            threads=int(args.threads),
            force=bool(args.force),
            runtime=runtime,
        )
        ui.complete(filter_task, current=Path(filtered_prefix).name)

        getchunks_module = _import_getchunks_module(Path(str(args.getchunks)), logger)
        planning_prefix = (
            filtered_prefix
            if (not args.dry_run and Path(f"{filtered_prefix}.bim").is_file())
            else bfile
        )
        chr_list = _parse_chr_list(args.chr, planning_prefix)
        chunks = _plan_chunks(
            planning_prefix,
            chr_list,
            repeats=int(args.chunk_repeats),
            chunk_len=int(args.chunk_len),
            out_dir=chunks_dir,
            prefix_tag=f"{prefix}.chunk",
            seed=int(args.seed),
            getchunks_module=getchunks_module,
            logger=logger,
        )

        extract_task = ui.add_stage(
            "2/6 Sample + extract chunks",
            max(1, len(chunks)),
            current=f"{len(chunks)} chunk(s) planned",
        )
        _extract_chunks(
            filtered_prefix,
            chunks,
            out_dir=chunks_dir,
            threads=int(args.threads),
            force=bool(args.force),
            runtime=runtime,
            task_id=extract_task,
        )
        ui.complete(extract_task, current=f"{len(chunks)} chunk(s)")
        chunk_bfile_by_id = {chunk.chunk_id: chunk.chunk_prefix for chunk in chunks}

        chunk_rows = [asdict(chunk) for chunk in chunks]
        _write_table(summary_dir / "chunks.tsv", chunk_rows)

        if args.dry_run:
            logger.info("Dry-run complete before experiment planning / execution.")
            ui.stop()
            ui.print_summary(
                title="Dry Run Complete",
                rows=[
                    ("Chunks planned", len(chunks)),
                    ("Command logs", logs_dir / "commands"),
                    ("Summary dir", summary_dir),
                    ("Elapsed", f"{time.time() - t0:.2f} sec"),
                ],
            )
            return 0

        logic_modes = _parse_logic_modes(str(args.logic_modes))
        causal_sizes = _parse_int_csv(str(args.causal_sizes))
        chunk_random_plans, skipped_chunk_random = _build_chunk_random_experiments(
            chunks=chunks,
            causal_sizes=causal_sizes,
            logic_modes=logic_modes,
            include_chunk_random=not bool(args.skip_chunk_random),
            seed=int(args.seed) + 1000,
            max_experiments=args.max_experiments,
            logger=logger,
        )
        subset_cache: dict[str, str] = {}
        subset_inflight: dict[str, threading.Event] = {}
        subset_lock = threading.Lock()
        results: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        next_plan_idx = 1
        phase_a_results, phase_a_failures, next_plan_idx = _run_plan_batch(
            task_title="3/6 Chunk-random simulation + GARFIELD",
            plans=chunk_random_plans,
            start_plan_idx=next_plan_idx,
            experiment_jobs=experiment_jobs,
            stage5_inner_threads=stage5_inner_threads,
            filtered_prefix=filtered_prefix,
            chunk_bfile_by_id=chunk_bfile_by_id,
            subset_dir=subset_dir,
            sim_dir=sim_dir,
            garfield_dir=garfield_dir,
            grm=str(args.grm),
            grm_id=args.grm_id,
            maf=float(args.maf),
            geno=float(args.geno),
            het=float(args.het),
            bg_pve=float(args.bg_pve),
            cs_pve=float(args.cs_pve),
            garfield_ext=int(args.garfield_ext),
            garfield_raw=(None if args.garfield_raw is None else int(args.garfield_raw)),
            seed=int(args.seed),
            force=bool(args.force),
            runtime=runtime,
            subset_cache=subset_cache,
            subset_inflight=subset_inflight,
            subset_lock=subset_lock,
        )
        results.extend(phase_a_results)
        failures.extend(phase_a_failures)

        remaining_budget = None
        if args.max_experiments is not None:
            remaining_budget = max(0, int(args.max_experiments) - len(chunk_random_plans))

        do_stratified = (
            int(args.stratified_chunks) > 0
            and remaining_budget != 0
            and not (
                bool(args.skip_maf_single)
                and bool(args.skip_ld_single)
                and bool(args.skip_maf_pair)
                and bool(args.skip_ld_pair)
            )
        )
        stratified_chunks = _choose_stratified_chunks(
            chunks,
            n_select=int(args.stratified_chunks),
            seed=int(args.seed) + 2000,
        ) if do_stratified else []
        _write_table(summary_dir / "selected_stratified_chunks.tsv", [asdict(chunk) for chunk in stratified_chunks])

        chunk_stats: dict[str, pd.DataFrame] = {}
        chunk_stat_rows: list[dict[str, object]] = []
        if do_stratified and stratified_chunks:
            chunk_stats, chunk_stat_rows = _collect_chunk_stats(
                task_title="4/6 Selected chunk MAF + LDscore",
                chunks=stratified_chunks,
                out_dir=chunk_stats_dir,
                summary_dir=summary_dir,
                ldsc_window=str(args.ldsc_window),
                stage3_jobs=stage3_jobs,
                stage3_inner_threads=stage3_inner_threads,
                force=bool(args.force),
                runtime=runtime,
            )
        else:
            stage4_task = ui.add_stage("4/6 Selected chunk MAF + LDscore", 1, current="skipped")
            ui.complete(stage4_task, current="skipped")
        _write_table(summary_dir / "chunk_stat_summary.tsv", chunk_stat_rows)

        stratified_plans, skipped_stratified = _build_stratified_experiments(
            chunks=stratified_chunks,
            chunk_stats=chunk_stats,
            logic_modes=logic_modes,
            include_maf_single=not bool(args.skip_maf_single),
            include_ld_single=not bool(args.skip_ld_single),
            include_maf_pair=not bool(args.skip_maf_pair),
            include_ld_pair=not bool(args.skip_ld_pair),
            seed=int(args.seed) + 3000,
            max_experiments=remaining_budget,
            logger=logger,
        ) if do_stratified and stratified_chunks else ([], [])

        all_plans = chunk_random_plans + stratified_plans
        all_skipped = skipped_chunk_random + skipped_stratified
        _write_table(summary_dir / "planned_experiments.tsv", [asdict(plan) for plan in all_plans])
        _write_table(
            summary_dir / "skipped_experiments.tsv",
            all_skipped,
            columns=["phase", "family", "chunk_id", "scenario", "replicate", "reason"],
        )

        phase_b_results, phase_b_failures, next_plan_idx = _run_plan_batch(
            task_title="5/6 Stratified simulation + GARFIELD",
            plans=stratified_plans,
            start_plan_idx=next_plan_idx,
            experiment_jobs=experiment_jobs,
            stage5_inner_threads=stage5_inner_threads,
            filtered_prefix=filtered_prefix,
            chunk_bfile_by_id=chunk_bfile_by_id,
            subset_dir=subset_dir,
            sim_dir=sim_dir,
            garfield_dir=garfield_dir,
            grm=str(args.grm),
            grm_id=args.grm_id,
            maf=float(args.maf),
            geno=float(args.geno),
            het=float(args.het),
            bg_pve=float(args.bg_pve),
            cs_pve=float(args.cs_pve),
            garfield_ext=int(args.garfield_ext),
            garfield_raw=(None if args.garfield_raw is None else int(args.garfield_raw)),
            seed=int(args.seed),
            force=bool(args.force),
            runtime=runtime,
            subset_cache=subset_cache,
            subset_inflight=subset_inflight,
            subset_lock=subset_lock,
        )
        results.extend(phase_b_results)
        failures.extend(phase_b_failures)

        summary_task = ui.add_stage("6/6 Write summaries", 1, current="results/failures/config")
        _finalize_result_tables(summary_dir=summary_dir, results=results, failures=failures)
        with open(summary_dir / "workflow_config.json", "w", encoding="utf-8") as fw:
            json.dump(vars(args), fw, indent=2, ensure_ascii=False)
        ui.complete(summary_task, current=summary_dir.name)

        logger.info("Completed experiments: %d", len(results))
        logger.info("Failed experiments: %d", len(failures))
        logger.info("Elapsed: %.2f sec", time.time() - t0)
        ui.stop()
        ui.print_summary(
            rows=[
                ("Experiments", len(all_plans)),
                ("Completed", len(results)),
                ("Failed", len(failures)),
                ("Stratified chunks", len(stratified_chunks)),
                ("Summary dir", summary_dir),
                ("Command logs", logs_dir / "commands"),
                ("Elapsed", f"{time.time() - t0:.2f} sec"),
            ],
        )
        if failures:
            ui.print_warning(
                f"{len(failures)} experiment(s) failed. See {summary_dir / 'failures.tsv'} and command logs."
            )
        return 0
    except Exception as exc:
        logger.exception("Tgarfield workflow failed.")
        ui.stop()
        ui.print_warning(
            f"Workflow failed: {exc}. Details were written to {logs_dir / 'Tgarfield.log'}."
        )
        return 1
    finally:
        ui.stop()


if __name__ == "__main__":
    raise SystemExit(main())
