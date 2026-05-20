from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def read_id_file(path: str) -> list[str]:
    out: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fr:
        for raw in fr:
            s = str(raw).strip()
            if s == "":
                continue
            out.append(s.split()[0])
    return out


def resolve_grm_id_path(grm_path: str, explicit: str | None = None) -> str | None:
    if explicit is not None and str(explicit).strip() != "":
        p = str(Path(str(explicit).strip()).expanduser())
        if not Path(p).is_file():
            raise FileNotFoundError(f"GRM ID file not found: {p}")
        return p

    direct = f"{grm_path}.id"
    if Path(direct).is_file():
        return direct

    p = Path(grm_path)
    txt_like = {".txt", ".tsv", ".csv", ".npy"}
    if p.suffix.lower() in txt_like:
        stem_cand = f"{str(p.with_suffix(''))}.id"
        if Path(stem_cand).is_file():
            return stem_cand
    return None


def load_grm_matrix(path: str) -> np.ndarray:
    low = str(path).lower()
    if low.endswith(".npy"):
        arr = np.asarray(np.load(path), dtype=np.float64)
    else:
        arr = np.asarray(np.genfromtxt(path, dtype=np.float64), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"GRM must be a square matrix, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("GRM matrix contains NaN/Inf values.")
    return arr


def load_and_align_grm(
    grm_path: str,
    target_ids: Sequence[str],
    *,
    grm_id_path: str | None = None,
    label: str = "GRM",
) -> tuple[np.ndarray, str | None]:
    target = [str(x) for x in target_ids]
    if len(target) == 0:
        raise ValueError(f"{label} alignment requires at least one target sample ID.")
    if len(set(target)) != len(target):
        raise ValueError(f"{label} alignment target sample IDs must be unique.")

    grm = load_grm_matrix(grm_path)
    id_path = resolve_grm_id_path(grm_path, grm_id_path)
    if id_path is None:
        if grm.shape[0] != len(target):
            raise ValueError(
                f"{label} shape {grm.shape} does not match target sample count {len(target)}, "
                "and no GRM ID file was found for reordering."
            )
        return np.asarray(grm, dtype=np.float64), None

    grm_ids = read_id_file(id_path)
    if len(grm_ids) != grm.shape[0]:
        raise ValueError(
            f"{label} ID count mismatch: matrix n={grm.shape[0]} but ID file has {len(grm_ids)} rows."
        )

    index: dict[str, int] = {}
    for i, sid in enumerate(grm_ids):
        if sid in index:
            raise ValueError(f"{label} ID file contains duplicate sample ID: {sid}")
        index[sid] = i

    missing = [sid for sid in target if sid not in index]
    if missing:
        preview = ", ".join(missing[:5])
        extra = "" if len(missing) <= 5 else f" ... (+{len(missing) - 5} more)"
        raise ValueError(f"{label} is missing target sample IDs: {preview}{extra}")

    order = np.asarray([index[sid] for sid in target], dtype=np.intp)
    if np.array_equal(order, np.arange(len(target), dtype=np.intp)) and grm.shape[0] == len(target):
        return np.asarray(grm, dtype=np.float64), id_path
    return np.asarray(grm[np.ix_(order, order)], dtype=np.float64), id_path
