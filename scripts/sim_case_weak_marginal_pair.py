#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy import stats

from janusx.janusx import bed_ldblock_r2_rust, load_bed_u8_matrix


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CHUNK_PREFIX = (
    REPO_ROOT
    / "test.Tgarfield_out/chunks/cubic_All.maf0.02.chunk.chr1.L1.R1.36299099_36799099"
)
DEFAULT_OUT_DIR = REPO_ROOT / "sim.case"
DEFAULT_GRM = Path("~/cubic/cubic_All.maf0.02.cGRM.npy").expanduser()
DEFAULT_GRM_ID = Path("~/cubic/cubic_All.maf0.02.cGRM.npy.id").expanduser()
DEFAULT_SITE1 = 36423175
DEFAULT_SITE2 = 36755292


def _out_path(prefix: Path, suffix: str) -> Path:
    return Path(f"{prefix}{suffix}")


def _read_fam_ids(prefix: Path) -> list[str]:
    fam_path = Path(f"{prefix}.fam")
    ids: list[str] = []
    with fam_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            parts = raw.strip().split()
            if parts:
                ids.append(parts[0])
    if not ids:
        raise ValueError(f"No sample IDs found in {fam_path}")
    return ids


def _read_bim_rows(prefix: Path) -> list[dict[str, object]]:
    bim_path = Path(f"{prefix}.bim")
    rows: list[dict[str, object]] = []
    with bim_path.open("r", encoding="utf-8", errors="replace") as fh:
        for idx, raw in enumerate(fh):
            chrom, snp, cm, pos, ref, alt = raw.strip().split()[:6]
            rows.append(
                {
                    "row_index": idx,
                    "chrom": chrom,
                    "snp": snp,
                    "cm": float(cm),
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                }
            )
    if not rows:
        raise ValueError(f"No BIM rows found in {bim_path}")
    return rows


def _read_site_stats_positions(site_stats_path: Path) -> tuple[list[int], dict[int, dict[str, object]]]:
    if not site_stats_path.is_file():
        return ([], {})
    with site_stats_path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        positions: list[int] = []
        info: dict[int, dict[str, object]] = {}
        for row in reader:
            pos = int(str(row["pos"]))
            positions.append(pos)
            info[pos] = row
    return (positions, info)


def _load_grm_matrix(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"GRM must be square, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"GRM contains non-finite values: {path}")
    return arr


def _read_grm_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                ids.append(line.split()[0])
    if not ids:
        raise ValueError(f"No IDs found in {path}")
    return ids


def _load_and_align_grm(grm_path: Path, grm_id_path: Path, target_ids: list[str]) -> np.ndarray:
    grm = _load_grm_matrix(grm_path)
    grm_ids = _read_grm_ids(grm_id_path)
    if len(grm_ids) != grm.shape[0]:
        raise ValueError(
            f"GRM/ID length mismatch: matrix n={grm.shape[0]}, ids={len(grm_ids)}"
        )
    index = {sid: i for i, sid in enumerate(grm_ids)}
    missing = [sid for sid in target_ids if sid not in index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"GRM is missing target IDs, first few: {preview}")
    order = np.asarray([index[sid] for sid in target_ids], dtype=np.intp)
    return np.asarray(grm[np.ix_(order, order)], dtype=np.float64)


def _variance(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return 0.0
    return float(np.var(arr, ddof=1))


def _parse_float_or_nan(value: object) -> float:
    try:
        return float(str(value))
    except Exception:
        return float("nan")


def _collapse_to_logic_bin01(row: np.ndarray) -> np.ndarray:
    arr = np.asarray(row, dtype=np.float64).reshape(-1)
    valid = np.isfinite(arr) & (arr >= 0.0)
    if not np.any(valid):
        raise ValueError("Genotype row has no valid values.")
    rounded = np.rint(arr[valid])
    geno = np.where(rounded <= 0.0, 0, np.where(rounded >= 2.0, 2, 1)).astype(np.uint8)
    c0 = int(np.sum(geno == 0))
    c2 = int(np.sum(geno == 2))
    mode02 = 2 if c2 > c0 else 0
    out = np.full(arr.shape[0], mode02, dtype=np.uint8)
    out[np.where(valid)[0]] = np.where(geno == 1, mode02, geno)
    return (out > 0).astype(np.float64)


def _residualize_indicator_against_main_effects(
    indicator: np.ndarray,
    member_rows: list[np.ndarray],
) -> np.ndarray:
    y = np.asarray(indicator, dtype=np.float64).reshape(-1)
    xcols = [np.ones(y.shape[0], dtype=np.float64)]
    xcols.extend(np.asarray(row, dtype=np.float64).reshape(-1) for row in member_rows)
    x = np.column_stack(xcols)
    gram = x.T @ x + 1e-8 * np.eye(x.shape[1], dtype=np.float64)
    beta = np.linalg.solve(gram, x.T @ y)
    return np.asarray(y - x @ beta, dtype=np.float64)


def _sample_background_from_grm(grm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    evals, evecs = np.linalg.eigh(np.asarray(grm, dtype=np.float64))
    z = rng.standard_normal(grm.shape[0])
    raw = evecs @ (np.sqrt(np.clip(evals, 0.0, None)) * z)
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    raw -= float(np.mean(raw))
    return raw


def _ols_assoc(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    design = np.column_stack([np.ones(xv.shape[0], dtype=np.float64), xv])
    beta = np.linalg.lstsq(design, yv, rcond=None)[0]
    resid = yv - design @ beta
    df = int(xv.shape[0] - design.shape[1])
    rss = float(resid @ resid)
    s2 = rss / max(df, 1)
    xtx_inv = np.linalg.inv(design.T @ design)
    se = float(np.sqrt(max(s2, 0.0) * xtx_inv[1, 1]))
    tval = float(beta[1] / se) if se > 0.0 else float("nan")
    pval = float(2.0 * stats.t.sf(abs(tval), df=max(df, 1))) if np.isfinite(tval) else float("nan")
    return {
        "beta": float(beta[1]),
        "se": se,
        "t": tval,
        "p": pval,
        "n": int(xv.shape[0]),
    }


def _read_validation_rows(result_path: Path, positions: set[int]) -> dict[int, dict[str, float | int | str]]:
    rows: dict[int, dict[str, float | int | str]] = {}
    with result_path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for rank, row in enumerate(reader, start=1):
            try:
                pos = int(str(row["pos"]))
            except Exception:
                continue
            if pos not in positions or pos in rows:
                continue
            rows[pos] = {
                "chrom": str(row.get("chrom", "")),
                "pos": pos,
                "rank": rank,
                "beta": _parse_float_or_nan(row.get("beta")),
                "se": _parse_float_or_nan(row.get("se")),
                "pwald": _parse_float_or_nan(row.get("pwald")),
                "chisq": _parse_float_or_nan(row.get("chisq")),
                "plrt": _parse_float_or_nan(row.get("plrt")),
            }
    return rows


def _run_gwas_validation(
    *,
    jx_exe: str,
    validate_bfile: Path,
    pheno_path: Path,
    grm_path: Path,
    out_dir: Path,
    prefix: str,
    model: str,
    threads: int,
    site_positions: list[int],
    weak_threshold: float,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(jx_exe),
        "gwas",
        "-bfile",
        str(validate_bfile),
        "-p",
        str(pheno_path),
        f"-{model}",
        "-k",
        str(grm_path),
        "-o",
        str(out_dir),
        "-prefix",
        str(prefix),
        "-t",
        str(max(1, int(threads))),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    result: dict[str, object] = {
        "enabled": True,
        "command": cmd,
        "returncode": int(proc.returncode),
        "scan_prefix": str(validate_bfile),
        "grm": str(grm_path),
        "model_requested": str(model),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        "status": "ok" if proc.returncode == 0 else "error",
    }
    if proc.returncode != 0:
        return result

    candidates: list[Path] = []
    for suffix in (f".{model}.tsv", ".fastlmm.tsv", ".lmm.tsv", ".lm.tsv"):
        candidates.extend(sorted(out_dir.glob(f"{prefix}.*{suffix}")))
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        result["status"] = "no_result_file"
        return result

    result_file = max(candidates, key=lambda p: p.stat().st_mtime)
    result["result_file"] = str(result_file)
    runtime_model = result_file.name.rsplit(".", 2)[-2]
    result["model_runtime"] = runtime_model

    exact_rows = _read_validation_rows(result_file, set(int(x) for x in site_positions))
    exact_serialized: list[dict[str, object]] = []
    valid_flags: list[bool] = []
    weak_flags: list[bool] = []
    for pos in site_positions:
        row = exact_rows.get(int(pos))
        if row is None:
            valid = False
            weak = False
            exact_serialized.append({"pos": int(pos), "found": False})
        else:
            beta = _parse_float_or_nan(row.get("beta"))
            pwald = _parse_float_or_nan(row.get("pwald"))
            valid = bool(np.isfinite(beta) and np.isfinite(pwald))
            weak = bool(valid and pwald > weak_threshold)
            exact_serialized.append({**row, "found": True, "valid": valid, "weak": weak})
        valid_flags.append(valid)
        weak_flags.append(weak)
    result["weak_threshold"] = float(weak_threshold)
    result["exact_sites"] = exact_serialized
    result["all_exact_found"] = bool(all(item.get("found", False) for item in exact_serialized))
    result["all_exact_valid"] = bool(all(valid_flags))
    result["both_sites_weak"] = bool(all(weak_flags))
    return result


def _write_pheno_outputs(out_prefix: Path, sample_ids: list[str], y: np.ndarray, seed: int, trait_name: str) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    with _out_path(out_prefix, ".pheno").open("w", encoding="utf-8") as fh:
        for sid, value in zip(sample_ids, yv):
            fh.write(f"{sid}\t{sid}\t{value:.6f}\n")
    with _out_path(out_prefix, ".pheno.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"IID\t{trait_name}\n")
        for sid, value in zip(sample_ids, yv):
            fh.write(f"{sid}\t{value:.6f}\n")
    rng = np.random.default_rng(int(seed) ^ 0xD9A35C714B1E208D)
    na_idx = set(rng.choice(len(sample_ids), size=int(round(0.1 * len(sample_ids))), replace=False).tolist())
    with _out_path(out_prefix, ".pheno.NA.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"IID\t{trait_name}\n")
        for i, (sid, value) in enumerate(zip(sample_ids, yv)):
            if i in na_idx:
                fh.write(f"{sid}\tNA\n")
            else:
                fh.write(f"{sid}\t{value:.6f}\n")


def _connected_components(r2: np.ndarray, threshold: float) -> tuple[np.ndarray, dict[int, int]]:
    n = int(r2.shape[0])
    comp = np.full(n, -1, dtype=np.int64)
    comp_id = 0
    for i in range(n):
        if comp[i] >= 0:
            continue
        stack = [i]
        comp[i] = comp_id
        while stack:
            u = stack.pop()
            nbr = np.where(r2[u] >= threshold)[0]
            for v in nbr:
                if comp[v] < 0:
                    comp[v] = comp_id
                    stack.append(int(v))
        comp_id += 1
    sizes: dict[int, int] = {}
    for cid in comp.tolist():
        sizes[cid] = sizes.get(cid, 0) + 1
    return (comp, sizes)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Simulate one two-site weak-marginal/strong-combination phenotype case "
            "from an existing chunk and a global GRM."
        )
    )
    p.add_argument("--chunk-prefix", default=str(DEFAULT_CHUNK_PREFIX), help="PLINK prefix for the chosen chunk.")
    p.add_argument("--site1", type=int, default=DEFAULT_SITE1, help="First causal site position.")
    p.add_argument("--site2", type=int, default=DEFAULT_SITE2, help="Second causal site position.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    p.add_argument("--prefix", default=None, help="Output prefix stem.")
    p.add_argument("--grm", default=str(DEFAULT_GRM), help="Global GRM .npy path.")
    p.add_argument("--grm-id", default=str(DEFAULT_GRM_ID), help="Global GRM ID file.")
    p.add_argument("--bg-pve", type=float, default=0.6, help="Background variance target.")
    p.add_argument("--cs-pve", type=float, default=0.12, help="Centered interaction variance target.")
    p.add_argument("--seed", type=int, default=20260520, help="Random seed.")
    p.add_argument(
        "--ld-block-r2-threshold",
        type=float,
        default=0.2,
        help="Heuristic r2 threshold used to define LD-block connected components.",
    )
    p.add_argument(
        "--validate-bfile",
        default=None,
        help=(
            "Optional genotype prefix used to validate the generated phenotype with `jx gwas`. "
            "Use the chunk prefix itself for chunk-only validation, or the whole-genome prefix "
            "for a genome-wide check."
        ),
    )
    p.add_argument(
        "--validate-model",
        choices=("fastlmm", "lmm", "lm"),
        default="fastlmm",
        help="Model used by optional `jx gwas` validation (default: fastlmm).",
    )
    p.add_argument(
        "--validate-threads",
        type=int,
        default=4,
        help="Threads used by optional `jx gwas` validation (default: 4).",
    )
    p.add_argument(
        "--validate-out-dir",
        default=None,
        help="Output directory for optional `jx gwas` validation files.",
    )
    p.add_argument(
        "--weak-site-p-threshold",
        type=float,
        default=0.05,
        help="Exact sites with validation pwald above this threshold are considered weak.",
    )
    p.add_argument(
        "--require-weak-sites",
        action="store_true",
        help="Exit non-zero if optional `jx gwas` validation does not show both exact sites as weak.",
    )
    p.add_argument(
        "--jx-exe",
        default="jx",
        help="Executable name/path used for optional `jx gwas` validation (default: jx).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    chunk_prefix = Path(args.chunk_prefix).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_id = chunk_prefix.name
    prefix_stem = str(args.prefix or f"{chunk_id}.weak_marginal_pair.centered.and")
    out_prefix = out_dir / prefix_stem
    site1 = int(args.site1)
    site2 = int(args.site2)
    bg_pve = float(args.bg_pve)
    cs_pve = float(args.cs_pve)
    residual_var = 1.0 - bg_pve - cs_pve
    if residual_var < 0.0:
        raise ValueError("bg_pve + cs_pve must be <= 1.0")

    bim_rows = _read_bim_rows(chunk_prefix)
    fam_ids = _read_fam_ids(chunk_prefix)
    pos_to_bim = {int(row["pos"]): row for row in bim_rows}
    if site1 not in pos_to_bim or site2 not in pos_to_bim:
        raise KeyError(f"Chosen positions are not both present in {chunk_prefix}")

    geno = load_bed_u8_matrix(str(chunk_prefix)).astype(np.float64)
    row1 = int(pos_to_bim[site1]["row_index"])
    row2 = int(pos_to_bim[site2]["row_index"])
    g1 = geno[row1].copy()
    g2 = geno[row2].copy()
    for arr in (g1, g2):
        arr[arr == 3.0] = np.nan

    b1 = _collapse_to_logic_bin01(g1)
    b2 = _collapse_to_logic_bin01(g2)
    pseudo = ((b1 > 0.0) & (b2 > 0.0)).astype(np.float64)
    centered_term = _residualize_indicator_against_main_effects(pseudo, [b1, b2])

    rng = np.random.default_rng(int(args.seed))
    grm = _load_and_align_grm(Path(args.grm).expanduser().resolve(), Path(args.grm_id).expanduser().resolve(), fam_ids)
    bg_raw = _sample_background_from_grm(grm, rng)
    bg_scale = np.sqrt(bg_pve / _variance(bg_raw)) if bg_pve > 0.0 and _variance(bg_raw) > 0.0 else 0.0
    bg_effect = bg_raw * bg_scale

    y = np.asarray(rng.standard_normal(len(fam_ids)) * np.sqrt(max(residual_var, 0.0)), dtype=np.float64)
    y += bg_effect
    coef0 = float(rng.uniform(-1.0, 1.0))
    causal_raw = centered_term * coef0
    causal_scale = np.sqrt(cs_pve / _variance(causal_raw)) if cs_pve > 0.0 and _variance(causal_raw) > 0.0 else 0.0
    causal_effect = coef0 * causal_scale
    y += centered_term * causal_effect

    trait_name = f"weakpair_bg{bg_pve:.3f}_cs{cs_pve:.3f}_grm_centered"
    _write_pheno_outputs(out_prefix, fam_ids, y, int(args.seed), trait_name)

    with _out_path(out_prefix, ".fixed.effects.tsv").open("w", encoding="utf-8") as fh:
        fh.write("term_id\tkind\tlogic\tsites\tlabel\teffect\n")
        fh.write(
            "1\tlogic_gate\tand\t"
            f"{pos_to_bim[site1]['chrom']}:{site1};{pos_to_bim[site2]['chrom']}:{site2}\t"
            f"{pos_to_bim[site1]['chrom']}_{site1}[{pos_to_bim[site1]['ref']}>{pos_to_bim[site1]['alt']}]&"
            f"{pos_to_bim[site2]['chrom']}_{site2}[{pos_to_bim[site2]['ref']}>{pos_to_bim[site2]['alt']}]\t"
            f"{causal_effect:.10f}\n"
        )

    with _out_path(out_prefix, ".causal.sites.tsv").open("w", encoding="utf-8") as fh:
        fh.write(f"{pos_to_bim[site1]['chrom']}\t{site1}\t{site1}\n")
        fh.write(f"{pos_to_bim[site2]['chrom']}\t{site2}\t{site2}\n")

    with _out_path(out_prefix, ".random.effects.tsv").open("w", encoding="utf-8") as fh:
        fh.write("sample_index\tsample_id\tsource\trole\teffect\n")
        for i, (sid, value) in enumerate(zip(fam_ids, bg_effect), start=1):
            fh.write(f"{i}\t{sid}\tgrm\tbackground\t{value:.10f}\n")

    with _out_path(out_prefix, ".pseudo.tsv").open("w", encoding="utf-8") as fh:
        fh.write("IID\tdosage1\tdosage2\tbinary1\tbinary2\tpseudo_and\tcentered_term\n")
        for sid, d1, d2, x1, x2, pp, cc in zip(fam_ids, g1, g2, b1, b2, pseudo, centered_term):
            d1_txt = "NA" if not np.isfinite(d1) else f"{d1:.0f}"
            d2_txt = "NA" if not np.isfinite(d2) else f"{d2:.0f}"
            fh.write(f"{sid}\t{d1_txt}\t{d2_txt}\t{x1:.0f}\t{x2:.0f}\t{pp:.0f}\t{cc:.10f}\n")

    site_stats_path = REPO_ROOT / "test.Tgarfield_out" / "summary" / f"{chunk_id}.site_stats.tsv"
    ld_positions, site_stats = _read_site_stats_positions(site_stats_path)
    pair_r2 = float("nan")
    block_site_count = {}
    block1 = None
    block2 = None
    if ld_positions:
        chrom = str(pos_to_bim[site1]["chrom"])
        r2_mat, _out_chrom, _out_pos = bed_ldblock_r2_rust(
            str(chunk_prefix),
            [chrom],
            [int(min(ld_positions))],
            [int(max(ld_positions))],
            [chrom] * len(ld_positions),
            [int(p) for p in ld_positions],
            threads=1,
        )
        r2_arr = np.asarray(r2_mat, dtype=np.float64)
        pos_to_idx = {int(pos): idx for idx, pos in enumerate(ld_positions)}
        if site1 in pos_to_idx and site2 in pos_to_idx:
            i = pos_to_idx[site1]
            j = pos_to_idx[site2]
            pair_r2 = float(r2_arr[i, j])
            comp, comp_sizes = _connected_components(r2_arr, float(args.ld_block_r2_threshold))
            block1 = int(comp[i])
            block2 = int(comp[j])
            block_site_count = {
                "site1_block_size": int(comp_sizes.get(block1, 0)),
                "site2_block_size": int(comp_sizes.get(block2, 0)),
            }

    assoc_rows = [
        {"feature": "dosage1", **_ols_assoc(g1, y)},
        {"feature": "dosage2", **_ols_assoc(g2, y)},
        {"feature": "binary1", **_ols_assoc(b1, y)},
        {"feature": "binary2", **_ols_assoc(b2, y)},
        {"feature": "pseudo_and", **_ols_assoc(pseudo, y)},
        {"feature": "centered_term", **_ols_assoc(centered_term, y)},
    ]
    with _out_path(out_prefix, ".assoc.tsv").open("w", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["feature", "beta", "se", "t", "p", "n"], delimiter="\t")
        writer.writeheader()
        for row in assoc_rows:
            writer.writerow(row)

    def _maf_from_dosage(row: np.ndarray) -> float:
        mask = np.isfinite(row)
        if not np.any(mask):
            return float("nan")
        af = float(np.nanmean(row[mask]) / 2.0)
        return float(min(af, 1.0 - af))

    validation: dict[str, object] | None = None
    if args.validate_bfile:
        validate_out_dir = (
            Path(args.validate_out_dir).expanduser().resolve()
            if args.validate_out_dir
            else out_dir / "validation"
        )
        validation = _run_gwas_validation(
            jx_exe=str(args.jx_exe),
            validate_bfile=Path(args.validate_bfile).expanduser().resolve(),
            pheno_path=_out_path(out_prefix, ".pheno.txt"),
            grm_path=Path(args.grm).expanduser().resolve(),
            out_dir=validate_out_dir,
            prefix=f"{prefix_stem}.validate",
            model=str(args.validate_model),
            threads=int(args.validate_threads),
            site_positions=[site1, site2],
            weak_threshold=float(args.weak_site_p_threshold),
        )

    summary = {
        "chunk_prefix": str(chunk_prefix),
        "chunk_id": chunk_id,
        "grm": str(Path(args.grm).expanduser().resolve()),
        "grm_id": str(Path(args.grm_id).expanduser().resolve()),
        "seed": int(args.seed),
        "bg_pve": bg_pve,
        "cs_pve": cs_pve,
        "residual_var": residual_var,
        "trait_name": trait_name,
        "logic": "and",
        "effect_model": "centered_interaction",
        "site1": {
            "chrom": str(pos_to_bim[site1]["chrom"]),
            "pos": site1,
            "snp": str(pos_to_bim[site1]["snp"]),
            "ref": str(pos_to_bim[site1]["ref"]),
            "alt": str(pos_to_bim[site1]["alt"]),
            "maf": float(site_stats.get(site1, {}).get("freq", _maf_from_dosage(g1))),
            "binary_freq": float(np.mean(b1)),
            "ld_block_id": block1,
        },
        "site2": {
            "chrom": str(pos_to_bim[site2]["chrom"]),
            "pos": site2,
            "snp": str(pos_to_bim[site2]["snp"]),
            "ref": str(pos_to_bim[site2]["ref"]),
            "alt": str(pos_to_bim[site2]["alt"]),
            "maf": float(site_stats.get(site2, {}).get("freq", _maf_from_dosage(g2))),
            "binary_freq": float(np.mean(b2)),
            "ld_block_id": block2,
        },
        "pair_r2": pair_r2,
        "ld_block_r2_threshold": float(args.ld_block_r2_threshold),
        "pseudo_and_freq": float(np.mean(pseudo)),
        "centered_term_var": _variance(centered_term),
        "background_var": _variance(bg_effect),
        "phenotype_var": _variance(y),
        "causal_effect": causal_effect,
        "raw_coef_before_scaling": coef0,
        **block_site_count,
        "assoc": assoc_rows,
    }
    if validation is not None:
        summary["validation"] = validation
    with _out_path(out_prefix, ".case.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=True, indent=2)

    print(f"Chunk        : {chunk_prefix}")
    print(f"Output prefix: {out_prefix}")
    print(f"Chosen sites : {site1}, {site2}")
    print(f"Pair r2      : {pair_r2:.6f}")
    print(f"Pseudo freq  : {np.mean(pseudo):.6f}")
    for row in assoc_rows:
        print(f"{row['feature']:12s} beta={row['beta']:.6f} p={row['p']:.6g}")
    if validation is not None:
        print(f"Validation   : {validation.get('status')}")
        for row in validation.get("exact_sites", []):
            pos = row.get("pos")
            if not row.get("found", False):
                print(f"  site {pos}: not found in validation result")
                continue
            beta = row.get("beta")
            pwald = row.get("pwald")
            weak = row.get("weak")
            print(f"  site {pos}: beta={beta} pwald={pwald} weak={weak}")
    print(f"Wrote: {_out_path(out_prefix, '.pheno.txt')}")
    print(f"Wrote: {_out_path(out_prefix, '.fixed.effects.tsv')}")
    print(f"Wrote: {_out_path(out_prefix, '.random.effects.tsv')}")
    print(f"Wrote: {_out_path(out_prefix, '.assoc.tsv')}")
    print(f"Wrote: {_out_path(out_prefix, '.case.json')}")
    if bool(args.require_weak_sites):
        if validation is None or not bool(validation.get("both_sites_weak", False)):
            print("Validation failed: both exact sites were not weak under the requested threshold.")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
