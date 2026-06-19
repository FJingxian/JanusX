#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from janusx.assoc import workflow_model_packed as w


GAMMA_RE = re.compile(r"Mean GRAMMAR-Gamma value = ([-+0-9.eE]+)")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "test" / "splmm_fastgwa_compare"
    return argparse.ArgumentParser(
        description="Compare GCTA fastGWA gamma against JanusX SparseLMM r_hat/c_trait across traits.",
    ).parse_args()


def build_config() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "test" / "splmm_fastgwa_compare"
    return {
        "repo_root": repo_root,
        "bfile": Path("/Users/jingxianfu/cubic/cubic_All.maf0.02"),
        "gcta_bin": Path("/Users/jingxianfu/Downloads/gcta-1.95.1-macOS-arm64/bin/gcta64"),
        "grm_sparse_prefix": data_root / "cubic_home_sp05",
        "jx_spgrm": data_root / "cubic_All.maf0.02.cGRM.spgrm",
        "pheno_gcta": repo_root / "test.cubic" / "cubic.gcta.traits.txt",
        "scale_summary": data_root / "multi_trait_scale" / "summary.json",
        "extract": data_root / "extract5000_ids.txt",
        "out_dir": data_root / "multi_trait_gamma",
        "gcta_threads": 4,
        "jx_threads": 8,
        "maf": 0.02,
        "geno": 0.05,
        "rhat_markers": 1000,
        "rhat_seed": 20260527,
        "traits": ["ATI", "ED", "EW", "KNPE", "LBT", "PH"],
    }


def load_scale_summary(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["traits"])


def load_extract_abs_indices(bim_path: Path, extract_path: Path) -> tuple[np.ndarray, int]:
    wanted = [line.strip() for line in extract_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted_set = set(wanted)
    abs_idx: list[int] = []
    found = 0
    with bim_path.open("r", encoding="utf-8") as fh:
        for row_idx, line in enumerate(fh):
            fields = line.rstrip("\n").split()
            if len(fields) >= 2 and fields[1] in wanted_set:
                abs_idx.append(row_idx)
                found += 1
    return np.asarray(abs_idx, dtype=np.int64), found


def load_fam_ids(fam_path: Path) -> list[tuple[str, str]]:
    ids: list[tuple[str, str]] = []
    with fam_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split()
            if len(fields) >= 2:
                ids.append((fields[0], fields[1]))
    return ids


def load_gcta_pheno_map(pheno_path: Path) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    with pheno_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split()
            if len(fields) >= 3:
                out[(fields[0], fields[1])] = fields
    return out


def is_missing_token(token: str) -> bool:
    x = token.strip()
    return (not x) or x.upper() == "NA" or x == "-9" or x.lower() == "nan"


def load_trait_vector(
    fam_ids: list[tuple[str, str]],
    pheno_map: dict[tuple[str, str], list[str]],
    mpheno: int,
) -> tuple[np.ndarray, np.ndarray]:
    col_idx = 1 + int(mpheno)
    sample_idx: list[int] = []
    y_vec: list[float] = []
    for i, key in enumerate(fam_ids):
        row = pheno_map.get(key)
        if row is None or col_idx >= len(row):
            continue
        token = row[col_idx]
        if is_missing_token(token):
            continue
        sample_idx.append(i)
        y_vec.append(float(token))
    return (
        np.asarray(sample_idx, dtype=np.int64),
        np.asarray(y_vec, dtype=np.float64),
    )


def ensure_gcta_gamma(
    *,
    gcta_bin: Path,
    bfile: Path,
    grm_sparse_prefix: Path,
    pheno_gcta: Path,
    extract: Path,
    out_prefix: Path,
    mpheno: int,
    threads: int,
    maf: float,
    geno: float,
) -> float:
    log_path = out_prefix.with_suffix(".log")
    if log_path.exists():
        gamma = parse_gamma_from_log(log_path)
        if gamma is not None:
            print(f"[reuse] trait={out_prefix.name} gamma={gamma:.9g}")
            return gamma

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(gcta_bin),
        "--bfile",
        str(bfile),
        "--grm-sparse",
        str(grm_sparse_prefix),
        "--fastGWA-mlm",
        "--pheno",
        str(pheno_gcta),
        "--mpheno",
        str(int(mpheno)),
        "--maf",
        str(float(maf)),
        "--geno",
        str(float(geno)),
        "--extract",
        str(extract),
        "--thread-num",
        str(int(threads)),
        "--out",
        str(out_prefix),
    ]
    print(f"[gcta] running trait={out_prefix.name} mpheno={mpheno}")
    subprocess.run(cmd, check=True, cwd=out_prefix.parent.parent.parent.parent)
    gamma = parse_gamma_from_log(log_path)
    if gamma is None:
        raise RuntimeError(f"failed to parse gamma from {log_path}")
    return gamma


def parse_gamma_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = GAMMA_RE.search(line)
        if m:
            return float(m.group(1))
    return None


def compute_jx_rhat(
    *,
    bfile: Path,
    jx_spgrm: Path,
    sample_idx: np.ndarray,
    y_vec: np.ndarray,
    sigma_g2: float,
    sigma_e2: float,
    extract_abs_idx: np.ndarray,
    maf: float,
    geno: float,
    threads: int,
    rhat_markers: int,
    rhat_seed: int,
) -> dict[str, float]:
    meta = w._splmm_bed_logic_meta_selected(
        str(bfile),
        sample_indices=sample_idx,
        maf_threshold=float(maf),
        max_missing_rate=float(geno),
        het_threshold=0.0,
        snps_only=False,
    )
    row_idx = np.asarray(meta["row_indices"], dtype=np.int64)
    mask = np.isin(row_idx, extract_abs_idx)
    row_idx_sub = np.ascontiguousarray(row_idx[mask], dtype=np.int64)
    maf_sub = np.ascontiguousarray(np.asarray(meta["maf"], dtype=np.float32)[mask], dtype=np.float32)
    miss_sub = np.ascontiguousarray(np.asarray(meta["missing_rate"], dtype=np.float32)[mask], dtype=np.float32)
    flip_sub = np.ascontiguousarray(np.asarray(meta["row_flip"], dtype=np.bool_)[mask], dtype=np.bool_)
    if row_idx_sub.size == 0:
        raise RuntimeError("no SNPs remained after intersecting trait QC rows with extract set")
    if (not np.isfinite(float(sigma_g2))) or float(sigma_g2) <= 0.0:
        raise RuntimeError(f"invalid sigma_g2 for SparseLMM lambda conversion: {sigma_g2}")
    if (not np.isfinite(float(sigma_e2))) or float(sigma_e2) < 0.0:
        raise RuntimeError(f"invalid sigma_e2 for SparseLMM lambda conversion: {sigma_e2}")
    lbd = float(float(sigma_e2) / float(sigma_g2))

    out = w.jxrs.splmm_assoc_pcg_bed(
        str(bfile),
        y_vec,
        float(lbd),
        x_cov=None,
        sample_indices=sample_idx,
        operator_sample_indices=sample_idx,
        site_keep=None,
        tol=1e-5,
        max_iter=200,
        block_rows=512,
        std_eps=1e-12,
        use_train_maf=True,
        threads=int(threads),
        model="add",
        rhat_markers=int(rhat_markers),
        rhat_seed=int(rhat_seed),
        packed=None,
        packed_n_samples=0,
        maf=maf_sub,
        row_flip=flip_sub,
        row_missing=miss_sub,
        row_indices=row_idx_sub,
        sparse_jxgrm_path=str(jx_spgrm),
        stage1_progress_callback=None,
        scan_progress_callback=None,
        progress_every=0,
        rhat_tol=1e-3,
        scan_mode="approx",
    )
    out_t = tuple(out)
    return {
        "r_hat": float(out_t[0]),
        "rhat_requested": int(out_t[7]),
        "rhat_used": int(out_t[8]),
        "subset_qc_snps": int(row_idx_sub.shape[0]),
    }


def main() -> int:
    _ = parse_args()
    cfg = build_config()
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg")
    os.environ.setdefault("PYTHONPATH", "python")

    fam_ids = load_fam_ids(Path(f"{cfg['bfile']}.fam"))
    pheno_map = load_gcta_pheno_map(Path(cfg["pheno_gcta"]))
    extract_abs_idx, extract_found = load_extract_abs_indices(
        Path(f"{cfg['bfile']}.bim"),
        Path(cfg["extract"]),
    )
    scale_rows = load_scale_summary(Path(cfg["scale_summary"]))
    by_trait = {str(row["trait"]): row for row in scale_rows}

    results: list[dict[str, object]] = []
    for trait in cfg["traits"]:
        row = by_trait[str(trait)]
        mpheno = int(row["mpheno"])
        sample_idx, y_vec = load_trait_vector(fam_ids, pheno_map, mpheno)
        if sample_idx.size != int(row["n"]):
            raise RuntimeError(
                f"trait={trait} sample count mismatch: current={sample_idx.size}, summary={row['n']}"
            )
        gamma = ensure_gcta_gamma(
            gcta_bin=Path(cfg["gcta_bin"]),
            bfile=Path(cfg["bfile"]),
            grm_sparse_prefix=Path(cfg["grm_sparse_prefix"]),
            pheno_gcta=Path(cfg["pheno_gcta"]),
            extract=Path(cfg["extract"]),
            out_prefix=out_dir / f"gcta_{str(trait).lower()}",
            mpheno=mpheno,
            threads=int(cfg["gcta_threads"]),
            maf=float(cfg["maf"]),
            geno=float(cfg["geno"]),
        )

        null_fit = w._splmm_sparse_null_fit(
            jxgrm_path=str(cfg["jx_spgrm"]),
            sample_idx=sample_idx,
            y_vec=y_vec,
            x_cov=None,
            progress_callback=None,
        )
        sigma_g2 = float(null_fit["sigma_g2"])
        sigma_e2 = float(null_fit["sigma_e2"])
        c_current = float(row["gcta_vp"]) / float(sigma_g2 + sigma_e2)
        rhat_info = compute_jx_rhat(
            bfile=Path(cfg["bfile"]),
            jx_spgrm=Path(cfg["jx_spgrm"]),
            sample_idx=sample_idx,
            y_vec=y_vec,
            sigma_g2=sigma_g2,
            sigma_e2=sigma_e2,
            extract_abs_idx=extract_abs_idx,
            maf=float(cfg["maf"]),
            geno=float(cfg["geno"]),
            threads=int(cfg["jx_threads"]),
            rhat_markers=int(cfg["rhat_markers"]),
            rhat_seed=int(cfg["rhat_seed"]),
        )
        r_hat = float(rhat_info["r_hat"])
        c_saved = float(row["scale_sum"])
        rhat_over_c_saved = r_hat / c_saved
        rhat_over_c_current = r_hat / c_current
        ratio_saved = rhat_over_c_saved / gamma if gamma > 0.0 else float("nan")
        ratio_current = rhat_over_c_current / gamma if gamma > 0.0 else float("nan")

        rec = {
            "trait": trait,
            "mpheno": mpheno,
            "n": int(sample_idx.size),
            "gcta_gamma": gamma,
            "r_hat": r_hat,
            "rhat_requested": int(rhat_info["rhat_requested"]),
            "rhat_used": int(rhat_info["rhat_used"]),
            "extract_requested": int(extract_abs_idx.shape[0]),
            "extract_found_in_bim": int(extract_found),
            "subset_qc_snps": int(rhat_info["subset_qc_snps"]),
            "sigma_g2_current": sigma_g2,
            "sigma_e2_current": sigma_e2,
            "lambda_current": float(null_fit["lambda"]),
            "c_trait_saved": c_saved,
            "c_trait_current": c_current,
            "c_saved_over_current": c_saved / c_current if c_current > 0.0 else float("nan"),
            "rhat_over_c_saved": rhat_over_c_saved,
            "rhat_over_c_current": rhat_over_c_current,
            "ratio_saved_over_gamma": ratio_saved,
            "ratio_current_over_gamma": ratio_current,
        }
        print(
            "[trait] "
            f"{trait} gamma={gamma:.9g} r_hat={r_hat:.9g} "
            f"r_hat/c_saved={rhat_over_c_saved:.9g} ratio={ratio_saved:.6f}"
        )
        results.append(rec)

    ratio_saved_arr = np.asarray([float(x["ratio_saved_over_gamma"]) for x in results], dtype=np.float64)
    ratio_current_arr = np.asarray([float(x["ratio_current_over_gamma"]) for x in results], dtype=np.float64)
    summary = {
        "config": {
            "bfile": str(cfg["bfile"]),
            "gcta_bin": str(cfg["gcta_bin"]),
            "grm_sparse_prefix": str(cfg["grm_sparse_prefix"]),
            "jx_spgrm": str(cfg["jx_spgrm"]),
            "pheno_gcta": str(cfg["pheno_gcta"]),
            "scale_summary": str(cfg["scale_summary"]),
            "extract": str(cfg["extract"]),
            "out_dir": str(out_dir),
            "gcta_threads": int(cfg["gcta_threads"]),
            "jx_threads": int(cfg["jx_threads"]),
            "maf": float(cfg["maf"]),
            "geno": float(cfg["geno"]),
            "rhat_markers": int(cfg["rhat_markers"]),
            "rhat_seed": int(cfg["rhat_seed"]),
        },
        "traits": results,
        "ratio_saved_over_gamma_summary": {
            "mean": float(np.nanmean(ratio_saved_arr)),
            "sd": float(np.nanstd(ratio_saved_arr)),
            "min": float(np.nanmin(ratio_saved_arr)),
            "max": float(np.nanmax(ratio_saved_arr)),
        },
        "ratio_current_over_gamma_summary": {
            "mean": float(np.nanmean(ratio_current_arr)),
            "sd": float(np.nanstd(ratio_current_arr)),
            "min": float(np.nanmin(ratio_current_arr)),
            "max": float(np.nanmax(ratio_current_arr)),
        },
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_json": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
