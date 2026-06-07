#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from janusx.assoc import workflow_model_packed as w
from janusx.janusx import bed_packed_decode_rows_f32, load_bed_2bit_packed


GAMMA_RE = re.compile(r"Mean GRAMMAR-Gamma value = ([-+0-9.eE]+)")


def load_fam_ids(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            fields = line.split()
            if len(fields) >= 2:
                out.append((fields[0], fields[1]))
    return out


def load_gcta_pheno_map(pheno_path: Path) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    with pheno_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            fields = line.split()
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


def load_extract_abs_indices(bim_path: Path, extract_path: Path) -> np.ndarray:
    wanted = {
        line.strip()
        for line in extract_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    out: list[int] = []
    with bim_path.open("r", encoding="utf-8") as fh:
        for row_idx, line in enumerate(fh):
            fields = line.split()
            if len(fields) >= 2 and fields[1] in wanted:
                out.append(row_idx)
    return np.asarray(out, dtype=np.int64)


def parse_gcta_gamma(log_path: Path) -> float:
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = GAMMA_RE.search(line)
        if m:
            return float(m.group(1))
    raise RuntimeError(f"failed to parse fastGWA gamma from {log_path}")


def summarize_metric(values: np.ndarray, gamma: float) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    mean = float(np.mean(vals))
    median = float(np.median(vals))
    return {
        "n": int(vals.shape[0]),
        "mean": mean,
        "median": median,
        "sd": float(np.std(vals)),
        "abs_err_mean": float(abs(mean - gamma)),
        "abs_err_median": float(abs(median - gamma)),
        "ratio_mean_over_gamma": float(mean / gamma),
        "ratio_median_over_gamma": float(median / gamma),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    cfg = {
        "bfile": Path("/Users/jingxianfu/cubic/cubic_All.maf0.02"),
        "jx_spgrm": repo / "test" / "splmm_fastgwa_compare" / "cubic_All.maf0.02.cGRM.spgrm",
        "pheno_gcta": repo / "test.cubic" / "cubic.gcta.traits.txt",
        "extract": repo / "test" / "splmm_fastgwa_compare" / "extract5000_ids.txt",
        "gcta_log": repo / "test" / "splmm_fastgwa_compare" / "gcta_fastgwa_ew.log",
        "out_json": repo / "test" / "splmm_fastgwa_compare" / "rhat_mode_compare_ew_q0.json",
        "mpheno": 12,
        "maf": 0.02,
        "geno": 0.05,
        "seed": 20260527,
        "sample_sizes": [1000, 2000, 5000],
    }

    gamma = parse_gcta_gamma(Path(cfg["gcta_log"]))
    fam_ids = load_fam_ids(Path(f"{cfg['bfile']}.fam"))
    pheno_map = load_gcta_pheno_map(Path(cfg["pheno_gcta"]))
    sample_idx, y_vec = load_trait_vector(fam_ids, pheno_map, int(cfg["mpheno"]))
    extract_abs_idx = load_extract_abs_indices(
        Path(f"{cfg['bfile']}.bim"),
        Path(cfg["extract"]),
    )

    null_fit = w._jxlmm_sparse_null_fit(
        jxgrm_path=str(cfg["jx_spgrm"]),
        sample_idx=sample_idx,
        y_vec=y_vec,
        x_cov=None,
        progress_callback=None,
    )
    sigma_g2 = float(null_fit["sigma_g2"])
    sigma_e2 = float(null_fit["sigma_e2"])

    meta = w._jxlmm_bed_logic_meta_selected(
        str(cfg["bfile"]),
        sample_indices=sample_idx,
        maf_threshold=float(cfg["maf"]),
        max_missing_rate=float(cfg["geno"]),
        het_threshold=0.0,
        snps_only=False,
    )
    row_idx = np.asarray(meta["row_indices"], dtype=np.int64)
    mask = np.isin(row_idx, extract_abs_idx)
    row_idx_sub = np.ascontiguousarray(row_idx[mask], dtype=np.int64)
    maf_sub = np.ascontiguousarray(
        np.asarray(meta["maf"], dtype=np.float32)[mask],
        dtype=np.float32,
    )
    flip_sub = np.ascontiguousarray(
        np.asarray(meta["row_flip"], dtype=np.bool_)[mask],
        dtype=np.bool_,
    )

    packed_raw, _miss_raw, _maf_raw, _std_raw, n_samples_full = load_bed_2bit_packed(
        str(cfg["bfile"])
    )
    packed = np.ascontiguousarray(np.asarray(packed_raw, dtype=np.uint8))

    k_dense = w._jxlmm_load_sparse_grm_subset_dense(str(cfg["jx_spgrm"]), sample_idx)
    n = int(sample_idx.shape[0])
    x = np.ones((n, 1), dtype=np.float64)
    v = sigma_g2 * k_dense + sigma_e2 * np.eye(n, dtype=np.float64)
    cf = cho_factor(v, lower=True, check_finite=False)
    vinv_x = cho_solve(cf, x, check_finite=False)
    xt_vinv_x_inv = np.linalg.inv(x.T @ vinv_x)

    rng = np.random.default_rng(int(cfg["seed"]))
    order = np.sort(
        rng.choice(
            row_idx_sub.shape[0],
            size=min(int(cfg["sample_sizes"][-1]), row_idx_sub.shape[0]),
            replace=False,
        )
    )

    out: dict[str, object] = {
        "gamma_gcta": gamma,
        "sigma_g2": sigma_g2,
        "sigma_e2": sigma_e2,
        "n_trait": int(sample_idx.shape[0]),
        "n_extract_qc": int(row_idx_sub.shape[0]),
        "seed": int(cfg["seed"]),
        "sample_sizes": list(cfg["sample_sizes"]),
        "subsets": {},
    }

    for size in cfg["sample_sizes"]:
        use = order[: min(int(size), order.shape[0])]
        rows_local = np.ascontiguousarray(row_idx_sub[use], dtype=np.int64)
        maf_local = np.ascontiguousarray(maf_sub[use], dtype=np.float32)
        flip_local = np.ascontiguousarray(flip_sub[use], dtype=np.bool_)
        g = np.ascontiguousarray(
            np.asarray(
                bed_packed_decode_rows_f32(
                    packed,
                    int(n_samples_full),
                    rows_local,
                    flip_local,
                    maf_local,
                    np.ascontiguousarray(sample_idx, dtype=np.int64),
                ),
                dtype=np.float64,
            ),
            dtype=np.float64,
        )
        s_sq = np.einsum("ij,ij->i", g, g)
        s_sum = np.sum(g, axis=1)
        s_m_s = s_sq - (s_sum * s_sum) / float(n)
        vinv_g = cho_solve(cf, g.T, check_finite=False).T
        s_vinv_s = np.einsum("ij,ij->i", g, vinv_g)
        sx = g @ vinv_x
        s_p_s = s_vinv_s - np.einsum("ij,jk,ik->i", sx, xt_vinv_x_inv, sx)

        subset_res = {
            "current_sp_over_sm": summarize_metric(s_p_s / s_m_s, gamma),
            "vinv_over_ss": summarize_metric(s_vinv_s / s_sq, gamma),
            "vinv_over_sm": summarize_metric(s_vinv_s / s_m_s, gamma),
            "sp_over_ss": summarize_metric(s_p_s / s_sq, gamma),
        }
        out["subsets"][f"n{int(size)}"] = subset_res

    out_path = Path(cfg["out_json"])
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
