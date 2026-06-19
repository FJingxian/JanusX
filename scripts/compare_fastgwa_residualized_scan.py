#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erfc

from janusx.assoc import workflow_model_packed as w
from janusx.janusx import bed_packed_decode_rows_f32, load_bed_2bit_packed


MED_CHI1 = 0.454936423119572
GAMMA_RE = re.compile(r"Mean GRAMMAR-Gamma value = ([-+0-9.eE]+)")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    data_root = repo / "test" / "splmm_fastgwa_compare"
    parser = argparse.ArgumentParser(
        description=(
            "Experimental fastGWA-style residualized scan: "
            "y_adj = M_X y, gamma = mean(g'V^-1 g / g'g), "
            "beta = g'V^-1 y_adj / (gamma g'g)."
        )
    )
    parser.add_argument(
        "--bfile",
        type=Path,
        default=Path("/Users/jingxianfu/cubic/cubic_All.maf0.02"),
    )
    parser.add_argument(
        "--spgrm",
        type=Path,
        default=Path("/Users/jingxianfu/cubic/cubic_All.maf0.02.spgrm"),
    )
    parser.add_argument(
        "--pheno",
        type=Path,
        default=repo / "test.cubic" / "cubic_All.Agronomic_23Traits.txt",
    )
    parser.add_argument(
        "--trait",
        default="EW",
        help="Trait name for JanusX-style header phenotype table.",
    )
    parser.add_argument(
        "--mpheno",
        type=int,
        default=None,
        help="Use GCTA-style phenotype file with FID IID phenotypes; 1-based phenotype index.",
    )
    parser.add_argument(
        "--cov-tsv",
        type=Path,
        default=None,
        help=(
            "Optional covariate table. Supported header formats: "
            "'IID cov1 ...' or 'FID IID cov1 ...'."
        ),
    )
    parser.add_argument("--maf", type=float, default=0.02)
    parser.add_argument("--geno", type=float, default=0.05)
    parser.add_argument(
        "--extract",
        type=Path,
        default=data_root / "extract5000_ids.txt",
        help="Evaluate on this SNP id list.",
    )
    parser.add_argument(
        "--fastgwa",
        type=Path,
        default=data_root / "gcta_fastgwa_ew.fastGWA",
    )
    parser.add_argument(
        "--fastgwa-log",
        type=Path,
        default=data_root / "gcta_fastgwa_ew.log",
    )
    parser.add_argument(
        "--jx-tsv",
        type=Path,
        default=repo / "test0012" / "cubic_All.maf0.02.EW.splmm.tsv",
        help="Optional current JanusX SparseLMM TSV for side-by-side comparison.",
    )
    parser.add_argument("--gamma-markers", type=int, default=2000)
    parser.add_argument("--gamma-seed", type=int, default=20260527)
    parser.add_argument("--scan-block", type=int, default=4096)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument(
        "--out-tsv",
        type=Path,
        default=None,
        help="Optional TSV for the experimental residualized scan.",
    )
    return parser.parse_args()


def parse_fastgwa_gamma(log_path: Path) -> float:
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = GAMMA_RE.search(line)
        if m:
            return float(m.group(1))
    raise RuntimeError(f"failed to parse fastGWA gamma from {log_path}")


def is_missing_token(token: str) -> bool:
    x = token.strip()
    return (not x) or x.upper() == "NA" or x.lower() == "NAN" or x == "-9"


def load_fam_iids(fam_path: Path) -> list[str]:
    iids: list[str] = []
    with fam_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            fields = line.split()
            if len(fields) >= 2:
                iids.append(fields[1])
    return iids


def load_trait_from_named_table(
    pheno_path: Path,
    fam_iids: list[str],
    trait: str,
) -> tuple[np.ndarray, np.ndarray]:
    value_by_iid: dict[str, float] = {}
    with pheno_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if trait not in header:
            raise RuntimeError(f"trait {trait} not found in {pheno_path}")
        trait_col = header.index(trait)
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if trait_col >= len(fields):
                continue
            iid = fields[0].strip()
            if iid == "":
                continue
            token = fields[trait_col]
            if is_missing_token(token):
                continue
            value_by_iid[iid] = float(token)
    sample_idx: list[int] = []
    y_vec: list[float] = []
    for i, iid in enumerate(fam_iids):
        if iid in value_by_iid:
            sample_idx.append(i)
            y_vec.append(value_by_iid[iid])
    return np.asarray(sample_idx, dtype=np.int64), np.asarray(y_vec, dtype=np.float64)


def load_trait_from_gcta_table(
    pheno_path: Path,
    fam_iids: list[str],
    mpheno: int,
) -> tuple[np.ndarray, np.ndarray]:
    if mpheno <= 0:
        raise RuntimeError(f"mpheno must be positive, got {mpheno}")
    value_by_iid: dict[str, float] = {}
    col_idx = 1 + int(mpheno)
    with pheno_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            fields = line.split()
            if len(fields) <= col_idx:
                continue
            iid = fields[1]
            token = fields[col_idx]
            if is_missing_token(token):
                continue
            value_by_iid[iid] = float(token)
    sample_idx: list[int] = []
    y_vec: list[float] = []
    for i, iid in enumerate(fam_iids):
        if iid in value_by_iid:
            sample_idx.append(i)
            y_vec.append(value_by_iid[iid])
    return np.asarray(sample_idx, dtype=np.int64), np.asarray(y_vec, dtype=np.float64)


def load_covariates(
    cov_path: Path,
    fam_iids: list[str],
) -> tuple[dict[str, np.ndarray], int]:
    with cov_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().split()
        if len(header) < 2:
            raise RuntimeError(f"invalid covariate header in {cov_path}")
        if header[0].upper() == "FID" and len(header) >= 3 and header[1].upper() == "IID":
            iid_col = 1
            cov_start = 2
        else:
            iid_col = 0
            cov_start = 1
        if len(header) <= cov_start:
            raise RuntimeError(f"no covariate columns found in {cov_path}")
        cov_by_iid: dict[str, np.ndarray] = {}
        n_cov = len(header) - cov_start
        for line in fh:
            fields = line.split()
            if len(fields) < cov_start + n_cov:
                continue
            iid = fields[iid_col]
            vals = []
            bad = False
            for token in fields[cov_start : cov_start + n_cov]:
                if is_missing_token(token):
                    bad = True
                    break
                vals.append(float(token))
            if not bad:
                cov_by_iid[iid] = np.asarray(vals, dtype=np.float64)
    missing = sum(1 for iid in fam_iids if iid not in cov_by_iid)
    return cov_by_iid, missing


def align_samples_with_optional_cov(
    fam_iids: list[str],
    sample_idx: np.ndarray,
    y_vec: np.ndarray,
    cov_by_iid: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if cov_by_iid is None:
        return sample_idx, y_vec, None
    keep_idx: list[int] = []
    keep_y: list[float] = []
    keep_cov: list[np.ndarray] = []
    for raw_i, y in zip(sample_idx.tolist(), y_vec.tolist()):
        iid = fam_iids[raw_i]
        cov = cov_by_iid.get(iid)
        if cov is None:
            continue
        keep_idx.append(raw_i)
        keep_y.append(y)
        keep_cov.append(cov)
    if not keep_idx:
        raise RuntimeError("no samples remained after covariate alignment")
    x_cov = np.vstack(keep_cov).astype(np.float64, copy=False)
    return (
        np.asarray(keep_idx, dtype=np.int64),
        np.asarray(keep_y, dtype=np.float64),
        x_cov,
    )


def residualize_y_ols(y_vec: np.ndarray, x_cov: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    n = int(y_vec.shape[0])
    if x_cov is None:
        x = np.ones((n, 1), dtype=np.float64)
    else:
        if x_cov.ndim != 2 or x_cov.shape[0] != n:
            raise RuntimeError(f"x_cov shape mismatch: got {x_cov.shape}, expected ({n}, p)")
        x = np.column_stack([np.ones(n, dtype=np.float64), x_cov])
    xtx = x.T @ x
    xtx_inv = np.linalg.pinv(xtx, hermitian=True)
    beta = xtx_inv @ (x.T @ y_vec)
    y_adj = y_vec - x @ beta
    return y_adj.astype(np.float64, copy=False), x


def load_extract_abs_indices(bim_path: Path, extract_path: Path) -> tuple[np.ndarray, set[str]]:
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
    return np.asarray(out, dtype=np.int64), wanted


def load_selected_bim_rows(
    bim_path: Path,
    wanted_abs_idx: np.ndarray,
) -> dict[int, tuple[str, int, str, str, str]]:
    wanted = set(int(x) for x in wanted_abs_idx.tolist())
    out: dict[int, tuple[str, int, str, str, str]] = {}
    with bim_path.open("r", encoding="utf-8") as fh:
        for row_idx, line in enumerate(fh):
            if row_idx not in wanted:
                continue
            fields = line.split()
            if len(fields) < 6:
                continue
            out[row_idx] = (
                fields[0],
                int(fields[3]),
                fields[1],
                fields[4],
                fields[5],
            )
            if len(out) == len(wanted):
                break
    return out


def estimate_gamma_vinv_over_ss(
    *,
    packed: np.ndarray,
    n_samples_full: int,
    row_indices: np.ndarray,
    row_flip: np.ndarray,
    row_maf: np.ndarray,
    sample_idx: np.ndarray,
    v_chol: tuple[np.ndarray, bool],
    seed: int,
    n_markers: int,
) -> dict[str, float]:
    if row_indices.size == 0:
        raise RuntimeError("no QC SNPs available for gamma estimation")
    use_n = min(int(n_markers), int(row_indices.shape[0]))
    chosen = np.sort(
        np.random.default_rng(int(seed)).choice(
            row_indices.shape[0],
            size=use_n,
            replace=False,
        )
    )
    rows_local = np.ascontiguousarray(row_indices[chosen], dtype=np.int64)
    maf_local = np.ascontiguousarray(row_maf[chosen], dtype=np.float32)
    flip_local = np.ascontiguousarray(row_flip[chosen], dtype=np.bool_)
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
    vinv_g = cho_solve(v_chol, g.T, check_finite=False).T
    s_vinv_s = np.einsum("ij,ij->i", g, vinv_g)
    valid = np.isfinite(s_sq) & np.isfinite(s_vinv_s) & (s_sq > 0.0) & (s_vinv_s > 0.0)
    if not np.any(valid):
        raise RuntimeError("no valid SNPs remained for vinv_over_ss gamma estimation")
    ratio = s_vinv_s[valid] / s_sq[valid]
    return {
        "gamma_mean": float(np.mean(ratio)),
        "gamma_median": float(np.median(ratio)),
        "n_requested": int(use_n),
        "n_used": int(valid.sum()),
    }


def compute_eval_scan(
    *,
    packed: np.ndarray,
    n_samples_full: int,
    eval_row_indices: np.ndarray,
    eval_row_flip: np.ndarray,
    eval_row_maf: np.ndarray,
    sample_idx: np.ndarray,
    v_inv_y: np.ndarray,
    gamma: float,
    block_rows: int,
    bim_info: dict[int, tuple[str, int, str, str, str]],
) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    block_use = max(1, int(block_rows))
    n_eval = int(eval_row_indices.shape[0])
    sample_idx_c = np.ascontiguousarray(sample_idx, dtype=np.int64)
    for start in range(0, n_eval, block_use):
        end = min(start + block_use, n_eval)
        rows_local = np.ascontiguousarray(eval_row_indices[start:end], dtype=np.int64)
        maf_local = np.ascontiguousarray(eval_row_maf[start:end], dtype=np.float32)
        flip_local = np.ascontiguousarray(eval_row_flip[start:end], dtype=np.bool_)
        g = np.ascontiguousarray(
            np.asarray(
                bed_packed_decode_rows_f32(
                    packed,
                    int(n_samples_full),
                    rows_local,
                    flip_local,
                    maf_local,
                    sample_idx_c,
                ),
                dtype=np.float64,
            ),
            dtype=np.float64,
        )
        s_sq = np.einsum("ij,ij->i", g, g)
        num = g @ v_inv_y
        denom = gamma * s_sq
        chisq = np.full(rows_local.shape[0], np.nan, dtype=np.float64)
        beta = np.full(rows_local.shape[0], np.nan, dtype=np.float64)
        se = np.full(rows_local.shape[0], np.nan, dtype=np.float64)
        good = np.isfinite(denom) & (denom > 0.0)
        beta[good] = num[good] / denom[good]
        se[good] = 1.0 / np.sqrt(denom[good])
        chisq[good] = (num[good] * num[good]) / denom[good]
        pwald = np.ones(rows_local.shape[0], dtype=np.float64)
        pwald[good] = erfc(np.sqrt(chisq[good] * 0.5))
        for local_i, abs_row in enumerate(rows_local.tolist()):
            chrom, pos, snp, a0, a1 = bim_info[int(abs_row)]
            out.append(
                {
                    "chrom": chrom,
                    "pos": pos,
                    "snp": snp,
                    "allele0": a0,
                    "allele1": a1,
                    "maf": float(maf_local[local_i]),
                    "beta": float(beta[local_i]),
                    "se": float(se[local_i]),
                    "chisq": float(chisq[local_i]),
                    "pwald": float(pwald[local_i]),
                }
            )
    return out


def write_scan_tsv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("chrom\tpos\tsnp\tallele0\tallele1\tmaf\tbeta\tse\tchisq\tpwald\n")
        for row in rows:
            fh.write(
                f"{row['chrom']}\t{row['pos']}\t{row['snp']}\t{row['allele0']}\t{row['allele1']}\t"
                f"{row['maf']:.7g}\t{row['beta']:.10g}\t{row['se']:.10g}\t{row['chisq']:.10g}\t{row['pwald']:.10g}\n"
            )


def load_fastgwa_subset(path: Path, wanted_snps: set[str]) -> dict[str, dict[str, float | int | str]]:
    out: dict[str, dict[str, float | int | str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue
            snp = fields[1]
            if snp not in wanted_snps:
                continue
            beta = float(fields[7])
            se = float(fields[8])
            out[snp] = {
                "beta": beta,
                "se": se,
                "chisq": (beta / se) ** 2 if se > 0.0 else float("nan"),
                "pwald": float(fields[9]),
            }
    return out


def load_jx_subset(path: Path, wanted_snps: set[str]) -> dict[str, dict[str, float | int | str]]:
    out: dict[str, dict[str, float | int | str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11:
                continue
            snp = fields[2]
            if snp not in wanted_snps:
                continue
            out[snp] = {
                "beta": float(fields[7]),
                "se": float(fields[8]),
                "chisq": float(fields[9]),
                "pwald": float(fields[10]),
            }
    return out


def summarize_vs_reference(
    name: str,
    test_rows: dict[str, dict[str, float | int | str]],
    ref_rows: dict[str, dict[str, float | int | str]],
    ordered_snps: list[str],
) -> dict[str, float | int | list[float] | str]:
    beta_ratio: list[float] = []
    se_ratio: list[float] = []
    chisq_ratio: list[float] = []
    log10_p_diff: list[float] = []
    test_beta: list[float] = []
    ref_beta: list[float] = []
    test_z: list[float] = []
    ref_z: list[float] = []
    test_chi: list[float] = []
    ref_chi: list[float] = []
    p_lt = 0
    matched = 0
    for snp in ordered_snps:
        t = test_rows.get(snp)
        r = ref_rows.get(snp)
        if t is None or r is None:
            continue
        tb = float(t["beta"])
        rb = float(r["beta"])
        ts = float(t["se"])
        rs = float(r["se"])
        tc = float(t["chisq"])
        rc = float(r["chisq"])
        tp = max(float(t["pwald"]), 1e-300)
        rp = max(float(r["pwald"]), 1e-300)
        if rb != 0.0:
            beta_ratio.append(tb / rb)
        if rs > 0.0:
            se_ratio.append(ts / rs)
        if rc > 0.0:
            chisq_ratio.append(tc / rc)
        log10_p_diff.append(math.log10(tp) - math.log10(rp))
        test_beta.append(tb)
        ref_beta.append(rb)
        if ts > 0.0 and rs > 0.0:
            test_z.append(tb / ts)
            ref_z.append(rb / rs)
        test_chi.append(tc)
        ref_chi.append(rc)
        p_lt += int(tp < rp)
        matched += 1
    if matched == 0:
        raise RuntimeError(f"no matched SNPs when comparing {name}")
    test_beta_arr = np.asarray(test_beta, dtype=np.float64)
    ref_beta_arr = np.asarray(ref_beta, dtype=np.float64)
    test_z_arr = np.asarray(test_z, dtype=np.float64)
    ref_z_arr = np.asarray(ref_z, dtype=np.float64)
    test_chi_arr = np.asarray(test_chi, dtype=np.float64)
    ref_chi_arr = np.asarray(ref_chi, dtype=np.float64)
    return {
        "name": name,
        "matched": matched,
        "lambda_gc_test": float(np.median(test_chi_arr) / MED_CHI1),
        "lambda_gc_ref": float(np.median(ref_chi_arr) / MED_CHI1),
        "median_beta_ratio_test_over_ref": float(np.median(np.asarray(beta_ratio, dtype=np.float64))),
        "median_se_ratio_test_over_ref": float(np.median(np.asarray(se_ratio, dtype=np.float64))),
        "median_chisq_ratio_test_over_ref": float(np.median(np.asarray(chisq_ratio, dtype=np.float64))),
        "median_log10_p_diff_test_minus_ref": float(np.median(np.asarray(log10_p_diff, dtype=np.float64))),
        "frac_p_test_lt_ref": float(p_lt / matched),
        "pearson_beta": float(np.corrcoef(test_beta_arr, ref_beta_arr)[0, 1]),
        "pearson_z": float(np.corrcoef(test_z_arr, ref_z_arr)[0, 1]),
    }


def main() -> int:
    args = parse_args()
    fam_iids = load_fam_iids(Path(f"{args.bfile}.fam"))
    if args.mpheno is not None:
        sample_idx, y_vec = load_trait_from_gcta_table(args.pheno, fam_iids, int(args.mpheno))
    else:
        sample_idx, y_vec = load_trait_from_named_table(args.pheno, fam_iids, str(args.trait))
    cov_by_iid = None
    if args.cov_tsv is not None:
        cov_by_iid, _ = load_covariates(args.cov_tsv, fam_iids)
    sample_idx, y_vec, x_cov = align_samples_with_optional_cov(fam_iids, sample_idx, y_vec, cov_by_iid)
    y_adj, x_design = residualize_y_ols(y_vec, x_cov)

    null_fit = w._splmm_sparse_null_fit(
        jxgrm_path=str(args.spgrm),
        sample_idx=sample_idx,
        y_vec=y_adj,
        x_cov=None,
        progress_callback=None,
    )
    sigma_g2 = float(null_fit["sigma_g2"])
    sigma_e2 = float(null_fit["sigma_e2"])

    meta = w._splmm_bed_logic_meta_selected(
        str(args.bfile),
        sample_indices=sample_idx,
        maf_threshold=float(args.maf),
        max_missing_rate=float(args.geno),
        het_threshold=0.0,
        snps_only=False,
    )
    row_idx_all = np.asarray(meta["row_indices"], dtype=np.int64)
    row_maf_all = np.ascontiguousarray(np.asarray(meta["maf"], dtype=np.float32), dtype=np.float32)
    row_flip_all = np.ascontiguousarray(np.asarray(meta["row_flip"], dtype=np.bool_), dtype=np.bool_)

    extract_abs_idx, _ = load_extract_abs_indices(Path(f"{args.bfile}.bim"), args.extract)
    eval_mask = np.isin(row_idx_all, extract_abs_idx)
    eval_row_idx = np.ascontiguousarray(row_idx_all[eval_mask], dtype=np.int64)
    eval_row_maf = np.ascontiguousarray(row_maf_all[eval_mask], dtype=np.float32)
    eval_row_flip = np.ascontiguousarray(row_flip_all[eval_mask], dtype=np.bool_)
    if eval_row_idx.size == 0:
        raise RuntimeError("no evaluation SNPs remained after QC/extract intersection")
    bim_info = load_selected_bim_rows(Path(f"{args.bfile}.bim"), eval_row_idx)

    packed_raw, _miss_raw, _maf_raw, _std_raw, n_samples_full = load_bed_2bit_packed(str(args.bfile))
    packed = np.ascontiguousarray(np.asarray(packed_raw, dtype=np.uint8))
    k_dense = w._splmm_load_sparse_grm_subset_dense(str(args.spgrm), sample_idx)
    n = int(sample_idx.shape[0])
    v = sigma_g2 * k_dense + sigma_e2 * np.eye(n, dtype=np.float64)
    v_chol = cho_factor(v, lower=True, check_finite=False)
    v_inv_y = cho_solve(v_chol, y_adj, check_finite=False)

    gamma_info = estimate_gamma_vinv_over_ss(
        packed=packed,
        n_samples_full=int(n_samples_full),
        row_indices=row_idx_all,
        row_flip=row_flip_all,
        row_maf=row_maf_all,
        sample_idx=sample_idx,
        v_chol=v_chol,
        seed=int(args.gamma_seed),
        n_markers=int(args.gamma_markers),
    )
    gamma = float(gamma_info["gamma_mean"])
    scan_rows = compute_eval_scan(
        packed=packed,
        n_samples_full=int(n_samples_full),
        eval_row_indices=eval_row_idx,
        eval_row_flip=eval_row_flip,
        eval_row_maf=eval_row_maf,
        sample_idx=sample_idx,
        v_inv_y=v_inv_y,
        gamma=gamma,
        block_rows=int(args.scan_block),
        bim_info=bim_info,
    )
    if args.out_tsv is not None:
        write_scan_tsv(args.out_tsv, scan_rows)

    ordered_snps = [str(row["snp"]) for row in scan_rows]
    scan_map = {str(row["snp"]): row for row in scan_rows}
    fastgwa_map = load_fastgwa_subset(args.fastgwa, set(ordered_snps))
    compare = {
        "residualized_vinv_over_ss_vs_fastgwa": summarize_vs_reference(
            "residualized_vinv_over_ss_vs_fastgwa",
            scan_map,
            fastgwa_map,
            ordered_snps,
        )
    }
    if args.jx_tsv is not None and args.jx_tsv.exists():
        jx_map = load_jx_subset(args.jx_tsv, set(ordered_snps))
        compare["residualized_vinv_over_ss_vs_current_jx"] = summarize_vs_reference(
            "residualized_vinv_over_ss_vs_current_jx",
            scan_map,
            jx_map,
            ordered_snps,
        )
        compare["current_jx_vs_fastgwa"] = summarize_vs_reference(
            "current_jx_vs_fastgwa",
            jx_map,
            fastgwa_map,
            ordered_snps,
        )

    out = {
        "method": {
            "y_adjustment": "OLS residualization: y_adj = M_X y",
            "null_model": "Sparse GRM null fit on y_adj with x_cov=None",
            "gamma": "mean(g'V^-1 g / g'g)",
            "beta": "g'V^-1 y_adj / (gamma g'g)",
            "se": "1 / sqrt(gamma g'g)",
        },
        "inputs": {
            "bfile": str(args.bfile),
            "spgrm": str(args.spgrm),
            "pheno": str(args.pheno),
            "trait": None if args.mpheno is not None else str(args.trait),
            "mpheno": args.mpheno,
            "cov_tsv": None if args.cov_tsv is None else str(args.cov_tsv),
            "maf": float(args.maf),
            "geno": float(args.geno),
            "extract": str(args.extract),
            "fastgwa": str(args.fastgwa),
            "jx_tsv": None if args.jx_tsv is None else str(args.jx_tsv),
        },
        "sample_info": {
            "n_samples": int(sample_idx.shape[0]),
            "n_design_cols": int(x_design.shape[1]),
            "n_eval_snps": int(len(scan_rows)),
            "y_adj_mean": float(np.mean(y_adj)),
            "y_adj_var": float(np.var(y_adj, ddof=1)),
        },
        "null_fit": {
            "sigma_g2": sigma_g2,
            "sigma_e2": sigma_e2,
        },
        "gamma": {
            "fastgwa_log_gamma": parse_fastgwa_gamma(args.fastgwa_log),
            **gamma_info,
        },
        "comparisons": compare,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
