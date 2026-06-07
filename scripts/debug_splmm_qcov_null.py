#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from janusx.assoc.workflow import prepare_streaming_context
from janusx.assoc.workflow_model_packed import (
    _ensure_splmm_sparse_grm,
    _jxlmm_sparse_null_fit,
)
from janusx.script._common.genoio import read_id_file
from janusx.script._common.log import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight SparseLMM null-fit debugger for qcov/PC effects."
    )
    parser.add_argument("-bfile", required=True, help="PLINK BED prefix.")
    parser.add_argument("-p", "--pheno", required=True, help="Phenotype file.")
    parser.add_argument(
        "-n",
        "--pheno-col",
        type=int,
        required=True,
        help="0-based phenotype column index, consistent with JanusX CLI.",
    )
    parser.add_argument(
        "--q-list",
        default="0,10",
        help="Comma-separated PC counts to compare, e.g. 0,10.",
    )
    parser.add_argument("--maf", type=float, default=0.001)
    parser.add_argument("--geno", type=float, default=0.1)
    parser.add_argument("--het", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--cutoff", type=float, default=0.05)
    parser.add_argument(
        "--log",
        default="test/splmm_qcov_debug/null_debug.log",
        help="Debug log path.",
    )
    return parser.parse_args()


def _design_stats(y_vec: np.ndarray, x_cov: Optional[np.ndarray]) -> dict[str, float | int]:
    n = int(y_vec.shape[0])
    if x_cov is None:
        return {
            "n": n,
            "p_design": 1,
            "rank_xtx": 1,
            "cond_xtx": 1.0,
            "resid_var_ols": float((y_vec @ y_vec) / max(1, n - 1)),
            "max_abs_intercept_dot_cov": 0.0,
        }
    x_design = np.ascontiguousarray(
        np.concatenate([np.ones((n, 1), dtype=np.float64), x_cov], axis=1),
        dtype=np.float64,
    )
    xtx = np.ascontiguousarray(x_design.T @ x_design, dtype=np.float64)
    evals = np.linalg.eigvalsh(xtx)
    rank = int(np.linalg.matrix_rank(xtx))
    beta = np.linalg.solve(xtx, x_design.T @ y_vec)
    resid = np.ascontiguousarray(y_vec - (x_design @ beta), dtype=np.float64)
    intercept_dot = np.abs(np.sum(x_cov, axis=0)) if x_cov.shape[1] > 0 else np.zeros(0, dtype=np.float64)
    return {
        "n": n,
        "p_design": int(x_design.shape[1]),
        "rank_xtx": rank,
        "cond_xtx": float(evals[-1] / max(evals[0], 1e-18)),
        "resid_var_ols": float((resid @ resid) / max(1, n - int(x_design.shape[1]))),
        "max_abs_intercept_dot_cov": float(np.max(intercept_dot)) if intercept_dot.size > 0 else 0.0,
    }


def main() -> int:
    args = parse_args()
    logger = setup_logging(str(args.log))
    qdims = [int(tok.strip()) for tok in str(args.q_list).split(",") if tok.strip()]
    results: dict[str, dict[str, object]] = {}

    for qdim in qdims:
        dense_grm_path_box: dict[str, Optional[str]] = {"path": None}

        def _capture_grm(_stream_genofile_ready: str, loaded_dense_grm_path: Optional[str]) -> None:
            dense_grm_path_box["path"] = str(loaded_dense_grm_path) if loaded_dense_grm_path else None

        pheno, ids, _n_snps, _grm, qmatrix, _cov_all, _eff_m, stream_genofile, _preloaded = prepare_streaming_context(
            genofile=str(args.bfile),
            phenofile=str(args.pheno),
            pheno_cols=[int(args.pheno_col)],
            maf_threshold=float(args.maf),
            max_missing_rate=float(args.geno),
            genetic_model="add",
            het_threshold=float(args.het),
            chunk_size=int(args.chunk_size),
            mgrm="1",
            pcdim=str(qdim),
            cov_inputs=None,
            threads=int(args.threads),
            mmap_limit=False,
            require_kinship=True,
            logger=logger,
            use_spinner=False,
            snps_only=False,
            allow_packed_grm=False,
            preload_packed_context=False,
            post_grm_hook=_capture_grm,
        )

        dense_grm_path = dense_grm_path_box["path"]
        sparse_out_prefix = (
            str(dense_grm_path)[: -len(".npy")]
            if dense_grm_path is not None and str(dense_grm_path).lower().endswith(".npy")
            else str(stream_genofile)
        )
        sparse_path = _ensure_splmm_sparse_grm(
            str(stream_genofile),
            sample_indices=None,
            out_prefix=str(sparse_out_prefix),
            dense_grm_path=dense_grm_path,
            cutoff=float(args.cutoff),
            maf_threshold=float(args.maf),
            max_missing_rate=float(args.geno),
            het_threshold=float(args.het),
            snps_only=False,
            threads=int(args.threads),
            logger=logger,
            use_spinner=False,
        )

        trait = str(pheno.columns[0])
        full_ids = read_id_file(
            f"{stream_genofile}.fam",
            logger,
            "genotype",
            use_spinner=False,
            show_status=False,
        )
        if full_ids is None:
            raise RuntimeError(f"Failed to read sample IDs from {stream_genofile}.fam")
        full_index = {str(sid): i for i, sid in enumerate(np.asarray(full_ids, dtype=str))}
        sample_map = np.asarray([full_index[str(sid)] for sid in np.asarray(ids, dtype=str)], dtype=np.int64)
        y_full = pheno[trait].to_numpy(dtype=np.float64, copy=False)
        keep_idx = np.flatnonzero(np.isfinite(y_full)).astype(np.int64, copy=False)
        sample_idx = np.ascontiguousarray(sample_map[keep_idx], dtype=np.int64)
        y_vec = np.ascontiguousarray(y_full[keep_idx], dtype=np.float64)
        x_cov = np.ascontiguousarray(qmatrix[keep_idx], dtype=np.float64)
        x_arg = x_cov if int(x_cov.shape[1]) > 0 else None

        fit = _jxlmm_sparse_null_fit(
            jxgrm_path=str(sparse_path),
            sample_idx=sample_idx,
            y_vec=y_vec,
            x_cov=x_arg,
            progress_callback=None,
        )
        diag = _design_stats(y_vec, x_arg)
        out = {
            "trait": trait,
            "q_shape": list(map(int, qmatrix.shape)),
            "sparse_path": str(sparse_path),
            **diag,
        }
        for key in (
            "strategy",
            "backend",
            "lambda",
            "sigma_g2",
            "sigma_e2",
            "pve",
            "rss_reml",
            "rss_mx_reml",
            "df_reml",
            "resid_var_reml",
            "resid_var_mx_reml",
            "ypy_reml",
            "sigma_sum_profile",
            "sigma_scale_reml",
            "mean_diag_k",
            "offdiag_density_k",
            "lambda_boundary",
        ):
            out[key] = fit.get(key)
        results[str(qdim)] = out

    print(json.dumps(results, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
