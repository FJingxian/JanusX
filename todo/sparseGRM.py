from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from janusx.gfreader import inspect_genotype_file, load_bed_2bit_packed, load_genotype_chunks
from janusx.gfreader.gfreader import (
    _ensure_plink_cache,
    _ensure_plink_cache_for_cli_source,
    _resolve_input,
)
from janusx.janusx import bed_packed_decode_rows_f32, bed_packed_row_flip_mask


@dataclass
class SparseGrmStats:
    backend: str
    n_samples: int
    n_snps_input: int
    eff_m: int
    scale: float
    nnz: int
    density: float


def _normalize_bed_prefix(genofile: str) -> str:
    path = str(genofile)
    lower = path.lower()
    if lower.endswith((".bed", ".bim", ".fam")):
        return path[:-4]
    return path


def _normalize_chunk(
    geno_chunk: np.ndarray,
    *,
    method: int,
    eps: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """
    Convert a SNP-major genotype chunk into a centered/standardized chunk.

    Parameters
    ----------
    geno_chunk:
        Array with shape (m_chunk, n_samples), coded as 0/1/2 and already
        filtered/imputed by janusx.gfreader.
    method:
        1 = centered VanRaden-style GRM
        2 = standardized GRM
    eps:
        Numerical floor for near-monomorphic SNPs.

    Returns
    -------
    z:
        Normalized chunk with the same shape as ``geno_chunk``.
    scale_add:
        Chunk contribution to the final GRM denominator.
        - method 1: sum_j 2 p_j (1 - p_j)
        - method 2: number of retained SNPs in the chunk
    """
    geno = np.asarray(geno_chunk, dtype=np.float32, order="C")
    p = geno.mean(axis=1, dtype=np.float32, keepdims=True) * np.float32(0.5)
    z = geno - np.float32(2.0) * p

    if method == 1:
        scale_add = float(
            np.sum(
                np.float32(2.0) * p * (np.float32(1.0) - p),
                dtype=np.float64,
            )
        )
        return z, scale_add

    if method == 2:
        denom = np.float32(2.0) * p * (np.float32(1.0) - p)
        np.maximum(denom, np.float32(max(eps, 1e-12)), out=denom)
        np.sqrt(denom, out=denom)
        z /= denom
        return z, float(geno.shape[0])

    raise ValueError(f"Unsupported GRM method: {method}")


def _iter_normalized_chunks(
    genofile: str,
    *,
    chunk_size: int,
    maf: float,
    missing_rate: float,
    method: int,
    eps: float,
):
    for geno_chunk, _sites in load_genotype_chunks(
        genofile,
        chunk_size=chunk_size,
        maf=maf,
        missing_rate=missing_rate,
        impute=True,
    ):
        z, scale_add = _normalize_chunk(geno_chunk, method=method, eps=eps)
        yield z, int(z.shape[0]), scale_add


def _resolve_packed_bed_prefix(genofile: str) -> str | None:
    """
    Resolve a genotype input to a PLINK BED prefix suitable for packed preload.

    Returns ``None`` when the source cannot be represented as a packed BED path
    via the currently available JanusX cache helpers.
    """
    kind, prefix, src_path = _resolve_input(genofile)
    if kind == "plink":
        return _normalize_bed_prefix(prefix)
    if src_path is None:
        return None
    if kind == "vcf":
        return _ensure_plink_cache(
            prefix,
            src_path,
            snps_only=True,
            maf=0.0,
            missing_rate=1.0,
        )
    if kind == "hmp":
        return _ensure_plink_cache_for_cli_source(
            prefix,
            src_path,
            source_label="HMP",
            snps_only=True,
        )
    if kind == "txt":
        return _ensure_plink_cache_for_cli_source(
            prefix,
            src_path,
            source_label="TXT",
            snps_only=True,
        )
    return None


def _packed_bed_payload_bytes(prefix: str) -> int:
    bed_path = f"{prefix}.bed"
    if not os.path.isfile(bed_path):
        raise FileNotFoundError(f"BED file not found: {bed_path}")
    return max(0, int(os.path.getsize(bed_path)) - 3)


def _load_packed_bed_context(
    genofile: str,
    *,
    maf: float,
    missing_rate: float,
) -> dict[str, Any]:
    """
    Preload a BED payload once, then filter SNP rows in-memory.

    The packed matrix remains 2-bit encoded; genotype rows are decoded only for
    the SNP chunk and sample block currently being processed.
    """
    prefix = _resolve_packed_bed_prefix(genofile)
    if prefix is None:
        raise ValueError(
            f"Packed BED preload is not available for input: {genofile}"
        )

    packed_raw, miss_raw, maf_raw, std_raw, n_samples = load_bed_2bit_packed(prefix)
    packed = np.ascontiguousarray(np.asarray(packed_raw, dtype=np.uint8))
    miss = np.ascontiguousarray(np.asarray(miss_raw, dtype=np.float32).reshape(-1))
    maf_arr = np.ascontiguousarray(np.asarray(maf_raw, dtype=np.float32).reshape(-1))
    std = np.ascontiguousarray(np.asarray(std_raw, dtype=np.float32).reshape(-1))
    if packed.ndim != 2:
        raise ValueError("Packed BED payload must be 2D.")
    if miss.shape[0] != packed.shape[0] or maf_arr.shape[0] != packed.shape[0]:
        raise ValueError("Packed BED row statistics do not match the packed row count.")

    keep = np.ones(packed.shape[0], dtype=np.bool_)
    maf_thr = float(maf)
    if maf_thr > 0.0:
        keep &= maf_arr >= maf_thr
    miss_thr = float(missing_rate)
    if miss_thr < 1.0:
        keep &= miss <= miss_thr
    if not np.any(keep):
        raise ValueError("No SNP remains after packed BED filtering.")

    if not np.all(keep):
        packed = np.ascontiguousarray(packed[keep], dtype=np.uint8)
        miss = np.ascontiguousarray(miss[keep], dtype=np.float32)
        maf_arr = np.ascontiguousarray(maf_arr[keep], dtype=np.float32)
        std = np.ascontiguousarray(std[keep], dtype=np.float32)

    row_flip = np.ascontiguousarray(
        np.asarray(
            bed_packed_row_flip_mask(packed, int(n_samples)),
            dtype=np.bool_,
        ).reshape(-1)
    )
    if row_flip.shape[0] != packed.shape[0]:
        raise ValueError("Packed BED row_flip length mismatch.")

    return {
        "prefix": prefix,
        "packed": packed,
        "missing_rate": miss,
        "maf": maf_arr,
        "std_denom": std,
        "row_flip": row_flip,
        "n_samples": int(n_samples),
        "payload_bytes": _packed_bed_payload_bytes(prefix),
    }


def _decode_packed_rows(
    packed_ctx: dict[str, Any],
    row_indices: np.ndarray,
    sample_indices: np.ndarray,
) -> np.ndarray:
    rows = np.ascontiguousarray(np.asarray(row_indices, dtype=np.int64).reshape(-1))
    sidx = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1))
    if rows.size == 0:
        return np.empty((0, int(sidx.size)), dtype=np.float32)
    decoded = bed_packed_decode_rows_f32(
        packed_ctx["packed"],
        int(packed_ctx["n_samples"]),
        rows,
        packed_ctx["row_flip"],
        packed_ctx["maf"],
        sidx,
    )
    return np.ascontiguousarray(np.asarray(decoded, dtype=np.float32), dtype=np.float32)


def _normalize_decoded_chunk(
    decoded_chunk: np.ndarray,
    *,
    maf_chunk: np.ndarray,
    std_chunk: np.ndarray,
    method: int,
    eps: float,
) -> np.ndarray:
    z = np.asarray(decoded_chunk, dtype=np.float32, order="C")
    z -= np.float32(2.0) * maf_chunk[:, None]
    if method == 2:
        denom = np.maximum(std_chunk, np.float32(max(eps, 1e-12)))
        z /= denom[:, None]
    return z


def _scan_grm_scale(
    genofile: str,
    *,
    chunk_size: int,
    maf: float,
    missing_rate: float,
    method: int,
    eps: float,
) -> tuple[float, int]:
    """
    First pass: compute the global denominator for the GRM.

    This pass is cheap in memory and avoids keeping any n x n state.
    """
    scale = 0.0
    eff_m = 0
    for _z, m_chunk, scale_add in _iter_normalized_chunks(
        genofile,
        chunk_size=chunk_size,
        maf=maf,
        missing_rate=missing_rate,
        method=method,
        eps=eps,
    ):
        scale += scale_add
        eff_m += m_chunk

    if eff_m <= 0:
        raise ValueError("No effective SNPs remained after filtering.")
    if scale <= 0.0:
        raise ValueError("Invalid GRM denominator; check the genotype input.")
    return float(scale), int(eff_m)


def _scan_grm_scale_packed(
    packed_ctx: dict[str, Any],
    *,
    method: int,
) -> tuple[float, int]:
    maf = np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1)
    eff_m = int(maf.shape[0])
    if eff_m <= 0:
        raise ValueError("No effective SNPs remained after packed BED filtering.")
    if method == 1:
        scale = float(
            np.sum(
                np.float32(2.0) * maf * (np.float32(1.0) - maf),
                dtype=np.float64,
            )
        )
    elif method == 2:
        scale = float(eff_m)
    else:
        raise ValueError(f"Unsupported GRM method: {method}")

    if scale <= 0.0:
        raise ValueError("Invalid GRM denominator from packed BED preload.")
    return scale, eff_m


def build_sparse_grm_blockwise(
    genofile: str,
    *,
    block_size: int = 256,
    chunk_size: int = 1024,
    threshold: float = 0.05,
    method: int = 1,
    maf: float = 0.02,
    missing_rate: float = 0.05,
    eps: float = 1e-8,
    absolute_threshold: bool = False,
    keep_diag: bool = True,
    backend: str = "auto",
    preload_limit_mb: int = 2048,
    verbose: bool = True,
) -> tuple[sp.csr_matrix, list[str], SparseGrmStats]:
    """
    Build an exact sparse GRM without materializing the full dense n x n matrix.

    Strategy
    --------
    1. First pass: stream genotype chunks and compute the global GRM denominator.
    2. Second stage: iterate over sample block pairs (I, J).
       For each block pair, rescan genotype chunks and accumulate only
       ``Z[:, I].T @ Z[:, J]`` into a local dense buffer of size ``B x B``.
    3. Threshold the local block and emit only surviving entries into COO triplets.

    Memory
    ------
    Peak memory is roughly:
      - one genotype chunk: O(chunk_size * n_samples)
      - one local accumulator: O(block_size^2)
      - final sparse output: O(nnz)

    Tradeoff
    --------
    This is exact and memory-friendly, but slower than the dense GRM because
    the genotype source is scanned once per sample-block pair.

    Backend
    -------
    - ``stream``: always stream genotype chunks from the source/cache.
    - ``packed-bed``: preload PLINK BED 2-bit payload and decode row/sample
      subsets on demand, reducing repeated IO at the cost of higher RAM.
    - ``auto``: use packed-bed only when the source can be resolved to BED and
      the BED payload size is within ``preload_limit_mb``.
    """
    sample_ids, n_snps_input = inspect_genotype_file(
        genofile,
        maf=maf,
        missing_rate=missing_rate,
    )
    n_samples = len(sample_ids)
    if n_samples <= 0:
        raise ValueError("No samples were found in the genotype input.")

    block_size = int(max(1, block_size))
    chunk_size = int(max(1, chunk_size))
    threshold = float(threshold)
    backend_key = str(backend).strip().lower()
    if backend_key not in {"auto", "stream", "packed-bed"}:
        raise ValueError("backend must be one of: auto, stream, packed-bed")

    packed_ctx: dict[str, Any] | None = None
    backend_used = "stream"
    if backend_key != "stream":
        prefix = _resolve_packed_bed_prefix(genofile)
        if prefix is not None:
            payload_bytes = _packed_bed_payload_bytes(prefix)
            limit_bytes = int(max(0, preload_limit_mb)) * 1024 * 1024
            should_preload = backend_key == "packed-bed"
            if backend_key == "auto":
                should_preload = (limit_bytes <= 0) or (payload_bytes <= limit_bytes)
            if should_preload:
                if verbose:
                    print(
                        f"[packed-bed] preload prefix={prefix}, "
                        f"payload={payload_bytes / 1024**2:.2f} MB"
                    )
                packed_ctx = _load_packed_bed_context(
                    genofile,
                    maf=maf,
                    missing_rate=missing_rate,
                )
                backend_used = "packed-bed"
            elif verbose and backend_key == "auto":
                print(
                    f"[packed-bed] skip preload: payload={payload_bytes / 1024**2:.2f} MB "
                    f"> limit={preload_limit_mb} MB"
                )
        elif backend_key == "packed-bed":
            raise ValueError(
                f"Cannot resolve packed BED backend for input: {genofile}"
            )

    if verbose:
        print(
            f"[scale pass] samples={n_samples}, input_snps={n_snps_input}, "
            f"chunk_size={chunk_size}, method={method}, backend={backend_used}"
        )
    if packed_ctx is None:
        scale, eff_m = _scan_grm_scale(
            genofile,
            chunk_size=chunk_size,
            maf=maf,
            missing_rate=missing_rate,
            method=method,
            eps=eps,
        )
    else:
        scale, eff_m = _scan_grm_scale_packed(
            packed_ctx,
            method=method,
        )

    n_blocks = (n_samples + block_size - 1) // block_size
    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []

    for bi, i0 in enumerate(range(0, n_samples, block_size)):
        i1 = min(i0 + block_size, n_samples)
        for bj, j0 in enumerate(range(i0, n_samples, block_size), start=bi):
            j1 = min(j0 + block_size, n_samples)
            acc = np.zeros((i1 - i0, j1 - j0), dtype=np.float32)
            sample_i = np.arange(i0, i1, dtype=np.int64)
            sample_j = np.arange(j0, j1, dtype=np.int64)

            if verbose:
                print(
                    f"[block {bi + 1}/{n_blocks}, {bj + 1}/{n_blocks}] "
                    f"samples=({i0}:{i1}, {j0}:{j1})"
                )

            if packed_ctx is None:
                for z, _m_chunk, _scale_add in _iter_normalized_chunks(
                    genofile,
                    chunk_size=chunk_size,
                    maf=maf,
                    missing_rate=missing_rate,
                    method=method,
                    eps=eps,
                ):
                    acc += z[:, i0:i1].T @ z[:, j0:j1]
            else:
                maf_all = np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1)
                std_all = np.asarray(packed_ctx["std_denom"], dtype=np.float32).reshape(-1)
                for r0 in range(0, eff_m, chunk_size):
                    r1 = min(r0 + chunk_size, eff_m)
                    row_idx = np.arange(r0, r1, dtype=np.int64)
                    maf_chunk = maf_all[r0:r1]
                    std_chunk = std_all[r0:r1]
                    decoded_i = _decode_packed_rows(
                        packed_ctx,
                        row_idx,
                        sample_i,
                    )
                    z_i = _normalize_decoded_chunk(
                        decoded_i,
                        maf_chunk=maf_chunk,
                        std_chunk=std_chunk,
                        method=method,
                        eps=eps,
                    )
                    if i0 == j0:
                        z_j = z_i
                    else:
                        decoded_j = _decode_packed_rows(
                            packed_ctx,
                            row_idx,
                            sample_j,
                        )
                        z_j = _normalize_decoded_chunk(
                            decoded_j,
                            maf_chunk=maf_chunk,
                            std_chunk=std_chunk,
                            method=method,
                            eps=eps,
                        )
                    acc += z_i.T @ z_j

            acc /= np.float32(scale)

            if absolute_threshold:
                keep = np.abs(acc) >= threshold
            else:
                keep = acc >= threshold

            if i0 == j0:
                keep = np.triu(keep)
                if keep_diag:
                    diag_idx = np.arange(i1 - i0)
                    keep[diag_idx, diag_idx] = True

            rr, cc = np.nonzero(keep)
            if rr.size == 0:
                continue

            vals = acc[rr, cc].astype(np.float32, copy=False)
            r_global = (rr + i0).astype(np.int32, copy=False)
            c_global = (cc + j0).astype(np.int32, copy=False)

            rows_parts.append(r_global)
            cols_parts.append(c_global)
            data_parts.append(vals)

            if i0 != j0:
                rows_parts.append(c_global)
                cols_parts.append(r_global)
                data_parts.append(vals)

    if rows_parts:
        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
        data = np.concatenate(data_parts)
    else:
        rows = np.empty(0, dtype=np.int32)
        cols = np.empty(0, dtype=np.int32)
        data = np.empty(0, dtype=np.float32)

    sparse_grm = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_samples, n_samples),
        dtype=np.float32,
    ).tocsr()
    sparse_grm.sum_duplicates()

    nnz = int(sparse_grm.nnz)
    density = float(nnz) / float(max(1, n_samples * n_samples))
    stats = SparseGrmStats(
        backend=backend_used,
        n_samples=n_samples,
        n_snps_input=int(n_snps_input),
        eff_m=int(eff_m),
        scale=float(scale),
        nnz=nnz,
        density=density,
    )
    return sparse_grm, sample_ids, stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a sparse genomic relationship matrix without "
            "materializing the full dense n x n matrix."
        )
    )
    parser.add_argument("genofile", help="Genotype input path/prefix understood by janusx.gfreader.")
    parser.add_argument("--block-size", type=int, default=256, help="Sample block size.")
    parser.add_argument("--chunk-size", type=int, default=1024, help="SNP chunk size.")
    parser.add_argument("--threshold", type=float, default=0.05, help="Keep entries >= threshold.")
    parser.add_argument("--absolute-threshold", action="store_true", help="Use abs(K_ij) >= threshold.")
    parser.add_argument("--method", type=int, default=1, choices=[1, 2], help="1=centered, 2=standardized.")
    parser.add_argument("--maf", type=float, default=0.02, help="MAF filter.")
    parser.add_argument("--missing-rate", type=float, default=0.05, help="Max missing rate.")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "stream", "packed-bed"],
        help="Chunk source backend. auto tries packed-bed when BED payload is small enough.",
    )
    parser.add_argument(
        "--preload-limit-mb",
        type=int,
        default=2048,
        help="When backend=auto, only preload packed BED if payload <= this limit (MB).",
    )
    parser.add_argument("--save-prefix", type=str, default=None, help="If set, write .npz and .id outputs.")
    parser.add_argument("--quiet", action="store_true", help="Suppress block progress logs.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    sparse_grm, sample_ids, stats = build_sparse_grm_blockwise(
        args.genofile,
        block_size=args.block_size,
        chunk_size=args.chunk_size,
        threshold=args.threshold,
        absolute_threshold=args.absolute_threshold,
        method=args.method,
        maf=args.maf,
        missing_rate=args.missing_rate,
        backend=args.backend,
        preload_limit_mb=args.preload_limit_mb,
        verbose=not args.quiet,
    )

    print(
        f"[done] backend={stats.backend}, shape={sparse_grm.shape}, nnz={stats.nnz}, "
        f"density={stats.density:.6f}, eff_m={stats.eff_m}"
    )

    if args.save_prefix:
        sp.save_npz(f"{args.save_prefix}.sparse_grm.npz", sparse_grm)
        np.savetxt(
            f"{args.save_prefix}.sparse_grm.id",
            np.asarray(sample_ids, dtype=str),
            fmt="%s",
        )
        print(
            f"[saved] {args.save_prefix}.sparse_grm.npz\n"
            f"[saved] {args.save_prefix}.sparse_grm.id"
        )


if __name__ == "__main__":
    main()
