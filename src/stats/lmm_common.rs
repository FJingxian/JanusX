//! Shared rotation, SIMD dot-product, streaming chunk, and missing-count helpers
//! used by both `fvlmm` (fixed-lambda GWAS) and `lmm` (exact REML/ML GWAS).

use matrixmultiply::sgemm;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::bedmath::packed_row_missing_count_selected;
use crate::blas::{
    cblas_sgemm_dispatch, rust_sgemm_backend_tag, BlasThreadGuard, CblasInt, CBLAS_NO_TRANS,
    CBLAS_ROW_MAJOR, CBLAS_TRANS,
};

pub(crate) fn rotate_snp_block_with_ut(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
) {
    if rows == 0 || n == 0 {
        return;
    }
    // C(rows, n) = A(rows, n) * U(n, n)
    // We store U^T in row-major (`u_t`). For matrixmultiply strides, expose U by
    // setting B row stride = 1 and col stride = n, i.e. B(i,j)=u_t[j*n + i].
    unsafe {
        sgemm(
            rows,
            n,
            n,
            1.0,
            snp_block.as_ptr(),
            n as isize,
            1,
            u_t.as_ptr(),
            1,
            n as isize,
            0.0,
            out_block.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[inline]
pub(crate) fn env_positive_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
}

#[inline]
pub(crate) fn choose_fvlmm_blas_tile_rows(rows: usize, threads: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    if threads <= 1 || rows <= 64 {
        return rows;
    }
    let target_tasks = threads.saturating_mul(2).max(1);
    let tile_rows = rows.div_ceil(target_tasks);
    tile_rows.clamp(64, 2048).min(rows)
}

#[inline]
pub(crate) fn fvlmm_rotate_kernel_prefers_tiled() -> bool {
    matches!(
        std::env::var("JX_FVLMM_ROTATE_KERNEL")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("blas_tiled") | Some("tiled")
    )
}

#[inline]
pub(crate) fn choose_rotate_row_tile_rows_fvlmm(
    rows: usize,
    n: usize,
    thread_hint: usize,
) -> usize {
    if let Some(v) = env_positive_usize("JX_GWAS_ROTATE_ROW_TILE") {
        return v.max(1).min(rows.max(1));
    }
    if n < 2048 {
        return choose_rotate_tile_rows(rows, thread_hint);
    }
    let target_mb = env_positive_usize("JX_GWAS_ROTATE_ROW_TILE_MB").unwrap_or_else(|| {
        if n >= 16384 {
            8usize
        } else if n >= 8192 {
            6usize
        } else {
            4usize
        }
    });
    let target_bytes = target_mb.saturating_mul(1024 * 1024);
    let bytes_per_row = n
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<f32>())
        .max(1);
    let mut tile_rows = (target_bytes / bytes_per_row).max(16).min(rows.max(1));
    let max_tile = if n >= 16384 {
        256usize
    } else if n >= 8192 {
        192usize
    } else {
        128usize
    };
    tile_rows = tile_rows.clamp(16, max_tile.min(rows.max(1)));
    tile_rows.min(rows.max(1))
}

#[inline]
pub(crate) fn choose_rotate_col_block_cols_fvlmm(n: usize) -> usize {
    if let Some(v) = env_positive_usize("JX_GWAS_ROTATE_COL_BLOCK") {
        return v.max(1).min(n.max(1));
    }
    if n < 2048 {
        return n.max(1);
    }
    let target_mb = env_positive_usize("JX_GWAS_ROTATE_COL_BLOCK_MB").unwrap_or_else(|| {
        if n >= 16384 {
            12usize
        } else if n >= 8192 {
            16usize
        } else {
            8usize
        }
    });
    let target_bytes = target_mb.saturating_mul(1024 * 1024);
    let bytes_per_col = n.saturating_mul(std::mem::size_of::<f32>()).max(1);
    let cols = (target_bytes / bytes_per_col).max(64);
    cols.clamp(64, 1024).min(n.max(1))
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rotate_snp_block_with_ut_parallel_blocked(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
    row_tile_rows: usize,
    col_block_cols: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    if rows == 0 || n == 0 {
        return;
    }
    debug_assert!(snp_block.len() >= rows.saturating_mul(n));
    debug_assert!(u_t.len() >= n.saturating_mul(n));
    debug_assert!(out_block.len() >= rows.saturating_mul(n));
    let tile_rows = row_tile_rows.max(1).min(rows);
    let col_block = col_block_cols.max(1).min(n);

    let mut run_parallel = || {
        out_block
            .par_chunks_mut(tile_rows * n)
            .enumerate()
            .for_each_init(
                || vec![0.0_f32; tile_rows.saturating_mul(col_block)],
                |scratch, (chunk_idx, out_chunk)| {
                    let row_start = chunk_idx * tile_rows;
                    let rows_here = out_chunk.len() / n;
                    let a_start = row_start * n;
                    let a_end = a_start + rows_here * n;
                    let a_block = &snp_block[a_start..a_end];
                    if col_block >= n {
                        rotate_snp_block_with_ut(a_block, rows_here, n, u_t, out_chunk);
                        return;
                    }
                    for col_start in (0..n).step_by(col_block) {
                        let cols_here = (n - col_start).min(col_block);
                        let scratch_sub = &mut scratch[..rows_here * cols_here];
                        matmul_rowmajor_rhs_transposed_from_rowmajor_f32(
                            a_block,
                            rows_here,
                            n,
                            &u_t[col_start * n..(col_start + cols_here) * n],
                            cols_here,
                            scratch_sub,
                        );
                        for r in 0..rows_here {
                            let src = &scratch_sub[r * cols_here..(r + 1) * cols_here];
                            let dst =
                                &mut out_chunk[r * n + col_start..r * n + col_start + cols_here];
                            dst.copy_from_slice(src);
                        }
                    }
                },
            );
    };

    if tile_rows >= rows {
        if col_block >= n {
            rotate_snp_block_with_ut(snp_block, rows, n, u_t, out_block);
            return;
        }
        let mut scratch = vec![0.0_f32; rows.saturating_mul(col_block)];
        for col_start in (0..n).step_by(col_block) {
            let cols_here = (n - col_start).min(col_block);
            let scratch_sub = &mut scratch[..rows * cols_here];
            matmul_rowmajor_rhs_transposed_from_rowmajor_f32(
                snp_block,
                rows,
                n,
                &u_t[col_start * n..(col_start + cols_here) * n],
                cols_here,
                scratch_sub,
            );
            for r in 0..rows {
                let src = &scratch_sub[r * cols_here..(r + 1) * cols_here];
                let dst = &mut out_block[r * n + col_start..r * n + col_start + cols_here];
                dst.copy_from_slice(src);
            }
        }
        return;
    }

    if let Some(tp) = pool {
        tp.install(run_parallel);
    } else {
        run_parallel();
    }
}

#[inline]
pub(crate) fn assoc_rotate_prefers_rayon_rowmajor_f32_kernel() -> bool {
    for env_name in ["JX_LMM_ROTATE_F32_KERNEL", "JX_ROWMAJOR_F32_KERNEL"] {
        if let Ok(raw) = std::env::var(env_name) {
            let norm = raw.trim().to_ascii_lowercase();
            match norm.as_str() {
                "rayon" | "parallel" | "custom" => return true,
                "blas" | "gemm" | "serial" => return false,
                _ => {}
            }
        }
    }
    // For LMM/FvLMM association rotate-proj, default to BLAS-backed GEMM
    // across platforms. This path is large dense projection, unlike some
    // HE/PCG row-major kernels that still prefer custom Rayon tiling.
    false
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn rotate_snp_block_with_ut_blas_impl(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
    threads: usize,
    proj_pool: Option<&Arc<rayon::ThreadPool>>,
    allow_fvlmm_tiled: bool,
) {
    if rows == 0 || n == 0 {
        return;
    }
    debug_assert!(snp_block.len() >= rows.saturating_mul(n));
    debug_assert!(u_t.len() >= n.saturating_mul(n));
    debug_assert!(out_block.len() >= rows.saturating_mul(n));

    if allow_fvlmm_tiled && fvlmm_rotate_kernel_prefers_tiled() {
        let tile_rows = choose_fvlmm_blas_tile_rows(rows, threads.max(1));
        let mut run = || {
            out_block
                .par_chunks_mut(tile_rows.saturating_mul(n).max(1))
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let row_start = chunk_idx * tile_rows;
                    let rows_here = out_chunk.len() / n;
                    let a_start = row_start * n;
                    let a_end = a_start + rows_here * n;
                    let a_block = &snp_block[a_start..a_end];
                    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
                    unsafe {
                        cblas_sgemm_dispatch(
                            CBLAS_ROW_MAJOR,
                            CBLAS_NO_TRANS,
                            CBLAS_TRANS,
                            rows_here as CblasInt,
                            n as CblasInt,
                            n as CblasInt,
                            1.0_f32,
                            a_block.as_ptr(),
                            n as CblasInt,
                            u_t.as_ptr(),
                            n as CblasInt,
                            0.0_f32,
                            out_chunk.as_mut_ptr(),
                            n as CblasInt,
                        );
                    }
                    #[cfg(not(any(
                        target_os = "macos",
                        target_os = "linux",
                        target_os = "windows"
                    )))]
                    {
                        rotate_snp_block_with_ut(a_block, rows_here, n, u_t, out_chunk);
                    }
                });
        };
        // Accelerate exposes only a process-wide single/multi-thread switch.
        // Tile-level Rayon parallelism is therefore paired with single-thread
        // BLAS calls to avoid nested oversubscription.
        let _blas_guard = BlasThreadGuard::enter(1);
        if let Some(tp) = proj_pool {
            tp.install(run);
        } else {
            run();
        }
        return;
    }

    if assoc_rotate_prefers_rayon_rowmajor_f32_kernel() {
        let thread_hint = if threads > 0 {
            threads
        } else {
            rayon::current_num_threads()
        };
        let row_tile = choose_rotate_row_tile_rows_fvlmm(rows, n, thread_hint);
        let col_block = choose_rotate_col_block_cols_fvlmm(n);
        rotate_snp_block_with_ut_parallel_blocked(
            snp_block, rows, n, u_t, out_block, row_tile, col_block, proj_pool,
        );
        return;
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        let _blas_guard = BlasThreadGuard::enter(threads.max(1));
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            rows as CblasInt,
            n as CblasInt,
            n as CblasInt,
            1.0_f32,
            snp_block.as_ptr(),
            n as CblasInt,
            u_t.as_ptr(),
            n as CblasInt,
            0.0_f32,
            out_block.as_mut_ptr(),
            n as CblasInt,
        );
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        rotate_snp_block_with_ut(snp_block, rows, n, u_t, out_block);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rotate_snp_block_with_ut_blas(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
    threads: usize,
    proj_pool: Option<&Arc<rayon::ThreadPool>>,
) {
    rotate_snp_block_with_ut_blas_impl(snp_block, rows, n, u_t, out_block, threads, proj_pool, false)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rotate_snp_block_with_ut_blas_fvlmm(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
    threads: usize,
    proj_pool: Option<&Arc<rayon::ThreadPool>>,
) {
    rotate_snp_block_with_ut_blas_impl(snp_block, rows, n, u_t, out_block, threads, proj_pool, true)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn matmul_rowmajor_rhs_transposed_from_rowmajor_f32(
    a: &[f32],
    m: usize,
    k: usize,
    b_rowmajor_nk: &[f32],
    n: usize,
    out: &mut [f32],
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    debug_assert!(a.len() >= m.saturating_mul(k));
    debug_assert!(b_rowmajor_nk.len() >= n.saturating_mul(k));
    debug_assert!(out.len() >= m.saturating_mul(n));
    // C(m, n) = A(m, k) * B^T(k, n)
    // `b_rowmajor_nk` stores B(n, k) row-major. Expose B^T by setting
    // B strides to: row_stride=1, col_stride=k.
    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b_rowmajor_nk.as_ptr(),
            1,
            k as isize,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[inline]
pub(crate) fn choose_rotate_tile_rows(rows: usize, thread_hint: usize) -> usize {
    if rows <= 1 {
        return 1;
    }
    let threads = thread_hint.max(1);
    // Avoid splitting when there is no real parallelism (or chunk is tiny):
    // one large SGEMM is much faster than many tiny calls in this case.
    if threads <= 1 || rows <= 32 {
        return rows;
    }
    // Keep enough independent tiles to feed the rayon pool, but avoid tiles
    // so small that SGEMM launch overhead dominates.
    let target_tasks = threads.saturating_mul(2).max(1);
    let mut tile_rows = (rows + target_tasks - 1) / target_tasks;
    tile_rows = tile_rows.clamp(16, 128);
    tile_rows.min(rows)
}

#[inline]
pub(crate) fn rotate_snp_block_with_ut_parallel(
    snp_block: &[f32],
    rows: usize,
    n: usize,
    u_t: &[f32],
    out_block: &mut [f32],
    tile_rows: usize,
) {
    if rows == 0 || n == 0 {
        return;
    }

    let tile_rows = tile_rows.max(1).min(rows);
    if tile_rows >= rows {
        rotate_snp_block_with_ut(snp_block, rows, n, u_t, out_block);
        return;
    }

    out_block
        .par_chunks_mut(tile_rows * n)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * tile_rows;
            let rows_here = out_chunk.len() / n;
            let a_start = row_start * n;
            let a_end = a_start + rows_here * n;
            rotate_snp_block_with_ut(&snp_block[a_start..a_end], rows_here, n, u_t, out_chunk);
        });
}

#[inline]
pub(crate) fn record_elapsed_nanos(acc: &AtomicU64, t0: Instant) {
    let nanos = t0.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    acc.fetch_add(nanos, Ordering::Relaxed);
}

#[inline]
pub(crate) fn elapsed_nanos_to_secs(acc: &AtomicU64) -> f64 {
    (acc.load(Ordering::Relaxed) as f64) * 1e-9
}

#[inline]
pub(crate) fn normalize_default_assoc_threads_for_backend(
    backend_tag: &str,
    proj_threads: usize,
    assoc_threads: usize,
) -> usize {
    let proj = proj_threads.max(1);
    let assoc = assoc_threads.max(1);
    if backend_tag == "openblas" {
        assoc.min(proj)
    } else {
        assoc
    }
}

#[inline]
pub(crate) fn stage_proj_threads_or(requested_threads: usize) -> usize {
    env_positive_usize("JX_FVLMM_PROJ_THREADS")
        .or_else(|| env_positive_usize("JX_MLM_RUST_THREADS"))
        .unwrap_or(requested_threads)
}

#[inline]
pub(crate) fn stage_assoc_threads_or(requested_threads: usize, proj_threads: usize) -> usize {
    if let Some(explicit) = env_positive_usize("JX_FVLMM_ASSOC_THREADS") {
        return explicit;
    }
    let default_threads = env_positive_usize("JX_MLM_RUST_THREADS").unwrap_or(requested_threads);
    normalize_default_assoc_threads_for_backend(
        rust_sgemm_backend_tag(),
        proj_threads,
        default_threads,
    )
}

/// Per-SNP filter results from parallel counting (Copy-friendly, no heap strings).
pub(crate) struct SnpCounts {
    pub(crate) flip: bool,
    pub(crate) maf: f32,
    pub(crate) miss_rate: f32,
    pub(crate) missing_count: usize,
}

/// Double-buffer for the streaming BED -> TSV pipeline.
pub(crate) struct StreamingChunk {
    // Stage 1 output: count + decode
    #[allow(dead_code)]
    pub(crate) packed_block: Vec<u8>,
    pub(crate) g_block: Vec<f32>,
    pub(crate) miss_block: Vec<usize>,
    pub(crate) indices: Vec<usize>,
    pub(crate) flip: Vec<bool>,
    pub(crate) maf: Vec<f32>,
    pub(crate) miss_rate: Vec<f32>,
    pub(crate) chrom: Vec<String>,
    pub(crate) pos: Vec<i64>,
    pub(crate) snp: Vec<String>,
    pub(crate) a0: Vec<String>,
    pub(crate) a1: Vec<String>,
    // Stage 2 output
    pub(crate) rot_block: Vec<f32>,
    pub(crate) out_block: Vec<f64>,
    // Bookkeeping
    pub(crate) rows: usize,
    pub(crate) scanned_to: usize,
}

impl StreamingChunk {
    pub(crate) fn new(
        capacity: usize,
        n: usize,
        out_cols: usize,
        bytes_per_snp: usize,
        retain_packed: bool,
    ) -> Self {
        Self {
            packed_block: if retain_packed {
                vec![0u8; capacity.saturating_mul(bytes_per_snp)]
            } else {
                Vec::new()
            },
            g_block: vec![0.0_f32; capacity.saturating_mul(n)],
            miss_block: vec![0usize; capacity],
            indices: Vec::with_capacity(capacity),
            flip: Vec::with_capacity(capacity),
            maf: Vec::with_capacity(capacity),
            miss_rate: Vec::with_capacity(capacity),
            chrom: Vec::with_capacity(capacity),
            pos: Vec::with_capacity(capacity),
            snp: Vec::with_capacity(capacity),
            a0: Vec::with_capacity(capacity),
            a1: Vec::with_capacity(capacity),
            rot_block: vec![0.0_f32; capacity.saturating_mul(n)],
            out_block: vec![0.0_f64; capacity.saturating_mul(out_cols)],
            rows: 0,
            scanned_to: 0,
        }
    }

    pub(crate) fn new_decoded(capacity: usize, n: usize, out_cols: usize) -> Self {
        Self::new(capacity, n, out_cols, 0, false)
    }

    pub(crate) fn clear(&mut self) {
        self.indices.clear();
        self.flip.clear();
        self.maf.clear();
        self.miss_rate.clear();
        self.chrom.clear();
        self.pos.clear();
        self.snp.clear();
        self.a0.clear();
        self.a1.clear();
        self.rows = 0;
        self.scanned_to = 0;
    }
}

#[inline]
pub(crate) fn dot_loop(a: &[f64], b: &[f64]) -> f64 {
    dot_loop_simd(a, b)
}

#[inline]
#[allow(dead_code)]
fn dot_loop_unrolled(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let mut i = 0usize;
    while i + 3 < n {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut s = s0 + s1 + s2 + s3;
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_loop_avx2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0usize;
    let mut acc: __m256d = _mm256_setzero_pd();
    while i + 4 <= n {
        let va = unsafe { _mm256_loadu_pd(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_pd(b.as_ptr().add(i)) };
        acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
        i += 4;
    }
    let mut lanes = [0.0_f64; 4];
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
    let mut s = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_loop_neon(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::aarch64::{vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64};

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0usize;
    let mut acc = vdupq_n_f64(0.0);
    while i + 2 <= n {
        let va = unsafe { vld1q_f64(a.as_ptr().add(i)) };
        let vb = unsafe { vld1q_f64(b.as_ptr().add(i)) };
        acc = vaddq_f64(acc, vmulq_f64(va, vb));
        i += 2;
    }
    let mut lanes = [0.0_f64; 2];
    unsafe { vst1q_f64(lanes.as_mut_ptr(), acc) };
    let mut s = lanes[0] + lanes[1];
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[inline]
pub(crate) fn dot_loop_simd(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { dot_loop_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_loop_neon(a, b) };
    }
    #[allow(unreachable_code)]
    dot_loop_unrolled(a, b)
}

pub(crate) struct AssocScratch {
    pub(crate) c: Vec<f64>,       // len p
    pub(crate) a_inv_c: Vec<f64>, // len p
}

impl AssocScratch {
    pub(crate) fn new(p: usize) -> Self {
        Self {
            c: vec![0.0; p],
            a_inv_c: vec![0.0; p],
        }
    }

    #[inline]
    pub(crate) fn reset(&mut self) {
        self.c.fill(0.0);
    }
}

#[inline]
pub(crate) fn resolve_snp_name(snp: &str, chrom: &str, pos: i32) -> String {
    if !snp.is_empty() && snp != "." {
        snp.to_string()
    } else {
        format!("{chrom}_{pos}")
    }
}

#[inline]
pub(crate) fn missing_count_from_rate(v: f32, n_samples: usize) -> usize {
    if !v.is_finite() || v <= 0.0 {
        0usize
    } else {
        ((v as f64) * (n_samples as f64)).round().max(0.0) as usize
    }
}

#[inline]
pub(crate) fn missing_rate_from_count(v: usize, n_samples: usize) -> f32 {
    if n_samples == 0 {
        0.0_f32
    } else {
        (v as f32) / (n_samples as f32)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_packed_missing_block(
    miss_sub: &mut [usize],
    sample_identity: bool,
    row_missing: &[f32],
    row_idx: Option<&[usize]>,
    start: usize,
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    sample_idx: &[usize],
    pool: Option<&Arc<rayon::ThreadPool>>,
) {
    if sample_identity {
        if let Some(tp) = pool {
            tp.install(|| {
                miss_sub.par_iter_mut().enumerate().for_each(|(off, dst)| {
                    let idx = start + off;
                    // row_missing is aligned to the active/output row order; only packed
                    // row fetches should use row_idx indirection.
                    *dst = missing_count_from_rate(row_missing[idx], n_samples);
                });
            });
        } else {
            miss_sub.iter_mut().enumerate().for_each(|(off, dst)| {
                let idx = start + off;
                *dst = missing_count_from_rate(row_missing[idx], n_samples);
            });
        }
        return;
    }

    if let Some(tp) = pool {
        tp.install(|| {
            miss_sub.par_iter_mut().enumerate().for_each(|(off, dst)| {
                let idx = start + off;
                let src_row = row_idx.map(|v| v[idx]).unwrap_or(idx);
                let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                *dst = packed_row_missing_count_selected(row, n_samples, sample_idx);
            });
        });
    } else {
        miss_sub.iter_mut().enumerate().for_each(|(off, dst)| {
            let idx = start + off;
            let src_row = row_idx.map(|v| v[idx]).unwrap_or(idx);
            let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
            *dst = packed_row_missing_count_selected(row, n_samples, sample_idx);
        });
    }
}
