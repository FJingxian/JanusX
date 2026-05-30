use memmap2::Mmap;
use rayon::prelude::*;
use std::borrow::Cow;
use std::fs::File;
use std::sync::Arc;

use crate::bedmath::{
    adaptive_grm_block_rows, decode_standardized_packed_block_rows_f32, is_identity_indices,
    packed_byte_lut,
};
use crate::gfcore::{read_bim, read_fam};
use crate::he::{
    apply_grm_to_mat_f32_with_workspace, build_row_standardization_stats_with_options,
    GrmApplyWorkspace,
};
use crate::linalg::{cholesky_inplace, cholesky_solve_into};
use crate::stats_common::get_cached_pool;

#[derive(Clone, Debug)]
pub(crate) struct PcgResult<T> {
    pub(crate) x: Vec<T>,
    pub(crate) converged: bool,
    pub(crate) iters: usize,
    pub(crate) rel_res: f64,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct PcgSolveInfo {
    pub(crate) converged: bool,
    pub(crate) iters: usize,
    pub(crate) rel_res: f64,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct PcgMatrixResult<T> {
    pub(crate) x: Vec<T>,
    pub(crate) n_rows: usize,
    pub(crate) n_cols: usize,
    pub(crate) converged_all: bool,
    pub(crate) max_iters: usize,
    pub(crate) max_rel_res: f64,
    pub(crate) column_info: Vec<PcgSolveInfo>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct PcgMatrixSolveInfo {
    pub(crate) n_rows: usize,
    pub(crate) n_cols: usize,
    pub(crate) converged_all: bool,
    pub(crate) max_iters: usize,
    pub(crate) max_rel_res: f64,
    pub(crate) column_info: Vec<PcgSolveInfo>,
}

const PCG_PAR_THRESHOLD: usize = 16_384;
const PCG_BED_BLOCK_ROWS_BASE: usize = 1024;

pub(crate) trait PcgScalar: Copy + Default + Send + Sync {
    fn dot_to_f64(lhs: Self, rhs: Self) -> f64;
    fn mul(lhs: Self, rhs: Self) -> Self;
    fn add_scaled_in_place(dst: &mut Self, alpha: f64, value: Self);
    fn sub_scaled_in_place(dst: &mut Self, alpha: f64, value: Self);
    fn update_direction(z: Self, beta: f64, prev_p: Self) -> Self;
}

impl PcgScalar for f32 {
    #[inline]
    fn dot_to_f64(lhs: Self, rhs: Self) -> f64 {
        (lhs as f64) * (rhs as f64)
    }

    #[inline]
    fn mul(lhs: Self, rhs: Self) -> Self {
        lhs * rhs
    }

    #[inline]
    fn add_scaled_in_place(dst: &mut Self, alpha: f64, value: Self) {
        *dst += (alpha as f32) * value;
    }

    #[inline]
    fn sub_scaled_in_place(dst: &mut Self, alpha: f64, value: Self) {
        *dst -= (alpha as f32) * value;
    }

    #[inline]
    fn update_direction(z: Self, beta: f64, prev_p: Self) -> Self {
        z + (beta as f32) * prev_p
    }
}

impl PcgScalar for f64 {
    #[inline]
    fn dot_to_f64(lhs: Self, rhs: Self) -> f64 {
        lhs * rhs
    }

    #[inline]
    fn mul(lhs: Self, rhs: Self) -> Self {
        lhs * rhs
    }

    #[inline]
    fn add_scaled_in_place(dst: &mut Self, alpha: f64, value: Self) {
        *dst += alpha * value;
    }

    #[inline]
    fn sub_scaled_in_place(dst: &mut Self, alpha: f64, value: Self) {
        *dst -= alpha * value;
    }

    #[inline]
    fn update_direction(z: Self, beta: f64, prev_p: Self) -> Self {
        z + beta * prev_p
    }
}

pub(crate) trait PcgOperator<T: PcgScalar> {
    fn apply(&mut self, input: &[T], output: &mut [T]) -> Result<(), String>;
}

#[allow(dead_code)]
pub(crate) struct PcgFnOperator<F> {
    inner: F,
}

#[inline]
#[allow(dead_code)]
pub(crate) fn pcg_operator_fn<F>(inner: F) -> PcgFnOperator<F> {
    PcgFnOperator { inner }
}

impl<T, F> PcgOperator<T> for PcgFnOperator<F>
where
    T: PcgScalar,
    F: FnMut(&[T], &mut [T]) -> Result<(), String>,
{
    #[inline]
    fn apply(&mut self, input: &[T], output: &mut [T]) -> Result<(), String> {
        (self.inner)(input, output)
    }
}

impl<T, O> PcgOperator<T> for &mut O
where
    T: PcgScalar,
    O: PcgOperator<T> + ?Sized,
{
    #[inline]
    fn apply(&mut self, input: &[T], output: &mut [T]) -> Result<(), String> {
        (**self).apply(input, output)
    }
}

pub(crate) trait PcgPreconditioner<T: PcgScalar> {
    fn apply(&mut self, rhs: &[T], output: &mut [T]) -> Result<(), String>;
}

#[allow(dead_code)]
pub(crate) struct PcgFnPreconditioner<F> {
    inner: F,
}

#[inline]
#[allow(dead_code)]
pub(crate) fn pcg_preconditioner_fn<F>(inner: F) -> PcgFnPreconditioner<F> {
    PcgFnPreconditioner { inner }
}

impl<T, F> PcgPreconditioner<T> for PcgFnPreconditioner<F>
where
    T: PcgScalar,
    F: FnMut(&[T], &mut [T]) -> Result<(), String>,
{
    #[inline]
    fn apply(&mut self, rhs: &[T], output: &mut [T]) -> Result<(), String> {
        (self.inner)(rhs, output)
    }
}

impl<T, M> PcgPreconditioner<T> for &mut M
where
    T: PcgScalar,
    M: PcgPreconditioner<T> + ?Sized,
{
    #[inline]
    fn apply(&mut self, rhs: &[T], output: &mut [T]) -> Result<(), String> {
        (**self).apply(rhs, output)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct IdentityPreconditioner;

impl<T: PcgScalar> PcgPreconditioner<T> for IdentityPreconditioner {
    #[inline]
    fn apply(&mut self, rhs: &[T], output: &mut [T]) -> Result<(), String> {
        if rhs.len() != output.len() {
            return Err(format!(
                "PCG identity preconditioner length mismatch: rhs={}, out={}",
                rhs.len(),
                output.len()
            ));
        }
        output.copy_from_slice(rhs);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ScaledIdentityPreconditioner {
    inv_scale: f64,
}

impl ScaledIdentityPreconditioner {
    #[inline]
    pub(crate) fn new(scale: f64) -> Result<Self, String> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!(
                "PCG scaled-identity preconditioner requires finite positive scale, got {scale}"
            ));
        }
        Ok(Self {
            inv_scale: 1.0 / scale,
        })
    }
}

impl<T: PcgScalar> PcgPreconditioner<T> for ScaledIdentityPreconditioner {
    #[inline]
    fn apply(&mut self, rhs: &[T], output: &mut [T]) -> Result<(), String> {
        if rhs.len() != output.len() {
            return Err(format!(
                "PCG scaled-identity preconditioner length mismatch: rhs={}, out={}",
                rhs.len(),
                output.len()
            ));
        }
        if rhs.len() >= PCG_PAR_THRESHOLD {
            output
                .par_iter_mut()
                .zip(rhs.par_iter())
                .for_each(|(outj, rhsj)| {
                    let mut value = T::default();
                    T::add_scaled_in_place(&mut value, self.inv_scale, *rhsj);
                    *outj = value;
                });
        } else {
            for j in 0..rhs.len() {
                let mut value = T::default();
                T::add_scaled_in_place(&mut value, self.inv_scale, rhs[j]);
                output[j] = value;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DiagonalPreconditioner<'a, T> {
    inv_diag: &'a [T],
}

#[derive(Clone, Debug)]
enum PcgVariancePreconditionerF64 {
    Scaled(ScaledIdentityPreconditioner),
    Diagonal(Vec<f64>),
}

impl PcgPreconditioner<f64> for PcgVariancePreconditionerF64 {
    #[inline]
    fn apply(&mut self, rhs: &[f64], output: &mut [f64]) -> Result<(), String> {
        match self {
            Self::Scaled(inner) => inner.apply(rhs, output),
            Self::Diagonal(inv_diag) => {
                DiagonalPreconditioner::new(inv_diag.as_slice()).apply(rhs, output)
            }
        }
    }
}

impl<'a, T> DiagonalPreconditioner<'a, T> {
    #[inline]
    pub(crate) fn new(inv_diag: &'a [T]) -> Self {
        Self { inv_diag }
    }
}

impl<T: PcgScalar> PcgPreconditioner<T> for DiagonalPreconditioner<'_, T> {
    #[inline]
    fn apply(&mut self, rhs: &[T], output: &mut [T]) -> Result<(), String> {
        if rhs.len() != output.len() || rhs.len() != self.inv_diag.len() {
            return Err(format!(
                "PCG diagonal preconditioner length mismatch: rhs={}, out={}, diag={}",
                rhs.len(),
                output.len(),
                self.inv_diag.len()
            ));
        }
        if rhs.len() >= PCG_PAR_THRESHOLD {
            output
                .par_iter_mut()
                .zip(rhs.par_iter())
                .zip(self.inv_diag.par_iter())
                .for_each(|((outj, rhsj), dj)| *outj = T::mul(*rhsj, *dj));
        } else {
            for j in 0..rhs.len() {
                output[j] = T::mul(rhs[j], self.inv_diag[j]);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct PcgGrmBedConfig {
    pub(crate) sample_idx: Vec<usize>,
    pub(crate) row_flip: Vec<bool>,
    pub(crate) row_maf: Vec<f32>,
    pub(crate) row_indices: Option<Vec<usize>>,
    pub(crate) threads: usize,
    pub(crate) block_rows: usize,
    pub(crate) std_eps: f32,
    pub(crate) use_train_maf: bool,
}

impl PcgGrmBedConfig {
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn new(row_flip: Vec<bool>, row_maf: Vec<f32>) -> Self {
        Self {
            sample_idx: Vec::new(),
            row_flip,
            row_maf,
            row_indices: None,
            threads: 0,
            block_rows: 0,
            std_eps: 1e-12_f32,
            use_train_maf: false,
        }
    }
}

enum PcgBedPayload<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
    Mmap { mmap: Mmap, payload_offset: usize },
}

impl PcgBedPayload<'_> {
    #[inline]
    fn as_packed_bytes(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes.as_slice(),
            Self::Mmap {
                mmap,
                payload_offset,
            } => &mmap[*payload_offset..],
        }
    }
}

#[allow(dead_code)]
pub(crate) struct PcgGrmBedOperator<'a> {
    payload: PcgBedPayload<'a>,
    bytes_per_snp: usize,
    n_samples_full: usize,
    n_samples_used: usize,
    sample_idx: Vec<usize>,
    full_sample_fast: bool,
    row_flip: Vec<bool>,
    row_mean: Vec<f32>,
    row_inv_sd: Vec<f32>,
    diag_k: Vec<f32>,
    row_indices: Option<Vec<usize>>,
    block_rows: usize,
    m_scale: f32,
    pool: Option<Arc<rayon::ThreadPool>>,
    workspace: GrmApplyWorkspace,
    rhs_f32: Vec<f32>,
    out_f32: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn pcg_compute_grm_diag_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples_full: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    row_indices: Option<&[usize]>,
    block_rows: usize,
    m_scale: f32,
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<Vec<f32>, String> {
    let m = row_flip.len();
    let n_out = sample_idx.len();
    let mut diag = vec![0.0_f32; n_out];
    if m == 0 || n_out == 0 {
        return Ok(diag);
    }
    let row_step = block_rows.max(1).min(m.max(1));
    let mut block = vec![0.0_f32; row_step.saturating_mul(n_out)];
    for st in (0..m).step_by(row_step) {
        let ed = (st + row_step).min(m);
        let cur_rows = ed - st;
        let blk_slice = &mut block[..cur_rows * n_out];
        decode_standardized_packed_block_rows_f32(
            packed_flat,
            bytes_per_snp,
            n_samples_full,
            row_flip,
            row_mean,
            row_inv_sd,
            sample_idx,
            full_sample_fast,
            row_indices,
            st,
            blk_slice,
            code4_lut,
            pool,
        )?;
        for row in blk_slice.chunks(n_out) {
            for (dst, &value) in diag.iter_mut().zip(row.iter()) {
                *dst += value * value;
            }
        }
    }
    let scale = m_scale.max(1e-12_f32);
    for value in diag.iter_mut() {
        *value /= scale;
    }
    Ok(diag)
}

impl<'a> PcgGrmBedOperator<'a> {
    #[allow(dead_code)]
    pub(crate) fn from_packed_rows(
        packed: Cow<'a, [u8]>,
        n_samples_full: usize,
        cfg: PcgGrmBedConfig,
    ) -> Result<Self, String> {
        let payload = match packed {
            Cow::Borrowed(bytes) => PcgBedPayload::Borrowed(bytes),
            Cow::Owned(bytes) => PcgBedPayload::Owned(bytes),
        };
        Self::from_payload(payload, n_samples_full, cfg)
    }

    pub(crate) fn from_packed_rows_with_progress<F>(
        packed: Cow<'a, [u8]>,
        n_samples_full: usize,
        cfg: PcgGrmBedConfig,
        progress_callback: Option<&mut F>,
        progress_every_rows: usize,
    ) -> Result<Self, String>
    where
        F: FnMut(usize, usize) -> Result<(), String>,
    {
        let payload = match packed {
            Cow::Borrowed(bytes) => PcgBedPayload::Borrowed(bytes),
            Cow::Owned(bytes) => PcgBedPayload::Owned(bytes),
        };
        Self::from_payload_with_progress(
            payload,
            n_samples_full,
            cfg,
            progress_callback,
            progress_every_rows,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn from_bed_mmap_prefix(prefix: &str, cfg: PcgGrmBedConfig) -> Result<Self, String> {
        Self::from_bed_mmap_prefix_with_progress(
            prefix,
            cfg,
            None::<&mut fn(usize, usize) -> Result<(), String>>,
            0,
        )
    }

    pub(crate) fn from_bed_mmap_prefix_with_progress<F>(
        prefix: &str,
        cfg: PcgGrmBedConfig,
        progress_callback: Option<&mut F>,
        progress_every_rows: usize,
    ) -> Result<Self, String>
    where
        F: FnMut(usize, usize) -> Result<(), String>,
    {
        let bed_prefix = pcg_normalize_plink_prefix(prefix);
        if bed_prefix.is_empty() {
            return Err("PCG BED mmap prefix must not be empty".to_string());
        }
        let n_samples_full = read_fam(&bed_prefix)
            .map_err(|e| format!("failed to read {bed_prefix}.fam: {e}"))?
            .len();
        if n_samples_full == 0 {
            return Err("PCG BED mmap requires at least one sample".to_string());
        }
        let n_snps_bim = read_bim(&bed_prefix)
            .map_err(|e| format!("failed to read {bed_prefix}.bim: {e}"))?
            .len();
        let bed_path = format!("{bed_prefix}.bed");
        let file = File::open(&bed_path).map_err(|e| format!("failed to open {bed_path}: {e}"))?;
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| format!("failed to mmap {bed_path}: {e}"))?;
        if mmap.len() < 3 {
            return Err(format!("{bed_path}: file too small for BED header"));
        }
        if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
            return Err(format!(
                "{bed_path}: only SNP-major BED (0x6C 0x1B 0x01) is supported"
            ));
        }
        let bytes_per_snp = n_samples_full.div_ceil(4);
        if bytes_per_snp == 0 {
            return Err("PCG BED mmap bytes_per_snp resolved to zero".to_string());
        }
        let payload_len = mmap.len() - 3;
        if payload_len % bytes_per_snp != 0 {
            return Err(format!(
                "{bed_path}: invalid BED payload length {}, bytes_per_snp={}",
                payload_len, bytes_per_snp
            ));
        }
        let n_snps = payload_len / bytes_per_snp;
        if n_snps != n_snps_bim {
            return Err(format!(
                "{bed_path}: BED/BIM SNP count mismatch: bed={n_snps}, bim={n_snps_bim}"
            ));
        }
        Self::from_payload_with_progress(
            PcgBedPayload::Mmap {
                mmap,
                payload_offset: 3,
            },
            n_samples_full,
            cfg,
            progress_callback,
            progress_every_rows,
        )
    }

    fn from_payload(
        payload: PcgBedPayload<'a>,
        n_samples_full: usize,
        cfg: PcgGrmBedConfig,
    ) -> Result<Self, String> {
        Self::from_payload_with_progress(
            payload,
            n_samples_full,
            cfg,
            None::<&mut fn(usize, usize) -> Result<(), String>>,
            0,
        )
    }

    fn from_payload_with_progress<F>(
        payload: PcgBedPayload<'a>,
        n_samples_full: usize,
        cfg: PcgGrmBedConfig,
        progress_callback: Option<&mut F>,
        progress_every_rows: usize,
    ) -> Result<Self, String>
    where
        F: FnMut(usize, usize) -> Result<(), String>,
    {
        let PcgGrmBedConfig {
            sample_idx,
            row_flip,
            row_maf,
            row_indices,
            threads,
            block_rows,
            std_eps,
            use_train_maf,
        } = cfg;
        if n_samples_full == 0 {
            return Err("PCG GRM BED operator requires n_samples_full > 0".to_string());
        }
        let bytes_per_snp = n_samples_full.div_ceil(4);
        if bytes_per_snp == 0 {
            return Err("PCG GRM BED operator bytes_per_snp resolved to zero".to_string());
        }
        let packed = payload.as_packed_bytes();
        if packed.len() % bytes_per_snp != 0 {
            return Err(format!(
                "PCG GRM BED packed length mismatch: packed_bytes={}, bytes_per_snp={bytes_per_snp}",
                packed.len()
            ));
        }
        let n_rows_total = packed.len() / bytes_per_snp;
        if row_flip.len() != row_maf.len() {
            return Err(format!(
                "PCG GRM BED row metadata mismatch: row_flip={}, row_maf={}",
                row_flip.len(),
                row_maf.len()
            ));
        }
        if let Some(indices) = row_indices.as_ref() {
            if indices.len() != row_flip.len() {
                return Err(format!(
                    "PCG GRM BED row_indices length mismatch: row_indices={}, row_meta={}",
                    indices.len(),
                    row_flip.len()
                ));
            }
            if indices.iter().any(|&idx| idx >= n_rows_total) {
                return Err("PCG GRM BED row_indices contains out-of-range SNP index".to_string());
            }
        } else if row_flip.len() != n_rows_total {
            return Err(format!(
                "PCG GRM BED row count mismatch: row_meta={}, packed_rows={n_rows_total}",
                row_flip.len()
            ));
        }
        let sample_idx = pcg_normalize_sample_idx(sample_idx, n_samples_full)?;
        if sample_idx.is_empty() {
            return Err("PCG GRM BED operator sample_idx must not be empty".to_string());
        }
        let n_samples_used = sample_idx.len();
        let full_sample_fast = is_identity_indices(&sample_idx, n_samples_full);
        let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
        let row_stats = build_row_standardization_stats_with_options(
            packed,
            bytes_per_snp,
            &row_flip,
            &row_maf,
            &sample_idx,
            row_indices.as_deref(),
            std_eps.max(1e-12_f32),
            use_train_maf,
            true,
            pool.as_ref(),
            progress_callback,
            progress_every_rows,
        )?;
        if row_stats.m_effective == 0 {
            return Err(
                "PCG GRM BED operator found zero effective SNPs after standardization".to_string(),
            );
        }
        let base_rows = if block_rows == 0 {
            PCG_BED_BLOCK_ROWS_BASE
        } else {
            block_rows
        };
        let block_rows =
            adaptive_grm_block_rows(base_rows, row_flip.len(), n_samples_used, 0usize, threads)
                .max(1);
        let mut diag_k = row_stats
            .sample_diag_sum
            .unwrap_or_else(|| vec![0.0_f32; n_samples_used]);
        let scale = (row_stats.m_effective as f32).max(1e-12_f32);
        for value in diag_k.iter_mut() {
            *value /= scale;
        }
        Ok(Self {
            payload,
            bytes_per_snp,
            n_samples_full,
            n_samples_used,
            sample_idx,
            full_sample_fast,
            row_flip,
            row_mean: row_stats.row_mean,
            row_inv_sd: row_stats.row_inv_sd,
            diag_k,
            row_indices,
            block_rows,
            m_scale: row_stats.m_effective as f32,
            pool,
            workspace: GrmApplyWorkspace::new(),
            rhs_f32: vec![0.0_f32; n_samples_used],
            out_f32: vec![0.0_f32; n_samples_used],
        })
    }

    pub(crate) fn variance_jacobi_inv_diag(
        &self,
        sigma_g2: f64,
        sigma_e2: f64,
    ) -> Result<Vec<f64>, String> {
        pcg_validate_variance_components(sigma_g2, sigma_e2)?;
        let diag_src: Cow<'_, [f32]> = if self.diag_k.len() == self.n_samples_used {
            Cow::Borrowed(self.diag_k.as_slice())
        } else {
            Cow::Owned(pcg_compute_grm_diag_f32(
                self.payload.as_packed_bytes(),
                self.bytes_per_snp,
                self.n_samples_full,
                &self.row_flip,
                &self.row_mean,
                &self.row_inv_sd,
                &self.sample_idx,
                self.full_sample_fast,
                self.row_indices.as_deref(),
                self.block_rows,
                self.m_scale,
                &packed_byte_lut().code4,
                self.pool.as_ref(),
            )?)
        };
        let mut inv_diag = vec![0.0_f64; diag_src.len()];
        for (dst, &kdiag) in inv_diag.iter_mut().zip(diag_src.iter()) {
            let denom = sigma_g2 * (kdiag as f64) + sigma_e2;
            if !denom.is_finite() || denom <= 0.0 {
                return Err(format!(
                    "PCG Jacobi preconditioner encountered non-positive V diagonal: sigma_g2={sigma_g2}, sigma_e2={sigma_e2}, Kdiag={kdiag}"
                ));
            }
            *dst = 1.0_f64 / denom;
        }
        Ok(inv_diag)
    }
}

impl PcgOperator<f64> for PcgGrmBedOperator<'_> {
    fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), String> {
        if input.len() != self.n_samples_used || output.len() != self.n_samples_used {
            return Err(format!(
                "PCG GRM BED operator length mismatch: input={}, out={}, expected={}",
                input.len(),
                output.len(),
                self.n_samples_used
            ));
        }
        for (dst, &src) in self.rhs_f32.iter_mut().zip(input.iter()) {
            if !src.is_finite() {
                return Err("PCG GRM BED operator received non-finite input vector".to_string());
            }
            *dst = src as f32;
        }
        apply_grm_to_mat_f32_with_workspace(
            self.payload.as_packed_bytes(),
            self.bytes_per_snp,
            self.n_samples_full,
            &self.row_flip,
            &self.row_mean,
            &self.row_inv_sd,
            &self.sample_idx,
            self.full_sample_fast,
            self.row_indices.as_deref(),
            self.block_rows,
            &self.rhs_f32,
            1usize,
            self.m_scale,
            &packed_byte_lut().code4,
            self.pool.as_ref(),
            &mut self.workspace,
            None,
            &mut self.out_f32,
        )?;
        for (dst, &src) in output.iter_mut().zip(self.out_f32.iter()) {
            *dst = src as f64;
        }
        Ok(())
    }
}

impl PcgGrmBedOperator<'_> {
    pub(crate) fn apply_mat_row_major_f64(
        &mut self,
        input: &[f64],
        n_rhs: usize,
        output: &mut [f64],
    ) -> Result<(), String> {
        let total = self.n_samples_used.saturating_mul(n_rhs);
        if input.len() != total || output.len() != total {
            return Err(format!(
                "PCG GRM BED matrix apply length mismatch: input={}, out={}, expected={}",
                input.len(),
                output.len(),
                total
            ));
        }
        if self.rhs_f32.len() != total {
            self.rhs_f32.resize(total, 0.0_f32);
        }
        if self.out_f32.len() != total {
            self.out_f32.resize(total, 0.0_f32);
        }
        for (dst, &src) in self.rhs_f32.iter_mut().zip(input.iter()) {
            if !src.is_finite() {
                return Err("PCG GRM BED matrix apply received non-finite input".to_string());
            }
            *dst = src as f32;
        }
        apply_grm_to_mat_f32_with_workspace(
            self.payload.as_packed_bytes(),
            self.bytes_per_snp,
            self.n_samples_full,
            &self.row_flip,
            &self.row_mean,
            &self.row_inv_sd,
            &self.sample_idx,
            self.full_sample_fast,
            self.row_indices.as_deref(),
            self.block_rows,
            &self.rhs_f32,
            n_rhs,
            self.m_scale,
            &packed_byte_lut().code4,
            self.pool.as_ref(),
            &mut self.workspace,
            None,
            &mut self.out_f32,
        )?;
        for (dst, &src) in output.iter_mut().zip(self.out_f32.iter()) {
            *dst = src as f64;
        }
        Ok(())
    }
}

#[allow(dead_code)]
pub(crate) struct VarianceComponentOperator<K, T> {
    k_operator: K,
    sigma_g2: f64,
    sigma_e2: f64,
    kx: Vec<T>,
}

impl<K, T> VarianceComponentOperator<K, T> {
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn new(k_operator: K, sigma_g2: f64, sigma_e2: f64) -> Self {
        Self {
            k_operator,
            sigma_g2,
            sigma_e2,
            kx: Vec::new(),
        }
    }
}

impl<K, T> PcgOperator<T> for VarianceComponentOperator<K, T>
where
    K: PcgOperator<T>,
    T: PcgScalar,
{
    fn apply(&mut self, input: &[T], output: &mut [T]) -> Result<(), String> {
        if input.len() != output.len() {
            return Err(format!(
                "PCG variance-component operator length mismatch: input={}, out={}",
                input.len(),
                output.len()
            ));
        }
        if !self.sigma_g2.is_finite() || !self.sigma_e2.is_finite() {
            return Err(
                "PCG variance-component operator requires finite sigma_g2/sigma_e2.".to_string(),
            );
        }
        if self.kx.len() != input.len() {
            self.kx.resize(input.len(), T::default());
        }
        self.k_operator.apply(input, &mut self.kx)?;
        if output.len() >= PCG_PAR_THRESHOLD {
            output
                .par_iter_mut()
                .zip(input.par_iter())
                .zip(self.kx.par_iter())
                .for_each(|((outj, xj), kxj)| {
                    let mut value = T::default();
                    T::add_scaled_in_place(&mut value, self.sigma_g2, *kxj);
                    T::add_scaled_in_place(&mut value, self.sigma_e2, *xj);
                    *outj = value;
                });
        } else {
            for j in 0..output.len() {
                let mut value = T::default();
                T::add_scaled_in_place(&mut value, self.sigma_g2, self.kx[j]);
                T::add_scaled_in_place(&mut value, self.sigma_e2, input[j]);
                output[j] = value;
            }
        }
        Ok(())
    }
}

#[inline]
fn pcg_dot<T: PcgScalar>(a: &[T], b: &[T]) -> f64 {
    if a.len() >= PCG_PAR_THRESHOLD {
        a.par_iter()
            .zip(b.par_iter())
            .map(|(x, y)| T::dot_to_f64(*x, *y))
            .sum()
    } else {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| T::dot_to_f64(*x, *y))
            .sum()
    }
}

#[inline]
fn pcg_sub_assign<T: PcgScalar>(dst: &mut [T], src: &[T]) {
    if dst.len() >= PCG_PAR_THRESHOLD {
        dst.par_iter_mut()
            .zip(src.par_iter())
            .for_each(|(dstj, srcj)| T::sub_scaled_in_place(dstj, 1.0_f64, *srcj));
    } else {
        for j in 0..dst.len() {
            T::sub_scaled_in_place(&mut dst[j], 1.0_f64, src[j]);
        }
    }
}

#[inline]
fn pcg_update_solution_and_residual<T: PcgScalar>(
    x: &mut [T],
    r: &mut [T],
    p: &[T],
    ap: &[T],
    alpha: f64,
) {
    if x.len() >= PCG_PAR_THRESHOLD {
        x.par_iter_mut()
            .zip(r.par_iter_mut())
            .zip(p.par_iter().zip(ap.par_iter()))
            .for_each(|((xj, rj), (pj, apj))| {
                T::add_scaled_in_place(xj, alpha, *pj);
                T::sub_scaled_in_place(rj, alpha, *apj);
            });
    } else {
        for j in 0..x.len() {
            T::add_scaled_in_place(&mut x[j], alpha, p[j]);
            T::sub_scaled_in_place(&mut r[j], alpha, ap[j]);
        }
    }
}

#[inline]
fn pcg_update_direction<T: PcgScalar>(p: &mut [T], z: &[T], beta: f64) {
    if p.len() >= PCG_PAR_THRESHOLD {
        p.par_iter_mut()
            .zip(z.par_iter())
            .for_each(|(pj, zj)| *pj = T::update_direction(*zj, beta, *pj));
    } else {
        for j in 0..p.len() {
            p[j] = T::update_direction(z[j], beta, p[j]);
        }
    }
}

#[inline]
fn pcg_init_solution_buffer<T: PcgScalar>(
    x_out: &mut [T],
    b_len: usize,
    x0: Option<&[T]>,
) -> Result<(), String> {
    if x_out.len() != b_len {
        return Err(format!(
            "PCG output length mismatch: got {}, expected {}",
            x_out.len(),
            b_len
        ));
    }
    if let Some(guess) = x0 {
        if guess.len() != b_len {
            return Err(format!(
                "PCG initial guess length mismatch: got {}, expected {}",
                guess.len(),
                b_len
            ));
        }
        x_out.copy_from_slice(guess);
    } else {
        x_out.fill(T::default());
    }
    Ok(())
}

#[inline]
fn pcg_validate_matrix_shape<T>(
    name: &str,
    values: &[T],
    n_rows: usize,
    n_cols: usize,
) -> Result<(), String> {
    let expected = n_rows
        .checked_mul(n_cols)
        .ok_or_else(|| format!("PCG {name} shape overflow: rows={n_rows}, cols={n_cols}"))?;
    if values.len() != expected {
        return Err(format!(
            "PCG {name} length mismatch: got {}, expected {} (rows={} cols={})",
            values.len(),
            expected,
            n_rows,
            n_cols
        ));
    }
    Ok(())
}

#[inline]
fn pcg_copy_matrix_column<T: Copy>(
    matrix: &[T],
    n_rows: usize,
    n_cols: usize,
    col: usize,
    out: &mut [T],
) {
    if n_cols == 1 {
        out.copy_from_slice(matrix);
        return;
    }
    for row in 0..n_rows {
        out[row] = matrix[row * n_cols + col];
    }
}

#[inline]
fn pcg_store_matrix_column<T: Copy>(
    matrix: &mut [T],
    n_rows: usize,
    n_cols: usize,
    col: usize,
    values: &[T],
) {
    if n_cols == 1 {
        matrix.copy_from_slice(values);
        return;
    }
    for row in 0..n_rows {
        matrix[row * n_cols + col] = values[row];
    }
}

#[inline]
fn pcg_validate_variance_components(sigma_g2: f64, sigma_e2: f64) -> Result<(), String> {
    if !sigma_g2.is_finite() || sigma_g2 < 0.0 {
        return Err(format!(
            "PCG sigma_g2 must be finite and >= 0, got {sigma_g2}"
        ));
    }
    if !sigma_e2.is_finite() || sigma_e2 < 0.0 {
        return Err(format!(
            "PCG sigma_e2 must be finite and >= 0, got {sigma_e2}"
        ));
    }
    if sigma_g2 <= 0.0 && sigma_e2 <= 0.0 {
        return Err("PCG requires sigma_g2 > 0 or sigma_e2 > 0.".to_string());
    }
    Ok(())
}

#[inline]
fn pcg_normalize_plink_prefix(prefix: &str) -> String {
    let mut out = prefix.trim().to_string();
    let lower = out.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        out.truncate(out.len().saturating_sub(4));
    }
    out
}

#[inline]
fn pcg_normalize_sample_idx(
    sample_idx: Vec<usize>,
    n_samples_full: usize,
) -> Result<Vec<usize>, String> {
    if n_samples_full == 0 {
        return Err("PCG sample index normalization requires n_samples_full > 0".to_string());
    }
    if sample_idx.is_empty() {
        return Ok((0..n_samples_full).collect());
    }
    if sample_idx.iter().any(|&idx| idx >= n_samples_full) {
        return Err("PCG sample_idx contains out-of-range sample index".to_string());
    }
    Ok(sample_idx)
}

#[derive(Clone, Debug)]
pub(crate) struct PcgJxlmmNullModel {
    pub(crate) n_samples: usize,
    pub(crate) n_covariates: usize,
    pub(crate) v_inv_x: Vec<f64>,
    pub(crate) beta_hat: Vec<f64>,
    pub(crate) py: Vec<f64>,
    pub(crate) xt_v_inv_x_chol: Vec<f64>,
    pub(crate) xtx_chol: Vec<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct PcgJxlmmNullModelInfo {
    pub(crate) v_inv_y: PcgSolveInfo,
    pub(crate) v_inv_x: PcgMatrixSolveInfo,
}

#[derive(Clone, Debug)]
pub(crate) struct PcgJxlmmRHatResult {
    pub(crate) r_hat: f64,
    pub(crate) n_markers_requested: usize,
    pub(crate) n_markers_used: usize,
    #[allow(dead_code)]
    pub(crate) solve_info: PcgMatrixSolveInfo,
}

#[inline]
fn pcg_variance_preconditioner_f64(
    sigma_g2: f64,
    sigma_e2: f64,
) -> Result<PcgVariancePreconditionerF64, String> {
    pcg_validate_variance_components(sigma_g2, sigma_e2)?;
    let scale = (sigma_g2 + sigma_e2)
        .max(sigma_e2)
        .max(sigma_g2)
        .max(1e-12_f64);
    Ok(PcgVariancePreconditionerF64::Scaled(
        ScaledIdentityPreconditioner::new(scale)?,
    ))
}

#[inline]
fn pcg_variance_preconditioner_jacobi_f64(
    sigma_g2: f64,
    sigma_e2: f64,
    inv_diag: Option<&[f64]>,
) -> Result<PcgVariancePreconditionerF64, String> {
    if let Some(inv_diag) = inv_diag {
        if inv_diag.is_empty() {
            return Err(
                "PCG Jacobi preconditioner requires non-empty inverse diagonal".to_string(),
            );
        }
        if inv_diag.iter().any(|&v| !v.is_finite() || v <= 0.0) {
            return Err(
                "PCG Jacobi preconditioner requires finite positive inverse diagonal entries"
                    .to_string(),
            );
        }
        return Ok(PcgVariancePreconditionerF64::Diagonal(inv_diag.to_vec()));
    }
    pcg_variance_preconditioner_f64(sigma_g2, sigma_e2)
}

#[inline]
fn pcg_xt_vec(
    x_row_major: &[f64],
    n_samples: usize,
    n_covariates: usize,
    vec_in: &[f64],
    out: &mut [f64],
) {
    out.fill(0.0);
    for i in 0..n_samples {
        let yi = vec_in[i];
        let row = &x_row_major[i * n_covariates..(i + 1) * n_covariates];
        for j in 0..n_covariates {
            out[j] += row[j] * yi;
        }
    }
}

#[inline]
fn pcg_xt_mat(
    x_row_major: &[f64],
    y_row_major: &[f64],
    n_samples: usize,
    n_covariates: usize,
    n_cols_y: usize,
    out: &mut [f64],
) {
    out.fill(0.0);
    for i in 0..n_samples {
        let x_row = &x_row_major[i * n_covariates..(i + 1) * n_covariates];
        let y_row = &y_row_major[i * n_cols_y..(i + 1) * n_cols_y];
        for r in 0..n_covariates {
            let xr = x_row[r];
            for c in 0..n_cols_y {
                out[r * n_cols_y + c] += xr * y_row[c];
            }
        }
    }
}

#[inline]
fn pcg_spd_cholesky_with_jitter(
    matrix: &[f64],
    dim: usize,
    label: &str,
) -> Result<Vec<f64>, String> {
    pcg_validate_matrix_shape(label, matrix, dim, dim)?;
    let mut chol = matrix.to_vec();
    if cholesky_inplace(&mut chol, dim).is_some() {
        return Ok(chol);
    }
    let trace = (0..dim).map(|i| matrix[i * dim + i].abs()).sum::<f64>();
    let base_jitter = (trace / (dim.max(1) as f64)).max(1.0) * 1e-10_f64;
    for attempt in 0..6 {
        let jitter = base_jitter * 10.0_f64.powi(attempt);
        chol.copy_from_slice(matrix);
        for i in 0..dim {
            chol[i * dim + i] += jitter;
        }
        if cholesky_inplace(&mut chol, dim).is_some() {
            return Ok(chol);
        }
    }
    Err(format!(
        "PCG SparseLMM failed to factor {label} as SPD after diagonal jitter"
    ))
}

#[inline]
fn pcg_validate_jxlmm_state(
    null_model: &PcgJxlmmNullModel,
    x_row_major: &[f64],
) -> Result<(), String> {
    pcg_validate_matrix_shape(
        "SparseLMM design matrix",
        x_row_major,
        null_model.n_samples,
        null_model.n_covariates,
    )?;
    pcg_validate_matrix_shape(
        "SparseLMM V^{-1}X matrix",
        &null_model.v_inv_x,
        null_model.n_samples,
        null_model.n_covariates,
    )?;
    if null_model.beta_hat.len() != null_model.n_covariates
        || null_model.py.len() != null_model.n_samples
    {
        return Err("PCG SparseLMM null model state length mismatch".to_string());
    }
    pcg_validate_matrix_shape(
        "SparseLMM XtVinvX chol",
        &null_model.xt_v_inv_x_chol,
        null_model.n_covariates,
        null_model.n_covariates,
    )?;
    pcg_validate_matrix_shape(
        "SparseLMM XtX chol",
        &null_model.xtx_chol,
        null_model.n_covariates,
        null_model.n_covariates,
    )?;
    Ok(())
}

pub(crate) fn pcg_build_jxlmm_null_model<FY, FX>(
    x_row_major: &[f64],
    y: &[f64],
    n_samples: usize,
    n_covariates: usize,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    k_operator: &mut PcgGrmBedOperator<'_>,
    sigma_g2: f64,
    sigma_e2: f64,
    jacobi_inv_diag: Option<&[f64]>,
    mut on_v_inv_y_iteration: FY,
    mut on_v_inv_x_column_iteration: FX,
) -> Result<(PcgJxlmmNullModel, PcgJxlmmNullModelInfo), String>
where
    FY: FnMut(usize, usize, f64) -> Result<(), String>,
    FX: FnMut(usize, usize, usize, f64) -> Result<(), String>,
{
    if n_samples == 0 || n_covariates == 0 {
        return Err("PCG SparseLMM null model requires n_samples > 0 and n_covariates > 0".to_string());
    }
    pcg_validate_matrix_shape("SparseLMM design matrix", x_row_major, n_samples, n_covariates)?;
    if y.len() != n_samples {
        return Err(format!(
            "PCG SparseLMM y length mismatch: got {}, expected {n_samples}",
            y.len()
        ));
    }

    let precond_y = pcg_variance_preconditioner_jacobi_f64(sigma_g2, sigma_e2, jacobi_inv_diag)?;
    let v_inv_y = pcg_solve_v_inv_y(
        y,
        None,
        max_iter,
        tol,
        tiny,
        &mut *k_operator,
        sigma_g2,
        sigma_e2,
        precond_y,
        |iter, iter_max, rel_res| on_v_inv_y_iteration(iter, iter_max, rel_res),
    )?;
    if v_inv_y.iters < max_iter {
        on_v_inv_y_iteration(max_iter, max_iter, v_inv_y.rel_res)?;
    }
    if !v_inv_y.converged {
        return Err(format!(
            "PCG SparseLMM V^-1 y did not converge: iters={}, rel_res={:.3e}",
            v_inv_y.iters, v_inv_y.rel_res
        ));
    }

    let v_inv_x = if let Some(inv_diag) = jacobi_inv_diag {
        pcg_solve_variance_matrix_batched_f64(
            x_row_major,
            n_samples,
            n_covariates,
            max_iter,
            tol,
            tiny,
            k_operator,
            sigma_g2,
            sigma_e2,
            inv_diag,
            |col, iter, iter_max, rel_res| {
                on_v_inv_x_column_iteration(col, iter, iter_max, rel_res)
            },
        )?
    } else {
        let precond_x =
            pcg_variance_preconditioner_jacobi_f64(sigma_g2, sigma_e2, jacobi_inv_diag)?;
        pcg_solve_v_inv_x(
            x_row_major,
            n_samples,
            n_covariates,
            None,
            max_iter,
            tol,
            tiny,
            &mut *k_operator,
            sigma_g2,
            sigma_e2,
            precond_x,
            |col, iter, iter_max, rel_res| {
                on_v_inv_x_column_iteration(col, iter, iter_max, rel_res)
            },
        )?
    };
    if v_inv_x
        .column_info
        .last()
        .map(|info| info.iters < max_iter)
        .unwrap_or(false)
    {
        on_v_inv_x_column_iteration(
            n_covariates.saturating_sub(1),
            max_iter,
            max_iter,
            v_inv_x.max_rel_res,
        )?;
    }
    if !v_inv_x.converged_all {
        return Err(format!(
            "PCG SparseLMM V^-1 X did not converge for all columns: max_iters={}, max_rel_res={:.3e}",
            v_inv_x.max_iters, v_inv_x.max_rel_res
        ));
    }

    let mut xt_v_inv_y = vec![0.0_f64; n_covariates];
    pcg_xt_vec(
        x_row_major,
        n_samples,
        n_covariates,
        &v_inv_y.x,
        &mut xt_v_inv_y,
    );

    let mut xt_v_inv_x = vec![0.0_f64; n_covariates * n_covariates];
    pcg_xt_mat(
        x_row_major,
        &v_inv_x.x,
        n_samples,
        n_covariates,
        n_covariates,
        &mut xt_v_inv_x,
    );
    let xt_v_inv_x_chol =
        pcg_spd_cholesky_with_jitter(&xt_v_inv_x, n_covariates, "SparseLMM XtVinvX")?;
    let mut beta_hat = vec![0.0_f64; n_covariates];
    cholesky_solve_into(&xt_v_inv_x_chol, n_covariates, &xt_v_inv_y, &mut beta_hat);

    let mut py = v_inv_y.x.clone();
    for i in 0..n_samples {
        let vinvx_row = &v_inv_x.x[i * n_covariates..(i + 1) * n_covariates];
        let adjust = vinvx_row
            .iter()
            .zip(beta_hat.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        py[i] -= adjust;
    }

    let mut xtx = vec![0.0_f64; n_covariates * n_covariates];
    pcg_xt_mat(
        x_row_major,
        x_row_major,
        n_samples,
        n_covariates,
        n_covariates,
        &mut xtx,
    );
    let xtx_chol = pcg_spd_cholesky_with_jitter(&xtx, n_covariates, "SparseLMM XtX")?;

    Ok((
        PcgJxlmmNullModel {
            n_samples,
            n_covariates,
            v_inv_x: v_inv_x.x,
            beta_hat,
            py,
            xt_v_inv_x_chol,
            xtx_chol,
        },
        PcgJxlmmNullModelInfo {
            v_inv_y: pcg_solve_info(v_inv_y.converged, v_inv_y.iters, v_inv_y.rel_res),
            v_inv_x: PcgMatrixSolveInfo {
                n_rows: v_inv_x.n_rows,
                n_cols: v_inv_x.n_cols,
                converged_all: v_inv_x.converged_all,
                max_iters: v_inv_x.max_iters,
                max_rel_res: v_inv_x.max_rel_res,
                column_info: v_inv_x.column_info,
            },
        },
    ))
}

pub(crate) fn pcg_jxlmm_s_m_s(
    null_model: &PcgJxlmmNullModel,
    x_row_major: &[f64],
    snp: &[f64],
) -> Result<f64, String> {
    pcg_validate_jxlmm_state(null_model, x_row_major)?;
    if snp.len() != null_model.n_samples {
        return Err(format!(
            "PCG SparseLMM s^T M s SNP length mismatch: got {}, expected {}",
            snp.len(),
            null_model.n_samples
        ));
    }
    let p = null_model.n_covariates;
    let mut xts = vec![0.0_f64; p];
    pcg_xt_vec(x_row_major, null_model.n_samples, p, snp, &mut xts);
    let mut alpha = vec![0.0_f64; p];
    cholesky_solve_into(&null_model.xtx_chol, p, &xts, &mut alpha);
    let s_sq = snp.iter().map(|v| v * v).sum::<f64>();
    let cross = xts
        .iter()
        .zip(alpha.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>();
    Ok((s_sq - cross).max(0.0_f64))
}

pub(crate) fn pcg_jxlmm_s_p_s_exact(
    null_model: &PcgJxlmmNullModel,
    x_row_major: &[f64],
    snp: &[f64],
    v_inv_snp: &[f64],
) -> Result<f64, String> {
    pcg_validate_jxlmm_state(null_model, x_row_major)?;
    if snp.len() != null_model.n_samples || v_inv_snp.len() != null_model.n_samples {
        return Err(format!(
            "PCG SparseLMM s^T P s length mismatch: snp={}, v_inv_snp={}, expected={}",
            snp.len(),
            v_inv_snp.len(),
            null_model.n_samples
        ));
    }
    let p = null_model.n_covariates;
    let mut xt_v_inv_s = vec![0.0_f64; p];
    pcg_xt_vec(
        x_row_major,
        null_model.n_samples,
        p,
        v_inv_snp,
        &mut xt_v_inv_s,
    );
    let mut gamma = vec![0.0_f64; p];
    cholesky_solve_into(&null_model.xt_v_inv_x_chol, p, &xt_v_inv_s, &mut gamma);
    let s_v_inv_s = snp
        .iter()
        .zip(v_inv_snp.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let quad = xt_v_inv_s
        .iter()
        .zip(gamma.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>();
    Ok((s_v_inv_s - quad).max(0.0_f64))
}

#[inline]
fn pcg_col_major_to_row_major_f64(
    matrix_col_major: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<f64>, String> {
    pcg_validate_matrix_shape("column-major matrix", matrix_col_major, n_rows, n_cols)?;
    let mut out = vec![0.0_f64; n_rows.saturating_mul(n_cols)];
    for col in 0..n_cols {
        let src = &matrix_col_major[col * n_rows..(col + 1) * n_rows];
        for row in 0..n_rows {
            out[row * n_cols + col] = src[row];
        }
    }
    Ok(out)
}

#[inline]
fn pcg_apply_diag_preconditioner_mat_f64(
    inv_diag: &[f64],
    rhs_row_major: &[f64],
    n_rows: usize,
    n_cols: usize,
    out_row_major: &mut [f64],
) -> Result<(), String> {
    pcg_validate_matrix_shape("PCG diag-precond rhs", rhs_row_major, n_rows, n_cols)?;
    pcg_validate_matrix_shape("PCG diag-precond out", out_row_major, n_rows, n_cols)?;
    if inv_diag.len() != n_rows {
        return Err(format!(
            "PCG diag-precond length mismatch: diag={}, rows={n_rows}",
            inv_diag.len()
        ));
    }
    if n_rows.saturating_mul(n_cols) >= PCG_PAR_THRESHOLD {
        out_row_major
            .par_chunks_mut(n_cols)
            .zip(rhs_row_major.par_chunks(n_cols))
            .zip(inv_diag.par_iter())
            .for_each(|((out_row, rhs_row), &scale)| {
                for col in 0..n_cols {
                    out_row[col] = rhs_row[col] * scale;
                }
            });
    } else {
        for row in 0..n_rows {
            let scale = inv_diag[row];
            let rhs_row = &rhs_row_major[row * n_cols..(row + 1) * n_cols];
            let out_row = &mut out_row_major[row * n_cols..(row + 1) * n_cols];
            for col in 0..n_cols {
                out_row[col] = rhs_row[col] * scale;
            }
        }
    }
    Ok(())
}

#[inline]
fn pcg_apply_variance_mat_f64(
    k_operator: &mut PcgGrmBedOperator<'_>,
    input_row_major: &[f64],
    n_rows: usize,
    n_cols: usize,
    sigma_g2: f64,
    sigma_e2: f64,
    kx_row_major: &mut Vec<f64>,
    output_row_major: &mut [f64],
) -> Result<(), String> {
    pcg_validate_matrix_shape("PCG variance-mat input", input_row_major, n_rows, n_cols)?;
    pcg_validate_matrix_shape("PCG variance-mat out", output_row_major, n_rows, n_cols)?;
    let total = n_rows.saturating_mul(n_cols);
    if kx_row_major.len() != total {
        kx_row_major.resize(total, 0.0_f64);
    }
    k_operator.apply_mat_row_major_f64(input_row_major, n_cols, kx_row_major)?;
    for ((dst, &kg), &xv) in output_row_major
        .iter_mut()
        .zip(kx_row_major.iter())
        .zip(input_row_major.iter())
    {
        *dst = sigma_g2.mul_add(kg, sigma_e2 * xv);
    }
    Ok(())
}

fn pcg_solve_variance_matrix_batched_f64<F>(
    rhs_row_major: &[f64],
    n_rows: usize,
    n_cols: usize,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    k_operator: &mut PcgGrmBedOperator<'_>,
    sigma_g2: f64,
    sigma_e2: f64,
    inv_diag: &[f64],
    mut on_column_iteration: F,
) -> Result<PcgMatrixResult<f64>, String>
where
    F: FnMut(usize, usize, usize, f64) -> Result<(), String>,
{
    pcg_validate_variance_components(sigma_g2, sigma_e2)?;
    pcg_validate_matrix_shape("PCG batched variance rhs", rhs_row_major, n_rows, n_cols)?;
    if n_cols == 0 {
        return Err("PCG batched variance solve requires at least one RHS column".to_string());
    }
    if inv_diag.len() != n_rows {
        return Err(format!(
            "PCG batched variance Jacobi diagonal mismatch: diag={}, rows={n_rows}",
            inv_diag.len()
        ));
    }
    if inv_diag.iter().any(|&v| !v.is_finite() || v <= 0.0) {
        return Err(
            "PCG batched variance solve requires finite positive Jacobi inverse diagonal entries"
                .to_string(),
        );
    }

    let total = n_rows.saturating_mul(n_cols);
    let mut x = vec![0.0_f64; total];
    let mut r = rhs_row_major.to_vec();
    let mut z = vec![0.0_f64; total];
    let mut p_mat = vec![0.0_f64; total];
    let mut ap = vec![0.0_f64; total];
    let mut kx = vec![0.0_f64; total];
    let mut rhs_norm = vec![0.0_f64; n_cols];
    let mut rz_old = vec![0.0_f64; n_cols];
    let mut rz_new = vec![0.0_f64; n_cols];
    let mut alpha = vec![0.0_f64; n_cols];
    let mut p_ap = vec![0.0_f64; n_cols];
    let mut rr = vec![0.0_f64; n_cols];
    let mut rel_res = vec![f64::INFINITY; n_cols];
    let mut converged = vec![false; n_cols];
    let mut column_info = vec![pcg_solve_info(false, 0, f64::INFINITY); n_cols];

    pcg_apply_diag_preconditioner_mat_f64(inv_diag, &r, n_rows, n_cols, &mut z)?;
    p_mat.copy_from_slice(&z);
    for row in 0..n_rows {
        let row_off = row * n_cols;
        let r_row = &r[row_off..row_off + n_cols];
        let z_row = &z[row_off..row_off + n_cols];
        for col in 0..n_cols {
            rhs_norm[col] += r_row[col] * r_row[col];
            rz_old[col] += r_row[col] * z_row[col];
        }
    }
    for col in 0..n_cols {
        let rr0 = rhs_norm[col];
        rhs_norm[col] = rr0.sqrt().max(tiny);
        rel_res[col] = rr0.sqrt() / rhs_norm[col];
        if rel_res[col] <= tol || rz_old[col].abs() <= tiny {
            converged[col] = true;
            column_info[col] = pcg_solve_info(true, 0, rel_res[col]);
        }
    }

    for iter in 1..=max_iter {
        if converged.iter().all(|&v| v) {
            break;
        }
        pcg_apply_variance_mat_f64(
            k_operator, &p_mat, n_rows, n_cols, sigma_g2, sigma_e2, &mut kx, &mut ap,
        )?;
        rz_new.fill(0.0_f64);
        alpha.fill(0.0_f64);
        p_ap.fill(0.0_f64);
        for row in 0..n_rows {
            let row_off = row * n_cols;
            let p_row = &p_mat[row_off..row_off + n_cols];
            let ap_row = &ap[row_off..row_off + n_cols];
            for col in 0..n_cols {
                if converged[col] {
                    continue;
                }
                p_ap[col] += p_row[col] * ap_row[col];
            }
        }
        for col in 0..n_cols {
            if converged[col] {
                continue;
            }
            if !p_ap[col].is_finite() || p_ap[col].abs() <= tiny {
                rel_res[col] = f64::INFINITY;
                continue;
            }
            alpha[col] = rz_old[col] / p_ap[col];
        }
        for row in 0..n_rows {
            let row_off = row * n_cols;
            for col in 0..n_cols {
                if converged[col] {
                    continue;
                }
                let idx = row_off + col;
                x[idx] += alpha[col] * p_mat[idx];
                r[idx] -= alpha[col] * ap[idx];
            }
        }
        pcg_apply_diag_preconditioner_mat_f64(inv_diag, &r, n_rows, n_cols, &mut z)?;
        rr.fill(0.0_f64);
        for row in 0..n_rows {
            let row_off = row * n_cols;
            let r_row = &r[row_off..row_off + n_cols];
            let z_row = &z[row_off..row_off + n_cols];
            for col in 0..n_cols {
                if converged[col] {
                    continue;
                }
                rr[col] += r_row[col] * r_row[col];
                rz_new[col] += r_row[col] * z_row[col];
            }
        }
        for col in 0..n_cols {
            if converged[col] {
                continue;
            }
            rel_res[col] = rr[col].sqrt() / rhs_norm[col];
            on_column_iteration(col, iter, max_iter, rel_res[col])?;
            if rel_res[col].is_finite() && rel_res[col] <= tol {
                converged[col] = true;
                column_info[col] = pcg_solve_info(true, iter, rel_res[col]);
                continue;
            }
            if !rz_new[col].is_finite() || rz_old[col].abs() <= tiny {
                column_info[col] = pcg_solve_info(false, iter, rel_res[col]);
                continue;
            }
            let beta = rz_new[col] / rz_old[col];
            for row in 0..n_rows {
                let idx = row * n_cols + col;
                p_mat[idx] = z[idx] + beta * p_mat[idx];
            }
            rz_old[col] = rz_new[col];
            column_info[col] = pcg_solve_info(false, iter, rel_res[col]);
        }
    }

    Ok(PcgMatrixResult {
        x,
        n_rows,
        n_cols,
        converged_all: converged.iter().all(|&v| v),
        max_iters: column_info
            .iter()
            .map(|info| info.iters)
            .max()
            .unwrap_or(0usize),
        max_rel_res: column_info
            .iter()
            .map(|info| info.rel_res)
            .filter(|v| v.is_finite())
            .fold(0.0_f64, f64::max),
        column_info,
    })
}

pub(crate) fn pcg_estimate_jxlmm_r_hat<F>(
    null_model: &PcgJxlmmNullModel,
    x_row_major: &[f64],
    sampled_markers_col_major: &[f64],
    n_markers: usize,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    k_operator: &mut PcgGrmBedOperator<'_>,
    sigma_g2: f64,
    sigma_e2: f64,
    jacobi_inv_diag: Option<&[f64]>,
    mut on_v_inv_s_column_iteration: F,
) -> Result<PcgJxlmmRHatResult, String>
where
    F: FnMut(usize, usize, usize, f64) -> Result<(), String>,
{
    pcg_validate_jxlmm_state(null_model, x_row_major)?;
    pcg_validate_matrix_shape(
        "SparseLMM sampled marker matrix",
        sampled_markers_col_major,
        null_model.n_samples,
        n_markers,
    )?;
    if n_markers == 0 {
        return Err("PCG SparseLMM r-hat estimation requires at least one sampled marker".to_string());
    }
    let inv_diag = jacobi_inv_diag.ok_or_else(|| {
        "PCG SparseLMM r-hat batched solver requires Jacobi inverse diagonal".to_string()
    })?;
    let rhs_row_major =
        pcg_col_major_to_row_major_f64(sampled_markers_col_major, null_model.n_samples, n_markers)?;
    let v_inv_s = pcg_solve_variance_matrix_batched_f64(
        &rhs_row_major,
        null_model.n_samples,
        n_markers,
        max_iter,
        tol,
        tiny,
        k_operator,
        sigma_g2,
        sigma_e2,
        inv_diag,
        |col, iter, iter_max, rel_res| on_v_inv_s_column_iteration(col, iter, iter_max, rel_res),
    )?;

    let mut ratio_sum = 0.0_f64;
    let mut n_used = 0usize;
    let mut v_inv_col = vec![0.0_f64; null_model.n_samples];
    for col in 0..n_markers {
        let snp_col = &sampled_markers_col_major
            [col * null_model.n_samples..(col + 1) * null_model.n_samples];
        pcg_copy_matrix_column(
            &v_inv_s.x,
            null_model.n_samples,
            n_markers,
            col,
            &mut v_inv_col,
        );
        let s_m_s = pcg_jxlmm_s_m_s(null_model, x_row_major, snp_col)?;
        if !s_m_s.is_finite() || s_m_s <= tiny.max(1e-30_f64) {
            continue;
        }
        let s_p_s = pcg_jxlmm_s_p_s_exact(null_model, x_row_major, snp_col, &v_inv_col)?;
        if !s_p_s.is_finite() || s_p_s <= tiny.max(1e-30_f64) {
            continue;
        }
        let ratio = s_p_s / s_m_s;
        if ratio.is_finite() && ratio > 0.0 {
            ratio_sum += ratio;
            n_used += 1;
        }
    }
    if n_used == 0 {
        return Err("PCG SparseLMM r-hat estimation found no valid sampled markers".to_string());
    }

    Ok(PcgJxlmmRHatResult {
        r_hat: ratio_sum / (n_used as f64),
        n_markers_requested: n_markers,
        n_markers_used: n_used,
        solve_info: PcgMatrixSolveInfo {
            n_rows: v_inv_s.n_rows,
            n_cols: v_inv_s.n_cols,
            converged_all: v_inv_s.converged_all,
            max_iters: v_inv_s.max_iters,
            max_rel_res: v_inv_s.max_rel_res,
            column_info: v_inv_s.column_info,
        },
    })
}

#[inline]
fn pcg_solve_info(converged: bool, iters: usize, rel_res: f64) -> PcgSolveInfo {
    PcgSolveInfo {
        converged,
        iters,
        rel_res,
    }
}

#[allow(dead_code)]
pub(crate) fn pcg_solve_into<T, FA, FM, FO>(
    x_out: &mut [T],
    b: &[T],
    x0: Option<&[T]>,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    mut apply_a: FA,
    mut apply_m_inv: FM,
    mut on_iteration: FO,
) -> Result<PcgSolveInfo, String>
where
    T: PcgScalar,
    FA: PcgOperator<T>,
    FM: PcgPreconditioner<T>,
    FO: FnMut(usize, usize, f64) -> Result<(), String>,
{
    let m = b.len();
    pcg_init_solution_buffer(x_out, m, x0)?;
    if m == 0 || max_iter == 0 {
        return Ok(pcg_solve_info(m == 0, 0, 0.0_f64));
    }

    let bnorm = pcg_dot(b, b).sqrt();
    if !bnorm.is_finite() {
        return Err("PCG invalid RHS norm.".to_string());
    }
    let denom_b = bnorm.max(1e-12_f64);

    let mut r = b.to_vec();
    let mut ap = vec![T::default(); m];
    if x0.is_some() {
        apply_a.apply(x_out, &mut ap)?;
        pcg_sub_assign(&mut r, &ap);
    }

    let mut z = vec![T::default(); m];
    apply_m_inv.apply(&r, &mut z)?;
    let mut p = z.clone();
    let mut rz_old = pcg_dot(&r, &z);
    let mut converged = false;
    let mut rel_res = (pcg_dot(&r, &r).sqrt() / denom_b).max(0.0_f64);
    let mut iters_done = 0usize;
    let tiny_use = tiny.max(1e-30_f64);
    let tol_use = tol.max(0.0_f64);

    on_iteration(0usize, max_iter, rel_res)?;
    if rel_res.is_finite() && rel_res <= tol_use {
        converged = true;
        return Ok(pcg_solve_info(converged, 0, rel_res));
    }

    for it in 0..max_iter {
        apply_a.apply(&p, &mut ap)?;
        let denom = pcg_dot(&p, &ap);
        if !denom.is_finite() || denom <= tiny_use {
            break;
        }
        let alpha = rz_old / denom;
        pcg_update_solution_and_residual(x_out, &mut r, &p, &ap, alpha);
        rel_res = (pcg_dot(&r, &r).sqrt() / denom_b).max(0.0_f64);
        iters_done = it + 1;
        on_iteration(iters_done, max_iter, rel_res)?;
        if rel_res.is_finite() && rel_res <= tol_use {
            converged = true;
            break;
        }

        apply_m_inv.apply(&r, &mut z)?;
        let rz_new = pcg_dot(&r, &z);
        if !rz_new.is_finite() || rz_new <= tiny_use {
            break;
        }
        let beta = rz_new / rz_old.max(tiny_use);
        pcg_update_direction(&mut p, &z, beta);
        rz_old = rz_new;
    }

    Ok(pcg_solve_info(converged, iters_done, rel_res))
}

pub(crate) fn pcg_solve<T, FA, FM, FO>(
    b: &[T],
    x0: Option<&[T]>,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    apply_a: FA,
    apply_m_inv: FM,
    on_iteration: FO,
) -> Result<PcgResult<T>, String>
where
    T: PcgScalar,
    FA: PcgOperator<T>,
    FM: PcgPreconditioner<T>,
    FO: FnMut(usize, usize, f64) -> Result<(), String>,
{
    let mut x = vec![T::default(); b.len()];
    let info = pcg_solve_into(
        &mut x,
        b,
        x0,
        max_iter,
        tol,
        tiny,
        apply_a,
        apply_m_inv,
        on_iteration,
    )?;
    Ok(PcgResult {
        x,
        converged: info.converged,
        iters: info.iters,
        rel_res: info.rel_res,
    })
}

#[allow(dead_code)]
pub(crate) fn pcg_solve_matrix_into<T, FA, FM, FO>(
    x_out_row_major: &mut [T],
    rhs_row_major: &[T],
    n_rows: usize,
    n_cols: usize,
    x0_row_major: Option<&[T]>,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    apply_a: &mut FA,
    apply_m_inv: &mut FM,
    mut on_column_iteration: FO,
) -> Result<PcgMatrixSolveInfo, String>
where
    T: PcgScalar,
    FA: PcgOperator<T> + ?Sized,
    FM: PcgPreconditioner<T> + ?Sized,
    FO: FnMut(usize, usize, usize, f64) -> Result<(), String>,
{
    pcg_validate_matrix_shape("output matrix", x_out_row_major, n_rows, n_cols)?;
    pcg_validate_matrix_shape("RHS matrix", rhs_row_major, n_rows, n_cols)?;
    if let Some(guess) = x0_row_major {
        pcg_validate_matrix_shape("initial-guess matrix", guess, n_rows, n_cols)?;
    }

    let mut rhs_col = vec![T::default(); n_rows];
    let mut x_col = vec![T::default(); n_rows];
    let mut guess_col = x0_row_major.map(|_| vec![T::default(); n_rows]);
    let mut converged_all = true;
    let mut max_iters = 0usize;
    let mut max_rel_res = 0.0_f64;
    let mut column_info = Vec::with_capacity(n_cols);

    for col in 0..n_cols {
        pcg_copy_matrix_column(rhs_row_major, n_rows, n_cols, col, &mut rhs_col);
        let guess_slice = if let (Some(guess), Some(buf)) = (x0_row_major, guess_col.as_mut()) {
            pcg_copy_matrix_column(guess, n_rows, n_cols, col, buf);
            Some(buf.as_slice())
        } else {
            None
        };

        let info = pcg_solve_into(
            &mut x_col,
            &rhs_col,
            guess_slice,
            max_iter,
            tol,
            tiny,
            &mut *apply_a,
            &mut *apply_m_inv,
            |iter, iter_max, rel_res| on_column_iteration(col, iter, iter_max, rel_res),
        )?;
        pcg_store_matrix_column(x_out_row_major, n_rows, n_cols, col, &x_col);
        converged_all &= info.converged;
        max_iters = max_iters.max(info.iters);
        max_rel_res = if info.rel_res.is_finite() {
            max_rel_res.max(info.rel_res)
        } else {
            f64::INFINITY
        };
        column_info.push(info);
    }

    Ok(PcgMatrixSolveInfo {
        n_rows,
        n_cols,
        converged_all,
        max_iters,
        max_rel_res,
        column_info,
    })
}

#[allow(dead_code)]
pub(crate) fn pcg_solve_matrix<T, FA, FM, FO>(
    rhs_row_major: &[T],
    n_rows: usize,
    n_cols: usize,
    x0_row_major: Option<&[T]>,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    apply_a: &mut FA,
    apply_m_inv: &mut FM,
    on_column_iteration: FO,
) -> Result<PcgMatrixResult<T>, String>
where
    T: PcgScalar,
    FA: PcgOperator<T> + ?Sized,
    FM: PcgPreconditioner<T> + ?Sized,
    FO: FnMut(usize, usize, usize, f64) -> Result<(), String>,
{
    let total = n_rows
        .checked_mul(n_cols)
        .ok_or_else(|| format!("PCG output shape overflow: rows={n_rows}, cols={n_cols}"))?;
    let mut x = vec![T::default(); total];
    let info = pcg_solve_matrix_into(
        &mut x,
        rhs_row_major,
        n_rows,
        n_cols,
        x0_row_major,
        max_iter,
        tol,
        tiny,
        apply_a,
        apply_m_inv,
        on_column_iteration,
    )?;
    Ok(PcgMatrixResult {
        x,
        n_rows: info.n_rows,
        n_cols: info.n_cols,
        converged_all: info.converged_all,
        max_iters: info.max_iters,
        max_rel_res: info.max_rel_res,
        column_info: info.column_info,
    })
}

#[allow(dead_code)]
pub(crate) fn pcg_solve_v_inv_y_into<K, M, FO>(
    y_out: &mut [f64],
    y: &[f64],
    x0: Option<&[f64]>,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    k_operator: K,
    sigma_g2: f64,
    sigma_e2: f64,
    apply_m_inv: M,
    on_iteration: FO,
) -> Result<PcgSolveInfo, String>
where
    K: PcgOperator<f64>,
    M: PcgPreconditioner<f64>,
    FO: FnMut(usize, usize, f64) -> Result<(), String>,
{
    pcg_validate_variance_components(sigma_g2, sigma_e2)?;
    let v_operator = VarianceComponentOperator::new(k_operator, sigma_g2, sigma_e2);
    pcg_solve_into(
        y_out,
        y,
        x0,
        max_iter,
        tol,
        tiny,
        v_operator,
        apply_m_inv,
        on_iteration,
    )
}

#[allow(dead_code)]
pub(crate) fn pcg_solve_v_inv_y<K, M, FO>(
    y: &[f64],
    x0: Option<&[f64]>,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    k_operator: K,
    sigma_g2: f64,
    sigma_e2: f64,
    apply_m_inv: M,
    on_iteration: FO,
) -> Result<PcgResult<f64>, String>
where
    K: PcgOperator<f64>,
    M: PcgPreconditioner<f64>,
    FO: FnMut(usize, usize, f64) -> Result<(), String>,
{
    let mut y_out = vec![0.0_f64; y.len()];
    let info = pcg_solve_v_inv_y_into(
        &mut y_out,
        y,
        x0,
        max_iter,
        tol,
        tiny,
        k_operator,
        sigma_g2,
        sigma_e2,
        apply_m_inv,
        on_iteration,
    )?;
    Ok(PcgResult {
        x: y_out,
        converged: info.converged,
        iters: info.iters,
        rel_res: info.rel_res,
    })
}

#[allow(dead_code)]
pub(crate) fn pcg_solve_v_inv_x_into<K, M, FO>(
    x_out_row_major: &mut [f64],
    x_row_major: &[f64],
    n_rows: usize,
    n_cols: usize,
    x0_row_major: Option<&[f64]>,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    k_operator: K,
    sigma_g2: f64,
    sigma_e2: f64,
    apply_m_inv: M,
    on_column_iteration: FO,
) -> Result<PcgMatrixSolveInfo, String>
where
    K: PcgOperator<f64>,
    M: PcgPreconditioner<f64>,
    FO: FnMut(usize, usize, usize, f64) -> Result<(), String>,
{
    pcg_validate_variance_components(sigma_g2, sigma_e2)?;
    let mut v_operator = VarianceComponentOperator::new(k_operator, sigma_g2, sigma_e2);
    let mut preconditioner = apply_m_inv;
    pcg_solve_matrix_into(
        x_out_row_major,
        x_row_major,
        n_rows,
        n_cols,
        x0_row_major,
        max_iter,
        tol,
        tiny,
        &mut v_operator,
        &mut preconditioner,
        on_column_iteration,
    )
}

#[allow(dead_code)]
pub(crate) fn pcg_solve_v_inv_x<K, M, FO>(
    x_row_major: &[f64],
    n_rows: usize,
    n_cols: usize,
    x0_row_major: Option<&[f64]>,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    k_operator: K,
    sigma_g2: f64,
    sigma_e2: f64,
    apply_m_inv: M,
    on_column_iteration: FO,
) -> Result<PcgMatrixResult<f64>, String>
where
    K: PcgOperator<f64>,
    M: PcgPreconditioner<f64>,
    FO: FnMut(usize, usize, usize, f64) -> Result<(), String>,
{
    let total = n_rows
        .checked_mul(n_cols)
        .ok_or_else(|| format!("PCG output shape overflow: rows={n_rows}, cols={n_cols}"))?;
    let mut x_out = vec![0.0_f64; total];
    let info = pcg_solve_v_inv_x_into(
        &mut x_out,
        x_row_major,
        n_rows,
        n_cols,
        x0_row_major,
        max_iter,
        tol,
        tiny,
        k_operator,
        sigma_g2,
        sigma_e2,
        apply_m_inv,
        on_column_iteration,
    )?;
    Ok(PcgMatrixResult {
        x: x_out,
        n_rows: info.n_rows,
        n_cols: info.n_cols,
        converged_all: info.converged_all,
        max_iters: info.max_iters,
        max_rel_res: info.max_rel_res,
        column_info: info.column_info,
    })
}

#[allow(dead_code)]
pub(crate) fn pcg_solve_v_inv_y_packed_bed<'a, M, FO>(
    packed: Cow<'a, [u8]>,
    n_samples_full: usize,
    bed_cfg: PcgGrmBedConfig,
    y: &[f64],
    x0: Option<&[f64]>,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    sigma_g2: f64,
    sigma_e2: f64,
    apply_m_inv: M,
    on_iteration: FO,
) -> Result<PcgResult<f64>, String>
where
    M: PcgPreconditioner<f64>,
    FO: FnMut(usize, usize, f64) -> Result<(), String>,
{
    let operator = PcgGrmBedOperator::from_packed_rows(packed, n_samples_full, bed_cfg)?;
    pcg_solve_v_inv_y(
        y,
        x0,
        max_iter,
        tol,
        tiny,
        operator,
        sigma_g2,
        sigma_e2,
        apply_m_inv,
        on_iteration,
    )
}

#[allow(dead_code)]
pub(crate) fn pcg_solve_v_inv_x_packed_bed<'a, M, FO>(
    packed: Cow<'a, [u8]>,
    n_samples_full: usize,
    bed_cfg: PcgGrmBedConfig,
    x_row_major: &[f64],
    n_rows: usize,
    n_cols: usize,
    x0_row_major: Option<&[f64]>,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    sigma_g2: f64,
    sigma_e2: f64,
    apply_m_inv: M,
    on_column_iteration: FO,
) -> Result<PcgMatrixResult<f64>, String>
where
    M: PcgPreconditioner<f64>,
    FO: FnMut(usize, usize, usize, f64) -> Result<(), String>,
{
    let operator = PcgGrmBedOperator::from_packed_rows(packed, n_samples_full, bed_cfg)?;
    pcg_solve_v_inv_x(
        x_row_major,
        n_rows,
        n_cols,
        x0_row_major,
        max_iter,
        tol,
        tiny,
        operator,
        sigma_g2,
        sigma_e2,
        apply_m_inv,
        on_column_iteration,
    )
}

#[allow(dead_code)]
pub(crate) fn pcg_solve_v_inv_y_mmap_bed<M, FO>(
    prefix: &str,
    bed_cfg: PcgGrmBedConfig,
    y: &[f64],
    x0: Option<&[f64]>,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    sigma_g2: f64,
    sigma_e2: f64,
    apply_m_inv: M,
    on_iteration: FO,
) -> Result<PcgResult<f64>, String>
where
    M: PcgPreconditioner<f64>,
    FO: FnMut(usize, usize, f64) -> Result<(), String>,
{
    let operator = PcgGrmBedOperator::from_bed_mmap_prefix(prefix, bed_cfg)?;
    pcg_solve_v_inv_y(
        y,
        x0,
        max_iter,
        tol,
        tiny,
        operator,
        sigma_g2,
        sigma_e2,
        apply_m_inv,
        on_iteration,
    )
}

#[allow(dead_code)]
pub(crate) fn pcg_solve_v_inv_x_mmap_bed<M, FO>(
    prefix: &str,
    bed_cfg: PcgGrmBedConfig,
    x_row_major: &[f64],
    n_rows: usize,
    n_cols: usize,
    x0_row_major: Option<&[f64]>,
    max_iter: usize,
    tol: f64,
    tiny: f64,
    sigma_g2: f64,
    sigma_e2: f64,
    apply_m_inv: M,
    on_column_iteration: FO,
) -> Result<PcgMatrixResult<f64>, String>
where
    M: PcgPreconditioner<f64>,
    FO: FnMut(usize, usize, usize, f64) -> Result<(), String>,
{
    let operator = PcgGrmBedOperator::from_bed_mmap_prefix(prefix, bed_cfg)?;
    pcg_solve_v_inv_x(
        x_row_major,
        n_rows,
        n_cols,
        x0_row_major,
        max_iter,
        tol,
        tiny,
        operator,
        sigma_g2,
        sigma_e2,
        apply_m_inv,
        on_column_iteration,
    )
}
