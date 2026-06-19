use crate::bedmath::SubsetDecodePlan;
use crate::bedmath::{
    decode_row_centered_full_lut, decode_row_centered_full_lut_f64,
    decode_standardized_packed_block_rows_f32_with_plan, decode_subset_row_from_full_scratch,
    decode_subset_row_from_full_scratch_f64, decode_subset_with_plan, packed_byte_lut,
};
use crate::gfreader::SampleSubsetPlan;
use crate::gload::WindowedBedMatrix;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::PyResult;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) struct AdditiveDecodePlan {
    subset: SampleSubsetPlan,
    subset_decode_plan: Option<SubsetDecodePlan>,
    dense_subset_pos: Option<Vec<isize>>,
    sample_byte_idx: Option<Vec<usize>>,
    sample_bit_shift: Option<Vec<u8>>,
}

#[inline]
fn maybe_build_dense_subset_pos(selected: &[usize], n_samples_full: usize) -> Option<Vec<isize>> {
    if selected.is_empty() || selected.len() >= n_samples_full {
        return None;
    }
    if selected.len().saturating_mul(4) < n_samples_full.saturating_mul(3) {
        return None;
    }
    let mut pos = vec![-1isize; n_samples_full];
    for (j, &sid) in selected.iter().enumerate() {
        if sid >= n_samples_full || pos[sid] >= 0 {
            return None;
        }
        pos[sid] = j as isize;
    }
    Some(pos)
}

impl AdditiveDecodePlan {
    pub(crate) fn from_sample_indices(n_samples_full: usize, sample_indices: &[usize]) -> Self {
        Self::from_optional_indices(n_samples_full, Some(sample_indices))
    }

    pub(crate) fn from_optional_indices(
        n_samples_full: usize,
        sample_indices: Option<&[usize]>,
    ) -> Self {
        let subset = SampleSubsetPlan::from_optional_indices(n_samples_full, sample_indices);
        let selected = subset.selected();
        let subset_decode_plan = selected
            .map(|idx| SubsetDecodePlan::from_sample_idx_with_n_samples(idx, n_samples_full));
        let dense_subset_pos =
            selected.and_then(|idx| maybe_build_dense_subset_pos(idx, n_samples_full));
        let sample_byte_idx = selected.map(|idx| idx.iter().map(|&sid| sid >> 2).collect());
        let sample_bit_shift =
            selected.map(|idx| idx.iter().map(|&sid| ((sid & 3) << 1) as u8).collect());
        Self {
            subset,
            subset_decode_plan,
            dense_subset_pos,
            sample_byte_idx,
            sample_bit_shift,
        }
    }

    #[inline]
    pub(crate) fn sample_identity(&self) -> bool {
        self.subset.is_identity()
    }

    #[inline]
    pub(crate) fn selected_excluded_sample_indices(&self) -> Option<&[usize]> {
        self.subset.excluded()
    }

    #[inline]
    pub(crate) fn subset_decode_plan(&self) -> Option<&SubsetDecodePlan> {
        self.subset_decode_plan.as_ref()
    }

    #[inline]
    pub(crate) fn dense_subset_pos(&self) -> Option<&[isize]> {
        self.dense_subset_pos.as_deref()
    }

    #[inline]
    pub(crate) fn sample_byte_idx(&self) -> Option<&[usize]> {
        self.sample_byte_idx.as_deref()
    }

    #[inline]
    pub(crate) fn sample_bit_shift(&self) -> Option<&[u8]> {
        self.sample_bit_shift.as_deref()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PackedGeneticModel {
    Add,
    Dom,
    Rec,
    Het,
}

impl PackedGeneticModel {
    pub(crate) fn parse(text: &str) -> PyResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "add" => Ok(Self::Add),
            "dom" => Ok(Self::Dom),
            "rec" => Ok(Self::Rec),
            "het" => Ok(Self::Het),
            _ => Err(PyRuntimeError::new_err(
                "model must be one of: add, dom, rec, het",
            )),
        }
    }

    #[inline]
    pub(crate) fn apply(self, g: f64) -> f64 {
        match self {
            Self::Add => g,
            Self::Dom => {
                if g > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Rec => {
                if (g - 2.0).abs() < 1e-6 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Het => {
                if (g - 1.0).abs() < 1e-6 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

pub(crate) enum PackedRowDecodeSource<'a> {
    Resident {
        packed_flat: &'a [u8],
        bytes_per_snp: usize,
    },
    Windowed(WindowedBedMatrix),
}

#[inline]
pub(crate) fn resolve_source_row_index(row_indices: Option<&[usize]>, row_idx: usize) -> usize {
    row_indices.map(|v| v[row_idx]).unwrap_or(row_idx)
}

#[inline]
fn packed_model_value_lut_f32<F>(apply: &F, flip: bool, mean_g: f32) -> [f32; 4]
where
    F: Fn(f64) -> f64 + Sync,
{
    let raw = if flip {
        [2.0_f32, mean_g, 1.0_f32, 0.0_f32]
    } else {
        [0.0_f32, mean_g, 1.0_f32, 2.0_f32]
    };
    [
        apply(raw[0] as f64) as f32,
        apply(raw[1] as f64) as f32,
        apply(raw[2] as f64) as f32,
        apply(raw[3] as f64) as f32,
    ]
}

#[inline]
fn center_decoded_row_inplace_f32(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let mean = (row.iter().map(|&v| v as f64).sum::<f64>() / (row.len() as f64)) as f32;
    for v in row.iter_mut() {
        *v -= mean;
    }
}

#[inline]
pub(crate) fn decode_centered_block_packed_f32<F>(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_indices: Option<&[usize]>,
    row_start: usize,
    rows: usize,
    n: usize,
    apply: &F,
    sample_identity: bool,
    subset_plan: Option<&SubsetDecodePlan>,
    dense_subset_pos: Option<&[isize]>,
    out: &mut [f32],
) where
    F: Fn(f64) -> f64 + Sync,
{
    if rows == 0 || n == 0 {
        return;
    }
    debug_assert_eq!(out.len(), rows * n);
    let code4_lut = &packed_byte_lut().code4;

    if sample_identity {
        out.par_chunks_mut(n).enumerate().for_each(|(off, dst)| {
            let idx = row_start + off;
            let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
            let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
            let mean_g = (2.0_f64 * (row_maf[idx] as f64)).max(0.0);
            let value_lut = packed_model_value_lut_f32(apply, row_flip[idx], mean_g as f32);
            decode_row_centered_full_lut(row, n, code4_lut, &value_lut, dst);
            center_decoded_row_inplace_f32(dst);
        });
        return;
    }

    if let Some(sample_pos) = dense_subset_pos {
        out.par_chunks_mut(n).enumerate().for_each_init(
            || vec![0.0_f32; sample_pos.len()],
            |full_row, (off, dst)| {
                let idx = row_start + off;
                let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
                let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
                let mean_g = (2.0_f64 * (row_maf[idx] as f64)).max(0.0);
                let value_lut = packed_model_value_lut_f32(apply, row_flip[idx], mean_g as f32);
                decode_row_centered_full_lut(
                    row,
                    sample_pos.len(),
                    code4_lut,
                    &value_lut,
                    full_row,
                );
                let mut sum_g = 0.0_f64;
                for (full_j, &out_j) in sample_pos.iter().enumerate() {
                    if out_j >= 0 {
                        let gv = full_row[full_j];
                        dst[out_j as usize] = gv;
                        sum_g += gv as f64;
                    }
                }
                let g_mean = (sum_g / (n as f64)) as f32;
                for v in dst.iter_mut() {
                    *v -= g_mean;
                }
            },
        );
        return;
    }

    let plan = subset_plan.expect("subset_plan must exist for non-identity mapping");
    out.par_chunks_mut(n).enumerate().for_each(|(off, dst)| {
        let idx = row_start + off;
        let src_row = row_indices.map(|v| v[idx]).unwrap_or(idx);
        let row = &packed_flat[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
        let mean_g = (2.0_f64 * (row_maf[idx] as f64)).max(0.0);
        let value_lut = packed_model_value_lut_f32(apply, row_flip[idx], mean_g as f32);
        decode_subset_with_plan(row, plan, &value_lut, dst);
        center_decoded_row_inplace_f32(dst);
    });
}

#[inline]
pub(crate) fn decode_centered_block_packed_model_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_indices: Option<&[usize]>,
    row_start: usize,
    rows: usize,
    n: usize,
    gm: PackedGeneticModel,
    sample_identity: bool,
    subset_plan: Option<&SubsetDecodePlan>,
    dense_subset_pos: Option<&[isize]>,
    out: &mut [f32],
) {
    decode_centered_block_packed_f32(
        packed_flat,
        bytes_per_snp,
        row_flip,
        row_maf,
        row_indices,
        row_start,
        rows,
        n,
        &|g| gm.apply(g),
        sample_identity,
        subset_plan,
        dense_subset_pos,
        out,
    );
}

#[inline]
pub(crate) fn decode_packed_row_model_into_f64<F>(
    row: &[u8],
    flip: bool,
    row_maf: f32,
    n: usize,
    apply: &F,
    sample_identity: bool,
    sample_byte_idx: Option<&[usize]>,
    sample_bit_shift: Option<&[u8]>,
    out: &mut [f64],
) where
    F: Fn(f64) -> f64 + Sync,
{
    debug_assert_eq!(out.len(), n);
    let mean_g = (2.0_f64 * row_maf as f64).max(0.0);
    if sample_identity {
        let mut j = 0usize;
        for &b in row.iter() {
            let mut lane = 0usize;
            while lane < 4 && j < n {
                let code = (b >> (lane * 2)) & 0b11;
                let mut gv = match code {
                    0b00 => 0.0_f64,
                    0b10 => 1.0_f64,
                    0b11 => 2.0_f64,
                    _ => mean_g,
                };
                if flip && code != 0b01 {
                    gv = 2.0_f64 - gv;
                }
                out[j] = apply(gv);
                lane += 1;
                j += 1;
            }
            if j >= n {
                break;
            }
        }
        return;
    }

    let byte_idx = sample_byte_idx.expect("sample_byte_idx must exist for non-identity mapping");
    let bit_shift = sample_bit_shift.expect("sample_bit_shift must exist for non-identity mapping");
    for j in 0..n {
        let b = row[byte_idx[j]];
        let code = (b >> bit_shift[j]) & 0b11;
        let mut gv = match code {
            0b00 => 0.0_f64,
            0b10 => 1.0_f64,
            0b11 => 2.0_f64,
            _ => mean_g,
        };
        if flip && code != 0b01 {
            gv = 2.0_f64 - gv;
        }
        out[j] = apply(gv);
    }
}

#[inline]
pub(crate) fn decode_packed_row_model_enum_into_f64(
    row: &[u8],
    flip: bool,
    row_maf: f32,
    n: usize,
    gm: PackedGeneticModel,
    sample_identity: bool,
    sample_byte_idx: Option<&[usize]>,
    sample_bit_shift: Option<&[u8]>,
    out: &mut [f64],
) {
    decode_packed_row_model_into_f64(
        row,
        flip,
        row_maf,
        n,
        &|g| gm.apply(g),
        sample_identity,
        sample_byte_idx,
        sample_bit_shift,
        out,
    );
}

#[inline]
pub(crate) fn decode_indexed_row_model_into_f64(
    row_source: &mut PackedRowDecodeSource<'_>,
    row_flip: &[bool],
    row_maf: &[f32],
    row_indices: Option<&[usize]>,
    row_idx: usize,
    n: usize,
    gm: PackedGeneticModel,
    sample_identity: bool,
    sample_byte_idx: Option<&[usize]>,
    sample_bit_shift: Option<&[u8]>,
    out: &mut [f64],
) -> Result<(), String> {
    let src = resolve_source_row_index(row_indices, row_idx);
    let flip = row_flip[row_idx];
    let maf = row_maf[row_idx];
    match row_source {
        PackedRowDecodeSource::Resident {
            packed_flat,
            bytes_per_snp,
        } => {
            let row = &packed_flat[src * *bytes_per_snp..(src + 1) * *bytes_per_snp];
            decode_packed_row_model_enum_into_f64(
                row,
                flip,
                maf,
                n,
                gm,
                sample_identity,
                sample_byte_idx,
                sample_bit_shift,
                out,
            );
            Ok(())
        }
        PackedRowDecodeSource::Windowed(matrix) => {
            let row = matrix.read_source_range(src, src + 1)?;
            decode_packed_row_model_enum_into_f64(
                row,
                flip,
                maf,
                n,
                gm,
                sample_identity,
                sample_byte_idx,
                sample_bit_shift,
                out,
            );
            Ok(())
        }
    }
}

#[inline]
pub(crate) fn apply_prepared_grm_stream_row_copy_f32(
    src: &[f32],
    dst: &mut [f32],
    mean_g: f32,
    std_scale: f32,
    flip_major: bool,
) {
    for (g_src, g_dst) in src.iter().zip(dst.iter_mut()) {
        if *g_src < 0.0_f32 {
            *g_dst = 0.0_f32;
            continue;
        }
        let gv = if flip_major { 2.0_f32 - *g_src } else { *g_src };
        *g_dst = (gv - mean_g) * std_scale;
    }
}

#[inline]
pub(crate) fn apply_prepared_grm_stream_row_copy_f64(
    src: &[f32],
    dst: &mut [f64],
    mean_g: f64,
    std_scale: f64,
    flip_major: bool,
) {
    for (g_src, g_dst) in src.iter().zip(dst.iter_mut()) {
        if *g_src < 0.0_f32 {
            *g_dst = 0.0_f64;
            continue;
        }
        let gv = if flip_major {
            2.0_f64 - (*g_src as f64)
        } else {
            *g_src as f64
        };
        *g_dst = (gv - mean_g) * std_scale;
    }
}

#[inline]
pub(crate) fn prepare_packed_block_row_indices(
    row_indices: Option<&[usize]>,
    row_start: usize,
    rows: usize,
    out_row_indices: &mut [usize],
) -> Result<(), String> {
    if out_row_indices.len() < rows {
        return Err("prepare_packed_block_row_indices: out_row_indices too small".to_string());
    }
    if let Some(indices) = row_indices {
        if indices.len() < row_start.saturating_add(rows) {
            return Err("prepare_packed_block_row_indices: row_indices too small".to_string());
        }
        out_row_indices[..rows].copy_from_slice(&indices[row_start..row_start + rows]);
    } else {
        for (off, dst) in out_row_indices[..rows].iter_mut().enumerate() {
            *dst = row_start + off;
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn prepare_packed_block_centered_mean_scale_f32(
    row_maf: &[f32],
    row_start: usize,
    rows: usize,
    out_mean: &mut [f32],
    out_scale: &mut [f32],
) -> Result<(), String> {
    if out_mean.len() < rows {
        return Err("prepare_packed_block_centered_mean_scale_f32: out_mean too small".to_string());
    }
    if out_scale.len() < rows {
        return Err(
            "prepare_packed_block_centered_mean_scale_f32: out_scale too small".to_string(),
        );
    }
    if row_maf.len() < row_start.saturating_add(rows) {
        return Err("prepare_packed_block_centered_mean_scale_f32: row_maf too small".to_string());
    }
    for off in 0..rows {
        out_mean[off] = 2.0_f32 * row_maf[row_start + off];
        out_scale[off] = 1.0_f32;
    }
    Ok(())
}

#[inline]
pub(crate) fn prepare_packed_block_standardized_mean_scale_f32(
    row_maf: &[f32],
    row_start: usize,
    rows: usize,
    eps: f32,
    out_mean: &mut [f32],
    out_scale: &mut [f32],
) -> Result<(), String> {
    if out_mean.len() < rows {
        return Err(
            "prepare_packed_block_standardized_mean_scale_f32: out_mean too small".to_string(),
        );
    }
    if out_scale.len() < rows {
        return Err(
            "prepare_packed_block_standardized_mean_scale_f32: out_scale too small".to_string(),
        );
    }
    if row_maf.len() < row_start.saturating_add(rows) {
        return Err(
            "prepare_packed_block_standardized_mean_scale_f32: row_maf too small".to_string(),
        );
    }
    for off in 0..rows {
        let p = row_maf[row_start + off].clamp(0.0_f32, 1.0_f32);
        let var = 2.0_f32 * p * (1.0_f32 - p);
        out_mean[off] = 2.0_f32 * p;
        out_scale[off] = if var > eps {
            1.0_f32 / var.sqrt()
        } else {
            0.0_f32
        };
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_prepared_additive_block_packed_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_scale: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    subset_plan: Option<&SubsetDecodePlan>,
    packed_row_indices: Option<&[usize]>,
    row_start: usize,
    row_end: usize,
    out_block: &mut [f32],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let cur_rows = row_end.saturating_sub(row_start);
    let n_out = sample_idx.len();
    if cur_rows == 0 || n_out == 0 {
        return Ok(());
    }
    if row_flip.len() < row_end {
        return Err("decode_prepared_additive_block_packed_f32: row_flip too small".to_string());
    }
    if row_mean.len() < row_end {
        return Err("decode_prepared_additive_block_packed_f32: row_mean too small".to_string());
    }
    if row_scale.len() < row_end {
        return Err("decode_prepared_additive_block_packed_f32: row_scale too small".to_string());
    }
    if let Some(indices) = packed_row_indices {
        if indices.len() < row_end {
            return Err(
                "decode_prepared_additive_block_packed_f32: packed_row_indices too small"
                    .to_string(),
            );
        }
    }
    if out_block.len() < cur_rows.saturating_mul(n_out) {
        return Err("decode_prepared_additive_block_packed_f32: out_block too small".to_string());
    }
    decode_standardized_packed_block_rows_f32_with_plan(
        packed_flat,
        bytes_per_snp,
        n_samples,
        row_flip,
        row_mean,
        row_scale,
        sample_idx,
        full_sample_fast,
        subset_plan,
        packed_row_indices,
        row_start,
        &mut out_block[..cur_rows * n_out],
        &packed_byte_lut().code4,
        pool,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_centered_additive_block_from_maf_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    subset_plan: Option<&SubsetDecodePlan>,
    packed_row_indices: Option<&[usize]>,
    row_start: usize,
    row_end: usize,
    scratch_row_indices: &mut [usize],
    scratch_mean: &mut [f32],
    scratch_scale: &mut [f32],
    out_block: &mut [f32],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let cur_rows = row_end.saturating_sub(row_start);
    prepare_packed_block_row_indices(packed_row_indices, row_start, cur_rows, scratch_row_indices)?;
    prepare_packed_block_centered_mean_scale_f32(
        row_maf,
        row_start,
        cur_rows,
        scratch_mean,
        scratch_scale,
    )?;
    decode_prepared_additive_block_packed_f32(
        packed_flat,
        bytes_per_snp,
        n_samples,
        &row_flip[row_start..row_end],
        &scratch_mean[..cur_rows],
        &scratch_scale[..cur_rows],
        sample_idx,
        full_sample_fast,
        subset_plan,
        Some(&scratch_row_indices[..cur_rows]),
        0usize,
        cur_rows,
        out_block,
        pool,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_standardized_additive_block_from_maf_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    subset_plan: Option<&SubsetDecodePlan>,
    packed_row_indices: Option<&[usize]>,
    row_start: usize,
    row_end: usize,
    eps: f32,
    scratch_row_indices: &mut [usize],
    scratch_mean: &mut [f32],
    scratch_scale: &mut [f32],
    out_block: &mut [f32],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let cur_rows = row_end.saturating_sub(row_start);
    prepare_packed_block_row_indices(packed_row_indices, row_start, cur_rows, scratch_row_indices)?;
    prepare_packed_block_standardized_mean_scale_f32(
        row_maf,
        row_start,
        cur_rows,
        eps,
        scratch_mean,
        scratch_scale,
    )?;
    decode_prepared_additive_block_packed_f32(
        packed_flat,
        bytes_per_snp,
        n_samples,
        &row_flip[row_start..row_end],
        &scratch_mean[..cur_rows],
        &scratch_scale[..cur_rows],
        sample_idx,
        full_sample_fast,
        subset_plan,
        Some(&scratch_row_indices[..cur_rows]),
        0usize,
        cur_rows,
        out_block,
        pool,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_additive_grm_block_f32(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    method: usize,
    eps: f32,
    row_start: usize,
    row_end: usize,
    n_out: usize,
    out_block: &mut [f32],
    out_varsum: &mut [f64],
    out_rowsum: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let cur_rows = row_end.saturating_sub(row_start);
    if cur_rows == 0 {
        return Ok(());
    }
    if out_block.len() < cur_rows.saturating_mul(n_out) {
        return Err("decode_additive_grm_block_f32: out_block too small".to_string());
    }
    if out_varsum.len() < cur_rows {
        return Err("decode_additive_grm_block_f32: out_varsum too small".to_string());
    }
    if out_rowsum.len() < cur_rows {
        return Err("decode_additive_grm_block_f32: out_rowsum too small".to_string());
    }

    let code4_lut = &packed_byte_lut().code4;
    let block = &mut out_block[..cur_rows * n_out];
    let varsum = &mut out_varsum[..cur_rows];
    let rowsum = &mut out_rowsum[..cur_rows];

    if method == 2 {
        let subset_plan = if full_sample_fast {
            None
        } else {
            Some(SubsetDecodePlan::from_sample_idx_with_n_samples(
                sample_idx, n_samples,
            ))
        };
        let mut scratch_row_indices = vec![0usize; cur_rows];
        let mut scratch_mean = vec![0.0_f32; cur_rows];
        let mut scratch_scale = vec![0.0_f32; cur_rows];
        decode_standardized_additive_block_from_maf_f32(
            packed_flat,
            bytes_per_snp,
            n_samples,
            row_flip,
            row_maf,
            sample_idx,
            full_sample_fast,
            subset_plan.as_ref(),
            None,
            row_start,
            row_end,
            eps,
            scratch_row_indices.as_mut_slice(),
            scratch_mean.as_mut_slice(),
            scratch_scale.as_mut_slice(),
            block,
            pool,
        )?;
        for off in 0..cur_rows {
            let p = row_maf[row_start + off].clamp(0.0_f32, 1.0_f32) as f64;
            varsum[off] = 0.0_f64;
            rowsum[off] = 2.0_f64 * p * (n_out as f64);
        }
        return Ok(());
    }

    let mut decode_run = || {
        if full_sample_fast {
            block
                .par_chunks_mut(n_out)
                .zip(varsum.par_iter_mut())
                .zip(rowsum.par_iter_mut())
                .enumerate()
                .for_each(|(off, ((out_row, row_varsum_dst), row_sum_dst))| {
                    let idx = row_start + off;
                    let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                    let p = row_maf[idx].clamp(0.0_f32, 1.0_f32);
                    let mean_g = 2.0_f32 * p;
                    let var = 2.0_f32 * p * (1.0_f32 - p);
                    let std_scale = if method == 2 {
                        if var > eps {
                            1.0_f32 / var.sqrt()
                        } else {
                            0.0_f32
                        }
                    } else {
                        1.0_f32
                    };
                    let value_lut: [f32; 4] = if row_flip[idx] {
                        [
                            (2.0_f32 - mean_g) * std_scale,
                            0.0_f32,
                            (1.0_f32 - mean_g) * std_scale,
                            (0.0_f32 - mean_g) * std_scale,
                        ]
                    } else {
                        [
                            (0.0_f32 - mean_g) * std_scale,
                            0.0_f32,
                            (1.0_f32 - mean_g) * std_scale,
                            (2.0_f32 - mean_g) * std_scale,
                        ]
                    };
                    decode_row_centered_full_lut(row, n_samples, code4_lut, &value_lut, out_row);
                    if method == 1 {
                        *row_varsum_dst = var as f64;
                    }
                    *row_sum_dst = (mean_g as f64) * (n_out as f64);
                });
        } else {
            block
                .par_chunks_mut(n_out)
                .zip(varsum.par_iter_mut())
                .zip(rowsum.par_iter_mut())
                .enumerate()
                .for_each_init(
                    || vec![0.0_f32; n_samples],
                    |full_row, (off, ((out_row, row_varsum_dst), row_sum_dst))| {
                        let idx = row_start + off;
                        let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                        let p = row_maf[idx].clamp(0.0_f32, 1.0_f32);
                        let default_mean_g = 2.0_f32 * p;
                        let (var_centered, row_sum) = decode_subset_row_from_full_scratch(
                            row,
                            n_samples,
                            sample_idx,
                            row_flip[idx],
                            default_mean_g,
                            method,
                            eps,
                            code4_lut,
                            full_row.as_mut_slice(),
                            out_row,
                        );
                        if method == 1 {
                            *row_varsum_dst = var_centered;
                        }
                        *row_sum_dst = row_sum;
                    },
                );
        }
    };

    if let Some(tp) = pool {
        tp.install(&mut decode_run);
    } else {
        decode_run();
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_additive_grm_block_f64(
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    sample_idx: &[usize],
    full_sample_fast: bool,
    method: usize,
    eps: f64,
    row_start: usize,
    row_end: usize,
    n_out: usize,
    out_block: &mut [f64],
    out_varsum: &mut [f64],
    out_rowsum: &mut [f64],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<(), String> {
    let cur_rows = row_end.saturating_sub(row_start);
    if cur_rows == 0 {
        return Ok(());
    }
    if out_block.len() < cur_rows.saturating_mul(n_out) {
        return Err("decode_additive_grm_block_f64: out_block too small".to_string());
    }
    if out_varsum.len() < cur_rows {
        return Err("decode_additive_grm_block_f64: out_varsum too small".to_string());
    }
    if out_rowsum.len() < cur_rows {
        return Err("decode_additive_grm_block_f64: out_rowsum too small".to_string());
    }

    let code4_lut = &packed_byte_lut().code4;
    let block = &mut out_block[..cur_rows * n_out];
    let varsum = &mut out_varsum[..cur_rows];
    let rowsum = &mut out_rowsum[..cur_rows];

    let mut decode_run = || {
        if full_sample_fast {
            block
                .par_chunks_mut(n_out)
                .zip(varsum.par_iter_mut())
                .zip(rowsum.par_iter_mut())
                .enumerate()
                .for_each(|(off, ((out_row, row_varsum_dst), row_sum_dst))| {
                    let idx = row_start + off;
                    let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                    let p = row_maf[idx].clamp(0.0_f32, 1.0_f32) as f64;
                    let mean_g = 2.0_f64 * p;
                    let var = 2.0_f64 * p * (1.0_f64 - p);
                    let std_scale = if method == 2 {
                        if var > eps {
                            1.0_f64 / var.sqrt()
                        } else {
                            0.0_f64
                        }
                    } else {
                        1.0_f64
                    };
                    let value_lut: [f64; 4] = if row_flip[idx] {
                        [
                            (2.0_f64 - mean_g) * std_scale,
                            0.0_f64,
                            (1.0_f64 - mean_g) * std_scale,
                            (0.0_f64 - mean_g) * std_scale,
                        ]
                    } else {
                        [
                            (0.0_f64 - mean_g) * std_scale,
                            0.0_f64,
                            (1.0_f64 - mean_g) * std_scale,
                            (2.0_f64 - mean_g) * std_scale,
                        ]
                    };
                    decode_row_centered_full_lut_f64(
                        row, n_samples, code4_lut, &value_lut, out_row,
                    );
                    if method == 1 {
                        *row_varsum_dst = var;
                    }
                    *row_sum_dst = mean_g * (n_out as f64);
                });
        } else {
            block
                .par_chunks_mut(n_out)
                .zip(varsum.par_iter_mut())
                .zip(rowsum.par_iter_mut())
                .enumerate()
                .for_each_init(
                    || vec![0.0_f64; n_samples],
                    |full_row, (off, ((out_row, row_varsum_dst), row_sum_dst))| {
                        let idx = row_start + off;
                        let row = &packed_flat[idx * bytes_per_snp..(idx + 1) * bytes_per_snp];
                        let p = row_maf[idx].clamp(0.0_f32, 1.0_f32) as f64;
                        let default_mean_g = 2.0_f64 * p;
                        let (var_centered, row_sum) = decode_subset_row_from_full_scratch_f64(
                            row,
                            n_samples,
                            sample_idx,
                            row_flip[idx],
                            default_mean_g,
                            method,
                            eps,
                            code4_lut,
                            full_row.as_mut_slice(),
                            out_row,
                        );
                        if method == 1 {
                            *row_varsum_dst = var_centered;
                        }
                        *row_sum_dst = row_sum;
                    },
                );
        }
    };

    if let Some(tp) = pool {
        tp.install(&mut decode_run);
    } else {
        decode_run();
    }
    Ok(())
}
