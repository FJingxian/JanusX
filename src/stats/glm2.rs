use nalgebra::DMatrix;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rayon::prelude::*;
use std::borrow::Cow;
use std::fmt::Write as _;
use std::sync::Arc;

use crate::assoc2tsv::AssocTsvSink;
use crate::decode::{
    decode_packed_row_model_enum_into_f64, AdditiveDecodePlan, PackedGeneticModel,
};
use crate::gfcore::{self, BimChunkReader};
use crate::gfreader::{
    build_sample_selection, count_packed_row_counts, count_packed_row_counts_selected_with_excluded,
};
use crate::glm::{
    is_simple_snp_allele, lm_resolve_snp_name, normalize_plink_prefix_local, student_t_p_two_sided,
    LmQrProjection,
};
use crate::gload::WindowedBedMatrix;
use crate::linalg::{chi2_sf, format_chisq_value, sanitize_assoc_pvalue};
use crate::stats_common::{get_cached_pool, parse_index_vec_i64};

fn pinv_svd_square(a: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let q = a.nrows();
    if q != a.ncols() {
        return Err("pinv_svd_square expects a square matrix".to_string());
    }
    let svd = a.clone().svd(true, true);
    let u = svd.u.ok_or_else(|| "SVD failed to produce U".to_string())?;
    let vt = svd
        .v_t
        .ok_or_else(|| "SVD failed to produce V^T".to_string())?;
    let mut s_inv = DMatrix::<f64>::zeros(q, q);
    let smax = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let rcond = 1e-12_f64;
    let cutoff = rcond * smax.max(1.0);
    for i in 0..q {
        let s = svd.singular_values[i];
        if s.is_finite() && s > cutoff {
            s_inv[(i, i)] = 1.0 / s;
        }
    }
    let v = vt.transpose();
    Ok(v * s_inv * u.transpose())
}

fn matrix_inverse_or_pinv(a: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    if let Some(inv) = a.clone().try_inverse() {
        return Ok(inv);
    }
    pinv_svd_square(a)
}

fn lm2_header(cov_indices: &[usize]) -> Vec<u8> {
    let mut header =
        String::from("chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald");
    for &idx in cov_indices {
        let _ = write!(&mut header, "\tbeta_i{idx}\tse_i{idx}\tpwald_i{idx}");
    }
    header.push_str("\tchisq_int_joint\tp_int_joint\tchisq_joint\tp_joint");
    header.push('\n');
    header.into_bytes()
}

struct Lm2CoefResult {
    beta: f64,
    se: f64,
    chisq: f64,
    pwald: f64,
}

struct Lm2JointResult {
    chisq: f64,
    pvalue: f64,
}

struct Lm2FitResult {
    snp_stat: Lm2CoefResult,
    interaction_stats: Vec<Lm2CoefResult>,
    interaction_joint: Lm2JointResult,
    full_joint: Lm2JointResult,
}

struct Lm2BaseCache {
    q_rank: usize,
    n_interactions: usize,
    qr: LmQrProjection,
    df: i32,
}

struct Lm2Scratch {
    z_row: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
    e: Vec<f64>,
}

impl Lm2Scratch {
    fn new(q_base: usize, n_interactions: usize) -> Self {
        let m = 1usize + n_interactions;
        Self {
            z_row: vec![0.0_f64; m],
            c: vec![0.0_f64; q_base * m],
            d: vec![0.0_f64; m * m],
            e: vec![0.0_f64; m],
        }
    }

    fn reset(&mut self) {
        self.z_row.fill(0.0);
        self.c.fill(0.0);
        self.d.fill(0.0);
        self.e.fill(0.0);
    }
}

struct Lm2ThreadState {
    g_raw: Vec<f64>,
    scratch: Lm2Scratch,
    row_text: String,
}

impl Lm2ThreadState {
    fn new(n: usize, q_rank: usize, n_interactions: usize) -> Self {
        Self {
            g_raw: vec![0.0_f64; n],
            scratch: Lm2Scratch::new(q_rank, n_interactions),
            row_text: String::with_capacity(256 + 96 * n_interactions),
        }
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn lm2_precompute_base_cache(
    x_base: &[f64],
    y: &[f64],
    n: usize,
    q_base: usize,
    n_interactions: usize,
) -> Result<Lm2BaseCache, String> {
    let m = 1usize + n_interactions;
    let p = q_base + m;
    if n <= p {
        return Err(format!(
            "n too small: require n > q_base + 1 + n_interactions, got n={n}, q_base={q_base}, n_interactions={n_interactions}"
        ));
    }
    let qr = LmQrProjection::from_design(x_base, y, n, q_base)?;
    Ok(Lm2BaseCache {
        q_rank: qr.rank(),
        n_interactions,
        qr,
        df: (n as i32) - (p as i32),
    })
}

fn lm2_fit_single_snp(
    g_raw: &[f64],
    cov_selected: &[f64],
    n: usize,
    base_cache: &Lm2BaseCache,
    scratch: &mut Lm2Scratch,
) -> Result<Lm2FitResult, String> {
    let q_rank = base_cache.q_rank;
    let n_interactions = base_cache.n_interactions;
    let m = 1usize + n_interactions;
    if g_raw.len() != n {
        return Err("LM2 genotype length mismatch".to_string());
    }
    if cov_selected.len() != n.saturating_mul(n_interactions) {
        return Err("LM2 selected covariates shape mismatch".to_string());
    }
    if base_cache.df <= 0 {
        return Ok(Lm2FitResult {
            snp_stat: Lm2CoefResult {
                beta: f64::NAN,
                se: f64::NAN,
                chisq: f64::NAN,
                pwald: 1.0,
            },
            interaction_stats: (0..n_interactions)
                .map(|_| Lm2CoefResult {
                    beta: f64::NAN,
                    se: f64::NAN,
                    chisq: f64::NAN,
                    pwald: 1.0,
                })
                .collect(),
            interaction_joint: Lm2JointResult {
                chisq: f64::NAN,
                pvalue: 1.0,
            },
            full_joint: Lm2JointResult {
                chisq: f64::NAN,
                pvalue: 1.0,
            },
        });
    }

    let q = base_cache.qr.q();
    let y_resid = base_cache.qr.y_resid();
    scratch.reset();
    for r in 0..n {
        let g = g_raw[r];
        let z_row = &mut scratch.z_row;
        z_row[0] = g;
        for j in 0..n_interactions {
            let cv = cov_selected[r * n_interactions + j];
            z_row[1 + j] = g * cv;
        }
        let yv = y_resid[r];
        let q_row = &q[r * q_rank..(r + 1) * q_rank];
        for a in 0..m {
            let za = z_row[a];
            scratch.e[a] += za * yv;
            for qj in 0..q_rank {
                scratch.c[qj * m + a] += q_row[qj] * za;
            }
            for b in 0..=a {
                scratch.d[a * m + b] += za * z_row[b];
            }
        }
    }
    for a in 0..m {
        for b in 0..a {
            scratch.d[b * m + a] = scratch.d[a * m + b];
        }
    }

    let c_mat = DMatrix::<f64>::from_row_slice(q_rank, m, &scratch.c);
    let d_mat = DMatrix::<f64>::from_row_slice(m, m, &scratch.d);
    let schur = d_mat - c_mat.transpose() * &c_mat;
    let schur_inv = matrix_inverse_or_pinv(&schur)?;
    let rhs = DMatrix::<f64>::from_column_slice(m, 1, &scratch.e);
    let beta = &schur_inv * &rhs;
    let beta_vals = beta.as_slice();
    let rss = (base_cache.qr.rss0() - dot(&scratch.e, beta_vals)).max(0.0);
    let sigma2 = rss / (base_cache.df as f64);

    let coef_stat = |coef_idx: usize| -> Lm2CoefResult {
        let b = beta[(coef_idx, 0)];
        let var = sigma2 * schur_inv[(coef_idx, coef_idx)];
        let se = if var.is_finite() && var > 0.0 {
            var.sqrt()
        } else {
            f64::NAN
        };
        let chisq = if b.is_finite() && se.is_finite() && se > 0.0 {
            let z = b / se;
            z * z
        } else {
            f64::NAN
        };
        let pwald = if b.is_finite() && se.is_finite() && se > 0.0 {
            let t = b / se;
            let p_raw = student_t_p_two_sided(t, base_cache.df);
            sanitize_assoc_pvalue(b, se, p_raw)
        } else {
            1.0
        };
        Lm2CoefResult {
            beta: b,
            se,
            chisq,
            pwald,
        }
    };

    let snp_stat = coef_stat(0);
    let mut out = Vec::<Lm2CoefResult>::with_capacity(n_interactions);
    for coef_idx in 1..m {
        out.push(coef_stat(coef_idx));
    }

    let interaction_joint = if n_interactions > 0 && sigma2.is_finite() && sigma2 > 0.0 {
        let mut cov_block = vec![0.0_f64; n_interactions * n_interactions];
        for i in 0..n_interactions {
            for j in 0..n_interactions {
                cov_block[i * n_interactions + j] = schur_inv[(1 + i, 1 + j)];
            }
        }
        let cov_block_mat =
            DMatrix::<f64>::from_row_slice(n_interactions, n_interactions, &cov_block);
        let cov_block_inv = matrix_inverse_or_pinv(&cov_block_mat)?;
        let beta_int = DMatrix::<f64>::from_column_slice(n_interactions, 1, &beta_vals[1..m]);
        let stat = ((beta_int.transpose() * &cov_block_inv * beta_int)[(0, 0)] / sigma2).max(0.0);
        Lm2JointResult {
            chisq: stat,
            pvalue: chi2_sf(stat, n_interactions as f64),
        }
    } else {
        Lm2JointResult {
            chisq: f64::NAN,
            pvalue: 1.0,
        }
    };

    let full_joint = if sigma2.is_finite() && sigma2 > 0.0 {
        let stat = (dot(&scratch.e, beta_vals) / sigma2).max(0.0);
        Lm2JointResult {
            chisq: stat,
            pvalue: chi2_sf(stat, m as f64),
        }
    } else {
        Lm2JointResult {
            chisq: f64::NAN,
            pvalue: 1.0,
        }
    };

    Ok(Lm2FitResult {
        snp_stat,
        interaction_stats: out,
        interaction_joint,
        full_joint,
    })
}

fn append_lm2_row_text(
    text_buf: &mut String,
    chrom: &str,
    pos: i64,
    snp: &str,
    allele0: &str,
    allele1: &str,
    maf: f32,
    miss_count: usize,
    fit: &Lm2FitResult,
) {
    let _ = write!(
        text_buf,
        "{chrom}\t{pos}\t{snp}\t{allele0}\t{allele1}\t{maf:.4}\t{miss_count}\t{:.4}\t{:.4}\t{}\t{:.4e}",
        fit.snp_stat.beta,
        fit.snp_stat.se,
        format_chisq_value(fit.snp_stat.chisq),
        fit.snp_stat.pwald,
    );
    for row in fit.interaction_stats.iter() {
        let _ = write!(
            text_buf,
            "\t{:.4}\t{:.4}\t{:.4e}",
            row.beta, row.se, row.pwald,
        );
    }
    let _ = write!(
        text_buf,
        "\t{}\t{:.4e}\t{}\t{:.4e}",
        format_chisq_value(fit.interaction_joint.chisq),
        fit.interaction_joint.pvalue,
        format_chisq_value(fit.full_joint.chisq),
        fit.full_joint.pvalue,
    );
    text_buf.push('\n');
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    x,
    cov_all,
    cov_indices,
    out_tsv,
    sample_ids=None,
    row_indices=None,
    row_flip=None,
    row_missing=None,
    row_maf=None,
    maf_threshold=0.0,
    max_missing_rate=1.0,
    genetic_model="add",
    het_threshold=0.02,
    snps_only=false,
    chunk_size=10000,
    threads=0,
    progress_callback=None,
    progress_every=0,
    mmap_window_mb=None,
))]
pub fn lm2_stream_bed_to_tsv(
    py: Python<'_>,
    prefix: String,
    y: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    cov_all: PyReadonlyArray2<'_, f64>,
    cov_indices: PyReadonlyArray1<'_, i64>,
    out_tsv: String,
    sample_ids: Option<Vec<String>>,
    row_indices: Option<PyReadonlyArray1<'_, i64>>,
    row_flip: Option<PyReadonlyArray1<'_, bool>>,
    row_missing: Option<PyReadonlyArray1<'_, f32>>,
    row_maf: Option<PyReadonlyArray1<'_, f32>>,
    maf_threshold: f32,
    max_missing_rate: f32,
    genetic_model: &str,
    het_threshold: f32,
    snps_only: bool,
    chunk_size: usize,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    mmap_window_mb: Option<usize>,
) -> PyResult<(usize, usize)> {
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size must be > 0"));
    }
    if !(0.0..=0.5).contains(&maf_threshold) {
        return Err(PyValueError::new_err(
            "maf_threshold must be within [0, 0.5]",
        ));
    }
    if !(0.0..=1.0).contains(&max_missing_rate) {
        return Err(PyValueError::new_err(
            "max_missing_rate must be within [0, 1.0]",
        ));
    }
    if !(0.0..=1.0).contains(&het_threshold) {
        return Err(PyValueError::new_err(
            "het_threshold must be within [0, 1.0]",
        ));
    }
    let gm = PackedGeneticModel::parse(genetic_model)?;

    let y = y.as_slice()?;
    let x_arr = x.as_array();
    let cov_arr = cov_all.as_array();
    let cov_idx = cov_indices.as_slice()?;
    let n = y.len();
    let (xn, q_base) = (x_arr.shape()[0], x_arr.shape()[1]);
    if xn != n {
        return Err(PyRuntimeError::new_err("X.n_rows must equal len(y)"));
    }
    if cov_arr.shape()[0] != n {
        return Err(PyRuntimeError::new_err("cov_all.n_rows must equal len(y)"));
    }
    let n_cov_all = cov_arr.shape()[1];
    if n_cov_all == 0 {
        return Err(PyRuntimeError::new_err(
            "LM2 requires cov_all with at least one column",
        ));
    }
    let mut cov_pick: Vec<usize> = Vec::with_capacity(cov_idx.len());
    for &idx in cov_idx {
        if idx < 0 {
            return Err(PyRuntimeError::new_err("cov_indices must be >= 0"));
        }
        let u = idx as usize;
        if u >= n_cov_all {
            return Err(PyRuntimeError::new_err(format!(
                "cov_indices out of range: {u} >= {n_cov_all}"
            )));
        }
        cov_pick.push(u);
    }
    if cov_pick.is_empty() {
        return Err(PyRuntimeError::new_err(
            "LM2 requires at least one explicitly selected covariate column.",
        ));
    }
    let n_interactions = cov_pick.len();
    if n <= q_base + 1 + n_interactions {
        return Err(PyRuntimeError::new_err(format!(
            "n too small: require n > q_base + 1 + n_interactions, got n={n}, q_base={q_base}, n_interactions={n_interactions}"
        )));
    }

    let x_flat: Cow<[f64]> = match x.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(x_arr.iter().copied().collect()),
    };
    let base_cache = lm2_precompute_base_cache(x_flat.as_ref(), y, n, q_base, n_interactions)
        .map_err(PyRuntimeError::new_err)?;
    let mut cov_selected = vec![0.0_f64; n * n_interactions];
    for r in 0..n {
        for (j, &src_j) in cov_pick.iter().enumerate() {
            cov_selected[r * n_interactions + j] = cov_arr[(r, src_j)];
        }
    }
    let prepared_meta_input = match (row_indices, row_flip, row_missing, row_maf) {
        (None, None, None, None) => None,
        (Some(row_idx), Some(row_flip), Some(row_missing), Some(row_maf)) => {
            let row_idx_vec = row_idx.as_slice()?.to_vec();
            let row_flip_vec = match row_flip.as_slice() {
                Ok(slc) => slc.to_vec(),
                Err(_) => row_flip.as_array().iter().copied().collect(),
            };
            let row_missing_vec = match row_missing.as_slice() {
                Ok(slc) => slc.to_vec(),
                Err(_) => row_missing.as_array().iter().copied().collect(),
            };
            let row_maf_vec = match row_maf.as_slice() {
                Ok(slc) => slc.to_vec(),
                Err(_) => row_maf.as_array().iter().copied().collect(),
            };
            Some((row_idx_vec, row_flip_vec, row_missing_vec, row_maf_vec))
        }
        _ => {
            return Err(PyRuntimeError::new_err(
                "prepared row metadata must provide all or none of: row_indices, row_flip, row_missing, row_maf",
            ))
        }
    };

    let norm_prefix = normalize_plink_prefix_local(&prefix);
    let samples = gfcore::read_fam(&norm_prefix).map_err(PyRuntimeError::new_err)?;
    let (sample_indices, _sample_ids_ordered) =
        build_sample_selection(&samples, sample_ids, None).map_err(PyRuntimeError::new_err)?;
    if sample_indices.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "sample_ids length mismatch: selected n={} but len(y)={n}",
            sample_indices.len()
        )));
    }

    let pool: Option<Arc<rayon::ThreadPool>> = get_cached_pool(threads)?;
    let sample_plan =
        AdditiveDecodePlan::from_sample_indices(samples.len(), sample_indices.as_slice());
    let sample_identity = sample_plan.sample_identity();
    let sample_byte_idx = sample_plan.sample_byte_idx();
    let sample_bit_shift = sample_plan.sample_bit_shift();
    let selected_excluded_sample_indices = sample_plan.selected_excluded_sample_indices();

    let out_path = out_tsv.clone();
    py.detach(move || -> PyResult<(usize, usize)> {
        let mut bim_reader = BimChunkReader::open(&norm_prefix).map_err(PyRuntimeError::new_err)?;
        let mut bed_window =
            WindowedBedMatrix::open(&norm_prefix, mmap_window_mb.unwrap_or(512usize))
                .map_err(PyRuntimeError::new_err)?;
        let n_snps = bed_window.n_source_snps();
        let bytes_per_snp = bed_window.bytes_per_snp();
        let prepared_meta = match prepared_meta_input.as_ref() {
            None => None,
            Some((row_idx64, row_flip_vec, row_missing_vec, row_maf_vec)) => {
                let m = row_idx64.len();
                if row_flip_vec.len() != m
                    || row_missing_vec.len() != m
                    || row_maf_vec.len() != m
                {
                    return Err(PyRuntimeError::new_err(format!(
                        "prepared row metadata length mismatch: row_indices={m}, row_flip={}, row_missing={}, row_maf={}",
                        row_flip_vec.len(),
                        row_missing_vec.len(),
                        row_maf_vec.len(),
                    )));
                }
                let source_rows =
                    parse_index_vec_i64(row_idx64.as_slice(), n_snps, "row_indices")
                        .map_err(PyRuntimeError::new_err)?;
                let mut prev_src: Option<usize> = None;
                for &src_row in source_rows.iter() {
                    if let Some(prev) = prev_src {
                        if src_row < prev {
                            return Err(PyRuntimeError::new_err(
                                "prepared row_indices must be sorted in ascending BED order",
                            ));
                        }
                    }
                    prev_src = Some(src_row);
                }
                let miss_counts: Vec<usize> = row_missing_vec
                    .iter()
                    .map(|&v| {
                        if !v.is_finite() || v < 0.0_f32 {
                            0usize
                        } else {
                            v.round() as usize
                        }
                    })
                    .collect();
                Some((source_rows, row_flip_vec.clone(), miss_counts, row_maf_vec.clone()))
            }
        };
        let total_scan_units = prepared_meta
            .as_ref()
            .map(|(source_rows, _, _, _)| source_rows.len())
            .unwrap_or(n_snps);

        let mut sink = AssocTsvSink::with_config(out_path, 8 * 1024 * 1024, 16);
        sink.ensure_custom_header(lm2_header(cov_pick.as_slice()).as_slice())
            .map_err(PyRuntimeError::new_err)?;

        let progress_block = if progress_every == 0 {
            chunk_size.max(1)
        } else {
            progress_every.max(1)
        };
        let mut next_progress_emit = progress_block.min(total_scan_units);
        let mut kept_total = 0usize;
        let mut chunk_start = 0usize;
        let mut text = String::with_capacity(8 * 1024 * 1024);
        let mut g_raw = vec![0.0_f64; n];
        let mut lm2_scratch = Lm2Scratch::new(base_cache.q_rank, n_interactions);
        let mut rel_row_indices = Vec::<usize>::with_capacity(chunk_size.max(1));

        while chunk_start < total_scan_units {
            let chunk_end = (chunk_start + chunk_size).min(total_scan_units);
            let prepared_batch = prepared_meta
                .as_ref()
                .map(|(source_rows, row_flip_all, miss_count_all, row_maf_all)| {
                    (
                        &source_rows[chunk_start..chunk_end],
                        &row_flip_all[chunk_start..chunk_end],
                        &miss_count_all[chunk_start..chunk_end],
                        &row_maf_all[chunk_start..chunk_end],
                    )
                });
            let chunk_packed = if let Some((source_rows, _, _, _)) = prepared_batch.as_ref() {
                bed_window
                    .prepare_source_rows(source_rows, &mut rel_row_indices)
                    .map_err(PyRuntimeError::new_err)?
            } else {
                bed_window
                    .read_source_range(chunk_start, chunk_end)
                    .map_err(PyRuntimeError::new_err)?
            };
            let chunk_sites = if let Some((source_rows, _, _, _)) = prepared_batch.as_ref() {
                bim_reader
                    .read_selected_rows(source_rows)
                    .map_err(PyRuntimeError::new_err)?
            } else {
                bim_reader
                    .read_range(chunk_start, chunk_end)
                    .map_err(PyRuntimeError::new_err)?
            };

            let counts: Vec<Option<(f32, usize)>> = if let Some((_, _, miss_count_batch, row_maf_batch)) =
                prepared_batch.as_ref()
            {
                row_maf_batch
                    .iter()
                    .copied()
                    .zip(miss_count_batch.iter().copied())
                    .map(|(maf, missing_count)| Some((maf, missing_count)))
                    .collect()
            } else if let Some(tp) = pool.as_ref() {
                tp.install(|| {
                    (chunk_start..chunk_end)
                        .into_par_iter()
                        .map(|snp_idx| {
                            let local_idx = snp_idx - chunk_start;
                            let row = &chunk_packed
                                [local_idx * bytes_per_snp..(local_idx + 1) * bytes_per_snp];
                            let (missing, het, hom_alt) = if sample_identity {
                                count_packed_row_counts(row, samples.len())
                            } else {
                                count_packed_row_counts_selected_with_excluded(
                                    row,
                                    samples.len(),
                                    sample_indices.as_slice(),
                                    selected_excluded_sample_indices,
                                )
                            };
                            let non_missing = n.saturating_sub(missing);
                            let miss_rate = if n > 0 {
                                missing as f32 / n as f32
                            } else {
                                1.0
                            };
                            if miss_rate > max_missing_rate {
                                return None;
                            }
                            if non_missing == 0 {
                                return None;
                            }
                            let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                            let alt_freq = alt_sum as f32 / (2.0_f32 * non_missing as f32);
                            let maf = alt_freq.min(1.0_f32 - alt_freq);
                            if maf < maf_threshold {
                                return None;
                            }
                            Some((alt_freq, missing))
                        })
                        .collect()
                })
            } else {
                (chunk_start..chunk_end)
                    .map(|snp_idx| {
                        let local_idx = snp_idx - chunk_start;
                        let row = &chunk_packed
                            [local_idx * bytes_per_snp..(local_idx + 1) * bytes_per_snp];
                        let (missing, het, hom_alt) = if sample_identity {
                            count_packed_row_counts(row, samples.len())
                        } else {
                            count_packed_row_counts_selected_with_excluded(
                                row,
                                samples.len(),
                                sample_indices.as_slice(),
                                selected_excluded_sample_indices,
                            )
                        };
                        let non_missing = n.saturating_sub(missing);
                        let miss_rate = if n > 0 {
                            missing as f32 / n as f32
                        } else {
                            1.0
                        };
                        if miss_rate > max_missing_rate || non_missing == 0 {
                            return None;
                        }
                        let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                        let alt_freq = alt_sum as f32 / (2.0_f32 * non_missing as f32);
                        let maf = alt_freq.min(1.0_f32 - alt_freq);
                        if maf < maf_threshold {
                            return None;
                        }
                        Some((alt_freq, missing))
                    })
                    .collect()
            };

            text.clear();
            let mut chunk_kept = 0usize;
            if let Some(tp) = pool.as_ref() {
                let row_texts = tp.install(|| {
                    (0..chunk_sites.len())
                        .into_par_iter()
                        .map_init(
                            || Lm2ThreadState::new(n, base_cache.q_rank, n_interactions),
                            |state, offset| -> Result<Option<String>, String> {
                                let Some((maf, missing_count)) = counts[offset] else {
                                    return Ok(None);
                                };
                                let site = &chunk_sites[offset];
                                if snps_only
                                    && (!is_simple_snp_allele(&site.ref_allele)
                                        || !is_simple_snp_allele(&site.alt_allele))
                                {
                                    return Ok(None);
                                }
                                let row_local_idx = if prepared_batch.is_some() {
                                    rel_row_indices[offset]
                                } else {
                                    offset
                                };
                                let row = &chunk_packed[row_local_idx * bytes_per_snp
                                    ..(row_local_idx + 1) * bytes_per_snp];
                                decode_packed_row_model_enum_into_f64(
                                    row,
                                    prepared_batch
                                        .as_ref()
                                        .map(|(_, row_flip_batch, _, _)| row_flip_batch[offset])
                                        .unwrap_or(false),
                                    maf,
                                    n,
                                    gm,
                                    sample_identity,
                                    sample_byte_idx,
                                    sample_bit_shift,
                                    &mut state.g_raw,
                                );
                                let fit = lm2_fit_single_snp(
                                    state.g_raw.as_slice(),
                                    cov_selected.as_slice(),
                                    n,
                                    &base_cache,
                                    &mut state.scratch,
                                )?;
                                state.row_text.clear();
                                let snp_name =
                                    lm_resolve_snp_name(&site.snp, &site.chrom, site.pos);
                                append_lm2_row_text(
                                    &mut state.row_text,
                                    site.chrom.as_str(),
                                    site.pos as i64,
                                    snp_name.as_str(),
                                    site.ref_allele.as_str(),
                                    site.alt_allele.as_str(),
                                    maf,
                                    missing_count,
                                    &fit,
                                );
                                Ok(Some(std::mem::take(&mut state.row_text)))
                            },
                        )
                        .collect::<Vec<_>>()
                });
                for row_text in row_texts {
                    if let Some(row) = row_text.map_err(PyRuntimeError::new_err)? {
                        text.push_str(&row);
                        chunk_kept = chunk_kept.saturating_add(1);
                    }
                }
            } else {
                for (offset, (site, counts_opt)) in
                    chunk_sites.iter().zip(counts.iter()).enumerate()
                {
                    let Some((maf, missing_count)) = counts_opt else {
                        continue;
                    };
                    if snps_only
                        && (!is_simple_snp_allele(&site.ref_allele)
                            || !is_simple_snp_allele(&site.alt_allele))
                    {
                        continue;
                    }
                    let row_local_idx = if prepared_batch.is_some() {
                        rel_row_indices[offset]
                    } else {
                        offset
                    };
                    let row = &chunk_packed
                        [row_local_idx * bytes_per_snp..(row_local_idx + 1) * bytes_per_snp];
                    decode_packed_row_model_enum_into_f64(
                        row,
                        prepared_batch
                            .as_ref()
                            .map(|(_, row_flip_batch, _, _)| row_flip_batch[offset])
                            .unwrap_or(false),
                        *maf,
                        n,
                        gm,
                        sample_identity,
                        sample_byte_idx,
                        sample_bit_shift,
                        &mut g_raw,
                    );
                    let fit = lm2_fit_single_snp(
                        g_raw.as_slice(),
                        cov_selected.as_slice(),
                        n,
                        &base_cache,
                        &mut lm2_scratch,
                    )
                    .map_err(PyRuntimeError::new_err)?;
                    let snp_name = lm_resolve_snp_name(&site.snp, &site.chrom, site.pos);
                    append_lm2_row_text(
                        &mut text,
                        site.chrom.as_str(),
                        site.pos as i64,
                        snp_name.as_str(),
                        site.ref_allele.as_str(),
                        site.alt_allele.as_str(),
                        *maf,
                        *missing_count,
                        &fit,
                    );
                    chunk_kept = chunk_kept.saturating_add(1);
                }
            }
            kept_total = kept_total.saturating_add(chunk_kept);
            sink.add_rows(chunk_kept);
            sink.flush_external_text(&mut text)
                .map_err(PyRuntimeError::new_err)?;

            let scanned_total = chunk_end;
            if scanned_total >= next_progress_emit {
                if let Some(cb) = progress_callback.as_ref() {
                    let _ = Python::attach(|py2| -> PyResult<()> {
                        py2.check_signals()?;
                        cb.call1(
                            py2,
                            (
                                scanned_total.min(total_scan_units),
                                total_scan_units,
                            ),
                        )?;
                        Ok(())
                    });
                }
                next_progress_emit = (scanned_total / progress_block + 1)
                    .saturating_mul(progress_block)
                    .min(total_scan_units);
            }
            chunk_start = chunk_end;
        }

        if let Some(cb) = progress_callback.as_ref() {
            let _ = Python::attach(|py2| -> PyResult<()> {
                py2.check_signals()?;
                cb.call1(py2, (total_scan_units, total_scan_units))?;
                Ok(())
            });
        }
        sink.finish().map_err(PyRuntimeError::new_err)?;
        Ok((kept_total, total_scan_units))
    })
}
