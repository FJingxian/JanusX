use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::exceptions::*;
use pyo3::prelude::*;
use pyo3::BoundObject;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::gfreader::SiteInfo;

#[inline]
fn sim_draw_hwe(maf: f32, rng: &mut StdRng) -> i8 {
    let p0 = (1.0_f32 - maf) * (1.0_f32 - maf);
    let p1 = p0 + 2.0_f32 * maf * (1.0_f32 - maf);
    let u = rng.random::<f32>();
    if u < p0 {
        0
    } else if u < p1 {
        1
    } else {
        2
    }
}

#[inline]
fn sim_draw_homo(maf: f32, rng: &mut StdRng) -> i8 {
    if rng.random::<f32>() < maf {
        2
    } else {
        0
    }
}

#[inline]
fn sim_row_seed(base_seed: u64, row_idx: usize) -> u64 {
    let mut x = base_seed ^ (row_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15_u64);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9_u64);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB_u64);
    x ^ (x >> 31)
}

#[inline]
fn sim_generate_beta(beta_rng: &mut StdRng, beta_sd: f32, m: usize) -> Vec<f32> {
    let mut beta = vec![0.0_f32; m];
    if beta_sd <= 0.0 || m == 0 {
        return beta;
    }

    let mut i = 0usize;
    while i + 1 < m {
        let u1 = beta_rng
            .random::<f32>()
            .clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON);
        let u2 = beta_rng.random::<f32>();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0_f32 * std::f32::consts::PI * u2;
        beta[i] = beta_sd * r * theta.cos();
        beta[i + 1] = beta_sd * r * theta.sin();
        i += 2;
    }
    if i < m {
        let u1 = beta_rng
            .random::<f32>()
            .clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON);
        let u2 = beta_rng.random::<f32>();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0_f32 * std::f32::consts::PI * u2;
        beta[i] = beta_sd * r * theta.cos();
    }
    beta
}

#[inline]
fn sim_apply_trait_to_y(y_slice: &mut [f32], g_data: &[i8], beta: &[f32], n: usize) {
    if beta.is_empty() || y_slice.is_empty() {
        return;
    }
    y_slice.par_iter_mut().enumerate().for_each(|(j, yj)| {
        let mut acc = 0.0_f32;
        let mut off = j;
        for &b in beta.iter() {
            acc += (g_data[off] as f32) * b;
            off += n;
        }
        *yj += acc;
    });
}

#[inline]
fn sim_generate_chunk_data(
    mafs: &[f32],
    base_row_idx: usize,
    nidv: usize,
    n_families: usize,
    n_unrelated: usize,
    sire_idx: &[usize],
    dam_idx: &[usize],
    child_groups: &[Vec<usize>],
    unrelated_idx: &[usize],
    homo: bool,
    row_seed_base: u64,
) -> Result<(Vec<i8>, Vec<SiteInfo>), String> {
    let m = mafs.len();
    let mut data = vec![0_i8; m * nidv];
    data.par_chunks_mut(nidv).enumerate().for_each(|(r, row)| {
        let maf = mafs[r];
        let mut rr = StdRng::seed_from_u64(sim_row_seed(row_seed_base, base_row_idx + r));

        if n_families > 0 {
            for fi in 0..n_families {
                let sire = if homo {
                    sim_draw_homo(maf, &mut rr)
                } else {
                    sim_draw_hwe(maf, &mut rr)
                };
                let dam = if homo {
                    sim_draw_homo(maf, &mut rr)
                } else {
                    sim_draw_hwe(maf, &mut rr)
                };
                row[sire_idx[fi]] = sire;
                row[dam_idx[fi]] = dam;

                for grp in child_groups.iter() {
                    let child = if homo {
                        if rr.random::<f32>() < 0.5_f32 {
                            sire
                        } else {
                            dam
                        }
                    } else {
                        let pat = (rr.random::<f32>() * 2.0_f32) < (sire as f32);
                        let mat = (rr.random::<f32>() * 2.0_f32) < (dam as f32);
                        (pat as i8) + (mat as i8)
                    };
                    row[grp[fi]] = child;
                }
            }
        }

        if n_unrelated > 0 {
            for &idx in unrelated_idx.iter() {
                row[idx] = if homo {
                    sim_draw_homo(maf, &mut rr)
                } else {
                    sim_draw_hwe(maf, &mut rr)
                };
            }
        }
    });

    let sites = (0..m)
        .map(|i| SiteInfo {
            chrom: "1".to_string(),
            pos: (base_row_idx + i) as i32,
            ref_allele: "A".to_string(),
            alt_allele: "T".to_string(),
        })
        .collect::<Vec<SiteInfo>>();
    Ok((data, sites))
}

#[pyclass]
pub struct SimChunkGenerator {
    nsnp: usize,
    nidv: usize,
    chunk_size: usize,
    maf_low: f32,
    maf_high: f32,
    homo: bool,
    n_done: usize,
    mafs_rng: StdRng,
    row_seed_base: u64,
    n_families: usize,
    n_unrelated: usize,
    sire_idx: Vec<usize>,
    dam_idx: Vec<usize>,
    child_groups: Vec<Vec<usize>>,
    unrelated_idx: Vec<usize>,
}

#[pyclass]
pub struct SimTraitAccumulator {
    beta_rng: StdRng,
    beta_sd: f32,
}

#[pyclass]
pub struct SimEngine {
    chunk: SimChunkGenerator,
    trait_acc: SimTraitAccumulator,
}

#[pymethods]
impl SimChunkGenerator {
    #[new]
    #[pyo3(signature = (
        nsnp,
        nidv,
        chunk_size=50_000,
        maf_low=0.02_f32,
        maf_high=0.45_f32,
        seed=1_u64,
        homo=false,
        sire_idx=Vec::new(),
        dam_idx=Vec::new(),
        child_groups=Vec::new(),
        unrelated_idx=Vec::new()
    ))]
    fn new(
        nsnp: usize,
        nidv: usize,
        chunk_size: usize,
        maf_low: f32,
        maf_high: f32,
        seed: u64,
        homo: bool,
        sire_idx: Vec<usize>,
        dam_idx: Vec<usize>,
        child_groups: Vec<Vec<usize>>,
        unrelated_idx: Vec<usize>,
    ) -> PyResult<Self> {
        if nsnp == 0 || nidv == 0 {
            return Err(PyValueError::new_err("nsnp and nidv must be > 0"));
        }
        if chunk_size == 0 {
            return Err(PyValueError::new_err("chunk_size must be > 0"));
        }
        if !(maf_low > 0.0_f32 && maf_low < maf_high && maf_high <= 0.5_f32) {
            return Err(PyValueError::new_err(
                "maf bounds must satisfy 0 < maf_low < maf_high <= 0.5",
            ));
        }
        if sire_idx.len() != dam_idx.len() {
            return Err(PyValueError::new_err(
                "sire_idx and dam_idx length mismatch",
            ));
        }

        let n_families = sire_idx.len();
        let n_family_samples = if n_families == 0 {
            0
        } else {
            2 * n_families + child_groups.len() * n_families
        };

        for &idx in sire_idx.iter() {
            if idx >= nidv {
                return Err(PyValueError::new_err(format!(
                    "sire index out of range: {idx} >= {nidv}"
                )));
            }
        }
        for &idx in dam_idx.iter() {
            if idx >= nidv {
                return Err(PyValueError::new_err(format!(
                    "dam index out of range: {idx} >= {nidv}"
                )));
            }
        }
        for (gi, grp) in child_groups.iter().enumerate() {
            if grp.len() != n_families {
                return Err(PyValueError::new_err(format!(
                    "child_groups[{gi}] length mismatch: got {}, expected {}",
                    grp.len(),
                    n_families
                )));
            }
            for &idx in grp.iter() {
                if idx >= nidv {
                    return Err(PyValueError::new_err(format!(
                        "child index out of range: {idx} >= {nidv}"
                    )));
                }
            }
        }
        for &idx in unrelated_idx.iter() {
            if idx >= nidv {
                return Err(PyValueError::new_err(format!(
                    "unrelated index out of range: {idx} >= {nidv}"
                )));
            }
        }

        let n_unrelated = if unrelated_idx.is_empty() {
            nidv.saturating_sub(n_family_samples)
        } else {
            unrelated_idx.len()
        };
        let unrelated_idx_use = if unrelated_idx.is_empty() {
            (n_family_samples..nidv).collect::<Vec<usize>>()
        } else {
            unrelated_idx
        };

        Ok(Self {
            nsnp,
            nidv,
            chunk_size,
            maf_low,
            maf_high,
            homo,
            n_done: 0,
            mafs_rng: StdRng::seed_from_u64(seed),
            row_seed_base: seed ^ 0xA5A5_5A5A_D3C1_BA7E_u64,
            n_families,
            n_unrelated,
            sire_idx,
            dam_idx,
            child_groups,
            unrelated_idx: unrelated_idx_use,
        })
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.nidv
    }

    #[getter]
    fn n_snps(&self) -> usize {
        self.nsnp
    }

    #[getter]
    fn generated_snps(&self) -> usize {
        self.n_done
    }

    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyArray2<i8>>, Vec<SiteInfo>)>> {
        if self.n_done >= self.nsnp {
            return Ok(None);
        }
        let m = std::cmp::min(self.chunk_size, self.nsnp - self.n_done);
        let mut mafs = vec![0.0_f32; m];
        for maf in mafs.iter_mut() {
            *maf = self.maf_low + (self.maf_high - self.maf_low) * self.mafs_rng.random::<f32>();
        }

        let base_row_idx = self.n_done;
        let sire_idx = self.sire_idx.clone();
        let dam_idx = self.dam_idx.clone();
        let child_groups = self.child_groups.clone();
        let unrelated_idx = self.unrelated_idx.clone();
        let n_families = self.n_families;
        let n_unrelated = self.n_unrelated;
        let nidv = self.nidv;
        let homo = self.homo;
        let row_seed_base = self.row_seed_base;

        let (data, sites) = py
            .detach(move || {
                sim_generate_chunk_data(
                    &mafs,
                    base_row_idx,
                    nidv,
                    n_families,
                    n_unrelated,
                    &sire_idx,
                    &dam_idx,
                    &child_groups,
                    &unrelated_idx,
                    homo,
                    row_seed_base,
                )
            })
            .map_err(PyRuntimeError::new_err)?;
        let arr = Array2::from_shape_vec((m, self.nidv), data)
            .map_err(|e| PyRuntimeError::new_err(format!("sim chunk shape error: {e}")))?;
        let out = PyArray2::from_owned_array(py, arr).into_bound();

        self.n_done += m;
        Ok(Some((out, sites)))
    }
}

#[pymethods]
impl SimTraitAccumulator {
    #[new]
    #[pyo3(signature = (seed=1_u64, beta_sd=0.0_f32))]
    fn new(seed: u64, beta_sd: f32) -> PyResult<Self> {
        if !beta_sd.is_finite() || beta_sd < 0.0 {
            return Err(PyValueError::new_err("beta_sd must be finite and >= 0"));
        }
        Ok(Self {
            beta_rng: StdRng::seed_from_u64(seed),
            beta_sd,
        })
    }

    #[getter]
    fn beta_sd(&self) -> f32 {
        self.beta_sd
    }

    fn set_beta_sd(&mut self, beta_sd: f32) -> PyResult<()> {
        if !beta_sd.is_finite() || beta_sd < 0.0 {
            return Err(PyValueError::new_err("beta_sd must be finite and >= 0"));
        }
        self.beta_sd = beta_sd;
        Ok(())
    }

    fn reset_seed(&mut self, seed: u64) {
        self.beta_rng = StdRng::seed_from_u64(seed);
    }

    fn accumulate_chunk<'py>(
        &mut self,
        py: Python<'py>,
        mut y: PyReadwriteArray1<'py, f32>,
        g: PyReadonlyArray2<'py, i8>,
    ) -> PyResult<()> {
        let y_slice = y
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("y must be contiguous float32 1D"))?;
        let g_view = g.as_array();
        if g_view.ndim() != 2 {
            return Err(PyValueError::new_err("g must be a 2D int8 array"));
        }
        let m = g_view.shape()[0];
        let n = g_view.shape()[1];
        if y_slice.len() != n {
            return Err(PyValueError::new_err(format!(
                "y length mismatch: got {}, expected {}",
                y_slice.len(),
                n
            )));
        }
        if m == 0 || n == 0 || self.beta_sd <= 0.0 {
            return Ok(());
        }

        let beta = sim_generate_beta(&mut self.beta_rng, self.beta_sd, m);

        let g_owned;
        let g_data: &[i8] = if g_view.is_standard_layout() {
            g_view
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("g must be C-contiguous or castable"))?
        } else {
            let (raw, offset) = g_view.to_owned().into_raw_vec_and_offset();
            let total = m.saturating_mul(n);
            let start = offset.unwrap_or(0);
            let end = start.saturating_add(total);
            if end > raw.len() {
                return Err(PyValueError::new_err(format!(
                    "g copy layout error: start={}, total={}, raw_len={}",
                    start,
                    total,
                    raw.len()
                )));
            }
            g_owned = raw[start..end].to_vec();
            &g_owned
        };

        py.detach(|| sim_apply_trait_to_y(y_slice, g_data, &beta, n));
        Ok(())
    }
}

#[pymethods]
impl SimEngine {
    #[new]
    #[pyo3(signature = (
        nsnp,
        nidv,
        chunk_size=50_000,
        maf_low=0.02_f32,
        maf_high=0.45_f32,
        seed=1_u64,
        homo=false,
        sire_idx=Vec::new(),
        dam_idx=Vec::new(),
        child_groups=Vec::new(),
        unrelated_idx=Vec::new(),
        beta_seed=1_u64,
        beta_sd=0.0_f32
    ))]
    fn new(
        nsnp: usize,
        nidv: usize,
        chunk_size: usize,
        maf_low: f32,
        maf_high: f32,
        seed: u64,
        homo: bool,
        sire_idx: Vec<usize>,
        dam_idx: Vec<usize>,
        child_groups: Vec<Vec<usize>>,
        unrelated_idx: Vec<usize>,
        beta_seed: u64,
        beta_sd: f32,
    ) -> PyResult<Self> {
        let chunk = SimChunkGenerator::new(
            nsnp,
            nidv,
            chunk_size,
            maf_low,
            maf_high,
            seed,
            homo,
            sire_idx,
            dam_idx,
            child_groups,
            unrelated_idx,
        )?;
        let trait_acc = SimTraitAccumulator::new(beta_seed, beta_sd)?;
        Ok(Self { chunk, trait_acc })
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.chunk.nidv
    }

    #[getter]
    fn n_snps(&self) -> usize {
        self.chunk.nsnp
    }

    #[getter]
    fn generated_snps(&self) -> usize {
        self.chunk.n_done
    }

    fn set_beta_sd(&mut self, beta_sd: f32) -> PyResult<()> {
        self.trait_acc.set_beta_sd(beta_sd)
    }

    fn reset_beta_seed(&mut self, seed: u64) {
        self.trait_acc.reset_seed(seed);
    }

    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
        mut y: PyReadwriteArray1<'py, f32>,
    ) -> PyResult<Option<(Bound<'py, PyArray2<i8>>, Vec<SiteInfo>)>> {
        if self.chunk.n_done >= self.chunk.nsnp {
            return Ok(None);
        }
        let y_slice = y
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("y must be contiguous float32 1D"))?;
        if y_slice.len() != self.chunk.nidv {
            return Err(PyValueError::new_err(format!(
                "y length mismatch: got {}, expected {}",
                y_slice.len(),
                self.chunk.nidv
            )));
        }

        let m = std::cmp::min(self.chunk.chunk_size, self.chunk.nsnp - self.chunk.n_done);
        let mut mafs = vec![0.0_f32; m];
        for maf in mafs.iter_mut() {
            *maf = self.chunk.maf_low
                + (self.chunk.maf_high - self.chunk.maf_low) * self.chunk.mafs_rng.random::<f32>();
        }
        let beta = sim_generate_beta(&mut self.trait_acc.beta_rng, self.trait_acc.beta_sd, m);

        let base_row_idx = self.chunk.n_done;
        let sire_idx = self.chunk.sire_idx.clone();
        let dam_idx = self.chunk.dam_idx.clone();
        let child_groups = self.chunk.child_groups.clone();
        let unrelated_idx = self.chunk.unrelated_idx.clone();
        let n_families = self.chunk.n_families;
        let n_unrelated = self.chunk.n_unrelated;
        let nidv = self.chunk.nidv;
        let homo = self.chunk.homo;
        let row_seed_base = self.chunk.row_seed_base;

        let (data, sites) = py
            .detach(move || -> Result<(Vec<i8>, Vec<SiteInfo>), String> {
                let (data, sites) = sim_generate_chunk_data(
                    &mafs,
                    base_row_idx,
                    nidv,
                    n_families,
                    n_unrelated,
                    &sire_idx,
                    &dam_idx,
                    &child_groups,
                    &unrelated_idx,
                    homo,
                    row_seed_base,
                )?;
                sim_apply_trait_to_y(y_slice, &data, &beta, nidv);
                Ok((data, sites))
            })
            .map_err(PyRuntimeError::new_err)?;
        let arr = Array2::from_shape_vec((m, nidv), data)
            .map_err(|e| PyRuntimeError::new_err(format!("sim chunk shape error: {e}")))?;
        let out = PyArray2::from_owned_array(py, arr).into_bound();
        self.chunk.n_done += m;
        Ok(Some((out, sites)))
    }
}

#[pyfunction]
pub fn sim_trait_accumulate_i8_f32<'py>(
    py: Python<'py>,
    mut y: PyReadwriteArray1<'py, f32>,
    g: PyReadonlyArray2<'py, i8>,
    beta: PyReadonlyArray1<'py, f32>,
) -> PyResult<()> {
    let y_slice = y
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("y must be contiguous float32 1D"))?;
    let g_view = g.as_array();
    if g_view.ndim() != 2 {
        return Err(PyValueError::new_err("g must be a 2D int8 array"));
    }
    let m = g_view.shape()[0];
    let n = g_view.shape()[1];
    if y_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "y length mismatch: got {}, expected {}",
            y_slice.len(),
            n
        )));
    }
    if m == 0 || n == 0 {
        return Ok(());
    }

    let beta_owned;
    let beta_data: &[f32] = if let Ok(s) = beta.as_slice() {
        s
    } else {
        let (raw, offset) = beta.as_array().to_owned().into_raw_vec_and_offset();
        let total = m;
        let start = offset.unwrap_or(0);
        let end = start.saturating_add(total);
        if end > raw.len() {
            return Err(PyValueError::new_err(format!(
                "beta copy layout error: start={}, total={}, raw_len={}",
                start,
                total,
                raw.len()
            )));
        }
        beta_owned = raw[start..end].to_vec();
        &beta_owned
    };
    if beta_data.len() != m {
        return Err(PyValueError::new_err(format!(
            "beta length mismatch: got {}, expected {}",
            beta_data.len(),
            m
        )));
    }

    let g_owned;
    let g_data: &[i8] = if g_view.is_standard_layout() {
        g_view
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("g must be C-contiguous or castable"))?
    } else {
        let (raw, offset) = g_view.to_owned().into_raw_vec_and_offset();
        let total = m.saturating_mul(n);
        let start = offset.unwrap_or(0);
        let end = start.saturating_add(total);
        if end > raw.len() {
            return Err(PyValueError::new_err(format!(
                "g copy layout error: start={}, total={}, raw_len={}",
                start,
                total,
                raw.len()
            )));
        }
        g_owned = raw[start..end].to_vec();
        &g_owned
    };

    py.detach(|| sim_apply_trait_to_y(y_slice, g_data, beta_data, n));
    Ok(())
}
