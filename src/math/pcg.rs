use rayon::prelude::*;

pub(crate) struct PcgResultF32 {
    pub(crate) x: Vec<f32>,
    pub(crate) converged: bool,
    pub(crate) iters: usize,
    pub(crate) rel_res: f64,
}

const PCG_PAR_THRESHOLD: usize = 16_384;

#[inline]
fn dot_f32_f64(a: &[f32], b: &[f32]) -> f64 {
    if a.len() >= PCG_PAR_THRESHOLD {
        a.par_iter()
            .zip(b.par_iter())
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum()
    } else {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum()
    }
}

pub(crate) fn pcg_solve_f32<FA, FM, FO>(
    b: &[f32],
    max_iter: usize,
    tol: f64,
    tiny: f64,
    mut apply_a: FA,
    mut apply_m_inv: FM,
    mut on_iteration: FO,
) -> Result<PcgResultF32, String>
where
    FA: FnMut(&[f32]) -> Result<Vec<f32>, String>,
    FM: FnMut(&[f32], &mut [f32]),
    FO: FnMut(usize, usize, f64) -> Result<(), String>,
{
    let m = b.len();
    let mut x = vec![0.0_f32; m];
    if m == 0 || max_iter == 0 {
        return Ok(PcgResultF32 {
            x,
            converged: m == 0,
            iters: 0,
            rel_res: 0.0_f64,
        });
    }

    let bnorm = dot_f32_f64(b, b).sqrt();
    if !bnorm.is_finite() {
        return Err("PCG invalid RHS norm.".to_string());
    }
    let denom_b = bnorm.max(1e-12_f64);

    let mut r = b.to_vec();
    let mut z = vec![0.0_f32; m];
    apply_m_inv(&r, &mut z);
    let mut p = z.clone();
    let mut rz_old = dot_f32_f64(&r, &z);
    let mut converged = false;
    let mut rel_res = (dot_f32_f64(&r, &r).sqrt() / denom_b).max(0.0_f64);
    let mut iters_done = 0usize;
    let tiny_use = tiny.max(1e-30_f64);
    let tol_use = tol.max(0.0_f64);

    for it in 0..max_iter {
        let ap = apply_a(&p)?;
        if ap.len() != m {
            return Err(format!(
                "PCG apply_a length mismatch: got {}, expected {}",
                ap.len(),
                m
            ));
        }
        let denom = dot_f32_f64(&p, &ap);
        if !denom.is_finite() || denom <= tiny_use {
            break;
        }
        let alpha = rz_old / denom;
        let alpha32 = alpha as f32;
        if m >= PCG_PAR_THRESHOLD {
            x.par_iter_mut()
                .zip(r.par_iter_mut())
                .zip(p.par_iter().zip(ap.par_iter()))
                .for_each(|((xj, rj), (pj, apj))| {
                    *xj += alpha32 * *pj;
                    *rj -= alpha32 * *apj;
                });
        } else {
            for j in 0..m {
                x[j] += alpha32 * p[j];
                r[j] -= alpha32 * ap[j];
            }
        }
        rel_res = (dot_f32_f64(&r, &r).sqrt() / denom_b).max(0.0_f64);
        iters_done = it + 1;
        on_iteration(iters_done, max_iter, rel_res)?;
        if rel_res.is_finite() && rel_res <= tol_use {
            converged = true;
            break;
        }

        apply_m_inv(&r, &mut z);
        let rz_new = dot_f32_f64(&r, &z);
        if !rz_new.is_finite() || rz_new <= tiny_use {
            break;
        }
        let beta_cg = rz_new / rz_old.max(tiny_use);
        let beta32 = beta_cg as f32;
        if m >= PCG_PAR_THRESHOLD {
            p.par_iter_mut()
                .zip(z.par_iter())
                .for_each(|(pj, zj)| *pj = *zj + beta32 * *pj);
        } else {
            for j in 0..m {
                p[j] = z[j] + beta32 * p[j];
            }
        }
        rz_old = rz_new;
    }

    Ok(PcgResultF32 {
        x,
        converged,
        iters: iters_done,
        rel_res,
    })
}
