#[inline]
pub(crate) fn normal_sf(z: f64) -> f64 {
    0.5 * libm::erfc(z / std::f64::consts::SQRT_2)
}

#[inline]
pub(crate) fn chi2_sf_df1(stat: f64) -> f64 {
    if !stat.is_finite() || stat <= 0.0 {
        return 1.0;
    }
    let p = libm::erfc((0.5 * stat).sqrt());
    if p.is_finite() {
        p.clamp(f64::MIN_POSITIVE, 1.0)
    } else {
        1.0
    }
}

pub(crate) fn cholesky_inplace(a: &mut [f64], dim: usize) -> Option<()> {
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = a[i * dim + j];
            for k in 0..j {
                sum -= a[i * dim + k] * a[j * dim + k];
            }
            if i == j {
                if sum <= 1e-18 {
                    return None;
                }
                a[i * dim + j] = sum.sqrt();
            } else {
                a[i * dim + j] = sum / a[j * dim + j];
            }
        }
        for j in (i + 1)..dim {
            a[i * dim + j] = 0.0;
        }
    }
    Some(())
}

#[inline]
pub(crate) fn cholesky_solve_into(l: &[f64], dim: usize, b: &[f64], out: &mut [f64]) {
    debug_assert_eq!(b.len(), dim);
    debug_assert_eq!(out.len(), dim);

    for i in 0..dim {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * dim + k] * out[k];
        }
        out[i] = sum / l[i * dim + i];
    }

    for ii in 0..dim {
        let i = dim - 1 - ii;
        let mut sum = out[i];
        for k in (i + 1)..dim {
            sum -= l[k * dim + i] * out[k];
        }
        out[i] = sum / l[i * dim + i];
    }
}

#[inline]
pub(crate) fn cholesky_logdet(l: &[f64], dim: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..dim {
        s += l[i * dim + i].ln();
    }
    2.0 * s
}
