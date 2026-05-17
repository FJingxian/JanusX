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

#[inline]
fn normal_ppf_acklam(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    if p > P_HIGH {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }

    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
}

#[inline]
pub(crate) fn chi2_stat_df1_from_sf(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::INFINITY;
    }
    if p >= 1.0 {
        return 0.0;
    }
    let z = normal_ppf_acklam(1.0 - 0.5 * p);
    if z.is_finite() {
        z * z
    } else {
        f64::NAN
    }
}

#[inline]
pub(crate) fn format_chisq_value(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    format!("{value:.4e}")
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
