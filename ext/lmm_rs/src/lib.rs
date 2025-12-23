use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::Bound;
use rayon::prelude::*;
use std::f64::consts::PI;

/// 标准正态 SF: P(Z > z)
#[inline]
fn normal_sf(z: f64) -> f64 {
    // sf = 0.5 * erfc(z / sqrt(2))
    0.5 * libm::erfc(z / std::f64::consts::SQRT_2)
}

/// cholesky 分解（就地），a 是 dim x dim，返回下三角 L，A = L L^T
fn cholesky_inplace(a: &mut [f64], dim: usize) -> Option<()> {
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
        // 上三角清零（可选）
        for j in (i + 1)..dim {
            a[i * dim + j] = 0.0;
        }
    }
    Some(())
}

/// 用 cholesky(L) 解 A x = b，a 中存的是 L（下三角）
fn cholesky_solve(a: &[f64], dim: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; dim];
    // 先解 L y = b
    for i in 0..dim {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i * dim + k] * y[k];
        }
        y[i] = sum / a[i * dim + i];
    }

    // 再解 L^T x = y
    let mut x = vec![0.0_f64; dim];
    for ii in 0..dim {
        let i = dim - 1 - ii;
        let mut sum = y[i];
        for k in (i + 1)..dim {
            sum -= a[k * dim + i] * x[k];
        }
        x[i] = sum / a[i * dim + i];
    }
    x
}

/// 从 cholesky(L) 计算 logdet(A) = 2 * sum(log(diag(L)))
fn cholesky_logdet(a: &[f64], dim: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..dim {
        let d = a[i * dim + i];
        s += d.ln();
    }
    2.0 * s
}

/// 计算 REML(lbd, snp_vec_rot)；
/// snp_vec_rot 已经是 Dh@y 之后同一空间里的 SNP（长度 n）
/// 逻辑严格对应你 Python 里的 _REML
fn reml_loglike(
    log10_lbd: f64,
    s: &[f64],         // eigen values S, len n
    xcov: &[f64],      // Xcov (n x p_cov) row-major
    y: &[f64],         // y_rot, len n
    snp: &[f64],       // snp_rot, len n
    n: usize,
    p_cov: usize,
) -> f64 {
    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return -1e8;
    }

    let p = p_cov + 1;
    if n <= p {
        return -1e8;
    }

    // V = S + lbd, V_inv = 1/V
    let mut v = vec![0.0_f64; n];
    let mut vinv = vec![0.0_f64; n];
    for i in 0..n {
        let vv = s[i] + lbd;
        if vv <= 0.0 {
            return -1e8;
        }
        v[i] = vv;
        vinv[i] = 1.0 / vv;
    }

    // XTV_invX (dim x dim)  和  XTV_invy (dim,)
    let dim = p;
    let mut xtv_inv_x = vec![0.0_f64; dim * dim];
    let mut xtv_inv_y = vec![0.0_f64; dim];

    for i in 0..n {
        let vi = vinv[i];
        let yi = y[i];

        // 第 p_cov 个之前是 Xcov，第 p_cov 是 snp
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xtv_inv_y[r] += vi * xir * yi;

            for c in 0..=r {
                let xic = if c < p_cov {
                    xcov[i * p_cov + c]
                } else {
                    snp[i]
                };
                xtv_inv_x[r * dim + c] += vi * xir * xic;
            }
        }
    }

    // 补上对称部分 + ridge
    let ridge = 1e-6;
    for r in 0..dim {
        xtv_inv_x[r * dim + r] += ridge;
        for c in 0..r {
            let vrc = xtv_inv_x[r * dim + c];
            xtv_inv_x[c * dim + r] = vrc;
        }
    }

    // chol
    if cholesky_inplace(&mut xtv_inv_x, dim).is_none() {
        return -1e8;
    }
    let log_det_xtv_inv_x = cholesky_logdet(&xtv_inv_x, dim);
    let beta = cholesky_solve(&xtv_inv_x, dim, &xtv_inv_y);

    // 残差 r = y - X_cov_snp @ beta
    let mut r_vec = vec![0.0_f64; n];
    for i in 0..n {
        let mut xb = 0.0;
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xb += xir * beta[r];
        }
        r_vec[i] = y[i] - xb;
    }

    let mut rtv_invr = 0.0_f64;
    for i in 0..n {
        rtv_invr += vinv[i] * r_vec[i] * r_vec[i];
    }

    let log_det_v: f64 = v.iter().map(|vv| vv.ln()).sum();
    let n_f = n as f64;
    let p_f = p as f64;

    let total_log = (n_f - p_f) * (rtv_invr.ln()) + log_det_v + log_det_xtv_inv_x;
    let c = (n_f - p_f) * ((n_f - p_f).ln() - 1.0 - (2.0 * PI).ln()) / 2.0;
    let reml = c - 0.5 * total_log;

    if !reml.is_finite() {
        -1e8
    } else {
        reml
    }
}

fn brent_max_reml(
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snp: &[f64],
    n: usize,
    p_cov: usize,
    low: f64,
    high: f64,
    tol: f64,
    max_iter: usize,
) -> (f64, f64) {
    // 容错：low < high
    let mut a = low;
    let mut c = high;
    if !(a < c) {
        std::mem::swap(&mut a, &mut c);
    }

    // 机器精度防护
    let eps = 2.220_446_049_250_313e-16_f64; // f64::EPSILON
    let tol = tol.abs().max(1e-12);

    // 目标函数：g(x) = -REML(x)，Brent 是最小化
    let f = |x: f64| -> f64 {
        -reml_loglike(x, s, xcov, y, snp, n, p_cov)
    };

    // 初始点：取中点
    let mut x = 0.5 * (a + c);
    let mut w = x;
    let mut v = x;

    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;

    // 这两个是最近一步和上上步的步长
    let mut d = 0.0_f64;
    let mut e = 0.0_f64;

    for _iter in 0..max_iter {
        let m = 0.5 * (a + c);
        let tol1 = tol * x.abs() + eps;
        let tol2 = 2.0 * tol1;

        // 收敛判据：当前 x 已经在区间中点附近，并且区间足够小
        if (x - m).abs() <= tol2 - 0.5 * (c - a) {
            break;
        }

        let mut p = 0.0_f64;
        let mut q = 0.0_f64;
        let mut r = 0.0_f64;
        let mut u: f64;

        let use_parabolic = if e.abs() > tol1 {
            // 尝试抛物线插值
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);

            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }

            let mut ok = false;
            if q.abs() > eps {
                let s = p / q;
                u = x + s;

                // 条件：u 在 (a,c) 内，并且步长不太大
                if (u - a) >= tol2
                    && (c - u) >= tol2
                    && s.abs() < 0.5 * e.abs()
                {
                    ok = true;
                }
            }

            if ok {
                d = p / q;
                u = x + d;
                // 如果太接近边界，再稍微偏移一下
                if (u - a) < tol2 || (c - u) < tol2 {
                    d = if x < m { tol1 } else { -tol1 };
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        if !use_parabolic {
            // 退回黄金分割步骤
            e = if x < m { c - x } else { x - a };
            d = 0.3819660_f64 * e; // (3 - sqrt(5)) / 2 ≈ 0.381966
        }

        // 确保步长至少为 tol1
        if d.abs() < tol1 {
            d = if d >= 0.0 { tol1 } else { -tol1 };
        }

        u = x + d;
        let fu = f(u);

        // 更新区间和三点 x,w,v
        if fu <= fx {
            // u 成为新的最好点
            if u >= x {
                a = x;
            } else {
                c = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            // u 只是次优点
            if u >= x {
                a = u;
            } else {
                c = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }

    let best_log10_lbd = x;
    let best_ll = -fx; // 因为 f = -REML

    (best_log10_lbd, best_ll)
}

// /// 简单的黄金分割搜索，在 log10(lbd) 上最大化 REML
// fn brent_max_reml(
//     s: &[f64],
//     xcov: &[f64],
//     y: &[f64],
//     snp: &[f64],
//     n: usize,
//     p_cov: usize,
//     low: f64,
//     high: f64,
//     tol: f64,
//     max_iter: usize,
// ) -> (f64, f64) {
//     let mut a = low;
//     let mut b = high;
//     let phi = 0.5 * (3.0_f64.sqrt() - 1.0); // ~0.618

//     let mut c = b - phi * (b - a);
//     let mut d = a + phi * (b - a);

//     let mut fc = reml_loglike(c, s, xcov, y, snp, n, p_cov);
//     let mut fd = reml_loglike(d, s, xcov, y, snp, n, p_cov);

//     for _ in 0..max_iter {
//         if (b - a).abs() < tol {
//             break;
//         }
//         if fc > fd {
//             b = d;
//             d = c;
//             fd = fc;
//             c = b - phi * (b - a);
//             fc = reml_loglike(c, s, xcov, y, snp, n, p_cov);
//         } else {
//             a = c;
//             c = d;
//             fc = fd;
//             d = a + phi * (b - a);
//             fd = reml_loglike(d, s, xcov, y, snp, n, p_cov);
//         }
//     }

//     if fc > fd {
//         (c, fc)
//     } else {
//         (d, fd)
//     }
// }

/// 计算最终 beta, se, lbd：
/// 逻辑对应 LMM._fit 里成功找到 lbd 后那一段
fn final_beta_se(
    log10_lbd: f64,
    s: &[f64],
    xcov: &[f64],
    y: &[f64],
    snp: &[f64],
    n: usize,
    p_cov: usize,
) -> (f64, f64, f64) {
    let lbd = 10.0_f64.powf(log10_lbd);
    if !lbd.is_finite() || lbd <= 0.0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let p = p_cov + 1;
    if n <= p {
        return (f64::NAN, f64::NAN, lbd);
    }

    // V / V_inv
    let mut v = vec![0.0_f64; n];
    let mut vinv = vec![0.0_f64; n];
    for i in 0..n {
        let vv = s[i] + lbd;
        if vv <= 0.0 {
            return (f64::NAN, f64::NAN, lbd);
        }
        v[i] = vv;
        vinv[i] = 1.0 / vv;
    }

    // X^T V^{-1} X & X^T V^{-1} y
    let dim = p;
    let mut xtv_inv_x = vec![0.0_f64; dim * dim];
    let mut xtv_inv_y = vec![0.0_f64; dim];

    for i in 0..n {
        let vi = vinv[i];
        let yi = y[i];
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xtv_inv_y[r] += vi * xir * yi;

            for c in 0..=r {
                let xic = if c < p_cov {
                    xcov[i * p_cov + c]
                } else {
                    snp[i]
                };
                xtv_inv_x[r * dim + c] += vi * xir * xic;
            }
        }
    }
    let ridge = 1e-6;
    for r in 0..dim {
        xtv_inv_x[r * dim + r] += ridge;
        for c in 0..r {
            let vrc = xtv_inv_x[r * dim + c];
            xtv_inv_x[c * dim + r] = vrc;
        }
    }

    if cholesky_inplace(&mut xtv_inv_x, dim).is_none() {
        return (f64::NAN, f64::NAN, lbd);
    }
    let beta = cholesky_solve(&xtv_inv_x, dim, &xtv_inv_y);

    // 残差
    let mut r_vec = vec![0.0_f64; n];
    for i in 0..n {
        let mut xb = 0.0;
        for r in 0..dim {
            let xir = if r < p_cov {
                xcov[i * p_cov + r]
            } else {
                snp[i]
            };
            xb += xir * beta[r];
        }
        r_vec[i] = y[i] - xb;
    }

    let mut rtv_invr = 0.0_f64;
    for i in 0..n {
        rtv_invr += vinv[i] * r_vec[i] * r_vec[i];
    }
    let n_f = n as f64;
    let p_f = p as f64;
    let sigma2 = rtv_invr / (n_f - p_f);

    // 为了得到最后一个系数的方差：解 A x = e_k, x_k 就是 (A^{-1})_{kk}
    let k = dim - 1;
    let mut e = vec![0.0_f64; dim];
    e[k] = 1.0;

    let x = cholesky_solve(&xtv_inv_x, dim, &e);
    let var_beta_k = sigma2 * x[k];
    if var_beta_k <= 0.0 || !var_beta_k.is_finite() {
        return (f64::NAN, f64::NAN, lbd);
    }
    let se = var_beta_k.sqrt();
    (beta[k], se, lbd)
}

///
/// Rust 加速版 MLM-GWAS：对一个块中的多个 SNP 做 REML λ 优化 + β/SE/p 计算。
///
/// Python 端需要先算好：
///   S, Xcov=self.Dh@X, y_rot=self.Dh@y, bounds=self.bounds
/// 然后对 snp_chunk 做：
///   snp_chunk_rot = snp_chunk @ self.Dh.T   # (m_chunk, n)
///
/// 本函数的 g_rot_chunk 就是这个 snp_chunk_rot（float32）
///
/// 参数：
///   s:      (n,)    eigenvalues S
///   xcov:   (n,p0)  Xcov in rotated space
///   y_rot:  (n,)    y in rotated space
///   low:    float   log10(lambda) 下界 (self.bounds[0])
///   high:   float   log10(lambda) 上界 (self.bounds[1])
///   g_rot_chunk: (m_chunk,n) float32，谱空间里的基因型
///   max_iter:     λ 搜索的最大迭代次数（默认 50）
///   tol:          λ 搜索的收敛阈值（默认 1e-2，对 log10(lambda)）
///   threads:      rayon 线程数，0 表示用默认全核
///
/// 返回：
///   (beta_se_p, lambdas)
///   beta_se_p: (m_chunk,3) [beta, se, p]
///   lambdas:   (m_chunk,)  每个 SNP 对应的 λ
///
#[pyfunction]
#[pyo3(signature = (s, xcov, y_rot, low, high, g_rot_chunk, max_iter=50, tol=1e-2, threads=0))]
fn lmm_reml_chunk_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<'py, f64>,
    xcov: PyReadonlyArray2<'py, f64>,
    y_rot: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    g_rot_chunk: PyReadonlyArray2<'py, f32>,
    max_iter: usize,
    tol: f64,
    threads: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    use numpy::PyArrayMethods; // 确保 as_slice_mut 在作用域内

    let s = s.as_slice()?;
    let xcov_arr = xcov.as_array();
    let y = y_rot.as_slice()?;
    let g_arr = g_rot_chunk.as_array(); // ArrayView2<f32>

    let n = y.len();
    let (xc_n, p_cov) = (xcov_arr.shape()[0], xcov_arr.shape()[1]);
    if xc_n != n {
        return Err(PyRuntimeError::new_err("Xcov.n_rows must equal len(y_rot)"));
    }
    if s.len() != n {
        return Err(PyRuntimeError::new_err("len(S) must equal len(y_rot)"));
    }
    if g_arr.shape()[1] != n {
        return Err(PyRuntimeError::new_err("g_rot_chunk must be (m_chunk, n)"));
    }
    if low >= high {
        return Err(PyRuntimeError::new_err("low must be < high (log10(lambda) bounds)"));
    }

    let m_chunk = g_arr.shape()[0];

    // 拍扁 xcov，方便快速索引
    let xcov_flat: Vec<f64> = xcov_arr.iter().cloned().collect();

    // 分配输出 numpy 数组（已是 Bound<PyArray>）
    let beta_se_p = PyArray2::<f64>::zeros_bound(py, [m_chunk, 3], false);
    let lambdas = PyArray1::<f64>::zeros_bound(py, [m_chunk], false);

    // 先把底层 buffer 取出来（后面串行写）
    let beta_se_p_slice: &mut [f64] = unsafe {
        beta_se_p
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("beta_se_p not contiguous"))?
    };
    let lambdas_slice: &mut [f64] = unsafe {
        lambdas
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("lambdas not contiguous"))?
    };

    // Rayon 线程池（可选设置线程数）
    let pool = if threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("rayon pool: {e}")))?,
        )
    } else {
        None
    };

    // 在不持有 GIL 的情况下做重计算
    py.allow_threads(|| {
        // 1. 并行算出每个 SNP 的 (beta,se,p,lambda)，放到 Vec 里
        let compute_all = || {
            (0..m_chunk)
                .into_par_iter()
                .map(|idx| {
                    let row = g_arr.row(idx);
                    // 把 float32 转成 float64 向量（已经是在 Dh 空间里的 snp）
                    let mut snp_vec = vec![0.0_f64; n];
                    for i in 0..n {
                        snp_vec[i] = row[i] as f64;
                    }

                    // 1) 黄金分割搜索最优 log10(lambda)
                    let (best_log10_lbd, _best_ll) = brent_max_reml(
                        s,
                        &xcov_flat,
                        y,
                        &snp_vec,
                        n,
                        p_cov,
                        low,
                        high,
                        tol,
                        max_iter,
                    );

                    // 2) 用最优 lambda 再算一次 beta, se
                    let (beta, se, lbd) =
                        final_beta_se(best_log10_lbd, s, &xcov_flat, y, &snp_vec, n, p_cov);

                    // 3) 计算 p 值
                    let p = if beta.is_finite() && se.is_finite() && se > 0.0 {
                        let z = beta / se;
                        2.0 * normal_sf(z.abs())
                    } else {
                        1.0
                    };

                    (beta, se, p, lbd)
                })
                .collect::<Vec<(f64, f64, f64, f64)>>()
        };

        let results = if let Some(pool) = &pool {
            pool.install(compute_all)
        } else {
            compute_all()
        };

        // 2. 串行写入 numpy buffer（避免 &mut slice 在多线程闭包中被捕获）
        for (idx, (beta, se, p, lbd)) in results.into_iter().enumerate() {
            let out_row = &mut beta_se_p_slice[idx * 3..(idx + 1) * 3];
            out_row[0] = beta;
            out_row[1] = se;
            out_row[2] = if p.is_finite() { p } else { 1.0 };
            lambdas_slice[idx] = lbd;
        }
    });

    Ok((beta_se_p, lambdas))
}

#[pymodule]
fn lmm_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lmm_reml_chunk_f32, m)?)?;
    Ok(())
}