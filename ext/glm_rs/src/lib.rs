use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use rayon::prelude::*;

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let maxit = 200;
    let eps = 3.0e-14;
    let fpmin = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=maxit {
        let m2 = 2.0 * (m as f64);

        let mut aa = (m as f64) * (b - (m as f64)) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + (m as f64)) * (qab + (m as f64)) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }
    h
}

fn betai(a: f64, b: f64, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    let ln_beta = libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b);

    if x < (a + 1.0) / (a + b + 2.0) {
        let front = ((a * x.ln()) + (b * (1.0 - x).ln()) - ln_beta).exp() / a;
        front * betacf(a, b, x)
    } else {
        let front = ((b * (1.0 - x).ln()) + (a * x.ln()) - ln_beta).exp() / b;
        1.0 - front * betacf(b, a, 1.0 - x)
    }
}

fn student_t_cdf(t: f64, df: i32) -> f64 {
    if df <= 0 {
        return f64::NAN;
    }
    if !t.is_finite() {
        return if t > 0.0 { 1.0 } else { 0.0 };
    }

    let v = df as f64;
    let x = v / (v + t * t);
    let a = v / 2.0;
    let b = 0.5;
    let ib = betai(a, b, x);

    if t >= 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

#[inline]
fn student_t_p_two_sided(t: f64, df: i32) -> f64 {
    if df <= 0 { return f64::NAN; }
    if !t.is_finite() { return if t.is_nan() { f64::NAN } else { 0.0 }; }

    let v = df as f64;
    let x = v / (v + t * t);
    let a = v / 2.0;
    let b = 0.5;

    // 双侧 p 直接等于 regularized incomplete beta
    let mut p = betai(a, b, x);

    // 兜底：NaN/Inf -> 1；过小 -> clamp 到最小正数，避免 0
    if !p.is_finite() { p = 1.0; }
    p = p.clamp(f64::MIN_POSITIVE, 1.0);
    p
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[allow(non_snake_case)]
fn xs_t_iXX(xs: &[f64], ixx: &[f64], q0: usize) -> Vec<f64> {
    let mut b21 = vec![0.0; q0];
    for j in 0..q0 {
        let mut acc = 0.0;
        for k in 0..q0 {
            acc += xs[k] * ixx[k * q0 + j];
        }
        b21[j] = acc;
    }
    b21
}

#[allow(non_snake_case)]
fn build_iXXs(iXX: &[f64], b21: &[f64], invb22: f64, q0: usize) -> Vec<f64> {
    let dim = q0 + 1;
    let mut ixxs = vec![0.0; dim * dim];

    for r in 0..q0 {
        for c in 0..q0 {
            ixxs[r * dim + c] = iXX[r * q0 + c] + invb22 * (b21[r] * b21[c]);
        }
    }
    ixxs[q0 * dim + q0] = invb22;

    for j in 0..q0 {
        let v = -invb22 * b21[j];
        ixxs[q0 * dim + j] = v;
        ixxs[j * dim + q0] = v;
    }
    ixxs
}

#[inline]
fn matvec(a: &[f64], dim: usize, rhs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; dim];
    for r in 0..dim {
        let row = &a[r * dim..(r + 1) * dim];
        out[r] = row.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum();
    }
    out
}

// -------------------------
// small matrix utils (for pool test)
// -------------------------
fn matmul_at_b(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    // (a^T) (ac x ar)  *  b (br x bc) ; require ar==br
    assert!(ar == br);
    let mut out = vec![0.0; ac * bc];
    for i in 0..ac {
        for k in 0..ar {
            let aik = a[k * ac + i];
            let brow = &b[k * bc..(k + 1) * bc];
            let outrow = &mut out[i * bc..(i + 1) * bc];
            for j in 0..bc {
                outrow[j] += aik * brow[j];
            }
        }
    }
    out
}

/// Gauss-Jordan inverse for small dense matrix (dim <= ~128 recommended)
fn invert_gauss_jordan(a: &[f64], dim: usize, ridge: f64) -> Option<Vec<f64>> {
    let mut aug = vec![0.0; dim * 2 * dim]; // dim x (2dim)
    for r in 0..dim {
        for c in 0..dim {
            let mut v = a[r * dim + c];
            if r == c {
                v += ridge;
            }
            aug[r * (2 * dim) + c] = v;
        }
        aug[r * (2 * dim) + (dim + r)] = 1.0;
    }

    for i in 0..dim {
        // pivot
        let mut pivot = aug[i * (2 * dim) + i];
        if pivot.abs() < 1e-18 {
            // find swap
            let mut swap = None;
            for r in (i + 1)..dim {
                let v = aug[r * (2 * dim) + i];
                if v.abs() > pivot.abs() {
                    pivot = v;
                    swap = Some(r);
                }
            }
            if let Some(r2) = swap {
                for c in 0..(2 * dim) {
                    aug.swap(i * (2 * dim) + c, r2 * (2 * dim) + c);
                }
                pivot = aug[i * (2 * dim) + i];
            }
        }
        if pivot.abs() < 1e-18 {
            return None;
        }

        // normalize row
        let invp = 1.0 / pivot;
        for c in 0..(2 * dim) {
            aug[i * (2 * dim) + c] *= invp;
        }

        // eliminate others
        for r in 0..dim {
            if r == i {
                continue;
            }
            let factor = aug[r * (2 * dim) + i];
            if factor == 0.0 {
                continue;
            }
            for c in 0..(2 * dim) {
                let idx_rc = r * (2 * dim) + c;
                let idx_ic = i * (2 * dim) + c;
                aug[idx_rc] -= factor * aug[idx_ic];
            }
        }
    }

    let mut inv = vec![0.0; dim * dim];
    for r in 0..dim {
        let src = &aug[r * (2 * dim) + dim..r * (2 * dim) + 2 * dim];
        inv[r * dim..(r + 1) * dim].copy_from_slice(src);
    }
    Some(inv)
}

// -------------------------
// MLM: per-marker (matches mlm_c core)
// Inputs:
//   y: (n,) float64
//   X: (n,q0) float64
//   UT: (n,n) float64  (U.t())
//   UX: (n,q0) float64  (UT @ X)  <-- precompute in python for speed
//   Uy: (n,) float64    (UT @ y)
//   iUXUX: (q0,q0) float64  pinv(UX^T UX)
//   UXUy: (q0,) float64     UX^T Uy
//   G: (m,n) int8 marker rows
//   vgs: float64
// Return: (m,3): beta, se, p
// -------------------------

#[pyfunction]
#[pyo3(signature = (uy, ux, iuxux, uxuy, ut, g, vgs, step=10000, threads=0))]
fn mlmi8<'py>(
    py: Python<'py>,
    uy: PyReadonlyArray1<'py, f64>,
    ux: PyReadonlyArray2<'py, f64>,
    iuxux: PyReadonlyArray2<'py, f64>,
    uxuy: PyReadonlyArray1<'py, f64>,
    ut: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray2<'py, i8>,
    vgs: f64,
    step: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let uy = uy.as_slice()?;
    let ux_arr = ux.as_array();
    let iuxux_arr = iuxux.as_array();
    let uxuy = uxuy.as_slice()?;
    let ut_arr = ut.as_array();
    let g_arr = g.as_array();

    let n = uy.len();
    let (uxn, q0) = (ux_arr.shape()[0], ux_arr.shape()[1]);
    if uxn != n {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("UX.n_rows must equal len(Uy)"));
    }
    if iuxux_arr.shape() != &[q0, q0] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("iUXUX must be (q0,q0)"));
    }
    if uxuy.len() != q0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("UXUy must have length q0"));
    }
    if ut_arr.shape() != &[n, n] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("UT must be (n,n)"));
    }
    if g_arr.shape()[1] != n {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("G must be (m,n) with marker rows"));
    }

    let m = g_arr.shape()[0];

    // flatten for fast indexing
    let ux_flat: Vec<f64> = ux_arr.iter().cloned().collect();        // (n,q0)
    let ut_flat: Vec<f64> = ut_arr.iter().cloned().collect();        // (n,n)
    let iuxux_flat: Vec<f64> = iuxux_arr.iter().cloned().collect();  // (q0,q0)

    // output
    let out = PyArray2::<f64>::zeros(py, [m, 3], false);
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("output array not contiguous")
        })?
    };

    let pool = if threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("rayon pool: {e}")))?,
        )
    } else {
        None
    };

    py.detach(|| {
        let mut runner = || {
            let mut i_marker = 0usize;
            while i_marker < m {
                let cnt = std::cmp::min(step, m - i_marker);
                let block = &mut out_slice[i_marker * 3..(i_marker + cnt) * 3];

                block.par_chunks_mut(3).enumerate().for_each(|(l, row)| {
                    let idx = i_marker + l;

                    // s: genotype (n,)
                    // compute Us = UT @ s  (n,)
                    let mut us = vec![0.0_f64; n];
                    for r in 0..n {
                        let mut acc = 0.0;
                        let ut_row = &ut_flat[r * n..(r + 1) * n];
                        for k in 0..n {
                            acc += ut_row[k] * (g_arr[(idx, k)] as f64);
                        }
                        us[r] = acc;
                    }

                    // UXUs = UX^T Us  (q0,)
                    let mut uxus = vec![0.0_f64; q0];
                    for r in 0..n {
                        let ur = us[r];
                        let ux_row = &ux_flat[r * q0..(r + 1) * q0];
                        for j in 0..q0 {
                            uxus[j] += ux_row[j] * ur;
                        }
                    }

                    let usus = dot(&us, &us);
                    let usuy = dot(&us, uy);

                    // tmp = iUXUX @ UXUs
                    let tmp = matvec(&iuxux_flat, q0, &uxus);
                    let quad = dot(&uxus, &tmp);
                    let b22 = usus - quad;

                    if !b22.is_finite() || b22.abs() < 1e-12 {
                        row[0] = f64::NAN;
                        row[1] = f64::NAN;
                        row[2] = 1.0;
                        return;
                    }
                    let invb22 = 1.0 / b22;

                    // build iXXs (dim=q0+1)
                    let dim = q0 + 1;
                    let mut ixxs = vec![0.0_f64; dim * dim];

                    // B21 = (UXUs^T iUXUX) => row vector length q0
                    // We already have tmp = iUXUX@UXUs ; so B21 = tmp^T
                    let b21 = tmp;

                    // top-left: iUXUX + invb22 * (B21^T B21)
                    for r in 0..q0 {
                        for c in 0..q0 {
                            ixxs[r * dim + c] = iuxux_flat[r * q0 + c] + invb22 * (b21[r] * b21[c]);
                        }
                    }
                    // bottom-right
                    ixxs[q0 * dim + q0] = invb22;
                    // off-diagonal
                    for j in 0..q0 {
                        let v = -invb22 * b21[j];
                        ixxs[q0 * dim + j] = v;
                        ixxs[j * dim + q0] = v;
                    }

                    // rhs = [UXUy; UsUy]
                    let mut rhs = vec![0.0_f64; dim];
                    rhs[..q0].copy_from_slice(uxuy);
                    rhs[q0] = usuy;

                    // beta = ixxs @ rhs
                    let beta = matvec(&ixxs, dim, &rhs);

                    // se_snp = sqrt(iXXs(q0,q0) * vgs)
                    let se_snp = (ixxs[q0 * dim + q0] * vgs).sqrt();
                    let df = (n as i32) - (q0 as i32) - 1;

                    if !se_snp.is_finite() || se_snp <= 0.0 || df <= 0 {
                        row[0] = f64::NAN;
                        row[1] = f64::NAN;
                        row[2] = 1.0;
                        return;
                    }

                    let t = beta[q0] / se_snp;
                    let p = student_t_p_two_sided(t, df);

                    row[0] = beta[q0];
                    row[1] = se_snp;
                    row[2] = if p.is_finite() { p } else { 1.0 };
                });

                i_marker += cnt;
            }
        };

        if let Some(p) = &pool {
            p.install(|| runner());
        } else {
            runner();
        }
    });

    Ok(out)
}

// -------------------------
// Multi-locus (pool) test:
// Put multiple loci (k) into the fixed effects at once and compute their pvalues.
// Inputs:
//   y, X, UT as above
//   g_pool: (k,n) int8   (pool loci rows)
//   vgs: float64
// Return: (k,3): beta, se, p for each locus in pool (coefs for those loci)
// -------------------------

#[pyfunction]
#[pyo3(signature = (y, x, ut, g_pool, vgs, ridge=1e-10))]
fn mlmpi8<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    ut: PyReadonlyArray2<'py, f64>,
    g_pool: PyReadonlyArray2<'py, i8>,
    vgs: f64,
    ridge: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y = y.as_slice()?;
    let x_arr = x.as_array();
    let ut_arr = ut.as_array();
    let gp = g_pool.as_array();

    let n = y.len();
    let (xn, q0) = (x_arr.shape()[0], x_arr.shape()[1]);
    if xn != n {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("X must have n rows"));
    }
    if ut_arr.shape() != &[n, n] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("UT must be (n,n)"));
    }
    if gp.shape()[1] != n {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("g_pool must be (k,n) row loci"));
    }
    let k = gp.shape()[0];
    if k == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("g_pool k must be > 0"));
    }

    // flatten UT
    let ut_flat: Vec<f64> = ut_arr.iter().cloned().collect();

    // Uy = UT @ y
    let mut uy = vec![0.0_f64; n];
    for r in 0..n {
        let ut_row = &ut_flat[r * n..(r + 1) * n];
        let mut acc = 0.0;
        for i in 0..n {
            acc += ut_row[i] * y[i];
        }
        uy[r] = acc;
    }

    // UX = UT @ X  => (n,q0)
    let x_flat: Vec<f64> = x_arr.iter().cloned().collect(); // (n,q0)
    let mut ux = vec![0.0_f64; n * q0];
    for r in 0..n {
        for c in 0..q0 {
            let mut acc = 0.0;
            for i in 0..n {
                acc += ut_flat[r * n + i] * x_flat[i * q0 + c];
            }
            ux[r * q0 + c] = acc;
        }
    }

    // UZ for pool loci: Z = g_pool^T (n,k), UZ = UT @ Z => (n,k)
    let mut uz = vec![0.0_f64; n * k];
    for r in 0..n {
        for col in 0..k {
            let mut acc = 0.0;
            for i in 0..n {
                acc += ut_flat[r * n + i] * (gp[(col, i)] as f64);
            }
            uz[r * k + col] = acc;
        }
    }

    // Build design matrix W = [UX | UZ]  (n, q0+k)
    let dim = q0 + k;
    let mut w = vec![0.0_f64; n * dim];
    for r in 0..n {
        // UX part
        w[r * dim..r * dim + q0].copy_from_slice(&ux[r * q0..(r + 1) * q0]);
        // UZ part
        w[r * dim + q0..(r + 1) * dim].copy_from_slice(&uz[r * k..(r + 1) * k]);
    }

    // XtX = W^T W  (dim,dim)
    let xtx = matmul_at_b(&w, n, dim, &w, n, dim);

    // inv(XtX)
    let inv = invert_gauss_jordan(&xtx, dim, ridge).ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("Failed to invert W'W (try larger ridge)")
    })?;

    // rhs = W^T Uy  (dim,)
    let mut rhs = vec![0.0_f64; dim];
    for r in 0..n {
        let ur = uy[r];
        let w_row = &w[r * dim..(r + 1) * dim];
        for j in 0..dim {
            rhs[j] += w_row[j] * ur;
        }
    }

    // beta = inv @ rhs
    let beta = matvec(&inv, dim, &rhs);

    // Return p for pool part only (k)
    let df = (n as i32) - (dim as i32);
    if df <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("df <= 0 (too many covariates)"));
    }

    let out = PyArray2::<f64>::zeros(py, [k, 3], false);
    let out_slice: &mut [f64] = unsafe { out.as_slice_mut().unwrap() };

    for j in 0..k {
        let idx = q0 + j; // pool coefficient index
        let se = (inv[idx * dim + idx] * vgs).sqrt();
        let b = beta[idx];

        let p = if se.is_finite() && se > 0.0 {
            student_t_p_two_sided(b / se, df)
        } else {
            1.0
        };

        out_slice[j * 3 + 0] = b;
        out_slice[j * 3 + 1] = se;
        out_slice[j * 3 + 2] = if p.is_finite() { p } else { 1.0 };
    }

    Ok(out)
}

/// Fast GLM for:
/// y: (n,) float64
/// X: (n, q0) float64
/// iXX: (q0, q0) float64
/// G: (m, n) int8   (marker rows)  <-- no transpose
///
/// Return: (m, q0+3) float64
#[pyfunction]
#[pyo3(signature = (y, x, ixx, g, step=10000, threads=0))]
fn glmi8<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    ixx: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray2<'py, i8>,
    step: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y = y.as_slice()?;
    let x_arr = x.as_array();
    let ixx_arr = ixx.as_array();
    let g_arr = g.as_array();

    let n = y.len();
    let (xn, q0) = (x_arr.shape()[0], x_arr.shape()[1]);
    if xn != n {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "X.n_rows must equal len(y)",
        ));
    }
    if ixx_arr.shape()[0] != q0 || ixx_arr.shape()[1] != q0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "iXX must be (q0,q0)",
        ));
    }
    if g_arr.shape()[1] != n {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "G must be shape (m, n) for int8 fast path",
        ));
    }
    let m = g_arr.shape()[0];
    let row_stride = q0 + 3;

    // flatten X and iXX for fast indexing
    let x_flat: Vec<f64> = x_arr.iter().cloned().collect();
    let ixx_flat: Vec<f64> = ixx_arr.iter().cloned().collect();

    // precompute xy and yy
    let mut xy = vec![0.0; q0];
    for i in 0..n {
        let yi = y[i];
        let row = &x_flat[i * q0..(i + 1) * q0];
        for j in 0..q0 {
            xy[j] += row[j] * yi;
        }
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();

    // allocate output
    let out = PyArray2::<f64>::zeros(py, [m, row_stride], false);

    // IMPORTANT: borrow output as mutable slice, then parallel-fill chunks_mut safely
    // This is safe because each thread writes a disjoint row slice.
    let out_slice: &mut [f64] = unsafe {
        out.as_slice_mut().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("output array not contiguous")
        })?
    };

    // optional pool
    let pool = if threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("rayon pool: {e}")))?,
        )
    } else {
        None
    };

    py.detach(|| {
        let mut runner = || {
            let mut i_marker = 0usize;

            while i_marker < m {
                let cnt = std::cmp::min(step, m - i_marker);

                // split output into mutable row chunks for this block
                let block = &mut out_slice[i_marker * row_stride..(i_marker + cnt) * row_stride];

                block
                    .par_chunks_mut(row_stride)
                    .enumerate()
                    .for_each(|(l, row_out)| {
                        let idx = i_marker + l;

                        let mut sy = 0.0_f64;
                        let mut ss = 0.0_f64;
                        let mut xs = vec![0.0_f64; q0];

                        for k in 0..n {
                            let gv = g_arr[(idx, k)] as f64;
                            sy += gv * y[k];
                            ss += gv * gv;

                            let row = &x_flat[k * q0..(k + 1) * q0];
                            for j in 0..q0 {
                                xs[j] += row[j] * gv;
                            }
                        }

                        let b21 = xs_t_iXX(&xs, &ixx_flat, q0);
                        let t2 = dot(&b21, &xs);
                        let b22 = ss - t2;

                        let (invb22, df) = if b22 < 1e-8 {
                            (0.0, (n as i32) - (q0 as i32))
                        } else {
                            (1.0 / b22, (n as i32) - (q0 as i32) - 1)
                        };

                        let dim = q0 + 1;
                        let ixxs = build_iXXs(&ixx_flat, &b21, invb22, q0);

                        let mut rhs = vec![0.0_f64; dim];
                        rhs[..q0].copy_from_slice(&xy);
                        rhs[q0] = sy;

                        let beta = matvec(&ixxs, dim, &rhs);
                        let beta_rhs = dot(&beta, &rhs);
                        let ve = (yy - beta_rhs) / (df as f64);

                        // pvalues for all coefficients
                        for ff in 0..dim {
                            // let se = (ixxs[ff * dim + ff] * ve).sqrt();
                            // let t = beta[ff] / se;
                            // row_out[2 + ff] = student_t_p_two_sided(t, df);
                            let se = (ixxs[ff * dim + ff] * ve).sqrt();
                            let t = beta[ff] / se;
                            let mut p = student_t_p_two_sided(t, df);
                            p = p.clamp(f64::MIN_POSITIVE, 1.0);
                            row_out[2 + ff] = p;
                        }

                        // beta/se for SNP
                        if invb22 == 0.0 {
                            row_out[0] = f64::NAN;
                            row_out[1] = f64::NAN;
                            row_out[2 + q0] = f64::NAN;
                        } else {
                            let beta_snp = beta[q0];
                            let se_snp = (ixxs[q0 * dim + q0] * ve).sqrt();
                            row_out[0] = beta_snp;
                            row_out[1] = se_snp;
                        }
                    });

                i_marker += cnt;
            }
        };

        if let Some(p) = &pool {
            p.install(runner);
        } else {
            runner();
        }
    });

    Ok(out)
}

#[pymodule]
fn jxglm_rs(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mlmi8, m)?)?;
    m.add_function(wrap_pyfunction!(mlmpi8, m)?)?;
    m.add_function(wrap_pyfunction!(glmi8, m)?)?;
    Ok(())
}