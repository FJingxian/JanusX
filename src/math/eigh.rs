use nalgebra::DMatrix;
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::BoundObject;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
use crate::blas::{lapack_dsyevd_dispatch, lapack_dsyevr_dispatch, CblasInt};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EighDriver {
    Dsyevd,
    Dsyevr,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EighJobz {
    ValuesOnly,
    ValuesAndVectors,
}

#[inline]
fn copy_col_major_to_row_major(src_col_major: &[f64], n: usize) -> Vec<f64> {
    let mut out_row_major = vec![0.0_f64; n * n];
    for col in 0..n {
        for row in 0..n {
            out_row_major[row * n + col] = src_col_major[row + col * n];
        }
    }
    out_row_major
}

#[inline]
fn parse_eigh_driver_token(token: &str) -> EighDriver {
    match token.trim().to_ascii_lowercase().as_str() {
        // Keep dsyevr as explicit opt-in; default/auto stay on dsyevd.
        "dsyevr" | "syevr" | "evr" => EighDriver::Dsyevr,
        "auto" => EighDriver::Dsyevd,
        "" | "dsyevd" | "syevd" | "evd" => EighDriver::Dsyevd,
        _ => EighDriver::Dsyevd,
    }
}

#[inline]
fn parse_eigh_jobz_token(token: &str) -> EighJobz {
    match token.trim().to_ascii_uppercase().as_str() {
        "N" | "NOVEC" | "NOVECTORS" | "EIGVALS" => EighJobz::ValuesOnly,
        "V" | "VEC" | "VECTORS" | "EIGENPAIRS" | "" => EighJobz::ValuesAndVectors,
        _ => EighJobz::ValuesAndVectors,
    }
}

#[inline]
fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let t = v.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on")
        })
        .unwrap_or(false)
}

#[inline]
fn eigh_jobz_label(jobz: EighJobz) -> &'static str {
    match jobz {
        EighJobz::ValuesOnly => "N",
        EighJobz::ValuesAndVectors => "V",
    }
}

#[inline]
fn is_lapack_backend(name: &str) -> bool {
    name.starts_with("lapack_")
}

#[inline]
fn resolve_eigh_driver_from_env() -> EighDriver {
    let req = std::env::var("JX_EIGH_DRIVER")
        .ok()
        .unwrap_or_else(|| "dsyevd".to_string());
    let parsed = parse_eigh_driver_token(&req);
    if matches!(parsed, EighDriver::Dsyevr) && !env_truthy("JX_EIGH_ALLOW_DSYEVR") {
        // Keep dsyevr opt-in for experiments; production default remains dsyevd.
        EighDriver::Dsyevd
    } else {
        parsed
    }
}

#[inline]
fn lapack_dsyevr_row_major_inplace(
    a_row_major: &mut [f64],
    n: usize,
    jobz: EighJobz,
) -> Result<(Vec<f64>, Option<Vec<f64>>, &'static str), String> {
    let n_i = n as CblasInt;
    let lda = n_i;
    let jobz_flag = [
        eigh_jobz_label(jobz).as_bytes()[0] as std::os::raw::c_char,
        0,
    ];
    let uplo = [b'U' as std::os::raw::c_char, 0];
    let need_vec = matches!(jobz, EighJobz::ValuesAndVectors);

    let mut w = vec![0.0_f64; n];
    let mut z = if need_vec {
        vec![0.0_f64; n * n]
    } else {
        vec![0.0_f64; 1]
    };
    let ldz = if need_vec { n_i } else { 1 as CblasInt };
    let range = [b'A' as std::os::raw::c_char, 0];
    let vl = 0.0_f64;
    let vu = 0.0_f64;
    let il: CblasInt = 0;
    let iu: CblasInt = 0;
    let abstol = 0.0_f64;
    let mut m_found: CblasInt = 0;
    let mut isuppz = vec![0 as CblasInt; 2 * n];
    let mut info: CblasInt = 0;
    let mut lwork: CblasInt = -1;
    let mut liwork: CblasInt = -1;
    let mut work_q = [0.0_f64];
    let mut iwork_q = [0 as CblasInt];

    let lapack_ready = unsafe {
        lapack_dsyevr_dispatch(
            jobz_flag.as_ptr(),
            range.as_ptr(),
            uplo.as_ptr(),
            &n_i,
            a_row_major.as_mut_ptr(),
            &lda,
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            &mut m_found,
            w.as_mut_ptr(),
            z.as_mut_ptr(),
            &ldz,
            isuppz.as_mut_ptr(),
            work_q.as_mut_ptr(),
            &lwork,
            iwork_q.as_mut_ptr(),
            &liwork,
            &mut info,
        )
    };
    if lapack_ready.is_err() {
        return Err("dsyevr workspace query dispatch unavailable".to_string());
    }
    if info != 0 {
        return Err(format!("dsyevr workspace query failed: info={info}"));
    }

    let lwork_use = (work_q[0] as CblasInt).max(1);
    let liwork_use = iwork_q[0].max(1);
    let mut work = vec![0.0_f64; lwork_use as usize];
    let mut iwork = vec![0 as CblasInt; liwork_use as usize];
    info = 0;
    lwork = lwork_use;
    liwork = liwork_use;
    m_found = 0;
    let run_ok = unsafe {
        lapack_dsyevr_dispatch(
            jobz_flag.as_ptr(),
            range.as_ptr(),
            uplo.as_ptr(),
            &n_i,
            a_row_major.as_mut_ptr(),
            &lda,
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            &mut m_found,
            w.as_mut_ptr(),
            z.as_mut_ptr(),
            &ldz,
            isuppz.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        )
    };
    if run_ok.is_err() {
        return Err("dsyevr run dispatch unavailable".to_string());
    }
    if info != 0 {
        return Err(format!("dsyevr run failed: info={info}"));
    }
    if (m_found as usize) != n {
        return Err(format!("dsyevr returned m={m_found}, expected n={n}"));
    }

    if need_vec {
        let u_row = copy_col_major_to_row_major(&z, n);
        Ok((w, Some(u_row), "lapack_dsyevr"))
    } else {
        Ok((w, None, "lapack_dsyevr"))
    }
}

#[inline]
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn lapack_dsyevd_row_major_inplace(
    a_row_major: &mut [f64],
    n: usize,
    jobz: EighJobz,
) -> Result<(Vec<f64>, Option<Vec<f64>>, &'static str), String> {
    let n_i = n as CblasInt;
    let lda = n_i;
    let jobz_flag = [
        eigh_jobz_label(jobz).as_bytes()[0] as std::os::raw::c_char,
        0,
    ];
    let uplo = [b'U' as std::os::raw::c_char, 0];
    let need_vec = matches!(jobz, EighJobz::ValuesAndVectors);

    let mut w = vec![0.0_f64; n];
    let mut info: CblasInt = 0;
    let mut lwork: CblasInt = -1;
    let mut liwork: CblasInt = -1;
    let mut work_q = [0.0_f64];
    let mut iwork_q = [0 as CblasInt];

    let lapack_ready = unsafe {
        lapack_dsyevd_dispatch(
            jobz_flag.as_ptr(),
            uplo.as_ptr(),
            &n_i,
            a_row_major.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work_q.as_mut_ptr(),
            &lwork,
            iwork_q.as_mut_ptr(),
            &liwork,
            &mut info,
        )
    };
    if lapack_ready.is_err() {
        return Err("dsyevd workspace query dispatch unavailable".to_string());
    }
    if info != 0 {
        return Err(format!("dsyevd workspace query failed: info={info}"));
    }

    let lwork_use = (work_q[0] as CblasInt).max(1);
    let liwork_use = iwork_q[0].max(1);
    let mut work = vec![0.0_f64; lwork_use as usize];
    let mut iwork = vec![0 as CblasInt; liwork_use as usize];
    info = 0;
    lwork = lwork_use;
    liwork = liwork_use;
    let run_ok = unsafe {
        lapack_dsyevd_dispatch(
            jobz_flag.as_ptr(),
            uplo.as_ptr(),
            &n_i,
            a_row_major.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        )
    };
    if run_ok.is_err() {
        return Err("dsyevd run dispatch unavailable".to_string());
    }
    if info != 0 {
        return Err(format!("dsyevd run failed: info={info}"));
    }

    if need_vec {
        let u_row = copy_col_major_to_row_major(a_row_major, n);
        Ok((w, Some(u_row), "lapack_dsyevd"))
    } else {
        Ok((w, None, "lapack_dsyevd"))
    }
}

#[inline]
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn lapack_try_driver_row_major_inplace(
    a_row_major: &mut [f64],
    n: usize,
    jobz: EighJobz,
    driver: EighDriver,
) -> Result<(Vec<f64>, Option<Vec<f64>>, &'static str), String> {
    match driver {
        EighDriver::Dsyevd => lapack_dsyevd_row_major_inplace(a_row_major, n, jobz),
        EighDriver::Dsyevr => lapack_dsyevr_row_major_inplace(a_row_major, n, jobz),
    }
}

#[inline]
fn symmetric_eigh_f64_row_major_with_driver(
    a_row_major: &[f64],
    n: usize,
    driver: EighDriver,
    jobz: EighJobz,
) -> Result<(Vec<f64>, Option<Vec<f64>>, &'static str), String> {
    if a_row_major.len() != n.saturating_mul(n) {
        return Err(format!(
            "symmetric_eigh_f64_row_major: shape mismatch, len={} expected={}",
            a_row_major.len(),
            n.saturating_mul(n)
        ));
    }
    if n == 0 {
        let vecs = if matches!(jobz, EighJobz::ValuesAndVectors) {
            Some(Vec::new())
        } else {
            None
        };
        return Ok((Vec::new(), vecs, "empty"));
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        let mut lapack_errors: Vec<String> = Vec::new();
        let mut a_primary = a_row_major.to_vec();
        let primary = driver;
        let secondary = match driver {
            EighDriver::Dsyevd => EighDriver::Dsyevr,
            EighDriver::Dsyevr => EighDriver::Dsyevd,
        };

        match lapack_try_driver_row_major_inplace(&mut a_primary, n, jobz, primary) {
            Ok(res) => return Ok(res),
            Err(err) => {
                lapack_errors.push(format!(
                    "{}: {err}",
                    match primary {
                        EighDriver::Dsyevd => "dsyevd",
                        EighDriver::Dsyevr => "dsyevr",
                    }
                ));
            }
        }

        let mut a_retry = a_row_major.to_vec();
        match lapack_try_driver_row_major_inplace(&mut a_retry, n, jobz, secondary) {
            Ok(res) => return Ok(res),
            Err(err) => {
                lapack_errors.push(format!(
                    "{}: {err}",
                    match secondary {
                        EighDriver::Dsyevd => "dsyevd",
                        EighDriver::Dsyevr => "dsyevr",
                    }
                ));
            }
        }

        let allow_nalgebra_fallback = if cfg!(target_os = "windows") {
            // Windows default keeps fallback enabled, but allow explicit env
            // override for diagnostics/perf debugging.
            match std::env::var("JX_EIGH_ALLOW_NALGEBRA_FALLBACK") {
                Ok(v) => {
                    let t = v.trim().to_ascii_lowercase();
                    matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on")
                }
                Err(_) => true,
            }
        } else {
            env_truthy("JX_EIGH_ALLOW_NALGEBRA_FALLBACK")
        };
        if !allow_nalgebra_fallback {
            return Err(format!(
                "LAPACK eigh failed (driver={}, jobz={}): {}",
                match driver {
                    EighDriver::Dsyevd => "dsyevd",
                    EighDriver::Dsyevr => "dsyevr",
                },
                eigh_jobz_label(jobz),
                lapack_errors.join("; ")
            ));
        }
    }

    let dm = DMatrix::from_row_slice(n, n, a_row_major);
    let se = nalgebra::linalg::SymmetricEigen::new(dm);
    let eval_unsorted = se.eigenvalues.as_slice();
    let evec_unsorted = se.eigenvectors;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eval_unsorted[a]
            .partial_cmp(&eval_unsorted[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut s = vec![0.0_f64; n];
    let mut u = if matches!(jobz, EighJobz::ValuesAndVectors) {
        Some(vec![0.0_f64; n * n])
    } else {
        None
    };
    for (col_new, &col_old) in order.iter().enumerate() {
        s[col_new] = eval_unsorted[col_old];
        if let Some(ref mut uu) = u {
            for row in 0..n {
                uu[row * n + col_new] = evec_unsorted[(row, col_old)];
            }
        }
    }
    Ok((s, u, "nalgebra"))
}

#[inline]
fn symmetric_eigh_f64_row_major_inplace_single_driver(
    a_row_major: &mut [f64],
    n: usize,
    driver: EighDriver,
    jobz: EighJobz,
) -> Result<(Vec<f64>, Option<Vec<f64>>, &'static str), String> {
    if a_row_major.len() != n.saturating_mul(n) {
        return Err(format!(
            "symmetric_eigh_f64_row_major_inplace: shape mismatch, len={} expected={}",
            a_row_major.len(),
            n.saturating_mul(n)
        ));
    }
    if n == 0 {
        let vecs = if matches!(jobz, EighJobz::ValuesAndVectors) {
            Some(Vec::new())
        } else {
            None
        };
        return Ok((Vec::new(), vecs, "empty"));
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        return lapack_try_driver_row_major_inplace(a_row_major, n, jobz, driver).map_err(|e| {
            format!(
                "inplace LAPACK eigh failed (driver={}, jobz={}): {e}",
                match driver {
                    EighDriver::Dsyevd => "dsyevd",
                    EighDriver::Dsyevr => "dsyevr",
                },
                eigh_jobz_label(jobz),
            )
        });
    }

    #[allow(unreachable_code)]
    Err(
        "inplace eigh requires LAPACK backend on macOS/Linux/Windows; unsupported target"
            .to_string(),
    )
}

pub(crate) fn symmetric_eigh_f64_row_major(
    a_row_major: &[f64],
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>, &'static str), String> {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    let driver = resolve_eigh_driver_from_env();
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let driver = EighDriver::Dsyevd;
    let (evals, evecs, backend) = symmetric_eigh_f64_row_major_with_driver(
        a_row_major,
        n,
        driver,
        EighJobz::ValuesAndVectors,
    )?;
    let vecs = evecs.ok_or_else(|| "eigh internal error: missing eigenvectors".to_string())?;
    Ok((evals, vecs, backend))
}

#[pyfunction]
#[pyo3(signature = (n=2048, threads=0, seed=20260501, driver=None, jobz="V", require_lapack=false))]
pub fn rust_eigh_debug_f64(
    n: usize,
    threads: usize,
    seed: u64,
    driver: Option<String>,
    jobz: &str,
    require_lapack: bool,
) -> PyResult<(
    String,
    String,
    String,
    usize,
    isize,
    isize,
    isize,
    bool,
    f64,
    f64,
    f64,
)> {
    if n == 0 {
        return Err(PyRuntimeError::new_err("n must be > 0"));
    }
    let driver_sel = parse_eigh_driver_token(driver.unwrap_or_else(|| "auto".to_string()).as_str());
    let jobz_sel = parse_eigh_jobz_token(jobz);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = rng.random::<f64>() - 0.5_f64;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }

    let backend = crate::blas::rust_sgemm_backend();
    let threads_before = crate::blas::rust_blas_get_num_threads();
    if threads > 0 {
        let _ = crate::blas::rust_blas_set_num_threads(threads);
    }
    let threads_in_stage = crate::blas::rust_blas_get_num_threads();

    let t0 = Instant::now();
    let run = symmetric_eigh_f64_row_major_with_driver(&a, n, driver_sel, jobz_sel);
    if threads > 0 && threads_before > 0 {
        let _ = crate::blas::rust_blas_set_num_threads(threads_before as usize);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let threads_after = crate::blas::rust_blas_get_num_threads();

    let (evals, _evecs, evd_backend) = run.map_err(PyRuntimeError::new_err)?;
    let lapack_used = is_lapack_backend(evd_backend);
    if require_lapack && !lapack_used {
        return Err(PyRuntimeError::new_err(format!(
            "rust_eigh_debug_f64 expected LAPACK backend, got {evd_backend}"
        )));
    }
    let min_eval = evals.first().copied().unwrap_or(f64::NAN);
    let max_eval = evals.last().copied().unwrap_or(f64::NAN);

    Ok((
        backend,
        evd_backend.to_string(),
        eigh_jobz_label(jobz_sel).to_string(),
        n,
        threads_before,
        threads_in_stage,
        threads_after,
        lapack_used,
        elapsed,
        min_eval,
        max_eval,
    ))
}

#[pyfunction]
#[pyo3(signature = (a, threads=0, driver=None, jobz="V", require_lapack=false))]
pub fn rust_eigh_from_array_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    threads: usize,
    driver: Option<String>,
    jobz: &str,
    require_lapack: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
    String,
    String,
    usize,
    isize,
    isize,
    isize,
    bool,
    f64,
)> {
    let a_view = a.as_array();
    let nrows = a_view.shape().first().copied().unwrap_or(0usize);
    let ncols = a_view.shape().get(1).copied().unwrap_or(0usize);
    if nrows == 0 || ncols == 0 || nrows != ncols {
        return Err(PyRuntimeError::new_err(format!(
            "rust_eigh_from_array_f64 expects a non-empty square matrix; got shape=({}, {})",
            nrows, ncols
        )));
    }
    let n = nrows;

    let driver_sel = parse_eigh_driver_token(driver.unwrap_or_else(|| "auto".to_string()).as_str());
    let jobz_sel = parse_eigh_jobz_token(jobz);

    let backend = crate::blas::rust_sgemm_backend();
    let threads_before = crate::blas::rust_blas_get_num_threads();
    if threads > 0 {
        let _ = crate::blas::rust_blas_set_num_threads(threads);
    }
    let threads_in_stage = crate::blas::rust_blas_get_num_threads();

    let t0 = Instant::now();
    let run = if let Some(a_contig) = a_view.as_slice_memory_order() {
        symmetric_eigh_f64_row_major_with_driver(a_contig, n, driver_sel, jobz_sel)
    } else {
        let a_tmp: Vec<f64> = a_view.iter().copied().collect();
        symmetric_eigh_f64_row_major_with_driver(&a_tmp, n, driver_sel, jobz_sel)
    };
    if threads > 0 && threads_before > 0 {
        let _ = crate::blas::rust_blas_set_num_threads(threads_before as usize);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let threads_after = crate::blas::rust_blas_get_num_threads();

    let (evals, evecs_opt, evd_backend) = run.map_err(PyRuntimeError::new_err)?;
    let lapack_used = is_lapack_backend(evd_backend);
    if require_lapack && !lapack_used {
        return Err(PyRuntimeError::new_err(format!(
            "rust_eigh_from_array_f64 expected LAPACK backend, got {evd_backend}"
        )));
    }

    let evals_arr = PyArray1::from_owned_array(py, Array1::from_vec(evals)).into_bound();
    let evecs_arr = match evecs_opt {
        Some(v) => Some(
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((n, n), v)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )
            .into_bound(),
        ),
        None => None,
    };

    Ok((
        evals_arr,
        evecs_arr,
        backend,
        evd_backend.to_string(),
        n,
        threads_before,
        threads_in_stage,
        threads_after,
        lapack_used,
        elapsed,
    ))
}

#[pyfunction]
#[pyo3(signature = (a, threads=0, driver=None, jobz="V", require_lapack=false))]
pub fn rust_eigh_from_array_f64_inplace<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
    threads: usize,
    driver: Option<String>,
    jobz: &str,
    require_lapack: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
    String,
    String,
    usize,
    isize,
    isize,
    isize,
    bool,
    f64,
)> {
    let shape = a.shape();
    let nrows = shape.first().copied().unwrap_or(0usize);
    let ncols = shape.get(1).copied().unwrap_or(0usize);
    if nrows == 0 || ncols == 0 || nrows != ncols {
        return Err(PyRuntimeError::new_err(format!(
            "rust_eigh_from_array_f64_inplace expects a non-empty square matrix; got shape=({}, {})",
            nrows, ncols
        )));
    }
    let n = nrows;
    let a_row_major: &mut [f64] = unsafe {
        a.as_slice_mut().map_err(|_| {
            PyRuntimeError::new_err(
                "rust_eigh_from_array_f64_inplace expects writable contiguous float64 matrix",
            )
        })?
    };

    let driver_sel = parse_eigh_driver_token(driver.unwrap_or_else(|| "auto".to_string()).as_str());
    let jobz_sel = parse_eigh_jobz_token(jobz);

    let backend = crate::blas::rust_sgemm_backend();
    let threads_before = crate::blas::rust_blas_get_num_threads();
    if threads > 0 {
        let _ = crate::blas::rust_blas_set_num_threads(threads);
    }
    let threads_in_stage = crate::blas::rust_blas_get_num_threads();

    let t0 = Instant::now();
    let run =
        symmetric_eigh_f64_row_major_inplace_single_driver(a_row_major, n, driver_sel, jobz_sel);
    if threads > 0 && threads_before > 0 {
        let _ = crate::blas::rust_blas_set_num_threads(threads_before as usize);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let threads_after = crate::blas::rust_blas_get_num_threads();

    let (evals, evecs_opt, evd_backend) = run.map_err(PyRuntimeError::new_err)?;
    let lapack_used = is_lapack_backend(evd_backend);
    if require_lapack && !lapack_used {
        return Err(PyRuntimeError::new_err(format!(
            "rust_eigh_from_array_f64_inplace expected LAPACK backend, got {evd_backend}"
        )));
    }

    let evals_arr = PyArray1::from_owned_array(py, Array1::from_vec(evals)).into_bound();
    let evecs_arr = match evecs_opt {
        Some(v) => Some(
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((n, n), v)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )
            .into_bound(),
        ),
        None => None,
    };

    Ok((
        evals_arr,
        evecs_arr,
        backend,
        evd_backend.to_string(),
        n,
        threads_before,
        threads_in_stage,
        threads_after,
        lapack_used,
        elapsed,
    ))
}
