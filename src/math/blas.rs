use pyo3::prelude::*;
use std::sync::OnceLock;

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) type CblasInt = std::os::raw::c_int;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_COL_MAJOR: CblasInt = 102;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_NO_TRANS: CblasInt = 111;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_TRANS: CblasInt = 112;

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SgemmBackend {
    Accelerate,
    OpenBlas,
    Blas,
    Rust,
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
static SGEMM_BACKEND: OnceLock<SgemmBackend> = OnceLock::new();

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const HAS_OPENBLAS_BACKEND: bool = cfg!(any(
    target_os = "windows",
    all(feature = "blas-openblas", jx_openblas_available)
));
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const HAS_ACCELERATE_BACKEND: bool = cfg!(all(
    target_os = "macos",
    not(all(feature = "blas-openblas", jx_openblas_available))
));
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const HAS_BLAS_BACKEND: bool = cfg!(all(
    target_os = "linux",
    jx_blas_available,
    not(all(feature = "blas-openblas", jx_openblas_available))
));

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn default_sgemm_backend() -> SgemmBackend {
    #[cfg(target_os = "macos")]
    {
        if HAS_OPENBLAS_BACKEND {
            return SgemmBackend::OpenBlas;
        }
        return SgemmBackend::Accelerate;
    }
    #[cfg(target_os = "windows")]
    {
        return SgemmBackend::OpenBlas;
    }
    #[cfg(target_os = "linux")]
    {
        if HAS_OPENBLAS_BACKEND {
            return SgemmBackend::OpenBlas;
        }
        if HAS_BLAS_BACKEND {
            return SgemmBackend::Blas;
        }
        return SgemmBackend::Rust;
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn resolve_sgemm_backend() -> SgemmBackend {
    let default_backend = default_sgemm_backend();
    let req = std::env::var("JX_RUST_BLAS_BACKEND")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "auto".to_string());
    match req.as_str() {
        "" | "auto" => default_backend,
        "openblas" => {
            if HAS_OPENBLAS_BACKEND {
                SgemmBackend::OpenBlas
            } else {
                default_backend
            }
        }
        "accelerate" => {
            if HAS_ACCELERATE_BACKEND {
                SgemmBackend::Accelerate
            } else {
                default_backend
            }
        }
        "blas" => {
            if HAS_BLAS_BACKEND {
                SgemmBackend::Blas
            } else {
                default_backend
            }
        }
        "rust" => SgemmBackend::Rust,
        _ => default_backend,
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn selected_sgemm_backend() -> SgemmBackend {
    *SGEMM_BACKEND.get_or_init(resolve_sgemm_backend)
}

#[cfg(all(
    target_os = "macos",
    not(all(feature = "blas-openblas", jx_openblas_available))
))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_accelerate(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_accelerate(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd_accelerate(
        jobz: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        w: *mut f64,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    #[link_name = "dsyevr_"]
    fn lapack_dsyevr_accelerate(
        jobz: *const std::os::raw::c_char,
        range: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        vl: *const f64,
        vu: *const f64,
        il: *const CblasInt,
        iu: *const CblasInt,
        abstol: *const f64,
        m: *mut CblasInt,
        w: *mut f64,
        z: *mut f64,
        ldz: *const CblasInt,
        isuppz: *mut CblasInt,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
}

#[cfg(all(
    target_os = "linux",
    jx_blas_available,
    not(all(feature = "blas-openblas", jx_openblas_available))
))]
#[link(name = "blas")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_blas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_blas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
}

#[cfg(all(
    target_os = "macos",
    feature = "blas-openblas",
    jx_openblas_available,
    not(jx_openblas_link_openblas0)
))]
#[link(name = "openblas")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd_openblas(
        jobz: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        w: *mut f64,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    #[link_name = "dsyevr_"]
    fn lapack_dsyevr_openblas(
        jobz: *const std::os::raw::c_char,
        range: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        vl: *const f64,
        vu: *const f64,
        il: *const CblasInt,
        iu: *const CblasInt,
        abstol: *const f64,
        m: *mut CblasInt,
        w: *mut f64,
        z: *mut f64,
        ldz: *const CblasInt,
        isuppz: *mut CblasInt,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    fn openblas_set_num_threads(num_threads: CblasInt);
    fn openblas_get_num_threads() -> CblasInt;
}

#[cfg(all(
    target_os = "macos",
    feature = "blas-openblas",
    jx_openblas_available,
    jx_openblas_link_openblas0
))]
#[link(name = "openblas.0")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd_openblas(
        jobz: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        w: *mut f64,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    #[link_name = "dsyevr_"]
    fn lapack_dsyevr_openblas(
        jobz: *const std::os::raw::c_char,
        range: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        vl: *const f64,
        vu: *const f64,
        il: *const CblasInt,
        iu: *const CblasInt,
        abstol: *const f64,
        m: *mut CblasInt,
        w: *mut f64,
        z: *mut f64,
        ldz: *const CblasInt,
        isuppz: *mut CblasInt,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    fn openblas_set_num_threads(num_threads: CblasInt);
    fn openblas_get_num_threads() -> CblasInt;
}

#[cfg(all(
    target_os = "linux",
    feature = "blas-openblas",
    jx_openblas_available,
    not(jx_openblas_link_openblas0)
))]
#[link(name = "openblas")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd_openblas(
        jobz: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        w: *mut f64,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    #[link_name = "dsyevr_"]
    fn lapack_dsyevr_openblas(
        jobz: *const std::os::raw::c_char,
        range: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        vl: *const f64,
        vu: *const f64,
        il: *const CblasInt,
        iu: *const CblasInt,
        abstol: *const f64,
        m: *mut CblasInt,
        w: *mut f64,
        z: *mut f64,
        ldz: *const CblasInt,
        isuppz: *mut CblasInt,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    fn openblas_set_num_threads(num_threads: CblasInt);
    fn openblas_get_num_threads() -> CblasInt;
}

#[cfg(all(
    target_os = "linux",
    feature = "blas-openblas",
    jx_openblas_available,
    jx_openblas_link_openblas0
))]
#[link(name = "openblas.0")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd_openblas(
        jobz: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        w: *mut f64,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    #[link_name = "dsyevr_"]
    fn lapack_dsyevr_openblas(
        jobz: *const std::os::raw::c_char,
        range: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        vl: *const f64,
        vu: *const f64,
        il: *const CblasInt,
        iu: *const CblasInt,
        abstol: *const f64,
        m: *mut CblasInt,
        w: *mut f64,
        z: *mut f64,
        ldz: *const CblasInt,
        isuppz: *mut CblasInt,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    fn openblas_set_num_threads(num_threads: CblasInt);
    fn openblas_get_num_threads() -> CblasInt;
}

#[cfg(all(target_os = "windows", not(jx_openblas_link_openblas_plain)))]
#[link(name = "libopenblas", kind = "static")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    fn openblas_set_num_threads(num_threads: CblasInt);
    fn openblas_get_num_threads() -> CblasInt;
}

#[cfg(all(target_os = "windows", jx_openblas_link_openblas_plain))]
#[link(name = "openblas", kind = "static")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    fn openblas_set_num_threads(num_threads: CblasInt);
    fn openblas_get_num_threads() -> CblasInt;
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_sgemm_rust(
    order: CblasInt,
    transa: CblasInt,
    transb: CblasInt,
    m: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f32,
    a: *const f32,
    lda: CblasInt,
    b: *const f32,
    ldb: CblasInt,
    beta: f32,
    c: *mut f32,
    ldc: CblasInt,
) {
    assert_eq!(
        order, CBLAS_COL_MAJOR,
        "Rust SGEMM fallback expects column-major order"
    );
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let (lda, ldb, ldc) = (lda as usize, ldb as usize, ldc as usize);

    let a_cols = if transa == CBLAS_NO_TRANS { k } else { m };
    let b_cols = if transb == CBLAS_NO_TRANS { n } else { k };
    let a_slice = std::slice::from_raw_parts(a, lda.saturating_mul(a_cols));
    let b_slice = std::slice::from_raw_parts(b, ldb.saturating_mul(b_cols));
    let c_slice = std::slice::from_raw_parts_mut(c, ldc.saturating_mul(n));

    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0_f32;
            for p in 0..k {
                let av = if transa == CBLAS_NO_TRANS {
                    a_slice[row + p * lda]
                } else {
                    a_slice[p + row * lda]
                };
                let bv = if transb == CBLAS_NO_TRANS {
                    b_slice[p + col * ldb]
                } else {
                    b_slice[col + p * ldb]
                };
                acc += av * bv;
            }
            let idx = row + col * ldc;
            c_slice[idx] = alpha * acc + beta * c_slice[idx];
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn cblas_sgemm_dispatch(
    order: CblasInt,
    transa: CblasInt,
    transb: CblasInt,
    m: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f32,
    a: *const f32,
    lda: CblasInt,
    b: *const f32,
    ldb: CblasInt,
    beta: f32,
    c: *mut f32,
    ldc: CblasInt,
) {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(all(
                target_os = "macos",
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_sgemm_accelerate(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(any(
                target_os = "windows",
                all(feature = "blas-openblas", jx_openblas_available)
            ))]
            {
                cblas_sgemm_openblas(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::Blas => {
            #[cfg(all(
                target_os = "linux",
                jx_blas_available,
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_sgemm_blas(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::Rust => {
            cblas_sgemm_rust(
                order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            );
            return;
        }
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_sgemm_accelerate(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(target_os = "linux", feature = "blas-openblas", jx_openblas_available))]
    {
        cblas_sgemm_openblas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        target_os = "linux",
        jx_blas_available,
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_sgemm_blas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        target_os = "linux",
        not(jx_blas_available),
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_sgemm_rust(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(target_os = "windows")]
    {
        cblas_sgemm_openblas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_dgemm_rust(
    order: CblasInt,
    transa: CblasInt,
    transb: CblasInt,
    m: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f64,
    a: *const f64,
    lda: CblasInt,
    b: *const f64,
    ldb: CblasInt,
    beta: f64,
    c: *mut f64,
    ldc: CblasInt,
) {
    assert_eq!(
        order, CBLAS_COL_MAJOR,
        "Rust DGEMM fallback expects column-major order"
    );
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let (lda, ldb, ldc) = (lda as usize, ldb as usize, ldc as usize);

    let a_cols = if transa == CBLAS_NO_TRANS { k } else { m };
    let b_cols = if transb == CBLAS_NO_TRANS { n } else { k };
    let a_slice = std::slice::from_raw_parts(a, lda.saturating_mul(a_cols));
    let b_slice = std::slice::from_raw_parts(b, ldb.saturating_mul(b_cols));
    let c_slice = std::slice::from_raw_parts_mut(c, ldc.saturating_mul(n));

    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0_f64;
            for p in 0..k {
                let av = if transa == CBLAS_NO_TRANS {
                    a_slice[row + p * lda]
                } else {
                    a_slice[p + row * lda]
                };
                let bv = if transb == CBLAS_NO_TRANS {
                    b_slice[p + col * ldb]
                } else {
                    b_slice[col + p * ldb]
                };
                acc += av * bv;
            }
            let idx = row + col * ldc;
            c_slice[idx] = alpha * acc + beta * c_slice[idx];
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn cblas_dgemm_dispatch(
    order: CblasInt,
    transa: CblasInt,
    transb: CblasInt,
    m: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f64,
    a: *const f64,
    lda: CblasInt,
    b: *const f64,
    ldb: CblasInt,
    beta: f64,
    c: *mut f64,
    ldc: CblasInt,
) {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(all(
                target_os = "macos",
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_dgemm_accelerate(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(any(
                target_os = "windows",
                all(feature = "blas-openblas", jx_openblas_available)
            ))]
            {
                cblas_dgemm_openblas(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::Blas => {
            #[cfg(all(
                target_os = "linux",
                jx_blas_available,
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_dgemm_blas(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::Rust => {
            cblas_dgemm_rust(
                order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            );
            return;
        }
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_dgemm_accelerate(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(target_os = "linux", feature = "blas-openblas", jx_openblas_available))]
    {
        cblas_dgemm_openblas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        target_os = "linux",
        jx_blas_available,
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_dgemm_blas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        target_os = "linux",
        not(jx_blas_available),
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_dgemm_rust(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(target_os = "windows")]
    {
        cblas_dgemm_openblas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn lapack_dsyevd_dispatch(
    jobz: *const std::os::raw::c_char,
    uplo: *const std::os::raw::c_char,
    n: *const CblasInt,
    a: *mut f64,
    lda: *const CblasInt,
    w: *mut f64,
    work: *mut f64,
    lwork: *const CblasInt,
    iwork: *mut CblasInt,
    liwork: *const CblasInt,
    info: *mut CblasInt,
) -> Result<(), &'static str> {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(all(
                target_os = "macos",
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                lapack_dsyevd_accelerate(
                    jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info,
                );
                return Ok(());
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(all(
                any(target_os = "macos", target_os = "linux"),
                feature = "blas-openblas",
                jx_openblas_available
            ))]
            {
                lapack_dsyevd_openblas(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
                return Ok(());
            }
        }
        SgemmBackend::Blas | SgemmBackend::Rust => {}
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        lapack_dsyevd_accelerate(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
        return Ok(());
    }
    #[cfg(all(target_os = "linux", feature = "blas-openblas", jx_openblas_available))]
    {
        lapack_dsyevd_openblas(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
        return Ok(());
    }
    #[cfg(target_os = "windows")]
    {
        let _ = (jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
        return Err("lapack_dsyevd unavailable on this Windows build");
    }
    #[allow(unreachable_code)]
    Err("lapack_dsyevd backend unavailable")
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn lapack_dsyevr_dispatch(
    jobz: *const std::os::raw::c_char,
    range: *const std::os::raw::c_char,
    uplo: *const std::os::raw::c_char,
    n: *const CblasInt,
    a: *mut f64,
    lda: *const CblasInt,
    vl: *const f64,
    vu: *const f64,
    il: *const CblasInt,
    iu: *const CblasInt,
    abstol: *const f64,
    m: *mut CblasInt,
    w: *mut f64,
    z: *mut f64,
    ldz: *const CblasInt,
    isuppz: *mut CblasInt,
    work: *mut f64,
    lwork: *const CblasInt,
    iwork: *mut CblasInt,
    liwork: *const CblasInt,
    info: *mut CblasInt,
) -> Result<(), &'static str> {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(all(
                target_os = "macos",
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                lapack_dsyevr_accelerate(
                    jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz,
                    work, lwork, iwork, liwork, info,
                );
                return Ok(());
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(all(
                any(target_os = "macos", target_os = "linux"),
                feature = "blas-openblas",
                jx_openblas_available
            ))]
            {
                lapack_dsyevr_openblas(
                    jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz,
                    work, lwork, iwork, liwork, info,
                );
                return Ok(());
            }
        }
        SgemmBackend::Blas | SgemmBackend::Rust => {}
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        lapack_dsyevr_accelerate(
            jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work,
            lwork, iwork, liwork, info,
        );
        return Ok(());
    }
    #[cfg(all(target_os = "linux", feature = "blas-openblas", jx_openblas_available))]
    {
        lapack_dsyevr_openblas(
            jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work,
            lwork, iwork, liwork, info,
        );
        return Ok(());
    }
    #[cfg(target_os = "windows")]
    {
        let _ = (
            jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work,
            lwork, iwork, liwork, info,
        );
        return Err("lapack_dsyevr unavailable on this Windows build");
    }
    #[allow(unreachable_code)]
    Err("lapack_dsyevr backend unavailable")
}

#[pyfunction]
pub fn rust_sgemm_backend() -> String {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        match selected_sgemm_backend() {
            SgemmBackend::Accelerate => "accelerate".to_string(),
            SgemmBackend::OpenBlas => "openblas".to_string(),
            SgemmBackend::Blas => "blas".to_string(),
            SgemmBackend::Rust => "rust".to_string(),
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "unsupported".to_string()
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn apply_blas_thread_env_hints(threads: usize) {
    let t = threads.max(1).to_string();
    for key in [
        "OMP_NUM_THREADS",
        "OMP_MAX_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OPENBLAS_MAX_THREADS",
        "MKL_NUM_THREADS",
        "MKL_MAX_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "JX_MLM_BLAS_THREADS",
    ] {
        std::env::set_var(key, &t);
    }
}

#[cfg(any(
    target_os = "windows",
    all(feature = "blas-openblas", jx_openblas_available)
))]
#[inline]
fn rust_openblas_set_threads_impl(threads: usize) -> bool {
    unsafe {
        let t = (threads.max(1)).min(i32::MAX as usize) as CblasInt;
        openblas_set_num_threads(t);
    }
    true
}

#[cfg(not(any(
    target_os = "windows",
    all(feature = "blas-openblas", jx_openblas_available)
)))]
#[inline]
fn rust_openblas_set_threads_impl(_threads: usize) -> bool {
    false
}

#[cfg(any(
    target_os = "windows",
    all(feature = "blas-openblas", jx_openblas_available)
))]
#[inline]
fn rust_openblas_get_threads_impl() -> Option<usize> {
    unsafe {
        let v = openblas_get_num_threads();
        if v > 0 {
            return Some(v as usize);
        }
    }
    None
}

#[cfg(not(any(
    target_os = "windows",
    all(feature = "blas-openblas", jx_openblas_available)
)))]
#[inline]
fn rust_openblas_get_threads_impl() -> Option<usize> {
    None
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) struct OpenBlasThreadGuard {
    prev_threads: Option<usize>,
    active: bool,
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
impl OpenBlasThreadGuard {
    #[inline]
    pub(crate) fn enter(target_threads: usize) -> Self {
        let prev = rust_openblas_get_threads_impl();
        let mut active = false;
        if target_threads > 0 && rust_openblas_set_threads_impl(target_threads) {
            active = true;
        }
        Self {
            prev_threads: prev,
            active,
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
impl Drop for OpenBlasThreadGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Some(prev) = self.prev_threads {
            let _ = rust_openblas_set_threads_impl(prev);
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub(crate) struct OpenBlasThreadGuard;

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
impl OpenBlasThreadGuard {
    #[inline]
    pub(crate) fn enter(_target_threads: usize) -> Self {
        Self
    }
}

#[pyfunction]
#[pyo3(signature = (threads))]
pub fn rust_blas_set_num_threads(threads: usize) -> bool {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        let t = threads.max(1);
        apply_blas_thread_env_hints(t);
        let backend = selected_sgemm_backend();
        if backend != SgemmBackend::OpenBlas {
            return false;
        }
        return rust_openblas_set_threads_impl(t);
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = threads;
        false
    }
}

#[pyfunction]
pub fn rust_blas_get_num_threads() -> isize {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        let backend = selected_sgemm_backend();
        if backend != SgemmBackend::OpenBlas {
            return -1;
        }
        return rust_openblas_get_threads_impl()
            .map(|v| v as isize)
            .unwrap_or(-1);
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        -1
    }
}
