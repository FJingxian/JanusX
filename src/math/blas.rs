use pyo3::prelude::*;
#[cfg(target_os = "macos")]
use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) type CblasInt = std::os::raw::c_int;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_COL_MAJOR: CblasInt = 102;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_ROW_MAJOR: CblasInt = 101;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_NO_TRANS: CblasInt = 111;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_TRANS: CblasInt = 112;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_UPPER: CblasInt = 121;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_LOWER: CblasInt = 122;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_LEFT: CblasInt = 141;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_DIAG_NON_UNIT: CblasInt = 131;

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
const HAS_OPENBLAS_BACKEND: bool = cfg!(all(feature = "blas-openblas", jx_openblas_available));
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
        // macOS policy:
        // - If Accelerate backend is compiled in, use it by default for BLAS
        //   (GRM/GEMM/SYRK-heavy paths).
        // - Otherwise, fall back to OpenBLAS when this wheel/build is
        //   OpenBLAS-only.
        if HAS_ACCELERATE_BACKEND {
            return SgemmBackend::Accelerate;
        }
        if HAS_OPENBLAS_BACKEND {
            return SgemmBackend::OpenBlas;
        }
        return SgemmBackend::Rust;
    }
    #[cfg(target_os = "windows")]
    {
        if HAS_OPENBLAS_BACKEND {
            return SgemmBackend::OpenBlas;
        }
        return SgemmBackend::Rust;
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
        "openblas" if HAS_OPENBLAS_BACKEND => SgemmBackend::OpenBlas,
        "accelerate" if HAS_ACCELERATE_BACKEND => SgemmBackend::Accelerate,
        "blas" if HAS_BLAS_BACKEND => SgemmBackend::Blas,
        "rust" => SgemmBackend::Rust,
        _ => default_backend,
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn selected_sgemm_backend() -> SgemmBackend {
    *SGEMM_BACKEND.get_or_init(resolve_sgemm_backend)
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn garfield_backend_guard_error(backend: SgemmBackend) -> Option<&'static str> {
    match backend {
        SgemmBackend::Rust => Some(
            "Garfield Max6 requires a linked BLAS backend; the pure-Rust BLAS fallback is disabled for this path",
        ),
        SgemmBackend::Accelerate | SgemmBackend::OpenBlas | SgemmBackend::Blas => None,
    }
}

/// Require a linked BLAS implementation for the performance-sensitive
/// Garfield Max6 paths. Other JanusX modules may still use the historical
/// Rust fallback, but Max6 and GRM-Max6 must never silently run through it.
pub(crate) fn ensure_garfield_blas_backend() -> Result<(), String> {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        let backend = selected_sgemm_backend();
        if let Some(message) = garfield_backend_guard_error(backend) {
            return Err(format!("{message} (selected backend: {backend:?})"));
        }
        return Ok(());
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        Err("Garfield Max6 requires a linked BLAS backend; unsupported target".to_string())
    }
}

// ---------------------------------------------------------------------------
// FFI declarations (deduplicated via macros across Accelerate / BLAS / OpenBLAS)
// ---------------------------------------------------------------------------

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[allow(unused_macros)]
macro_rules! declare_cblas_extern_block {
    (
        $(#[$block_attr:meta])*
        cblas: {
            sgemm => $sgemm:ident,
            dgemm => $dgemm:ident,
            dtrsm => $dtrsm:ident,
            ddot  => $ddot:ident,
            daxpy => $daxpy:ident,
            ssyrk => $ssyrk:ident,
            dsyrk => $dsyrk:ident,
        }
        $($extra:tt)*
    ) => {
        $(#[$block_attr])*
        unsafe extern "C" {
            #[link_name = "cblas_sgemm"]
            fn $sgemm(
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
            #[allow(dead_code)]
            fn $dgemm(
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
            #[link_name = "cblas_dtrsm"]
            fn $dtrsm(
                order: CblasInt,
                side: CblasInt,
                uplo: CblasInt,
                transa: CblasInt,
                diag: CblasInt,
                m: CblasInt,
                n: CblasInt,
                alpha: f64,
                a: *const f64,
                lda: CblasInt,
                b: *mut f64,
                ldb: CblasInt,
            );
            #[link_name = "cblas_ddot"]
            fn $ddot(
                n: CblasInt,
                x: *const f64,
                incx: CblasInt,
                y: *const f64,
                incy: CblasInt,
            ) -> f64;
            #[link_name = "cblas_daxpy"]
            fn $daxpy(
                n: CblasInt,
                alpha: f64,
                x: *const f64,
                incx: CblasInt,
                y: *mut f64,
                incy: CblasInt,
            );
            #[link_name = "cblas_ssyrk"]
            fn $ssyrk(
                order: CblasInt,
                uplo: CblasInt,
                trans: CblasInt,
                n: CblasInt,
                k: CblasInt,
                alpha: f32,
                a: *const f32,
                lda: CblasInt,
                beta: f32,
                c: *mut f32,
                ldc: CblasInt,
            );
            #[link_name = "cblas_dsyrk"]
            fn $dsyrk(
                order: CblasInt,
                uplo: CblasInt,
                trans: CblasInt,
                n: CblasInt,
                k: CblasInt,
                alpha: f64,
                a: *const f64,
                lda: CblasInt,
                beta: f64,
                c: *mut f64,
                ldc: CblasInt,
            );
            $($extra)*
        }
    };
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[allow(unused_macros)]
macro_rules! declare_openblas_extern {
    ($(#[$block_attr:meta])*) => {
        declare_cblas_extern_block! {
            $(#[$block_attr])*
            cblas: {
                sgemm => cblas_sgemm_openblas,
                dgemm => cblas_dgemm_openblas,
                dtrsm => cblas_dtrsm_openblas,
                ddot  => cblas_ddot_openblas,
                daxpy => cblas_daxpy_openblas,
                ssyrk => cblas_ssyrk_openblas,
                dsyrk => cblas_dsyrk_openblas,
            }
            #[cfg(jx_openblas_lapack_available)]
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
            #[cfg(jx_openblas_lapack_available)]
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
    };
}

// macOS Accelerate framework (linked when OpenBLAS is not statically configured)
#[cfg(all(
    target_os = "macos",
    not(all(feature = "blas-openblas", jx_openblas_available))
))]
declare_cblas_extern_block! {
    #[link(name = "Accelerate", kind = "framework")]
    cblas: {
        sgemm => cblas_sgemm_accelerate,
        dgemm => cblas_dgemm_accelerate,
        dtrsm => cblas_dtrsm_accelerate,
        ddot  => cblas_ddot_accelerate,
        daxpy => cblas_daxpy_accelerate,
        ssyrk => cblas_ssyrk_accelerate,
        dsyrk => cblas_dsyrk_accelerate,
    }
    // Legacy CLAPACK 3.2.1 fallback symbols kept for macOS < 13.3 compatibility;
    // on macOS 13.3+, `call_accelerate_dsyevd` / `call_accelerate_dsyevr`
    // dynamically resolve `dsyevd$NEWLAPACK` / `dsyevr$NEWLAPACK` first.
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd_accelerate_legacy(
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
    fn lapack_dsyevr_accelerate_legacy(
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

// Linux generic BLAS fallback
#[cfg(all(
    target_os = "linux",
    jx_blas_available,
    not(all(feature = "blas-openblas", jx_openblas_available))
))]
declare_cblas_extern_block! {
    #[link(name = "blas")]
    cblas: {
        sgemm => cblas_sgemm_blas,
        dgemm => cblas_dgemm_blas,
        dtrsm => cblas_dtrsm_blas,
        ddot  => cblas_ddot_blas,
        daxpy => cblas_daxpy_blas,
        ssyrk => cblas_ssyrk_blas,
        dsyrk => cblas_dsyrk_blas,
    }
}

// OpenBLAS link variants across macOS / Linux / Windows
#[cfg(all(
    target_os = "macos",
    feature = "blas-openblas",
    jx_openblas_available,
    not(jx_openblas_link_openblas0)
))]
declare_openblas_extern!(#[link(name = "openblas")]);

#[cfg(all(
    target_os = "macos",
    feature = "blas-openblas",
    jx_openblas_available,
    jx_openblas_link_openblas0
))]
declare_openblas_extern!(#[link(name = "openblas.0")]);

#[cfg(all(
    target_os = "linux",
    feature = "blas-openblas",
    jx_openblas_available,
    not(jx_openblas_link_openblas0)
))]
declare_openblas_extern!(#[link(name = "openblas")]);

#[cfg(all(
    target_os = "linux",
    feature = "blas-openblas",
    jx_openblas_available,
    jx_openblas_link_openblas0
))]
declare_openblas_extern!(#[link(name = ":libopenblas.so.0")]);

#[cfg(all(
    target_os = "windows",
    feature = "blas-openblas",
    jx_openblas_available,
    not(jx_openblas_link_openblas_plain)
))]
declare_openblas_extern!(
    #[cfg_attr(jx_openblas_static_link, link(name = "libopenblas", kind = "static"))]
    #[cfg_attr(not(jx_openblas_static_link), link(name = "libopenblas"))]
);

#[cfg(all(
    target_os = "windows",
    feature = "blas-openblas",
    jx_openblas_available,
    jx_openblas_link_openblas_plain
))]
declare_openblas_extern!(
    #[cfg_attr(jx_openblas_static_link, link(name = "openblas", kind = "static"))]
    #[cfg_attr(not(jx_openblas_static_link), link(name = "openblas"))]
);

// ---------------------------------------------------------------------------
// macOS Accelerate NEWLAPACK dynamic symbol resolution (macOS 13.3+)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
type LapackDsyevdFn = unsafe extern "C" fn(
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

#[cfg(target_os = "macos")]
type LapackDsyevrFn = unsafe extern "C" fn(
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

#[cfg(all(
    target_os = "macos",
    not(all(feature = "blas-openblas", jx_openblas_available))
))]
struct AccelerateLapackDyn {
    #[allow(dead_code)]
    handle: usize,
    dsyevd_newlapack: Option<LapackDsyevdFn>,
    dsyevr_newlapack: Option<LapackDsyevrFn>,
}

#[cfg(all(
    target_os = "macos",
    not(all(feature = "blas-openblas", jx_openblas_available))
))]
static ACCELERATE_LAPACK_DYN: OnceLock<AccelerateLapackDyn> = OnceLock::new();

#[cfg(all(
    target_os = "macos",
    not(all(feature = "blas-openblas", jx_openblas_available))
))]
fn accelerate_lapack_dyn() -> &'static AccelerateLapackDyn {
    ACCELERATE_LAPACK_DYN.get_or_init(|| unsafe {
        let path = b"/System/Library/Frameworks/Accelerate.framework/Accelerate\0";
        let handle = libc::dlopen(
            path.as_ptr() as *const std::os::raw::c_char,
            libc::RTLD_LAZY | libc::RTLD_LOCAL,
        );
        let lookup = if !handle.is_null() {
            handle
        } else {
            libc::RTLD_DEFAULT
        };
        let d_dsyevd = libc::dlsym(
            lookup,
            b"dsyevd$NEWLAPACK\0".as_ptr() as *const std::os::raw::c_char,
        );
        let d_dsyevr = libc::dlsym(
            lookup,
            b"dsyevr$NEWLAPACK\0".as_ptr() as *const std::os::raw::c_char,
        );
        AccelerateLapackDyn {
            handle: handle as usize,
            dsyevd_newlapack: if d_dsyevd.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut libc::c_void, LapackDsyevdFn>(
                    d_dsyevd,
                ))
            },
            dsyevr_newlapack: if d_dsyevr.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut libc::c_void, LapackDsyevrFn>(
                    d_dsyevr,
                ))
            },
        }
    })
}

#[cfg(all(
    target_os = "macos",
    not(all(feature = "blas-openblas", jx_openblas_available))
))]
#[inline]
fn accelerate_force_legacy_lapack() -> bool {
    std::env::var("JX_ACCELERATE_LEGACY_LAPACK")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(all(
    target_os = "macos",
    not(all(feature = "blas-openblas", jx_openblas_available))
))]
#[inline]
unsafe fn lapack_dsyevd_accelerate(
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
) {
    if !accelerate_force_legacy_lapack() {
        if let Some(f) = accelerate_lapack_dyn().dsyevd_newlapack {
            f(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
            return;
        }
    }
    lapack_dsyevd_accelerate_legacy(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
}

#[cfg(all(
    target_os = "macos",
    not(all(feature = "blas-openblas", jx_openblas_available))
))]
#[inline]
unsafe fn lapack_dsyevr_accelerate(
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
) {
    if !accelerate_force_legacy_lapack() {
        if let Some(f) = accelerate_lapack_dyn().dsyevr_newlapack {
            f(
                jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work,
                lwork, iwork, liwork, info,
            );
            return;
        }
    }
    lapack_dsyevr_accelerate_legacy(
        jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work, lwork,
        iwork, liwork, info,
    );
}

// ---------------------------------------------------------------------------
// Pure-Rust BLAS fallback implementations
// ---------------------------------------------------------------------------

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
#[allow(dead_code)]
fn checked_dgemm_matrix_span(
    order: CblasInt,
    rows: usize,
    columns: usize,
    leading_dimension: usize,
) -> Option<usize> {
    if rows == 0 || columns == 0 {
        return Some(0);
    }
    let minimum_leading_dimension = if order == CBLAS_ROW_MAJOR {
        columns
    } else if order == CBLAS_COL_MAJOR {
        rows
    } else {
        return None;
    };
    if leading_dimension < minimum_leading_dimension {
        return None;
    }
    if order == CBLAS_ROW_MAJOR {
        (rows - 1)
            .checked_mul(leading_dimension)
            .and_then(|offset| offset.checked_add(columns))
    } else {
        (columns - 1)
            .checked_mul(leading_dimension)
            .and_then(|offset| offset.checked_add(rows))
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
#[allow(dead_code)]
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
    assert!(
        m >= 0 && n >= 0 && k >= 0,
        "Rust DGEMM dimensions must be nonnegative"
    );
    assert!(
        lda >= 0 && ldb >= 0 && ldc >= 0,
        "Rust DGEMM leading dimensions must be nonnegative"
    );
    assert!(
        order == CBLAS_ROW_MAJOR || order == CBLAS_COL_MAJOR,
        "Rust DGEMM received an unsupported matrix order"
    );

    if m == 0 || n == 0 {
        return;
    }

    let (m, n, k) = (m as usize, n as usize, k as usize);
    let (lda, ldb, ldc) = (lda as usize, ldb as usize, ldc as usize);

    let c_span =
        checked_dgemm_matrix_span(order, m, n, ldc).expect("Rust DGEMM C matrix span overflow");
    let c_slice = std::slice::from_raw_parts_mut(c, c_span);

    if alpha == 0.0 || k == 0 {
        for col in 0..n {
            for row in 0..m {
                let idx = if order == CBLAS_ROW_MAJOR {
                    row * ldc + col
                } else {
                    row + col * ldc
                };
                c_slice[idx] = if beta == 0.0 {
                    0.0
                } else {
                    beta * c_slice[idx]
                };
            }
        }
        return;
    }

    let (a_rows, a_columns) = if transa == CBLAS_NO_TRANS {
        (m, k)
    } else {
        (k, m)
    };
    let (b_rows, b_columns) = if transb == CBLAS_NO_TRANS {
        (k, n)
    } else {
        (n, k)
    };
    let a_span = checked_dgemm_matrix_span(order, a_rows, a_columns, lda)
        .expect("Rust DGEMM A matrix span overflow");
    let b_span = checked_dgemm_matrix_span(order, b_rows, b_columns, ldb)
        .expect("Rust DGEMM B matrix span overflow");
    let a_slice = std::slice::from_raw_parts(a, a_span);
    let b_slice = std::slice::from_raw_parts(b, b_span);

    if order == CBLAS_ROW_MAJOR {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0_f64;
                for p in 0..k {
                    let av = if transa == CBLAS_NO_TRANS {
                        a_slice[row * lda + p]
                    } else {
                        a_slice[p * lda + row]
                    };
                    let bv = if transb == CBLAS_NO_TRANS {
                        b_slice[p * ldb + col]
                    } else {
                        b_slice[col * ldb + p]
                    };
                    acc += av * bv;
                }
                let idx = row * ldc + col;
                c_slice[idx] = if beta == 0.0 {
                    alpha * acc
                } else {
                    alpha * acc + beta * c_slice[idx]
                };
            }
        }
        return;
    }

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
            c_slice[idx] = if beta == 0.0 {
                alpha * acc
            } else {
                alpha * acc + beta * c_slice[idx]
            };
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_dtrsm_rust(
    order: CblasInt,
    side: CblasInt,
    uplo: CblasInt,
    transa: CblasInt,
    diag: CblasInt,
    m: CblasInt,
    n: CblasInt,
    alpha: f64,
    a: *const f64,
    lda: CblasInt,
    b: *mut f64,
    ldb: CblasInt,
) {
    assert_eq!(
        side, CBLAS_LEFT,
        "Rust DTRSM fallback currently supports left-side solves only"
    );
    assert_eq!(
        uplo, CBLAS_LOWER,
        "Rust DTRSM fallback currently supports lower triangles only"
    );
    assert_eq!(
        diag, CBLAS_DIAG_NON_UNIT,
        "Rust DTRSM fallback expects non-unit diagonals"
    );
    assert!(order == CBLAS_ROW_MAJOR || order == CBLAS_COL_MAJOR);
    assert!(m >= 0 && n >= 0 && lda >= m);
    let rows = m as usize;
    let columns = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    assert!(if order == CBLAS_ROW_MAJOR {
        ldb >= columns
    } else {
        ldb >= rows
    });
    let a_slice = std::slice::from_raw_parts(a, rows.saturating_mul(lda));
    let b_slice = std::slice::from_raw_parts_mut(b, rows.saturating_mul(ldb));
    assert!(transa == CBLAS_NO_TRANS || transa == CBLAS_TRANS);
    if order == CBLAS_ROW_MAJOR {
        if transa == CBLAS_NO_TRANS {
            for column in 0..columns {
                for row in 0..rows {
                    let mut value = alpha * b_slice[row * ldb + column];
                    for pivot in 0..row {
                        value -= a_slice[row * lda + pivot] * b_slice[pivot * ldb + column];
                    }
                    b_slice[row * ldb + column] = value / a_slice[row * lda + row];
                }
            }
        } else {
            for column in 0..columns {
                for row in (0..rows).rev() {
                    let mut value = alpha * b_slice[row * ldb + column];
                    for pivot in (row + 1)..rows {
                        value -= a_slice[pivot * lda + row] * b_slice[pivot * ldb + column];
                    }
                    b_slice[row * ldb + column] = value / a_slice[row * lda + row];
                }
            }
        }
    } else if transa == CBLAS_NO_TRANS {
        for column in 0..columns {
            for row in 0..rows {
                let mut value = alpha * b_slice[row + column * ldb];
                for pivot in 0..row {
                    value -= a_slice[row + pivot * lda] * b_slice[pivot + column * ldb];
                }
                b_slice[row + column * ldb] = value / a_slice[row + row * lda];
            }
        }
    } else {
        for column in 0..columns {
            for row in (0..rows).rev() {
                let mut value = alpha * b_slice[row + column * ldb];
                for pivot in (row + 1)..rows {
                    value -= a_slice[pivot + row * lda] * b_slice[pivot + column * ldb];
                }
                b_slice[row + column * ldb] = value / a_slice[row + row * lda];
            }
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_ddot_rust(
    n: CblasInt,
    x: *const f64,
    incx: CblasInt,
    y: *const f64,
    incy: CblasInt,
) -> f64 {
    if n <= 0 {
        return 0.0_f64;
    }
    assert!(
        incx > 0 && incy > 0,
        "Rust DDOT fallback expects positive increments"
    );
    let n_usize = n as usize;
    let incx_usize = incx as usize;
    let incy_usize = incy as usize;
    if incx_usize == 1 && incy_usize == 1 {
        let xs = std::slice::from_raw_parts(x, n_usize);
        let ys = std::slice::from_raw_parts(y, n_usize);
        xs.iter().zip(ys.iter()).map(|(a, b)| (*a) * (*b)).sum()
    } else {
        let xs = std::slice::from_raw_parts(x, n_usize.saturating_mul(incx_usize));
        let ys = std::slice::from_raw_parts(y, n_usize.saturating_mul(incy_usize));
        let mut acc = 0.0_f64;
        for i in 0..n_usize {
            acc += xs[i * incx_usize] * ys[i * incy_usize];
        }
        acc
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_daxpy_rust(
    n: CblasInt,
    alpha: f64,
    x: *const f64,
    incx: CblasInt,
    y: *mut f64,
    incy: CblasInt,
) {
    if n <= 0 || alpha == 0.0_f64 {
        return;
    }
    assert!(
        incx > 0 && incy > 0,
        "Rust DAXPY fallback expects positive increments"
    );
    let n_usize = n as usize;
    let incx_usize = incx as usize;
    let incy_usize = incy as usize;
    if incx_usize == 1 && incy_usize == 1 {
        let xs = std::slice::from_raw_parts(x, n_usize);
        let ys = std::slice::from_raw_parts_mut(y, n_usize);
        for i in 0..n_usize {
            ys[i] = alpha.mul_add(xs[i], ys[i]);
        }
    } else {
        let xs = std::slice::from_raw_parts(x, n_usize.saturating_mul(incx_usize));
        let ys = std::slice::from_raw_parts_mut(y, n_usize.saturating_mul(incy_usize));
        for i in 0..n_usize {
            let yi = i * incy_usize;
            ys[yi] = alpha.mul_add(xs[i * incx_usize], ys[yi]);
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_ssyrk_rust(
    order: CblasInt,
    uplo: CblasInt,
    trans: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f32,
    a: *const f32,
    lda: CblasInt,
    beta: f32,
    c: *mut f32,
    ldc: CblasInt,
) {
    assert_eq!(
        order, CBLAS_COL_MAJOR,
        "Rust SSYRK fallback expects column-major order"
    );
    let (n, k) = (n as usize, k as usize);
    let (lda, ldc) = (lda as usize, ldc as usize);
    let a_cols = if trans == CBLAS_NO_TRANS { k } else { n };
    let a_slice = std::slice::from_raw_parts(a, lda.saturating_mul(a_cols));
    let c_slice = std::slice::from_raw_parts_mut(c, ldc.saturating_mul(n));

    for col in 0..n {
        let row_start = if uplo == CBLAS_UPPER { 0 } else { col };
        let row_end = if uplo == CBLAS_UPPER { col + 1 } else { n };
        for row in row_start..row_end {
            let mut acc = 0.0_f32;
            for p in 0..k {
                let av = if trans == CBLAS_NO_TRANS {
                    a_slice[row + p * lda]
                } else {
                    a_slice[p + row * lda]
                };
                let bv = if trans == CBLAS_NO_TRANS {
                    a_slice[col + p * lda]
                } else {
                    a_slice[p + col * lda]
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
unsafe fn cblas_dsyrk_rust(
    order: CblasInt,
    uplo: CblasInt,
    trans: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f64,
    a: *const f64,
    lda: CblasInt,
    beta: f64,
    c: *mut f64,
    ldc: CblasInt,
) {
    assert_eq!(
        order, CBLAS_COL_MAJOR,
        "Rust DSYRK fallback expects column-major order"
    );
    let (n, k) = (n as usize, k as usize);
    let (lda, ldc) = (lda as usize, ldc as usize);
    let a_cols = if trans == CBLAS_NO_TRANS { k } else { n };
    let a_slice = std::slice::from_raw_parts(a, lda.saturating_mul(a_cols));
    let c_slice = std::slice::from_raw_parts_mut(c, ldc.saturating_mul(n));

    for col in 0..n {
        let row_start = if uplo == CBLAS_UPPER { 0 } else { col };
        let row_end = if uplo == CBLAS_UPPER { col + 1 } else { n };
        for row in row_start..row_end {
            let mut acc = 0.0_f64;
            for p in 0..k {
                let av = if trans == CBLAS_NO_TRANS {
                    a_slice[row + p * lda]
                } else {
                    a_slice[p + row * lda]
                };
                let bv = if trans == CBLAS_NO_TRANS {
                    a_slice[col + p * lda]
                } else {
                    a_slice[p + col * lda]
                };
                acc += av * bv;
            }
            let idx = row + col * ldc;
            c_slice[idx] = alpha * acc + beta * c_slice[idx];
        }
    }
}

// ---------------------------------------------------------------------------
// Unified CBLAS & LAPACK dispatch
// ---------------------------------------------------------------------------

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
macro_rules! dispatch_cblas {
    ($acc:ident, $ob:ident, $blas:ident, $rust:ident, ($($arg:expr),* $(,)?)) => {{
        match selected_sgemm_backend() {
            SgemmBackend::Accelerate => {
                #[cfg(all(
                    target_os = "macos",
                    not(all(feature = "blas-openblas", jx_openblas_available))
                ))]
                {
                    return $acc($($arg),*);
                }
                #[cfg(not(all(
                    target_os = "macos",
                    not(all(feature = "blas-openblas", jx_openblas_available))
                )))]
                {
                    return $rust($($arg),*);
                }
            }
            SgemmBackend::OpenBlas => {
                #[cfg(all(feature = "blas-openblas", jx_openblas_available))]
                {
                    return $ob($($arg),*);
                }
                #[cfg(not(all(feature = "blas-openblas", jx_openblas_available)))]
                {
                    return $rust($($arg),*);
                }
            }
            SgemmBackend::Blas => {
                #[cfg(all(
                    target_os = "linux",
                    jx_blas_available,
                    not(all(feature = "blas-openblas", jx_openblas_available))
                ))]
                {
                    return $blas($($arg),*);
                }
                #[cfg(not(all(
                    target_os = "linux",
                    jx_blas_available,
                    not(all(feature = "blas-openblas", jx_openblas_available))
                )))]
                {
                    return $rust($($arg),*);
                }
            }
            SgemmBackend::Rust => return $rust($($arg),*),
        }
    }};
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
    dispatch_cblas!(
        cblas_sgemm_accelerate,
        cblas_sgemm_openblas,
        cblas_sgemm_blas,
        cblas_sgemm_rust,
        (order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    );
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
#[allow(dead_code)]
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
    dispatch_cblas!(
        cblas_dgemm_accelerate,
        cblas_dgemm_openblas,
        cblas_dgemm_blas,
        cblas_dgemm_rust,
        (order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    );
}

/// Dispatch a row/column-major double-precision triangular solve to the
/// selected CBLAS backend. The Rust fallback intentionally implements only
/// the lower, left-side, non-unit case used by the GRM interaction scanner;
/// unsupported combinations fail loudly instead of silently changing the
/// solve semantics.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
#[allow(dead_code)]
pub(crate) unsafe fn cblas_dtrsm_dispatch(
    order: CblasInt,
    side: CblasInt,
    uplo: CblasInt,
    transa: CblasInt,
    diag: CblasInt,
    m: CblasInt,
    n: CblasInt,
    alpha: f64,
    a: *const f64,
    lda: CblasInt,
    b: *mut f64,
    ldb: CblasInt,
) {
    dispatch_cblas!(
        cblas_dtrsm_accelerate,
        cblas_dtrsm_openblas,
        cblas_dtrsm_blas,
        cblas_dtrsm_rust,
        (order, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
    );
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn cblas_ddot_dispatch(
    n: CblasInt,
    x: *const f64,
    incx: CblasInt,
    y: *const f64,
    incy: CblasInt,
) -> f64 {
    dispatch_cblas!(
        cblas_ddot_accelerate,
        cblas_ddot_openblas,
        cblas_ddot_blas,
        cblas_ddot_rust,
        (n, x, incx, y, incy)
    );
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn cblas_daxpy_dispatch(
    n: CblasInt,
    alpha: f64,
    x: *const f64,
    incx: CblasInt,
    y: *mut f64,
    incy: CblasInt,
) {
    dispatch_cblas!(
        cblas_daxpy_accelerate,
        cblas_daxpy_openblas,
        cblas_daxpy_blas,
        cblas_daxpy_rust,
        (n, alpha, x, incx, y, incy)
    );
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn cblas_ssyrk_dispatch(
    order: CblasInt,
    uplo: CblasInt,
    trans: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f32,
    a: *const f32,
    lda: CblasInt,
    beta: f32,
    c: *mut f32,
    ldc: CblasInt,
) {
    dispatch_cblas!(
        cblas_ssyrk_accelerate,
        cblas_ssyrk_openblas,
        cblas_ssyrk_blas,
        cblas_ssyrk_rust,
        (order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
    );
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn cblas_dsyrk_dispatch(
    order: CblasInt,
    uplo: CblasInt,
    trans: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f64,
    a: *const f64,
    lda: CblasInt,
    beta: f64,
    c: *mut f64,
    ldc: CblasInt,
) {
    dispatch_cblas!(
        cblas_dsyrk_accelerate,
        cblas_dsyrk_openblas,
        cblas_dsyrk_blas,
        cblas_dsyrk_rust,
        (order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
    );
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
    #[cfg(target_os = "macos")]
    {
        if should_try_openblas_lapack_on_macos() {
            if let Some(ob) = openblas_lapack_dyn() {
                let (prev_threads, prev_omp_threads) =
                    openblas_lapack_dyn_threads_guard(ob, preferred_openblas_thread_cap());
                (ob.dsyevd)(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
                openblas_lapack_dyn_threads_restore(ob, prev_threads, prev_omp_threads);
                return Ok(());
            }
            if matches!(mac_eigh_lapack_pref(), MacEighLapackPref::OpenBlas) {
                return Err("macOS OpenBLAS LAPACK requested but dynamic OpenBLAS loader failed");
            }
        }
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        lapack_dsyevd_accelerate(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
        return Ok(());
    }
    #[cfg(all(
        feature = "blas-openblas",
        jx_openblas_available,
        jx_openblas_lapack_available
    ))]
    {
        lapack_dsyevd_openblas(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
        return Ok(());
    }
    #[allow(unreachable_code)]
    {
        let _ = (jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
        Err("lapack_dsyevd backend unavailable")
    }
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
    #[cfg(target_os = "macos")]
    {
        if should_try_openblas_lapack_on_macos() {
            if let Some(ob) = openblas_lapack_dyn() {
                let (prev_threads, prev_omp_threads) =
                    openblas_lapack_dyn_threads_guard(ob, preferred_openblas_thread_cap());
                (ob.dsyevr)(
                    jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz,
                    work, lwork, iwork, liwork, info,
                );
                openblas_lapack_dyn_threads_restore(ob, prev_threads, prev_omp_threads);
                return Ok(());
            }
            if matches!(mac_eigh_lapack_pref(), MacEighLapackPref::OpenBlas) {
                return Err("macOS OpenBLAS LAPACK requested but dynamic OpenBLAS loader failed");
            }
        }
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
    #[cfg(all(
        feature = "blas-openblas",
        jx_openblas_available,
        jx_openblas_lapack_available
    ))]
    {
        lapack_dsyevr_openblas(
            jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work,
            lwork, iwork, liwork, info,
        );
        return Ok(());
    }
    #[allow(unreachable_code)]
    {
        let _ = (
            jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work,
            lwork, iwork, liwork, info,
        );
        Err("lapack_dsyevr backend unavailable")
    }
}

// ---------------------------------------------------------------------------
// Backend inspection & kernel preference helpers
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn rust_sgemm_backend() -> String {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        rust_sgemm_backend_tag().to_string()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "unsupported".to_string()
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) fn rust_sgemm_backend_tag() -> &'static str {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => "accelerate",
        SgemmBackend::OpenBlas => "openblas",
        SgemmBackend::Blas => "blas",
        SgemmBackend::Rust => "rust",
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) fn rust_sgemm_prefers_rayon_rowmajor_f32_kernel() -> bool {
    if let Ok(raw) = std::env::var("JX_ROWMAJOR_F32_KERNEL") {
        match raw.trim().to_ascii_lowercase().as_str() {
            "rayon" | "parallel" | "custom" => return true,
            "blas" | "gemm" | "serial" => return false,
            _ => {}
        }
    }
    #[cfg(target_os = "windows")]
    {
        matches!(
            selected_sgemm_backend(),
            SgemmBackend::OpenBlas | SgemmBackend::Blas | SgemmBackend::Rust
        )
    }
    #[cfg(not(target_os = "windows"))]
    {
        matches!(
            selected_sgemm_backend(),
            SgemmBackend::OpenBlas | SgemmBackend::Rust
        )
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
#[inline]
pub(crate) fn rust_sgemm_prefers_rayon_rowmajor_f32_kernel() -> bool {
    true
}

#[pyfunction]
pub fn rust_eigh_lapack_backend() -> String {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        rust_eigh_lapack_backend_tag().to_string()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "unsupported".to_string()
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) fn rust_eigh_lapack_backend_tag() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        if should_try_openblas_lapack_on_macos() && openblas_lapack_dyn().is_some() {
            return "openblas_dyn";
        }
    }
    rust_sgemm_backend_tag()
}

// ---------------------------------------------------------------------------
// macOS Accelerate threading API & dynamic OpenBLAS LAPACK loader
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[inline]
fn accelerate_blas_set_threading_mode(multithreaded: bool) -> Option<bool> {
    type BlasSetThreadingFn = unsafe extern "C" fn(std::os::raw::c_uint) -> std::os::raw::c_int;
    const BLAS_THREADING_MULTI_THREADED: std::os::raw::c_uint = 0;
    const BLAS_THREADING_SINGLE_THREADED: std::os::raw::c_uint = 1;
    unsafe {
        let fp = libc::dlsym(
            libc::RTLD_DEFAULT,
            b"BLASSetThreading\0".as_ptr() as *const std::os::raw::c_char,
        );
        if fp.is_null() {
            return None;
        }
        let f: BlasSetThreadingFn = std::mem::transmute(fp);
        let mode = if multithreaded {
            BLAS_THREADING_MULTI_THREADED
        } else {
            BLAS_THREADING_SINGLE_THREADED
        };
        Some(f(mode) == 0)
    }
}

#[cfg(not(target_os = "macos"))]
#[inline]
fn accelerate_blas_set_threading_mode(_multithreaded: bool) -> Option<bool> {
    None
}

#[cfg(target_os = "macos")]
#[inline]
fn accelerate_blas_get_threading_mode() -> Option<isize> {
    type BlasGetThreadingFn = unsafe extern "C" fn() -> std::os::raw::c_uint;
    // See Accelerate vecLib thread_api.h:
    // 0 => BLAS_THREADING_MULTI_THREADED, 1 => BLAS_THREADING_SINGLE_THREADED.
    const BLAS_THREADING_SINGLE_THREADED: std::os::raw::c_uint = 1;
    unsafe {
        let fp = libc::dlsym(
            libc::RTLD_DEFAULT,
            b"BLASGetThreading\0".as_ptr() as *const std::os::raw::c_char,
        );
        if fp.is_null() {
            return None;
        }
        let f: BlasGetThreadingFn = std::mem::transmute(fp);
        if f() == BLAS_THREADING_SINGLE_THREADED {
            Some(1_isize)
        } else {
            // For multi-threaded mode, thread count is decided internally by
            // Accelerate and capped by VECLIB_MAXIMUM_THREADS when set at startup.
            std::env::var("VECLIB_MAXIMUM_THREADS")
                .ok()
                .and_then(|s| s.trim().parse::<isize>().ok())
                .filter(|v| *v > 0)
                .or(Some(-1))
        }
    }
}

#[cfg(not(target_os = "macos"))]
#[inline]
fn accelerate_blas_get_threading_mode() -> Option<isize> {
    None
}

#[cfg(target_os = "macos")]
type OpenBlasSetThreadsFn = unsafe extern "C" fn(CblasInt);
#[cfg(target_os = "macos")]
type OpenBlasGetThreadsFn = unsafe extern "C" fn() -> CblasInt;
#[cfg(target_os = "macos")]
type OmpSetNumThreadsFn = unsafe extern "C" fn(std::os::raw::c_int);
#[cfg(target_os = "macos")]
type OmpGetMaxThreadsFn = unsafe extern "C" fn() -> std::os::raw::c_int;

#[cfg(target_os = "macos")]
struct OpenBlasLapackDyn {
    #[allow(dead_code)]
    handle: usize,
    dsyevd: LapackDsyevdFn,
    dsyevr: LapackDsyevrFn,
    set_threads: Option<OpenBlasSetThreadsFn>,
    get_threads: Option<OpenBlasGetThreadsFn>,
    omp_set_threads: Option<OmpSetNumThreadsFn>,
    omp_get_max_threads: Option<OmpGetMaxThreadsFn>,
}

#[cfg(target_os = "macos")]
static OPENBLAS_LAPACK_DYN: OnceLock<Option<OpenBlasLapackDyn>> = OnceLock::new();
#[cfg(target_os = "macos")]
static EIGH_OPENBLAS_DYN_THREAD_HINT: AtomicUsize = AtomicUsize::new(0);

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn _dyld_image_count() -> u32;
    fn _dyld_get_image_name(image_index: u32) -> *const std::os::raw::c_char;
}

#[cfg(target_os = "macos")]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MacEighLapackPref {
    Auto,
    Accelerate,
    OpenBlas,
}

#[cfg(target_os = "macos")]
#[inline]
fn mac_eigh_lapack_pref() -> MacEighLapackPref {
    let raw = std::env::var("JX_RUST_EIGH_LAPACK_BACKEND")
        .ok()
        .or_else(|| std::env::var("JX_RUST_LAPACK_BACKEND").ok())
        .unwrap_or_else(|| "auto".to_string())
        .trim()
        .to_ascii_lowercase();
    match raw.as_str() {
        "accelerate" | "veclib" => MacEighLapackPref::Accelerate,
        "openblas" => MacEighLapackPref::OpenBlas,
        _ => MacEighLapackPref::Auto,
    }
}

#[cfg(target_os = "macos")]
#[inline]
fn should_try_openblas_lapack_on_macos() -> bool {
    // Recheck loaded OpenMP runtimes: mixed libomp worker/barrier state can
    // segfault inside either runtime.
    if macos_has_multiple_openmp_runtimes_loaded() {
        return false;
    }
    match mac_eigh_lapack_pref() {
        MacEighLapackPref::OpenBlas => true,
        MacEighLapackPref::Accelerate => false,
        MacEighLapackPref::Auto => {
            macos_has_loaded_openblas_runtime() || !macos_has_foreign_openmp_runtime_loaded()
        }
    }
}

#[cfg(target_os = "macos")]
#[inline]
fn preferred_openblas_thread_cap() -> usize {
    let hinted = EIGH_OPENBLAS_DYN_THREAD_HINT.load(Ordering::SeqCst);
    if hinted > 0 {
        return hinted;
    }
    for key in [
        "JX_MLM_BLAS_THREADS",
        "OPENBLAS_NUM_THREADS",
        "JX_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ] {
        if let Ok(raw) = std::env::var(key) {
            if let Ok(v) = raw.trim().parse::<usize>() {
                if v > 0 {
                    return v;
                }
            }
        }
    }
    std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
}

#[cfg(target_os = "macos")]
#[inline]
fn push_unique_candidate(out: &mut Vec<String>, path: String) {
    if !path.is_empty() && !out.iter().any(|v| v == &path) {
        out.push(path);
    }
}

#[cfg(target_os = "macos")]
#[inline]
fn push_openblas_candidates_in_dir(out: &mut Vec<String>, dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut preferred = Vec::<String>::new();
    let mut others = Vec::<String>::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        let lname = name.to_ascii_lowercase();
        if !lname.starts_with("libopenblas") || !lname.ends_with(".dylib") {
            continue;
        }
        let path_s = path.to_string_lossy().to_string();
        if lname == "libopenblas.0.dylib" || lname == "libopenblas.dylib" {
            preferred.push(path_s);
        } else {
            others.push(path_s);
        }
    }
    preferred.sort_unstable();
    others.sort_unstable();
    for p in preferred.into_iter().chain(others.into_iter()) {
        push_unique_candidate(out, p);
    }
}

#[cfg(target_os = "macos")]
fn extension_module_dir() -> Option<PathBuf> {
    unsafe {
        let mut info: libc::Dl_info = std::mem::zeroed();
        let sym = rust_sgemm_backend as *const () as *const libc::c_void;
        if libc::dladdr(sym, &mut info as *mut libc::Dl_info) == 0 || info.dli_fname.is_null() {
            return None;
        }
        let c_path = std::ffi::CStr::from_ptr(info.dli_fname);
        PathBuf::from(c_path.to_string_lossy().to_string())
            .parent()
            .map(|p| p.to_path_buf())
    }
}

#[cfg(target_os = "macos")]
fn macos_openblas_bundle_dirs() -> Vec<PathBuf> {
    let mut out = Vec::<PathBuf>::new();
    if let Some(mod_dir) = extension_module_dir() {
        for rel in [".dylibs", ".libs", ""] {
            let probe = if rel.is_empty() {
                mod_dir.clone()
            } else {
                mod_dir.join(rel)
            };
            if probe.is_dir() && !out.iter().any(|p| p == &probe) {
                out.push(probe);
            }
        }
        if let Some(parent) = mod_dir.parent() {
            let sib = parent.join("janusx.libs");
            if sib.is_dir() && !out.iter().any(|p| p == &sib) {
                out.push(sib);
            }
        }
    }
    out
}

#[cfg(target_os = "macos")]
#[inline]
fn macos_is_openmp_runtime_name(name: &str) -> bool {
    let low = name.to_ascii_lowercase();
    low.starts_with("libomp") || low.starts_with("libgomp") || low.starts_with("libiomp")
}

#[cfg(target_os = "macos")]
fn macos_loaded_image_paths() -> Vec<PathBuf> {
    let mut out = Vec::<PathBuf>::new();
    let nimg = unsafe { _dyld_image_count() };
    for i in 0..nimg {
        let p = unsafe { _dyld_get_image_name(i) };
        if p.is_null() {
            continue;
        }
        let raw = unsafe { std::ffi::CStr::from_ptr(p) };
        let path = PathBuf::from(raw.to_string_lossy().to_string());
        if !out.iter().any(|v| v == &path) {
            out.push(path);
        }
    }
    out
}

#[cfg(target_os = "macos")]
fn macos_has_multiple_openmp_runtimes_loaded() -> bool {
    let mut seen = Vec::<PathBuf>::new();
    for path in macos_loaded_image_paths() {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !macos_is_openmp_runtime_name(name) {
            continue;
        }
        let canonical = std::fs::canonicalize(&path).unwrap_or(path);
        if !seen.contains(&canonical) {
            seen.push(canonical);
            if seen.len() > 1 {
                return true;
            }
        }
    }
    false
}

#[cfg(target_os = "macos")]
fn macos_loaded_openblas_candidates() -> Vec<String> {
    let mut out = Vec::<String>::new();
    for path in macos_loaded_image_paths() {
        let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        let low = name.to_ascii_lowercase();
        if low.ends_with(".dylib") && low.contains("openblas") {
            push_unique_candidate(&mut out, path.to_string_lossy().to_string());
        }
    }
    out
}

#[cfg(target_os = "macos")]
#[inline]
fn macos_has_loaded_openblas_runtime() -> bool {
    !macos_loaded_openblas_candidates().is_empty()
}

#[cfg(target_os = "macos")]
fn macos_has_foreign_openmp_runtime_loaded() -> bool {
    let bundle_dirs = macos_openblas_bundle_dirs();
    if bundle_dirs.is_empty() {
        return false;
    }
    for path in macos_loaded_image_paths() {
        let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        if macos_is_openmp_runtime_name(name)
            && !bundle_dirs.iter().any(|root| path.starts_with(root))
        {
            return true;
        }
    }
    false
}

#[cfg(target_os = "macos")]
fn openblas_lapack_candidates(pref: MacEighLapackPref) -> Vec<String> {
    let mut out = Vec::<String>::new();

    if matches!(pref, MacEighLapackPref::Auto) {
        for cand in macos_loaded_openblas_candidates() {
            push_unique_candidate(&mut out, cand);
        }
    }

    for key in ["JX_OPENBLAS_LIB_PATH", "OPENBLAS_LIB_PATH"] {
        if let Ok(v) = std::env::var(key) {
            let s = v.trim();
            if !s.is_empty() {
                let p = Path::new(s);
                if p.is_dir() {
                    push_openblas_candidates_in_dir(&mut out, p);
                } else {
                    push_unique_candidate(&mut out, s.to_string());
                }
            }
        }
    }

    if matches!(pref, MacEighLapackPref::OpenBlas) {
        for cand in macos_loaded_openblas_candidates() {
            push_unique_candidate(&mut out, cand);
        }
    }

    for probe_dir in macos_openblas_bundle_dirs() {
        push_openblas_candidates_in_dir(&mut out, &probe_dir);
    }

    if matches!(pref, MacEighLapackPref::OpenBlas) {
        if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
            let base = Path::new(&prefix).join("lib");
            for leaf in [
                "libopenblas.dylib",
                "libopenblas.0.dylib",
                "libopenblasp-r0.3.32.dylib",
                "libopenblas_armv8p-r0.3.32.dylib",
                "libopenblas_vortexp-r0.3.32.dylib",
            ] {
                let p = base.join(leaf);
                if p.exists() {
                    push_unique_candidate(&mut out, p.to_string_lossy().to_string());
                }
            }
        }
        for p in [
            "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
            "/opt/homebrew/opt/openblas/lib/libopenblas.0.dylib",
            "/usr/local/opt/openblas/lib/libopenblas.dylib",
            "/usr/local/opt/openblas/lib/libopenblas.0.dylib",
            "libopenblas.dylib",
            "libopenblas.0.dylib",
        ] {
            push_unique_candidate(&mut out, p.to_string());
        }
    }
    out
}

#[cfg(target_os = "macos")]
unsafe fn dlsym_opt<T>(handle: *mut libc::c_void, names: &[&[u8]]) -> Option<T> {
    for sym in names {
        let ptr = libc::dlsym(handle, sym.as_ptr() as *const std::os::raw::c_char);
        if !ptr.is_null() {
            return Some(std::mem::transmute_copy::<*mut libc::c_void, T>(&ptr));
        }
    }
    None
}

#[cfg(target_os = "macos")]
unsafe fn openblas_lapack_dyn_load_once() -> Option<OpenBlasLapackDyn> {
    let pref = mac_eigh_lapack_pref();
    let mut handle: *mut libc::c_void = std::ptr::null_mut();
    for cand in openblas_lapack_candidates(pref) {
        let Ok(cpath) = std::ffi::CString::new(cand.as_bytes()) else {
            continue;
        };
        let h = libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if !h.is_null() {
            handle = h;
            break;
        }
    }
    if handle.is_null() {
        return None;
    }

    let (Some(dsyevd), Some(dsyevr)) = (
        dlsym_opt::<LapackDsyevdFn>(handle, &[b"dsyevd_\0"]),
        dlsym_opt::<LapackDsyevrFn>(handle, &[b"dsyevr_\0"]),
    ) else {
        let _ = libc::dlclose(handle);
        return None;
    };

    Some(OpenBlasLapackDyn {
        handle: handle as usize,
        dsyevd,
        dsyevr,
        set_threads: dlsym_opt(
            handle,
            &[
                b"openblas_set_num_threads\0",
                b"openblas_set_num_threads_\0",
            ],
        ),
        get_threads: dlsym_opt(
            handle,
            &[
                b"openblas_get_num_threads\0",
                b"openblas_get_num_threads_\0",
            ],
        ),
        omp_set_threads: dlsym_opt(handle, &[b"omp_set_num_threads\0"]),
        omp_get_max_threads: dlsym_opt(handle, &[b"omp_get_max_threads\0"]),
    })
}

#[cfg(target_os = "macos")]
#[inline]
fn openblas_lapack_dyn() -> Option<&'static OpenBlasLapackDyn> {
    OPENBLAS_LAPACK_DYN
        .get_or_init(|| unsafe { openblas_lapack_dyn_load_once() })
        .as_ref()
}

#[cfg(target_os = "macos")]
#[inline]
pub(crate) fn prewarm_eigh_openblas_dyn_runtime() {
    if should_try_openblas_lapack_on_macos() {
        let _ = openblas_lapack_dyn();
    }
}

#[cfg(target_os = "macos")]
#[inline]
pub(crate) fn macos_eigh_uses_openblas_dyn() -> bool {
    should_try_openblas_lapack_on_macos() && openblas_lapack_dyn().is_some()
}

#[cfg(target_os = "macos")]
#[inline]
fn macos_openblas_dyn_current_threads() -> isize {
    let Some(ob) = openblas_lapack_dyn() else {
        return -1;
    };
    if let Some(getter) = ob.get_threads {
        let v = unsafe { getter() };
        if v > 0 {
            return v as isize;
        }
    }
    if let Some(getter) = ob.omp_get_max_threads {
        let v = unsafe { getter() };
        if v > 0 {
            return v as isize;
        }
    }
    -1
}

#[cfg(target_os = "macos")]
#[inline]
unsafe fn openblas_lapack_dyn_threads_guard(
    ob: &OpenBlasLapackDyn,
    target_threads: usize,
) -> (Option<usize>, Option<usize>) {
    let prev = ob.get_threads.and_then(|g| {
        let v = g();
        (v > 0).then_some(v as usize)
    });
    let prev_omp = ob.omp_get_max_threads.and_then(|g| {
        let v = g();
        (v > 0).then_some(v as usize)
    });
    let clamped = target_threads.max(1).min(i32::MAX as usize);
    if let Some(setter) = ob.omp_set_threads {
        setter(clamped as std::os::raw::c_int);
    }
    if let Some(setter) = ob.set_threads {
        setter(clamped as CblasInt);
    }
    (prev, prev_omp)
}

#[cfg(target_os = "macos")]
#[inline]
unsafe fn openblas_lapack_dyn_threads_restore(
    ob: &OpenBlasLapackDyn,
    prev_threads: Option<usize>,
    prev_omp_threads: Option<usize>,
) {
    if let (Some(prev), Some(setter)) = (prev_omp_threads, ob.omp_set_threads) {
        setter(prev.min(i32::MAX as usize) as std::os::raw::c_int);
    }
    if let (Some(prev), Some(setter)) = (prev_threads, ob.set_threads) {
        setter(prev.min(i32::MAX as usize) as CblasInt);
    }
}

// ---------------------------------------------------------------------------
// Thread guards & PyO3 thread control API
// ---------------------------------------------------------------------------

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) fn with_eigh_thread_stage<T, F>(threads: usize, f: F) -> (isize, isize, T, isize)
where
    F: FnOnce() -> T,
{
    #[cfg(target_os = "macos")]
    {
        prewarm_eigh_openblas_dyn_runtime();
        if macos_eigh_uses_openblas_dyn() {
            let before = macos_openblas_dyn_current_threads();
            let new_hint = if threads > 0 { threads.max(1) } else { 0 };
            let prev_hint = EIGH_OPENBLAS_DYN_THREAD_HINT.swap(new_hint, Ordering::SeqCst);
            let in_stage = if threads > 0 {
                threads as isize
            } else {
                before
            };
            let out = f();
            EIGH_OPENBLAS_DYN_THREAD_HINT.store(prev_hint, Ordering::SeqCst);
            let after = macos_openblas_dyn_current_threads();
            return (before, in_stage, out, after);
        }
    }

    let before = rust_blas_get_num_threads();
    let (in_stage, out, after) = {
        let _guard = BlasThreadGuard::enter(threads);
        let in_stage = rust_blas_get_num_threads();
        let out = f();
        let after = rust_blas_get_num_threads();
        (in_stage, out, after)
    };
    (before, in_stage, out, after)
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
#[inline]
pub(crate) fn with_eigh_thread_stage<T, F>(_threads: usize, f: F) -> (isize, isize, T, isize)
where
    F: FnOnce() -> T,
{
    (-1, -1, f(), -1)
}

#[cfg(all(feature = "blas-openblas", jx_openblas_available))]
#[inline]
fn rust_openblas_set_threads_impl(threads: usize) -> bool {
    unsafe {
        openblas_set_num_threads(threads.max(1).min(i32::MAX as usize) as CblasInt);
    }
    true
}

#[cfg(not(all(feature = "blas-openblas", jx_openblas_available)))]
#[inline]
fn rust_openblas_set_threads_impl(_threads: usize) -> bool {
    false
}

#[cfg(all(feature = "blas-openblas", jx_openblas_available))]
#[inline]
fn rust_openblas_get_threads_impl() -> Option<usize> {
    let v = unsafe { openblas_get_num_threads() };
    (v > 0).then_some(v as usize)
}

#[cfg(not(all(feature = "blas-openblas", jx_openblas_available)))]
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
        let active = target_threads > 0 && rust_openblas_set_threads_impl(target_threads);
        Self {
            prev_threads: prev,
            active,
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
impl Drop for OpenBlasThreadGuard {
    fn drop(&mut self) {
        if self.active {
            if let Some(prev) = self.prev_threads {
                let _ = rust_openblas_set_threads_impl(prev);
            }
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
enum BlasThreadState {
    OpenBlas { prev_threads: Option<usize> },
    Accelerate { prev_mode: Option<isize> },
    Other,
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) struct BlasThreadGuard {
    state: BlasThreadState,
    active: bool,
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
impl BlasThreadGuard {
    #[inline]
    pub(crate) fn enter(target_threads: usize) -> Self {
        if target_threads == 0 {
            return Self {
                state: BlasThreadState::Other,
                active: false,
            };
        }

        let backend = selected_sgemm_backend();
        let state = match backend {
            SgemmBackend::OpenBlas => BlasThreadState::OpenBlas {
                prev_threads: rust_openblas_get_threads_impl(),
            },
            SgemmBackend::Accelerate => BlasThreadState::Accelerate {
                prev_mode: accelerate_blas_get_threading_mode(),
            },
            _ => BlasThreadState::Other,
        };

        let t = target_threads.max(1);
        match backend {
            SgemmBackend::OpenBlas => {
                let _ = rust_openblas_set_threads_impl(t);
            }
            SgemmBackend::Accelerate => {
                let _ = accelerate_blas_set_threading_mode(t > 1);
            }
            _ => {}
        }

        Self {
            state,
            active: true,
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
impl Drop for BlasThreadGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        match self.state {
            BlasThreadState::OpenBlas { prev_threads } => {
                if let Some(prev) = prev_threads {
                    let _ = rust_openblas_set_threads_impl(prev);
                }
            }
            BlasThreadState::Accelerate { prev_mode } => match prev_mode {
                Some(1) => {
                    let _ = accelerate_blas_set_threading_mode(false);
                }
                Some(_) => {
                    let _ = accelerate_blas_set_threading_mode(true);
                }
                None => {}
            },
            BlasThreadState::Other => {}
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

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub(crate) struct BlasThreadGuard;

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
impl BlasThreadGuard {
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
        match selected_sgemm_backend() {
            SgemmBackend::OpenBlas => rust_openblas_set_threads_impl(t),
            SgemmBackend::Accelerate => accelerate_blas_set_threading_mode(t > 1).unwrap_or(true),
            _ => false,
        }
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
        match selected_sgemm_backend() {
            SgemmBackend::OpenBlas => rust_openblas_get_threads_impl()
                .map(|v| v as isize)
                .unwrap_or(-1),
            SgemmBackend::Accelerate => accelerate_blas_get_threading_mode().unwrap_or_else(|| {
                std::env::var("VECLIB_MAXIMUM_THREADS")
                    .ok()
                    .and_then(|s| s.trim().parse::<isize>().ok())
                    .filter(|x| *x > 0)
                    .unwrap_or(-1)
            }),
            _ => -1,
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        -1
    }
}

#[cfg(all(
    test,
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod tests {
    use super::*;

    #[test]
    fn garfield_blas_guard_rejects_rust_backend() {
        assert!(garfield_backend_guard_error(SgemmBackend::Rust).is_some());
        assert!(garfield_backend_guard_error(SgemmBackend::Accelerate).is_none());
        assert!(garfield_backend_guard_error(SgemmBackend::OpenBlas).is_none());
        assert!(garfield_backend_guard_error(SgemmBackend::Blas).is_none());
    }

    #[cfg(all(target_os = "linux", feature = "blas-openblas"))]
    #[test]
    fn linux_openblas_feature_does_not_use_rust_fallback() {
        assert_ne!(rust_sgemm_backend_tag(), "rust");
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    #[test]
    fn accelerate_newlapack_dsyevd_and_dsyevr_solve_symmetric_matrix() {
        // Verify dynamic resolution & numerical correctness of Accelerate LAPACK
        // (dsyevd$NEWLAPACK / dsyevr$NEWLAPACK on macOS 13.3+, legacy fallback on older macOS).
        let dyn_syms = accelerate_lapack_dyn();
        assert!(
            dyn_syms.dsyevd_newlapack.is_some() == dyn_syms.dsyevr_newlapack.is_some(),
            "Accelerate NEWLAPACK dsyevd and dsyevr availability should match"
        );

        let n: CblasInt = 2;
        let lda: CblasInt = 2;
        let jobz = [b'V' as std::os::raw::c_char, 0];
        let uplo = [b'U' as std::os::raw::c_char, 0];

        // dsyevd
        let mut a = vec![2.0_f64, 1.0, 1.0, 2.0];
        let mut w = vec![0.0_f64; 2];
        let mut work_q = [0.0_f64];
        let mut iwork_q = [0 as CblasInt];
        let mut lwork: CblasInt = -1;
        let mut liwork: CblasInt = -1;
        let mut info: CblasInt = 0;
        unsafe {
            lapack_dsyevd_accelerate(
                jobz.as_ptr(),
                uplo.as_ptr(),
                &n,
                a.as_mut_ptr(),
                &lda,
                w.as_mut_ptr(),
                work_q.as_mut_ptr(),
                &lwork,
                iwork_q.as_mut_ptr(),
                &liwork,
                &mut info,
            );
        }
        assert_eq!(info, 0);
        lwork = work_q[0].ceil() as CblasInt;
        liwork = iwork_q[0];
        let mut work = vec![0.0_f64; lwork as usize];
        let mut iwork = vec![0 as CblasInt; liwork as usize];
        unsafe {
            lapack_dsyevd_accelerate(
                jobz.as_ptr(),
                uplo.as_ptr(),
                &n,
                a.as_mut_ptr(),
                &lda,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                iwork.as_mut_ptr(),
                &liwork,
                &mut info,
            );
        }
        assert_eq!(info, 0);
        assert!((w[0] - 1.0).abs() < 1e-12);
        assert!((w[1] - 3.0).abs() < 1e-12);

        // dsyevr
        let mut a_r = vec![2.0_f64, 1.0, 1.0, 2.0];
        let mut w_r = vec![0.0_f64; 2];
        let mut z_r = vec![0.0_f64; 4];
        let mut isuppz = vec![0 as CblasInt; 4];
        let range = [b'A' as std::os::raw::c_char, 0];
        let vl = 0.0_f64;
        let vu = 0.0_f64;
        let il: CblasInt = 0;
        let iu: CblasInt = 0;
        let abstol = 0.0_f64;
        let mut m_found: CblasInt = 0;
        lwork = -1;
        liwork = -1;
        info = 0;
        unsafe {
            lapack_dsyevr_accelerate(
                jobz.as_ptr(),
                range.as_ptr(),
                uplo.as_ptr(),
                &n,
                a_r.as_mut_ptr(),
                &lda,
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                &mut m_found,
                w_r.as_mut_ptr(),
                z_r.as_mut_ptr(),
                &lda,
                isuppz.as_mut_ptr(),
                work_q.as_mut_ptr(),
                &lwork,
                iwork_q.as_mut_ptr(),
                &liwork,
                &mut info,
            );
        }
        assert_eq!(info, 0);
        lwork = work_q[0].ceil() as CblasInt;
        liwork = iwork_q[0];
        let mut work_r = vec![0.0_f64; lwork as usize];
        let mut iwork_r = vec![0 as CblasInt; liwork as usize];
        unsafe {
            lapack_dsyevr_accelerate(
                jobz.as_ptr(),
                range.as_ptr(),
                uplo.as_ptr(),
                &n,
                a_r.as_mut_ptr(),
                &lda,
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                &mut m_found,
                w_r.as_mut_ptr(),
                z_r.as_mut_ptr(),
                &lda,
                isuppz.as_mut_ptr(),
                work_r.as_mut_ptr(),
                &lwork,
                iwork_r.as_mut_ptr(),
                &liwork,
                &mut info,
            );
        }
        assert_eq!(info, 0);
        assert_eq!(m_found, 2);
        assert!((w_r[0] - 1.0).abs() < 1e-12);
        assert!((w_r[1] - 3.0).abs() < 1e-12);
    }

    fn matrix_storage(matrix: &[Vec<f64>], order: CblasInt) -> Vec<f64> {
        let rows = matrix.len();
        let columns = matrix[0].len();
        if order == CBLAS_ROW_MAJOR {
            matrix.iter().flat_map(|row| row.iter().copied()).collect()
        } else {
            (0..columns)
                .flat_map(|column| (0..rows).map(move |row| matrix[row][column]))
                .collect()
        }
    }

    fn padded_matrix_storage(
        matrix: &[Vec<f64>],
        order: CblasInt,
        leading_dimension: usize,
    ) -> Vec<f64> {
        let rows = matrix.len();
        let columns = matrix[0].len();
        let span = checked_dgemm_matrix_span(order, rows, columns, leading_dimension).unwrap();
        let mut storage = vec![f64::NAN; span];
        for row in 0..rows {
            for column in 0..columns {
                let index = if order == CBLAS_ROW_MAJOR {
                    row * leading_dimension + column
                } else {
                    row + column * leading_dimension
                };
                storage[index] = matrix[row][column];
            }
        }
        storage
    }

    #[test]
    fn checked_dgemm_matrix_span_stops_at_final_accessed_element() {
        assert_eq!(checked_dgemm_matrix_span(CBLAS_ROW_MAJOR, 2, 3, 5), Some(8));
        assert_eq!(
            checked_dgemm_matrix_span(CBLAS_COL_MAJOR, 2, 3, 5),
            Some(12)
        );
        assert_eq!(checked_dgemm_matrix_span(CBLAS_ROW_MAJOR, 2, 3, 2), None);
        assert_eq!(checked_dgemm_matrix_span(CBLAS_COL_MAJOR, 2, 3, 1), None);
        assert_eq!(checked_dgemm_matrix_span(999, 2, 3, 5), None);
        assert_eq!(checked_dgemm_matrix_span(CBLAS_ROW_MAJOR, 0, 3, 5), Some(0));
        assert_eq!(checked_dgemm_matrix_span(CBLAS_COL_MAJOR, 2, 0, 5), Some(0));
    }

    #[test]
    fn rust_dgemm_fallback_supports_padded_leading_dimensions() {
        let m = 2;
        let n = 3;
        let k = 2;
        let logical_a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let logical_b = vec![vec![5.0, 6.0, 7.0], vec![8.0, 9.0, 10.0]];
        let initial_c = vec![vec![0.5, 1.0, 1.5], vec![2.0, 2.5, 3.0]];

        for &order in &[CBLAS_ROW_MAJOR, CBLAS_COL_MAJOR] {
            for &transa in &[CBLAS_NO_TRANS, CBLAS_TRANS] {
                for &transb in &[CBLAS_NO_TRANS, CBLAS_TRANS] {
                    let physical_a = if transa == CBLAS_NO_TRANS {
                        logical_a.clone()
                    } else {
                        (0..k)
                            .map(|row| (0..m).map(|column| logical_a[column][row]).collect())
                            .collect()
                    };
                    let physical_b = if transb == CBLAS_NO_TRANS {
                        logical_b.clone()
                    } else {
                        (0..n)
                            .map(|row| (0..k).map(|column| logical_b[column][row]).collect())
                            .collect()
                    };
                    let lda = 5;
                    let ldb = 6;
                    let ldc = 7;
                    let a = padded_matrix_storage(&physical_a, order, lda);
                    let b = padded_matrix_storage(&physical_b, order, ldb);
                    let mut c = padded_matrix_storage(&initial_c, order, ldc);

                    unsafe {
                        cblas_dgemm_rust(
                            order,
                            transa,
                            transb,
                            m as CblasInt,
                            n as CblasInt,
                            k as CblasInt,
                            1.5,
                            a.as_ptr(),
                            lda as CblasInt,
                            b.as_ptr(),
                            ldb as CblasInt,
                            -0.25,
                            c.as_mut_ptr(),
                            ldc as CblasInt,
                        );
                    }

                    for row in 0..m {
                        for column in 0..n {
                            let product = (0..k)
                                .map(|p| logical_a[row][p] * logical_b[p][column])
                                .sum::<f64>();
                            let index = if order == CBLAS_ROW_MAJOR {
                                row * ldc + column
                            } else {
                                row + column * ldc
                            };
                            let expected = 1.5 * product - 0.25 * initial_c[row][column];
                            assert_eq!(c[index], expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rust_dgemm_fallback_alpha_zero_skips_nan_operands_and_honors_beta() {
        let m = 2;
        let n = 2;
        let k = 3;
        let initial_c = vec![vec![1.5, f64::NAN], vec![-2.0, 4.25]];

        for &order in &[CBLAS_ROW_MAJOR, CBLAS_COL_MAJOR] {
            for &transa in &[CBLAS_NO_TRANS, CBLAS_TRANS] {
                for &transb in &[CBLAS_NO_TRANS, CBLAS_TRANS] {
                    let physical_a = if transa == CBLAS_NO_TRANS {
                        vec![vec![f64::NAN; k]; m]
                    } else {
                        vec![vec![f64::NAN; m]; k]
                    };
                    let physical_b = if transb == CBLAS_NO_TRANS {
                        vec![vec![f64::NAN; n]; k]
                    } else {
                        vec![vec![f64::NAN; k]; n]
                    };
                    let lda = 5;
                    let ldb = 6;
                    let ldc = 7;
                    let a = padded_matrix_storage(&physical_a, order, lda);
                    let b = padded_matrix_storage(&physical_b, order, ldb);
                    let mut c = padded_matrix_storage(&initial_c, order, ldc);

                    unsafe {
                        cblas_dgemm_rust(
                            order,
                            transa,
                            transb,
                            m as CblasInt,
                            n as CblasInt,
                            k as CblasInt,
                            0.0,
                            a.as_ptr(),
                            lda as CblasInt,
                            b.as_ptr(),
                            ldb as CblasInt,
                            2.0,
                            c.as_mut_ptr(),
                            ldc as CblasInt,
                        );
                    }

                    let expected = [[3.0, f64::NAN], [-4.0, 8.5]];
                    for row in 0..m {
                        for column in 0..n {
                            let index = if order == CBLAS_ROW_MAJOR {
                                row * ldc + column
                            } else {
                                row + column * ldc
                            };
                            if expected[row][column].is_nan() {
                                assert!(c[index].is_nan());
                            } else {
                                assert_eq!(c[index], expected[row][column]);
                            }
                        }
                    }

                    let mut c =
                        vec![f64::NAN; checked_dgemm_matrix_span(order, m, n, ldc).unwrap()];
                    unsafe {
                        cblas_dgemm_rust(
                            order,
                            transa,
                            transb,
                            m as CblasInt,
                            n as CblasInt,
                            k as CblasInt,
                            0.0,
                            std::ptr::null(),
                            lda as CblasInt,
                            std::ptr::null(),
                            ldb as CblasInt,
                            0.0,
                            c.as_mut_ptr(),
                            ldc as CblasInt,
                        );
                    }
                    for row in 0..m {
                        for column in 0..n {
                            let index = if order == CBLAS_ROW_MAJOR {
                                row * ldc + column
                            } else {
                                row + column * ldc
                            };
                            assert_eq!(c[index], 0.0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rust_dgemm_fallback_k_zero_skips_operands_and_scales_c() {
        let m = 2;
        let n = 3;
        let ldc = 7;
        let initial_c = vec![vec![1.5, f64::NAN, -2.0], vec![4.25, 0.0, 3.0]];

        for &order in &[CBLAS_ROW_MAJOR, CBLAS_COL_MAJOR] {
            let mut c = padded_matrix_storage(&initial_c, order, ldc);
            unsafe {
                cblas_dgemm_rust(
                    order,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    m as CblasInt,
                    n as CblasInt,
                    0,
                    f64::NAN,
                    std::ptr::null(),
                    1,
                    std::ptr::null(),
                    1,
                    -0.5,
                    c.as_mut_ptr(),
                    ldc as CblasInt,
                );
            }

            let expected = [[-0.75, f64::NAN, 1.0], [-2.125, -0.0, -1.5]];
            for row in 0..m {
                for column in 0..n {
                    let index = if order == CBLAS_ROW_MAJOR {
                        row * ldc + column
                    } else {
                        row + column * ldc
                    };
                    if expected[row][column].is_nan() {
                        assert!(c[index].is_nan());
                    } else {
                        assert_eq!(c[index], expected[row][column]);
                    }
                }
            }
        }
    }

    #[test]
    fn rust_dgemm_fallback_zero_dimensions_do_not_dereference_inputs() {
        unsafe {
            cblas_dgemm_rust(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                0,
                2,
                3,
                1.0,
                std::ptr::null(),
                3,
                std::ptr::null(),
                2,
                1.0,
                std::ptr::null_mut(),
                2,
            );
            cblas_dgemm_rust(
                CBLAS_COL_MAJOR,
                CBLAS_TRANS,
                CBLAS_TRANS,
                2,
                0,
                3,
                f64::NAN,
                std::ptr::null(),
                3,
                std::ptr::null(),
                2,
                f64::NAN,
                std::ptr::null_mut(),
                2,
            );
        }
    }

    #[test]
    fn rust_dgemm_fallback_supports_cblas_layouts_transposes_and_scalars() {
        let m = 2;
        let n = 3;
        let k = 2;
        let logical_a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let logical_b = vec![vec![5.0, 6.0, 7.0], vec![8.0, 9.0, 10.0]];
        let initial_c = vec![vec![0.5, 1.0, 1.5], vec![2.0, 2.5, 3.0]];
        let expected = vec![vec![41.75, 47.5, 53.25], vec![93.0, 106.75, 120.5]];

        for &order in &[CBLAS_ROW_MAJOR, CBLAS_COL_MAJOR] {
            for &transa in &[CBLAS_NO_TRANS, CBLAS_TRANS] {
                for &transb in &[CBLAS_NO_TRANS, CBLAS_TRANS] {
                    let physical_a = if transa == CBLAS_NO_TRANS {
                        logical_a.clone()
                    } else {
                        (0..k)
                            .map(|row| (0..m).map(|column| logical_a[column][row]).collect())
                            .collect()
                    };
                    let physical_b = if transb == CBLAS_NO_TRANS {
                        logical_b.clone()
                    } else {
                        (0..n)
                            .map(|row| (0..k).map(|column| logical_b[column][row]).collect())
                            .collect()
                    };
                    let a = matrix_storage(&physical_a, order);
                    let b = matrix_storage(&physical_b, order);
                    let mut c = matrix_storage(&initial_c, order);
                    let lda = if order == CBLAS_ROW_MAJOR {
                        physical_a[0].len()
                    } else {
                        physical_a.len()
                    } as CblasInt;
                    let ldb = if order == CBLAS_ROW_MAJOR {
                        physical_b[0].len()
                    } else {
                        physical_b.len()
                    } as CblasInt;
                    let ldc = if order == CBLAS_ROW_MAJOR { n } else { m } as CblasInt;

                    unsafe {
                        cblas_dgemm_rust(
                            order,
                            transa,
                            transb,
                            m as CblasInt,
                            n as CblasInt,
                            k as CblasInt,
                            2.0,
                            a.as_ptr(),
                            lda,
                            b.as_ptr(),
                            ldb,
                            -0.5,
                            c.as_mut_ptr(),
                            ldc,
                        );
                    }

                    for row in 0..m {
                        for column in 0..n {
                            let actual = if order == CBLAS_ROW_MAJOR {
                                c[row * ldc as usize + column]
                            } else {
                                c[row + column * ldc as usize]
                            };
                            assert!(
                                (actual - expected[row][column]).abs() < 1.0e-12,
                                "order={order}, transa={transa}, transb={transb}, ({row}, {column}): actual={actual:.16e}, expected={:.16e}",
                                expected[row][column]
                            );
                        }
                    }
                }
            }
        }

        let expected_product = vec![vec![21.0, 24.0, 27.0], vec![47.0, 54.0, 61.0]];
        for &order in &[CBLAS_ROW_MAJOR, CBLAS_COL_MAJOR] {
            let a = matrix_storage(&logical_a, order);
            let b = matrix_storage(&logical_b, order);
            let mut c = vec![f64::NAN; m * n];
            let lda = if order == CBLAS_ROW_MAJOR { k } else { m } as CblasInt;
            let ldb = if order == CBLAS_ROW_MAJOR { n } else { k } as CblasInt;
            let ldc = if order == CBLAS_ROW_MAJOR { n } else { m } as CblasInt;

            unsafe {
                cblas_dgemm_rust(
                    order,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    m as CblasInt,
                    n as CblasInt,
                    k as CblasInt,
                    1.0,
                    a.as_ptr(),
                    lda,
                    b.as_ptr(),
                    ldb,
                    0.0,
                    c.as_mut_ptr(),
                    ldc,
                );
            }

            for row in 0..m {
                for column in 0..n {
                    let actual = if order == CBLAS_ROW_MAJOR {
                        c[row * ldc as usize + column]
                    } else {
                        c[row + column * ldc as usize]
                    };
                    assert_eq!(actual, expected_product[row][column]);
                }
            }
        }
    }
}
