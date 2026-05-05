use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn env_flag(name: &str) -> bool {
    match env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

fn env_flag_default_true(name: &str) -> bool {
    match env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !matches!(s.as_str(), "0" | "false" | "no" | "off")
        }
        Err(_) => true,
    }
}

fn emit_verbose_note(msg: impl AsRef<str>) {
    if env_flag("JANUSX_BUILD_VERBOSE") {
        println!("cargo:warning={}", msg.as_ref());
    }
}

fn has_library_with_prefix(lib_dir_s: &str, prefix: &str) -> bool {
    let Ok(entries) = fs::read_dir(lib_dir_s) else {
        return false;
    };
    let pfx = prefix.to_ascii_lowercase();
    for entry in entries.flatten() {
        let Some(name_raw) = entry.file_name().to_str().map(|s| s.to_ascii_lowercase()) else {
            continue;
        };
        if !name_raw.starts_with(&pfx) {
            continue;
        }
        let is_lib = name_raw.ends_with(".dylib")
            || name_raw.contains(".dylib.")
            || name_raw.ends_with(".so")
            || name_raw.contains(".so.")
            || name_raw.ends_with(".a")
            || name_raw.ends_with(".lib")
            || name_raw.ends_with(".dll")
            || name_raw.ends_with(".dll.a");
        if is_lib {
            return true;
        }
    }
    false
}

fn maybe_link_blas_family_from_dir(lib_dir_s: &str, source_label: &str) {
    // Explicitly link liblapack/libblas when available in the same prefix
    // (typically conda's lib directory). OpenBLAS itself is still linked by
    // the Rust extern blocks, including versioned-soname handling.
    if !env_flag_default_true("JANUSX_LINK_BLAS_FAMILY") {
        return;
    }
    let has_lapack = has_library_with_prefix(lib_dir_s, "liblapack")
        || has_library_with_prefix(lib_dir_s, "lapack");
    let has_blas =
        has_library_with_prefix(lib_dir_s, "libblas") || has_library_with_prefix(lib_dir_s, "blas");

    let mut linked: Vec<&str> = Vec::new();
    if has_lapack {
        println!("cargo:rustc-link-lib=dylib=lapack");
        linked.push("lapack");
    }
    if has_blas {
        println!("cargo:rustc-link-lib=dylib=blas");
        linked.push("blas");
    }
    if !linked.is_empty() {
        emit_verbose_note(format!(
            "BLAS family linkage via {source_label}: openblas + {}.",
            linked.join(" + ")
        ));
    }
}

fn maybe_prebuild_kmc_bind() {
    println!("cargo:rerun-if-env-changed=JANUSX_PREBUILD_KMC_BIND");
    println!("cargo:rerun-if-env-changed=JANUSX_STRICT_KMC_BIND");
    println!("cargo:rerun-if-env-changed=JANUSX_KMC_SRC");
    println!("cargo:rerun-if-env-changed=JANUSX_BUILD_VERBOSE");
    println!("cargo:rerun-if-env-changed=JANUSX_KMC_REBUILD");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=PYTHON");
    println!("cargo:rerun-if-changed=python/janusx/native/build_kmc_bind.py");
    println!("cargo:rerun-if-changed=python/janusx/native/kmc_count_bind.cpp");

    if matches!(env::var("JANUSX_PREBUILD_KMC_BIND"), Ok(v) if v.trim() == "0") {
        emit_verbose_note("Skip KMC prebuild (JANUSX_PREBUILD_KMC_BIND=0).");
        return;
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let script_path = Path::new(&manifest_dir)
        .join("python")
        .join("janusx")
        .join("native")
        .join("build_kmc_bind.py");
    if !script_path.exists() {
        println!(
            "cargo:warning=KMC prebuild script not found: {}",
            script_path.to_string_lossy()
        );
        return;
    }

    let python = env::var("PYO3_PYTHON")
        .or_else(|_| env::var("PYTHON_SYS_EXECUTABLE"))
        .or_else(|_| env::var("PYTHON"))
        .unwrap_or_else(|_| "python3".to_string());

    let mut cmd = Command::new(&python);
    cmd.current_dir(&manifest_dir).arg(script_path.as_os_str());
    if let Ok(kmc_src) = env::var("JANUSX_KMC_SRC") {
        if !kmc_src.trim().is_empty() {
            cmd.arg("--kmc-src").arg(kmc_src);
        }
    }
    if env_flag("JANUSX_BUILD_VERBOSE") {
        cmd.arg("--verbose");
    }
    if env_flag("JANUSX_KMC_REBUILD") {
        cmd.arg("--rebuild");
    }

    match cmd.output() {
        Ok(out) if out.status.success() => {
            emit_verbose_note("KMC bind prebuild finished.");
            let stdout = String::from_utf8_lossy(&out.stdout);
            let msg = stdout.trim();
            if !msg.is_empty() {
                for line in msg.lines() {
                    emit_verbose_note(line);
                }
            }
        }
        Ok(out) => {
            let status = out.status;
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let msg = format!(
                "KMC bind prebuild failed (status: {}). Will fallback to runtime build.",
                status
            );
            if !stdout.trim().is_empty() {
                for line in stdout.lines() {
                    println!("cargo:warning=[kmc-prebuild stdout] {line}");
                }
            }
            if !stderr.trim().is_empty() {
                for line in stderr.lines() {
                    println!("cargo:warning=[kmc-prebuild stderr] {line}");
                }
            }
            if env_flag("JANUSX_STRICT_KMC_BIND") {
                panic!("{msg}");
            }
            println!("cargo:warning={msg}");
        }
        Err(err) => {
            let msg = format!(
                "KMC bind prebuild command failed to start ({err}). Will fallback to runtime build."
            );
            if env_flag("JANUSX_STRICT_KMC_BIND") {
                panic!("{msg}");
            }
            println!("cargo:warning={msg}");
        }
    }
}

fn configure_openblas_from_dir(lib_dir_s: &str, source_label: &str) -> bool {
    if !Path::new(lib_dir_s).exists() {
        return false;
    }

    let has_win_libopenblas = Path::new(lib_dir_s).join("libopenblas.lib").exists()
        || Path::new(lib_dir_s).join("libopenblas.dll.a").exists()
        || Path::new(lib_dir_s).join("libopenblas.a").exists()
        || Path::new(lib_dir_s).join("libopenblas.dll").exists();
    let has_win_openblas = Path::new(lib_dir_s).join("openblas.lib").exists()
        || Path::new(lib_dir_s).join("openblas.dll.a").exists()
        || Path::new(lib_dir_s).join("openblas.a").exists()
        || Path::new(lib_dir_s).join("openblas.dll").exists();

    // Conda/macOS can expose versioned soname only (e.g. libopenblas.0.dylib)
    // without a generic libopenblas.dylib symlink.
    let has_generic = Path::new(lib_dir_s).join("libopenblas.dylib").exists()
        || Path::new(lib_dir_s).join("libopenblas.so").exists()
        || Path::new(lib_dir_s).join("libopenblas.a").exists()
        // Windows (vcpkg/conda) layouts.
        || has_win_libopenblas
        || has_win_openblas;
    let has_versioned = Path::new(lib_dir_s).join("libopenblas.0.dylib").exists()
        || Path::new(lib_dir_s).join("libopenblas.so.0").exists();
    if !has_generic && !has_versioned {
        return false;
    }
    println!("cargo:rustc-link-search=native={lib_dir_s}");
    if cfg!(any(target_os = "linux", target_os = "macos")) {
        // Ensure wheel-repair tools can resolve @rpath-linked dependencies
        // (e.g. libopenblas.0.dylib -> libiconv.2.dylib in conda environments).
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir_s}");
    }
    maybe_link_blas_family_from_dir(lib_dir_s, source_label);
    println!("cargo:rustc-cfg=jx_openblas_available");
    if cfg!(target_os = "windows") && has_win_openblas && !has_win_libopenblas {
        println!("cargo:rustc-cfg=jx_openblas_link_openblas_plain");
    }
    if !has_generic && has_versioned {
        println!("cargo:rustc-cfg=jx_openblas_link_openblas0");
        emit_verbose_note(format!(
            "OpenBLAS detected via {source_label} with versioned soname only; linking as openblas.0."
        ));
    } else {
        emit_verbose_note(format!(
            "OpenBLAS enabled via {source_label}; backend auto can use openblas."
        ));
    }
    true
}

fn main() {
    maybe_prebuild_kmc_bind();

    // Allow conditional compilation checks for our custom cfg.
    println!("cargo:rustc-check-cfg=cfg(jx_openblas_available)");
    println!("cargo:rustc-check-cfg=cfg(jx_openblas_link_openblas0)");
    println!("cargo:rustc-check-cfg=cfg(jx_openblas_link_openblas_plain)");
    println!("cargo:rustc-check-cfg=cfg(jx_blas_available)");
    println!("cargo:rerun-if-env-changed=JANUSX_REQUIRE_OPENBLAS");
    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=OPENBLAS_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=BLAS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=LIBRARY_LIB");
    println!("cargo:rerun-if-env-changed=LIBRARY_PREFIX");
    println!("cargo:rerun-if-env-changed=JANUSX_LINK_BLAS_FAMILY");
    let require_openblas = env_flag("JANUSX_REQUIRE_OPENBLAS");

    // Only probe OpenBLAS when user explicitly enabled the feature.
    if env::var_os("CARGO_FEATURE_BLAS_OPENBLAS").is_none() {
        if require_openblas {
            panic!(
                "JANUSX_REQUIRE_OPENBLAS=1 but feature `blas-openblas` is not enabled. \
Enable default features or pass --features blas-openblas."
            );
        }
        if cfg!(target_os = "linux") {
            if let Some(lib_dir) = env::var_os("BLAS_LIB_DIR") {
                let lib_dir_s = lib_dir.to_string_lossy().to_string();
                if Path::new(&lib_dir_s).exists() {
                    println!("cargo:rustc-link-search=native={lib_dir_s}");
                    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir_s}");
                    println!("cargo:rustc-cfg=jx_blas_available");
                    println!("cargo:warning=BLAS enabled via BLAS_LIB_DIR.");
                    return;
                }
            }
            if pkg_config::Config::new()
                .cargo_metadata(true)
                .probe("blas")
                .is_ok()
            {
                println!("cargo:rustc-cfg=jx_blas_available");
                println!("cargo:warning=BLAS detected by pkg-config.");
            } else {
                println!(
                    "cargo:warning=No BLAS/OpenBLAS detected on Linux; falling back to pure Rust SGEMM."
                );
            }
        }
        return;
    }

    // Manual override via environment (useful for conda/local installs).
    // - OPENBLAS_LIB_DIR: directory containing libopenblas.*
    // - OPENBLAS_INCLUDE_DIR: optional include dir
    if let Some(lib_dir) = env::var_os("OPENBLAS_LIB_DIR") {
        let lib_dir_s = lib_dir.to_string_lossy().to_string();
        if configure_openblas_from_dir(&lib_dir_s, "OPENBLAS_LIB_DIR") {
            if let Some(include_dir) = env::var_os("OPENBLAS_INCLUDE_DIR") {
                let include_dir_s = include_dir.to_string_lossy().to_string();
                if Path::new(&include_dir_s).exists() {
                    println!("cargo:include={include_dir_s}");
                }
            }
            return;
        }
    }

    // Conda environments may provide OpenBLAS without pkg-config metadata.
    // On Windows, libraries are usually in %CONDA_PREFIX%/Library/lib.
    if let Some(conda_prefix) = env::var_os("CONDA_PREFIX") {
        let mut candidates: Vec<(String, String)> = Vec::new();
        let p = Path::new(&conda_prefix);
        candidates.push((
            p.join("lib").to_string_lossy().to_string(),
            "CONDA_PREFIX/lib".to_string(),
        ));
        candidates.push((
            p.join("Library").join("lib").to_string_lossy().to_string(),
            "CONDA_PREFIX/Library/lib".to_string(),
        ));
        for (lib_dir_s, label) in candidates {
            if configure_openblas_from_dir(&lib_dir_s, &label) {
                let conda_inc = if label.contains("Library") {
                    Path::new(&conda_prefix).join("Library").join("include")
                } else {
                    Path::new(&conda_prefix).join("include")
                };
                if conda_inc.exists() {
                    println!("cargo:include={}", conda_inc.to_string_lossy());
                }
                return;
            }
        }
    }

    // Windows conda-build style helper envs.
    if let Some(library_lib) = env::var_os("LIBRARY_LIB") {
        let library_lib_s = library_lib.to_string_lossy().to_string();
        if configure_openblas_from_dir(&library_lib_s, "LIBRARY_LIB") {
            if let Some(library_prefix) = env::var_os("LIBRARY_PREFIX") {
                let inc = Path::new(&library_prefix).join("include");
                if inc.exists() {
                    println!("cargo:include={}", inc.to_string_lossy());
                }
            }
            return;
        }
    }

    // Try pkg-config discovery.
    match pkg_config::Config::new()
        .cargo_metadata(true)
        .probe("openblas")
    {
        Ok(_) => {
            println!("cargo:rustc-cfg=jx_openblas_available");
            println!("cargo:warning=OpenBLAS detected by pkg-config.");
        }
        Err(err) => {
            if require_openblas {
                panic!(
                    "JANUSX_REQUIRE_OPENBLAS=1 but OpenBLAS was not detected ({err}). \
Set OPENBLAS_LIB_DIR/OPENBLAS_INCLUDE_DIR or install OpenBLAS + pkg-config metadata."
                );
            }
            if cfg!(target_os = "linux") {
                if let Some(lib_dir) = env::var_os("BLAS_LIB_DIR") {
                    let lib_dir_s = lib_dir.to_string_lossy().to_string();
                    if Path::new(&lib_dir_s).exists() {
                        println!("cargo:rustc-link-search=native={lib_dir_s}");
                        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir_s}");
                        println!("cargo:rustc-cfg=jx_blas_available");
                        println!(
                            "cargo:warning=OpenBLAS feature requested but library not found ({err}); \
falling back to BLAS via BLAS_LIB_DIR."
                        );
                        return;
                    }
                }
                match pkg_config::Config::new().cargo_metadata(true).probe("blas") {
                    Ok(_) => {
                        println!("cargo:rustc-cfg=jx_blas_available");
                        println!(
                            "cargo:warning=OpenBLAS feature requested but library not found ({err}); \
falling back to pkg-config BLAS."
                        );
                    }
                    Err(blas_err) => {
                        println!(
                            "cargo:warning=OpenBLAS feature requested but library not found ({err}); \
BLAS also unavailable ({blas_err}); falling back to pure Rust SGEMM."
                        );
                    }
                }
            } else {
                println!(
                    "cargo:warning=OpenBLAS feature requested but library not found ({err}); \
falling back to platform backend."
                );
            }
        }
    }
}
