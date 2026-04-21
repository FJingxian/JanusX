use std::env;
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
        println!("cargo:warning=Skip KMC prebuild (JANUSX_PREBUILD_KMC_BIND=0).");
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
            println!("cargo:warning=KMC bind prebuild finished.");
            let stdout = String::from_utf8_lossy(&out.stdout);
            let msg = stdout.trim();
            if !msg.is_empty() {
                for line in msg.lines() {
                    println!("cargo:warning={line}");
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

    // Conda/macOS can expose versioned soname only (e.g. libopenblas.0.dylib)
    // without a generic libopenblas.dylib symlink.
    let has_generic = Path::new(lib_dir_s).join("libopenblas.dylib").exists()
        || Path::new(lib_dir_s).join("libopenblas.so").exists()
        || Path::new(lib_dir_s).join("libopenblas.a").exists();
    let has_versioned = Path::new(lib_dir_s).join("libopenblas.0.dylib").exists()
        || Path::new(lib_dir_s).join("libopenblas.so.0").exists();
    if !has_generic && !has_versioned {
        return false;
    }
    println!("cargo:rustc-link-search=native={lib_dir_s}");
    // Ensure wheel-repair tools can resolve @rpath-linked dependencies
    // (e.g. libopenblas.0.dylib -> libiconv.2.dylib in conda environments).
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir_s}");
    println!("cargo:rustc-cfg=jx_openblas_available");
    if !has_generic && has_versioned {
        println!("cargo:rustc-cfg=jx_openblas_link_openblas0");
        println!(
            "cargo:warning=OpenBLAS detected via {source_label} with versioned soname \
only; linking as openblas.0."
        );
    } else {
        println!(
            "cargo:warning=OpenBLAS enabled via {source_label}; backend auto can use openblas."
        );
    }
    true
}

fn main() {
    maybe_prebuild_kmc_bind();

    // Allow conditional compilation checks for our custom cfg.
    println!("cargo:rustc-check-cfg=cfg(jx_openblas_available)");
    println!("cargo:rustc-check-cfg=cfg(jx_openblas_link_openblas0)");
    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=OPENBLAS_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    // Only probe OpenBLAS when user explicitly enabled the feature.
    if env::var_os("CARGO_FEATURE_BLAS_OPENBLAS").is_none() {
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
    if let Some(conda_prefix) = env::var_os("CONDA_PREFIX") {
        let conda_lib = Path::new(&conda_prefix).join("lib");
        let conda_lib_s = conda_lib.to_string_lossy().to_string();
        if configure_openblas_from_dir(&conda_lib_s, "CONDA_PREFIX/lib") {
            let conda_inc = Path::new(&conda_prefix).join("include");
            if conda_inc.exists() {
                println!("cargo:include={}", conda_inc.to_string_lossy());
            }
            return;
        }
    }

    // Try pkg-config discovery.
    match pkg_config::Config::new().cargo_metadata(true).probe("openblas") {
        Ok(_) => {
            println!("cargo:rustc-cfg=jx_openblas_available");
            println!("cargo:warning=OpenBLAS detected by pkg-config.");
        }
        Err(err) => {
            println!(
                "cargo:warning=OpenBLAS feature requested but library not found ({err}); \
falling back to platform backend."
            );
        }
    }
}
