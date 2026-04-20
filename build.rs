use std::env;
use std::path::Path;

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
