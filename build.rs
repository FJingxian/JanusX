use std::env;
use std::fs;
use std::path::{Path, PathBuf};

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

fn env_value(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn parse_boolish(value: &str) -> Option<bool> {
    let key = value.trim().to_ascii_lowercase();
    match key.as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

fn should_prefer_accelerate_on_macos(require_openblas: bool) -> bool {
    if !cfg!(target_os = "macos") {
        return false;
    }
    if require_openblas {
        return false;
    }
    // Default policy on macOS: prefer Accelerate unless explicitly disabled.
    env_flag_default_true("JANUSX_MACOS_PREFER_ACCELERATE")
}

fn emit_verbose_note(msg: impl AsRef<str>) {
    if env_flag("JANUSX_BUILD_VERBOSE") {
        println!("cargo:warning={}", msg.as_ref());
    }
}

fn configure_macos_deployment_target() {
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");

    // When building from source on Apple Silicon, cc-rs may derive the C/C++
    // deployment target from the installed SDK if this variable is unset.
    // That can produce an SDK-version target (for example 27.2), which is not
    // a valid minimum target and breaks conda clang's libc++ headers. Keep the
    // native build aligned with JanusX's macOS arm64 wheel minimum while
    // preserving any explicit user choice.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
        || env::var_os("MACOSX_DEPLOYMENT_TARGET").is_some()
    {
        return;
    }

    env::set_var("MACOSX_DEPLOYMENT_TARGET", "11.0");
    println!(
        "cargo:warning=MACOSX_DEPLOYMENT_TARGET was unset; defaulting to 11.0 for macOS arm64 C/C++ builds."
    );
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

fn has_windows_openblas_runtime_dll(lib_dir_s: &str) -> bool {
    if !cfg!(target_os = "windows") {
        return false;
    }
    let lib_dir = Path::new(lib_dir_s);
    let mut probe_dirs: Vec<std::path::PathBuf> = vec![lib_dir.to_path_buf()];
    if let Some(parent) = lib_dir.parent() {
        probe_dirs.push(parent.join("bin"));
        probe_dirs.push(parent.join("Library").join("bin"));
    }
    for d in probe_dirs {
        if !(d.exists() && d.is_dir()) {
            continue;
        }
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let Some(name_raw) = entry.file_name().to_str().map(|s| s.to_ascii_lowercase()) else {
                continue;
            };
            if !name_raw.ends_with(".dll") {
                continue;
            }
            if name_raw.starts_with("openblas") || name_raw.starts_with("libopenblas") {
                return true;
            }
        }
    }
    false
}

fn file_contains_all_ascii_markers(path: &Path, markers: &[&[u8]]) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    markers.iter().all(|marker| {
        if marker.is_empty() {
            return true;
        }
        bytes.windows(marker.len()).any(|w| w == *marker)
    })
}

fn windows_openblas_candidate_files(lib_dir_s: &str) -> Vec<std::path::PathBuf> {
    if !cfg!(target_os = "windows") {
        return Vec::new();
    }
    let lib_dir = Path::new(lib_dir_s);
    let mut probe_dirs: Vec<std::path::PathBuf> = vec![lib_dir.to_path_buf()];
    if let Some(parent) = lib_dir.parent() {
        probe_dirs.push(parent.join("bin"));
        probe_dirs.push(parent.join("DLLs"));
        probe_dirs.push(parent.join("Library").join("bin"));
        if let Some(grand) = parent.parent() {
            probe_dirs.push(grand.join("bin"));
            probe_dirs.push(grand.join("DLLs"));
            probe_dirs.push(grand.join("Library").join("bin"));
        }
    }

    let mut out: Vec<std::path::PathBuf> = Vec::new();
    for d in probe_dirs {
        if !(d.exists() && d.is_dir()) {
            continue;
        }
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name_raw) = path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase())
            else {
                continue;
            };
            if !(name_raw.starts_with("openblas") || name_raw.starts_with("libopenblas")) {
                continue;
            }
            let is_candidate = name_raw.ends_with(".dll")
                || name_raw.ends_with(".lib")
                || name_raw.ends_with(".a")
                || name_raw.ends_with(".dll.a");
            if is_candidate {
                out.push(path);
            }
        }
    }
    out
}

fn windows_openblas_has_lapack_symbols(lib_dir_s: &str) -> bool {
    if !cfg!(target_os = "windows") {
        return false;
    }
    let markers: [&[u8]; 2] = [b"dsyevd_", b"dsyevr_"];
    for path in windows_openblas_candidate_files(lib_dir_s) {
        if file_contains_all_ascii_markers(&path, &markers) {
            emit_verbose_note(format!(
                "OpenBLAS LAPACK symbols detected directly from {}",
                path.to_string_lossy()
            ));
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

fn python_prefix_from_executable(raw: &str) -> Option<PathBuf> {
    let executable = Path::new(raw);
    let bin_dir = executable.parent()?;
    let dir_name = bin_dir.file_name()?.to_string_lossy().to_ascii_lowercase();
    if matches!(dir_name.as_str(), "bin" | "scripts") {
        Some(bin_dir.parent()?.to_path_buf())
    } else {
        // Windows conda Python is commonly <prefix>/python.exe, while venv
        // Python can be <prefix>/Scripts/python.exe.
        Some(bin_dir.to_path_buf())
    }
}

fn configure_openblas_from_python_env() -> bool {
    for variable in ["PYO3_PYTHON", "PYTHON_SYS_EXECUTABLE"] {
        let Some(raw) = env_value(variable) else {
            continue;
        };
        let Some(prefix) = python_prefix_from_executable(&raw) else {
            continue;
        };
        let lib_dir = prefix.join("lib");
        let lib_dir_s = lib_dir.to_string_lossy().to_string();
        if configure_openblas_from_dir(&lib_dir_s, variable) {
            let include_dir = prefix.join("include");
            if include_dir.exists() {
                println!("cargo:include={}", include_dir.to_string_lossy());
            }
            return true;
        }
    }
    false
}

fn build_python_prefixes() -> Vec<PathBuf> {
    let mut prefixes = Vec::<PathBuf>::new();
    for variable in ["PYO3_PYTHON", "PYTHON_SYS_EXECUTABLE"] {
        let Some(raw) = env_value(variable) else {
            continue;
        };
        let Some(prefix) = python_prefix_from_executable(&raw) else {
            continue;
        };
        let resolved = fs::canonicalize(&prefix).unwrap_or(prefix);
        if !prefixes.iter().any(|candidate| candidate == &resolved) {
            prefixes.push(resolved);
        }
    }
    prefixes
}

fn conda_prefix_matches_build_python(prefix: &Path) -> bool {
    let python_prefixes = build_python_prefixes();
    if python_prefixes.is_empty() {
        // Plain Cargo/conda builds may not have a PyO3 interpreter variable;
        // in that case CONDA_PREFIX is the only available build environment.
        return true;
    }
    let resolved = fs::canonicalize(prefix).unwrap_or_else(|_| prefix.to_path_buf());
    python_prefixes
        .iter()
        .any(|candidate| candidate == &resolved)
}

fn resolve_kmc_source_dir() -> PathBuf {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    if let Some(raw) = env_value("JANUSX_KMC_SRC") {
        let cand = PathBuf::from(raw);
        if cand.join("kmc_core").join("kmc_runner.h").is_file()
            && cand.join("kmc_api").join("kmc_file.h").is_file()
        {
            return cand;
        }
        panic!(
            "JANUSX_KMC_SRC does not look like a valid KMC source tree: {}",
            cand.to_string_lossy()
        );
    }

    let vendored = Path::new(&manifest_dir)
        .join("python")
        .join("janusx")
        .join("native")
        .join("vendor")
        .join("kmc");
    if vendored.join("kmc_core").join("kmc_runner.h").is_file()
        && vendored.join("kmc_api").join("kmc_file.h").is_file()
    {
        return vendored;
    }

    panic!(
        "KMC source not found. Set JANUSX_KMC_SRC or ensure vendored source exists under python/janusx/native/vendor/kmc."
    );
}

fn ensure_kmc_compat_include(kmc_src_dir: &Path) -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR missing"));
    let compat_root = out_dir.join("jx_kmc_compat");
    let ext_dir = compat_root.join("ext");
    fs::create_dir_all(&ext_dir).expect("failed to create KMC compat include dir");
    let algo_hdr = ext_dir.join("algorithm");
    if !algo_hdr.is_file() {
        fs::write(
            &algo_hdr,
            "#pragma once\n#include <algorithm>\nnamespace __gnu_cxx { using std::copy_n; }\n",
        )
        .expect("failed to write KMC compat ext/algorithm header");
    }

    let zlib_hdr = kmc_src_dir
        .join("3rd_party")
        .join("cloudflare")
        .join("zlib.h");
    if !zlib_hdr.is_file() {
        if let Some(parent) = zlib_hdr.parent() {
            fs::create_dir_all(parent).expect("failed to create KMC cloudflare zlib dir");
        }
        fs::write(
            &zlib_hdr,
            "/* JANUSX_KMC_ZLIB_SHIM */\n#pragma once\n#if defined(__has_include_next)\n#  if __has_include_next(<zlib.h>)\n#    include_next <zlib.h>\n#  else\n#    include <zlib.h>\n#  endif\n#else\n#  include <zlib.h>\n#endif\n",
        )
        .expect("failed to write KMC zlib shim");
    }

    compat_root
}

fn kmc_common_source_files(kmc_src_dir: &Path) -> Vec<PathBuf> {
    let rel = [
        "kmc_core/mem_disk_file.cpp",
        "kmc_core/rev_byte.cpp",
        "kmc_core/bkb_writer.cpp",
        "kmc_core/cpu_info.cpp",
        "kmc_core/bkb_reader.cpp",
        "kmc_core/fastq_reader.cpp",
        "kmc_core/timer.cpp",
        "kmc_core/develop.cpp",
        "kmc_core/kb_completer.cpp",
        "kmc_core/kb_storer.cpp",
        "kmc_core/kmer.cpp",
        "kmc_core/splitter.cpp",
        "kmc_core/kb_collector.cpp",
        "kmc_core/kmc_runner.cpp",
        "kmc_core/kff_writer.cpp",
        "kmc_api/mmer.cpp",
        "kmc_api/kmc_file.cpp",
        "kmc_api/kmer_api.cpp",
    ];
    rel.iter()
        .map(|path| kmc_src_dir.join(path))
        .collect::<Vec<_>>()
}

fn kmc_arch_source_files(kmc_src_dir: &Path) -> Vec<PathBuf> {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "aarch64" {
        return vec![kmc_src_dir.join("kmc_core").join("raduls_neon.cpp")];
    }
    if target_arch == "x86_64" || target_arch == "x86" {
        return vec![
            kmc_src_dir.join("kmc_core").join("raduls_sse2.cpp"),
            kmc_src_dir.join("kmc_core").join("raduls_sse41.cpp"),
            kmc_src_dir.join("kmc_core").join("raduls_avx.cpp"),
            kmc_src_dir.join("kmc_core").join("raduls_avx2.cpp"),
        ];
    }
    Vec::new()
}

fn configure_kmc_build(build: &mut cc::Build, compat_include: &Path, kmc_src_dir: &Path) {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    build.cpp(true).warnings(false);
    build.include(compat_include);
    build.include(kmc_src_dir);
    build.include(kmc_src_dir.join("kmc_core"));
    build.include(kmc_src_dir.join("kmc_api"));
    // Vendored KMC detects AArch64 only via __aarch64__. MSVC ARM64 cross
    // builds define _M_ARM64 instead, which otherwise falls through to the
    // x86 cpuid path and fails in cpu_info.cpp.
    if target_arch == "aarch64" && target_os == "windows" {
        build.define("__aarch64__", None);
    }
    if target_os == "windows" && target_env == "msvc" {
        // Vendored KMC's template-heavy translation units exceed MSVC's default
        // COFF section limit (C1128) and require C++ exception unwinding (/EHsc).
        // Under recent MSVC toolsets, optimizing KMC with /bigobj can also hit
        // compiler ICE C1001, so default to /Od unless overridden.
        build.flag_if_supported("/EHsc");
        build.flag_if_supported("/bigobj");
        if !env_flag("JANUSX_KMC_MSVC_OPT") && env_value("CXXFLAGS").is_none() {
            build.opt_level(0);
        }
    }
    build.flag_if_supported("-std=c++14");
    build.flag_if_supported("/std:c++14");
    build.flag_if_supported("-Wno-unused-parameter");
    build.flag_if_supported("-Wno-sign-compare");
    if cfg!(not(target_os = "windows")) {
        build.flag_if_supported("-pthread");
    }
}

fn compile_kmc_arch_sources(compat_include: &Path, kmc_src_dir: &Path, arch_sources: &[PathBuf]) {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "aarch64" {
        return;
    }
    if target_arch != "x86_64" && target_arch != "x86" {
        return;
    }

    for src in arch_sources {
        let Some(stem) = src.file_stem().and_then(|x| x.to_str()) else {
            continue;
        };
        let mut build = cc::Build::new();
        configure_kmc_build(&mut build, compat_include, kmc_src_dir);
        build.file(src);
        match stem {
            "raduls_sse2" => {
                build.flag_if_supported("-msse2");
                build.flag_if_supported("/arch:SSE2");
                build.define("__SSE2__", None);
            }
            "raduls_sse41" => {
                build.flag_if_supported("-msse4.1");
                build.flag_if_supported("/arch:SSE2");
                build.define("__SSE4_1__", None);
            }
            "raduls_avx" => {
                build.flag_if_supported("-mavx");
                build.flag_if_supported("/arch:AVX");
                build.define("__AVX__", None);
            }
            "raduls_avx2" => {
                build.flag_if_supported("-mavx2");
                build.flag_if_supported("/arch:AVX2");
                build.define("__AVX2__", None);
            }
            _ => continue,
        }
        build.compile(&format!("jx_kmc_wrapper_{stem}"));
    }
}

fn compile_kmc_wrapper() {
    println!("cargo:rerun-if-env-changed=JANUSX_KMC_SRC");
    println!("cargo:rerun-if-env-changed=JANUSX_KMC_MSVC_OPT");
    println!("cargo:rerun-if-env-changed=JANUSX_ZLIB_LIB");
    println!("cargo:rerun-if-env-changed=JANUSX_ZLIB_LIB_DIR");
    println!("cargo:rerun-if-env-changed=VCPKG_LIB_DIR");
    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LIBRARY_LIB");
    println!("cargo:rerun-if-env-changed=LIB");
    println!("cargo:rerun-if-changed=src/kmer/ffi/kmc_wrapper.cpp");
    println!("cargo:rerun-if-changed=src/kmer/ffi/kmc_wrapper.h");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let kmc_src_dir = resolve_kmc_source_dir();
    let compat_include = ensure_kmc_compat_include(&kmc_src_dir);
    let wrapper_cpp = Path::new(&manifest_dir)
        .join("src")
        .join("kmer")
        .join("ffi")
        .join("kmc_wrapper.cpp");
    let common_sources = kmc_common_source_files(&kmc_src_dir);
    let arch_sources = kmc_arch_source_files(&kmc_src_dir);
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    for src in common_sources.iter().chain(arch_sources.iter()) {
        println!("cargo:rerun-if-changed={}", src.to_string_lossy());
    }

    let mut build = cc::Build::new();
    configure_kmc_build(&mut build, &compat_include, &kmc_src_dir);
    build.file(&wrapper_cpp);
    for src in common_sources.iter() {
        build.file(src);
    }
    if target_arch == "aarch64" {
        for src in arch_sources.iter() {
            build.file(src);
        }
    }
    build.compile("jx_kmc_wrapper");
    compile_kmc_arch_sources(&compat_include, &kmc_src_dir, &arch_sources);

    if target_os == "windows" {
        if !try_link_windows_kmc_zlib(&target_arch, &kmc_src_dir) {
            println!("cargo:rustc-link-lib=zlib");
        }
    } else {
        println!("cargo:rustc-link-lib=z");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
    }
    emit_verbose_note(format!(
        "KMC wrapper enabled with source tree {}",
        kmc_src_dir.to_string_lossy()
    ));
}

fn path_env_dirs(names: &[&str]) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for name in names {
        let Some(raw) = env::var_os(name) else {
            continue;
        };
        for path in env::split_paths(&raw) {
            if path.is_dir() {
                out.push(path);
            }
        }
    }
    out
}

fn unique_existing_dirs(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for path in paths {
        if !path.is_dir() {
            continue;
        }
        let canon = fs::canonicalize(&path).unwrap_or(path.clone());
        let key = canon.to_string_lossy().to_ascii_lowercase();
        if seen.iter().any(|s| s == &key) {
            continue;
        }
        seen.push(key);
        out.push(canon);
    }
    out
}

fn windows_kmc_zlib_probe_dirs() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for name in [
        "JANUSX_ZLIB_LIB_DIR",
        "VCPKG_LIB_DIR",
        "OPENBLAS_LIB_DIR",
        "LIBRARY_LIB",
    ] {
        if let Some(dir) = env::var_os(name).map(PathBuf::from) {
            out.push(dir);
        }
    }
    out.extend(path_env_dirs(&["LIB"]));
    unique_existing_dirs(out)
}

fn windows_find_kmc_zlib_candidate() -> Option<PathBuf> {
    if let Some(raw) = env::var_os("JANUSX_ZLIB_LIB") {
        let cand = PathBuf::from(raw);
        if cand.is_file() {
            return Some(cand);
        }
    }

    let preferred_names = [
        "libzs.lib",
        "libzsd.lib",
        "zlibstatic.lib",
        "zlib.lib",
        "libz.lib",
        "zlib1.lib",
        "z.lib",
    ];
    for dir in windows_kmc_zlib_probe_dirs() {
        for name in preferred_names {
            let cand = dir.join(name);
            if cand.is_file() {
                return Some(cand);
            }
        }
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase())
            else {
                continue;
            };
            if !name.ends_with(".lib") {
                continue;
            }
            if name == "libzs.lib"
                || name == "libzsd.lib"
                || name == "libz.lib"
                || name == "z.lib"
                || name.starts_with("zlib")
            {
                return Some(path);
            }
        }
    }
    None
}

fn try_link_windows_kmc_zlib(target_arch: &str, kmc_src_dir: &Path) -> bool {
    let vendored_zlib_static = kmc_src_dir
        .join("kmc_core")
        .join("libs")
        .join("zlibstat.lib");

    if (target_arch == "x86_64" || target_arch == "x86") && vendored_zlib_static.is_file() {
        if let Some(parent) = vendored_zlib_static.parent() {
            println!(
                "cargo:rustc-link-search=native={}",
                parent.to_string_lossy()
            );
        }
        println!("cargo:rustc-link-lib=static=zlibstat");
        return true;
    }

    if let Some(cand) = windows_find_kmc_zlib_candidate() {
        if let Some(parent) = cand.parent() {
            println!(
                "cargo:rustc-link-search=native={}",
                parent.to_string_lossy()
            );
        }
        println!("cargo:rustc-link-arg={}", cand.to_string_lossy());
        emit_verbose_note(format!(
            "KMC zlib linked from {} for target arch {}",
            cand.to_string_lossy(),
            target_arch
        ));
        return true;
    }

    false
}

fn emit_bed_decode_simd_defaults() {
    println!("cargo:rerun-if-env-changed=JANUSX_BED_DECODE_SIMD_DEFAULT");
    println!("cargo:rerun-if-env-changed=JANUSX_BED_SUBSET_SIMD_DEFAULT");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let simd_cap = match target_arch.as_str() {
        "x86" | "x86_64" => Some("AVX2"),
        "aarch64" => Some("NEON"),
        _ => None,
    };

    let full_default = env_value("JANUSX_BED_DECODE_SIMD_DEFAULT")
        .as_deref()
        .and_then(parse_boolish)
        .unwrap_or(simd_cap.is_some());
    let subset_default = env_value("JANUSX_BED_SUBSET_SIMD_DEFAULT")
        .as_deref()
        .and_then(parse_boolish)
        .unwrap_or(simd_cap.is_some());

    println!(
        "cargo:rustc-env=JANUSX_BED_DECODE_SIMD_DEFAULT={}",
        if full_default { "1" } else { "0" }
    );
    println!(
        "cargo:rustc-env=JANUSX_BED_SUBSET_SIMD_DEFAULT={}",
        if subset_default { "1" } else { "0" }
    );

    if let Some(kind) = simd_cap {
        emit_verbose_note(format!(
            "JanusX BED decode runtime SIMD compiled in ({kind}); full-row default={}, subset default={}. Override with JX_BED_DECODE_SIMD / JX_BED_SUBSET_SIMD.",
            if full_default { "on" } else { "off" },
            if subset_default { "on" } else { "off" }
        ));
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
    let has_win_lapack = has_library_with_prefix(lib_dir_s, "liblapack")
        || has_library_with_prefix(lib_dir_s, "lapack")
        || has_library_with_prefix(lib_dir_s, "lapacke");
    let has_win_openblas_lapack = windows_openblas_has_lapack_symbols(lib_dir_s);
    let mut lapack_available = !cfg!(target_os = "windows");
    if cfg!(target_os = "windows") {
        // Windows packages are heterogeneous:
        // - some ship separate lapack/lapacke import libs
        // - others embed dsyevd_/dsyevr_ directly inside openblas/libopenblas
        //   import libs / DLLs.
        // Treat either case as LAPACK-capable.
        lapack_available = has_win_lapack || has_win_openblas_lapack;
    }
    if env_flag("JANUSX_FORCE_OPENBLAS_LAPACK") {
        lapack_available = true;
    }
    if env_flag("JANUSX_DISABLE_OPENBLAS_LAPACK") {
        lapack_available = false;
    }
    if lapack_available {
        println!("cargo:rustc-cfg=jx_openblas_lapack_available");
    } else {
        emit_verbose_note(format!(
            "OpenBLAS detected via {source_label}, but LAPACK symbols are treated as unavailable."
        ));
    }
    if cfg!(target_os = "windows") && !has_windows_openblas_runtime_dll(lib_dir_s) {
        // Static-only layout (e.g. vcpkg *-windows-static-md): use static link kind
        // in Rust extern blocks to avoid unresolved __imp_* imports.
        println!("cargo:rustc-cfg=jx_openblas_static_link");
    }
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
    println!("cargo:rerun-if-env-changed=PYO3_BUILD_EXTENSION_MODULE");
    println!("cargo:rerun-if-env-changed=JANUSX_BUILD_VERBOSE");
    configure_macos_deployment_target();
    let build_extension_module = env_value("PYO3_BUILD_EXTENSION_MODULE")
        .as_deref()
        .and_then(parse_boolish)
        .unwrap_or(false);
    if build_extension_module {
        pyo3_build_config::add_extension_module_link_args();
    } else {
        pyo3_build_config::add_python_framework_link_args();
    }

    compile_kmc_wrapper();
    emit_bed_decode_simd_defaults();

    // Allow conditional compilation checks for our custom cfg.
    println!("cargo:rustc-check-cfg=cfg(jx_openblas_available)");
    println!("cargo:rustc-check-cfg=cfg(jx_openblas_link_openblas0)");
    println!("cargo:rustc-check-cfg=cfg(jx_openblas_link_openblas_plain)");
    println!("cargo:rustc-check-cfg=cfg(jx_openblas_static_link)");
    println!("cargo:rustc-check-cfg=cfg(jx_openblas_lapack_available)");
    println!("cargo:rustc-check-cfg=cfg(jx_blas_available)");
    println!("cargo:rerun-if-env-changed=JANUSX_REQUIRE_OPENBLAS");
    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=OPENBLAS_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=BLAS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=LIBRARY_LIB");
    println!("cargo:rerun-if-env-changed=LIBRARY_PREFIX");
    println!("cargo:rerun-if-env-changed=JANUSX_LINK_BLAS_FAMILY");
    println!("cargo:rerun-if-env-changed=JANUSX_ALLOW_RUST_BLAS_FALLBACK");
    println!("cargo:rerun-if-env-changed=JANUSX_FORCE_OPENBLAS_LAPACK");
    println!("cargo:rerun-if-env-changed=JANUSX_DISABLE_OPENBLAS_LAPACK");
    println!("cargo:rerun-if-env-changed=JANUSX_MACOS_PREFER_ACCELERATE");
    let openblas_feature_enabled = env::var_os("CARGO_FEATURE_BLAS_OPENBLAS").is_some();
    let allow_rust_fallback = env_flag("JANUSX_ALLOW_RUST_BLAS_FALLBACK");
    let require_openblas = env_flag("JANUSX_REQUIRE_OPENBLAS")
        || (cfg!(target_os = "linux") && openblas_feature_enabled && !allow_rust_fallback);

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
                    emit_verbose_note("BLAS enabled via BLAS_LIB_DIR.");
                    return;
                }
            }
            if pkg_config::Config::new()
                .cargo_metadata(true)
                .probe("blas")
                .is_ok()
            {
                println!("cargo:rustc-cfg=jx_blas_available");
                emit_verbose_note("BLAS detected by pkg-config.");
            } else {
                println!(
                    "cargo:warning=No BLAS/OpenBLAS detected on Linux; falling back to pure Rust SGEMM."
                );
            }
        }
        return;
    }

    if should_prefer_accelerate_on_macos(require_openblas) {
        emit_verbose_note(
            "macOS build: prefer Accelerate backend by default (set JANUSX_MACOS_PREFER_ACCELERATE=0 or JANUSX_REQUIRE_OPENBLAS=1 to enable OpenBLAS probing).",
        );
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

    // Prefer the Python interpreter selected by maturin/PyO3 before consulting
    // CONDA_PREFIX, which a caller can inherit from a different environment.
    if configure_openblas_from_python_env() {
        return;
    }

    // Conda environments may provide OpenBLAS without pkg-config metadata.
    // On Windows, libraries are usually in %CONDA_PREFIX%/Library/lib.
    if let Some(conda_prefix) = env::var_os("CONDA_PREFIX")
        .map(PathBuf::from)
        .filter(|prefix| conda_prefix_matches_build_python(prefix))
    {
        let mut candidates: Vec<(String, String)> = Vec::new();
        let p = conda_prefix.as_path();
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
            let mut lapack_available = !cfg!(target_os = "windows");
            if env_flag("JANUSX_FORCE_OPENBLAS_LAPACK") {
                lapack_available = true;
            }
            if env_flag("JANUSX_DISABLE_OPENBLAS_LAPACK") {
                lapack_available = false;
            }
            if lapack_available {
                println!("cargo:rustc-cfg=jx_openblas_lapack_available");
            } else {
                emit_verbose_note(
                    "OpenBLAS detected by pkg-config, but LAPACK symbols are treated as unavailable.",
                );
            }
            emit_verbose_note("OpenBLAS detected by pkg-config.");
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
