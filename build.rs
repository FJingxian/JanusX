use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FastTreeOpenMpMode {
    Disabled,
    Auto,
    Required,
}

fn parse_fasttree_openmp_mode() -> FastTreeOpenMpMode {
    let Some(raw) = env_value("JANUSX_FASTTREE_OPENMP") else {
        return FastTreeOpenMpMode::Auto;
    };
    let key = raw.to_ascii_lowercase();
    match key.as_str() {
        "0" | "false" | "off" | "no" | "disable" | "disabled" | "serial" => {
            FastTreeOpenMpMode::Disabled
        }
        "auto" | "default" => FastTreeOpenMpMode::Auto,
        "1" | "true" | "on" | "yes" | "required" | "require" | "force" => {
            FastTreeOpenMpMode::Required
        }
        other => {
            println!("cargo:warning=Unrecognized JANUSX_FASTTREE_OPENMP={other:?}; using auto.");
            FastTreeOpenMpMode::Auto
        }
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

fn find_command_on_path(name: &str) -> Option<PathBuf> {
    let candidate = PathBuf::from(name);
    if candidate.components().count() > 1 {
        return candidate.is_file().then_some(candidate);
    }
    find_command_in_path_var(name, env::var_os("PATH").as_ref())
}

fn find_command_in_path_var(name: &str, path_var: Option<&OsString>) -> Option<PathBuf> {
    let candidate = PathBuf::from(name);
    if candidate.components().count() > 1 {
        return candidate.is_file().then_some(candidate);
    }
    let path_var = path_var?;
    for dir in env::split_paths(path_var) {
        let full = dir.join(name);
        if full.is_file() {
            return Some(full);
        }
    }
    None
}

fn tool_command_is_resolvable(tool: &cc::Tool) -> bool {
    let path = tool.path();
    if path.is_file() {
        return true;
    }
    let Some(name) = path.to_str() else {
        return false;
    };
    if path.components().count() > 1 {
        return false;
    }
    if find_command_on_path(name).is_some() {
        return true;
    }
    let tool_path_env = tool.env().iter().find_map(|(k, v)| {
        k.to_str()
            .filter(|key| key.eq_ignore_ascii_case("PATH"))
            .map(|_| v)
    });
    find_command_in_path_var(name, tool_path_env).is_some()
}

fn compiler_version_text(path: &Path) -> String {
    match Command::new(path).arg("--version").output() {
        Ok(out) => {
            let mut txt = String::new();
            if !out.stdout.is_empty() {
                txt.push_str(&String::from_utf8_lossy(&out.stdout));
            }
            if !out.stderr.is_empty() {
                txt.push_str(&String::from_utf8_lossy(&out.stderr));
            }
            txt
        }
        Err(_) => String::new(),
    }
}

fn is_apple_clang(path: &Path) -> bool {
    compiler_version_text(path)
        .to_ascii_lowercase()
        .contains("apple clang")
}

fn existing_dir(path: PathBuf) -> Option<PathBuf> {
    if path.is_dir() {
        Some(path)
    } else {
        None
    }
}

fn clear_dir_files(path: &Path) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let file_path = entry.path();
        if file_path.is_file() {
            let _ = fs::remove_file(file_path);
        }
    }
}

fn command_stdout_trim(cmd: &mut Command) -> Option<String> {
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if stdout.is_empty() {
        None
    } else {
        Some(stdout)
    }
}

fn compiler_print_file_name(
    compiler: &Path,
    compiler_args: &[OsString],
    name: &str,
) -> Option<PathBuf> {
    let mut cmd = Command::new(compiler);
    cmd.args(compiler_args)
        .arg(format!("-print-file-name={name}"));
    let stdout = command_stdout_trim(&mut cmd)?;
    let path = PathBuf::from(stdout);
    if path.as_os_str().is_empty() || path == Path::new(name) || !path.is_file() {
        return None;
    }
    Some(path)
}

fn unique_existing_files(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for path in paths {
        if !path.is_file() {
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

fn file_name_lower(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
}

fn is_openmp_runtime_name(path: &Path) -> bool {
    let Some(name) = file_name_lower(path) else {
        return false;
    };
    name.starts_with("libomp")
        || name.starts_with("libgomp")
        || name.starts_with("libiomp")
        || name.starts_with("vcomp")
}

fn parse_ldd_openmp_runtime_paths(exe: &Path) -> Vec<PathBuf> {
    let Some(ldd_path) = find_command_on_path("ldd") else {
        return Vec::new();
    };
    let Ok(out) = Command::new(ldd_path).arg(exe).output() else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut out_paths: Vec<PathBuf> = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || !trimmed.contains("=>") {
            continue;
        }
        let mut parts = trimmed.splitn(2, "=>");
        let _lhs = parts.next();
        let rhs = parts.next().unwrap_or("").trim();
        if rhs.is_empty() || rhs.starts_with("not found") {
            continue;
        }
        let Some(path_token) = rhs.split_whitespace().next() else {
            continue;
        };
        let path = PathBuf::from(path_token);
        if is_openmp_runtime_name(&path) && path.is_file() {
            out_paths.push(path);
        }
    }
    unique_existing_files(out_paths)
}

fn linux_openmp_runtime_artifacts(
    exe: &Path,
    compiler: &Path,
    compiler_args: &[OsString],
) -> Vec<PathBuf> {
    let mut out = parse_ldd_openmp_runtime_paths(exe);
    if out.is_empty() {
        for name in ["libgomp.so.1", "libomp.so", "libomp.so.5", "libiomp5.so"] {
            if let Some(path) = compiler_print_file_name(compiler, compiler_args, name) {
                out.push(path);
            }
        }
    }
    unique_existing_files(out)
}

fn windows_runtime_probe_dirs(compiler: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    out.extend(path_env_dirs(&["PATH"]));
    if let Some(dir) = env::var_os("JANUSX_FASTTREE_OPENMP_LIB_DIR").map(PathBuf::from) {
        if dir.is_dir() {
            out.push(dir);
        }
    }
    for name in [
        "LIBRARY_BIN",
        "OPENBLAS_BIN_DIR",
        "LIBRARY_LIB",
        "OPENBLAS_LIB_DIR",
    ] {
        if let Some(dir) = env::var_os(name).map(PathBuf::from) {
            if dir.is_dir() {
                out.push(dir);
            }
        }
    }
    if let Some(conda_prefix) = env::var_os("CONDA_PREFIX").map(PathBuf::from) {
        out.push(conda_prefix.join("Library").join("bin"));
        out.push(conda_prefix.join("Library").join("lib"));
        out.push(conda_prefix.join("bin"));
        out.push(conda_prefix.join("DLLs"));
    }
    if let Some(parent) = compiler.parent() {
        out.push(parent.to_path_buf());
        out.push(parent.join("..").join("bin"));
        out.push(parent.join("..").join("lib"));
    }
    out.into_iter().filter(|p| p.is_dir()).collect()
}

fn windows_find_runtime_by_name(name: &str, probe_dirs: &[PathBuf]) -> Option<PathBuf> {
    if let Some(path) = find_command_on_path(name) {
        return Some(path);
    }
    for dir in probe_dirs {
        let cand = dir.join(name);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

fn windows_openmp_runtime_artifacts(
    compiler: &Path,
    compiler_args: &[OsString],
    is_msvc: bool,
) -> Vec<PathBuf> {
    let names: Vec<&str> = if is_msvc {
        vec![
            "libomp.dll",
            "vcomp140.dll",
            "vcomp140_1.dll",
            "libiomp5md.dll",
        ]
    } else {
        vec![
            "libgomp-1.dll",
            "libomp.dll",
            "libstdc++-6.dll",
            "libgcc_s_seh-1.dll",
            "libgcc_s_dw2-1.dll",
            "libgcc_s_sjlj-1.dll",
            "libwinpthread-1.dll",
        ]
    };
    let probe_dirs = windows_runtime_probe_dirs(compiler);
    let mut out: Vec<PathBuf> = Vec::new();
    for name in [
        "libgomp-1.dll",
        "libomp.dll",
        "libstdc++-6.dll",
        "libgcc_s_seh-1.dll",
        "libgcc_s_dw2-1.dll",
        "libgcc_s_sjlj-1.dll",
        "libwinpthread-1.dll",
    ] {
        if let Some(path) = compiler_print_file_name(compiler, compiler_args, name) {
            out.push(path);
        }
    }
    for name in names {
        if let Some(path) = windows_find_runtime_by_name(name, &probe_dirs) {
            out.push(path);
        }
    }
    unique_existing_files(out)
}

fn emit_command_output(prefix: &str, out: &Output) {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !stdout.trim().is_empty() {
        for line in stdout.lines() {
            println!("cargo:warning=[{prefix} stdout] {line}");
        }
    }
    if !stderr.trim().is_empty() {
        for line in stderr.lines() {
            println!("cargo:warning=[{prefix} stderr] {line}");
        }
    }
}

fn macos_openmp_paths() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let explicit_inc = env::var_os("JANUSX_FASTTREE_OPENMP_INCLUDE_DIR").map(PathBuf::from);
    let explicit_lib = env::var_os("JANUSX_FASTTREE_OPENMP_LIB_DIR").map(PathBuf::from);
    if let (Some(inc), Some(lib)) = (explicit_inc, explicit_lib) {
        let dylib = lib.join("libomp.dylib");
        if inc.join("omp.h").is_file() && dylib.is_file() {
            return Some((inc, lib, dylib));
        }
    }

    let mut include_dirs: Vec<PathBuf> = Vec::new();
    let mut lib_dirs: Vec<PathBuf> = Vec::new();
    if let Some(conda_prefix) = env::var_os("CONDA_PREFIX").map(PathBuf::from) {
        if let Some(p) = existing_dir(conda_prefix.join("include")) {
            include_dirs.push(p);
        }
        if let Some(p) = existing_dir(conda_prefix.join("lib")) {
            lib_dirs.push(p);
        }
        if let Some(parent) = conda_prefix.parent() {
            if let Some(p) = existing_dir(parent.join("include")) {
                include_dirs.push(p);
            }
            if let Some(p) = existing_dir(parent.join("lib")) {
                lib_dirs.push(p);
            }
        }
    }
    if let Some(p) = existing_dir(PathBuf::from("/opt/homebrew/opt/libomp/include")) {
        include_dirs.push(p);
    }
    if let Some(p) = existing_dir(PathBuf::from("/opt/homebrew/opt/libomp/lib")) {
        lib_dirs.push(p);
    }
    if let Some(p) = existing_dir(PathBuf::from("/usr/local/opt/libomp/include")) {
        include_dirs.push(p);
    }
    if let Some(p) = existing_dir(PathBuf::from("/usr/local/opt/libomp/lib")) {
        lib_dirs.push(p);
    }

    for inc in include_dirs {
        if !inc.join("omp.h").is_file() {
            continue;
        }
        for lib in lib_dirs.iter() {
            let dylib = lib.join("libomp.dylib");
            if dylib.is_file() {
                return Some((inc.clone(), lib.clone(), dylib));
            }
        }
    }
    None
}

#[cfg(unix)]
fn set_exec_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;

    if let Ok(meta) = fs::metadata(path) {
        let mut perms = meta.permissions();
        perms.set_mode(0o755);
        let _ = fs::set_permissions(path, perms);
    }
}

#[cfg(not(unix))]
fn set_exec_permissions(_path: &Path) {}

fn compile_fasttree_executable() {
    println!("cargo:rerun-if-changed=todo/FastTree.c");
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CFLAGS");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=JANUSX_FASTTREE_OPENMP");
    println!("cargo:rerun-if-env-changed=JANUSX_FASTTREE_OPENMP_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=JANUSX_FASTTREE_OPENMP_LIB_DIR");
    println!("cargo:rerun-if-env-changed=JANUSX_FASTTREE_NATIVE");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let src_path = Path::new(&manifest_dir).join("todo").join("FastTree.c");
    if !src_path.is_file() {
        panic!("FastTree source not found: {}", src_path.to_string_lossy());
    }

    let target = env::var("TARGET").unwrap_or_else(|_| "unknown-target".to_string());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| env::consts::OS.to_string());
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let openmp_mode = parse_fasttree_openmp_mode();
    let temp_exe_name = if target_os == "windows" {
        "FastTree-build.exe"
    } else {
        "FastTree-build"
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR missing"));
    let temp_exe = out_dir.join(temp_exe_name);
    let final_dir = Path::new(&manifest_dir)
        .join("target")
        .join("janusx-artifacts")
        .join(&target)
        .join(&profile)
        .join("fasttree");
    fs::create_dir_all(&final_dir).expect("failed to create FastTree artifact directory");
    clear_dir_files(&final_dir);

    let mut build = cc::Build::new();
    build.file(&src_path).warnings(false);
    let compiler_opt = build.try_get_compiler().ok();
    let use_compiler_tool = compiler_opt
        .as_ref()
        .map(tool_command_is_resolvable)
        .unwrap_or(false);
    let compiler_args: Vec<OsString> = compiler_opt
        .as_ref()
        .filter(|_| use_compiler_tool)
        .map(|tool| tool.args().iter().map(|arg| arg.to_os_string()).collect())
        .unwrap_or_default();
    let compiler_path = if let Some(tool) = compiler_opt.as_ref().filter(|_| use_compiler_tool) {
        if !tool.path().is_file() {
            emit_verbose_note(format!(
                "FastTree compiler resolved via cc crate as `{}`; relying on cc::Tool command env/wrapper for execution.",
                tool.path().to_string_lossy()
            ));
        }
        tool.path().to_path_buf()
    } else {
        let fallbacks: Vec<&str> = if target_os == "windows" {
            vec!["cl.exe", "clang-cl.exe", "clang.exe", "gcc.exe"]
        } else {
            vec!["cc", "clang", "gcc"]
        };
        let found = fallbacks
            .into_iter()
            .find_map(find_command_on_path)
            .unwrap_or_else(|| {
                panic!(
                    "failed to find a usable C compiler for FastTree; configured compiler from cc crate was unavailable"
                )
            });
        if let Some(tool) = compiler_opt.as_ref() {
            emit_verbose_note(format!(
                "Configured compiler {} was not directly runnable with the detected cc::Tool env; fallback to {}",
                tool.path().to_string_lossy(),
                found.to_string_lossy()
            ));
        } else {
            emit_verbose_note(format!(
                "cc crate compiler detection failed; fallback to {}",
                found.to_string_lossy()
            ));
        }
        found
    };

    let is_msvc = compiler_path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("cl.exe") || s.eq_ignore_ascii_case("clang-cl.exe"))
        .unwrap_or(false);
    let native_tuning = env_flag("JANUSX_FASTTREE_NATIVE");
    if native_tuning {
        println!("cargo:warning=FastTree native CPU tuning enabled (JANUSX_FASTTREE_NATIVE=1).");
    }
    let explicit_omp_include = env::var_os("JANUSX_FASTTREE_OPENMP_INCLUDE_DIR").map(PathBuf::from);
    let explicit_omp_lib = env::var_os("JANUSX_FASTTREE_OPENMP_LIB_DIR").map(PathBuf::from);
    let mut runtime_artifacts: Vec<PathBuf> = Vec::new();
    let mut openmp_enabled = false;

    let build_command = |use_openmp: bool| -> Result<(Command, Vec<PathBuf>), String> {
        let mut cmd = if use_compiler_tool {
            let tool = compiler_opt
                .as_ref()
                .expect("use_compiler_tool implies compiler_opt");
            tool.to_command()
        } else {
            let mut cmd = Command::new(&compiler_path);
            cmd.args(&compiler_args);
            cmd
        };
        cmd.current_dir(&manifest_dir);
        let mut runtimes: Vec<PathBuf> = Vec::new();
        if is_msvc {
            cmd.arg("/nologo")
                .arg("/O2")
                .arg("/std:c11")
                .arg("/D_CRT_SECURE_NO_WARNINGS");
            if use_openmp {
                cmd.arg("/DOPENMP").arg("/openmp");
            }
            cmd.arg(format!("/Fe:{}", temp_exe.to_string_lossy()))
                .arg(src_path.as_os_str());
            return Ok((cmd, runtimes));
        }

        cmd.arg("-O3")
            .arg("-std=c99")
            .arg("-finline-functions")
            .arg("-funroll-loops")
            .arg("-o")
            .arg(temp_exe.as_os_str());
        if native_tuning {
            match target_arch.as_str() {
                "x86" | "x86_64" => {
                    cmd.arg("-march=native").arg("-mtune=native");
                }
                "arm" | "aarch64" => {
                    cmd.arg("-mcpu=native");
                }
                _ => {}
            }
        }
        if target_os != "windows" {
            cmd.arg("-lm");
        }

        if use_openmp {
            cmd.arg("-DOPENMP");
            if let Some(inc) = explicit_omp_include.as_ref() {
                cmd.arg(format!("-I{}", inc.to_string_lossy()));
            }
            if let Some(lib) = explicit_omp_lib.as_ref() {
                cmd.arg(format!("-L{}", lib.to_string_lossy()));
            }
            match target_os.as_str() {
                "macos" => {
                    let Some((omp_inc, omp_lib, omp_dylib)) = macos_openmp_paths() else {
                        return Err(
                            "libomp/omp.h not found on macOS; set JANUSX_FASTTREE_OPENMP_INCLUDE_DIR and JANUSX_FASTTREE_OPENMP_LIB_DIR, or install llvm-openmp/libomp."
                                .to_string(),
                        );
                    };
                    if is_apple_clang(&compiler_path) {
                        cmd.arg("-Xpreprocessor").arg("-fopenmp");
                    } else {
                        cmd.arg("-fopenmp");
                    }
                    cmd.arg(format!("-I{}", omp_inc.to_string_lossy()));
                    cmd.arg(format!("-L{}", omp_lib.to_string_lossy()));
                    cmd.arg("-Wl,-rpath,@loader_path");
                    cmd.arg("-lomp");
                    runtimes.push(omp_dylib);
                }
                "linux" => {
                    cmd.arg("-fopenmp");
                    cmd.arg("-Wl,-rpath,$ORIGIN");
                    if let Some(lib) = explicit_omp_lib.as_ref() {
                        cmd.arg(format!("-Wl,-rpath,{}", lib.to_string_lossy()));
                    }
                }
                "windows" => {
                    cmd.arg("-fopenmp");
                }
                _ => {
                    cmd.arg("-fopenmp");
                }
            }
        }
        cmd.arg(src_path.as_os_str());
        Ok((cmd, runtimes))
    };

    if openmp_mode != FastTreeOpenMpMode::Disabled {
        match build_command(true) {
            Ok((mut omp_cmd, mut omp_runtimes)) => {
                let out = omp_cmd
                    .output()
                    .unwrap_or_else(|e| panic!("failed to start FastTree compiler: {e}"));
                if out.status.success() {
                    openmp_enabled = true;
                    runtime_artifacts.append(&mut omp_runtimes);
                    match target_os.as_str() {
                        "linux" => {
                            runtime_artifacts.extend(linux_openmp_runtime_artifacts(
                                &temp_exe,
                                &compiler_path,
                                &compiler_args,
                            ));
                        }
                        "windows" => {
                            runtime_artifacts.extend(windows_openmp_runtime_artifacts(
                                &compiler_path,
                                &compiler_args,
                                is_msvc,
                            ));
                        }
                        _ => {}
                    }
                } else {
                    emit_command_output("fasttree-build-openmp", &out);
                    if openmp_mode == FastTreeOpenMpMode::Required {
                        panic!(
                            "FastTree OpenMP compilation failed with status {}",
                            out.status
                        );
                    }
                    println!(
                        "cargo:warning=FastTree OpenMP build failed; falling back to serial binary."
                    );
                }
            }
            Err(msg) => {
                if openmp_mode == FastTreeOpenMpMode::Required {
                    panic!("JANUSX_FASTTREE_OPENMP requires OpenMP, but {msg}");
                }
                println!("cargo:warning=FastTree OpenMP disabled: {msg}");
            }
        }
    }

    if !openmp_enabled {
        let (mut serial_cmd, _) = build_command(false)
            .unwrap_or_else(|e| panic!("failed to prepare serial FastTree compiler command: {e}"));
        let out = serial_cmd
            .output()
            .unwrap_or_else(|e| panic!("failed to start FastTree compiler: {e}"));
        if !out.status.success() {
            emit_command_output("fasttree-build", &out);
            panic!("FastTree compilation failed with status {}", out.status);
        }
    }

    runtime_artifacts = unique_existing_files(runtime_artifacts);
    let final_names: Vec<&str> = if target_os == "windows" && openmp_enabled {
        vec!["FastTreeMP.exe", "FastTree.exe"]
    } else if target_os == "windows" {
        vec!["FastTree.exe"]
    } else {
        vec!["FastTree"]
    };

    for exe_name in final_names.iter() {
        let final_exe = final_dir.join(exe_name);
        fs::copy(&temp_exe, &final_exe).unwrap_or_else(|e| {
            panic!(
                "failed to copy FastTree executable to {}: {e}",
                final_exe.to_string_lossy()
            )
        });
        set_exec_permissions(&final_exe);
    }
    for runtime in runtime_artifacts.iter() {
        let Some(name) = runtime.file_name() else {
            continue;
        };
        let target_path = final_dir.join(name);
        fs::copy(runtime, &target_path).unwrap_or_else(|e| {
            panic!(
                "failed to copy FastTree runtime artifact {} to {}: {e}",
                runtime.to_string_lossy(),
                target_path.to_string_lossy()
            )
        });
    }
    emit_verbose_note(format!(
        "Prepared FastTree artifacts in {}",
        final_dir.to_string_lossy()
    ));
    if openmp_enabled {
        if target_os == "windows" {
            println!(
                "cargo:warning=FastTree OpenMP enabled; emitted FastTreeMP.exe and FastTree.exe."
            );
        } else {
            println!("cargo:warning=FastTree OpenMP enabled.");
        }
    } else {
        println!("cargo:warning=FastTree built without OpenMP support.");
    }
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
        println!(
            "cargo:warning=JanusX BED decode runtime SIMD compiled in ({kind}); full-row default={}, subset default={}. Override with JX_BED_DECODE_SIMD / JX_BED_SUBSET_SIMD.",
            if full_default { "on" } else { "off" },
            if subset_default { "on" } else { "off" }
        );
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
    maybe_prebuild_kmc_bind();
    emit_bed_decode_simd_defaults();
    compile_fasttree_executable();

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
    println!("cargo:rerun-if-env-changed=LIBRARY_LIB");
    println!("cargo:rerun-if-env-changed=LIBRARY_PREFIX");
    println!("cargo:rerun-if-env-changed=JANUSX_LINK_BLAS_FAMILY");
    println!("cargo:rerun-if-env-changed=JANUSX_FORCE_OPENBLAS_LAPACK");
    println!("cargo:rerun-if-env-changed=JANUSX_DISABLE_OPENBLAS_LAPACK");
    println!("cargo:rerun-if-env-changed=JANUSX_MACOS_PREFER_ACCELERATE");
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

    if should_prefer_accelerate_on_macos(require_openblas) {
        println!(
            "cargo:warning=macOS build: prefer Accelerate backend by default \
(set JANUSX_MACOS_PREFER_ACCELERATE=0 or JANUSX_REQUIRE_OPENBLAS=1 to enable OpenBLAS probing)."
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
