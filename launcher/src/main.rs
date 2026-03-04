use std::env;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command, ExitStatus, Output, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const PYPI_SPEC: &str = "janusx";
const GITHUB_SPEC: &str = "git+https://github.com/FJingxian/JanusX.git";
const GITHUB_PROXY_SPEC: &str = "git+https://gh-proxy.org/https://github.com/FJingxian/JanusX.git";
const UPDATE_TIME_MARKER: &str = ".python_core_updated_at";
const COMMIT_MARKER: &str = ".python_core_commit";
const VERSION_AUTHOR: &str = "Jingxian FU, Yazhouwan National Laboratory";
const VERSION_CONTACT: &str = "fujingxian@yzwlab.cn";
const RUNTIME_HOME_CONFIG: &str = ".jx_home";
const LOGO: &str = r#"
       _                      __   __
      | |                     \ \ / /
      | | __ _ _ __  _   _ ___ \ V / 
  _   | |/ _` | '_ \| | | / __| > <  
 | |__| | (_| | | | | |_| \__ \/ . \ 
  \____/ \__,_|_| |_|\__,_|___/_/ \_\ Tools for GWAS and GS
  ---------------------------------------------------------
"#;

#[derive(Clone, Debug)]
enum UpdateSource {
    Pypi,
    Latest,
    Local(String),
}

#[derive(Clone, Debug)]
struct UpdateOptions {
    source: UpdateSource,
    verbose: bool,
    force_reinstall: bool,
}

fn main() {
    let code = match run() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            1
        }
    };
    process::exit(code);
}

fn run() -> Result<i32, String> {
    if is_installer_binary() {
        return run_installer();
    }

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        let python = ensure_runtime(false)?;
        return run_python_janusx(&python, &[]);
    }

    let head = args[0].as_str();
    if matches!(head, "-h" | "--help") {
        let python = ensure_runtime(false)?;
        return run_python_janusx(&python, &args);
    }
    if matches!(head, "-v" | "--version") {
        let home = runtime_home()?;
        let python = ensure_runtime(false)?;
        print_version(&python, &home);
        return Ok(0);
    }

    if matches!(head, "-update" | "--update") {
        let opts = match parse_update_args(&args[1..]) {
            Ok(v) => v,
            Err(e) if e.is_empty() => return Ok(0),
            Err(e) => return Err(e),
        };
        return run_update(opts);
    }

    let python = ensure_runtime(false)?;
    run_python_janusx(&python, &args)
}

fn is_installer_binary() -> bool {
    let exe = match env::current_exe() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let name = match exe.file_name().and_then(|x| x.to_str()) {
        Some(v) => v,
        None => return false,
    };
    name.starts_with("JanusX-")
}

fn run_installer() -> Result<i32, String> {
    println!("{LOGO}");
    println!("JanusX installer\n");

    let runtime_home = prompt_runtime_home()?;
    env::set_var("JX_HOME", &runtime_home);
    println!("Runtime home: {}", runtime_home.display());

    let python = ensure_runtime(false)?;
    if !is_janusx_installed(&python) {
        return Err("Failed to initialize JanusX runtime in selected JX_HOME.".to_string());
    }

    let install_dir = prompt_install_dir()?;
    let installed_jx = install_jx_binary(&install_dir)?;
    write_runtime_home_config_near_binary(&installed_jx, &runtime_home);
    print_success_line(&format!("Installed jx to {}", installed_jx.display()));

    println!("Warming up runtime with `jx -h` ...");
    let mut cmd = Command::new(&installed_jx);
    cmd.arg("-h")
        .env("JX_HOME", &runtime_home)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    let status = cmd
        .status()
        .map_err(|e| format!("Failed to run `{}` -h: {e}", installed_jx.display()))?;
    if !status.success() {
        return Err(format!(
            "Warm-up command failed: {} -h (exit={})",
            installed_jx.display(),
            exit_code(status)
        ));
    }
    print_success_line("Installation completed.");
    Ok(0)
}

fn default_runtime_home() -> Result<PathBuf, String> {
    #[cfg(windows)]
    {
        if let Some(v) = env::var_os("LOCALAPPDATA") {
            return Ok(PathBuf::from(v).join("JanusX"));
        }
    }
    #[cfg(not(windows))]
    {
        if let Some(v) = env::var_os("HOME") {
            return Ok(PathBuf::from(v).join(".janusx"));
        }
    }
    env::current_dir()
        .map(|p| p.join(".janusx"))
        .map_err(|e| format!("Failed to resolve runtime home: {e}"))
}

fn expand_tilde(input: &str) -> PathBuf {
    let s = input.trim();
    if !s.starts_with('~') {
        return PathBuf::from(s);
    }
    let home = env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from));
    if let Some(h) = home {
        if s == "~" {
            return h;
        }
        if let Some(rest) = s.strip_prefix("~/") {
            return h.join(rest);
        }
        if let Some(rest) = s.strip_prefix("~\\") {
            return h.join(rest);
        }
    }
    PathBuf::from(s)
}

fn prompt_runtime_home() -> Result<PathBuf, String> {
    let default_home = default_runtime_home()?;
    loop {
        println!(
            "Choose JanusX runtime home [y/n/path]\n  y: {}\n  n: cancel\n  path: existing directory",
            default_home.display()
        );
        print!("Input [y]: ");
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush stdout: {e}"))?;
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .map_err(|e| format!("Failed to read input: {e}"))?;
        let v = line.trim();
        if v.is_empty() || v.eq_ignore_ascii_case("y") {
            return Ok(default_home.clone());
        }
        if v.eq_ignore_ascii_case("n") {
            return Err("Installer cancelled.".to_string());
        }
        let p = expand_tilde(v);
        if p.exists() && p.is_dir() {
            return Ok(p);
        }
        eprintln!(
            "Invalid path: {} (must already exist). Please choose [y/n/path] again.",
            p.display()
        );
    }
}

fn is_dir_writable(dir: &Path) -> bool {
    if !dir.exists() || !dir.is_dir() {
        return false;
    }
    let probe = dir.join(format!(
        ".jx_write_test_{}_{}",
        process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_millis()
    ));
    match std::fs::write(&probe, b"ok") {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

fn default_install_dir() -> Result<PathBuf, String> {
    #[cfg(windows)]
    {
        // Keep default conservative on Windows to avoid UAC issues.
        return env::current_dir().map_err(|e| format!("Failed to resolve current directory: {e}"));
    }
    #[cfg(not(windows))]
    {
        let preferred = PathBuf::from("/usr/bin");
        if is_dir_writable(&preferred) {
            return Ok(preferred);
        }
        return env::current_dir().map_err(|e| format!("Failed to resolve current directory: {e}"));
    }
}

fn prompt_install_dir() -> Result<PathBuf, String> {
    let default_dir = default_install_dir()?;
    loop {
        println!(
            "Choose install directory for `jx` [y/n/path]\n  y: {}\n  n: cancel\n  path: existing writable directory",
            default_dir.display()
        );
        print!("Input [y]: ");
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush stdout: {e}"))?;
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .map_err(|e| format!("Failed to read input: {e}"))?;
        let v = line.trim();
        if v.is_empty() || v.eq_ignore_ascii_case("y") {
            if is_dir_writable(&default_dir) {
                return Ok(default_dir.clone());
            }
            eprintln!(
                "Default install dir is not writable: {}",
                default_dir.display()
            );
            continue;
        }
        if v.eq_ignore_ascii_case("n") {
            return Err("Installer cancelled.".to_string());
        }
        let p = expand_tilde(v);
        if !p.exists() || !p.is_dir() {
            eprintln!("Invalid path: {} (must already exist).", p.display());
            continue;
        }
        if !is_dir_writable(&p) {
            eprintln!("Path is not writable: {}", p.display());
            continue;
        }
        return Ok(p);
    }
}

fn install_jx_binary(install_dir: &Path) -> Result<PathBuf, String> {
    let src = env::current_exe().map_err(|e| format!("Failed to locate installer binary: {e}"))?;
    let dst_name = if cfg!(windows) { "jx.exe" } else { "jx" };
    let dst = install_dir.join(dst_name);
    if dst.exists() {
        std::fs::remove_file(&dst)
            .map_err(|e| format!("Failed to replace existing {}: {e}", dst.display()))?;
    }
    std::fs::copy(&src, &dst).map_err(|e| {
        format!(
            "Failed to copy installer binary {} -> {}: {e}",
            src.display(),
            dst.display()
        )
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&dst)
            .map_err(|e| format!("Failed to stat {}: {e}", dst.display()))?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&dst, perms)
            .map_err(|e| format!("Failed to chmod {}: {e}", dst.display()))?;
    }
    Ok(dst)
}

fn write_runtime_home_config_near_binary(binary_path: &Path, runtime_home: &Path) {
    let Some(parent) = binary_path.parent() else {
        return;
    };
    let cfg = parent.join(RUNTIME_HOME_CONFIG);
    let _ = std::fs::write(cfg, format!("{}\n", runtime_home.display()));
}

fn parse_update_args(args: &[String]) -> Result<UpdateOptions, String> {
    let mut latest = false;
    let mut local_source: Option<String> = None;
    let mut verbose = false;
    let mut force_reinstall = false;

    for token in args {
        match token.as_str() {
            "latest" | "--latest" => latest = true,
            "--verbose" => verbose = true,
            "--force-reinstall" | "--reinstall" | "--full" => force_reinstall = true,
            "-h" | "--help" | "help" => {
                println!(
                    "Update usage:\n  jx --update [latest|<local_path>] [--verbose] [--force-reinstall]\n  Example: jx --update ."
                );
                return Err(String::new());
            }
            other => {
                if Path::new(other).exists() {
                    if local_source.is_some() {
                        return Err(
                            "Only one local source path is allowed for `--update`.".to_string()
                        );
                    }
                    local_source = Some(other.to_string());
                } else {
                    return Err(format!(
                        "Unknown update option: {other}\nUpdate usage:\n  jx --update [latest|<local_path>] [--verbose] [--force-reinstall]\n  Example: jx --update ."
                    ));
                }
            }
        }
    }

    if latest && local_source.is_some() {
        return Err("`latest` and local path cannot be used together.".to_string());
    }

    let source = if latest {
        UpdateSource::Latest
    } else if let Some(path) = local_source {
        UpdateSource::Local(path)
    } else {
        UpdateSource::Pypi
    };

    Ok(UpdateOptions {
        source,
        verbose,
        force_reinstall,
    })
}

fn run_update(opts: UpdateOptions) -> Result<i32, String> {
    let home = runtime_home()?;
    let python = ensure_venv()?;
    let before = installed_version(&python);

    match &opts.source {
        UpdateSource::Latest => {
            ensure_git_available()?;
            let repo_url = repo_url_from_spec(GITHUB_SPEC);
            let remote_head = remote_head_commit(&repo_url);
            let local_commit = read_commit_marker(&home);
            if let (Some(local), Some(remote)) = (local_commit.as_deref(), remote_head.as_deref()) {
                if local == remote {
                    print_success_line("Already at latest GitHub commit.");
                    print_current_version(&python);
                    return Ok(0);
                }
                let start = Instant::now();
                match try_fast_python_update_latest(
                    &python,
                    &home,
                    &repo_url,
                    local,
                    remote,
                    opts.verbose,
                ) {
                    Ok(true) => {
                        print_success_line(&format!(
                            "JanusX python-only update completed. [{}]",
                            format_elapsed(start.elapsed())
                        ));
                        print_current_version(&python);
                        return Ok(0);
                    }
                    Ok(false) => {
                        if opts.verbose {
                            println!("Rust or build files changed; falling back to full update.");
                        }
                    }
                    Err(err) => {
                        if opts.verbose {
                            eprintln!("Fast update path failed: {err}");
                            eprintln!("Falling back to full update...");
                        }
                    }
                }
            }
            // GitHub latest should refresh even when package version string is unchanged.
            let gh_force_reinstall = true;
            if opts.verbose {
                println!("Updating from GitHub...");
            }
            match pip_install(
                &python,
                &home,
                GITHUB_SPEC,
                gh_force_reinstall,
                opts.verbose,
                "Updating from GitHub...",
            ) {
                Ok(elapsed) => {
                    if let Some(remote) = remote_head.as_deref() {
                        write_commit_marker(&home, remote);
                    }
                    print_success_line(&format!(
                        "JanusX update completed. [{}]",
                        format_elapsed(elapsed)
                    ));
                    print_current_version(&python);
                    return Ok(0);
                }
                Err(err_primary) => {
                    eprintln!("Direct GitHub update failed, retrying with proxy...");
                    if opts.verbose {
                        eprintln!("Reason: {err_primary}");
                    }
                    let elapsed = pip_install(
                        &python,
                        &home,
                        GITHUB_PROXY_SPEC,
                        gh_force_reinstall,
                        opts.verbose,
                        "Updating from proxy...",
                    )?;
                    if let Some(remote) = remote_head.as_deref() {
                        write_commit_marker(&home, remote);
                    }
                    print_success_line(&format!(
                        "JanusX update completed (via proxy). [{}]",
                        format_elapsed(elapsed)
                    ));
                    print_current_version(&python);
                    return Ok(0);
                }
            }
        }
        UpdateSource::Local(path) => {
            if opts.verbose {
                println!("Updating from local source: {path}");
            }
            let elapsed = pip_install(
                &python,
                &home,
                path,
                opts.force_reinstall,
                opts.verbose,
                "Updating from local source...",
            )?;
            print_success_line(&format!(
                "JanusX local update completed. [{}]",
                format_elapsed(elapsed)
            ));
            print_current_version(&python);
            return Ok(0);
        }
        UpdateSource::Pypi => {}
    }

    if opts.verbose {
        println!("Updating from PyPI...");
    }
    let elapsed = pip_install(
        &python,
        &home,
        PYPI_SPEC,
        opts.force_reinstall,
        opts.verbose,
        "Updating from PyPI...",
    )?;
    let after = installed_version(&python);
    if !opts.force_reinstall && before.is_some() && before == after {
        if let Some(v) = after {
            print_success_line(&format!(
                "It is the latest PyPI version ({v}) [{}]",
                format_elapsed(elapsed)
            ));
        } else {
            print_success_line(&format!(
                "It is already the latest PyPI version. [{}]",
                format_elapsed(elapsed)
            ));
        }
        println!("Use `jx --update latest` for GitHub latest.");
    } else {
        print_success_line(&format!(
            "JanusX update completed. [{}]",
            format_elapsed(elapsed)
        ));
    }
    print_current_version(&python);
    Ok(0)
}

fn run_python_janusx(python: &Path, args: &[String]) -> Result<i32, String> {
    let status = Command::new(python)
        .arg("-m")
        .arg("janusx.script.JanusX")
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("Failed to run JanusX module: {e}"))?;
    Ok(exit_code(status))
}

fn ensure_runtime(verbose_bootstrap: bool) -> Result<PathBuf, String> {
    let home = runtime_home()?;
    let python = ensure_venv()?;
    if !is_janusx_installed(&python) {
        if verbose_bootstrap {
            println!("Bootstrapping JanusX from PyPI...");
        } else {
            println!("JanusX runtime not found in venv. Installing from PyPI...");
        }
        let _ = pip_install(
            &python,
            &home,
            PYPI_SPEC,
            false,
            true,
            "Updating from PyPI...",
        )?;
    }
    Ok(python)
}

fn ensure_venv() -> Result<PathBuf, String> {
    let home = runtime_home()?;
    if should_rebuild_runtime(&home)? {
        println!(
            "Detected newer jx binary than runtime cache. Rebuilding {} ...",
            home.display()
        );
        std::fs::remove_dir_all(&home)
            .map_err(|e| format!("Failed to remove runtime directory {}: {e}", home.display()))?;
    }
    if !home.exists() {
        println!(
            "Runtime directory not found. Initializing {} ...",
            home.display()
        );
        std::fs::create_dir_all(&home)
            .map_err(|e| format!("Failed to create runtime directory {}: {e}", home.display()))?;
    }
    let venv = home.join("venv");
    let py = venv_python(&venv);
    if py.exists() {
        return Ok(py);
    }

    let sys_py = find_system_python().ok_or_else(|| python_install_hint())?;
    let status = Command::new(&sys_py)
        .arg("-m")
        .arg("venv")
        .arg(&venv)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("Failed to create venv with `{sys_py}`: {e}"))?;
    if !status.success() {
        return Err(format!(
            "Failed to create venv at {} (exit={}).",
            venv.display(),
            exit_code(status)
        ));
    }
    if !py.exists() {
        return Err(format!(
            "venv was created but Python executable not found at {}",
            py.display()
        ));
    }
    Ok(py)
}

fn should_rebuild_runtime(home: &Path) -> Result<bool, String> {
    if !home.exists() {
        return Ok(false);
    }
    let launcher_ts = launcher_mtime()?;
    let runtime_ts = path_mtime(home)?;
    match (launcher_ts, runtime_ts) {
        (Some(l), Some(r)) => Ok(l > r),
        _ => Ok(false),
    }
}

fn launcher_mtime() -> Result<Option<SystemTime>, String> {
    let exe =
        env::current_exe().map_err(|e| format!("Failed to locate current executable: {e}"))?;
    path_mtime(&exe)
}

fn path_mtime(path: &Path) -> Result<Option<SystemTime>, String> {
    let md = std::fs::metadata(path)
        .map_err(|e| format!("Failed to read metadata {}: {e}", path.display()))?;
    Ok(md.modified().ok())
}

fn runtime_home() -> Result<PathBuf, String> {
    if let Some(v) = env::var_os("JX_HOME") {
        return Ok(PathBuf::from(v));
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            let cfg = parent.join(RUNTIME_HOME_CONFIG);
            if cfg.exists() {
                if let Ok(text) = std::fs::read_to_string(&cfg) {
                    let v = text.trim();
                    if !v.is_empty() {
                        return Ok(expand_tilde(v));
                    }
                }
            }
        }
    }
    default_runtime_home()
}

fn venv_python(venv: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        return venv.join("Scripts").join("python.exe");
    }
    #[cfg(not(windows))]
    {
        venv.join("bin").join("python")
    }
}

fn find_system_python() -> Option<String> {
    if let Ok(val) = env::var("JX_PYTHON") {
        if command_ok(&val, &["--version"]) {
            return Some(val);
        }
    }
    for cand in ["python3", "python", "py"] {
        if command_ok(cand, &["--version"]) {
            return Some(cand.to_string());
        }
    }
    None
}

fn command_ok(bin: &str, args: &[&str]) -> bool {
    Command::new(bin)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn is_janusx_installed(python: &Path) -> bool {
    Command::new(python)
        .arg("-c")
        .arg("import janusx")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn installed_version(python: &Path) -> Option<String> {
    let out = Command::new(python)
        .arg("-c")
        .arg("import importlib.metadata as m; print(m.version('janusx'))")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn print_current_version(python: &Path) {
    if let Some(v) = installed_version(python) {
        println!("Current version: {v}");
    } else {
        println!("Current version: unknown");
    }
}

fn print_version(python: &Path, runtime_home: &Path) {
    let version = installed_version(python).unwrap_or_else(|| "0.0.0".to_string());
    let build_time =
        python_core_update_time(python, runtime_home).unwrap_or_else(|| "unknown".to_string());
    println!("{LOGO}");
    println!("JanusX v{version} by {VERSION_AUTHOR}");
    println!("Please report issues to <{VERSION_CONTACT}>");
    println!("Build time: {build_time}");
    println!();
}

fn python_core_update_time(python: &Path, runtime_home: &Path) -> Option<String> {
    let marker = runtime_home.join(UPDATE_TIME_MARKER);
    if marker.exists() {
        if let Ok(text) = std::fs::read_to_string(&marker) {
            let t = text.trim().to_string();
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    let out = Command::new(python)
        .arg("-c")
        .arg("import janusx, pathlib, datetime; p=pathlib.Path(janusx.__file__).resolve().parent; print(datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M'))")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn read_commit_marker(runtime_home: &Path) -> Option<String> {
    let p = runtime_home.join(COMMIT_MARKER);
    let text = std::fs::read_to_string(p).ok()?;
    let v = text.trim().to_string();
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

fn write_commit_marker(runtime_home: &Path, commit: &str) {
    let p = runtime_home.join(COMMIT_MARKER);
    let _ = std::fs::write(p, format!("{}\n", commit.trim()));
}

fn repo_url_from_spec(spec: &str) -> String {
    if let Some(s) = spec.strip_prefix("git+") {
        return s.to_string();
    }
    spec.to_string()
}

fn git_output(cwd: Option<&Path>, args: &[&str]) -> Option<String> {
    let mut cmd = Command::new("git");
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::null());
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn remote_head_commit(repo_url: &str) -> Option<String> {
    let out = git_output(None, &["ls-remote", repo_url, "HEAD"])?;
    let first = out.lines().next()?.trim();
    let sha = first.split_whitespace().next()?.trim().to_string();
    if sha.len() >= 7 {
        Some(sha)
    } else {
        None
    }
}

fn is_python_only_change(path: &str) -> bool {
    let p = path.trim();
    if p.is_empty() {
        return true;
    }
    if p.starts_with("python/") {
        return true;
    }
    matches!(
        p,
        "README.md" | "README.zh-CN.md" | "README_cn.md" | "LICENSE" | "MANIFEST.in" | ".gitignore"
    )
}

fn installed_package_dir(python: &Path) -> Option<PathBuf> {
    let out = Command::new(python)
        .arg("-c")
        .arg("import janusx, pathlib; print(pathlib.Path(janusx.__file__).resolve().parent)")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(PathBuf::from(text))
    }
}

fn overlay_dir(src: &Path, dst: &Path) -> Result<(), String> {
    if !src.exists() {
        return Err(format!("Source directory not found: {}", src.display()));
    }
    if !dst.exists() {
        std::fs::create_dir_all(dst)
            .map_err(|e| format!("Failed to create {}: {e}", dst.display()))?;
    }
    for entry in
        std::fs::read_dir(src).map_err(|e| format!("Failed to read {}: {e}", src.display()))?
    {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
        let p = entry.path();
        let name = entry.file_name();
        let target = dst.join(name);
        let ft = entry
            .file_type()
            .map_err(|e| format!("Failed to read file type {}: {e}", p.display()))?;
        if ft.is_dir() {
            if p.file_name().and_then(|x| x.to_str()) == Some("__pycache__") {
                continue;
            }
            overlay_dir(&p, &target)?;
        } else if ft.is_file() {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
            }
            std::fs::copy(&p, &target).map_err(|e| {
                format!(
                    "Failed to copy {} -> {}: {e}",
                    p.display(),
                    target.display()
                )
            })?;
        }
    }
    Ok(())
}

fn try_fast_python_update_latest(
    python: &Path,
    runtime_home: &Path,
    repo_url: &str,
    local_commit: &str,
    remote_commit: &str,
    verbose: bool,
) -> Result<bool, String> {
    let temp = env::temp_dir().join(format!(
        "jx_fastupdate_{}_{}",
        process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_millis()
    ));
    std::fs::create_dir_all(&temp)
        .map_err(|e| format!("Failed to create temp dir {}: {e}", temp.display()))?;

    let run_git_status = |args: &[&str]| -> bool {
        Command::new("git")
            .args(args)
            .current_dir(&temp)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };

    let cleanup = || {
        let _ = std::fs::remove_dir_all(&temp);
    };

    if !run_git_status(&["init"]) {
        cleanup();
        return Ok(false);
    }
    if !run_git_status(&["remote", "add", "origin", repo_url]) {
        cleanup();
        return Ok(false);
    }
    if !run_git_status(&["fetch", "--depth=1", "origin", local_commit]) {
        cleanup();
        return Ok(false);
    }
    if !run_git_status(&["fetch", "--depth=1", "origin", remote_commit]) {
        cleanup();
        return Ok(false);
    }

    let diff = git_output(
        Some(&temp),
        &["diff", "--name-only", local_commit, remote_commit],
    );
    let Some(diff_text) = diff else {
        cleanup();
        return Ok(false);
    };
    let changed: Vec<String> = diff_text
        .lines()
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect();
    if changed.is_empty() {
        write_commit_marker(runtime_home, remote_commit);
        cleanup();
        return Ok(true);
    }
    if !changed.iter().all(|p| is_python_only_change(p)) {
        cleanup();
        return Ok(false);
    }

    if !run_git_status(&["checkout", "--detach", remote_commit]) {
        cleanup();
        return Ok(false);
    }
    let src_pkg = temp.join("python").join("janusx");
    let dst_pkg = installed_package_dir(python).ok_or_else(|| {
        "Failed to locate installed janusx package directory in launcher venv.".to_string()
    })?;
    if verbose {
        println!(
            "Applying Python-only fast update: {} -> {}",
            src_pkg.display(),
            dst_pkg.display()
        );
    }
    overlay_dir(&src_pkg, &dst_pkg)?;
    write_commit_marker(runtime_home, remote_commit);
    write_python_core_update_marker(runtime_home, python);
    cleanup();
    Ok(true)
}

fn write_python_core_update_marker(runtime_home: &Path, python: &Path) {
    let marker = runtime_home.join(UPDATE_TIME_MARKER);
    let marker_s = marker.to_string_lossy().to_string();
    let _ = Command::new(python)
        .arg("-c")
        .arg("import datetime, pathlib, sys; p=pathlib.Path(sys.argv[1]); p.write_text(datetime.datetime.now().strftime('%Y-%m-%d %H:%M') + '\\n', encoding='utf-8')")
        .arg(marker_s)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn format_elapsed(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 60.0 {
        return format!("{secs:.1}s");
    }
    let total = secs.round() as u64;
    if total < 3600 {
        let m = total / 60;
        let s = total % 60;
        return format!("{m}m{s:02}s");
    }
    let h = total / 3600;
    let m = (total % 3600) / 60;
    format!("{h}h{m:02}m")
}

fn print_success_line(msg: &str) {
    if io::stdout().is_terminal() {
        println!("\x1b[32m✔︎ {}\x1b[0m", msg);
    } else {
        println!("✔︎ {}", msg);
    }
}

fn run_with_spinner(cmd: &mut Command, desc: &str) -> Result<(Output, Duration), String> {
    let start = Instant::now();
    let is_tty = io::stdout().is_terminal();
    let done = Arc::new(AtomicBool::new(false));
    let done_c = Arc::clone(&done);
    let desc_s = desc.to_string();

    let spinner = if is_tty {
        Some(thread::spawn(move || {
            let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut i = 0usize;
            while !done_c.load(Ordering::Relaxed) {
                let elapsed = format_elapsed(start.elapsed());
                print!("\r{} {} [{}]", frames[i % frames.len()], desc_s, elapsed);
                let _ = io::stdout().flush();
                i += 1;
                thread::sleep(Duration::from_millis(80));
            }
            print!("\r{}\r", " ".repeat(120));
            let _ = io::stdout().flush();
        }))
    } else {
        None
    };

    let out = cmd
        .output()
        .map_err(|e| format!("Failed to run command: {e}"))?;
    done.store(true, Ordering::Relaxed);
    if let Some(h) = spinner {
        let _ = h.join();
    }
    Ok((out, start.elapsed()))
}

fn pip_install(
    python: &Path,
    runtime_home: &Path,
    spec: &str,
    force_reinstall: bool,
    verbose: bool,
    desc: &str,
) -> Result<Duration, String> {
    let mut cmd = Command::new(python);
    cmd.arg("-m")
        .arg("pip")
        .arg("install")
        .arg("--upgrade")
        .arg("--disable-pip-version-check");
    if force_reinstall {
        cmd.arg("--force-reinstall").arg("--no-cache-dir");
    }
    cmd.arg(spec);

    if verbose {
        let start = Instant::now();
        cmd.stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        let status = cmd
            .status()
            .map_err(|e| format!("Failed to run pip install for {spec}: {e}"))?;
        if status.success() {
            remove_conflicting_jx_entrypoints(python);
            write_python_core_update_marker(runtime_home, python);
            return Ok(start.elapsed());
        }
        return Err(format!(
            "pip install failed (exit={}) for {spec}",
            exit_code(status)
        ));
    }

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let (out, elapsed) = run_with_spinner(&mut cmd, desc)?;
    if out.status.success() {
        remove_conflicting_jx_entrypoints(python);
        write_python_core_update_marker(runtime_home, python);
        return Ok(elapsed);
    }
    let mut msg = String::new();
    msg.push_str(&String::from_utf8_lossy(&out.stdout));
    msg.push_str(&String::from_utf8_lossy(&out.stderr));
    Err(format!(
        "pip install failed (exit={}) for {spec}\n{}",
        exit_code(out.status),
        msg.trim()
    ))
}

fn remove_conflicting_jx_entrypoints(python: &Path) {
    let Some(dir) = python.parent() else {
        return;
    };
    let candidates = ["jx", "jx.exe", "jx-script.py", "jx.cmd", "jx.bat"];
    for name in candidates {
        let p = dir.join(name);
        if p.exists() {
            let _ = std::fs::remove_file(&p);
        }
    }
}

fn ensure_git_available() -> Result<(), String> {
    if command_ok("git", &["--version"]) {
        return Ok(());
    }
    Err(git_install_hint())
}

fn git_install_hint() -> String {
    if cfg!(windows) {
        return "Git is required for `jx --update latest`, but it was not found in PATH.\n\
Windows install options:\n\
  1) winget: winget install --id Git.Git -e\n\
  2) choco:  choco install git -y\n\
  3) installer: https://git-scm.com/download/win"
            .to_string();
    }
    if cfg!(target_os = "macos") {
        return "Git is required for `jx --update latest`, but it was not found in PATH.\n\
macOS install options:\n\
  1) xcode-select --install\n\
  2) brew install git"
            .to_string();
    }
    "Git is required for `jx --update latest`, but it was not found in PATH.\n\
Linux install options:\n\
  Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y git\n\
  RHEL/CentOS/Fedora: sudo dnf install -y git (or: sudo yum install -y git)\n\
  openSUSE: sudo zypper install -y git"
        .to_string()
}

fn python_install_hint() -> String {
    if cfg!(windows) {
        return "No system Python found. Install Python 3 first, or set JX_PYTHON.\n\
Windows examples:\n\
  winget install Python.Python.3.12\n\
  set JX_PYTHON=C:\\Python312\\python.exe"
            .to_string();
    }
    if cfg!(target_os = "macos") {
        return "No system Python found. Install Python 3 first, or set JX_PYTHON.\n\
macOS examples:\n\
  brew install python\n\
  export JX_PYTHON=/opt/homebrew/bin/python3"
            .to_string();
    }
    "No system Python found. Install Python 3 first, or set JX_PYTHON.\n\
Linux examples:\n\
  sudo apt-get install -y python3 python3-venv\n\
  export JX_PYTHON=/usr/bin/python3"
        .to_string()
}

fn exit_code(status: ExitStatus) -> i32 {
    status.code().unwrap_or(1)
}
