use std::env;
use std::path::{Path, PathBuf};
use std::process::{self, Command, ExitStatus, Stdio};
use std::time::SystemTime;

const PYPI_SPEC: &str = "janusx";
const GITHUB_SPEC: &str = "git+https://github.com/FJingxian/JanusX.git";
const GITHUB_PROXY_SPEC: &str = "git+https://gh-proxy.org/https://github.com/FJingxian/JanusX.git";

#[derive(Clone, Copy, Debug)]
struct UpdateOptions {
    latest: bool,
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
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        let python = ensure_runtime(false)?;
        return run_python_janusx(&python, &[]);
    }

    let head = args[0].as_str();
    if matches!(head, "-h" | "--help" | "-v" | "--version") {
        let python = ensure_runtime(false)?;
        return run_python_janusx(&python, &args);
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

fn parse_update_args(args: &[String]) -> Result<UpdateOptions, String> {
    let mut latest = false;
    let mut verbose = false;
    let mut force_reinstall = false;

    for token in args {
        match token.as_str() {
            "latest" | "--latest" => latest = true,
            "--verbose" => verbose = true,
            "--force-reinstall" | "--reinstall" | "--full" => force_reinstall = true,
            "-h" | "--help" | "help" => {
                println!("Update usage:\n  jx --update [latest] [--verbose] [--force-reinstall]");
                return Err(String::new());
            }
            other => {
                return Err(format!(
                    "Unknown update option: {other}\nUpdate usage:\n  jx --update [latest] [--verbose] [--force-reinstall]"
                ));
            }
        }
    }

    Ok(UpdateOptions {
        latest,
        verbose,
        force_reinstall,
    })
}

fn run_update(opts: UpdateOptions) -> Result<i32, String> {
    let python = ensure_venv()?;
    let before = installed_version(&python);

    if opts.latest {
        ensure_git_available()?;
        // GitHub latest should refresh even when package version string is unchanged.
        let gh_force_reinstall = true;
        if opts.verbose {
            println!("Updating from GitHub...");
        }
        match pip_install(&python, GITHUB_SPEC, gh_force_reinstall, opts.verbose) {
            Ok(_) => {
                println!("JanusX update completed.");
                return Ok(0);
            }
            Err(err_primary) => {
                eprintln!("Direct GitHub update failed, retrying with proxy...");
                if opts.verbose {
                    eprintln!("Reason: {err_primary}");
                }
                pip_install(&python, GITHUB_PROXY_SPEC, gh_force_reinstall, opts.verbose)?;
                println!("JanusX update completed (via proxy).");
                return Ok(0);
            }
        }
    }

    if opts.verbose {
        println!("Updating from PyPI...");
    }
    pip_install(&python, PYPI_SPEC, opts.force_reinstall, opts.verbose)?;
    let after = installed_version(&python);
    if !opts.force_reinstall && before.is_some() && before == after {
        if let Some(v) = after {
            println!("It is the latest PyPI version ({v})");
        } else {
            println!("It is already the latest PyPI version.");
        }
        println!("Use `jx --update latest` for GitHub latest.");
    } else {
        println!("JanusX update completed.");
    }
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
    let python = ensure_venv()?;
    if !is_janusx_installed(&python) {
        if verbose_bootstrap {
            println!("Bootstrapping JanusX from PyPI...");
        } else {
            println!("JanusX runtime not found in venv. Installing from PyPI...");
        }
        pip_install(&python, PYPI_SPEC, false, true)?;
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

fn pip_install(
    python: &Path,
    spec: &str,
    force_reinstall: bool,
    verbose: bool,
) -> Result<(), String> {
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
        cmd.stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        let status = cmd
            .status()
            .map_err(|e| format!("Failed to run pip install for {spec}: {e}"))?;
        if status.success() {
            remove_conflicting_jx_entrypoints(python);
            return Ok(());
        }
        return Err(format!(
            "pip install failed (exit={}) for {spec}",
            exit_code(status)
        ));
    }

    let out = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run pip install for {spec}: {e}"))?;
    if out.status.success() {
        remove_conflicting_jx_entrypoints(python);
        return Ok(());
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
