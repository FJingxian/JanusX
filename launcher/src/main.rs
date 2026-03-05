use std::collections::{BTreeSet, VecDeque};
use std::env;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command, ExitStatus, Output, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
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
const INSTALLER_RELAUNCH_ENV: &str = "JX_INSTALLER_RELAUNCHED";
const SKIP_RUNTIME_REBUILD_ENV: &str = "JX_SKIP_RUNTIME_REBUILD";
const SKIP_WARMUP_ENV: &str = "JX_SKIP_WARMUP";
const WARMUP_MARKER: &str = ".runtime_warmed";
const CLI_HELP_TEXT: &str = r#"Usage:
    jx <module> [options]

Options:
    -h, --help             Show this help message
    -v, --version          Show version/build information
    -update, --update      Update JanusX: `jx --update [latest] [--verbose]`
    -uninstall, --uninstall  Remove JanusX runtime and launcher files

Modules:
    Genome-wide Association Studies (GWAS):
    grm           Build genomic relationship matrix
    pca           Principal component analysis for population structure
    gwas          Run genome-wide association analysis
    postgwas      Post-process GWAS results and downstream plots

    Genomic Selection (GS):
    gs            Genomic prediction and model-based selection

    GARFIELD:
    garfield      Random-forest based marker-trait association
    postgarfield  Summarize and visualize GARFIELD outputs

    Bulk Segregation Analysis (BSA):
    postbsa       Post-process and visualize BSA results

    Pipeline and utility:
    fastq2vcf     Variant-calling pipeline from FASTQ to VCF
    gmerge        Merge genotype/variant tables

    Benchmark:
    sim           Quick simulation workflow
    simulation    Extended simulation and benchmarking workflow
"#;
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
    let installer_mode = is_installer_binary();
    let code = match run() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            1
        }
    };
    if installer_mode {
        pause_before_exit();
    }
    process::exit(code);
}

fn pause_before_exit() {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return;
    }
    println!();
    print!("Press Enter to exit...");
    let _ = io::stdout().flush();
    let mut buf = String::new();
    let _ = io::stdin().read_line(&mut buf);
}

fn run() -> Result<i32, String> {
    if is_installer_binary() {
        return run_installer();
    }

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        let home = runtime_home()?;
        let python = ensure_runtime(false)?;
        let _ = maybe_auto_warmup(&home)?;
        return run_python_janusx(&python, &[]);
    }

    let head = args[0].as_str();
    if matches!(head, "-h" | "--help") {
        println!("{LOGO}");
        println!("{}", CLI_HELP_TEXT.trim());
        return Ok(0);
    }
    if matches!(head, "-v" | "--version") {
        let home = runtime_home()?;
        let python = ensure_runtime(false)?;
        if maybe_auto_warmup(&home)? {
            return Ok(0);
        }
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
    if matches!(head, "-uninstall" | "--uninstall") {
        return run_uninstall(&args[1..]);
    }

    let home = runtime_home()?;
    let python = ensure_runtime(false)?;
    let _ = maybe_auto_warmup(&home)?;
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
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        if env::var_os(INSTALLER_RELAUNCH_ENV).is_none() && relaunch_installer_in_terminal()? {
            return Ok(0);
        }
        return Err("Installer requires a terminal UI.\n\
macOS: please double-click the `.command` installer file.\n\
Linux: please run the `.run` installer file."
            .to_string());
    }

    println!("{LOGO}");
    println!("JanusX Installer");
    println!("Guided setup: Runtime home -> jx location -> PATH\n");
    check_and_handle_existing_jx_in_path()?;

    let (runtime_home, install_dir) = prompt_runtime_home()?;
    env::set_var("JX_HOME", &runtime_home);
    println!("Runtime home: {}", runtime_home.display());

    ensure_runtime(false)?;

    let installed_jx = install_jx_binary(&install_dir)?;
    write_runtime_home_config_near_binary(&installed_jx, &runtime_home);
    warm_up_jx(&installed_jx, &runtime_home)?;
    println!("Installation complete.");
    print_path_setup_hint(&installed_jx);
    Ok(0)
}

fn warm_up_jx(jx_bin: &Path, runtime_home: &Path) -> Result<(), String> {
    let warm_start = Instant::now();
    let warm_done = Arc::new(AtomicBool::new(false));
    let warm_done_c = Arc::clone(&warm_done);
    let warm_progress = if io::stdout().is_terminal() {
        Some(thread::spawn(move || {
            let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut i = 0usize;
            while !warm_done_c.load(Ordering::Relaxed) {
                let line = format!(
                    "{} Warming up ...[{}]",
                    frames[i % frames.len()],
                    format_elapsed(warm_start.elapsed())
                );
                if supports_color() {
                    print!("\r{}", style_green(&line));
                } else {
                    print!("\r{}", line);
                }
                let _ = io::stdout().flush();
                i += 1;
                thread::sleep(Duration::from_millis(100));
            }
            print!("\r{}\r", " ".repeat(120));
            let _ = io::stdout().flush();
        }))
    } else {
        None
    };

    let stop_warm_progress = |done: Arc<AtomicBool>, handle: Option<thread::JoinHandle<()>>| {
        done.store(true, Ordering::Relaxed);
        if let Some(h) = handle {
            let _ = h.join();
        }
    };

    let h_status = Command::new(jx_bin)
        .arg("-h")
        .env("JX_HOME", runtime_home)
        .env(SKIP_RUNTIME_REBUILD_ENV, "1")
        .env(SKIP_WARMUP_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("Failed to run `{}` -h: {e}", jx_bin.display()))?;
    if !h_status.success() {
        stop_warm_progress(warm_done, warm_progress);
        return Err(format!(
            "Warm-up command failed: {} -h (exit={})",
            jx_bin.display(),
            exit_code(h_status)
        ));
    }

    let py = venv_python(&runtime_home.join("venv"));
    if py.exists() {
        let py_status = Command::new(&py)
            .arg("-m")
            .arg("janusx.script.JanusX")
            .arg("-h")
            .env("JX_HOME", runtime_home)
            .env(SKIP_RUNTIME_REBUILD_ENV, "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| {
                format!(
                    "Failed to run `{}` -m janusx.script.JanusX -h: {e}",
                    py.display()
                )
            })?;
        if !py_status.success() {
            stop_warm_progress(warm_done, warm_progress);
            return Err(format!(
                "Warm-up command failed: {} -m janusx.script.JanusX -h (exit={})",
                py.display(),
                exit_code(py_status)
            ));
        }
    }

    let output = Command::new(jx_bin)
        .arg("-v")
        .env("JX_HOME", runtime_home)
        .env(SKIP_RUNTIME_REBUILD_ENV, "1")
        .env(SKIP_WARMUP_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run `{}` -v: {e}", jx_bin.display()))?;
    stop_warm_progress(warm_done, warm_progress);

    print_success_line(&format!(
        "Warming up ...[{}]",
        format_elapsed(warm_start.elapsed())
    ));
    print_dim_block(&String::from_utf8_lossy(&output.stdout));
    print_dim_block(&String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        return Err(format!(
            "Warm-up command failed: {} -v (exit={})",
            jx_bin.display(),
            exit_code(output.status)
        ));
    }
    write_warmup_marker(runtime_home);
    Ok(())
}

fn maybe_auto_warmup(runtime_home: &Path) -> Result<bool, String> {
    if should_skip_auto_warmup() || is_warmup_done(runtime_home) {
        return Ok(false);
    }
    let Some(jx_bin) = env::current_exe().ok() else {
        return Ok(false);
    };
    warm_up_jx(&jx_bin, runtime_home)?;
    Ok(true)
}

fn should_skip_auto_warmup() -> bool {
    env::var(SKIP_WARMUP_ENV)
        .ok()
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

fn warmup_marker_path(runtime_home: &Path) -> PathBuf {
    runtime_home.join(WARMUP_MARKER)
}

fn is_warmup_done(runtime_home: &Path) -> bool {
    warmup_marker_path(runtime_home).exists()
}

fn write_warmup_marker(runtime_home: &Path) {
    let marker = warmup_marker_path(runtime_home);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_secs();
    let _ = std::fs::write(marker, format!("{ts}\n"));
}

fn relaunch_installer_in_terminal() -> Result<bool, String> {
    #[cfg(target_os = "macos")]
    {
        let exe =
            env::current_exe().map_err(|e| format!("Failed to locate installer binary: {e}"))?;
        let cmdline = format!(
            "export {flag}=1; {exe}",
            flag = INSTALLER_RELAUNCH_ENV,
            exe = sh_quote(&exe.to_string_lossy())
        );
        let osa_expr = format!(
            "tell application \"Terminal\" to do script {}",
            applescript_quote(&cmdline)
        );
        let ok = Command::new("osascript")
            .arg("-e")
            .arg(osa_expr)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        return Ok(ok);
    }
    #[cfg(target_os = "linux")]
    {
        let exe =
            env::current_exe().map_err(|e| format!("Failed to locate installer binary: {e}"))?;
        let cmdline = format!(
            "export {flag}=1; {exe}",
            flag = INSTALLER_RELAUNCH_ENV,
            exe = sh_quote(&exe.to_string_lossy())
        );
        let attempts: [(&str, &[&str]); 5] = [
            ("x-terminal-emulator", &["-e", "sh", "-lc"]),
            ("gnome-terminal", &["--", "sh", "-lc"]),
            ("konsole", &["-e", "sh", "-lc"]),
            ("xfce4-terminal", &["--command", "sh -lc"]),
            ("xterm", &["-e", "sh", "-lc"]),
        ];
        for (bin, fixed) in attempts {
            let mut cmd = Command::new(bin);
            if bin == "xfce4-terminal" {
                let wrapped = format!("{} {}", fixed[1], sh_quote(&cmdline));
                cmd.arg(fixed[0]).arg(wrapped);
            } else {
                cmd.args(fixed).arg(&cmdline);
            }
            let launched = cmd
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if launched {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Ok(false)
    }
}

fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

fn applescript_quote(s: &str) -> String {
    let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn print_path_setup_hint(installed_jx: &Path) {
    let Some(install_dir) = installed_jx.parent() else {
        return;
    };

    if is_dir_in_path(install_dir) {
        println!("`jx` is already in PATH.");
        return;
    }

    println!();
    println!("`jx` is not in PATH yet.");
    println!("Add this directory to PATH (persistent):");
    println!("  {}", install_dir.display());

    #[cfg(target_os = "windows")]
    {
        let d = install_dir.display();
        match detect_windows_shell() {
            WindowsShell::Cmd => {
                println!("cmd (persistent, user):");
                println!("  setx PATH \"{d};%PATH%\"");
            }
            WindowsShell::PowerShell | WindowsShell::Unknown => {
                println!("PowerShell (persistent, user):");
                println!(
                    "  [Environment]::SetEnvironmentVariable('Path', \"{d};\" + [Environment]::GetEnvironmentVariable('Path','User'), 'User')"
                );
            }
        }
        println!("Reopen terminal.");
    }

    #[cfg(target_os = "macos")]
    {
        let d = install_dir.display();
        println!("zsh:");
        println!("  echo 'export PATH=\"{d}:$PATH\"' >> ~/.zshrc");
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let d = install_dir.display();
        println!("bash:");
        println!("  echo 'export PATH=\"{d}:$PATH\"' >> ~/.bashrc");
    }
}

fn check_and_handle_existing_jx_in_path() -> Result<(), String> {
    let found = find_existing_jx_in_path();
    if found.is_empty() {
        return Ok(());
    }

    eprintln!("Warning: detected existing `jx` in PATH:");
    for p in &found {
        eprintln!("  {}", p.display());
    }

    loop {
        print!("Delete existing `jx` and continue? [y/n]: ");
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush stdout: {e}"))?;
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .map_err(|e| format!("Failed to read input: {e}"))?;
        let v = line.trim();
        if v.eq_ignore_ascii_case("n") {
            return Err("Installer cancelled.".to_string());
        }
        if v.eq_ignore_ascii_case("y") {
            let mut failed: Vec<String> = Vec::new();
            let mut cleaned_dirs: BTreeSet<PathBuf> = BTreeSet::new();
            for p in &found {
                if let Err(e) = std::fs::remove_file(p) {
                    failed.push(format!("{} ({e})", p.display()));
                }
                if let Some(parent) = p.parent() {
                    let parent_buf = parent.to_path_buf();
                    if cleaned_dirs.insert(parent_buf.clone()) {
                        cleanup_generated_artifacts_in_install_dir(&parent_buf, &mut failed);
                    }
                }
            }
            if !failed.is_empty() {
                return Err(format!(
                    "Failed to delete existing `jx` executable(s):\n{}",
                    failed.join("\n")
                ));
            }
            println!("Existing `jx` removed from PATH locations.");
            return Ok(());
        }
    }
}

fn cleanup_generated_artifacts_in_install_dir(install_dir: &Path, failed: &mut Vec<String>) {
    let mut remove_path = |p: PathBuf| {
        if !p.exists() {
            return;
        }
        let res = if p.is_dir() {
            std::fs::remove_dir_all(&p)
        } else {
            std::fs::remove_file(&p)
        };
        if let Err(e) = res {
            failed.push(format!("{} ({e})", p.display()));
        }
    };

    // Capture linked runtime path before removing .jx_home
    let linked_runtime = {
        let cfg = install_dir.join(RUNTIME_HOME_CONFIG);
        std::fs::read_to_string(&cfg)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(|s| expand_tilde(&s))
    };

    // Cleanup current and legacy runtime/artifact layouts under jx install directory.
    for name in [
        ".janusx",
        "venv",
        RUNTIME_HOME_CONFIG,
        UPDATE_TIME_MARKER,
        COMMIT_MARKER,
        WARMUP_MARKER,
    ] {
        remove_path(install_dir.join(name));
    }

    // Also cleanup runtime linked by .jx_home (cross-platform default/override locations).
    if let Some(runtime_home) = linked_runtime {
        // Guard against accidentally deleting install dir/root-like paths.
        if runtime_home != install_dir
            && runtime_home.parent().is_some()
            && runtime_home.as_os_str() != "/"
            && runtime_home.as_os_str() != "\\"
        {
            remove_path(runtime_home);
        }
    }
}

fn find_existing_jx_in_path() -> Vec<PathBuf> {
    let Some(path_var) = env::var_os("PATH") else {
        return Vec::new();
    };
    let mut out: BTreeSet<PathBuf> = BTreeSet::new();
    let names: &[&str] = if cfg!(windows) {
        &["jx.exe", "jx.cmd", "jx.bat", "jx"]
    } else {
        &["jx"]
    };
    for dir in env::split_paths(&path_var) {
        for name in names {
            let p = dir.join(name);
            if p.exists() && p.is_file() {
                out.insert(p);
            }
        }
    }
    out.into_iter().collect()
}

fn is_dir_in_path(dir: &Path) -> bool {
    let Some(path_var) = env::var_os("PATH") else {
        return false;
    };
    let want = canonical_or_self(dir);
    env::split_paths(&path_var).any(|p| path_eq(&canonical_or_self(&p), &want))
}

fn canonical_or_self(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(windows)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum WindowsShell {
    PowerShell,
    Cmd,
    Unknown,
}

#[cfg(windows)]
fn detect_windows_shell() -> WindowsShell {
    if env::var_os("POWERSHELL_DISTRIBUTION_CHANNEL").is_some()
        || env::var_os("PSExecutionPolicyPreference").is_some()
    {
        return WindowsShell::PowerShell;
    }
    if env::var_os("PROMPT").is_some() {
        return WindowsShell::Cmd;
    }
    if let Some(comspec) = env::var_os("ComSpec") {
        let s = comspec.to_string_lossy().to_ascii_lowercase();
        if s.contains("pwsh") || s.contains("powershell") {
            return WindowsShell::PowerShell;
        }
        if s.ends_with("cmd.exe") {
            return WindowsShell::Cmd;
        }
    }
    WindowsShell::Unknown
}

#[cfg(windows)]
fn path_eq(a: &Path, b: &Path) -> bool {
    a.to_string_lossy()
        .eq_ignore_ascii_case(&b.to_string_lossy())
}

#[cfg(not(windows))]
fn path_eq(a: &Path, b: &Path) -> bool {
    a == b
}

fn default_runtime_home() -> Result<PathBuf, String> {
    #[cfg(windows)]
    {
        if let Some(v) = env::var_os("LOCALAPPDATA") {
            return Ok(PathBuf::from(v).join("JanusX").join(".janusx"));
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

fn prompt_runtime_home() -> Result<(PathBuf, PathBuf), String> {
    let default_home = default_runtime_home()?;
    loop {
        let default_home_s = default_home.display().to_string();
        print!(
            "JanusX runtime home builds in {} [y/n/path]: ",
            default_home_s
        );
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush stdout: {e}"))?;
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .map_err(|e| format!("Failed to read input: {e}"))?;
        let v = line.trim();
        let runtime_home = if v.is_empty() || v.eq_ignore_ascii_case("y") {
            default_home.clone()
        } else if v.eq_ignore_ascii_case("n") {
            return Err("Installer cancelled.".to_string());
        } else {
            let p = expand_tilde(v);
            if !p.exists() || !p.is_dir() {
                eprintln!(
                    "Invalid path: {} (must already exist). Please choose [y/n/path] again.",
                    p.display()
                );
                continue;
            }
            if p.file_name().and_then(|x| x.to_str()) == Some(".janusx") {
                p
            } else {
                p.join(".janusx")
            }
        };
        let install_dir = runtime_home
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| runtime_home.clone());
        let runtime_parent = runtime_home
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| runtime_home.clone());
        if !runtime_parent.exists() {
            if let Err(e) = std::fs::create_dir_all(&runtime_parent) {
                eprintln!(
                    "Failed to create runtime parent directory {}: {}",
                    runtime_parent.display(),
                    e
                );
                continue;
            }
        }
        if !runtime_parent.is_dir() {
            eprintln!(
                "Runtime parent is not a directory: {}",
                runtime_parent.display()
            );
            continue;
        }
        if !is_dir_writable(&runtime_parent) {
            eprintln!(
                "Runtime parent directory is not writable: {}",
                runtime_parent.display()
            );
            continue;
        }
        if !install_dir.exists() {
            if let Err(e) = std::fs::create_dir_all(&install_dir) {
                eprintln!(
                    "Failed to create install directory {}: {}",
                    install_dir.display(),
                    e
                );
                continue;
            }
        }
        if !install_dir.is_dir() {
            eprintln!("Install path is not a directory: {}", install_dir.display());
            continue;
        }
        if !is_dir_writable(&install_dir) {
            eprintln!(
                "Install directory is not writable: {}",
                install_dir.display()
            );
            continue;
        }
        println!("Binary `jx` builds in {}", install_dir.display());
        return Ok((runtime_home, install_dir));
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
    let jx_bin = env::current_exe().ok();

    match &opts.source {
        UpdateSource::Latest => {
            ensure_git_available()?;
            let repo_url = repo_url_from_spec(GITHUB_SPEC);
            let remote_head = remote_head_commit(&repo_url);
            let local_commit = read_commit_marker(&home);
            if let (Some(local), Some(remote)) = (local_commit.as_deref(), remote_head.as_deref()) {
                if local == remote {
                    print_success_line("Already at latest GitHub commit.");
                    warm_up_after_update(jx_bin.as_deref(), &home)?;
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
                        warm_up_after_update(jx_bin.as_deref(), &home)?;
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
                println!("Updating from GitHub ...");
            }
            match pip_install_update(
                &python,
                &home,
                GITHUB_SPEC,
                gh_force_reinstall,
                opts.verbose,
                "Updating from GitHub ...",
            ) {
                Ok(_) => {
                    if let Some(remote) = remote_head.as_deref() {
                        write_commit_marker(&home, remote);
                    }
                    warm_up_after_update(jx_bin.as_deref(), &home)?;
                    return Ok(0);
                }
                Err(err_primary) => {
                    eprintln!("Direct GitHub update failed, retrying with proxy...");
                    if opts.verbose {
                        eprintln!("Reason: {err_primary}");
                    }
                    let _ = pip_install_update(
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
                    warm_up_after_update(jx_bin.as_deref(), &home)?;
                    return Ok(0);
                }
            }
        }
        UpdateSource::Local(path) => {
            if opts.verbose {
                println!("Updating from local source: {path}");
            }
            let _ = pip_install_update(
                &python,
                &home,
                path,
                opts.force_reinstall,
                opts.verbose,
                "Updating from local source...",
            )?;
            warm_up_after_update(jx_bin.as_deref(), &home)?;
            return Ok(0);
        }
        UpdateSource::Pypi => {}
    }

    if opts.verbose {
        println!("Updating from PyPI...");
    }
    let elapsed = pip_install_update(
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
    }
    warm_up_after_update(jx_bin.as_deref(), &home)?;
    Ok(0)
}

fn run_uninstall(args: &[String]) -> Result<i32, String> {
    let mut yes = false;
    for token in args {
        match token.as_str() {
            "-y" | "--yes" => yes = true,
            "-h" | "--help" | "help" => {
                println!("Uninstall usage:\n  jx --uninstall [--yes]");
                return Ok(0);
            }
            other => {
                return Err(format!(
                    "Unknown uninstall option: {other}\nUninstall usage:\n  jx --uninstall [--yes]"
                ));
            }
        }
    }

    if !yes {
        print!("This will remove JanusX launcher/runtime files. Continue? [y/n]: ");
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush stdout: {e}"))?;
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .map_err(|e| format!("Failed to read input: {e}"))?;
        if !line.trim().eq_ignore_ascii_case("y") {
            println!("Uninstall cancelled.");
            return Ok(0);
        }
    }

    let exe = env::current_exe().map_err(|e| format!("Failed to locate current jx binary: {e}"))?;
    let install_dir = exe
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Failed to resolve jx install directory.".to_string())?;

    let mut failed: Vec<String> = Vec::new();
    cleanup_generated_artifacts_in_install_dir(&install_dir, &mut failed);

    if let Ok(home) = runtime_home() {
        cleanup_runtime_home(&home, &mut failed);
    }
    if let Ok(def_home) = default_runtime_home() {
        cleanup_runtime_home(&def_home, &mut failed);
    }

    let mut remove_path = |p: &Path| {
        if !p.exists() {
            return;
        }
        if let Err(e) = std::fs::remove_file(p) {
            failed.push(format!("{} ({e})", p.display()));
        }
    };

    remove_path(&install_dir.join("jx"));
    remove_path(&install_dir.join("jx.cmd"));
    remove_path(&install_dir.join("jx.bat"));
    remove_path(&install_dir.join("jx-script.py"));

    #[cfg(not(windows))]
    {
        remove_path(&install_dir.join("jx.exe"));
        remove_path(&exe);
    }
    #[cfg(windows)]
    {
        let jx_exe = install_dir.join("jx.exe");
        if jx_exe.exists() {
            if let Err(e) = std::fs::remove_file(&jx_exe) {
                let escaped = jx_exe.to_string_lossy().replace('"', "\"\"");
                let cmdline =
                    format!("ping 127.0.0.1 -n 2 >NUL & del /f /q \"{escaped}\" >NUL 2>NUL");
                let spawned = Command::new("cmd")
                    .arg("/C")
                    .arg(cmdline)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn();
                if spawned.is_err() {
                    failed.push(format!("{} ({e})", jx_exe.display()));
                }
            }
        }
    }

    if !failed.is_empty() {
        return Err(format!(
            "JanusX uninstall finished with errors:\n{}",
            failed.join("\n")
        ));
    }

    print_success_line("JanusX uninstall completed.");
    println!("Genotype caches were not removed.");
    Ok(0)
}

fn cleanup_runtime_home(runtime_home: &Path, failed: &mut Vec<String>) {
    if !runtime_home.exists() {
        return;
    }
    let marker_hits = runtime_home.join("venv").exists()
        || runtime_home.join(UPDATE_TIME_MARKER).exists()
        || runtime_home.join(COMMIT_MARKER).exists()
        || runtime_home.join(WARMUP_MARKER).exists();
    if marker_hits {
        if let Err(e) = std::fs::remove_dir_all(runtime_home) {
            failed.push(format!("{} ({e})", runtime_home.display()));
        }
        return;
    }

    // Fallback: only prune known runtime artifacts if directory doesn't look fully managed by JanusX.
    for name in ["venv", UPDATE_TIME_MARKER, COMMIT_MARKER, WARMUP_MARKER] {
        let p = runtime_home.join(name);
        if !p.exists() {
            continue;
        }
        let res = if p.is_dir() {
            std::fs::remove_dir_all(&p)
        } else {
            std::fs::remove_file(&p)
        };
        if let Err(e) = res {
            failed.push(format!("{} ({e})", p.display()));
        }
    }
}

fn pip_install_update(
    python: &Path,
    runtime_home: &Path,
    spec: &str,
    force_reinstall: bool,
    verbose: bool,
    desc: &str,
) -> Result<Duration, String> {
    if verbose {
        return pip_install(python, runtime_home, spec, force_reinstall, true, desc);
    }
    pip_install_tail(python, runtime_home, spec, force_reinstall, desc, 10)
}

fn warm_up_after_update(jx_bin: Option<&Path>, runtime_home: &Path) -> Result<(), String> {
    let Some(jx_path) = jx_bin else {
        return Ok(());
    };
    warm_up_jx(jx_path, runtime_home)
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
        let _ = std::fs::remove_file(warmup_marker_path(&home));
        if verbose_bootstrap {
            println!("Bootstrapping JanusX from PyPI...");
        }
        let _ = pip_install_tail(
            &python,
            &home,
            PYPI_SPEC,
            false,
            "Building runtime from PyPI ...",
            10,
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
        ensure_pip_in_venv(&py)?;
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
    ensure_pip_in_venv(&py)?;
    Ok(py)
}

fn has_pip(python: &Path) -> bool {
    Command::new(python)
        .arg("-m")
        .arg("pip")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn ensure_pip_in_venv(python: &Path) -> Result<(), String> {
    if has_pip(python) {
        return Ok(());
    }

    let out = Command::new(python)
        .arg("-m")
        .arg("ensurepip")
        .arg("--upgrade")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to bootstrap pip with ensurepip: {e}"))?;

    if !out.status.success() || !has_pip(python) {
        let mut msg = String::new();
        msg.push_str(&String::from_utf8_lossy(&out.stdout));
        msg.push_str(&String::from_utf8_lossy(&out.stderr));
        let details = msg.trim();
        let details = if details.is_empty() {
            "(no output)".to_string()
        } else {
            details.to_string()
        };
        return Err(format!(
            "Runtime venv Python has no pip, and ensurepip bootstrap failed.\n{}\n{}",
            details,
            pip_bootstrap_hint()
        ));
    }

    Ok(())
}

fn should_rebuild_runtime(home: &Path) -> Result<bool, String> {
    let skip = env::var(SKIP_RUNTIME_REBUILD_ENV)
        .ok()
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    if skip {
        return Ok(false);
    }
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
    if supports_color() {
        println!("\x1b[32m✔︎ {}\x1b[0m", msg);
    } else {
        println!("✔︎ {}", msg);
    }
}

fn supports_color() -> bool {
    io::stdout().is_terminal()
}

fn style_green(text: &str) -> String {
    if supports_color() {
        format!("\x1b[32m{text}\x1b[0m")
    } else {
        text.to_string()
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
    let mut cmd = build_pip_install_cmd(python, spec, force_reinstall);

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

fn build_pip_install_cmd(python: &Path, spec: &str, force_reinstall: bool) -> Command {
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
    cmd
}

fn pip_install_tail(
    python: &Path,
    runtime_home: &Path,
    spec: &str,
    force_reinstall: bool,
    desc: &str,
    max_lines: usize,
) -> Result<Duration, String> {
    let is_tty = io::stdout().is_terminal();
    let mut cmd = build_pip_install_cmd(python, spec, force_reinstall);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let start = Instant::now();
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to run pip install for {spec}: {e}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to capture pip stdout".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Failed to capture pip stderr".to_string())?;

    let (tx, rx) = mpsc::channel::<(bool, String)>();
    let tx_out = tx.clone();
    let tx_err = tx.clone();
    let out_handle = thread::spawn(move || {
        let reader = io::BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            if tx_out.send((false, line)).is_err() {
                break;
            }
        }
    });
    let err_handle = thread::spawn(move || {
        let reader = io::BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            if tx_err.send((true, line)).is_err() {
                break;
            }
        }
    });
    drop(tx);

    let mut rendered = false;
    let mut out_text = String::new();
    let mut err_text = String::new();
    let mut tail: VecDeque<String> = VecDeque::with_capacity(max_lines.max(1));
    let mut channel_open = true;
    let mut exit_status: Option<ExitStatus> = None;
    let mut last_render = Instant::now();
    let mut last_line_seen: Option<String> = None;

    while channel_open {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok((is_err, line)) => {
                let is_duplicate = last_line_seen.as_deref() == Some(line.as_str());
                last_line_seen = Some(line.clone());
                if is_err {
                    err_text.push_str(&line);
                    err_text.push('\n');
                } else {
                    out_text.push_str(&line);
                    out_text.push('\n');
                }
                if !is_duplicate {
                    if max_lines > 0 {
                        if tail.len() == max_lines {
                            tail.pop_front();
                        }
                        tail.push_back(line);
                    }
                }
                if is_tty {
                    render_tail_block(desc, "", start.elapsed(), &tail, max_lines, rendered)?;
                    rendered = true;
                    last_render = Instant::now();
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if exit_status.is_none() {
                    exit_status = child
                        .try_wait()
                        .map_err(|e| format!("Failed to poll pip install status: {e}"))?;
                }
                if is_tty && last_render.elapsed() >= Duration::from_secs(2) {
                    render_tail_block(desc, "", start.elapsed(), &tail, max_lines, rendered)?;
                    rendered = true;
                    last_render = Instant::now();
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                channel_open = false;
            }
        }
    }

    let _ = out_handle.join();
    let _ = err_handle.join();

    let status = if let Some(s) = exit_status {
        s
    } else {
        child
            .wait()
            .map_err(|e| format!("Failed to wait for pip install process: {e}"))?
    };

    if is_tty {
        let final_symbol = if status.success() { "✔︎" } else { "✘" };
        render_tail_block(
            desc,
            final_symbol,
            start.elapsed(),
            &tail,
            max_lines,
            rendered,
        )?;
    }

    if status.success() {
        remove_conflicting_jx_entrypoints(python);
        write_python_core_update_marker(runtime_home, python);
        return Ok(start.elapsed());
    }

    let mut msg = String::new();
    msg.push_str(&out_text);
    msg.push_str(&err_text);
    Err(format!(
        "pip install failed (exit={}) for {spec}\n{}",
        exit_code(status),
        msg.trim()
    ))
}

fn render_tail_block(
    desc: &str,
    symbol: &str,
    elapsed: Duration,
    tail: &VecDeque<String>,
    max_lines: usize,
    rendered_before: bool,
) -> Result<(), String> {
    if rendered_before {
        print!("\x1b[{}A", max_lines.saturating_add(1));
    }
    let width = terminal_line_width();
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    if symbol.is_empty() {
        let idx = ((elapsed.as_millis() / 120) as usize) % frames.len();
        let title = truncate_plain_line(
            &format!("{} {}[{}]", frames[idx], desc, format_elapsed(elapsed)),
            width,
        );
        print!("\x1b[2K\r{}\n", style_green(&title));
    } else {
        let title = truncate_plain_line(
            &format!("{symbol} {desc}[{}]", format_elapsed(elapsed)),
            width,
        );
        print!("\x1b[2K\r{}\n", style_green(&title));
    }
    let mut shown = 0usize;
    for line in tail {
        let trimmed = truncate_plain_line(line, width);
        print!("\x1b[2K\r\x1b[2m{}\x1b[0m\n", trimmed);
        shown += 1;
    }
    while shown < max_lines {
        print!("\x1b[2K\r\n");
        shown += 1;
    }
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush pip progress output: {e}"))?;
    Ok(())
}

fn terminal_line_width() -> usize {
    let cols = env::var("COLUMNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(80);
    // Keep a conservative width to avoid visual wrap; wrapped lines break block redraw.
    cols.saturating_sub(4).clamp(40, 68)
}

fn truncate_plain_line(s: &str, max_chars: usize) -> String {
    if max_chars <= 3 {
        return "...".to_string();
    }
    let mut out = String::new();
    let mut n = 0usize;
    for ch in s.chars() {
        if n + 1 >= max_chars {
            out.push_str("...");
            return out;
        }
        out.push(ch);
        n += 1;
    }
    out
}

fn print_dim_block(text: &str) {
    for line in text.lines() {
        if supports_color() {
            println!("\x1b[2m{}\x1b[0m", line);
        } else {
            println!("{line}");
        }
    }
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

fn pip_bootstrap_hint() -> String {
    if cfg!(windows) {
        return "Try recreating runtime venv with a full Python distribution that includes ensurepip.\n\
Suggested fix:\n\
  1) Remove runtime: %LOCALAPPDATA%\\JanusX\\venv\n\
  2) Re-run installer or `jx -h`"
            .to_string();
    }
    if cfg!(target_os = "macos") {
        return "Try recreating runtime venv with a Python that includes ensurepip.\n\
Suggested fix:\n\
  1) rm -rf ~/.janusx/venv\n\
  2) Re-run installer or `jx -h`\n\
If system Python lacks ensurepip, install python via Homebrew: `brew install python`."
            .to_string();
    }
    "Try recreating runtime venv with a Python that includes ensurepip.\n\
Suggested fix:\n\
  1) rm -rf ~/.janusx/venv\n\
  2) Re-run installer or `jx -h`"
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
