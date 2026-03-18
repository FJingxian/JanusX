use std::collections::{BTreeSet, VecDeque};
use std::env;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command, ExitStatus, Output, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};
#[cfg(target_os = "windows")]
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

mod dlc;
mod fastq2count;
mod fastq2vcf;
mod pipeline;

const PYPI_SPEC: &str = "janusx";
const GITHUB_SPEC_CN: &str = "git+https://gh-proxy.org/https://github.com/FJingxian/JanusX.git";
const GITHUB_SPEC_ORIGIN: &str = "git+https://github.com/FJingxian/JanusX.git";
const GITHUB_ARCHIVE_CN: &str =
    "https://gh-proxy.org/https://github.com/FJingxian/JanusX/archive/refs/heads/main.tar.gz";
const GITHUB_ARCHIVE_ORIGIN: &str =
    "https://github.com/FJingxian/JanusX/archive/refs/heads/main.tar.gz";
const UPDATE_TIME_MARKER: &str = ".python_core_updated_at";
const COMMIT_MARKER: &str = ".python_core_commit";
const LAUNCHER_CARGO_VERSION: &str = env!("CARGO_PKG_VERSION");
const VERSION_AUTHOR: &str = "Jingxian FU, Yazhouwan National Laboratory";
const VERSION_CONTACT: &str = "fujingxian@yzwlab.cn";
const RUNTIME_HOME_CONFIG: &str = ".jx_home";
const INSTALLER_RELAUNCH_ENV: &str = "JX_INSTALLER_RELAUNCHED";
const SKIP_RUNTIME_REBUILD_ENV: &str = "JX_SKIP_RUNTIME_REBUILD";
const SKIP_WARMUP_ENV: &str = "JX_SKIP_WARMUP";
const WARMUP_MARKER: &str = ".runtime_warmed";
const LAUNCHER_VERSION_MARKER: &str = ".launcher_version";
const GWAS_HISTORY_DB_FILE: &str = "janusx_tasks.db";
const KNOWN_MODULES: [&str; 18] = [
    "grm",
    "pca",
    "gwas",
    "postgwas",
    "gs",
    "reml",
    "garfield",
    "postgarfield",
    "postbsa",
    "fastq2count",
    "fastq2vcf",
    "adamixture",
    "hybrid",
    "gformat",
    "gmerge",
    "webui",
    "sim",
    "simulation",
];
const MODULE_LIST_ENTRIES: [(&str, &str); 18] = [
    ("grm", "Build genomic relationship matrix"),
    (
        "pca",
        "Principal component analysis for population structure",
    ),
    ("gwas", "Run genome-wide association analysis"),
    ("postgwas", "Post-process GWAS results and downstream plots"),
    ("gs", "Genomic prediction and model-based selection"),
    (
        "reml",
        "Estimate heritability/effect components by REML-BLUP",
    ),
    ("garfield", "Random-forest based marker-trait association"),
    ("postgarfield", "Summarize and visualize GARFIELD outputs"),
    ("postbsa", "Post-process and visualize BSA results"),
    (
        "fastq2count",
        "RNA-seq pipeline from FASTQ to gene count/FPKM/TPM",
    ),
    ("fastq2vcf", "Variant-calling pipeline from FASTQ to VCF"),
    ("adamixture", "ADAMIXTURE ancestry inference"),
    (
        "hybrid",
        "Build pairwise hybrid genotype matrix from parent lists",
    ),
    ("gformat", "Convert genotype files across plink/vcf/txt/npy"),
    ("gmerge", "Merge genotype/variant tables"),
    ("webui", "Start JanusX web UI (postgwas first)"),
    ("sim", "Quick simulation workflow"),
    (
        "simulation",
        "Extended simulation and benchmarking workflow",
    ),
];
const MIN_PYTHON_MAJOR: u32 = 3;
const MIN_PYTHON_MINOR: u32 = 9;
const RUSTUP_DIST_SERVER_CN: &str = "https://rsproxy.cn";
const RUSTUP_UPDATE_ROOT_CN: &str = "https://rsproxy.cn/rustup";
const RUSTUP_DIST_SERVER_ORIGIN: &str = "https://static.rust-lang.org";
const RUSTUP_UPDATE_ROOT_ORIGIN: &str = "https://static.rust-lang.org/rustup";
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
    Dlc,
    Pypi,
    Latest,
    Local(String),
}

#[derive(Clone, Debug)]
struct UpdateOptions {
    source: UpdateSource,
    verbose: bool,
    force_reinstall: bool,
    editable: bool,
}

#[derive(Clone, Debug)]
enum UpgradeSource {
    Latest,
    Local(String),
}

#[derive(Clone, Debug)]
struct UpgradeOptions {
    source: UpgradeSource,
    verbose: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum LauncherReplaceResult {
    ReplacedNow,
    #[cfg(windows)]
    ReplacedAfterExit,
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
        print_cli_help();
        return Ok(0);
    }
    if matches!(head, "-v" | "--version") {
        let home = runtime_home()?;
        let core_python = ready_core_python(&home);
        if core_python.is_some() && maybe_auto_warmup(&home)? {
            return Ok(0);
        }
        print_version(core_python.as_deref(), &home);
        return Ok(0);
    }
    if matches!(head, "-list" | "--list") {
        return run_list(&args[1..]);
    }

    if matches!(head, "-update" | "--update") {
        let opts = match parse_update_args(&args[1..]) {
            Ok(v) => v,
            Err(e) if e.is_empty() => return Ok(0),
            Err(e) => return Err(e),
        };
        return run_update(opts);
    }
    if matches!(head, "-upgrade" | "--upgrade") {
        let opts = match parse_upgrade_args(&args[1..]) {
            Ok(v) => v,
            Err(e) if e.is_empty() => return Ok(0),
            Err(e) => return Err(e),
        };
        return run_upgrade(opts);
    }
    if matches!(head, "-clean" | "--clean") {
        return run_clean(&args[1..]);
    }
    if matches!(head, "-uninstall" | "--uninstall") {
        return run_uninstall(&args[1..]);
    }
    if dlc::is_dlc_tool(head) {
        let home = runtime_home()?;
        return dlc::run_dlc_tool(head, &args[1..], &home);
    }
    if head == "fastq2count" {
        return fastq2count::run_fastq2count_module(&args[1..]);
    }
    if head == "fastq2vcf" {
        return fastq2vcf::run_fastq2vcf_module(&args[1..]);
    }
    if !is_known_module(head) {
        println!("{}", style_yellow(&format!("Unknown module: {head}")));
        print_cli_help();
        return Ok(1);
    }

    let home = runtime_home()?;
    let python = ensure_runtime(false)?;
    let _ = maybe_auto_warmup(&home)?;
    run_python_janusx(&python, &args)
}

fn is_known_module(name: &str) -> bool {
    KNOWN_MODULES.contains(&name)
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

    println!("JanusX Installer");
    println!("Guided setup: Runtime home -> jx location -> PATH\n");
    check_and_handle_existing_jx_in_path()?;

    let (runtime_home, install_dir) = prompt_runtime_home()?;
    env::set_var("JX_HOME", &runtime_home);
    println!("Runtime home: {}", runtime_home.display());

    ensure_runtime(false)?;

    let installed_jx = install_jx_binary(&install_dir)?;
    write_runtime_home_config_near_binary(&installed_jx, &runtime_home);
    write_launcher_version_marker_near_binary(&installed_jx);
    warm_up_jx(&installed_jx, &runtime_home)?;
    print_path_setup_hint(&installed_jx);
    Ok(0)
}

fn warm_up_jx(jx_bin: &Path, runtime_home: &Path) -> Result<(), String> {
    let warm_start = Instant::now();
    let warm_done = Arc::new(AtomicBool::new(false));
    let warm_done_c = Arc::clone(&warm_done);
    let warm_progress = if io::stdout().is_terminal() {
        Some(thread::spawn(move || {
            while !warm_done_c.load(Ordering::Relaxed) {
                let elapsed = warm_start.elapsed();
                let line = format!(
                    "{} Warming up ...[{}]",
                    spinner_frame_for_elapsed(elapsed),
                    format_elapsed_live(elapsed)
                );
                if supports_color() {
                    print!("\r{}", style_green(&line));
                } else {
                    print!("\r{}", line);
                }
                let _ = io::stdout().flush();
                thread::sleep(spinner_refresh_interval(elapsed));
            }
            print!("\r\x1b[2K");
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

    // Force-load Python core path via a real module entry.
    let module_help_status = Command::new(jx_bin)
        .arg("gwas")
        .arg("-h")
        .env("JX_HOME", runtime_home)
        .env(SKIP_RUNTIME_REBUILD_ENV, "1")
        .env(SKIP_WARMUP_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("Failed to run `{}` gwas -h: {e}", jx_bin.display()))?;
    if !module_help_status.success() {
        stop_warm_progress(warm_done, warm_progress);
        return Err(format!(
            "Warm-up command failed: {} gwas -h (exit={})",
            jx_bin.display(),
            exit_code(module_help_status)
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

    match ensure_install_dir_in_path_persistent(install_dir) {
        Ok(PathPersistStatus::Added) => {
            print_success_line("JanusX has been added to PATH.");
            println!("Reopen terminal.");
        }
        Ok(PathPersistStatus::AlreadyConfigured) => {
            println!("JanusX PATH is already configured, but current session may not be reloaded.");
            #[cfg(target_os = "macos")]
            println!("Run: source ~/.zshrc");
            #[cfg(all(unix, not(target_os = "macos")))]
            println!("Run: source ~/.bashrc");
            #[cfg(target_os = "windows")]
            println!("Reopen terminal.");
        }
        Err(e) => {
            eprintln!("Warning: failed to auto-add JanusX to PATH: {e}");
            println!("Add JanusX to PATH:");

            #[cfg(target_os = "windows")]
            {
                let d = install_dir.display();
                println!("setx PATH \"{d};%PATH%\"");
            }

            #[cfg(target_os = "macos")]
            {
                let d = install_dir.display();
                println!("echo 'export PATH=\"{d}:$PATH\"' >> ~/.zshrc");
            }

            #[cfg(all(unix, not(target_os = "macos")))]
            {
                let d = install_dir.display();
                println!("echo 'export PATH=\"{d}:$PATH\"' >> ~/.bashrc");
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum PathPersistStatus {
    Added,
    AlreadyConfigured,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum PathRemoveStatus {
    Removed,
    NotFound,
}

fn ensure_install_dir_in_path_persistent(install_dir: &Path) -> Result<PathPersistStatus, String> {
    #[cfg(target_os = "windows")]
    {
        return ensure_install_dir_in_path_persistent_windows(install_dir);
    }
    #[cfg(target_os = "macos")]
    {
        return ensure_install_dir_in_path_persistent_unix(install_dir, ".zshrc");
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        return ensure_install_dir_in_path_persistent_unix(install_dir, ".bashrc");
    }
    #[allow(unreachable_code)]
    Err("Unsupported platform for PATH auto-setup.".to_string())
}

fn remove_install_dir_from_path_persistent(install_dir: &Path) -> Result<PathRemoveStatus, String> {
    #[cfg(target_os = "windows")]
    {
        return remove_install_dir_from_path_persistent_windows(install_dir);
    }
    #[cfg(target_os = "macos")]
    {
        return remove_install_dir_from_path_persistent_unix(install_dir, ".zshrc");
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        return remove_install_dir_from_path_persistent_unix(install_dir, ".bashrc");
    }
    #[allow(unreachable_code)]
    Err("Unsupported platform for PATH removal.".to_string())
}

#[cfg(target_os = "windows")]
fn ensure_install_dir_in_path_persistent_windows(
    install_dir: &Path,
) -> Result<PathPersistStatus, String> {
    let target = canonical_or_self(install_dir);
    let target_s = target.to_string_lossy().to_string();
    let script = r#"
$target = "$env:JX_TARGET_DIR".Trim()
if ([string]::IsNullOrWhiteSpace($target)) { exit 12 }
$target = [System.IO.Path]::GetFullPath($target)
$old = [Environment]::GetEnvironmentVariable('Path','User')
$parts = @()
if (-not [string]::IsNullOrWhiteSpace($old)) {
  $parts = @(
    $old.Split(';') |
      ForEach-Object { $_.Trim() } |
      Where-Object { $_ -ne '' }
  )
}
$parts = @(
  $parts | Where-Object {
    -not [System.StringComparer]::OrdinalIgnoreCase.Equals($_, $target)
  }
)
$newParts = @($target) + $parts
$new = [string]::Join(';', $newParts)
$oldNorm = [string]::Join(';', @($parts))
if (-not [string]::IsNullOrWhiteSpace($oldNorm)) {
  $oldWithTarget = [string]::Join(';', @($target) + $parts)
  if ([System.StringComparer]::OrdinalIgnoreCase.Equals($old, $oldWithTarget)) {
    exit 11
  }
}
[Environment]::SetEnvironmentVariable('Path', $new, 'User')
exit 10
"#;

    for shell in ["powershell", "pwsh"] {
        if !command_ok(
            shell,
            &["-NoProfile", "-Command", "$PSVersionTable.PSVersion"],
        ) {
            continue;
        }
        let status = Command::new(shell)
            .arg("-NoProfile")
            .arg("-Command")
            .arg(script)
            .env("JX_TARGET_DIR", &target_s)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| format!("Failed to run {shell} for PATH setup: {e}"))?;
        let code = exit_code(status);
        if code == 10 {
            return Ok(PathPersistStatus::Added);
        }
        if code == 11 {
            return Ok(PathPersistStatus::AlreadyConfigured);
        }
        if code == 12 {
            return Err("Resolved install directory is empty; cannot update PATH.".to_string());
        }
    }

    let status = Command::new("cmd")
        .arg("/C")
        .arg("setx")
        .arg("PATH")
        .arg(format!("{};%PATH%", target_s))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("Failed to run setx for PATH setup: {e}"))?;
    if status.success() {
        return Ok(PathPersistStatus::Added);
    }
    Err(format!("setx failed with exit={}", exit_code(status)))
}

#[cfg(target_os = "windows")]
fn remove_install_dir_from_path_persistent_windows(
    install_dir: &Path,
) -> Result<PathRemoveStatus, String> {
    let target = canonical_or_self(install_dir);
    let target_s = target.to_string_lossy().to_string();
    let script = r#"
$target = "$env:JX_TARGET_DIR".Trim()
if ([string]::IsNullOrWhiteSpace($target)) { exit 21 }
$target = [System.IO.Path]::GetFullPath($target)
$old = [Environment]::GetEnvironmentVariable('Path','User')
if ([string]::IsNullOrWhiteSpace($old)) { exit 21 }
$parts = @()
$removed = $false
foreach ($p in $old.Split(';')) {
  $t = $p.Trim()
  if ($t -eq '') { continue }
  if ([System.StringComparer]::OrdinalIgnoreCase.Equals($t, $target)) {
    $removed = $true
    continue
  }
  $parts += $t
}
if (-not $removed) { exit 21 }
$new = [string]::Join(';', $parts)
[Environment]::SetEnvironmentVariable('Path', $new, 'User')
exit 20
"#;

    for shell in ["powershell", "pwsh"] {
        if !command_ok(
            shell,
            &["-NoProfile", "-Command", "$PSVersionTable.PSVersion"],
        ) {
            continue;
        }
        let status = Command::new(shell)
            .arg("-NoProfile")
            .arg("-Command")
            .arg(script)
            .env("JX_TARGET_DIR", &target_s)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| format!("Failed to run {shell} for PATH removal: {e}"))?;
        let code = exit_code(status);
        if code == 20 {
            return Ok(PathRemoveStatus::Removed);
        }
        if code == 21 {
            return Ok(PathRemoveStatus::NotFound);
        }
    }
    Err("No PowerShell runtime available to remove PATH entry.".to_string())
}

#[cfg(any(target_os = "macos", all(unix, not(target_os = "macos"))))]
fn ensure_install_dir_in_path_persistent_unix(
    install_dir: &Path,
    rc_name: &str,
) -> Result<PathPersistStatus, String> {
    let home = env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| "HOME is not set; cannot update shell rc file.".to_string())?;
    let rc = home.join(rc_name);
    let d = canonical_or_self(install_dir).to_string_lossy().to_string();
    let line = format!("export PATH=\"{}:$PATH\"", d.replace('"', "\\\""));

    let mut existing = String::new();
    if rc.exists() {
        existing = std::fs::read_to_string(&rc)
            .map_err(|e| format!("Failed to read {}: {e}", rc.display()))?;
    }
    if existing.lines().any(|l| l.trim() == line) {
        return Ok(PathPersistStatus::AlreadyConfigured);
    }

    let mut out = existing;
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&line);
    out.push('\n');
    std::fs::write(&rc, out).map_err(|e| format!("Failed to write {}: {e}", rc.display()))?;
    Ok(PathPersistStatus::Added)
}

#[cfg(any(target_os = "macos", all(unix, not(target_os = "macos"))))]
fn remove_install_dir_from_path_persistent_unix(
    install_dir: &Path,
    rc_name: &str,
) -> Result<PathRemoveStatus, String> {
    let home = env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| "HOME is not set; cannot update shell rc file.".to_string())?;
    let rc = home.join(rc_name);
    if !rc.exists() {
        return Ok(PathRemoveStatus::NotFound);
    }

    let d = canonical_or_self(install_dir).to_string_lossy().to_string();
    let line = format!("export PATH=\"{}:$PATH\"", d.replace('"', "\\\""));
    let text = std::fs::read_to_string(&rc)
        .map_err(|e| format!("Failed to read {}: {e}", rc.display()))?;

    let kept: Vec<&str> = text.lines().filter(|l| l.trim() != line).collect();
    if kept.len() == text.lines().count() {
        return Ok(PathRemoveStatus::NotFound);
    }

    let mut out = kept.join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    std::fs::write(&rc, out).map_err(|e| format!("Failed to write {}: {e}", rc.display()))?;
    Ok(PathRemoveStatus::Removed)
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
        LAUNCHER_VERSION_MARKER,
        UPDATE_TIME_MARKER,
        COMMIT_MARKER,
        WARMUP_MARKER,
    ] {
        remove_path(install_dir.join(name));
    }

    // Also cleanup runtime linked by .jx_home (cross-platform default/override locations).
    if let Some(runtime_home) = linked_runtime {
        if !is_managed_runtime_home_path(&runtime_home) {
            failed.push(format!(
                "Refusing to remove linked runtime outside JanusX/.janusx: {}",
                runtime_home.display()
            ));
        } else {
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
    let p = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    #[cfg(windows)]
    {
        return de_verbatim_windows_path(p);
    }
    #[cfg(not(windows))]
    {
        p
    }
}

#[cfg(windows)]
fn de_verbatim_windows_path(path: PathBuf) -> PathBuf {
    let s = path.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{}", rest));
    }
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        return PathBuf::from(rest);
    }
    path
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
            return Ok(PathBuf::from(v).join("JanusX").join(".janusx"));
        }
    }
    env::current_dir()
        .map(|p| p.join("JanusX").join(".janusx"))
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
    let default_install = default_home
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_home.clone());
    loop {
        let default_install_s = default_install.display().to_string();
        print!(
            "JanusX will be installed in {} [y/n/path]: ",
            default_install_s
        );
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush stdout: {e}"))?;
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .map_err(|e| format!("Failed to read input: {e}"))?;
        let v = line.trim();
        let (runtime_home, install_dir) = if v.is_empty() || v.eq_ignore_ascii_case("y") {
            let home = default_home.clone();
            let install = home
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| home.clone());
            (home, install)
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
            let install = if p.file_name().and_then(|x| x.to_str()) == Some("JanusX") {
                p
            } else {
                p.join("JanusX")
            };
            (install.join(".janusx"), install)
        };

        if install_dir.exists() {
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
            if !is_dir_empty(&install_dir)? {
                print!(
                    "{} already exists and is not empty. Delete and install? [y/n]: ",
                    install_dir.display()
                );
                io::stdout()
                    .flush()
                    .map_err(|e| format!("Failed to flush stdout: {e}"))?;
                let mut confirm = String::new();
                io::stdin()
                    .read_line(&mut confirm)
                    .map_err(|e| format!("Failed to read input: {e}"))?;
                if !confirm.trim().eq_ignore_ascii_case("y") {
                    continue;
                }
                std::fs::remove_dir_all(&install_dir).map_err(|e| {
                    format!(
                        "Failed to remove existing install directory {}: {e}",
                        install_dir.display()
                    )
                })?;
                std::fs::create_dir_all(&install_dir).map_err(|e| {
                    format!(
                        "Failed to recreate install directory {}: {e}",
                        install_dir.display()
                    )
                })?;
            }
        } else {
            std::fs::create_dir_all(&install_dir).map_err(|e| {
                format!(
                    "Failed to create install directory {}: {e}",
                    install_dir.display()
                )
            })?;
        }

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
        let install_dir_abs = canonical_or_self(&install_dir);
        let runtime_home_abs = install_dir_abs.join(".janusx");
        println!("Binary `jx` builds in {}", install_dir_abs.display());
        return Ok((runtime_home_abs, install_dir_abs));
    }
}

fn is_dir_empty(dir: &Path) -> Result<bool, String> {
    let mut it =
        std::fs::read_dir(dir).map_err(|e| format!("Failed to read {}: {e}", dir.display()))?;
    Ok(it.next().is_none())
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

fn write_launcher_version_marker_near_binary(binary_path: &Path) {
    let Some(version) = detect_expected_janusx_version(None) else {
        return;
    };
    let Some(parent) = binary_path.parent() else {
        return;
    };
    let marker = parent.join(LAUNCHER_VERSION_MARKER);
    let _ = std::fs::write(marker, format!("{version}\n"));
}

fn resolve_pypi_spec(runtime_home: &Path) -> String {
    if let Some(v) = detect_expected_janusx_version(Some(runtime_home)) {
        return format!("{PYPI_SPEC}=={v}");
    }
    PYPI_SPEC.to_string()
}

fn detect_expected_janusx_version(runtime_home: Option<&Path>) -> Option<String> {
    if let Some(v) = option_env!("JANUSX_EXPECTED_VERSION").and_then(normalize_version_token) {
        return Some(v);
    }
    if let Ok(v) = env::var("JX_EXPECTED_VERSION") {
        if let Some(s) = normalize_version_token(&v) {
            return Some(s);
        }
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            let marker = parent.join(LAUNCHER_VERSION_MARKER);
            if let Ok(text) = std::fs::read_to_string(marker) {
                if let Some(s) = normalize_version_token(&text) {
                    return Some(s);
                }
            }
        }
        if let Some(name) = exe.file_name().and_then(|x| x.to_str()) {
            if let Some(s) = extract_version_from_filename(name) {
                return Some(s);
            }
        }
    }
    if let Some(home) = runtime_home {
        if let Some(parent) = home.parent() {
            let marker = parent.join(LAUNCHER_VERSION_MARKER);
            if let Ok(text) = std::fs::read_to_string(marker) {
                if let Some(s) = normalize_version_token(&text) {
                    return Some(s);
                }
            }
        }
    }
    None
}

fn extract_version_from_filename(name: &str) -> Option<String> {
    let chars: Vec<char> = name.chars().collect();
    let n = chars.len();
    for i in 0..n {
        if chars[i] != 'v' && chars[i] != 'V' {
            continue;
        }
        if i + 1 >= n || !chars[i + 1].is_ascii_digit() {
            continue;
        }
        let mut j = i + 1;
        while j < n && (chars[j].is_ascii_digit() || chars[j] == '.') {
            j += 1;
        }
        let cand: String = chars[i + 1..j].iter().collect();
        if let Some(v) = normalize_version_token(&cand) {
            return Some(v);
        }
    }
    None
}

fn normalize_version_token(raw: &str) -> Option<String> {
    let t = raw.trim().trim_start_matches('v').trim_start_matches('V');
    if t.is_empty() {
        return None;
    }
    if !t.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return None;
    }
    let parts: Vec<&str> = t.split('.').collect();
    if parts.len() < 2 || parts.iter().any(|x| x.is_empty()) {
        return None;
    }
    Some(t.to_string())
}

fn parse_upgrade_args(args: &[String]) -> Result<UpgradeOptions, String> {
    let mut latest = false;
    let mut local_source: Option<String> = None;
    let mut verbose = false;

    for token in args {
        match token.as_str() {
            "latest" | "-latest" | "--latest" => latest = true,
            "-verbose" | "--verbose" => verbose = true,
            "-h" | "--help" | "help" => {
                println!(
                    "Upgrade usage:\n  jx -upgrade\n  jx -upgrade latest [-verbose]\n  jx -upgrade <local_path> [-verbose]\nNote: `jx -upgrade` upgrades launcher first, then runs core update logic equivalent to `jx -update latest`."
                );
                return Err(String::new());
            }
            other => {
                if Path::new(other).exists() {
                    if local_source.is_some() {
                        return Err(
                            "Only one local source path is allowed for `-upgrade`.".to_string()
                        );
                    }
                    local_source = Some(other.to_string());
                } else {
                    return Err(format!(
                        "Unknown upgrade option: {other}\nUpgrade usage:\n  jx -upgrade\n  jx -upgrade latest [-verbose]\n  jx -upgrade <local_path> [-verbose]\nNote: `jx -upgrade` upgrades launcher first, then runs core update logic equivalent to `jx -update latest`."
                    ));
                }
            }
        }
    }

    if latest && local_source.is_some() {
        return Err("`latest` and local path cannot be used together.".to_string());
    }

    let source = if let Some(path) = local_source {
        UpgradeSource::Local(path)
    } else {
        UpgradeSource::Latest
    };

    Ok(UpgradeOptions { source, verbose })
}

fn run_upgrade(opts: UpgradeOptions) -> Result<i32, String> {
    let home = runtime_home()?;
    let python = select_python_for_launcher_upgrade(&home)?;
    let rust_check_source = match &opts.source {
        UpgradeSource::Latest => UpdateSource::Latest,
        UpgradeSource::Local(path) => UpdateSource::Local(path.clone()),
    };
    ensure_rust_toolchain_for_update_source(&rust_check_source, &home, &python, opts.verbose)?;

    let (launcher_dir, cleanup_dir) = prepare_upgrade_launcher_source(&opts, &home, &python)?;
    let replace_result = (|| -> Result<LauncherReplaceResult, String> {
        let built = build_launcher_binary_from_source(&launcher_dir, &home, opts.verbose)?;
        replace_current_launcher_binary(&built, opts.verbose)
    })();
    if let Some(dir) = cleanup_dir {
        let _ = std::fs::remove_dir_all(&dir);
    }
    let replace_state = replace_result?;
    match replace_state {
        LauncherReplaceResult::ReplacedNow => {
            print_success_line("Launcher upgrade completed. Current `jx` has been replaced.");
        }
        #[cfg(windows)]
        LauncherReplaceResult::ReplacedAfterExit => {
            print_success_line(
                "Launcher upgrade compiled and staged. `jx` will be replaced after this command exits.",
            );
            println!("Keep this command running until update steps finish, then run `jx -v` to verify.");
        }
    }
    let (core_update, core_update_hint) = match &opts.source {
        UpgradeSource::Latest => (
            UpdateOptions {
                source: UpdateSource::Latest,
                verbose: opts.verbose,
                force_reinstall: false,
                editable: false,
            },
            "jx -update latest".to_string(),
        ),
        UpgradeSource::Local(path) => (
            UpdateOptions {
                source: UpdateSource::Local(path.clone()),
                verbose: opts.verbose,
                force_reinstall: false,
                editable: true,
            },
            format!("jx -update -e {path}"),
        ),
    };
    let core_update_result = run_update_internal(core_update, false);

    #[cfg(windows)]
    if matches!(replace_state, LauncherReplaceResult::ReplacedAfterExit) {
        schedule_windows_staged_launcher_replace(opts.verbose)?;
    }

    match core_update_result {
        Ok(_) => {
            print_success_line(&format!(
                "Core package update completed via `{core_update_hint}`."
            ));
            Ok(0)
        }
        Err(e) => Err(format!(
            "Launcher upgrade completed, but core package update (`{core_update_hint}`) failed: {e}"
        )),
    }
}

fn select_python_for_launcher_upgrade(runtime_home: &Path) -> Result<PathBuf, String> {
    let runtime_py = venv_python(&runtime_home.join("venv"));
    if runtime_py.exists() {
        return Ok(runtime_py);
    }
    let Some(sys_py) = find_system_python() else {
        return Err(python_install_hint());
    };
    Ok(PathBuf::from(sys_py))
}

fn prepare_upgrade_launcher_source(
    opts: &UpgradeOptions,
    runtime_home: &Path,
    python: &Path,
) -> Result<(PathBuf, Option<PathBuf>), String> {
    match &opts.source {
        UpgradeSource::Local(raw) => {
            let source = canonical_or_self(Path::new(raw));
            let launcher_dir = resolve_launcher_dir_from_source_path(&source)?;
            return Ok((launcher_dir, None));
        }
        UpgradeSource::Latest => {}
    }

    let work_dir = runtime_home.join(".launcher-upgrade-src");
    if work_dir.exists() {
        std::fs::remove_dir_all(&work_dir).map_err(|e| {
            format!(
                "Failed to clear launcher upgrade workspace {}: {e}",
                work_dir.display()
            )
        })?;
    }
    std::fs::create_dir_all(&work_dir).map_err(|e| {
        format!(
            "Failed to create launcher upgrade workspace {}: {e}",
            work_dir.display()
        )
    })?;

    let repo_dir = work_dir.join("repo");
    let repo_cn = repo_url_from_spec(GITHUB_SPEC_CN);
    let repo_origin = repo_url_from_spec(GITHUB_SPEC_ORIGIN);
    let mut git_ok = false;
    if command_ok("git", &["--version"]) {
        match git_clone_source_repo(
            &repo_cn,
            &repo_dir,
            "Cloning JanusX source (CN mirror) ...",
            opts.verbose,
        ) {
            Ok(_) => git_ok = true,
            Err(err_cn) => {
                eprintln!("GitHub CN mirror clone failed, retrying source...");
                if opts.verbose {
                    eprintln!("Reason: {err_cn}");
                }
                git_clone_source_repo(
                    &repo_origin,
                    &repo_dir,
                    "Cloning JanusX source (source) ...",
                    opts.verbose,
                )?;
                git_ok = true;
            }
        }
    }
    if git_ok {
        let launcher_dir = resolve_launcher_dir_from_source_path(&repo_dir)?;
        return Ok((launcher_dir, Some(work_dir)));
    }

    let archive_path = work_dir.join("JanusX-latest-main.tar.gz");
    match download_file_with_http_tools(
        GITHUB_ARCHIVE_CN,
        &archive_path,
        "Downloading GitHub source archive (CN mirror) ...",
        opts.verbose,
    ) {
        Ok(_) => {}
        Err(err_cn) => {
            eprintln!("GitHub source archive CN mirror failed, retrying source...");
            if opts.verbose {
                eprintln!("Reason: {err_cn}");
            }
            download_file_with_http_tools(
                GITHUB_ARCHIVE_ORIGIN,
                &archive_path,
                "Downloading GitHub source archive (source) ...",
                opts.verbose,
            )?;
        }
    }

    let extract_dir = work_dir.join("extract");
    std::fs::create_dir_all(&extract_dir).map_err(|e| {
        format!(
            "Failed to create extraction directory {}: {e}",
            extract_dir.display()
        )
    })?;
    extract_tar_gz_with_python(python, &archive_path, &extract_dir, opts.verbose)?;
    let repo_root = find_repo_root_with_launcher(&extract_dir).ok_or_else(|| {
        format!(
            "Failed to locate JanusX source root containing launcher/Cargo.toml under {}",
            extract_dir.display()
        )
    })?;
    let launcher_dir = resolve_launcher_dir_from_source_path(&repo_root)?;
    Ok((launcher_dir, Some(work_dir)))
}

fn git_clone_source_repo(
    repo_url: &str,
    dst: &Path,
    desc: &str,
    verbose: bool,
) -> Result<(), String> {
    if dst.exists() {
        std::fs::remove_dir_all(dst)
            .map_err(|e| format!("Failed to clear {} before clone: {e}", dst.display()))?;
    }
    let mut cmd = Command::new("git");
    cmd.arg("clone")
        .arg("--depth")
        .arg("1")
        .arg(repo_url)
        .arg(dst)
        .stdin(Stdio::null());
    match run_cmd_with_optional_spinner(&mut cmd, desc, verbose) {
        Ok(_) => Ok(()),
        Err(err) => {
            if should_retry_without_proxy_env(&err) {
                let mut retry = Command::new("git");
                retry
                    .arg("clone")
                    .arg("--depth")
                    .arg("1")
                    .arg(repo_url)
                    .arg(dst)
                    .stdin(Stdio::null());
                apply_temporary_no_proxy(&mut retry);
                let retry_desc = format!("{desc} (retry without proxy) ...");
                run_cmd_with_optional_spinner(&mut retry, &retry_desc, verbose).map_err(|e| {
                    format!("Failed to clone source repo: {err}\nRetry without proxy failed: {e}")
                })
            } else {
                Err(format!("Failed to clone source repo: {err}"))
            }
        }
    }
}

fn find_repo_root_with_launcher(root: &Path) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if dir.join("launcher").join("Cargo.toml").exists() {
            return Some(dir);
        }
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            }
        }
    }
    None
}

fn resolve_launcher_dir_from_source_path(source: &Path) -> Result<PathBuf, String> {
    let source = canonical_or_self(source);
    let direct = source.join("Cargo.toml");
    let direct_main = source.join("src").join("main.rs");
    if direct.exists() && direct_main.exists() {
        return Ok(source);
    }
    let nested = source.join("launcher");
    let nested_cargo = nested.join("Cargo.toml");
    if nested_cargo.exists() {
        return Ok(nested);
    }
    Err(format!(
        "Launcher source not found from {} (expect `<src>/launcher/Cargo.toml` or `<src>/Cargo.toml`).",
        source.display()
    ))
}

fn build_launcher_binary_from_source(
    launcher_dir: &Path,
    runtime_home: &Path,
    verbose: bool,
) -> Result<PathBuf, String> {
    if !launcher_dir.join("Cargo.toml").exists() {
        return Err(format!(
            "Launcher Cargo.toml not found in {}",
            launcher_dir.display()
        ));
    }

    let local_cargo = local_cargo_path(runtime_home);
    let use_local_toolchain = local_rust_toolchain_ready(runtime_home) && local_cargo.exists();

    let mut cmd = if use_local_toolchain {
        Command::new(&local_cargo)
    } else {
        Command::new("cargo")
    };
    cmd.arg("build")
        .arg("--release")
        .arg("--bin")
        .arg("jx")
        .current_dir(launcher_dir)
        .stdin(Stdio::null());
    if launcher_dir.join("Cargo.lock").exists() {
        cmd.arg("--locked");
    }
    if use_local_toolchain {
        apply_local_rust_env(&mut cmd, runtime_home, true);
    }
    run_cmd_with_optional_spinner(&mut cmd, "Compiling launcher from source ...", verbose)
        .map_err(|e| format!("Launcher source build failed: {e}"))?;

    let built = launcher_dir
        .join("target")
        .join("release")
        .join(if cfg!(windows) { "jx.exe" } else { "jx" });
    if !built.exists() {
        return Err(format!(
            "Build completed but launcher binary not found: {}",
            built.display()
        ));
    }
    Ok(built)
}

fn replace_current_launcher_binary(
    built_binary: &Path,
    verbose: bool,
) -> Result<LauncherReplaceResult, String> {
    let current = env::current_exe().map_err(|e| format!("Failed to locate current jx: {e}"))?;
    let install_dir = current
        .parent()
        .ok_or_else(|| "Failed to resolve current jx install directory.".to_string())?;
    let staged = install_dir.join(if cfg!(windows) {
        "jx.new.exe"
    } else {
        "jx.new"
    });
    if staged.exists() {
        std::fs::remove_file(&staged).map_err(|e| {
            format!(
                "Failed to remove stale staged launcher {}: {e}",
                staged.display()
            )
        })?;
    }
    std::fs::copy(built_binary, &staged).map_err(|e| {
        format!(
            "Failed to stage launcher binary {} -> {}: {e}",
            built_binary.display(),
            staged.display()
        )
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&staged)
            .map_err(|e| format!("Failed to stat staged launcher {}: {e}", staged.display()))?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&staged, perms)
            .map_err(|e| format!("Failed to chmod staged launcher {}: {e}", staged.display()))?;
    }

    #[cfg(windows)]
    {
        let _ = verbose;
        return Ok(LauncherReplaceResult::ReplacedAfterExit);
    }

    #[cfg(not(windows))]
    {
        match std::fs::rename(&staged, &current) {
            Ok(_) => Ok(LauncherReplaceResult::ReplacedNow),
            Err(rename_err) => {
                let _ = std::fs::remove_file(&current);
                std::fs::rename(&staged, &current).map_err(|retry_err| {
                    format!(
                        "Failed to replace current launcher {} (first: {}; retry: {})",
                        current.display(),
                        rename_err,
                        retry_err
                    )
                })?;
                Ok(LauncherReplaceResult::ReplacedNow)
            }
        }
    }
}

#[cfg(windows)]
fn schedule_windows_staged_launcher_replace(verbose: bool) -> Result<(), String> {
    let current = env::current_exe().map_err(|e| format!("Failed to locate current jx: {e}"))?;
    let install_dir = current
        .parent()
        .ok_or_else(|| "Failed to resolve current jx install directory.".to_string())?;
    let staged = install_dir.join("jx.new.exe");
    if !staged.exists() {
        // Nothing to do (already replaced, or staging failed earlier and was surfaced).
        return Ok(());
    }
    let helper = install_dir.join("jx_replace_launcher.cmd");
    let esc_stage = staged.to_string_lossy().replace('"', "\"\"");
    let esc_current = current.to_string_lossy().replace('"', "\"\"");
    let helper_script = format!(
        "@echo off\r\n\
setlocal enableextensions\r\n\
set \"SRC={esc_stage}\"\r\n\
set \"DST={esc_current}\"\r\n\
for /L %%i in (1,1,900) do (\r\n\
  move /Y \"%SRC%\" \"%DST%\" >NUL 2>NUL && goto :done\r\n\
  >NUL 2>NUL ping 127.0.0.1 -n 2\r\n\
)\r\n\
:done\r\n\
del /F /Q \"%~f0\" >NUL 2>NUL\r\n\
endlocal\r\n"
    );
    std::fs::write(&helper, helper_script).map_err(|e| {
        format!(
            "Failed to create launcher replacement helper {}: {e}",
            helper.display()
        )
    })?;
    if verbose {
        println!(
            "Applying launcher replacement after exit via helper: {}",
            helper.display()
        );
    }
    Command::new("cmd")
        .arg("/C")
        .arg(&helper)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to stage launcher replacement: {e}"))?;
    Ok(())
}

fn parse_update_args(args: &[String]) -> Result<UpdateOptions, String> {
    let mut dlc = false;
    let mut latest = false;
    let mut local_source: Option<String> = None;
    let mut verbose = false;
    let mut editable = false;

    for token in args {
        match token.as_str() {
            "dlc" => dlc = true,
            "latest" | "-latest" | "--latest" => latest = true,
            "-e" | "-editable" | "--editable" => editable = true,
            "-verbose" | "--verbose" => verbose = true,
            "-h" | "--help" | "help" => {
                println!(
                    "Update usage:\n  jx -update dlc [-verbose]\n  jx -update [latest|<local_path>] [-e|-editable] [-verbose]\n  Example: jx -update -e ."
                );
                return Err(String::new());
            }
            other => {
                if Path::new(other).exists() {
                    if local_source.is_some() {
                        return Err(
                            "Only one local source path is allowed for `-update`.".to_string()
                        );
                    }
                    local_source = Some(other.to_string());
                } else {
                    return Err(format!(
                        "Unknown update option: {other}\nUpdate usage:\n  jx -update dlc [-verbose]\n  jx -update [latest|<local_path>] [-e|-editable] [-verbose]\n  Example: jx -update -e ."
                    ));
                }
            }
        }
    }

    if dlc && (latest || local_source.is_some() || editable) {
        return Err("`jx -update dlc` does not accept other update source/options.".to_string());
    }
    if latest && local_source.is_some() {
        return Err("`latest` and local path cannot be used together.".to_string());
    }

    let source = if dlc {
        UpdateSource::Dlc
    } else if latest {
        UpdateSource::Latest
    } else if let Some(path) = local_source {
        UpdateSource::Local(path)
    } else {
        UpdateSource::Pypi
    };

    if editable && !matches!(source, UpdateSource::Local(_)) {
        return Err(
            "`-e/-editable` requires a local source path, e.g. `jx -update -e .`.".to_string(),
        );
    }

    Ok(UpdateOptions {
        source,
        verbose,
        force_reinstall: false,
        editable,
    })
}

fn run_update(opts: UpdateOptions) -> Result<i32, String> {
    run_update_internal(opts, true)
}

fn run_update_internal(
    opts: UpdateOptions,
    allow_launcher_self_update: bool,
) -> Result<i32, String> {
    if matches!(opts.source, UpdateSource::Dlc) {
        let home = runtime_home()?;
        let python = ensure_runtime(false)?;
        return dlc::run_dlc_update(&home, &python, opts.verbose);
    }

    let home = runtime_home()?;
    let pypi_spec = resolve_pypi_spec(&home);
    let python = ensure_venv()?;
    ensure_rust_toolchain_for_update_source(&opts.source, &home, &python, opts.verbose)?;
    let before = installed_version(&python);
    let jx_bin = env::current_exe().ok();

    if allow_launcher_self_update
        && matches!(opts.source, UpdateSource::Pypi)
        && maybe_self_update_launcher_from_pypi_check(&python, &home, opts.verbose)?
    {
        return Ok(0);
    }
    if matches!(opts.source, UpdateSource::Pypi) && !opts.force_reinstall {
        let check_start = Instant::now();
        if let (Some(current), Some(latest)) =
            (before.clone(), pypi_latest_version(&python, opts.verbose))
        {
            if compare_version_tokens(&current, &latest) != std::cmp::Ordering::Less {
                print_success_line(&format!(
                    "Already at latest PyPI version ({current})[{}]",
                    format_elapsed(check_start.elapsed())
                ));
                if github_has_newer_release_hint(&python, &home, &current, opts.verbose) {
                    println!("Use `jx -update latest` for GitHub latest.");
                }
                return Ok(0);
            }
        }
    }

    match &opts.source {
        UpdateSource::Latest => {
            if allow_launcher_self_update
                && maybe_self_update_launcher_from_release(&python, &home, opts.verbose)?
            {
                return Ok(0);
            }
            if ensure_git_available().is_err() {
                if !has_http_download_tool() {
                    return Err(git_install_hint());
                }
                let tmp_dir = home.join(".update-src");
                let archive_path = tmp_dir.join("JanusX-latest-main.tar.gz");
                let archive_update_result: Result<(), String> = (|| {
                    std::fs::create_dir_all(&tmp_dir).map_err(|e| {
                        format!(
                            "Failed to create GitHub source cache directory {}: {e}",
                            tmp_dir.display()
                        )
                    })?;
                    match download_file_with_http_tools(
                        GITHUB_ARCHIVE_CN,
                        &archive_path,
                        "Downloading GitHub source archive (CN mirror) ...",
                        opts.verbose,
                    ) {
                        Ok(_) => {}
                        Err(err_cn) => {
                            eprintln!("GitHub source archive CN mirror failed, retrying source...");
                            if opts.verbose {
                                eprintln!("Reason: {err_cn}");
                            }
                            download_file_with_http_tools(
                                GITHUB_ARCHIVE_ORIGIN,
                                &archive_path,
                                "Downloading GitHub source archive (source) ...",
                                opts.verbose,
                            )?;
                        }
                    }
                    let archive_spec = archive_path.to_string_lossy().to_string();
                    let _ = pip_install_update(
                        &python,
                        &home,
                        &archive_spec,
                        true,
                        false,
                        opts.verbose,
                        "Updating from GitHub source archive ...",
                        true,
                    )?;
                    warm_up_after_update(jx_bin.as_deref(), &home)?;
                    Ok(())
                })();

                // Always clean downloaded source archive after update attempt.
                let _ = std::fs::remove_file(&archive_path);
                let _ = std::fs::remove_dir(&tmp_dir);

                archive_update_result?;
                return Ok(0);
            }
            let repo_url_cn = repo_url_from_spec(GITHUB_SPEC_CN);
            let repo_url_origin = repo_url_from_spec(GITHUB_SPEC_ORIGIN);
            let (_repo_url, remote_head, using_cn_repo) =
                if let Some(head) = remote_head_commit(&repo_url_cn) {
                    (repo_url_cn.clone(), Some(head), true)
                } else if let Some(head) = remote_head_commit(&repo_url_origin) {
                    (repo_url_origin.clone(), Some(head), false)
                } else {
                    (repo_url_cn.clone(), None, true)
                };
            if opts.verbose && !using_cn_repo {
                println!("CN GitHub mirror is unavailable; using source.");
            }
            // `jx -update latest` always performs a full reinstall.
            let gh_force_reinstall = true;
            if opts.verbose {
                println!("Updating from GitHub (CN mirror) ...");
            }
            match pip_install_update(
                &python,
                &home,
                GITHUB_SPEC_CN,
                gh_force_reinstall,
                false,
                opts.verbose,
                "Updating from GitHub (CN mirror) ...",
                true,
            ) {
                Ok(_) => {
                    if let Some(remote) = remote_head.as_deref() {
                        write_commit_marker(&home, remote);
                    }
                    warm_up_after_update(jx_bin.as_deref(), &home)?;
                    return Ok(0);
                }
                Err(err_primary) => {
                    eprintln!("GitHub CN mirror update failed, retrying with source...");
                    if opts.verbose {
                        eprintln!("Reason: {err_primary}");
                    }
                    let _ = pip_install_update(
                        &python,
                        &home,
                        GITHUB_SPEC_ORIGIN,
                        gh_force_reinstall,
                        false,
                        opts.verbose,
                        "Updating from GitHub (source)...",
                        false,
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
            let local_spec = resolve_local_update_spec(path)?;
            if opts.verbose {
                println!("Updating from local source: {local_spec}");
            }
            let local_res = pip_install_update(
                &python,
                &home,
                &local_spec,
                opts.force_reinstall,
                opts.editable,
                opts.verbose,
                "Updating from local source...",
                true,
            );
            #[cfg(target_os = "windows")]
            let local_res = match local_res {
                Ok(v) => Ok(v),
                Err(e) => {
                    if opts.editable && is_editable_local_pyd_copy_failure(&e) {
                        eprintln!(
                            "Editable local install failed due to in-tree .pyd copy/lock conflict; retrying non-editable local install..."
                        );
                        pip_install_update(
                            &python,
                            &home,
                            &local_spec,
                            true,
                            false,
                            opts.verbose,
                            "Retrying local source install (non-editable)...",
                            true,
                        )
                        .map_err(|e2| format!("{e}\n\nRetry without editable failed: {e2}"))
                    } else {
                        Err(e)
                    }
                }
            };
            let _ = local_res?;
            warm_up_after_update(jx_bin.as_deref(), &home)?;
            return Ok(0);
        }
        UpdateSource::Pypi => {}
        UpdateSource::Dlc => {
            return Err("Unhandled update source: dlc".to_string());
        }
    }

    if opts.verbose {
        println!("Updating from PyPI...");
    }
    let elapsed = pip_install_update(
        &python,
        &home,
        &pypi_spec,
        opts.force_reinstall,
        false,
        opts.verbose,
        "Updating from PyPI...",
        true,
    )?;
    let after = installed_version(&python);
    if !opts.force_reinstall && before.is_some() && before == after {
        if let Some(v) = after {
            print_success_line(&format!(
                "Already at latest PyPI version ({v}) [{}]",
                format_elapsed(elapsed)
            ));
        } else {
            print_success_line(&format!(
                "Already at latest PyPI version. [{}]",
                format_elapsed(elapsed)
            ));
        }
        if let Some(v) = before.as_deref() {
            if github_has_newer_release_hint(&python, &home, v, opts.verbose) {
                println!("Use `jx -update latest` for GitHub latest.");
            }
        }
        return Ok(0);
    }
    warm_up_after_update(jx_bin.as_deref(), &home)?;
    Ok(0)
}

fn is_python_project_root(path: &Path) -> bool {
    path.join("pyproject.toml").is_file() || path.join("setup.py").is_file()
}

fn resolve_local_update_spec(path: &str) -> Result<String, String> {
    let p = Path::new(path);
    if !p.exists() {
        return Err(format!("Local source path does not exist: {path}"));
    }
    if p.is_file() {
        return Ok(path.to_string());
    }
    if is_python_project_root(p) {
        return Ok(path.to_string());
    }
    let python_sub = p.join("python");
    if is_python_project_root(&python_sub) {
        return Ok(python_sub.to_string_lossy().to_string());
    }
    Err(format!(
        "Local source path `{}` is not a Python project (missing pyproject.toml/setup.py).\n\
Try repo root with pyproject.toml, or use `{}`.",
        path,
        python_sub.display()
    ))
}

fn ensure_rust_toolchain_for_update_source(
    source: &UpdateSource,
    runtime_home: &Path,
    python: &Path,
    verbose: bool,
) -> Result<(), String> {
    if !matches!(source, UpdateSource::Latest | UpdateSource::Local(_)) {
        return Ok(());
    }
    if verbose {
        eprintln!("Checking Rust toolchain for source update...");
    }
    if any_rust_toolchain_ready(runtime_home) {
        #[cfg(target_os = "windows")]
        ensure_windows_msvc_linker_ready(runtime_home)?;
        if verbose {
            eprintln!("Rust toolchain is available.");
        }
        return Ok(());
    }
    if verbose {
        eprintln!("Rust toolchain not detected; installing local Rust toolchain...");
    }
    ensure_local_rust_toolchain(runtime_home, python, verbose)?;
    #[cfg(target_os = "windows")]
    ensure_windows_msvc_linker_ready(runtime_home)?;
    Ok(())
}

fn rustup_home(runtime_home: &Path) -> PathBuf {
    runtime_home.join(".rustup")
}

fn cargo_home(runtime_home: &Path) -> PathBuf {
    runtime_home.join(".cargo")
}

fn cargo_bin_dir(runtime_home: &Path) -> PathBuf {
    cargo_home(runtime_home).join("bin")
}

fn local_rustc_path(runtime_home: &Path) -> PathBuf {
    if cfg!(windows) {
        cargo_bin_dir(runtime_home).join("rustc.exe")
    } else {
        cargo_bin_dir(runtime_home).join("rustc")
    }
}

fn local_cargo_path(runtime_home: &Path) -> PathBuf {
    if cfg!(windows) {
        cargo_bin_dir(runtime_home).join("cargo.exe")
    } else {
        cargo_bin_dir(runtime_home).join("cargo")
    }
}

fn apply_local_rust_env(cmd: &mut Command, runtime_home: &Path, use_cn_mirror: bool) {
    let rustup_dir = rustup_home(runtime_home);
    let cargo_dir = cargo_home(runtime_home);
    cmd.env("RUSTUP_HOME", &rustup_dir);
    cmd.env("CARGO_HOME", &cargo_dir);
    if use_cn_mirror {
        cmd.env("RUSTUP_DIST_SERVER", RUSTUP_DIST_SERVER_CN);
        cmd.env("RUSTUP_UPDATE_ROOT", RUSTUP_UPDATE_ROOT_CN);
    } else {
        cmd.env("RUSTUP_DIST_SERVER", RUSTUP_DIST_SERVER_ORIGIN);
        cmd.env("RUSTUP_UPDATE_ROOT", RUSTUP_UPDATE_ROOT_ORIGIN);
    }
    cmd.env("CARGO_REGISTRIES_CRATES_IO_PROTOCOL", "sparse");
    cmd.env("CARGO_NET_GIT_FETCH_WITH_CLI", "true");
    cmd.env("CARGO_TERM_COLOR", "always");

    let mut paths: Vec<PathBuf> = vec![cargo_bin_dir(runtime_home)];
    if let Some(curr) = env::var_os("PATH") {
        paths.extend(env::split_paths(&curr));
    }
    if let Ok(joined) = env::join_paths(paths) {
        cmd.env("PATH", joined);
    }
}

fn local_rust_toolchain_ready(runtime_home: &Path) -> bool {
    let rustc = local_rustc_path(runtime_home);
    let cargo = local_cargo_path(runtime_home);
    if !(rustc.exists() && cargo.exists()) {
        return false;
    }
    let mut cmd_rustc = Command::new(&rustc);
    cmd_rustc
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    apply_local_rust_env(&mut cmd_rustc, runtime_home, true);
    let rustc_ok = cmd_rustc.status().map(|s| s.success()).unwrap_or(false);

    if !rustc_ok {
        return false;
    }

    let mut cmd_cargo = Command::new(&cargo);
    cmd_cargo
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    apply_local_rust_env(&mut cmd_cargo, runtime_home, true);
    cmd_cargo.status().map(|s| s.success()).unwrap_or(false)
}

fn system_rust_toolchain_ready() -> bool {
    command_ok("rustc", &["--version"]) && command_ok("cargo", &["--version"])
}

fn any_rust_toolchain_ready(runtime_home: &Path) -> bool {
    local_rust_toolchain_ready(runtime_home) || system_rust_toolchain_ready()
}

#[cfg(target_os = "windows")]
fn parse_rustc_host_triple(text: &str) -> Option<String> {
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("host:") {
            let host = rest.trim();
            if !host.is_empty() {
                return Some(host.to_string());
            }
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn local_rust_host_triple(runtime_home: &Path) -> Option<String> {
    let rustc = local_rustc_path(runtime_home);
    if !rustc.exists() {
        return None;
    }
    let mut cmd = Command::new(&rustc);
    cmd.arg("-vV")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    apply_local_rust_env(&mut cmd, runtime_home, true);
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    parse_rustc_host_triple(&String::from_utf8_lossy(&out.stdout))
}

#[cfg(target_os = "windows")]
fn system_rust_host_triple() -> Option<String> {
    let out = Command::new("rustc")
        .arg("-vV")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_rustc_host_triple(&String::from_utf8_lossy(&out.stdout))
}

#[cfg(target_os = "windows")]
fn windows_rust_host_triple(runtime_home: &Path) -> Option<String> {
    local_rust_host_triple(runtime_home).or_else(system_rust_host_triple)
}

#[cfg(target_os = "windows")]
fn windows_rust_requires_msvc_linker(runtime_home: &Path) -> bool {
    let host = windows_rust_host_triple(runtime_home)
        .unwrap_or_else(|| "x86_64-pc-windows-msvc".to_string())
        .to_ascii_lowercase();
    host.contains("windows-msvc")
}

#[cfg(target_os = "windows")]
fn windows_has_link_exe_in_path() -> bool {
    if let Some(path_var) = env::var_os("PATH") {
        for d in env::split_paths(&path_var) {
            if d.join("link.exe").exists() {
                return true;
            }
        }
    }
    command_ok("where", &["link.exe"]) || command_ok("where.exe", &["link.exe"])
}

#[cfg(target_os = "windows")]
fn windows_parse_first_existing_path(output: &[u8]) -> Option<PathBuf> {
    let text = String::from_utf8_lossy(output);
    for line in text.lines() {
        let p = PathBuf::from(line.trim());
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn windows_vswhere_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for key in ["ProgramFiles(x86)", "ProgramFiles"] {
        if let Some(base) = env::var_os(key) {
            paths.push(
                PathBuf::from(base)
                    .join("Microsoft Visual Studio")
                    .join("Installer")
                    .join("vswhere.exe"),
            );
        }
    }
    paths
}

#[cfg(target_os = "windows")]
fn windows_find_vswhere() -> Option<PathBuf> {
    for p in windows_vswhere_candidates() {
        if p.is_file() {
            return Some(p);
        }
    }
    if command_ok("vswhere", &["-help"]) {
        return Some(PathBuf::from("vswhere"));
    }
    None
}

#[cfg(target_os = "windows")]
fn windows_find_vsdevcmd_with_vswhere(vswhere: &Path) -> Option<PathBuf> {
    let out = Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-find",
            "**\\Common7\\Tools\\VsDevCmd.bat",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    windows_parse_first_existing_path(&out.stdout)
}

#[cfg(target_os = "windows")]
fn windows_default_vsdevcmd_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let years = ["2022", "2019", "2017"];
    let editions = ["BuildTools", "Community", "Professional", "Enterprise"];
    for base_key in ["ProgramFiles", "ProgramFiles(x86)"] {
        let Some(base) = env::var_os(base_key) else {
            continue;
        };
        let base = PathBuf::from(base).join("Microsoft Visual Studio");
        for year in years {
            for edition in editions {
                out.push(
                    base.join(year)
                        .join(edition)
                        .join("Common7")
                        .join("Tools")
                        .join("VsDevCmd.bat"),
                );
            }
        }
    }
    out
}

#[cfg(target_os = "windows")]
fn windows_find_vsdevcmd() -> Option<PathBuf> {
    if let Some(vswhere) = windows_find_vswhere() {
        if let Some(vsdevcmd) = windows_find_vsdevcmd_with_vswhere(&vswhere) {
            return Some(vsdevcmd);
        }
    }
    windows_default_vsdevcmd_candidates()
        .into_iter()
        .find(|p| p.is_file())
}

#[cfg(target_os = "windows")]
fn windows_find_vcvars64_with_vswhere(vswhere: &Path) -> Option<PathBuf> {
    let out = Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-find",
            "**\\VC\\Auxiliary\\Build\\vcvars64.bat",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    windows_parse_first_existing_path(&out.stdout)
}

#[cfg(target_os = "windows")]
fn windows_default_vcvars64_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let years = ["2022", "2019", "2017"];
    let editions = ["BuildTools", "Community", "Professional", "Enterprise"];
    for base_key in ["ProgramFiles", "ProgramFiles(x86)"] {
        let Some(base) = env::var_os(base_key) else {
            continue;
        };
        let base = PathBuf::from(base).join("Microsoft Visual Studio");
        for year in years {
            for edition in editions {
                out.push(
                    base.join(year)
                        .join(edition)
                        .join("VC")
                        .join("Auxiliary")
                        .join("Build")
                        .join("vcvars64.bat"),
                );
            }
        }
    }
    out
}

#[cfg(target_os = "windows")]
fn windows_find_vcvars64() -> Option<PathBuf> {
    if let Some(vswhere) = windows_find_vswhere() {
        if let Some(vcvars) = windows_find_vcvars64_with_vswhere(&vswhere) {
            return Some(vcvars);
        }
    }
    windows_default_vcvars64_candidates()
        .into_iter()
        .find(|p| p.is_file())
}

#[cfg(target_os = "windows")]
fn windows_find_link_exe_with_vswhere(vswhere: &Path) -> Option<PathBuf> {
    let patterns = [
        "**\\VC\\Tools\\MSVC\\**\\bin\\Hostx64\\x64\\link.exe",
        "**\\VC\\Tools\\MSVC\\**\\bin\\Hostx64\\x86\\link.exe",
        "**\\VC\\Tools\\MSVC\\**\\bin\\Hostx86\\x64\\link.exe",
        "**\\VC\\Tools\\MSVC\\**\\bin\\Hostx86\\x86\\link.exe",
        "**\\VC\\Tools\\MSVC\\**\\bin\\HostArm64\\arm64\\link.exe",
        "**\\VC\\Tools\\MSVC\\**\\bin\\HostArm64\\x64\\link.exe",
    ];
    for pattern in patterns {
        let out = Command::new(vswhere)
            .args([
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-find",
                pattern,
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        if !out.status.success() {
            continue;
        }
        if let Some(p) = windows_parse_first_existing_path(&out.stdout) {
            return Some(p);
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn windows_find_link_exe_direct() -> Option<PathBuf> {
    if let Some(vswhere) = windows_find_vswhere() {
        if let Some(link) = windows_find_link_exe_with_vswhere(&vswhere) {
            return Some(link);
        }
    }
    let years = ["2022", "2019", "2017"];
    let editions = ["BuildTools", "Community", "Professional", "Enterprise"];
    for base_key in ["ProgramFiles", "ProgramFiles(x86)"] {
        let Some(base) = env::var_os(base_key) else {
            continue;
        };
        let base = PathBuf::from(base).join("Microsoft Visual Studio");
        for year in years {
            for edition in editions {
                let msvc_root = base
                    .join(year)
                    .join(edition)
                    .join("VC")
                    .join("Tools")
                    .join("MSVC");
                if !msvc_root.is_dir() {
                    continue;
                }
                let Ok(vers) = std::fs::read_dir(&msvc_root) else {
                    continue;
                };
                for v in vers.flatten() {
                    let link = v
                        .path()
                        .join("bin")
                        .join("Hostx64")
                        .join("x64")
                        .join("link.exe");
                    if link.is_file() {
                        return Some(link);
                    }
                }
            }
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn windows_capture_batch_env(script_candidates: &[String]) -> Result<Vec<(String, String)>, String> {
    fn path_has_link(path_value: &str) -> bool {
        for d in env::split_paths(path_value) {
            if d.join("link.exe").is_file() {
                return true;
            }
        }
        false
    }

    fn kvs_has_link(kvs: &[(String, String)]) -> bool {
        for (k, v) in kvs {
            if k.eq_ignore_ascii_case("PATH") && path_has_link(v) {
                return true;
            }
        }
        false
    }

    fn parse_env_pairs(stdout: &[u8]) -> Vec<(String, String)> {
        let mut kvs = Vec::new();
        for line in String::from_utf8_lossy(stdout).lines() {
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            if k.is_empty() || k.starts_with('=') {
                continue;
            }
            kvs.push((k.to_string(), v.to_string()));
        }
        kvs
    }

    let mut last_exit = 1i32;
    for script in script_candidates {
        let cmdline = format!("{script} & set");
        let out = Command::new("cmd")
            .arg("/d")
            .arg("/c")
            .arg(cmdline)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .map_err(|e| format!("Failed to run MSVC setup script: {e}"))?;
        last_exit = exit_code(out.status);
        let kvs = parse_env_pairs(&out.stdout);
        if !kvs.is_empty() && kvs_has_link(&kvs) {
            return Ok(kvs);
        }
    }

    Err(format!(
        "MSVC setup script exited with code {} while probing environment (no PATH containing link.exe found).",
        last_exit
    ))
}

#[cfg(target_os = "windows")]
fn windows_try_apply_msvc_env() -> Result<(), String> {
    let mut probe_scripts: Vec<String> = Vec::new();
    if let Some(vsdevcmd) = windows_find_vsdevcmd() {
        let path = vsdevcmd.display();
        probe_scripts.push(format!(
            "call \"{}\" -no_logo -arch=x64 -host_arch=x64 >NUL 2>&1",
            path
        ));
        probe_scripts.push(format!(
            "call \"{}\" -no_logo -arch=x64 >NUL 2>&1",
            path
        ));
        probe_scripts.push(format!("call \"{}\" >NUL 2>&1", path));
    }
    if let Some(vcvars64) = windows_find_vcvars64() {
        let path = vcvars64.display();
        probe_scripts.push(format!("call \"{}\" amd64 >NUL 2>&1", path));
        probe_scripts.push(format!("call \"{}\" >NUL 2>&1", path));
    }
    if probe_scripts.is_empty() {
        // No setup scripts found; still allow direct link.exe path probing below.
    }

    let mut probe_err: Option<String> = None;
    if !probe_scripts.is_empty() {
        match windows_capture_batch_env(&probe_scripts) {
            Ok(kvs) => {
                const KEYS: &[&str] = &[
                    "PATH",
                    "INCLUDE",
                    "LIB",
                    "LIBPATH",
                    "VCToolsInstallDir",
                    "VCINSTALLDIR",
                    "VSINSTALLDIR",
                    "WindowsSdkDir",
                    "WindowsSdkVersion",
                    "UniversalCRTSdkDir",
                    "UCRTVersion",
                ];
                for (k, v) in kvs {
                    if KEYS.contains(&k.as_str()) && !v.trim().is_empty() {
                        env::set_var(k, v);
                    }
                }
            }
            Err(e) => {
                probe_err = Some(e);
            }
        }
    }
    if windows_has_link_exe_in_path() {
        return Ok(());
    }
    if let Some(link) = windows_find_link_exe_direct() {
        if let Some(dir) = link.parent() {
            let mut paths = vec![dir.to_path_buf()];
            if let Some(current) = env::var_os("PATH") {
                paths.extend(env::split_paths(&current));
            }
            if let Ok(joined) = env::join_paths(paths) {
                env::set_var("PATH", joined);
                if windows_has_link_exe_in_path() {
                    return Ok(());
                }
            }
        }
    }
    if let Some(e) = probe_err {
        return Err(format!(
            "{e} (fallback direct scan also failed to locate usable link.exe)"
        ));
    }
    Err("MSVC environment loaded, but `link.exe` is still not discoverable in PATH.".to_string())
}

#[cfg(target_os = "windows")]
static WINDOWS_MSVC_ENV_INIT: OnceLock<Result<(), String>> = OnceLock::new();

#[cfg(target_os = "windows")]
fn windows_msvc_linker_missing_message() -> String {
    "Rust toolchain is installed, but MSVC linker `link.exe` was not found.\n\
JanusX source build on Windows currently uses MSVC target and requires Visual C++ Build Tools.\n\n\
Fix options:\n\
1) Install Build Tools for Visual Studio 2022 (Desktop development with C++), then reopen terminal.\n\
2) Or run JanusX commands from \"x64 Native Tools Command Prompt for VS 2022\".\n\n\
Quick install (PowerShell, admin):\n\
  winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override \"--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools\"\n\n\
After install, run:\n\
  jx -upgrade\n\
or\n\
  jx -update latest"
        .to_string()
}

#[cfg(target_os = "windows")]
fn ensure_windows_msvc_linker_ready(runtime_home: &Path) -> Result<(), String> {
    if !windows_rust_requires_msvc_linker(runtime_home) {
        return Ok(());
    }
    if windows_has_link_exe_in_path() {
        return Ok(());
    }
    let auto = WINDOWS_MSVC_ENV_INIT.get_or_init(windows_try_apply_msvc_env);
    match auto {
        Ok(()) if windows_has_link_exe_in_path() => Ok(()),
        Ok(()) => Err(windows_msvc_linker_missing_message()),
        Err(e) => Err(format!(
            "{}\n\nAuto-detect attempt failed: {}",
            windows_msvc_linker_missing_message(),
            e
        )),
    }
}

#[cfg(target_os = "windows")]
fn is_windows_linker_missing_error(err: &str) -> bool {
    let e = err.to_ascii_lowercase();
    e.contains("linker `link.exe` not found")
        || (e.contains("program not found") && e.contains("link.exe"))
        || e.contains("the msvc targets depend on the msvc linker")
        || (e.contains("visual studio") && e.contains("link.exe"))
}

fn rustup_target_triple() -> Option<&'static str> {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("x86_64-unknown-linux-gnu")
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Some("aarch64-unknown-linux-gnu")
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Some("aarch64-apple-darwin")
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        Some("x86_64-apple-darwin")
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Some("x86_64-pc-windows-msvc")
    } else if cfg!(all(target_os = "windows", target_arch = "aarch64")) {
        Some("aarch64-pc-windows-msvc")
    } else {
        None
    }
}

fn rustup_init_file_name() -> &'static str {
    if cfg!(windows) {
        "rustup-init.exe"
    } else {
        "rustup-init"
    }
}

fn rustup_init_download_url_cn() -> Option<String> {
    let triple = rustup_target_triple()?;
    Some(format!(
        "{}/rustup/dist/{}/{}",
        RUSTUP_DIST_SERVER_CN,
        triple,
        rustup_init_file_name()
    ))
}

fn rustup_init_download_url_origin() -> Option<String> {
    let triple = rustup_target_triple()?;
    Some(format!(
        "{}/rustup/dist/{}/{}",
        RUSTUP_DIST_SERVER_ORIGIN,
        triple,
        rustup_init_file_name()
    ))
}

fn ensure_local_rust_toolchain(
    runtime_home: &Path,
    python: &Path,
    verbose: bool,
) -> Result<(), String> {
    if local_rust_toolchain_ready(runtime_home) {
        return Ok(());
    }

    let (Some(url_cn), Some(url_origin)) = (
        rustup_init_download_url_cn(),
        rustup_init_download_url_origin(),
    ) else {
        return Err(
            "Current platform is not supported for local rustup bootstrap in `jx -update latest`."
                .to_string(),
        );
    };

    std::fs::create_dir_all(rustup_home(runtime_home)).map_err(|e| {
        format!(
            "Failed to create {}: {e}",
            rustup_home(runtime_home).display()
        )
    })?;
    std::fs::create_dir_all(cargo_home(runtime_home)).map_err(|e| {
        format!(
            "Failed to create {}: {e}",
            cargo_home(runtime_home).display()
        )
    })?;

    let install_tmp = runtime_home.join(".rustup-bootstrap");
    std::fs::create_dir_all(&install_tmp)
        .map_err(|e| format!("Failed to create {}: {e}", install_tmp.display()))?;
    let installer = install_tmp.join(rustup_init_file_name());

    match download_file_with_fallback(
        python,
        &url_cn,
        &installer,
        "Downloading local Rust toolchain bootstrap (CN mirror) ...",
        verbose,
    ) {
        Ok(_) => {}
        Err(err_cn) => {
            eprintln!("CN Rust bootstrap mirror failed, retrying source...");
            if verbose {
                eprintln!("Reason: {err_cn}");
            }
            download_file_with_fallback(
                python,
                &url_origin,
                &installer,
                "Downloading local Rust toolchain bootstrap (source) ...",
                verbose,
            )?;
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&installer)
            .map_err(|e| format!("Failed to stat {}: {e}", installer.display()))?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&installer, perms)
            .map_err(|e| format!("Failed to chmod {}: {e}", installer.display()))?;
    }

    let run_rustup_install = |use_cn_mirror: bool, desc: &str| -> Result<(), String> {
        let mut cmd = Command::new(&installer);
        cmd.arg("-y")
            .arg("--profile")
            .arg("minimal")
            .arg("--default-toolchain")
            .arg("stable")
            .arg("--no-modify-path")
            .stdin(Stdio::null());
        apply_local_rust_env(&mut cmd, runtime_home, use_cn_mirror);
        if verbose {
            let status = cmd
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .status()
                .map_err(|e| format!("Failed to install local Rust toolchain: {e}"))?;
            if !status.success() {
                return Err(format!(
                    "Failed to install local Rust toolchain: command failed with exit={}",
                    exit_code(status)
                ));
            }
        } else {
            cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
            let (out, elapsed) = run_with_spinner(&mut cmd, desc)
                .map_err(|e| format!("Failed to install local Rust toolchain: {e}"))?;
            if out.status.success() {
                print_success_line(&format!("{desc}[{}]", format_elapsed(elapsed)));
            } else {
                let mut msg = String::new();
                msg.push_str(&String::from_utf8_lossy(&out.stdout));
                msg.push_str(&String::from_utf8_lossy(&out.stderr));
                return Err(format!(
                    "Failed to install local Rust toolchain: command failed with exit={}\n{}",
                    exit_code(out.status),
                    msg.trim()
                ));
            }
        }
        Ok(())
    };

    match run_rustup_install(true, "Installing local Rust toolchain (CN mirror) ...") {
        Ok(_) => {}
        Err(err_cn) => {
            eprintln!("CN Rust toolchain install failed, retrying source...");
            if verbose {
                eprintln!("Reason: {err_cn}");
            }
            run_rustup_install(false, "Installing local Rust toolchain (source) ...")?;
        }
    }

    if !local_rust_toolchain_ready(runtime_home) {
        return Err(
            "Local Rust toolchain bootstrap finished, but `rustc/cargo` are still unavailable."
                .to_string(),
        );
    }
    Ok(())
}

fn maybe_self_update_launcher_from_pypi_check(
    python: &Path,
    runtime_home: &Path,
    verbose: bool,
) -> Result<bool, String> {
    let Some(current_version) =
        detect_expected_janusx_version(Some(runtime_home)).or_else(|| installed_version(python))
    else {
        return Ok(false);
    };
    let Some(latest_pypi) = pypi_latest_version(python, verbose) else {
        return Ok(false);
    };
    if compare_version_tokens(&current_version, &latest_pypi) != std::cmp::Ordering::Less {
        return Ok(false);
    }
    println!(
        "Detected newer PyPI janusx: current v{}, latest v{}",
        current_version, latest_pypi
    );
    println!("Switching to launcher self-update flow...");
    maybe_self_update_launcher_from_release(python, runtime_home, verbose)
}

fn maybe_self_update_launcher_from_release(
    python: &Path,
    runtime_home: &Path,
    verbose: bool,
) -> Result<bool, String> {
    let Some(asset_suffix) = release_asset_suffix_for_platform() else {
        return Ok(false);
    };
    let Some(current_version) =
        detect_expected_janusx_version(Some(runtime_home)).or_else(|| installed_version(python))
    else {
        return Ok(false);
    };
    let Some((tag_name, asset_name, asset_url)) =
        github_latest_release_asset(python, asset_suffix, verbose)?
    else {
        return Ok(false);
    };
    let Some(latest_version) = normalize_version_token(&tag_name) else {
        return Ok(false);
    };
    if compare_version_tokens(&current_version, &latest_version) != std::cmp::Ordering::Less {
        return Ok(false);
    }

    println!(
        "Detected newer launcher release: current v{}, latest v{}",
        current_version, latest_version
    );
    println!("Auto-updating launcher from GitHub release asset: {asset_name}");

    let work_dir = env::temp_dir().join(format!(
        "jx_selfupdate_{}_{}",
        process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_millis()
    ));
    std::fs::create_dir_all(&work_dir).map_err(|e| {
        format!(
            "Failed to create self-update temp dir {}: {e}",
            work_dir.display()
        )
    })?;
    let archive_path = work_dir.join(&asset_name);
    download_release_asset(python, &asset_url, &archive_path, verbose)?;
    let extract_dir = work_dir.join("extract");
    std::fs::create_dir_all(&extract_dir).map_err(|e| {
        format!(
            "Failed to create extraction directory {}: {e}",
            extract_dir.display()
        )
    })?;
    extract_tar_gz_with_python(python, &archive_path, &extract_dir, verbose)?;
    let installer = find_installer_in_tree(&extract_dir).ok_or_else(|| {
        format!(
            "Installer executable not found after extraction in {}",
            extract_dir.display()
        )
    })?;
    run_downloaded_installer(&installer, verbose)?;
    Ok(true)
}

fn release_asset_suffix_for_platform() -> Option<&'static str> {
    #[cfg(target_os = "windows")]
    {
        return Some("windows-x86_64.tar.gz");
    }
    #[cfg(target_os = "macos")]
    {
        return Some("darwin-universal.tar.gz");
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        return Some("linux-x86_64.tar.gz");
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        return Some("linux-aarch64.tar.gz");
    }
    #[allow(unreachable_code)]
    None
}

fn github_latest_release_asset(
    python: &Path,
    suffix: &str,
    verbose: bool,
) -> Result<Option<(String, String, String)>, String> {
    let script = r#"
import json, sys, urllib.request
url = "https://api.github.com/repos/FJingxian/JanusX/releases/latest"
req = urllib.request.Request(url, headers={
    "Accept": "application/vnd.github+json",
    "User-Agent": "jx-launcher"
})
with urllib.request.urlopen(req, timeout=20) as resp:
    obj = json.load(resp)
tag = str(obj.get("tag_name", "")).strip()
suffix = sys.argv[1]
for a in (obj.get("assets") or []):
    name = str(a.get("name", "")).strip()
    if name.endswith(suffix):
        print(tag)
        print(name)
        print(str(a.get("browser_download_url", "")).strip())
        break
"#;

    let out = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(suffix)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    let Ok(out) = out else {
        return Ok(None);
    };
    if !out.status.success() {
        if verbose {
            eprintln!(
                "Warning: failed to query GitHub latest release.\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        return Ok(None);
    }
    let lines: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect();
    if lines.len() < 3 {
        return Ok(None);
    }
    Ok(Some((lines[0].clone(), lines[1].clone(), lines[2].clone())))
}

fn pypi_latest_version(python: &Path, verbose: bool) -> Option<String> {
    let script = r#"
import json, urllib.request
url = "https://pypi.org/pypi/janusx/json"
req = urllib.request.Request(url, headers={"User-Agent": "jx-launcher"})
with urllib.request.urlopen(req, timeout=20) as resp:
    obj = json.load(resp)
info = obj.get("info") or {}
v = str(info.get("version", "")).strip()
if v:
    print(v)
"#;
    let out = Command::new(python)
        .arg("-c")
        .arg(script)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    if !out.status.success() {
        if verbose {
            eprintln!(
                "Warning: failed to query PyPI latest janusx version.\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    normalize_version_token(&s)
}

fn github_has_newer_release_hint(
    python: &Path,
    runtime_home: &Path,
    current_version: &str,
    verbose: bool,
) -> bool {
    let Some((tag, published_at)) = github_latest_release_meta(python, verbose) else {
        return false;
    };
    let latest_ver = normalize_version_token(&tag).unwrap_or_default();
    if !latest_ver.is_empty()
        && compare_version_tokens(current_version, &latest_ver) == std::cmp::Ordering::Less
    {
        return true;
    }
    if latest_ver == current_version {
        if let Some(local_time) = python_core_update_time(python, runtime_home) {
            let local_date = local_time.split_whitespace().next().unwrap_or_default();
            let remote_date = published_at.get(0..10).unwrap_or_default();
            return !local_date.is_empty() && !remote_date.is_empty() && remote_date > local_date;
        }
    }
    false
}

fn github_latest_release_meta(python: &Path, verbose: bool) -> Option<(String, String)> {
    let script = r#"
import json, urllib.request
url = "https://api.github.com/repos/FJingxian/JanusX/releases/latest"
req = urllib.request.Request(url, headers={
    "Accept": "application/vnd.github+json",
    "User-Agent": "jx-launcher"
})
with urllib.request.urlopen(req, timeout=20) as resp:
    obj = json.load(resp)
tag = str(obj.get("tag_name", "")).strip()
published = str(obj.get("published_at", "")).strip()
if tag:
    print(tag)
    print(published)
"#;
    let out = Command::new(python)
        .arg("-c")
        .arg(script)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    if !out.status.success() {
        if verbose {
            eprintln!(
                "Warning: failed to query GitHub latest release metadata.\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        return None;
    }
    let lines: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect();
    if lines.is_empty() {
        return None;
    }
    Some((lines[0].clone(), lines.get(1).cloned().unwrap_or_default()))
}

fn compare_version_tokens(a: &str, b: &str) -> std::cmp::Ordering {
    let pa: Vec<u32> = a
        .split('.')
        .map(|x| x.parse::<u32>().ok().unwrap_or(0))
        .collect();
    let pb: Vec<u32> = b
        .split('.')
        .map(|x| x.parse::<u32>().ok().unwrap_or(0))
        .collect();
    let n = pa.len().max(pb.len());
    for i in 0..n {
        let va = *pa.get(i).unwrap_or(&0);
        let vb = *pb.get(i).unwrap_or(&0);
        match va.cmp(&vb) {
            std::cmp::Ordering::Equal => {}
            o => return o,
        }
    }
    std::cmp::Ordering::Equal
}

fn download_release_asset(
    python: &Path,
    url: &str,
    output: &Path,
    verbose: bool,
) -> Result<(), String> {
    let mirror_url = github_mirror_url(url);
    if mirror_url != url {
        match download_file_with_fallback(
            python,
            &mirror_url,
            output,
            "Downloading launcher installer (CN mirror) ...",
            verbose,
        ) {
            Ok(_) => return Ok(()),
            Err(err_cn) => {
                eprintln!("GitHub CN mirror download failed, retrying source...");
                if verbose {
                    eprintln!("Reason: {err_cn}");
                }
            }
        }
    }
    download_file_with_fallback(
        python,
        url,
        output,
        "Downloading launcher installer (source) ...",
        verbose,
    )
}

fn github_mirror_url(url: &str) -> String {
    let t = url.trim();
    if t.starts_with("https://github.com/") {
        format!("https://gh-proxy.org/{t}")
    } else {
        t.to_string()
    }
}

fn download_file_with_fallback(
    python: &Path,
    url: &str,
    output: &Path,
    desc: &str,
    verbose: bool,
) -> Result<(), String> {
    if command_ok("wget", &["--version"]) {
        let mut cmd = Command::new("wget");
        cmd.arg("-c")
            .arg("-O")
            .arg(output)
            .arg(url)
            .stdin(Stdio::null());
        match run_cmd_with_optional_spinner(&mut cmd, desc, verbose) {
            Ok(_) => {}
            Err(err) => {
                if should_retry_without_proxy_env(&err) {
                    let mut retry = Command::new("wget");
                    retry
                        .arg("-c")
                        .arg("-O")
                        .arg(output)
                        .arg(url)
                        .stdin(Stdio::null());
                    apply_temporary_no_proxy(&mut retry);
                    let retry_desc = format!("{desc} (retry without proxy) ...");
                    run_cmd_with_optional_spinner(&mut retry, &retry_desc, verbose).map_err(
                        |e| format!("wget download failed: {err}\nRetry without proxy failed: {e}"),
                    )?;
                } else {
                    return Err(format!("wget download failed: {err}"));
                }
            }
        }
        return Ok(());
    }
    if command_ok("curl", &["--version"]) {
        let mut cmd = Command::new("curl");
        cmd.arg("-L")
            .arg("--fail")
            .arg("--continue-at")
            .arg("-")
            .arg("-o")
            .arg(output)
            .arg(url)
            .stdin(Stdio::null());
        match run_cmd_with_optional_spinner(&mut cmd, desc, verbose) {
            Ok(_) => {}
            Err(err) => {
                if should_retry_without_proxy_env(&err) {
                    let mut retry = Command::new("curl");
                    retry
                        .arg("-L")
                        .arg("--fail")
                        .arg("--continue-at")
                        .arg("-")
                        .arg("-o")
                        .arg(output)
                        .arg(url)
                        .stdin(Stdio::null());
                    apply_temporary_no_proxy(&mut retry);
                    let retry_desc = format!("{desc} (retry without proxy) ...");
                    run_cmd_with_optional_spinner(&mut retry, &retry_desc, verbose).map_err(
                        |e| format!("curl download failed: {err}\nRetry without proxy failed: {e}"),
                    )?;
                } else {
                    return Err(format!("curl download failed: {err}"));
                }
            }
        }
        return Ok(());
    }

    let script = r#"
import urllib.request, sys
url = sys.argv[1]
out = sys.argv[2]
urllib.request.urlretrieve(url, out)
"#;
    let mut cmd = Command::new(python);
    cmd.arg("-c")
        .arg(script)
        .arg(url)
        .arg(output)
        .stdin(Stdio::null());
    match run_cmd_with_optional_spinner(&mut cmd, desc, verbose) {
        Ok(_) => {}
        Err(err) => {
            if should_retry_without_proxy_env(&err) {
                let mut retry = Command::new(python);
                retry
                    .arg("-c")
                    .arg(script)
                    .arg(url)
                    .arg(output)
                    .stdin(Stdio::null());
                apply_temporary_no_proxy(&mut retry);
                let retry_desc = format!("{desc} (retry without proxy) ...");
                run_cmd_with_optional_spinner(&mut retry, &retry_desc, verbose).map_err(|e| {
                    format!("Python download failed: {err}\nRetry without proxy failed: {e}")
                })?;
            } else {
                return Err(format!("Python download failed: {err}"));
            }
        }
    }
    Ok(())
}

fn has_http_download_tool() -> bool {
    command_ok("wget", &["--version"]) || command_ok("curl", &["--version"])
}

fn download_file_with_http_tools(
    url: &str,
    output: &Path,
    desc: &str,
    verbose: bool,
) -> Result<(), String> {
    if command_ok("wget", &["--version"]) {
        let mut cmd = Command::new("wget");
        cmd.arg("-c")
            .arg("-O")
            .arg(output)
            .arg(url)
            .stdin(Stdio::null());
        match run_cmd_with_optional_spinner(&mut cmd, desc, verbose) {
            Ok(_) => {}
            Err(err) => {
                if should_retry_without_proxy_env(&err) {
                    let mut retry = Command::new("wget");
                    retry
                        .arg("-c")
                        .arg("-O")
                        .arg(output)
                        .arg(url)
                        .stdin(Stdio::null());
                    apply_temporary_no_proxy(&mut retry);
                    let retry_desc = format!("{desc} (retry without proxy) ...");
                    run_cmd_with_optional_spinner(&mut retry, &retry_desc, verbose).map_err(
                        |e| format!("wget download failed: {err}\nRetry without proxy failed: {e}"),
                    )?;
                } else {
                    return Err(format!("wget download failed: {err}"));
                }
            }
        }
        return Ok(());
    }
    if command_ok("curl", &["--version"]) {
        let mut cmd = Command::new("curl");
        cmd.arg("-L")
            .arg("--fail")
            .arg("--continue-at")
            .arg("-")
            .arg("-o")
            .arg(output)
            .arg(url)
            .stdin(Stdio::null());
        match run_cmd_with_optional_spinner(&mut cmd, desc, verbose) {
            Ok(_) => {}
            Err(err) => {
                if should_retry_without_proxy_env(&err) {
                    let mut retry = Command::new("curl");
                    retry
                        .arg("-L")
                        .arg("--fail")
                        .arg("--continue-at")
                        .arg("-")
                        .arg("-o")
                        .arg(output)
                        .arg(url)
                        .stdin(Stdio::null());
                    apply_temporary_no_proxy(&mut retry);
                    let retry_desc = format!("{desc} (retry without proxy) ...");
                    run_cmd_with_optional_spinner(&mut retry, &retry_desc, verbose).map_err(
                        |e| format!("curl download failed: {err}\nRetry without proxy failed: {e}"),
                    )?;
                } else {
                    return Err(format!("curl download failed: {err}"));
                }
            }
        }
        return Ok(());
    }
    Err("Neither wget nor curl is available.".to_string())
}

fn extract_tar_gz_with_python(
    python: &Path,
    archive: &Path,
    dest: &Path,
    verbose: bool,
) -> Result<(), String> {
    let script = r#"
import tarfile, os, sys
arc = sys.argv[1]
dst = sys.argv[2]
os.makedirs(dst, exist_ok=True)
with tarfile.open(arc, "r:gz") as tf:
    tf.extractall(dst)
"#;
    let mut cmd = Command::new(python);
    cmd.arg("-c")
        .arg(script)
        .arg(archive)
        .arg(dest)
        .stdin(Stdio::null());
    run_cmd_with_optional_spinner(&mut cmd, "Extracting launcher installer ...", verbose)
        .map_err(|e| format!("Failed to extract installer archive: {e}"))?;
    Ok(())
}

fn run_cmd_with_optional_spinner(
    cmd: &mut Command,
    desc: &str,
    verbose: bool,
) -> Result<(), String> {
    if verbose {
        let status = cmd
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .map_err(|e| format!("Failed to run command: {e}"))?;
        if status.success() {
            return Ok(());
        }
        return Err(format!("command failed with exit={}", exit_code(status)));
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let (out, _) = run_with_spinner(cmd, desc)?;
    if out.status.success() {
        return Ok(());
    }
    let mut msg = String::new();
    msg.push_str(&String::from_utf8_lossy(&out.stdout));
    msg.push_str(&String::from_utf8_lossy(&out.stderr));
    Err(format!(
        "command failed with exit={}\n{}",
        exit_code(out.status),
        msg.trim()
    ))
}

fn find_installer_in_tree(root: &Path) -> Option<PathBuf> {
    let expected_ext = if cfg!(windows) {
        "exe"
    } else if cfg!(target_os = "macos") {
        "command"
    } else {
        "run"
    };
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let ext = p
                .extension()
                .and_then(|x| x.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if ext == expected_ext {
                return Some(p);
            }
        }
    }
    None
}

fn run_downloaded_installer(installer: &Path, verbose: bool) -> Result<(), String> {
    #[cfg(windows)]
    {
        let status = Command::new("cmd")
            .arg("/C")
            .arg("start")
            .arg("")
            .arg(installer)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| format!("Failed to launch installer {}: {e}", installer.display()))?;
        if !status.success() {
            return Err(format!(
                "Failed to launch installer {} (exit={})",
                installer.display(),
                exit_code(status)
            ));
        }
        println!("Launched installer: {}", installer.display());
        return Ok(());
    }

    #[cfg(not(windows))]
    {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(installer)
                .map_err(|e| format!("Failed to stat installer {}: {e}", installer.display()))?
                .permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(installer, perms).map_err(|e| {
                format!(
                    "Failed to set executable permission on {}: {e}",
                    installer.display()
                )
            })?;
        }
        let mut cmd = Command::new(installer);
        cmd.stdin(Stdio::inherit());
        run_cmd_with_optional_spinner(&mut cmd, "Running launcher installer ...", verbose)
            .map_err(|e| format!("Installer execution failed: {e}"))?;
        return Ok(());
    }
}

fn run_clean(args: &[String]) -> Result<i32, String> {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help" | "help") {
        println!("Clean usage:\n  jx -clean\n  jx -clean <history_id>");
        return Ok(0);
    }
    if args.len() > 1 {
        return Err("Clean usage:\n  jx -clean\n  jx -clean <history_id>".to_string());
    }
    if args.len() == 1 {
        let python = ensure_runtime(false)?;
        let mut cmd = Command::new(&python);
        cmd.arg("-m")
            .arg("janusx.script.loadanno")
            .arg("--clean-id")
            .arg(&args[0]);
        let status = cmd
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .map_err(|e| format!("Failed to run load cleaner: {e}"))?;
        return Ok(exit_code(status));
    }

    let clean_home = runtime_home().or_else(|_| default_runtime_home())?;
    let uniq: Vec<PathBuf> = vec![clean_home.join(GWAS_HISTORY_DB_FILE)];
    let mut removed: Vec<PathBuf> = Vec::new();
    let mut failed_primary: Vec<String> = Vec::new();
    for p in uniq {
        if !p.exists() {
            continue;
        }
        let res = if p.is_dir() {
            std::fs::remove_dir_all(&p)
        } else {
            std::fs::remove_file(&p)
        };
        match res {
            Ok(_) => removed.push(p),
            Err(e) => failed_primary.push(format!("{} ({e})", p.display())),
        }
    }

    if !failed_primary.is_empty() {
        return Err(format!(
            "Failed to clean GWAS history DB:\n{}",
            failed_primary.join("\n")
        ));
    }

    if removed.is_empty() {
        print_success_line("GWAS history DB already clean.");
    } else {
        print_success_line("GWAS history DB cleaned.");
        for p in removed {
            println!("  removed {}", p.display());
        }
    }
    println!("A new history DB will be created automatically on next `jx gwas`.");
    Ok(0)
}

fn run_uninstall(args: &[String]) -> Result<i32, String> {
    let mut yes = false;
    for token in args {
        match token.as_str() {
            "-y" | "-yes" | "--yes" => yes = true,
            "-h" | "--help" | "help" => {
                println!("Uninstall usage:\n  jx -uninstall [-yes]");
                return Ok(0);
            }
            other => {
                return Err(format!(
                    "Unknown uninstall option: {other}\nUninstall usage:\n  jx -uninstall [-yes]"
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
    let path_remove_status = match remove_install_dir_from_path_persistent(&install_dir) {
        Ok(v) => Some(v),
        Err(e) => {
            failed.push(format!("PATH cleanup ({e})"));
            None
        }
    };
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
    match path_remove_status {
        Some(PathRemoveStatus::Removed) => println!("Removed JanusX from PATH."),
        Some(PathRemoveStatus::NotFound) => println!("JanusX PATH entry was not found."),
        None => {}
    }
    println!("Genotype caches were not removed.");
    Ok(0)
}

fn cleanup_runtime_home(runtime_home: &Path, failed: &mut Vec<String>) {
    if !runtime_home.exists() {
        return;
    }
    if !is_managed_runtime_home_path(runtime_home) {
        failed.push(format!(
            "Refusing to remove runtime outside JanusX/.janusx: {}",
            runtime_home.display()
        ));
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

fn is_managed_runtime_home_path(runtime_home: &Path) -> bool {
    let p = canonical_or_self(runtime_home);
    let Some(name) = p.file_name().and_then(|v| v.to_str()) else {
        return false;
    };
    if !runtime_path_name_eq(name, ".janusx") {
        return false;
    }
    let Some(parent_name) = p
        .parent()
        .and_then(|pp| pp.file_name())
        .and_then(|v| v.to_str())
    else {
        return false;
    };
    runtime_path_name_eq(parent_name, "JanusX")
}

#[cfg(windows)]
fn runtime_path_name_eq(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

#[cfg(not(windows))]
fn runtime_path_name_eq(a: &str, b: &str) -> bool {
    a == b
}

const PROXY_ENV_KEYS: [&str; 8] = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
];

fn has_proxy_env() -> bool {
    PROXY_ENV_KEYS
        .iter()
        .any(|k| env::var(k).map(|v| !v.trim().is_empty()).unwrap_or(false))
}

fn apply_temporary_no_proxy(cmd: &mut Command) {
    for key in PROXY_ENV_KEYS {
        cmd.env_remove(key);
    }
    cmd.env_remove("PIP_PROXY");
    cmd.env_remove("GIT_PROXY_COMMAND");
}

fn proxy_error_like(err: &str) -> bool {
    let e = err.to_ascii_lowercase();
    e.contains("proxy")
        || e.contains("proxyconnect")
        || e.contains("could not connect to server")
        || e.contains("failed to connect to")
        || e.contains("connection refused")
        || e.contains("127.0.0.1 port")
        || e.contains("localhost port")
}

fn should_retry_without_proxy_env(err: &str) -> bool {
    has_proxy_env() && proxy_error_like(err)
}

fn pip_install_update(
    python: &Path,
    runtime_home: &Path,
    spec: &str,
    force_reinstall: bool,
    editable: bool,
    verbose: bool,
    desc: &str,
    use_cn_rust_mirror: bool,
) -> Result<Duration, String> {
    let attempt_install = |force_source_build: bool,
                           suppress_failure_status_line: bool,
                           desc_override: Option<&str>,
                           disable_proxy: bool|
     -> Result<Duration, String> {
        let effective_desc = desc_override.unwrap_or(desc);
        if verbose {
            pip_install(
                python,
                runtime_home,
                spec,
                force_reinstall,
                editable,
                true,
                effective_desc,
                use_cn_rust_mirror,
                force_source_build,
                disable_proxy,
            )
        } else {
            pip_install_tail(
                python,
                runtime_home,
                spec,
                force_reinstall,
                editable,
                effective_desc,
                10,
                use_cn_rust_mirror,
                force_source_build,
                suppress_failure_status_line,
                disable_proxy,
            )
        }
    };

    let attempt_install_with_proxy_retry = |force_source_build: bool,
                                            suppress_failure_status_line: bool,
                                            desc_override: Option<&str>|
     -> Result<Duration, String> {
        match attempt_install(
            force_source_build,
            suppress_failure_status_line,
            desc_override,
            false,
        ) {
            Ok(d) => Ok(d),
            Err(err) => {
                if should_retry_without_proxy_env(&err) {
                    if verbose {
                        eprintln!(
                            "Detected proxy connectivity issue; retrying once with temporary no-proxy environment."
                        );
                    }
                    attempt_install(
                        force_source_build,
                        suppress_failure_status_line,
                        desc_override,
                        true,
                    )
                    .map_err(|retry_err| {
                        format!("{err}\nRetry without proxy failed: {retry_err}")
                    })
                } else {
                    Err(err)
                }
            }
        }
    };

    let is_pypi_spec = is_pypi_janusx_spec(spec);
    // For explicit source installs (GitHub/local path), ensure Rust toolchain upfront.
    // For PyPI, keep the preferred order: wheel first, then source+Rust fallback.
    if spec_requires_rust_build(spec) && !is_pypi_spec {
        if !any_rust_toolchain_ready(runtime_home) {
            if verbose {
                eprintln!(
                    "Rust toolchain not detected for source build; installing local Rust toolchain..."
                );
            }
            ensure_local_rust_toolchain(runtime_home, python, verbose)?;
        }
        #[cfg(target_os = "windows")]
        ensure_windows_msvc_linker_ready(runtime_home)?;
    }

    match attempt_install_with_proxy_retry(false, false, None) {
        Ok(d) => return Ok(d),
        Err(e) => {
            #[cfg(target_os = "windows")]
            if is_windows_linker_missing_error(&e) {
                return Err(format!(
                    "{}\n\n{}",
                    windows_msvc_linker_missing_message(),
                    e
                ));
            }
            if spec_requires_rust_build(spec)
                && !any_rust_toolchain_ready(runtime_home)
                && should_retry_after_installing_rust(&e)
            {
                if verbose {
                    eprintln!("Rust toolchain is required by this update; installing local Rust and retrying.");
                }
                ensure_local_rust_toolchain(runtime_home, python, verbose)?;
                #[cfg(target_os = "windows")]
                ensure_windows_msvc_linker_ready(runtime_home)?;
                return attempt_install_with_proxy_retry(false, false, None);
            }
            if is_pypi_spec && should_retry_with_source_build(&e) {
                if verbose {
                    eprintln!("PyPI wheel install is unavailable; retrying source build.");
                }
                if !any_rust_toolchain_ready(runtime_home) {
                    if verbose {
                        eprintln!(
                            "Rust toolchain not detected; installing local Rust before source build."
                        );
                    }
                    ensure_local_rust_toolchain(runtime_home, python, verbose)?;
                }
                #[cfg(target_os = "windows")]
                ensure_windows_msvc_linker_ready(runtime_home)?;
                match attempt_install_with_proxy_retry(
                    true,
                    false,
                    Some("Retrying source build from PyPI ..."),
                ) {
                    Ok(d) => return Ok(d),
                    Err(e2) => {
                        #[cfg(target_os = "windows")]
                        if is_windows_linker_missing_error(&e2) {
                            return Err(format!(
                                "{}\n\n{}",
                                windows_msvc_linker_missing_message(),
                                e2
                            ));
                        }
                        if !any_rust_toolchain_ready(runtime_home)
                            && should_retry_after_installing_rust(&e2)
                        {
                            if verbose {
                                eprintln!(
                                    "Source build requires Rust toolchain; installing local Rust and retrying."
                                );
                            }
                            ensure_local_rust_toolchain(runtime_home, python, verbose)?;
                            #[cfg(target_os = "windows")]
                            ensure_windows_msvc_linker_ready(runtime_home)?;
                            return attempt_install_with_proxy_retry(
                                true,
                                false,
                                Some("Retrying source build from PyPI ..."),
                            );
                        }
                        return Err(e2);
                    }
                }
            }
            return Err(e);
        }
    }
}

fn should_retry_with_source_build(err: &str) -> bool {
    let e = err.to_ascii_lowercase();
    e.contains("no matching distribution found for janusx")
        || e.contains("could not find a version that satisfies the requirement janusx")
}

fn should_retry_after_installing_rust(err: &str) -> bool {
    let e = err.to_ascii_lowercase();
    e.contains("can't find rust compiler")
        || e.contains("cannot find rust compiler")
        || e.contains("rust compiler")
        || e.contains("cargo, the rust package manager, is not installed")
        || e.contains("is not installed or is not on path")
        || e.contains("no such file or directory (os error 2)") && e.contains("rust")
}

#[cfg(target_os = "windows")]
fn is_editable_local_pyd_copy_failure(err: &str) -> bool {
    let e = err.to_ascii_lowercase();
    // Typical maturin editable failure when in-tree .pyd is locked by another process.
    (e.contains("building editable for janusx")
        || e.contains("--editable")
        || e.contains("editable for janusx"))
        && e.contains("maturin")
        && e.contains("failed to copy")
        && e.contains(".pyd")
}

fn warm_up_after_update(jx_bin: Option<&Path>, runtime_home: &Path) -> Result<(), String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = jx_bin {
        candidates.push(p.to_path_buf());
        if let Some(stripped) = strip_deleted_exe_suffix(p) {
            candidates.push(stripped);
        }
    }
    if let Ok(exe) = env::current_exe() {
        candidates.push(exe.clone());
        if let Some(stripped) = strip_deleted_exe_suffix(&exe) {
            candidates.push(stripped);
        }
    }

    let mut errs: Vec<String> = Vec::new();
    for cand in candidates {
        if !(cand.exists() && cand.is_file()) {
            continue;
        }
        match warm_up_jx(&cand, runtime_home) {
            Ok(()) => return Ok(()),
            Err(e) => errs.push(e),
        }
    }

    // Fallback: warm core directly with runtime python even if no runnable jx binary is found.
    let py = venv_python(&runtime_home.join("venv"));
    if py.exists() {
        let py_status = Command::new(&py)
            .arg("-m")
            .arg("janusx.script.gwas")
            .arg("-h")
            .env("JX_HOME", runtime_home)
            .env(SKIP_RUNTIME_REBUILD_ENV, "1")
            .env(SKIP_WARMUP_ENV, "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| format!("Failed to run `{}` -m janusx.script.gwas -h: {e}", py.display()))?;
        if py_status.success() {
            write_warmup_marker(runtime_home);
            return Ok(());
        }
        errs.push(format!(
            "Warm-up command failed: {} -m janusx.script.gwas -h (exit={})",
            py.display(),
            exit_code(py_status)
        ));
    }

    if errs.is_empty() {
        return Err("Warm-up skipped: no runnable `jx` binary and no runtime python found.".to_string());
    }
    Err(format!("Warm-up failed after update:\n{}", errs.join("\n")))
}

fn strip_deleted_exe_suffix(path: &Path) -> Option<PathBuf> {
    let raw = path.to_string_lossy();
    let suffix = " (deleted)";
    if !raw.ends_with(suffix) {
        return None;
    }
    let trimmed = &raw[..raw.len().saturating_sub(suffix.len())];
    if trimmed.trim().is_empty() {
        return None;
    }
    Some(PathBuf::from(trimmed))
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
    let pypi_spec = resolve_pypi_spec(&home);
    let python = ensure_venv()?;
    if !is_janusx_installed(&python) {
        let _ = std::fs::remove_file(warmup_marker_path(&home));
        if verbose_bootstrap {
            println!("Bootstrapping JanusX from PyPI...");
        }
        let _ = pip_install_update(
            &python,
            &home,
            &pypi_spec,
            false,
            false,
            verbose_bootstrap,
            "Building runtime from PyPI ...",
            true,
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
        std::fs::create_dir_all(&home)
            .map_err(|e| format!("Failed to create runtime directory {}: {e}", home.display()))?;
    }
    let venv = home.join("venv");
    let py = venv_python(&venv);
    if py.exists() {
        if !python_path_meets_min_version(&py) {
            println!(
                "Runtime Python is below {}.{}; rebuilding venv ...",
                MIN_PYTHON_MAJOR, MIN_PYTHON_MINOR
            );
            std::fs::remove_dir_all(&venv)
                .map_err(|e| format!("Failed to remove old venv {}: {e}", venv.display()))?;
        } else {
            ensure_pip_in_venv(&py)?;
            return Ok(py);
        }
    }
    if py.exists() {
        ensure_pip_in_venv(&py)?;
        return Ok(py);
    }

    let sys_py = find_system_python().ok_or_else(|| python_install_hint())?;
    let mut cmd = Command::new(&sys_py);
    cmd.arg("-m")
        .arg("venv")
        .arg("--without-pip")
        .arg(&venv)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let (out, elapsed) = run_with_spinner(&mut cmd, "Creating runtime venv ...")
        .map_err(|e| format!("Failed to create venv with `{sys_py}`: {e}"))?;
    if !out.status.success() {
        let mut msg = String::new();
        msg.push_str(&String::from_utf8_lossy(&out.stdout));
        msg.push_str(&String::from_utf8_lossy(&out.stderr));
        return Err(format!(
            "Failed to create venv at {} (exit={}).\n{}",
            venv.display(),
            exit_code(out.status),
            msg.trim()
        ));
    }
    print_success_line(&format!(
        "Creating runtime venv ...[{}]",
        format_elapsed(elapsed)
    ));
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

    let mut cmd = Command::new(python);
    cmd.arg("-m")
        .arg("ensurepip")
        .arg("--upgrade")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let (out, elapsed) = run_with_spinner(&mut cmd, "Bootstrapping pip in venv ...")
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
    print_success_line(&format!(
        "Bootstrapping pip in venv ...[{}]",
        format_elapsed(elapsed)
    ));

    Ok(())
}

fn should_rebuild_runtime(home: &Path) -> Result<bool, String> {
    // Auto rebuild is disabled by default to avoid implicit update/reinstall side effects.
    // Opt-in with JX_AUTO_RUNTIME_REBUILD=1 when forced rebuild is desired.
    let auto = env::var("JX_AUTO_RUNTIME_REBUILD")
        .ok()
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    if !auto {
        return Ok(false);
    }

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
        if python_bin_meets_min_version(&val) {
            return Some(val);
        }
    }

    #[cfg(windows)]
    let candidates: &[(&str, &[&str])] = &[
        ("python3.14", &[]),
        ("python3.13", &[]),
        ("python3.12", &[]),
        ("python3.11", &[]),
        ("python3.10", &[]),
        ("python3.9", &[]),
        ("python3", &[]),
        ("py", &[]),
        ("python", &[]),
    ];

    #[cfg(not(windows))]
    let candidates: &[(&str, &[&str])] = &[
        ("python3.14", &[]),
        ("python3.13", &[]),
        ("python3.12", &[]),
        ("python3.11", &[]),
        ("python3.10", &[]),
        ("python3.9", &[]),
        ("python3", &[]),
        // Safe fallback for Unix environments that only expose `python`.
        // python_bin_meets_min_version_with_prefix() still enforces >= MIN_PYTHON.
        ("python", &[]),
    ];

    for (bin, prefix) in candidates {
        if python_bin_meets_min_version_with_prefix(bin, prefix) {
            return Some((*bin).to_string());
        }
    }
    None
}

fn python_bin_meets_min_version(bin: &str) -> bool {
    python_bin_meets_min_version_with_prefix(bin, &[])
}

fn python_bin_meets_min_version_with_prefix(bin: &str, prefix: &[&str]) -> bool {
    python_version_with_prefix(bin, prefix)
        .map(python_version_is_compatible)
        .unwrap_or(false)
}

fn python_path_meets_min_version(python: &Path) -> bool {
    python_version_for_path(python)
        .map(python_version_is_compatible)
        .unwrap_or(false)
}

fn python_version_is_compatible(v: (u32, u32)) -> bool {
    v.0 > MIN_PYTHON_MAJOR || (v.0 == MIN_PYTHON_MAJOR && v.1 >= MIN_PYTHON_MINOR)
}

fn python_version_with_prefix(bin: &str, prefix: &[&str]) -> Option<(u32, u32)> {
    let mut cmd = Command::new(bin);
    cmd.args(prefix)
        .arg("-c")
        .arg("import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    parse_python_version(&String::from_utf8_lossy(&out.stdout))
}

fn python_version_for_path(python: &Path) -> Option<(u32, u32)> {
    let out = Command::new(python)
        .arg("-c")
        .arg("import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_python_version(&String::from_utf8_lossy(&out.stdout))
}

fn parse_python_version(s: &str) -> Option<(u32, u32)> {
    let t = s.trim();
    let mut it = t.split('.');
    let major = it.next()?.parse::<u32>().ok()?;
    let minor = it.next()?.parse::<u32>().ok()?;
    Some((major, minor))
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

fn ready_core_python(runtime_home: &Path) -> Option<PathBuf> {
    let py = venv_python(&runtime_home.join("venv"));
    if !py.exists() {
        return None;
    }
    if !python_path_meets_min_version(&py) {
        return None;
    }
    if !is_janusx_installed(&py) {
        return None;
    }
    Some(py)
}

fn print_version(core_python: Option<&Path>, runtime_home: &Path) {
    let launcher_version = normalize_version_token(LAUNCHER_CARGO_VERSION)
        .unwrap_or_else(|| LAUNCHER_CARGO_VERSION.to_string());
    let launcher_time =
        launcher_build_time(core_python, runtime_home).unwrap_or_else(|| "unknown".to_string());
    let core_version = core_python.and_then(installed_version);
    let core_time = core_python.and_then(|py| python_core_update_time(py, runtime_home));

    println!("{LOGO}");
    let title_version = core_version
        .clone()
        .unwrap_or_else(|| launcher_version.clone());
    if supports_color() {
        println!(
            "{}",
            style_white(&format!("JanusX v{title_version}, {VERSION_AUTHOR}"))
        );
        println!("{}", style_white(&format!("Contact: <{VERSION_CONTACT}>")));
        println!(
            "{} {} {}",
            style_green("Launcher:"),
            style_white(&format!("v{launcher_version}")),
            style_white(&launcher_time)
        );
        if let Some(core_ver) = &core_version {
            println!(
                "{} {} {}",
                style_green("Core packages:"),
                style_white(&format!("v{core_ver}")),
                style_white(core_time.as_deref().unwrap_or("unknown"))
            );
            if let Some(warn) = version_mismatch_hint(&launcher_version, core_ver) {
                println!("{}", style_yellow(&format!("Warning: {warn}")));
            }
        } else {
            println!("{}", style_green("Core packages: not installed"));
        }
        println!();
        return;
    }
    println!("JanusX v{title_version}, {VERSION_AUTHOR}");
    println!("Contact: <{VERSION_CONTACT}>");
    println!("Launcher: v{launcher_version} {launcher_time}");
    if let Some(core_ver) = &core_version {
        println!(
            "Core packages: v{core_ver} {}",
            core_time.as_deref().unwrap_or("unknown")
        );
        if let Some(warn) = version_mismatch_hint(&launcher_version, core_ver) {
            println!("Warning: {warn}");
        }
    } else {
        println!("Core packages: not installed");
    }
    println!();
}

fn version_mismatch_hint(launcher_version: &str, core_version: &str) -> Option<String> {
    match compare_version_tokens(launcher_version, core_version) {
        std::cmp::Ordering::Equal => None,
        std::cmp::Ordering::Less => Some(
            "launcher version is lower than core packages. Please run `jx -upgrade`.".to_string(),
        ),
        std::cmp::Ordering::Greater => Some(
            "core packages version is lower than launcher. Please run `jx -update latest`."
                .to_string(),
        ),
    }
}

fn launcher_build_time(core_python: Option<&Path>, _runtime_home: &Path) -> Option<String> {
    let exe = env::current_exe().ok()?;
    let exe_s = exe.to_string_lossy().to_string();
    let py = match core_python {
        Some(v) => v.to_path_buf(),
        None => {
            let sys = find_system_python()?;
            PathBuf::from(sys)
        }
    };
    let out = Command::new(&py)
        .arg("-c")
        .arg("import datetime, pathlib, sys; p=pathlib.Path(sys.argv[1]); print(datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M'))")
        .arg(exe_s)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if out.status.success() {
        let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
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

fn is_rust_core_change(path: &str) -> bool {
    let p = path.trim();
    if p.is_empty() {
        return false;
    }
    if p.starts_with("src/") {
        return true;
    }
    matches!(p, "Cargo.toml" | "Cargo.lock" | "build.rs")
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
    if changed.iter().any(|p| is_rust_core_change(p)) {
        cleanup();
        return Ok(false);
    }

    let has_python_change = changed.iter().any(|p| p.starts_with("python/"));
    if !has_python_change {
        write_commit_marker(runtime_home, remote_commit);
        write_python_core_update_marker(runtime_home, python);
        cleanup();
        return Ok(true);
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

const SPINNER_FRAMES: [&str; 4] = ["/", "-", "\\", "|"];

pub(crate) fn spinner_refresh_interval(elapsed: Duration) -> Duration {
    if elapsed < Duration::from_secs(60) {
        Duration::from_millis(100)
    } else {
        Duration::from_secs(1)
    }
}

pub(crate) fn format_elapsed_live(elapsed: Duration) -> String {
    let secs = elapsed.as_secs_f64();
    if secs < 60.0 {
        let tenths = (secs * 10.0).floor() / 10.0;
        return format!("{tenths:.1}s");
    }
    if secs < 3600.0 {
        let total = secs.floor() as u64;
        let m = total / 60;
        let s = total % 60;
        return format!("{m}m{s:02}s");
    }
    let total_min = (secs / 60.0).floor() as u64;
    let h = total_min / 60;
    let m = total_min % 60;
    format!("{h}h{m:02}m")
}

pub(crate) fn spinner_frame_for_elapsed(elapsed: Duration) -> &'static str {
    let step_ms = spinner_refresh_interval(elapsed).as_millis().max(1);
    let idx = ((elapsed.as_millis() / step_ms) as usize) % SPINNER_FRAMES.len();
    SPINNER_FRAMES[idx]
}

fn print_success_line(msg: &str) {
    let tty = io::stdout().is_terminal();
    if tty && supports_color() {
        println!("\r\x1b[2K\x1b[32m✔︎ {}\x1b[0m", msg);
    } else if tty {
        println!("\r\x1b[2K✔︎ {}", msg);
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

fn style_yellow(text: &str) -> String {
    if supports_color() {
        format!("\x1b[33m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

fn style_blue(text: &str) -> String {
    if supports_color() {
        format!("\x1b[34m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

fn style_orange(text: &str) -> String {
    if supports_color() {
        format!("\x1b[38;5;208m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

fn style_white(text: &str) -> String {
    if supports_color() {
        format!("\x1b[37m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

fn print_cli_help() {
    let width = help_line_width();
    let flag_key_width = 12usize;
    println!("{LOGO}");
    println!("{}", style_orange("Usage:"));
    println!("  {}", style_green("jx [flags]"));
    println!("  {}", style_green("jx [module] -h"));
    println!();

    println!("{}", style_orange("Flags:"));
    print_help_entry(2, "-h", "Show this help message", flag_key_width, width);
    print_help_entry(
        2,
        "-v",
        "Show version/build information",
        flag_key_width,
        width,
    );
    print_help_entry(
        2,
        "-update",
        "Update JanusX: `jx -update [dlc|latest|<local_path>] [-e|-editable] [-verbose]`",
        flag_key_width,
        width,
    );
    print_help_entry(
        2,
        "-upgrade",
        "Upgrade launcher from source, then update core (`latest` or `-e <local_path>`): `jx -upgrade [latest|<local_path>] [-verbose]`",
        flag_key_width,
        width,
    );
    print_help_entry(
        2,
        "-list",
        "List available modules or DLC tools: `jx -list [module|dlc]`",
        flag_key_width,
        width,
    );
    print_help_entry(
        2,
        "-clean",
        "Clear GWAS history DB, or `jx -clean <history_id>` to remove one history record",
        flag_key_width,
        width,
    );
    print_help_entry(
        2,
        "-uninstall",
        "Remove JanusX runtime and launcher files",
        flag_key_width,
        width,
    );
    println!();

    println!("{}", style_orange("Modules:"));
    println!(
        "  {}",
        style_blue("Genome-wide Association Studies (GWAS):")
    );
    print_help_entry(4, "grm", "Build genomic relationship matrix", 12, width);
    print_help_entry(
        4,
        "pca",
        "Principal component analysis for population structure",
        12,
        width,
    );
    print_help_entry(4, "gwas", "Run genome-wide association analysis", 12, width);
    print_help_entry(
        4,
        "postgwas",
        "Post-process GWAS results and downstream plots",
        12,
        width,
    );
    println!();
    println!("  {}", style_blue("Genomic Selection (GS):"));
    print_help_entry(
        4,
        "gs",
        "Genomic prediction and model-based selection",
        12,
        width,
    );
    print_help_entry(
        4,
        "reml",
        "Estimate heritability/effect components by REML-BLUP",
        12,
        width,
    );
    println!();
    println!(
        "  {}",
        style_blue("Genetic Association by Random Forest and InterpretivE Logic Decisions (GARFIELD):")
    );
    print_help_entry(
        4,
        "garfield",
        "Random-forest based marker-trait association",
        12,
        width,
    );
    print_help_entry(
        4,
        "postgarfield",
        "Summarize and visualize GARFIELD outputs",
        12,
        width,
    );
    println!();
    println!("  {}", style_blue("Bulk Segregation Analysis (BSA):"));
    print_help_entry(
        4,
        "postbsa",
        "Post-process and visualize BSA results",
        12,
        width,
    );
    println!();
    println!("  {}", style_blue("Pipeline and utility:"));
    print_help_entry(
        4,
        "fastq2count",
        "RNA-seq pipeline from FASTQ to gene count/FPKM/TPM",
        12,
        width,
    );
    print_help_entry(
        4,
        "fastq2vcf",
        "Variant-calling pipeline from FASTQ to VCF",
        12,
        width,
    );
    print_help_entry(4, "adamixture", "ADAMIXTURE ancestry inference", 12, width);
    print_help_entry(
        4,
        "hybrid",
        "Build pairwise hybrid genotype matrix from parent lists",
        12,
        width,
    );
    print_help_entry(
        4,
        "gformat",
        "Convert genotype files across plink/vcf/txt/npy",
        12,
        width,
    );
    print_help_entry(4, "gmerge", "Merge genotype/variant tables", 12, width);
    print_help_entry(
        4,
        "webui",
        "Start JanusX web UI (postgwas first)",
        12,
        width,
    );
    println!();
    println!("  {}", style_blue("Benchmark:"));
    print_help_entry(4, "sim", "Quick simulation workflow", 12, width);
    print_help_entry(
        4,
        "simulation",
        "Extended simulation and benchmarking workflow",
        12,
        width,
    );
}

fn run_list(args: &[String]) -> Result<i32, String> {
    if args.len() > 1 {
        return Err("List usage:\n  jx -list [module|dlc]".to_string());
    }
    let target = args
        .first()
        .map(|x| x.to_ascii_lowercase())
        .unwrap_or_else(|| "module".to_string());
    match target.as_str() {
        "module" | "modules" => {
            print_module_list();
            Ok(0)
        }
        "dlc" => print_dlc_list().map(|_| 0),
        _ => Err(format!(
            "Unknown list target: {target}\nList usage:\n  jx -list [module|dlc]"
        )),
    }
}

fn print_module_list() {
    let width = help_line_width();
    println!("{}", style_orange("Available Modules:"));
    for (name, desc) in MODULE_LIST_ENTRIES {
        print_help_entry(2, name, desc, 14, width);
    }
}

fn print_dlc_list() -> Result<(), String> {
    let home = runtime_home()?;
    let tools = dlc::list_dlc_tool_locators(&home)?;
    println!("{}", style_orange("Available External Tools (DLC):"));
    for item in tools {
        let head = format!("{:<14}", item.tool);
        let route = format_dlc_route_for_display(&item.route);
        let line = format!("{head} {route}");
        if item.ready {
            println!("  {}", style_green(&line));
        } else {
            println!("  {}", style_yellow(&line));
        }
    }
    Ok(())
}

fn format_dlc_route_for_display(route: &str) -> String {
    let (backend, locator) = match route.split_once(':') {
        Some((b, l)) => (b.trim(), l.trim()),
        None => ("none", route.trim()),
    };
    match backend {
        "host" => {
            if locator.is_empty() || locator == "-" {
                "host:-".to_string()
            } else {
                format!("host:{locator}")
            }
        }
        "conda" => {
            let env_name = locator
                .split_once('|')
                .map(|(n, _)| n.trim())
                .unwrap_or_default();
            if env_name.is_empty() {
                "conda".to_string()
            } else {
                format!("conda:{env_name}")
            }
        }
        "docker" => format!("docker:{locator}"),
        "sif" => format!("sif:{locator}"),
        "env" => format!("env:{locator}"),
        _ => route.to_string(),
    }
}

fn print_help_entry(indent: usize, key: &str, desc: &str, key_width: usize, total_width: usize) {
    let lead = " ".repeat(indent);
    let min_desc_width = 20usize;
    let key_padded = format!("{key:<key_width$}");
    let min_total = indent + key_width + 2 + min_desc_width;
    if total_width <= min_total {
        println!("{}{}", lead, style_green(key));
        for line in wrap_help_text(
            desc,
            total_width.saturating_sub(indent + 2).max(min_desc_width),
        ) {
            println!("{}  {}", lead, style_white(&line));
        }
        return;
    }

    let desc_width = total_width - indent - key_width - 2;
    let wrapped = wrap_help_text(desc, desc_width);
    if wrapped.is_empty() {
        println!("{}{}", lead, style_green(&key_padded));
        return;
    }
    println!(
        "{}{}  {}",
        lead,
        style_green(&key_padded),
        style_white(&wrapped[0])
    );
    let pad = " ".repeat(key_width);
    for line in wrapped.iter().skip(1) {
        println!("{}{}  {}", lead, pad, style_white(line));
    }
}

fn help_line_width() -> usize {
    let cols = detect_help_terminal_columns().unwrap_or(100);
    cols.saturating_sub(2).clamp(48, 160)
}

fn detect_help_terminal_columns() -> Option<usize> {
    if let Ok(v) = env::var("COLUMNS") {
        if let Ok(cols) = v.parse::<usize>() {
            if cols > 0 {
                return Some(cols);
            }
        }
    }

    if io::stdout().is_terminal() {
        if let Some(cols) = query_stty_columns() {
            return Some(cols);
        }
        if let Some(cols) = query_tput_columns() {
            return Some(cols);
        }
    }
    None
}

fn query_stty_columns() -> Option<usize> {
    if !io::stdin().is_terminal() {
        return None;
    }
    let out = Command::new("stty")
        .arg("size")
        .stdin(Stdio::inherit())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let cols = text
        .split_whitespace()
        .nth(1)
        .and_then(|x| x.parse::<usize>().ok())?;
    (cols > 0).then_some(cols)
}

fn query_tput_columns() -> Option<usize> {
    let out = Command::new("tput")
        .arg("cols")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let cols = text.trim().parse::<usize>().ok()?;
    (cols > 0).then_some(cols)
}

fn wrap_help_text(text: &str, width: usize) -> Vec<String> {
    let w = width.max(8);
    let mut out: Vec<String> = Vec::new();
    let mut line = String::new();

    for word in text.split_whitespace() {
        let len_word = word.chars().count();
        if line.is_empty() {
            if len_word <= w {
                line.push_str(word);
            } else {
                let chunks = split_long_word(word, w);
                if chunks.is_empty() {
                    continue;
                }
                for ch in chunks.iter().take(chunks.len().saturating_sub(1)) {
                    out.push(ch.to_string());
                }
                line = chunks.last().cloned().unwrap_or_default();
            }
            continue;
        }

        let len_line = line.chars().count();
        if len_line + 1 + len_word <= w {
            line.push(' ');
            line.push_str(word);
            continue;
        }

        out.push(std::mem::take(&mut line));
        if len_word <= w {
            line.push_str(word);
        } else {
            let chunks = split_long_word(word, w);
            for ch in chunks.iter().take(chunks.len().saturating_sub(1)) {
                out.push(ch.to_string());
            }
            line = chunks.last().cloned().unwrap_or_default();
        }
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

fn split_long_word(word: &str, width: usize) -> Vec<String> {
    let w = width.max(1);
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut n = 0usize;
    for ch in word.chars() {
        cur.push(ch);
        n += 1;
        if n >= w {
            out.push(std::mem::take(&mut cur));
            n = 0;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn run_with_spinner(cmd: &mut Command, desc: &str) -> Result<(Output, Duration), String> {
    let start = Instant::now();
    let is_tty = io::stdout().is_terminal();
    let done = Arc::new(AtomicBool::new(false));
    let done_c = Arc::clone(&done);
    let desc_s = desc.to_string();

    let spinner = if is_tty {
        Some(thread::spawn(move || {
            while !done_c.load(Ordering::Relaxed) {
                let elapsed = start.elapsed();
                let line = format!(
                    "\r{} {} [{}]",
                    spinner_frame_for_elapsed(elapsed),
                    desc_s,
                    format_elapsed_live(elapsed)
                );
                if supports_color() {
                    print!("{}", style_green(&line));
                } else {
                    print!("{line}");
                }
                let _ = io::stdout().flush();
                thread::sleep(spinner_refresh_interval(elapsed));
            }
            print!("\r\x1b[2K");
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
    editable: bool,
    verbose: bool,
    desc: &str,
    use_cn_rust_mirror: bool,
    force_source_build: bool,
    disable_proxy: bool,
) -> Result<Duration, String> {
    let mut cmd = build_pip_install_cmd(
        python,
        runtime_home,
        spec,
        force_reinstall,
        editable,
        use_cn_rust_mirror,
        force_source_build,
    );
    if disable_proxy {
        apply_temporary_no_proxy(&mut cmd);
    }

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

fn build_pip_install_cmd(
    python: &Path,
    runtime_home: &Path,
    spec: &str,
    force_reinstall: bool,
    editable: bool,
    use_cn_rust_mirror: bool,
    force_source_build: bool,
) -> Command {
    let mut cmd = Command::new(python);
    cmd.arg("-m")
        .arg("pip")
        .arg("install")
        .arg("--upgrade")
        .arg("--disable-pip-version-check");
    if force_source_build || should_force_source_build_for_spec(spec) {
        cmd.arg("--no-binary").arg("janusx");
    } else if is_pypi_janusx_spec(spec) {
        // Prefer wheel-only first for PyPI installs.
        // If wheel is unavailable, pip_install_update() will fall back to source build.
        cmd.arg("--only-binary")
            .arg("janusx")
            .arg("--prefer-binary");
    }
    if force_reinstall {
        cmd.arg("--force-reinstall").arg("--no-cache-dir");
    }
    if spec_requires_rust_build(spec) && local_rust_toolchain_ready(runtime_home) {
        apply_local_rust_env(&mut cmd, runtime_home, use_cn_rust_mirror);
    }
    if editable {
        cmd.arg("-e").arg(spec);
    } else {
        cmd.arg(spec);
    }
    cmd
}

fn should_force_source_build_for_spec(spec: &str) -> bool {
    let _ = spec;
    false
}

fn is_pypi_janusx_spec(spec: &str) -> bool {
    let s = spec.trim().to_ascii_lowercase();
    s == "janusx" || s.starts_with("janusx==")
}

fn spec_requires_rust_build(spec: &str) -> bool {
    let s = spec.trim();
    if s.starts_with("git+") {
        return true;
    }
    if Path::new(s).exists() {
        return true;
    }
    should_force_source_build_for_spec(spec)
}

fn pip_install_tail(
    python: &Path,
    runtime_home: &Path,
    spec: &str,
    force_reinstall: bool,
    editable: bool,
    desc: &str,
    max_lines: usize,
    use_cn_rust_mirror: bool,
    force_source_build: bool,
    suppress_failure_status_line: bool,
    disable_proxy: bool,
) -> Result<Duration, String> {
    let is_tty = io::stdout().is_terminal();
    let mut cmd = build_pip_install_cmd(
        python,
        runtime_home,
        spec,
        force_reinstall,
        editable,
        use_cn_rust_mirror,
        force_source_build,
    );
    if disable_proxy {
        apply_temporary_no_proxy(&mut cmd);
    }
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
    let mut tail_mode_started = false;
    let mut prelog_spinner_shown = false;
    let mut streaming_title_started = false;
    let mut out_text = String::new();
    let mut err_text = String::new();
    let mut tail: VecDeque<String> = VecDeque::with_capacity(max_lines.max(1));
    let mut channel_open = true;
    let mut exit_status: Option<ExitStatus> = None;
    let mut last_render = Instant::now();
    let mut last_line_seen: Option<String> = None;
    let mut streamed_lines_before_tail = 0usize;
    let mut saw_any_line = false;

    if is_tty {
        render_prelog_spinner_line(desc, start.elapsed())?;
        prelog_spinner_shown = true;
        last_render = Instant::now();
    }

    while channel_open {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok((is_err, line)) => {
                if is_tty && prelog_spinner_shown && !tail_mode_started {
                    if !streaming_title_started {
                        render_streaming_title_init(desc, start.elapsed())?;
                        streaming_title_started = true;
                    }
                    prelog_spinner_shown = false;
                }
                saw_any_line = true;
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
                        tail.push_back(line.clone());
                    }
                }
                if is_tty {
                    if max_lines == 0 {
                        tail_mode_started = true;
                        render_tail_block(desc, "", start.elapsed(), &tail, max_lines, rendered)?;
                        rendered = true;
                        last_render = Instant::now();
                    } else if !tail_mode_started {
                        if !is_duplicate {
                            if tail.len() < max_lines {
                                let width = terminal_line_width();
                                let trimmed = truncate_plain_line(&line, width);
                                if supports_color() {
                                    println!("\x1b[2m{}\x1b[0m", trimmed);
                                } else {
                                    println!("{trimmed}");
                                }
                                streamed_lines_before_tail += 1;
                                io::stdout().flush().map_err(|e| {
                                    format!("Failed to flush pip progress output: {e}")
                                })?;
                            } else {
                                // Switch from normal streaming log to fixed tail-refresh mode.
                                if streamed_lines_before_tail > 0 {
                                    let mut up = streamed_lines_before_tail;
                                    if streaming_title_started {
                                        up = up.saturating_add(1);
                                    }
                                    print!("\x1b[{}A", up);
                                }
                                render_tail_block(
                                    desc,
                                    "",
                                    start.elapsed(),
                                    &tail,
                                    max_lines,
                                    false,
                                )?;
                                rendered = true;
                                tail_mode_started = true;
                                last_render = Instant::now();
                            }
                        }
                    } else {
                        render_tail_block(desc, "", start.elapsed(), &tail, max_lines, rendered)?;
                        rendered = true;
                        last_render = Instant::now();
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if exit_status.is_none() {
                    exit_status = child
                        .try_wait()
                        .map_err(|e| format!("Failed to poll pip install status: {e}"))?;
                }
                let elapsed = start.elapsed();
                let refresh_every = spinner_refresh_interval(elapsed);
                if is_tty
                    && !tail_mode_started
                    && !saw_any_line
                    && last_render.elapsed() >= refresh_every
                {
                    render_prelog_spinner_line(desc, elapsed)?;
                    prelog_spinner_shown = true;
                    last_render = Instant::now();
                }
                if is_tty
                    && !tail_mode_started
                    && streaming_title_started
                    && last_render.elapsed() >= refresh_every
                {
                    render_streaming_title_only(desc, elapsed, streamed_lines_before_tail)?;
                    last_render = Instant::now();
                }
                if is_tty && tail_mode_started && last_render.elapsed() >= refresh_every {
                    render_tail_title_only(desc, elapsed, max_lines)?;
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
        if prelog_spinner_shown {
            print!("\r\x1b[2K");
            io::stdout()
                .flush()
                .map_err(|e| format!("Failed to flush pip progress output: {e}"))?;
        }
        let should_render_failure = !suppress_failure_status_line;
        if status.success() || should_render_failure {
            let final_symbol = if status.success() { "✔︎" } else { "✘" };
            if tail_mode_started {
                if status.success() {
                    render_tail_success_compact(desc, start.elapsed(), max_lines)?;
                } else {
                    render_tail_block(
                        desc,
                        final_symbol,
                        start.elapsed(),
                        &tail,
                        max_lines,
                        rendered,
                    )?;
                }
            } else {
                if !status.success() && streaming_title_started {
                    render_streaming_failure_compact(
                        desc,
                        start.elapsed(),
                        streamed_lines_before_tail,
                    )?;
                } else {
                    let width = terminal_line_width();
                    let title = truncate_plain_line(
                        &format!(
                            "{final_symbol} {desc}[{}]",
                            format_elapsed_live(start.elapsed())
                        ),
                        width,
                    );
                    if status.success() {
                        println!("{}", style_green(&title));
                    } else {
                        println!("{}", style_yellow(&title));
                    }
                    io::stdout()
                        .flush()
                        .map_err(|e| format!("Failed to flush pip progress output: {e}"))?;
                }
            }
        } else if tail_mode_started {
            clear_tail_title_only(max_lines)?;
        } else {
            print!("\r\x1b[2K");
            io::stdout()
                .flush()
                .map_err(|e| format!("Failed to flush pip progress output: {e}"))?;
        }
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
    if symbol.is_empty() {
        let title = truncate_plain_line(
            &format!(
                "{} {}[{}]",
                spinner_frame_for_elapsed(elapsed),
                desc,
                format_elapsed_live(elapsed)
            ),
            width,
        );
        print!("\x1b[2K\r{}\n", style_green(&title));
    } else {
        let title = truncate_plain_line(
            &format!("{symbol} {desc}[{}]", format_elapsed_live(elapsed)),
            width,
        );
        if symbol == "✘" {
            print!("\x1b[2K\r{}\n", style_yellow(&title));
        } else {
            print!("\x1b[2K\r{}\n", style_green(&title));
        }
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

fn render_tail_title_only(desc: &str, elapsed: Duration, max_lines: usize) -> Result<(), String> {
    let width = terminal_line_width();
    let title = truncate_plain_line(
        &format!(
            "{} {}[{}]",
            spinner_frame_for_elapsed(elapsed),
            desc,
            format_elapsed_live(elapsed)
        ),
        width,
    );
    // Cursor is at the bottom of the fixed block; move to title row, rewrite title only, then return.
    print!("\x1b[{}A", max_lines.saturating_add(1));
    print!("\x1b[2K\r{}\n", style_green(&title));
    print!("\x1b[{}B", max_lines);
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush pip progress output: {e}"))?;
    Ok(())
}

fn render_streaming_failure_compact(
    desc: &str,
    elapsed: Duration,
    lines_below: usize,
) -> Result<(), String> {
    let width = terminal_line_width();
    let title = truncate_plain_line(
        &format!("✘ {desc}[{}]", format_elapsed_live(elapsed)),
        width,
    );
    // Move to title row, rewrite as failure, then delete streamed log rows entirely.
    let up = lines_below.saturating_add(1);
    print!("\x1b[{}A", up);
    print!("\x1b[2K\r{}\n", style_yellow(&title));
    if lines_below > 0 {
        print!("\x1b[{}M", lines_below);
    }
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush pip progress output: {e}"))?;
    Ok(())
}

fn render_tail_success_compact(
    desc: &str,
    elapsed: Duration,
    max_lines: usize,
) -> Result<(), String> {
    let width = terminal_line_width();
    let title = truncate_plain_line(
        &format!("✔︎ {desc}[{}]", format_elapsed_live(elapsed)),
        width,
    );
    // Cursor is currently below the fixed block.
    // Move to title row, print success title, then delete tail log rows entirely.
    print!("\x1b[{}A", max_lines.saturating_add(1));
    print!("\x1b[2K\r{}\n", style_green(&title));
    if max_lines > 0 {
        print!("\x1b[{}M", max_lines);
    }
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush pip progress output: {e}"))?;
    Ok(())
}

fn clear_tail_title_only(max_lines: usize) -> Result<(), String> {
    // Cursor is currently below the fixed block.
    // Move to title row, clear it, keep logs, then return to bottom.
    print!("\x1b[{}A", max_lines.saturating_add(1));
    print!("\x1b[2K\r\n");
    if max_lines > 0 {
        print!("\x1b[{}B", max_lines);
    }
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush pip progress output: {e}"))?;
    Ok(())
}

fn render_prelog_spinner_line(desc: &str, elapsed: Duration) -> Result<(), String> {
    let width = terminal_line_width();
    let title = truncate_plain_line(
        &format!(
            "{} {}[{}]",
            spinner_frame_for_elapsed(elapsed),
            desc,
            format_elapsed_live(elapsed)
        ),
        width,
    );
    if supports_color() {
        print!("\r{}", style_green(&title));
    } else {
        print!("\r{title}");
    }
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush pip progress output: {e}"))?;
    Ok(())
}

fn render_streaming_title_init(desc: &str, elapsed: Duration) -> Result<(), String> {
    let width = terminal_line_width();
    let title = truncate_plain_line(
        &format!(
            "{} {}[{}]",
            spinner_frame_for_elapsed(elapsed),
            desc,
            format_elapsed_live(elapsed)
        ),
        width,
    );
    print!("\r\x1b[2K{}\n", style_green(&title));
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush pip progress output: {e}"))?;
    Ok(())
}

fn render_streaming_title_only(
    desc: &str,
    elapsed: Duration,
    lines_below: usize,
) -> Result<(), String> {
    let width = terminal_line_width();
    let title = truncate_plain_line(
        &format!(
            "{} {}[{}]",
            spinner_frame_for_elapsed(elapsed),
            desc,
            format_elapsed_live(elapsed)
        ),
        width,
    );
    let up = lines_below.saturating_add(1);
    print!("\x1b[{}A", up);
    print!("\x1b[2K\r{}\n", style_green(&title));
    if lines_below > 0 {
        print!("\x1b[{}B", lines_below);
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
        println!("{line}");
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
        return "Git is required for `jx -update latest`, but it was not found in PATH.\n\
Windows install options:\n\
  1) winget: winget install --id Git.Git -e\n\
  2) choco:  choco install git -y\n\
  3) installer: https://git-scm.com/download/win"
            .to_string();
    }
    if cfg!(target_os = "macos") {
        return "Git is required for `jx -update latest`, but it was not found in PATH.\n\
macOS install options:\n\
  1) xcode-select --install\n\
  2) brew install git"
            .to_string();
    }
    "Git is required for `jx -update latest`, but it was not found in PATH.\n\
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
  1) rm -rf ~/JanusX/.janusx/venv\n\
  2) Re-run installer or `jx -h`\n\
If system Python lacks ensurepip, install python via Homebrew: `brew install python`."
            .to_string();
    }
    "Try recreating runtime venv with a Python that includes ensurepip.\n\
Suggested fix:\n\
  1) rm -rf ~/JanusX/.janusx/venv\n\
  2) Re-run installer or `jx -h`"
        .to_string()
}

fn python_install_hint() -> String {
    if cfg!(windows) {
        return "No compatible Python found. JanusX requires Python >=3.9.\n\
Install Python 3.9+ first, or set JX_PYTHON.\n\
Note: a venv cannot upgrade Python major/minor by itself.\n\
Windows examples:\n\
  winget install Python.Python.3.12\n\
  set JX_PYTHON=C:\\Python312\\python.exe"
            .to_string();
    }
    if cfg!(target_os = "macos") {
        return "No compatible Python found. JanusX requires Python >=3.9.\n\
Install Python 3.9+ first, or set JX_PYTHON.\n\
Note: a venv cannot upgrade Python major/minor by itself.\n\
macOS examples:\n\
  brew install python\n\
  export JX_PYTHON=/opt/homebrew/bin/python3   # Apple Silicon\n\
  export JX_PYTHON=/usr/local/bin/python3      # Intel x86_64"
            .to_string();
    }
    "No compatible Python found. JanusX requires Python >=3.9.\n\
Install Python 3.9+ first, or set JX_PYTHON.\n\
Note: a venv cannot upgrade Python major/minor by itself.\n\
Linux examples:\n\
  sudo apt-get install -y python3 python3-venv\n\
  export JX_PYTHON=/usr/bin/python3"
        .to_string()
}

fn exit_code(status: ExitStatus) -> i32 {
    status.code().unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn spinner_refresh_interval_rule() {
        assert_eq!(
            spinner_refresh_interval(Duration::from_millis(0)),
            Duration::from_millis(100)
        );
        assert_eq!(
            spinner_refresh_interval(Duration::from_millis(59_999)),
            Duration::from_millis(100)
        );
        assert_eq!(
            spinner_refresh_interval(Duration::from_secs(60)),
            Duration::from_secs(1)
        );
        assert_eq!(
            spinner_refresh_interval(Duration::from_secs(3600)),
            Duration::from_secs(1)
        );
    }

    #[test]
    fn format_elapsed_live_rule() {
        assert_eq!(format_elapsed_live(Duration::from_millis(0)), "0.0s");
        assert_eq!(format_elapsed_live(Duration::from_millis(50)), "0.0s");
        assert_eq!(format_elapsed_live(Duration::from_millis(59900)), "59.9s");
        assert_eq!(format_elapsed_live(Duration::from_secs(60)), "1m00s");
        assert_eq!(format_elapsed_live(Duration::from_secs(3599)), "59m59s");
        assert_eq!(format_elapsed_live(Duration::from_secs(3600)), "1h00m");
        assert_eq!(format_elapsed_live(Duration::from_secs(3661)), "1h01m");
    }

    #[test]
    fn spinner_frame_changes_follow_refresh_step() {
        assert_eq!(spinner_frame_for_elapsed(Duration::from_millis(0)), "/");
        assert_eq!(spinner_frame_for_elapsed(Duration::from_millis(50)), "/");
        assert_eq!(spinner_frame_for_elapsed(Duration::from_millis(100)), "-");

        let frame_60_100 = spinner_frame_for_elapsed(Duration::from_millis(60_100));
        let frame_60_900 = spinner_frame_for_elapsed(Duration::from_millis(60_900));
        let frame_61_000 = spinner_frame_for_elapsed(Duration::from_millis(61_000));
        assert_eq!(frame_60_100, frame_60_900);
        assert_ne!(frame_60_900, frame_61_000);
    }
}
