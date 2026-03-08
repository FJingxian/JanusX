use std::collections::{BTreeSet, HashMap, VecDeque};
use std::env;
use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DLC_SLOT: &str = "active";
const DLC_TOOL_CACHE_META_KEY: &str = "__meta__";
const DEFAULT_IMAGE_TAG: &str = "janusxdlc:latest";
const CONDA_ENV: &str = "janusxdlc";
const CONDA_FORCE_RUNTIME_TOOLS: [&str; 2] = ["gatk", "beagle"];
const REQUIRED_TOOLS: [&str; 9] = [
    "fastp",
    "bwa",
    "samtools",
    "gatk",
    "bcftools",
    "tabix",
    "bgzip",
    "plink",
    "beagle",
];
const DLC_TOOL_ENTRIES: [(&str, &str); 9] = [
    ("bcftools", "VCF/BCF manipulation"),
    ("bgzip", "BGZF block compression"),
    ("beagle", "Phasing and imputation"),
    ("bwa", "Short-read alignment"),
    ("fastp", "FASTQ quality control"),
    ("gatk", "Variant discovery toolkit"),
    ("plink", "Genotype association toolkit"),
    ("samtools", "Alignment and BAM/CRAM utilities"),
    ("tabix", "BGZF indexing and queries"),
];
const JANUSX_SIF_MIRROR_URL: &str =
    "https://gh-proxy.org/https://github.com/FJingxian/JanusX/releases/download/v1.0.10/janusxext.sif";
const JANUSX_SIF_ORIGIN_URL: &str =
    "https://github.com/FJingxian/JanusX/releases/download/v1.0.10/janusxext.sif";
const JANUSX_SIF_URLS: [&str; 2] = [JANUSX_SIF_MIRROR_URL, JANUSX_SIF_ORIGIN_URL];
const MICROMAMBA_BIN_MIRROR_URL: &str = "https://gh-proxy.org/https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64";
const MICROMAMBA_BIN_ORIGIN_URL: &str =
    "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64";
const MICROMAMBA_BIN_URLS: [&str; 2] = [MICROMAMBA_BIN_MIRROR_URL, MICROMAMBA_BIN_ORIGIN_URL];
const MICROMAMBA_ARCHIVE_ORIGIN_URL: &str = "https://micro.mamba.pm/api/micromamba/linux-64/latest";
const MICROMAMBA_ARCHIVE_URLS: [&str; 1] = [MICROMAMBA_ARCHIVE_ORIGIN_URL];
const CONDA_BIOCONDA_MIRROR_URL: &str =
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda";
const CONDA_FORGE_MIRROR_URL: &str =
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge";
const CONDA_BIOCONDA_OFFICIAL: &str = "bioconda";
const CONDA_FORGE_OFFICIAL: &str = "conda-forge";
static DLC_INLINE_STEP_LINE_OPEN: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug)]
struct MethodStatus {
    id: i32,
    name: &'static str,
    builder_id: i32,
    checks: Vec<(&'static str, bool)>,
    selectable: bool,
}

#[derive(Clone, Debug)]
struct RuntimeRecord {
    method_id: i32,
    runtime_mode: String,
    image_tag: String,
    conda_env: String,
    sif_path: String,
    installed_tools: Vec<String>,
    status: String,
    updated_at: String,
}

#[derive(Clone, Debug)]
struct ToolCacheEntry {
    tool: String,
    backend: String,
    locator: String,
    runtime_sig: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ToolLocatorStatus {
    pub tool: String,
    pub route: String,
    pub ready: bool,
}

pub(crate) fn is_dlc_tool(name: &str) -> bool {
    DLC_TOOL_ENTRIES.iter().any(|(tool, _)| *tool == name)
}

pub(crate) fn list_dlc_tool_locators(runtime_home: &Path) -> Result<Vec<ToolLocatorStatus>, String> {
    let mut tools = REQUIRED_TOOLS
        .iter()
        .map(|x| (*x).to_string())
        .collect::<Vec<String>>();
    tools.sort();

    let python = super::venv_python(&runtime_home.join("venv"));
    if !python.exists() {
        return Ok(tools
            .into_iter()
            .map(|tool| ToolLocatorStatus {
                tool,
                route: "none:-".to_string(),
                ready: false,
            })
            .collect());
    }
    let db_path = runtime_home.join(super::GWAS_HISTORY_DB_FILE);
    let Some(record) = load_runtime_record(&python, &db_path)? else {
        return Ok(tools
            .into_iter()
            .map(|tool| ToolLocatorStatus {
                tool,
                route: "none:-".to_string(),
                ready: false,
            })
            .collect());
    };

    let runtime_sig = runtime_signature(&record);
    let (cached_sig, cached_entries) = load_tool_cache_entries(&python, &db_path)?;
    let use_cache = cached_sig
        .map(|s| s == runtime_sig)
        .unwrap_or(false);

    if !use_cache {
        let probed = probe_tool_locators_parallel(&record, runtime_home)?;
        let ready_cache = probed
            .iter()
            .filter(|x| x.ready)
            .map(|x| ToolCacheEntry {
                tool: x.tool.clone(),
                backend: route_backend_name(&record.runtime_mode).to_string(),
                locator: route_locator_from_route(&x.route),
                runtime_sig: runtime_sig.clone(),
            })
            .collect::<Vec<ToolCacheEntry>>();
        save_tool_cache_entries(&python, &db_path, &runtime_sig, &ready_cache)?;
        return Ok(probed);
    }

    let mut cache_map: HashMap<String, ToolCacheEntry> = HashMap::new();
    for entry in cached_entries {
        if entry.runtime_sig == runtime_sig {
            cache_map.insert(entry.tool.clone(), entry);
        }
    }
    let mut out = Vec::with_capacity(tools.len());
    let mut handles: Vec<(String, thread::JoinHandle<bool>)> = Vec::new();
    let mut pending_routes: HashMap<String, String> = HashMap::new();

    for tool in &tools {
        if let Some(entry) = cache_map.get(tool) {
            let route = format!("{}:{}", entry.backend, entry.locator);
            pending_routes.insert(tool.clone(), route);
            let backend = entry.backend.clone();
            let locator = entry.locator.clone();
            handles.push((
                tool.clone(),
                thread::spawn(move || validate_cached_locator(&backend, &locator)),
            ));
        } else {
            let backend = route_backend_name(&record.runtime_mode);
            out.push(ToolLocatorStatus {
                tool: tool.clone(),
                route: format!("{backend}:-"),
                ready: false,
            });
        }
    }

    let mut ready_map: HashMap<String, bool> = HashMap::new();
    for (tool, handle) in handles {
        let ready = handle.join().unwrap_or(false);
        ready_map.insert(tool, ready);
    }

    for tool in &tools {
        if let Some(route) = pending_routes.get(tool) {
            let ready = ready_map.get(tool).copied().unwrap_or(false);
            out.push(ToolLocatorStatus {
                tool: tool.clone(),
                route: route.clone(),
                ready,
            });
        }
    }
    out.sort_by(|a, b| a.tool.cmp(&b.tool));

    let ready_cache = out
        .iter()
        .filter(|x| x.ready)
        .map(|x| {
            let (backend, locator) = split_route(&x.route);
            ToolCacheEntry {
                tool: x.tool.clone(),
                backend: backend.to_string(),
                locator: locator.to_string(),
                runtime_sig: runtime_sig.clone(),
            }
        })
        .collect::<Vec<ToolCacheEntry>>();
    save_tool_cache_entries(&python, &db_path, &runtime_sig, &ready_cache)?;
    Ok(out)
}

pub(crate) fn run_dlc_update(runtime_home: &Path, python: &Path) -> Result<i32, String> {
    fs::create_dir_all(runtime_home).map_err(|e| {
        format!(
            "Failed to create runtime home {}: {e}",
            runtime_home.display()
        )
    })?;
    let db_path = runtime_home.join(super::GWAS_HISTORY_DB_FILE);

    println!(
        "DLC check/build mode. db={}",
        db_path.to_string_lossy().to_string()
    );
    let methods = collect_build_method_status();
    print_build_method_status(&methods);
    let selectable: Vec<i32> = methods
        .iter()
        .filter(|x| x.selectable)
        .map(|x| x.id)
        .collect();
    if selectable.is_empty() {
        println!(
            "{}",
            super::style_yellow("No available build method. Exit.")
        );
        return Ok(1);
    }
    let selected = prompt_build_method(&methods)?;
    let Some(selected_entry) = methods.iter().find(|x| x.id == selected).cloned() else {
        return Err("Invalid build method selection.".to_string());
    };

    if let Some(existing) = load_runtime_record(python, &db_path)? {
        if existing.status == "ok" {
            let (complete, detail) = runtime_completeness(&existing, runtime_home);
            if complete {
                println!(
                    "{}",
                    super::style_green(&format!(
                        "Detected existing complete DLC runtime (mode={}).",
                        existing.runtime_mode
                    ))
                );
                let confirmed = prompt_yes_no(
                    &format!(
                        "Clear previous toolchain and reinstall via {}? [y/N]: ",
                        selected_entry.name
                    ),
                    false,
                )?;
                if !confirmed {
                    println!(
                        "{}",
                        super::style_yellow("Keep existing DLC runtime. Exit without reinstall.")
                    );
                    return Ok(0);
                }
            } else {
                println!(
                    "{}",
                    super::style_yellow(&format!(
                        "Detected existing incomplete DLC runtime (mode={}): {}. Reinstalling via {}.",
                        existing.runtime_mode, detail, selected_entry.name
                    ))
                );
            }
        } else {
            println!(
                "{}",
                super::style_yellow(&format!(
                    "Detected previous DLC runtime with status=`{}`. Reinstalling via {}.",
                    existing.status, selected_entry.name
                ))
            );
        }
        if let Err(e) = clear_runtime_record(python, &db_path) {
            eprintln!(
                "{}",
                super::style_yellow(&format!(
                    "Warning: failed to clear previous DLC runtime record: {e}"
                ))
            );
        }
        cleanup_previous_runtime(&existing, runtime_home)?;
    }

    let mut record = match build_runtime(selected_entry.builder_id, python, runtime_home) {
        Ok(r) => r,
        Err(e) => {
            ensure_dlc_inline_step_newline();
            return Err(e);
        }
    };
    ensure_dlc_inline_step_newline();
    record.method_id = selected;
    let missing_after = missing_tools_for_record(&record, runtime_home)?;
    if !missing_after.is_empty() {
        return Err(format!(
            "Build failed: missing required tools -> {}",
            missing_after.join(", ")
        ));
    }
    record.installed_tools = REQUIRED_TOOLS.iter().map(|x| (*x).to_string()).collect();
    record.status = "ok".to_string();
    record.updated_at = now_utc_epoch();
    save_runtime_record(python, &db_path, &record)?;
    println!(
        "{}",
        super::style_green(&format!(
            "Build success. Saved to DB: {}",
            db_path.to_string_lossy()
        ))
    );
    println!(
        "{}",
        super::style_green(&format!(
            "Installed tools: {}",
            record.installed_tools.join(", ")
        ))
    );
    Ok(0)
}

pub(crate) fn run_dlc_tool(
    tool: &str,
    args: &[String],
    runtime_home: &Path,
) -> Result<i32, String> {
    if !is_dlc_tool(tool) {
        return Err(format!("Unsupported DLC tool: {tool}"));
    }
    let python = super::venv_python(&runtime_home.join("venv"));
    if !python.exists() {
        return Err("DLC runtime is not ready. Please run `jx -update dlc`.".to_string());
    }
    let db_path = runtime_home.join(super::GWAS_HISTORY_DB_FILE);
    if let Some(parent) = db_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let Some(record) = load_runtime_record(&python, &db_path)? else {
        return Err("DLC runtime is not ready. Please run `jx -update dlc`.".to_string());
    };
    if record.status != "ok" {
        return Err("DLC runtime is not ready. Please run `jx -update dlc`.".to_string());
    }
    let installed: BTreeSet<String> = record.installed_tools.iter().cloned().collect();
    if !installed.contains(tool) {
        return Err(format!(
            "DLC runtime does not contain `{tool}`. Please run `jx -update dlc`."
        ));
    }

    let mut cmd = build_tool_command(tool, args, &record, runtime_home)?;
    let status = cmd
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("Failed to run `jx {tool}`: {e}"))?;
    Ok(super::exit_code(status))
}

fn route_backend_name(runtime_mode: &str) -> &'static str {
    match runtime_mode {
        "env" | "local" => "env",
        "conda" => "conda",
        "docker" => "docker",
        "singularity" => "sif",
        _ => "none",
    }
}

fn runtime_signature(record: &RuntimeRecord) -> String {
    format!(
        "{}|{}|{}|{}",
        record.runtime_mode.trim(),
        record.image_tag.trim(),
        record.conda_env.trim(),
        record.sif_path.trim()
    )
}

fn split_route(route: &str) -> (&str, &str) {
    match route.split_once(':') {
        Some((a, b)) => (a, b),
        None => ("none", route),
    }
}

fn route_locator_from_route(route: &str) -> String {
    split_route(route).1.to_string()
}

fn validate_cached_locator(backend: &str, locator: &str) -> bool {
    let loc = locator.trim();
    if loc.is_empty() || loc == "-" {
        return false;
    }
    match backend {
        "env" | "conda" | "sif" => Path::new(loc).exists(),
        "docker" => !loc.is_empty(),
        _ => false,
    }
}

fn probe_tool_locators_parallel(
    record: &RuntimeRecord,
    runtime_home: &Path,
) -> Result<Vec<ToolLocatorStatus>, String> {
    let mut tools = REQUIRED_TOOLS
        .iter()
        .map(|x| (*x).to_string())
        .collect::<Vec<String>>();
    tools.sort();
    let mode = record.runtime_mode.clone();
    let env_dir = env_runtime_bin_dir(runtime_home);
    let conda_env = if record.conda_env.trim().is_empty() {
        CONDA_ENV.to_string()
    } else {
        record.conda_env.trim().to_string()
    };
    let docker_image = if record.image_tag.trim().is_empty() {
        DEFAULT_IMAGE_TAG.to_string()
    } else {
        record.image_tag.trim().to_string()
    };
    let docker_ready_once = if mode == "docker" {
        docker_image_ready(&docker_image)
    } else {
        false
    };
    let sif_path = record.sif_path.trim().to_string();
    let backend = route_backend_name(&mode).to_string();

    let mut handles: Vec<(String, thread::JoinHandle<ToolLocatorStatus>)> = Vec::new();
    for tool in tools {
        let tool_name = tool.clone();
        let mode_c = mode.clone();
        let env_dir_c = env_dir.clone();
        let conda_env_c = conda_env.clone();
        let docker_image_c = docker_image.clone();
        let docker_ready_c = docker_ready_once;
        let sif_path_c = sif_path.clone();
        let backend_c = backend.clone();
        handles.push((
            tool_name.clone(),
            thread::spawn(move || {
                match mode_c.as_str() {
                    "env" | "local" => {
                        let p = env_dir_c.join(&tool_name);
                        let loc = p.to_string_lossy().to_string();
                        let ready = p.exists();
                        ToolLocatorStatus {
                            tool: tool_name,
                            route: format!("{backend_c}:{loc}"),
                            ready,
                        }
                    }
                    "conda" => {
                        let loc = resolve_conda_tool_path(&conda_env_c, &tool_name)
                            .unwrap_or_else(|| "-".to_string());
                        let ready = Path::new(&loc).exists();
                        ToolLocatorStatus {
                            tool: tool_name,
                            route: format!("{backend_c}:{loc}"),
                            ready,
                        }
                    }
                    "docker" => {
                        ToolLocatorStatus {
                            tool: tool_name,
                            route: format!("{backend_c}:{docker_image_c}"),
                            ready: docker_ready_c,
                        }
                    }
                    "singularity" => {
                        let ready = !sif_path_c.is_empty() && Path::new(&sif_path_c).exists();
                        ToolLocatorStatus {
                            tool: tool_name,
                            route: format!("{backend_c}:{sif_path_c}"),
                            ready,
                        }
                    }
                    _ => ToolLocatorStatus {
                        tool: tool_name,
                        route: "none:-".to_string(),
                        ready: false,
                    },
                }
            }),
        ));
    }

    let mut out = Vec::new();
    for (_, handle) in handles {
        let item = handle.join().unwrap_or(ToolLocatorStatus {
            tool: String::new(),
            route: "none:-".to_string(),
            ready: false,
        });
        if !item.tool.is_empty() {
            out.push(item);
        }
    }
    out.sort_by(|a, b| a.tool.cmp(&b.tool));
    Ok(out)
}

fn resolve_conda_tool_path(env_name: &str, tool: &str) -> Option<String> {
    let conda_bin = find_bin("conda")?;
    let out = Command::new(conda_bin)
        .arg("run")
        .arg("--no-capture-output")
        .arg("-n")
        .arg(env_name)
        .arg("sh")
        .arg("-lc")
        .arg(format!("command -v {tool} || true"))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let p = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|x| x.trim())
        .find(|x| !x.is_empty())?
        .to_string();
    if p.is_empty() {
        None
    } else {
        Some(p)
    }
}

fn docker_image_ready(image: &str) -> bool {
    if image.trim().is_empty() {
        return false;
    }
    let Some(docker_bin) = find_bin("docker") else {
        return false;
    };
    if !docker_daemon_running().0 {
        return false;
    }
    Command::new(docker_bin)
        .arg("image")
        .arg("inspect")
        .arg(image)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .status()
        .map(|x| x.success())
        .unwrap_or(false)
}

fn collect_build_method_status() -> Vec<MethodStatus> {
    let linux = cfg!(target_os = "linux");
    let has_conda = command_in_path("conda");
    let has_docker = command_in_path("docker");
    let docker_daemon = if has_docker {
        docker_daemon_running().0
    } else {
        false
    };
    let singularity_ready = match find_bin("singularity") {
        Some(bin) => singularity_runtime_ready(&bin).0,
        None => false,
    };

    if !linux {
        let checks = vec![("docker", has_docker), ("daemon", docker_daemon)];
        return vec![MethodStatus {
            id: 1,
            name: "docker",
            builder_id: 3,
            selectable: checks.iter().all(|x| x.1),
            checks,
        }];
    }

    let mut out = vec![
        MethodStatus {
            id: 1,
            name: "env",
            builder_id: 1,
            checks: vec![("Linux", linux)],
            selectable: false,
        },
        MethodStatus {
            id: 2,
            name: "conda",
            builder_id: 2,
            checks: vec![("Linux", linux), ("conda", has_conda)],
            selectable: false,
        },
        MethodStatus {
            id: 3,
            name: "docker",
            builder_id: 3,
            checks: vec![("docker", has_docker), ("daemon", docker_daemon)],
            selectable: false,
        },
        MethodStatus {
            id: 4,
            name: "singularity",
            builder_id: 4,
            checks: vec![("Linux", linux), ("singularity", singularity_ready)],
            selectable: false,
        },
    ];
    for m in &mut out {
        m.selectable = m.checks.iter().all(|x| x.1);
    }
    out
}

fn print_build_method_status(methods: &[MethodStatus]) {
    for m in methods {
        let mut segs = vec![format!("[{}] {}", m.id, m.name)];
        for (n, ok) in &m.checks {
            segs.push(format!("{n} {}", mark(*ok)));
        }
        let line = segs.join(" ");
        if m.selectable {
            println!("{}", super::style_green(&line));
        } else {
            println!("{}", super::style_yellow(&line));
        }
    }
}

fn prompt_build_method(methods: &[MethodStatus]) -> Result<i32, String> {
    let selectable: Vec<i32> = methods
        .iter()
        .filter(|x| x.selectable)
        .map(|x| x.id)
        .collect();
    if selectable.is_empty() {
        return Err("No available build method in current system.".to_string());
    }
    let default_pick = selectable[0];

    loop {
        print!("Select build method (default {}): ", default_pick);
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush stdout: {e}"))?;
        let mut raw = String::new();
        io::stdin()
            .read_line(&mut raw)
            .map_err(|e| format!("Failed to read input: {e}"))?;
        let v = raw.trim();
        if v.is_empty() {
            return Ok(default_pick);
        }
        let picked = v
            .parse::<i32>()
            .map_err(|_| "Please input method number.".to_string());
        let picked = match picked {
            Ok(x) => x,
            Err(msg) => {
                println!("{}", super::style_yellow(&msg));
                continue;
            }
        };
        let Some(entry) = methods.iter().find(|x| x.id == picked) else {
            println!("{}", super::style_yellow("Invalid method number."));
            continue;
        };
        if !entry.selectable {
            println!("{}", super::style_yellow("Selected method is unavailable."));
            continue;
        }
        return Ok(picked);
    }
}

fn prompt_yes_no(prompt: &str, default_yes: bool) -> Result<bool, String> {
    loop {
        print!("{prompt}");
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush stdout: {e}"))?;
        let mut raw = String::new();
        io::stdin()
            .read_line(&mut raw)
            .map_err(|e| format!("Failed to read input: {e}"))?;
        let v = raw.trim().to_ascii_lowercase();
        if v.is_empty() {
            return Ok(default_yes);
        }
        if matches!(v.as_str(), "y" | "yes") {
            return Ok(true);
        }
        if matches!(v.as_str(), "n" | "no") {
            return Ok(false);
        }
        println!("{}", super::style_yellow("Please input y or n."));
    }
}

fn build_runtime(
    method_id: i32,
    python: &Path,
    runtime_home: &Path,
) -> Result<RuntimeRecord, String> {
    match method_id {
        1 => build_env_runtime(runtime_home),
        2 => build_conda_runtime(runtime_home),
        3 => build_docker_runtime(python),
        4 => build_singularity_runtime(python),
        _ => Err("Unknown build method.".to_string()),
    }
}

fn runtime_completeness(record: &RuntimeRecord, runtime_home: &Path) -> (bool, String) {
    match missing_tools_for_record(record, runtime_home) {
        Ok(missing) => {
            if missing.is_empty() {
                (true, "ok".to_string())
            } else {
                (false, format!("missing {}", missing.join(", ")))
            }
        }
        Err(e) => (false, e),
    }
}

fn cleanup_previous_runtime(record: &RuntimeRecord, runtime_home: &Path) -> Result<(), String> {
    match record.runtime_mode.as_str() {
        "docker" => {
            if let Some(docker_bin) = find_bin("docker") {
                let image = if record.image_tag.trim().is_empty() {
                    DEFAULT_IMAGE_TAG
                } else {
                    record.image_tag.as_str()
                };
                let _ = run_cmd(
                    "cleanup old docker image",
                    &vec![
                        docker_bin.to_string_lossy().to_string(),
                        "image".to_string(),
                        "rm".to_string(),
                        "-f".to_string(),
                        image.to_string(),
                    ],
                    false,
                );
            }
        }
        "conda" => {
            if let Some(conda_bin) = find_bin("conda") {
                let env_name = if record.conda_env.trim().is_empty() {
                    CONDA_ENV
                } else {
                    record.conda_env.as_str()
                };
                let _ = run_cmd(
                    "cleanup old conda env",
                    &vec![
                        conda_bin.to_string_lossy().to_string(),
                        "env".to_string(),
                        "remove".to_string(),
                        "-n".to_string(),
                        env_name.to_string(),
                        "-y".to_string(),
                    ],
                    false,
                );
            }
        }
        "singularity" => {
            let sif = record.sif_path.trim();
            if !sif.is_empty() {
                let p = PathBuf::from(sif);
                if p.exists() {
                    fs::remove_file(&p)
                        .map_err(|e| format!("Failed to remove old SIF {}: {e}", p.display()))?;
                }
            }
        }
        "env" | "local" => {}
        _ => {}
    }
    let dlc_dir = runtime_home.join("dlc");
    if dlc_dir.exists() {
        fs::remove_dir_all(&dlc_dir).map_err(|e| {
            format!(
                "Failed to remove old DLC directory {}: {e}",
                dlc_dir.display()
            )
        })?;
    }
    Ok(())
}

fn build_env_runtime(runtime_home: &Path) -> Result<RuntimeRecord, String> {
    if !cfg!(target_os = "linux") {
        return Err("env method is Linux only.".to_string());
    }
    prepare_env_runtime_bin(runtime_home)?;
    let env_prefix = env_runtime_prefix_dir(runtime_home);
    let host_missing = missing_host_tools();
    if host_missing.is_empty() {
        sync_env_runtime_bin_mixed(runtime_home, &env_prefix)?;
        return Ok(RuntimeRecord {
            method_id: 1,
            runtime_mode: "env".to_string(),
            image_tag: String::new(),
            conda_env: String::new(),
            sif_path: String::new(),
            installed_tools: REQUIRED_TOOLS.iter().map(|x| (*x).to_string()).collect(),
            status: "ok".to_string(),
            updated_at: now_utc_epoch(),
        });
    }
    let ordered_missing = ordered_required_tools(&host_missing);
    let mm_bin = env_runtime_micromamba_bin(runtime_home);
    let need_mm_setup = !mm_bin.exists();
    let need_env_create = !env_prefix.exists();
    let total_steps =
        ordered_missing.len() + usize::from(need_mm_setup) + usize::from(need_env_create);
    let mut step_idx = 1usize;

    let mut dl_desc: Option<String> = None;
    if need_mm_setup {
        dl_desc = Some(format!("[{}/{}] Download micromamba", step_idx, total_steps));
        step_idx += 1;
    }
    let micromamba_bin = ensure_micromamba(runtime_home, dl_desc.as_deref(), None)?;
    if need_env_create {
        let desc = format!("[{}/{}] Create local env via env", step_idx, total_steps);
        run_cmd_tail_with_conda_channel_fallback(
            &desc,
            &[
                micromamba_bin.to_string_lossy().to_string(),
                "create".to_string(),
                "-y".to_string(),
                "-p".to_string(),
                env_prefix.to_string_lossy().to_string(),
            ],
            &["python=3.10".to_string()],
            10,
        )?;
        step_idx += 1;
    }
    let _ = install_tools_stepwise_env(
        &micromamba_bin.to_string_lossy(),
        &env_prefix,
        &ordered_missing,
        "Build",
        step_idx,
        total_steps.max(step_idx.saturating_sub(1)),
    )?;
    sync_env_runtime_bin_mixed(runtime_home, &env_prefix)?;
    let missing_after = missing_env_runtime_tools(runtime_home);
    if !missing_after.is_empty() {
        return Err(format!(
            "DLC env runtime is still missing tools: {}",
            missing_after.join(", ")
        ));
    }
    Ok(RuntimeRecord {
        method_id: 1,
        runtime_mode: "env".to_string(),
        image_tag: String::new(),
        conda_env: String::new(),
        sif_path: String::new(),
        installed_tools: REQUIRED_TOOLS.iter().map(|x| (*x).to_string()).collect(),
        status: "ok".to_string(),
        updated_at: now_utc_epoch(),
    })
}

fn build_conda_runtime(runtime_home: &Path) -> Result<RuntimeRecord, String> {
    if !cfg!(target_os = "linux") {
        return Err("conda method is Linux only.".to_string());
    }
    let Some(conda_bin) = find_bin("conda") else {
        return Err("conda command not found in PATH.".to_string());
    };
    let conda_required = conda_required_tools_for_runtime();
    let conda_required_set: BTreeSet<String> = conda_required.iter().cloned().collect();
    let conda_bin_s = conda_bin.to_string_lossy().to_string();

    let mirror_yml = write_conda_dlc_yml(runtime_home, &conda_required, true)?;
    let official_yml = write_conda_dlc_yml(runtime_home, &conda_required, false)?;

    let mut removed_broken_env = false;
    let mut env_exists = conda_env_named_exists(&conda_bin, CONDA_ENV);
    if env_exists && !conda_env_exists(&conda_bin, CONDA_ENV) {
        run_cmd_tail(
            "build via conda [1/2] remove broken env",
            &vec![
                conda_bin_s.clone(),
                "env".to_string(),
                "remove".to_string(),
                "-n".to_string(),
                CONDA_ENV.to_string(),
                "-y".to_string(),
            ],
            true,
            10,
        )?;
        removed_broken_env = true;
        env_exists = false;
    }
    let apply_desc = if removed_broken_env {
        "build via conda [2/2] apply dlc.yml".to_string()
    } else {
        "build via conda [1/1] apply dlc.yml".to_string()
    };
    run_conda_dlc_yml_with_fallback(
        &conda_bin_s,
        env_exists,
        &mirror_yml,
        &official_yml,
        &apply_desc,
    )?;

    let prefix = vec![
        conda_bin_s.clone(),
        "run".to_string(),
        "--no-capture-output".to_string(),
        "-n".to_string(),
        CONDA_ENV.to_string(),
    ];

    let missing_after = ordered_required_tools(
        &missing_tools_in_prefixed_runtime(&prefix)?
            .into_iter()
            .filter(|x| conda_required_set.contains(x))
            .collect::<Vec<String>>(),
    );
    if !missing_after.is_empty() {
        run_cmd_tail(
            "build via conda [1/3] rebuild remove env",
            &vec![
                conda_bin_s.clone(),
                "env".to_string(),
                "remove".to_string(),
                "-n".to_string(),
                CONDA_ENV.to_string(),
                "-y".to_string(),
            ],
            true,
            10,
        )?;
        run_conda_dlc_yml_with_fallback(
            &conda_bin_s,
            false,
            &mirror_yml,
            &official_yml,
            "build via conda [2/3] recreate via dlc.yml",
        )?;
        let missing_rebuilt = ordered_required_tools(
            &missing_tools_in_prefixed_runtime(&prefix)?
                .into_iter()
                .filter(|x| conda_required_set.contains(x))
                .collect::<Vec<String>>(),
        );
        if !missing_rebuilt.is_empty() {
            return Err(format!(
                "Conda runtime is still missing tools: {}",
                missing_rebuilt.join(", ")
            ));
        }
    }
    Ok(RuntimeRecord {
        method_id: 2,
        runtime_mode: "conda".to_string(),
        image_tag: String::new(),
        conda_env: CONDA_ENV.to_string(),
        sif_path: String::new(),
        installed_tools: REQUIRED_TOOLS.iter().map(|x| (*x).to_string()).collect(),
        status: "ok".to_string(),
        updated_at: now_utc_epoch(),
    })
}

fn build_docker_runtime(python: &Path) -> Result<RuntimeRecord, String> {
    let Some(docker_bin) = find_bin("docker") else {
        return Err("docker command not found in PATH.".to_string());
    };
    let (ready, err) = docker_daemon_running();
    if !ready {
        return Err(format!(
            "Docker daemon is not running. detail: {}",
            if err.is_empty() {
                "docker daemon unavailable"
            } else {
                &err
            }
        ));
    }
    let dockerfile = pipeline_dockerfile_from_python(python)?;
    if !dockerfile.exists() {
        return Err(format!("Dockerfile not found: {}", dockerfile.display()));
    }
    let mut cmd = vec![
        docker_bin.to_string_lossy().to_string(),
        "build".to_string(),
        "--progress=plain".to_string(),
    ];
    if let Some(platform) = docker_default_platform() {
        cmd.push("--platform".to_string());
        cmd.push(platform.to_string());
    }
    cmd.extend([
        "-f".to_string(),
        dockerfile.to_string_lossy().to_string(),
        "-t".to_string(),
        DEFAULT_IMAGE_TAG.to_string(),
        dockerfile
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_string_lossy()
            .to_string(),
    ]);
    run_cmd_tail("build via docker", &cmd, true, 10)?;

    let mut verify_prefix = vec![
        docker_bin.to_string_lossy().to_string(),
        "run".to_string(),
        "--rm".to_string(),
    ];
    if let Some(platform) = docker_default_platform() {
        verify_prefix.push("--platform".to_string());
        verify_prefix.push(platform.to_string());
    }
    verify_prefix.push(DEFAULT_IMAGE_TAG.to_string());
    let missing = missing_tools_in_prefixed_runtime(&verify_prefix)?;
    if !missing.is_empty() {
        let _ = run_cmd(
            "rollback via docker [remove failed image]",
            &vec![
                docker_bin.to_string_lossy().to_string(),
                "image".to_string(),
                "rm".to_string(),
                "-f".to_string(),
                DEFAULT_IMAGE_TAG.to_string(),
            ],
            false,
        );
        return Err(format!(
            "Docker runtime is missing required tools: {}",
            missing.join(", ")
        ));
    }
    Ok(RuntimeRecord {
        method_id: 3,
        runtime_mode: "docker".to_string(),
        image_tag: DEFAULT_IMAGE_TAG.to_string(),
        conda_env: String::new(),
        sif_path: String::new(),
        installed_tools: REQUIRED_TOOLS.iter().map(|x| (*x).to_string()).collect(),
        status: "ok".to_string(),
        updated_at: now_utc_epoch(),
    })
}

fn build_singularity_runtime(python: &Path) -> Result<RuntimeRecord, String> {
    if !cfg!(target_os = "linux") {
        return Err("singularity method is Linux only.".to_string());
    }
    let Some(singularity_bin) = find_bin("singularity") else {
        return Err("singularity command not found in PATH".to_string());
    };
    let (ready, detail) = singularity_runtime_ready(&singularity_bin);
    if !ready {
        return Err(format!("Singularity runtime is not usable: {detail}"));
    }
    let pipeline_dir = pipeline_dir_from_python(python)?;
    let bin_dir = pipeline_dir.join("bin");
    fs::create_dir_all(&bin_dir)
        .map_err(|e| format!("Failed to create {}: {e}", bin_dir.display()))?;
    let sif_path = bin_dir.join("janusxext.sif");
    if !sif_path.exists() {
        print!("Download JanusX Singularity Image (~800MB)? [y/N/path.sif]: ");
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush stdout: {e}"))?;
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .map_err(|e| format!("Failed to read input: {e}"))?;
        let input = line.trim();
        if input.is_empty() || input.eq_ignore_ascii_case("n") {
            return Err("No SIF provided.".to_string());
        }
        let tmp_path = bin_dir.join("janusxext.tmp.sif");
        if input.eq_ignore_ascii_case("y") {
            download_with_fallback("download SIF", &JANUSX_SIF_URLS, &tmp_path)?;
        } else {
            let src = PathBuf::from(input);
            if !src.exists() {
                return Err(format!("SIF path not found: {}", src.display()));
            }
            fs::copy(&src, &tmp_path).map_err(|e| {
                format!(
                    "Failed to copy SIF {} -> {}: {e}",
                    src.display(),
                    tmp_path.display()
                )
            })?;
        }
        fs::rename(&tmp_path, &sif_path).map_err(|e| {
            format!(
                "Failed to move {} -> {}: {e}",
                tmp_path.display(),
                sif_path.display()
            )
        })?;
    }
    let verify_prefix = vec![
        singularity_bin.to_string_lossy().to_string(),
        "exec".to_string(),
        sif_path.to_string_lossy().to_string(),
    ];
    let missing = missing_tools_in_prefixed_runtime(&verify_prefix)?;
    if !missing.is_empty() {
        return Err(format!(
            "Singularity image is missing required tools: {}",
            missing.join(", ")
        ));
    }
    Ok(RuntimeRecord {
        method_id: 4,
        runtime_mode: "singularity".to_string(),
        image_tag: String::new(),
        conda_env: String::new(),
        sif_path: sif_path.to_string_lossy().to_string(),
        installed_tools: REQUIRED_TOOLS.iter().map(|x| (*x).to_string()).collect(),
        status: "ok".to_string(),
        updated_at: now_utc_epoch(),
    })
}

fn build_tool_command(
    tool: &str,
    args: &[String],
    record: &RuntimeRecord,
    runtime_home: &Path,
) -> Result<Command, String> {
    match record.runtime_mode.as_str() {
        "env" | "local" => {
            let tool_path = env_runtime_tool_path(runtime_home, tool);
            if !tool_path.exists() {
                return Err(format!(
                    "DLC env runtime is missing `{}` at {}. Please run `jx -update dlc`.",
                    tool,
                    tool_path.display()
                ));
            }
            let mut cmd = Command::new(tool_path);
            apply_env_runtime_exec_env(&mut cmd, runtime_home);
            cmd.args(args);
            Ok(cmd)
        }
        "conda" => {
            if !should_force_conda_run(tool) {
                if let Some(host_tool) = find_bin(tool) {
                    let mut cmd = Command::new(host_tool);
                    cmd.args(args);
                    return Ok(cmd);
                }
            }
            let Some(conda_bin) = find_bin("conda") else {
                return Err("conda command not found in PATH.".to_string());
            };
            let env_name = if record.conda_env.trim().is_empty() {
                CONDA_ENV
            } else {
                record.conda_env.as_str()
            };
            let mut cmd = Command::new(conda_bin);
            cmd.arg("run")
                .arg("--no-capture-output")
                .arg("-n")
                .arg(env_name)
                .arg(tool)
                .args(args);
            Ok(cmd)
        }
        "docker" => {
            let Some(docker_bin) = find_bin("docker") else {
                return Err("docker command not found in PATH.".to_string());
            };
            let (ready, _) = docker_daemon_running();
            if !ready {
                return Err("Docker daemon is not running.".to_string());
            }
            let image = if record.image_tag.trim().is_empty() {
                DEFAULT_IMAGE_TAG
            } else {
                record.image_tag.as_str()
            };
            let cwd = env::current_dir().map_err(|e| format!("Failed to get current dir: {e}"))?;
            let mounts = collect_docker_mounts(args, &cwd);

            let mut cmd = Command::new(docker_bin);
            cmd.arg("run").arg("--rm");
            if let Some(platform) = docker_default_platform() {
                cmd.arg("--platform").arg(platform);
            }
            if let Some((uid, gid)) = current_uid_gid() {
                cmd.arg("--user").arg(format!("{uid}:{gid}"));
            }
            for m in mounts {
                let s = m.to_string_lossy().to_string();
                cmd.arg("-v").arg(format!("{s}:{s}"));
            }
            cmd.arg("-w")
                .arg(cwd.to_string_lossy().to_string())
                .arg(image)
                .arg(tool)
                .args(args);
            Ok(cmd)
        }
        "singularity" => {
            let Some(singularity_bin) = find_bin("singularity") else {
                return Err("singularity command not found in PATH.".to_string());
            };
            let (ready, detail) = singularity_runtime_ready(&singularity_bin);
            if !ready {
                return Err(format!("Singularity runtime is not usable: {detail}"));
            }
            let sif = PathBuf::from(record.sif_path.trim());
            if !sif.exists() {
                return Err(
                    "Saved singularity runtime is invalid. Please run `jx -update dlc`."
                        .to_string(),
                );
            }
            let mut cmd = Command::new(singularity_bin);
            cmd.arg("exec").arg(sif).arg(tool).args(args);
            Ok(cmd)
        }
        _ => Err("Unknown DLC runtime mode. Please run `jx -update dlc`.".to_string()),
    }
}

fn apply_env_runtime_exec_env(cmd: &mut Command, runtime_home: &Path) {
    let env_prefix = env_runtime_prefix_dir(runtime_home);
    let env_bin = env_prefix.join("bin");

    let mut paths: Vec<PathBuf> = Vec::new();
    if env_bin.exists() && env_bin.is_dir() {
        paths.push(env_bin.clone());
    }
    if let Some(curr) = env::var_os("PATH") {
        paths.extend(env::split_paths(&curr));
    }
    if let Ok(joined) = env::join_paths(paths) {
        cmd.env("PATH", joined);
    }
    if env_prefix.exists() && env_prefix.is_dir() {
        cmd.env("CONDA_PREFIX", &env_prefix);
    }
    let java_bin = if cfg!(windows) {
        env_bin.join("java.exe")
    } else {
        env_bin.join("java")
    };
    if java_bin.exists() && java_bin.is_file() {
        cmd.env("JAVA_HOME", &env_prefix);
        cmd.env("GATK_JAVA", &java_bin);
    }
}

fn missing_tools_for_record(
    record: &RuntimeRecord,
    runtime_home: &Path,
) -> Result<Vec<String>, String> {
    match record.runtime_mode.as_str() {
        "env" | "local" => Ok(missing_env_runtime_tools(runtime_home)),
        "conda" => {
            let conda_required_set: BTreeSet<String> =
                conda_required_tools_for_runtime().into_iter().collect();
            let Some(conda_bin) = find_bin("conda") else {
                return Err("conda command not found in PATH.".to_string());
            };
            let env_name = if record.conda_env.trim().is_empty() {
                CONDA_ENV
            } else {
                record.conda_env.as_str()
            };
            Ok(ordered_required_tools(
                &missing_tools_in_prefixed_runtime(&[
                conda_bin.to_string_lossy().to_string(),
                "run".to_string(),
                "--no-capture-output".to_string(),
                "-n".to_string(),
                env_name.to_string(),
                ])?
                .into_iter()
                .filter(|x| conda_required_set.contains(x))
                .collect::<Vec<String>>(),
            ))
        }
        "docker" => {
            let Some(docker_bin) = find_bin("docker") else {
                return Err("docker command not found in PATH.".to_string());
            };
            let image = if record.image_tag.trim().is_empty() {
                DEFAULT_IMAGE_TAG
            } else {
                record.image_tag.as_str()
            };
            let mut prefix = vec![
                docker_bin.to_string_lossy().to_string(),
                "run".to_string(),
                "--rm".to_string(),
            ];
            if let Some(platform) = docker_default_platform() {
                prefix.push("--platform".to_string());
                prefix.push(platform.to_string());
            }
            prefix.push(image.to_string());
            missing_tools_in_prefixed_runtime(&prefix)
        }
        "singularity" => {
            let Some(singularity_bin) = find_bin("singularity") else {
                return Err("singularity command not found in PATH.".to_string());
            };
            let (ready, detail) = singularity_runtime_ready(&singularity_bin);
            if !ready {
                return Err(format!("Singularity runtime is not usable: {detail}"));
            }
            let sif = record.sif_path.trim();
            if sif.is_empty() {
                return Err("Saved singularity runtime is invalid.".to_string());
            }
            missing_tools_in_prefixed_runtime(&[
                singularity_bin.to_string_lossy().to_string(),
                "exec".to_string(),
                sif.to_string(),
            ])
        }
        _ => Err("Unknown DLC runtime mode.".to_string()),
    }
}

fn prepare_env_runtime_bin(runtime_home: &Path) -> Result<(), String> {
    let bin_dir = env_runtime_bin_dir(runtime_home);
    fs::create_dir_all(&bin_dir).map_err(|e| {
        format!(
            "Failed to prepare DLC env bin directory {}: {e}",
            bin_dir.display()
        )
    })
}

fn env_runtime_bin_dir(runtime_home: &Path) -> PathBuf {
    runtime_home.join("dlc").join("bin")
}

fn env_runtime_prefix_dir(runtime_home: &Path) -> PathBuf {
    runtime_home.join("dlc").join("env")
}

fn env_runtime_micromamba_root(runtime_home: &Path) -> PathBuf {
    runtime_home.join("dlc").join("_micromamba")
}

fn env_runtime_micromamba_bin(runtime_home: &Path) -> PathBuf {
    env_runtime_micromamba_root(runtime_home)
        .join("bin")
        .join("micromamba")
}

fn env_runtime_tool_path(runtime_home: &Path, tool: &str) -> PathBuf {
    env_runtime_bin_dir(runtime_home).join(tool)
}

fn missing_host_tools() -> Vec<String> {
    REQUIRED_TOOLS
        .iter()
        .filter(|x| !command_in_path(x))
        .map(|x| (*x).to_string())
        .collect()
}

fn ensure_micromamba(
    runtime_home: &Path,
    download_desc: Option<&str>,
    extract_desc: Option<&str>,
) -> Result<PathBuf, String> {
    let mm_bin = env_runtime_micromamba_bin(runtime_home);
    if mm_bin.exists() {
        return Ok(mm_bin);
    }
    let mm_root = env_runtime_micromamba_root(runtime_home);
    fs::create_dir_all(&mm_root).map_err(|e| {
        format!(
            "Failed to prepare micromamba directory {}: {e}",
            mm_root.display()
        )
    })?;
    let desc = download_desc.unwrap_or("download micromamba");
    if let Some(mm_parent) = mm_bin.parent() {
        fs::create_dir_all(mm_parent).map_err(|e| {
            format!(
                "Failed to prepare micromamba bin directory {}: {e}",
                mm_parent.display()
            )
        })?;
    }
    match download_with_fallback(desc, &MICROMAMBA_BIN_URLS, &mm_bin) {
        Ok(()) => {
            #[cfg(unix)]
            {
                let perms = fs::Permissions::from_mode(0o755);
                fs::set_permissions(&mm_bin, perms).map_err(|e| {
                    format!(
                        "Failed to set executable permission {}: {e}",
                        mm_bin.display()
                    )
                })?;
            }
            return Ok(mm_bin);
        }
        Err(bin_err) => {
            eprintln!(
                "{}",
                super::style_yellow(&format!(
                    "Micromamba binary download failed, fallback to archive source: {bin_err}"
                ))
            );
        }
    }

    if !command_in_path("tar") {
        return Err(
            "`tar` command not found in PATH (required for micromamba archive fallback)."
                .to_string(),
        );
    }
    let archive = mm_root.join("micromamba.tar.bz2");
    if !archive.exists() {
        download_with_fallback(desc, &MICROMAMBA_ARCHIVE_URLS, &archive)?;
    }
    let extract_desc = extract_desc.unwrap_or("build via env [extract micromamba]");
    run_cmd(
        extract_desc,
        &vec![
            "tar".to_string(),
            "-xjf".to_string(),
            archive.to_string_lossy().to_string(),
            "-C".to_string(),
            mm_root.to_string_lossy().to_string(),
        ],
        true,
    )?;
    if !mm_bin.exists() {
        return Err(format!(
            "micromamba binary not found after extraction: {}",
            mm_bin.display()
        ));
    }
    Ok(mm_bin)
}

fn sync_env_runtime_bin_mixed(runtime_home: &Path, env_prefix: &Path) -> Result<(), String> {
    prepare_env_runtime_bin(runtime_home)?;
    let bin_dir = env_runtime_bin_dir(runtime_home);
    for tool in REQUIRED_TOOLS {
        let local_src = env_prefix.join("bin").join(tool);
        let src = if local_src.exists() {
            local_src
        } else {
            let Some(host_src) = find_bin(tool) else {
                return Err(format!(
                    "Tool `{tool}` not found in either host PATH or DLC env prefix {}",
                    env_prefix.display()
                ));
            };
            host_src
        };
        let dst = bin_dir.join(tool);
        write_tool_wrapper(&dst, &src)?;
    }
    Ok(())
}

fn toolchain_packages_for_tools(tools: &[String]) -> Vec<String> {
    let mut out: BTreeSet<String> = BTreeSet::new();
    for t in tools {
        match t.as_str() {
            "fastp" => {
                out.insert("fastp".to_string());
            }
            "bwa" => {
                out.insert("bwa".to_string());
            }
            "samtools" => {
                out.insert("samtools".to_string());
            }
            "bcftools" => {
                out.insert("bcftools".to_string());
            }
            "tabix" => {
                out.insert("htslib".to_string());
            }
            "bgzip" => {
                out.insert("htslib".to_string());
            }
            "plink" => {
                out.insert("plink".to_string());
            }
            "beagle" => {
                out.insert("beagle".to_string());
                out.insert("openjdk=17".to_string());
            }
            "gatk" => {
                out.insert("gatk4".to_string());
                out.insert("picard".to_string());
                out.insert("openjdk=17".to_string());
            }
            other => {
                out.insert(other.to_string());
            }
        }
    }
    out.into_iter().collect()
}

fn write_tool_wrapper(dst: &Path, src: &Path) -> Result<(), String> {
    let src_s = src.to_string_lossy().to_string();
    let body = format!(
        "#!/usr/bin/env bash\nexec '{}' \"$@\"\n",
        sh_single_quote(&src_s)
    );
    if dst.exists() {
        fs::remove_file(dst).map_err(|e| format!("Failed to replace {}: {e}", dst.display()))?;
    }
    fs::write(dst, body).map_err(|e| format!("Failed to write {}: {e}", dst.display()))?;
    #[cfg(unix)]
    {
        let perms = fs::Permissions::from_mode(0o755);
        fs::set_permissions(dst, perms)
            .map_err(|e| format!("Failed to set executable permission {}: {e}", dst.display()))?;
    }
    Ok(())
}

fn sh_single_quote(raw: &str) -> String {
    raw.replace('\'', "'\"'\"'")
}

fn missing_env_runtime_tools(runtime_home: &Path) -> Vec<String> {
    REQUIRED_TOOLS
        .iter()
        .filter_map(|tool| {
            let p = env_runtime_tool_path(runtime_home, tool);
            if p.exists() && p.is_file() {
                None
            } else {
                Some((*tool).to_string())
            }
        })
        .collect()
}

fn missing_tools_in_prefixed_runtime(prefix: &[String]) -> Result<Vec<String>, String> {
    let script = format!(
        "for t in {}; do command -v \"$t\" >/dev/null 2>&1 || echo \"$t\"; done",
        REQUIRED_TOOLS.join(" ")
    );
    let mut cmd = Command::new(&prefix[0]);
    cmd.args(&prefix[1..])
        .arg("sh")
        .arg("-lc")
        .arg(script)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null());
    let out = cmd
        .output()
        .map_err(|e| format!("Failed to verify runtime tools: {e}"))?;
    if !out.status.success() {
        let mut msg = String::new();
        msg.push_str(&String::from_utf8_lossy(&out.stdout));
        msg.push_str(&String::from_utf8_lossy(&out.stderr));
        return Err(format!(
            "Runtime tool check command failed (exit={}): {}",
            super::exit_code(out.status),
            msg.trim()
        ));
    }
    let mut missing: Vec<String> = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let v = line.trim();
        if !v.is_empty() {
            missing.push(v.to_string());
        }
    }
    missing.sort();
    missing.dedup();
    Ok(missing)
}

fn run_cmd(desc: &str, argv: &[String], required: bool) -> Result<(), String> {
    if argv.is_empty() {
        return Err(format!("{desc}: empty command"));
    }
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let (out, elapsed) = super::run_with_spinner(&mut cmd, &format!("{desc} ..."))?;
    if out.status.success() {
        super::print_success_line(&format!("{desc}[{}]", super::format_elapsed(elapsed)));
        return Ok(());
    }
    let mut msg = String::new();
    msg.push_str(&String::from_utf8_lossy(&out.stdout));
    msg.push_str(&String::from_utf8_lossy(&out.stderr));
    if required {
        return Err(format!(
            "{desc} failed with exit={}.\n{}",
            super::exit_code(out.status),
            msg.trim()
        ));
    }
    eprintln!(
        "{}",
        super::style_yellow(&format!(
            "Warning: {desc} failed (optional, exit={}).",
            super::exit_code(out.status)
        ))
    );
    Ok(())
}

struct LiveCmdResult {
    status: ExitStatus,
    stdout: String,
    stderr: String,
    streamed: bool,
}

fn run_cmd_tail(
    desc: &str,
    argv: &[String],
    required: bool,
    max_lines: usize,
) -> Result<(), String> {
    if argv.is_empty() {
        return Err(format!("{desc}: empty command"));
    }
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let out = run_with_live_tail(&mut cmd, desc, max_lines)?;
    if out.status.success() {
        return Ok(());
    }

    let mut msg = String::new();
    msg.push_str(&out.stdout);
    msg.push_str(&out.stderr);
    let code = super::exit_code(out.status);

    if required {
        if out.streamed {
            return Err(format!("{desc} failed with exit={code}."));
        }
        return Err(format!("{desc} failed with exit={code}.\n{}", msg.trim()));
    }

    eprintln!(
        "{}",
        super::style_yellow(&format!("Warning: {desc} failed (optional, exit={code})."))
    );
    Ok(())
}

fn append_conda_channel_args(cmd: &mut Vec<String>, use_mirror: bool) {
    cmd.push("--override-channels".to_string());
    cmd.push("-c".to_string());
    cmd.push(if use_mirror {
        CONDA_BIOCONDA_MIRROR_URL.to_string()
    } else {
        CONDA_BIOCONDA_OFFICIAL.to_string()
    });
    cmd.push("-c".to_string());
    cmd.push(if use_mirror {
        CONDA_FORGE_MIRROR_URL.to_string()
    } else {
        CONDA_FORGE_OFFICIAL.to_string()
    });
}

fn run_cmd_tail_with_conda_channel_fallback(
    desc: &str,
    base_argv: &[String],
    packages: &[String],
    max_lines: usize,
) -> Result<(), String> {
    let mut mirror_cmd: Vec<String> = base_argv.to_vec();
    append_conda_channel_args(&mut mirror_cmd, true);
    mirror_cmd.extend(packages.iter().cloned());
    match run_cmd_tail(desc, &mirror_cmd, true, max_lines) {
        Ok(()) => Ok(()),
        Err(mirror_err) => {
            eprintln!(
                "{}",
                super::style_yellow(&format!(
                    "{desc}: mirror channels failed, fallback to official channels."
                ))
            );
            let mut official_cmd: Vec<String> = base_argv.to_vec();
            append_conda_channel_args(&mut official_cmd, false);
            official_cmd.extend(packages.iter().cloned());
            run_cmd_tail(desc, &official_cmd, true, max_lines)
                .map_err(|e| format!("{e}\nMirror attempt error: {mirror_err}"))
        }
    }
}

fn run_with_live_tail(
    cmd: &mut Command,
    desc: &str,
    max_lines: usize,
) -> Result<LiveCmdResult, String> {
    let is_tty = io::stdout().is_terminal();
    let start = Instant::now();
    if !is_tty {
        let out = cmd
            .output()
            .map_err(|e| format!("Failed to run command: {e}"))?;
        return Ok(LiveCmdResult {
            status: out.status,
            stdout: String::from_utf8_lossy(&out.stdout).to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).to_string(),
            streamed: false,
        });
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to run command: {e}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to capture command stdout".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Failed to capture command stderr".to_string())?;

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
    let mut prelog_spinner_shown = true;
    let mut streaming_title_started = false;
    let mut out_text = String::new();
    let mut err_text = String::new();
    let mut tail: VecDeque<String> = VecDeque::with_capacity(max_lines.max(1));
    let mut channel_open = true;
    let mut exit_status: Option<ExitStatus> = None;
    let mut last_render = Instant::now();
    let mut streamed_lines_before_tail = 0usize;
    let mut saw_any_line = false;

    render_prelog_spinner_line_dlc(desc, start.elapsed())?;

    while channel_open {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok((is_err, line)) => {
                if prelog_spinner_shown && !tail_mode_started {
                    if !streaming_title_started {
                        render_streaming_title_init_dlc(desc, start.elapsed())?;
                        streaming_title_started = true;
                    }
                    prelog_spinner_shown = false;
                }
                saw_any_line = true;
                if is_err {
                    err_text.push_str(&line);
                    err_text.push('\n');
                } else {
                    out_text.push_str(&line);
                    out_text.push('\n');
                }
                if max_lines > 0 {
                    if tail.len() == max_lines {
                        tail.pop_front();
                    }
                    tail.push_back(line.clone());
                }

                if max_lines == 0 {
                    tail_mode_started = true;
                    render_tail_block_dlc(desc, "", start.elapsed(), &tail, max_lines, rendered)?;
                    rendered = true;
                    last_render = Instant::now();
                    continue;
                }

                if !tail_mode_started {
                    if tail.len() < max_lines {
                        let width = dlc_terminal_line_width();
                        let trimmed = dlc_truncate_plain_line(&line, width);
                        if super::supports_color() {
                            println!("\x1b[2m{}\x1b[0m", trimmed);
                        } else {
                            println!("{trimmed}");
                        }
                        streamed_lines_before_tail += 1;
                        io::stdout()
                            .flush()
                            .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
                    } else {
                        if streamed_lines_before_tail > 0 {
                            let mut up = streamed_lines_before_tail;
                            if streaming_title_started {
                                up = up.saturating_add(1);
                            }
                            print!("\x1b[{}A", up);
                        }
                        render_tail_block_dlc(desc, "", start.elapsed(), &tail, max_lines, false)?;
                        rendered = true;
                        tail_mode_started = true;
                        last_render = Instant::now();
                    }
                } else {
                    render_tail_block_dlc(desc, "", start.elapsed(), &tail, max_lines, rendered)?;
                    rendered = true;
                    last_render = Instant::now();
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if exit_status.is_none() {
                    exit_status = child
                        .try_wait()
                        .map_err(|e| format!("Failed to poll command status: {e}"))?;
                }
                if !tail_mode_started && !saw_any_line {
                    render_prelog_spinner_line_dlc(desc, start.elapsed())?;
                    prelog_spinner_shown = true;
                }
                if !tail_mode_started
                    && streaming_title_started
                    && last_render.elapsed() >= Duration::from_millis(100)
                {
                    render_streaming_title_only_dlc(
                        desc,
                        start.elapsed(),
                        streamed_lines_before_tail,
                    )?;
                    last_render = Instant::now();
                }
                if tail_mode_started && last_render.elapsed() >= Duration::from_millis(100) {
                    render_tail_title_only_dlc(desc, start.elapsed(), max_lines)?;
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
            .map_err(|e| format!("Failed to wait command process: {e}"))?
    };
    let elapsed = start.elapsed();

    if prelog_spinner_shown {
        print!("\r\x1b[2K");
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
    }
    if tail_mode_started {
        DLC_INLINE_STEP_LINE_OPEN.store(false, Ordering::Relaxed);
        if status.success() {
            render_tail_success_compact_dlc(desc, elapsed, max_lines)?;
        } else {
            render_tail_block_dlc(desc, "✘", elapsed, &tail, max_lines, rendered)?;
        }
    } else if !status.success() && streaming_title_started {
        DLC_INLINE_STEP_LINE_OPEN.store(false, Ordering::Relaxed);
        render_streaming_failure_compact_dlc(desc, elapsed, streamed_lines_before_tail)?;
    } else {
        let width = dlc_terminal_line_width();
        let title = dlc_truncate_plain_line(
            &format!(
                "{} {desc}[{}]",
                if status.success() { "✔︎" } else { "✘" },
                super::format_elapsed(elapsed)
            ),
            width,
        );
        if status.success() {
            if is_inline_step_desc(desc) {
                print!("\r\x1b[2K{}", super::style_green(&title));
                DLC_INLINE_STEP_LINE_OPEN.store(true, Ordering::Relaxed);
            } else {
                println!("{}", super::style_green(&title));
                DLC_INLINE_STEP_LINE_OPEN.store(false, Ordering::Relaxed);
            }
        } else {
            println!("{}", super::style_yellow(&title));
            DLC_INLINE_STEP_LINE_OPEN.store(false, Ordering::Relaxed);
        }
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
    }

    Ok(LiveCmdResult {
        status,
        stdout: out_text,
        stderr: err_text,
        streamed: saw_any_line,
    })
}

fn render_tail_block_dlc(
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
    let width = dlc_terminal_line_width();
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    if symbol.is_empty() {
        let idx = ((elapsed.as_millis() / 100) as usize) % frames.len();
        let title = dlc_truncate_plain_line(
            &format!(
                "{} {}[{}]",
                frames[idx],
                desc,
                super::format_elapsed(elapsed)
            ),
            width,
        );
        print!("\x1b[2K\r{}\n", super::style_green(&title));
    } else {
        let title = dlc_truncate_plain_line(
            &format!("{symbol} {desc}[{}]", super::format_elapsed(elapsed)),
            width,
        );
        print!("\x1b[2K\r{}\n", super::style_yellow(&title));
    }
    let mut shown = 0usize;
    for line in tail {
        let trimmed = dlc_truncate_plain_line(line, width);
        print!("\x1b[2K\r\x1b[2m{}\x1b[0m\n", trimmed);
        shown += 1;
    }
    while shown < max_lines {
        print!("\x1b[2K\r\n");
        shown += 1;
    }
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
    Ok(())
}

fn render_tail_title_only_dlc(
    desc: &str,
    elapsed: Duration,
    max_lines: usize,
) -> Result<(), String> {
    let width = dlc_terminal_line_width();
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let idx = ((elapsed.as_millis() / 100) as usize) % frames.len();
    let title = dlc_truncate_plain_line(
        &format!(
            "{} {}[{}]",
            frames[idx],
            desc,
            super::format_elapsed(elapsed)
        ),
        width,
    );
    print!("\x1b[{}A", max_lines.saturating_add(1));
    print!("\x1b[2K\r{}\n", super::style_green(&title));
    print!("\x1b[{}B", max_lines);
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
    Ok(())
}

fn render_streaming_failure_compact_dlc(
    desc: &str,
    elapsed: Duration,
    lines_below: usize,
) -> Result<(), String> {
    let width = dlc_terminal_line_width();
    let title = dlc_truncate_plain_line(
        &format!("✘ {desc}[{}]", super::format_elapsed(elapsed)),
        width,
    );
    let up = lines_below.saturating_add(1);
    print!("\x1b[{}A", up);
    print!("\x1b[2K\r{}\n", super::style_yellow(&title));
    if lines_below > 0 {
        print!("\x1b[{}M", lines_below);
    }
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
    Ok(())
}

fn render_tail_success_compact_dlc(
    desc: &str,
    elapsed: Duration,
    max_lines: usize,
) -> Result<(), String> {
    let width = dlc_terminal_line_width();
    let title = dlc_truncate_plain_line(
        &format!("✔︎ {desc}[{}]", super::format_elapsed(elapsed)),
        width,
    );
    print!("\x1b[{}A", max_lines.saturating_add(1));
    print!("\x1b[2K\r{}\n", super::style_green(&title));
    if max_lines > 0 {
        print!("\x1b[{}M", max_lines);
    }
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
    Ok(())
}

fn render_prelog_spinner_line_dlc(desc: &str, elapsed: Duration) -> Result<(), String> {
    let width = dlc_terminal_line_width();
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let idx = ((elapsed.as_millis() / 100) as usize) % frames.len();
    let title = dlc_truncate_plain_line(
        &format!(
            "{} {}[{}]",
            frames[idx],
            desc,
            super::format_elapsed(elapsed)
        ),
        width,
    );
    print!("\r\x1b[2K{}", super::style_green(&title));
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
    Ok(())
}

fn render_streaming_title_init_dlc(desc: &str, elapsed: Duration) -> Result<(), String> {
    let width = dlc_terminal_line_width();
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let idx = ((elapsed.as_millis() / 100) as usize) % frames.len();
    let title = dlc_truncate_plain_line(
        &format!(
            "{} {}[{}]",
            frames[idx],
            desc,
            super::format_elapsed(elapsed)
        ),
        width,
    );
    print!("\r\x1b[2K{}\n", super::style_green(&title));
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
    Ok(())
}

fn render_streaming_title_only_dlc(
    desc: &str,
    elapsed: Duration,
    lines_below: usize,
) -> Result<(), String> {
    let width = dlc_terminal_line_width();
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let idx = ((elapsed.as_millis() / 100) as usize) % frames.len();
    let title = dlc_truncate_plain_line(
        &format!(
            "{} {}[{}]",
            frames[idx],
            desc,
            super::format_elapsed(elapsed)
        ),
        width,
    );
    let up = lines_below.saturating_add(1);
    print!("\x1b[{}A", up);
    print!("\x1b[2K\r{}\n", super::style_green(&title));
    if lines_below > 0 {
        print!("\x1b[{}B", lines_below);
    }
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
    Ok(())
}

fn dlc_terminal_line_width() -> usize {
    let cols = env::var("COLUMNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(80);
    cols.saturating_sub(4).clamp(40, 68)
}

fn dlc_truncate_plain_line(s: &str, max_chars: usize) -> String {
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

fn is_inline_step_desc(desc: &str) -> bool {
    desc.starts_with('[') && desc.contains('/') && desc.contains(']')
}

fn ensure_dlc_inline_step_newline() {
    if !DLC_INLINE_STEP_LINE_OPEN.swap(false, Ordering::Relaxed) {
        return;
    }
    println!();
    let _ = io::stdout().flush();
}

fn pipeline_dir_from_python(python: &Path) -> Result<PathBuf, String> {
    let out = Command::new(python)
        .arg("-c")
        .arg("import janusx.pipeline, pathlib; print(pathlib.Path(janusx.pipeline.__file__).resolve().parent)")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("Failed to resolve janusx.pipeline path: {e}"))?;
    if !out.status.success() {
        let mut msg = String::new();
        msg.push_str(&String::from_utf8_lossy(&out.stdout));
        msg.push_str(&String::from_utf8_lossy(&out.stderr));
        return Err(format!(
            "Failed to locate janusx.pipeline path (exit={}): {}",
            super::exit_code(out.status),
            msg.trim()
        ));
    }
    let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if p.is_empty() {
        return Err("janusx.pipeline path is empty.".to_string());
    }
    Ok(PathBuf::from(p))
}

fn pipeline_dockerfile_from_python(python: &Path) -> Result<PathBuf, String> {
    let pipeline_dir = pipeline_dir_from_python(python)?;
    Ok(pipeline_dir.join("docker").join("dockerfile"))
}

fn download_with_fallback(desc: &str, urls: &[&str], dst: &Path) -> Result<(), String> {
    let tmp = PathBuf::from(format!("{}.tmp.download", dst.to_string_lossy()));
    let mut errs: Vec<String> = Vec::new();
    for (idx, url) in urls.iter().enumerate() {
        let _ = fs::remove_file(&tmp);
        if idx > 0 {
            eprintln!(
                "{}",
                super::style_yellow(&format!(
                    "{desc}: source {} failed, fallback to source {}/{}",
                    idx,
                    idx + 1,
                    urls.len()
                ))
            );
        }
        let mut ok = false;
        if command_in_path("aria2c") {
            let total_bytes = probe_content_length_with_curl(url)
                .or_else(|| probe_content_length_with_wget(url));
            let tmp_dir = tmp
                .parent()
                .ok_or_else(|| format!("Invalid temporary download path: {}", tmp.display()))?
                .to_string_lossy()
                .to_string();
            let tmp_name = tmp
                .file_name()
                .ok_or_else(|| format!("Invalid temporary download path: {}", tmp.display()))?
                .to_string_lossy()
                .to_string();
            let cmd = vec![
                "aria2c".to_string(),
                "--allow-overwrite=true".to_string(),
                "--auto-file-renaming=false".to_string(),
                "--file-allocation=none".to_string(),
                "--console-log-level=error".to_string(),
                "--summary-interval=0".to_string(),
                "-x".to_string(),
                "8".to_string(),
                "-s".to_string(),
                "8".to_string(),
                "-k".to_string(),
                "1M".to_string(),
                "-d".to_string(),
                tmp_dir,
                "-o".to_string(),
                tmp_name,
                (*url).to_string(),
            ];
            if run_download_with_progress(desc, &cmd, &tmp, total_bytes).is_ok() {
                ok = true;
            }
        } else if command_in_path("curl") {
            let total_bytes = probe_content_length_with_curl(url);
            let cmd = vec![
                "curl".to_string(),
                "-L".to_string(),
                "--fail".to_string(),
                "--silent".to_string(),
                "--show-error".to_string(),
                "--retry".to_string(),
                "2".to_string(),
                "--retry-delay".to_string(),
                "2".to_string(),
                "--connect-timeout".to_string(),
                "15".to_string(),
                "-o".to_string(),
                tmp.to_string_lossy().to_string(),
                (*url).to_string(),
            ];
            if run_download_with_progress(desc, &cmd, &tmp, total_bytes).is_ok() {
                ok = true;
            }
        } else if command_in_path("wget") {
            let total_bytes = probe_content_length_with_wget(url);
            let cmd = vec![
                "wget".to_string(),
                "-q".to_string(),
                "--tries=1".to_string(),
                "--timeout=15".to_string(),
                "-O".to_string(),
                tmp.to_string_lossy().to_string(),
                (*url).to_string(),
            ];
            if run_download_with_progress(desc, &cmd, &tmp, total_bytes).is_ok() {
                ok = true;
            }
        } else {
            return Err("Need `curl` or `wget` to download SIF.".to_string());
        }
        if ok {
            fs::rename(&tmp, dst).map_err(|e| {
                format!("Failed to move {} -> {}: {e}", tmp.display(), dst.display())
            })?;
            return Ok(());
        }
        errs.push((*url).to_string());
    }
    Err(format!(
        "All SIF download sources failed: {}",
        errs.join(", ")
    ))
}

fn run_download_with_progress(
    desc: &str,
    argv: &[String],
    target_path: &Path,
    total_bytes: Option<u64>,
) -> Result<(), String> {
    if argv.is_empty() {
        return Err(format!("{desc}: empty command"));
    }

    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..]).stdin(Stdio::null());
    if !io::stdout().is_terminal() {
        let out = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| format!("Failed to run download command: {e}"))?;
        if out.status.success() {
            return Ok(());
        }
        let mut msg = String::new();
        msg.push_str(&String::from_utf8_lossy(&out.stdout));
        msg.push_str(&String::from_utf8_lossy(&out.stderr));
        return Err(format!(
            "{desc} failed with exit={}: {}",
            super::exit_code(out.status),
            msg.trim()
        ));
    }

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to run download command: {e}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to capture download stdout".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Failed to capture download stderr".to_string())?;

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

    let start = Instant::now();
    let max_lines = 10usize;
    let mut rendered = false;
    let mut tail_mode_started = false;
    let mut prelog_spinner_shown = true;
    let mut streaming_title_started = false;
    let mut tail: VecDeque<String> = VecDeque::with_capacity(max_lines.max(1));
    let mut status: Option<ExitStatus> = None;
    let mut channel_open = true;
    let mut out_text = String::new();
    let mut err_text = String::new();
    let mut last_render = Instant::now();
    let mut streamed_lines_before_tail = 0usize;
    let mut saw_any_line = false;

    let downloaded0 = fs::metadata(target_path).map(|m| m.len()).unwrap_or(0);
    let mut last_progress_desc = download_desc_with_progress(desc, downloaded0, total_bytes);
    render_prelog_spinner_line_dlc(&last_progress_desc, start.elapsed())?;

    while channel_open {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok((is_err, line)) => {
                if prelog_spinner_shown && !tail_mode_started {
                    if !streaming_title_started {
                        render_streaming_title_init_dlc(&last_progress_desc, start.elapsed())?;
                        streaming_title_started = true;
                    }
                    prelog_spinner_shown = false;
                }
                saw_any_line = true;
                if is_err {
                    err_text.push_str(&line);
                    err_text.push('\n');
                } else {
                    out_text.push_str(&line);
                    out_text.push('\n');
                }
                if line.trim().is_empty() {
                    continue;
                }

                if max_lines > 0 {
                    if tail.len() == max_lines {
                        tail.pop_front();
                    }
                    tail.push_back(line.clone());
                }

                let downloaded = fs::metadata(target_path).map(|m| m.len()).unwrap_or(0);
                last_progress_desc = download_desc_with_progress(desc, downloaded, total_bytes);

                if max_lines == 0 {
                    tail_mode_started = true;
                    render_tail_block_dlc(
                        &last_progress_desc,
                        "",
                        start.elapsed(),
                        &tail,
                        max_lines,
                        rendered,
                    )?;
                    rendered = true;
                    last_render = Instant::now();
                    continue;
                }

                if !tail_mode_started {
                    if tail.len() < max_lines {
                        let width = dlc_terminal_line_width();
                        let trimmed = dlc_truncate_plain_line(&line, width);
                        if super::supports_color() {
                            println!("\x1b[2m{}\x1b[0m", trimmed);
                        } else {
                            println!("{trimmed}");
                        }
                        streamed_lines_before_tail += 1;
                        io::stdout()
                            .flush()
                            .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
                    } else {
                        if streamed_lines_before_tail > 0 {
                            let mut up = streamed_lines_before_tail;
                            if streaming_title_started {
                                up = up.saturating_add(1);
                            }
                            print!("\x1b[{}A", up);
                        }
                        render_tail_block_dlc(
                            &last_progress_desc,
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
                } else {
                    render_tail_block_dlc(
                        &last_progress_desc,
                        "",
                        start.elapsed(),
                        &tail,
                        max_lines,
                        rendered,
                    )?;
                    rendered = true;
                    last_render = Instant::now();
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if status.is_none() {
                    status = child
                        .try_wait()
                        .map_err(|e| format!("Failed to poll download status: {e}"))?;
                }
                if status.is_none() {
                    let downloaded = fs::metadata(target_path).map(|m| m.len()).unwrap_or(0);
                    last_progress_desc =
                        download_desc_with_progress(desc, downloaded, total_bytes);
                }
                if !tail_mode_started && !saw_any_line {
                    render_prelog_spinner_line_dlc(&last_progress_desc, start.elapsed())?;
                    prelog_spinner_shown = true;
                }
                if !tail_mode_started
                    && streaming_title_started
                    && last_render.elapsed() >= Duration::from_millis(100)
                {
                    render_streaming_title_only_dlc(
                        &last_progress_desc,
                        start.elapsed(),
                        streamed_lines_before_tail,
                    )?;
                    last_render = Instant::now();
                }
                if tail_mode_started && last_render.elapsed() >= Duration::from_millis(100) {
                    render_tail_title_only_dlc(&last_progress_desc, start.elapsed(), max_lines)?;
                    rendered = true;
                    last_render = Instant::now();
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                channel_open = false;
            }
        }

        if status.is_none() {
            status = child
                .try_wait()
                .map_err(|e| format!("Failed to poll download status: {e}"))?;
        }
    }

    let _ = out_handle.join();
    let _ = err_handle.join();

    let status = match status {
        Some(s) => s,
        None => child
            .wait()
            .map_err(|e| format!("Failed to wait download process: {e}"))?,
    };
    let elapsed = start.elapsed();

    if prelog_spinner_shown {
        print!("\r\x1b[2K");
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
    }
    if tail_mode_started {
        DLC_INLINE_STEP_LINE_OPEN.store(false, Ordering::Relaxed);
        if status.success() {
            render_tail_success_compact_dlc(desc, elapsed, max_lines)?;
            return Ok(());
        }
        render_tail_block_dlc(&last_progress_desc, "✘", elapsed, &tail, max_lines, rendered)?;
    } else if !status.success() && streaming_title_started {
        DLC_INLINE_STEP_LINE_OPEN.store(false, Ordering::Relaxed);
        render_streaming_failure_compact_dlc(&last_progress_desc, elapsed, streamed_lines_before_tail)?;
    } else {
        let width = dlc_terminal_line_width();
        let title = dlc_truncate_plain_line(
            &format!(
                "{} {}[{}]",
                if status.success() { "✔︎" } else { "✘" },
                desc,
                super::format_elapsed(elapsed)
            ),
            width,
        );
        if status.success() {
            if is_inline_step_desc(desc) {
                print!("\r\x1b[2K{}", super::style_green(&title));
                DLC_INLINE_STEP_LINE_OPEN.store(true, Ordering::Relaxed);
            } else {
                println!("{}", super::style_green(&title));
                DLC_INLINE_STEP_LINE_OPEN.store(false, Ordering::Relaxed);
            }
            io::stdout()
                .flush()
                .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
            return Ok(());
        }
        println!("{}", super::style_yellow(&title));
        DLC_INLINE_STEP_LINE_OPEN.store(false, Ordering::Relaxed);
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush DLC progress output: {e}"))?;
    }
    let mut msg = String::new();
    msg.push_str(out_text.trim());
    if !msg.is_empty() && !err_text.trim().is_empty() {
        msg.push('\n');
    }
    msg.push_str(err_text.trim());
    Err(format!(
        "{desc} failed with exit={}: {}",
        super::exit_code(status),
        msg.trim()
    ))
}

fn human_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn download_desc_with_progress(base: &str, downloaded: u64, total_bytes: Option<u64>) -> String {
    match total_bytes {
        Some(total) if total > 0 => format!("{base} [{}/{}]", human_bytes(downloaded), human_bytes(total)),
        _ => format!("{base} [{}]", human_bytes(downloaded)),
    }
}

fn probe_content_length_with_curl(url: &str) -> Option<u64> {
    let out = Command::new("curl")
        .arg("-L")
        .arg("--silent")
        .arg("--show-error")
        .arg("--head")
        .arg("--connect-timeout")
        .arg("10")
        .arg(url)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_content_length_from_headers(&String::from_utf8_lossy(&out.stdout))
}

fn probe_content_length_with_wget(url: &str) -> Option<u64> {
    let out = Command::new("wget")
        .arg("--spider")
        .arg("--server-response")
        .arg("--timeout=10")
        .arg("--tries=1")
        .arg(url)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_content_length_from_headers(&String::from_utf8_lossy(&out.stderr))
}

fn parse_content_length_from_headers(text: &str) -> Option<u64> {
    let mut out: Option<u64> = None;
    for line in text.lines() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        let lower = s.to_ascii_lowercase();
        if !lower.starts_with("content-length:") {
            continue;
        }
        let n = s.split_once(':').map(|(_, v)| v.trim()).and_then(|v| {
            let digits: String = v.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() {
                None
            } else {
                digits.parse::<u64>().ok()
            }
        });
        if let Some(v) = n {
            out = Some(v);
        }
    }
    out
}

fn save_tool_cache_entries(
    python: &Path,
    db_path: &Path,
    runtime_sig: &str,
    entries: &[ToolCacheEntry],
) -> Result<(), String> {
    if let Some(parent) = db_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let payload = entries
        .iter()
        .map(|e| {
            format!(
                "{}\t{}\t{}\t{}",
                sanitize_db_field(&e.tool),
                sanitize_db_field(&e.backend),
                sanitize_db_field(&e.locator),
                sanitize_db_field(&e.runtime_sig)
            )
        })
        .collect::<Vec<String>>()
        .join("\n");
    let script = r#"
import sqlite3, sys
db, slot, meta_key, runtime_sig, payload = sys.argv[1:6]
conn = sqlite3.connect(db, timeout=30)
conn.execute("""
CREATE TABLE IF NOT EXISTS dlc_tool_cache (
    slot TEXT NOT NULL,
    tool TEXT NOT NULL,
    backend TEXT NOT NULL,
    locator TEXT NOT NULL,
    runtime_sig TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (slot, tool)
)
""")
conn.execute("DELETE FROM dlc_tool_cache WHERE slot = ?", (slot,))
conn.execute(
    "INSERT INTO dlc_tool_cache (slot, tool, backend, locator, runtime_sig, updated_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
    (slot, meta_key, "", "", runtime_sig),
)
for raw in payload.splitlines():
    if not raw.strip():
        continue
    parts = raw.split("\t")
    if len(parts) < 4:
        continue
    tool, backend, locator, sig = parts[:4]
    conn.execute(
        "INSERT INTO dlc_tool_cache (slot, tool, backend, locator, runtime_sig, updated_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
        (slot, tool, backend, locator, sig),
    )
conn.commit()
conn.close()
"#;
    let out = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(db_path.to_string_lossy().to_string())
        .arg(DLC_SLOT)
        .arg(DLC_TOOL_CACHE_META_KEY)
        .arg(runtime_sig)
        .arg(payload)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("Failed to save DLC tool cache: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let mut msg = String::new();
    msg.push_str(&String::from_utf8_lossy(&out.stdout));
    msg.push_str(&String::from_utf8_lossy(&out.stderr));
    Err(format!(
        "Failed to save DLC tool cache (exit={}): {}",
        super::exit_code(out.status),
        msg.trim()
    ))
}

fn load_tool_cache_entries(
    python: &Path,
    db_path: &Path,
) -> Result<(Option<String>, Vec<ToolCacheEntry>), String> {
    if !db_path.exists() {
        return Ok((None, Vec::new()));
    }
    let script = r#"
import sqlite3, sys
db, slot = sys.argv[1:3]
conn = sqlite3.connect(db, timeout=30)
conn.execute("""
CREATE TABLE IF NOT EXISTS dlc_tool_cache (
    slot TEXT NOT NULL,
    tool TEXT NOT NULL,
    backend TEXT NOT NULL,
    locator TEXT NOT NULL,
    runtime_sig TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (slot, tool)
)
""")
rows = conn.execute(
    "SELECT tool, backend, locator, runtime_sig FROM dlc_tool_cache WHERE slot = ? ORDER BY tool",
    (slot,)
).fetchall()
conn.close()
for row in rows:
    vals = ["" if v is None else str(v).replace("\t", " ").replace("\n", " ") for v in row]
    print("\t".join(vals))
"#;
    let out = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(db_path.to_string_lossy().to_string())
        .arg(DLC_SLOT)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("Failed to load DLC tool cache: {e}"))?;
    if !out.status.success() {
        let mut msg = String::new();
        msg.push_str(&String::from_utf8_lossy(&out.stdout));
        msg.push_str(&String::from_utf8_lossy(&out.stderr));
        if msg.contains("unable to open database file") {
            return Ok((None, Vec::new()));
        }
        return Err(format!(
            "Failed to load DLC tool cache (exit={}): {}",
            super::exit_code(out.status),
            msg.trim()
        ));
    }
    let mut meta_sig: Option<String> = None;
    let mut entries: Vec<ToolCacheEntry> = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let p = line.trim();
        if p.is_empty() {
            continue;
        }
        let parts: Vec<&str> = p.split('\t').collect();
        if parts.len() < 4 {
            continue;
        }
        let tool = parts[0].trim().to_string();
        let backend = parts[1].trim().to_string();
        let locator = parts[2].trim().to_string();
        let runtime_sig = parts[3].trim().to_string();
        if tool == DLC_TOOL_CACHE_META_KEY {
            meta_sig = Some(runtime_sig);
            continue;
        }
        entries.push(ToolCacheEntry {
            tool,
            backend,
            locator,
            runtime_sig,
        });
    }
    Ok((meta_sig, entries))
}

fn sanitize_db_field(raw: &str) -> String {
    raw.replace('\t', " ")
        .replace('\n', " ")
        .replace('\r', " ")
}

fn save_runtime_record(
    python: &Path,
    db_path: &Path,
    record: &RuntimeRecord,
) -> Result<(), String> {
    let installed_csv = record.installed_tools.join(",");
    let script = r#"
import sqlite3, sys
db, slot, method_id, runtime_mode, image_tag, conda_env, sif_path, installed_csv, status, updated_at = sys.argv[1:11]
conn = sqlite3.connect(db, timeout=30)
conn.execute("""
CREATE TABLE IF NOT EXISTS dlc_builds (
    slot TEXT PRIMARY KEY,
    method_id INTEGER NOT NULL,
    runtime_mode TEXT NOT NULL,
    image_tag TEXT NOT NULL,
    conda_env TEXT NOT NULL,
    sif_path TEXT NOT NULL,
    installed_tools_csv TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
""")
conn.execute("DELETE FROM dlc_builds")
conn.execute(
    "INSERT INTO dlc_builds (slot, method_id, runtime_mode, image_tag, conda_env, sif_path, installed_tools_csv, status, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    (slot, int(method_id), runtime_mode, image_tag, conda_env, sif_path, installed_csv, status, updated_at),
)
conn.commit()
conn.close()
"#;
    let out = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(db_path.to_string_lossy().to_string())
        .arg(DLC_SLOT)
        .arg(record.method_id.to_string())
        .arg(&record.runtime_mode)
        .arg(&record.image_tag)
        .arg(&record.conda_env)
        .arg(&record.sif_path)
        .arg(installed_csv)
        .arg(&record.status)
        .arg(&record.updated_at)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("Failed to save DLC runtime DB record: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let mut msg = String::new();
    msg.push_str(&String::from_utf8_lossy(&out.stdout));
    msg.push_str(&String::from_utf8_lossy(&out.stderr));
    Err(format!(
        "Failed to save DLC runtime DB record (exit={}): {}",
        super::exit_code(out.status),
        msg.trim()
    ))
}

fn clear_runtime_record(python: &Path, db_path: &Path) -> Result<(), String> {
    if let Some(parent) = db_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let script = r#"
import sqlite3, sys
db, slot = sys.argv[1:3]
conn = sqlite3.connect(db, timeout=30)
conn.execute("""
CREATE TABLE IF NOT EXISTS dlc_builds (
    slot TEXT PRIMARY KEY,
    method_id INTEGER NOT NULL,
    runtime_mode TEXT NOT NULL,
    image_tag TEXT NOT NULL,
    conda_env TEXT NOT NULL,
    sif_path TEXT NOT NULL,
    installed_tools_csv TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
""")
conn.execute("""
CREATE TABLE IF NOT EXISTS dlc_tool_cache (
    slot TEXT NOT NULL,
    tool TEXT NOT NULL,
    backend TEXT NOT NULL,
    locator TEXT NOT NULL,
    runtime_sig TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (slot, tool)
)
""")
conn.execute("DELETE FROM dlc_builds WHERE slot = ?", (slot,))
conn.execute("DELETE FROM dlc_tool_cache WHERE slot = ?", (slot,))
conn.commit()
conn.close()
"#;
    let out = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(db_path.to_string_lossy().to_string())
        .arg(DLC_SLOT)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("Failed to clear DLC runtime DB record: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let mut msg = String::new();
    msg.push_str(&String::from_utf8_lossy(&out.stdout));
    msg.push_str(&String::from_utf8_lossy(&out.stderr));
    Err(format!(
        "Failed to clear DLC runtime DB record (exit={}): {}",
        super::exit_code(out.status),
        msg.trim()
    ))
}

fn load_runtime_record(python: &Path, db_path: &Path) -> Result<Option<RuntimeRecord>, String> {
    if !db_path.exists() {
        return Ok(None);
    }
    let script = r#"
import sqlite3, sys
db, slot = sys.argv[1:3]
conn = sqlite3.connect(db, timeout=30)
conn.execute("""
CREATE TABLE IF NOT EXISTS dlc_builds (
    slot TEXT PRIMARY KEY,
    method_id INTEGER NOT NULL,
    runtime_mode TEXT NOT NULL,
    image_tag TEXT NOT NULL,
    conda_env TEXT NOT NULL,
    sif_path TEXT NOT NULL,
    installed_tools_csv TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
""")
row = conn.execute(
    "SELECT method_id, runtime_mode, image_tag, conda_env, sif_path, installed_tools_csv, status, updated_at FROM dlc_builds WHERE slot = ?",
    (slot,)
).fetchone()
conn.close()
if row is None:
    sys.exit(2)
vals = ["" if v is None else str(v).replace("\t", " ") for v in row]
print("\t".join(vals))
"#;
    let out = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(db_path.to_string_lossy().to_string())
        .arg(DLC_SLOT)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("Failed to load DLC runtime DB record: {e}"))?;
    let code = super::exit_code(out.status);
    if code == 2 {
        return Ok(None);
    }
    if !out.status.success() {
        let mut msg = String::new();
        msg.push_str(&String::from_utf8_lossy(&out.stdout));
        msg.push_str(&String::from_utf8_lossy(&out.stderr));
        if msg.contains("unable to open database file") {
            return Ok(None);
        }
        return Err(format!(
            "Failed to load DLC runtime DB record (exit={}): {}",
            code,
            msg.trim()
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        return Ok(None);
    }
    let parts: Vec<&str> = text.split('\t').collect();
    if parts.len() < 8 {
        return Err("Invalid DLC runtime DB record format.".to_string());
    }
    let method_id = parts[0]
        .parse::<i32>()
        .map_err(|_| "Invalid method_id in DLC runtime DB record.".to_string())?;
    let installed_tools = if parts[5].trim().is_empty() {
        Vec::new()
    } else {
        parts[5]
            .split(',')
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .collect()
    };
    Ok(Some(RuntimeRecord {
        method_id,
        runtime_mode: parts[1].to_string(),
        image_tag: parts[2].to_string(),
        conda_env: parts[3].to_string(),
        sif_path: parts[4].to_string(),
        installed_tools,
        status: parts[6].to_string(),
        updated_at: parts[7].to_string(),
    }))
}

fn conda_env_named_exists(conda_bin: &Path, env_name: &str) -> bool {
    let out = Command::new(conda_bin)
        .arg("env")
        .arg("list")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .output();
    let Ok(out) = out else {
        return false;
    };
    if !out.status.success() {
        return false;
    }
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = s.split_whitespace().collect();
        if cols.iter().any(|x| *x == env_name) {
            return true;
        }
    }
    false
}

fn write_conda_dlc_yml(
    runtime_home: &Path,
    missing_tools: &[String],
    use_mirror: bool,
) -> Result<PathBuf, String> {
    let yml_dir = runtime_home.join("dlc").join("conda");
    fs::create_dir_all(&yml_dir)
        .map_err(|e| format!("Failed to create {}: {e}", yml_dir.display()))?;
    let yml_path = yml_dir.join(if use_mirror {
        "dlc.mirror.yml"
    } else {
        "dlc.official.yml"
    });

    let mut content = String::new();
    content.push_str(&format!("name: {CONDA_ENV}\n"));
    content.push_str("channels:\n");
    if use_mirror {
        content.push_str(&format!("  - {CONDA_BIOCONDA_MIRROR_URL}\n"));
        content.push_str(&format!("  - {CONDA_FORGE_MIRROR_URL}\n"));
    } else {
        content.push_str(&format!("  - {CONDA_BIOCONDA_OFFICIAL}\n"));
        content.push_str(&format!("  - {CONDA_FORGE_OFFICIAL}\n"));
    }
    content.push_str("dependencies:\n");
    content.push_str("  - python=3.10\n");
    for pkg in toolchain_packages_for_tools(missing_tools) {
        content.push_str(&format!("  - {pkg}\n"));
    }
    fs::write(&yml_path, content)
        .map_err(|e| format!("Failed to write {}: {e}", yml_path.display()))?;
    Ok(yml_path)
}

fn build_conda_apply_dlc_yml_cmd(
    conda_bin: &str,
    env_exists: bool,
    yml_path: &Path,
) -> Vec<String> {
    let mut cmd = vec![
        conda_bin.to_string(),
        "env".to_string(),
        if env_exists {
            "update".to_string()
        } else {
            "create".to_string()
        },
        "-n".to_string(),
        CONDA_ENV.to_string(),
        "-f".to_string(),
        yml_path.to_string_lossy().to_string(),
    ];
    if env_exists {
        cmd.push("--prune".to_string());
    }
    cmd
}

fn run_conda_dlc_yml_with_fallback(
    conda_bin: &str,
    env_exists: bool,
    mirror_yml: &Path,
    official_yml: &Path,
    desc: &str,
) -> Result<(), String> {
    let mirror_cmd = build_conda_apply_dlc_yml_cmd(conda_bin, env_exists, mirror_yml);
    match run_cmd_tail(desc, &mirror_cmd, true, 10) {
        Ok(()) => Ok(()),
        Err(mirror_err) => {
            eprintln!(
                "{}",
                super::style_yellow(
                    "conda dlc.yml mirror channels failed, fallback to official channels.",
                )
            );
            let official_cmd = build_conda_apply_dlc_yml_cmd(conda_bin, env_exists, official_yml);
            run_cmd_tail(desc, &official_cmd, true, 10)
                .map_err(|e| format!("{e}\nMirror attempt error: {mirror_err}"))
        }
    }
}

fn install_tools_stepwise_env(
    micromamba_bin: &str,
    env_prefix: &Path,
    tools: &[String],
    action: &str,
    start_index: usize,
    total_steps: usize,
) -> Result<usize, String> {
    let ordered = ordered_required_tools(tools);
    if ordered.is_empty() {
        return Ok(start_index);
    }
    let total = total_steps.max(start_index.saturating_sub(1) + ordered.len());
    for (idx, tool) in ordered.iter().enumerate() {
        let pkgs = toolchain_packages_for_tools(std::slice::from_ref(tool));
        if pkgs.is_empty() {
            continue;
        }
        let cmd = vec![
            micromamba_bin.to_string(),
            "install".to_string(),
            "-p".to_string(),
            env_prefix.to_string_lossy().to_string(),
        ];
        let mut packages = vec!["-y".to_string()];
        packages.extend(pkgs);
        let step_no = start_index + idx;
        let desc = format!("[{}/{}] {} {} via env", step_no, total, action, tool);
        run_cmd_tail_with_conda_channel_fallback(&desc, &cmd, &packages, 10)?;
    }
    Ok(start_index + ordered.len())
}

fn ordered_required_tools(tools: &[String]) -> Vec<String> {
    let mut raw: BTreeSet<String> = BTreeSet::new();
    for t in tools {
        let x = t.trim();
        if !x.is_empty() {
            raw.insert(x.to_string());
        }
    }
    let mut ordered: Vec<String> = Vec::new();
    for t in REQUIRED_TOOLS {
        if raw.remove(t) {
            ordered.push(t.to_string());
        }
    }
    for x in raw {
        ordered.push(x);
    }
    ordered
}

fn should_force_conda_run(tool: &str) -> bool {
    CONDA_FORCE_RUNTIME_TOOLS.contains(&tool)
}

fn conda_required_tools_for_runtime() -> Vec<String> {
    let mut merged: BTreeSet<String> = BTreeSet::new();
    for t in missing_host_tools() {
        merged.insert(t);
    }
    for t in CONDA_FORCE_RUNTIME_TOOLS {
        merged.insert(t.to_string());
    }
    ordered_required_tools(&merged.into_iter().collect::<Vec<String>>())
}

fn conda_env_exists(conda_bin: &Path, env_name: &str) -> bool {
    Command::new(conda_bin)
        .arg("run")
        .arg("--no-capture-output")
        .arg("-n")
        .arg(env_name)
        .arg("python")
        .arg("-V")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn docker_daemon_running() -> (bool, String) {
    let out = Command::new("docker")
        .arg("info")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .output();
    let Ok(out) = out else {
        return (false, "failed to call docker info".to_string());
    };
    if out.status.success() {
        return (true, String::new());
    }
    let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
    if err.is_empty() {
        (false, "docker daemon unavailable".to_string())
    } else {
        (false, err)
    }
}

fn singularity_runtime_ready(singularity_bin: &Path) -> (bool, String) {
    let out = Command::new(singularity_bin)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .output();
    let Ok(out) = out else {
        return (false, "failed to call singularity --version".to_string());
    };
    if out.status.success() {
        return (true, String::new());
    }
    let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
    if err.contains("singularity.conf") {
        return (
            false,
            "missing singularity configuration file (e.g. /etc/singularity/singularity.conf)"
                .to_string(),
        );
    }
    if err.is_empty() {
        (false, "singularity runtime unavailable".to_string())
    } else {
        (false, err)
    }
}

fn docker_default_platform() -> Option<&'static str> {
    if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        return Some("linux/amd64");
    }
    None
}

fn command_in_path(bin: &str) -> bool {
    find_bin(bin).is_some()
}

fn find_bin(bin: &str) -> Option<PathBuf> {
    let p = PathBuf::from(bin);
    if p.components().count() > 1 {
        if p.exists() {
            return Some(p);
        }
        return None;
    }
    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        let cand = dir.join(bin);
        if cand.exists() && cand.is_file() {
            return Some(cand);
        }
        #[cfg(windows)]
        {
            for ext in [".exe", ".cmd", ".bat"] {
                let c = dir.join(format!("{bin}{ext}"));
                if c.exists() && c.is_file() {
                    return Some(c);
                }
            }
        }
    }
    None
}

fn collect_docker_mounts(args: &[String], cwd: &Path) -> Vec<PathBuf> {
    let mut out: BTreeSet<PathBuf> = BTreeSet::new();
    if let Ok(c) = fs::canonicalize(cwd) {
        out.insert(c);
    } else {
        out.insert(cwd.to_path_buf());
    }

    for arg in args {
        maybe_add_mount(&mut out, cwd, arg);
        if let Some((_, rhs)) = arg.split_once('=') {
            maybe_add_mount(&mut out, cwd, rhs);
        }
    }
    out.into_iter().collect()
}

fn maybe_add_mount(out: &mut BTreeSet<PathBuf>, cwd: &Path, raw: &str) {
    let v = raw.trim();
    if v.is_empty() || v.starts_with('-') || v.contains("://") {
        return;
    }
    let p = PathBuf::from(v);
    let abs = if p.is_absolute() { p } else { cwd.join(p) };
    if abs.exists() {
        if abs.is_dir() {
            if let Ok(c) = fs::canonicalize(&abs) {
                out.insert(c);
            } else {
                out.insert(abs);
            }
        } else if let Some(parent) = abs.parent() {
            if parent.exists() {
                if let Ok(c) = fs::canonicalize(parent) {
                    out.insert(c);
                } else {
                    out.insert(parent.to_path_buf());
                }
            }
        }
        return;
    }
    if abs.is_absolute() {
        if let Some(parent) = abs.parent() {
            if parent.exists() {
                if let Ok(c) = fs::canonicalize(parent) {
                    out.insert(c);
                } else {
                    out.insert(parent.to_path_buf());
                }
            }
        }
    }
}

fn current_uid_gid() -> Option<(u32, u32)> {
    if cfg!(windows) {
        return None;
    }
    let uid = Command::new("id")
        .arg("-u")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let gid = Command::new("id")
        .arg("-g")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !(uid.status.success() && gid.status.success()) {
        return None;
    }
    let u = String::from_utf8_lossy(&uid.stdout)
        .trim()
        .parse::<u32>()
        .ok()?;
    let g = String::from_utf8_lossy(&gid.stdout)
        .trim()
        .parse::<u32>()
        .ok()?;
    Some((u, g))
}

fn mark(ok: bool) -> &'static str {
    if ok {
        "✔︎"
    } else {
        "✘"
    }
}

fn now_utc_epoch() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_secs();
    ts.to_string()
}
