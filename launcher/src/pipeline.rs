use std::collections::HashSet;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub(crate) enum Scheduler {
    Nohup,
    Csub,
}

#[derive(Clone, Debug)]
pub(crate) struct StepItem {
    pub id: String,
    pub outputs: Vec<PathBuf>,
}

#[derive(Clone, Debug)]
pub(crate) struct PipelineStep {
    pub id: String,
    pub name: String,
    pub eta_minutes: Option<u64>,
    pub commands: Vec<String>,
    pub inputs: Vec<PathBuf>,
    pub outputs: Vec<PathBuf>,
    pub items: Vec<StepItem>,
}

#[derive(Clone, Debug)]
pub(crate) struct PipelineOptions {
    pub scheduler: Scheduler,
    pub nohup_max_jobs: usize,
    pub skip_if_outputs_exist: bool,
    pub poll_sec: f64,
    pub detect_failed_logs: bool,
    pub no_progress_timeout_sec: Option<u64>,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            scheduler: Scheduler::Nohup,
            nohup_max_jobs: 1,
            skip_if_outputs_exist: true,
            poll_sec: 2.0,
            detect_failed_logs: true,
            no_progress_timeout_sec: None,
        }
    }
}

pub(crate) trait PipelineHook {
    fn on_item_completed(&mut self, _step: &PipelineStep, _item: &StepItem) -> Result<(), String> {
        Ok(())
    }
}

impl<F> PipelineHook for F
where
    F: FnMut(&PipelineStep, &StepItem) -> Result<(), String>,
{
    fn on_item_completed(&mut self, step: &PipelineStep, item: &StepItem) -> Result<(), String> {
        self(step, item)
    }
}

pub(crate) fn run_pipeline_with_hook<H: PipelineHook>(
    workdir: &Path,
    steps: &[PipelineStep],
    opts: &PipelineOptions,
    hook: &mut H,
) -> Result<(), String> {
    fs::create_dir_all(workdir.join("tmp"))
        .map_err(|e| format!("Failed to create {}: {e}", workdir.join("tmp").display()))?;
    fs::create_dir_all(workdir.join("log"))
        .map_err(|e| format!("Failed to create {}: {e}", workdir.join("log").display()))?;

    for (idx, step) in steps.iter().enumerate() {
        let step_text = format_step_label(idx, steps.len(), step);
        if opts.skip_if_outputs_exist && all_outputs_ready(&step.outputs) {
            super::print_success_line(&format!("{step_text} ...Skipped"));
            continue;
        }
        if !all_paths_exist(&step.inputs) {
            return Err(format!(
                "{step_text} inputs not ready: {}",
                step.inputs
                    .iter()
                    .filter(|x| !x.exists())
                    .map(|x| x.display().to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ));
        }

        let sh = workdir.join("tmp").join(format!("{}.sh", idx));
        let script = build_step_script(step, opts);
        fs::write(&sh, script).map_err(|e| format!("Failed to write {}: {e}", sh.display()))?;
        set_script_executable(&sh)?;

        let existing_error_logs = if opts.detect_failed_logs && matches!(opts.scheduler, Scheduler::Csub) {
            collect_existing_item_error_logs(workdir, step)?
        } else {
            HashSet::new()
        };
        run_step_script(workdir, &sh, opts)?;
        wait_step_outputs(workdir, &step_text, step, opts, hook, &existing_error_logs)?;
        let elapsed = super::format_elapsed(Duration::from_secs(0));
        let _ = elapsed;
    }
    Ok(())
}

fn format_step_label(step_idx: usize, total_steps: usize, step: &PipelineStep) -> String {
    let step_name = if step.name.trim().is_empty() {
        step.id.as_str()
    } else {
        step.name.as_str()
    };
    let mut title = format!("Step {}/{}: {}", step_idx + 1, total_steps, step_name);
    if let Some(eta) = step.eta_minutes {
        title.push(' ');
        title.push('(');
        title.push_str(&format_eta(eta));
        title.push(')');
    }
    title
}

fn format_eta(minutes: u64) -> String {
    if minutes < 60 {
        return format!("~{}m", minutes);
    }
    let h = minutes / 60;
    let m = minutes % 60;
    if m == 0 {
        format!("~{}h", h)
    } else {
        format!("~{}h{}m", h, m)
    }
}

fn build_step_script(step: &PipelineStep, opts: &PipelineOptions) -> String {
    if step.commands.is_empty() {
        return "echo 'empty step'\n".to_string();
    }
    match opts.scheduler {
        Scheduler::Csub => {
            let mut text = step.commands.join("\n");
            if !text.ends_with('\n') {
                text.push('\n');
            }
            text
        }
        Scheduler::Nohup => {
            if opts.nohup_max_jobs > 0 {
                let mut parts = vec![format!("MAX_JOBS={}", opts.nohup_max_jobs)];
                for line in &step.commands {
                    parts.push(
                        "while [ \"$(jobs -pr | wc -l)\" -ge \"$MAX_JOBS\" ]; do sleep 2; done"
                            .to_string(),
                    );
                    parts.push(format!("{line} &"));
                }
                parts.push("wait".to_string());
                parts.join("\n") + "\n"
            } else {
                let mut lines: Vec<String> =
                    step.commands.iter().map(|x| format!("{x} &")).collect();
                lines.push("wait".to_string());
                lines.join("\n") + "\n"
            }
        }
    }
}

fn run_step_script(workdir: &Path, sh: &Path, opts: &PipelineOptions) -> Result<(), String> {
    let mut cmd = Command::new("bash");
    cmd.arg(sh)
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(if matches!(opts.scheduler, Scheduler::Csub) {
            Stdio::piped()
        } else {
            Stdio::piped()
        });
    let out = cmd
        .output()
        .map_err(|e| format!("Failed to run {}: {e}", sh.display()))?;
    if out.status.success() {
        return Ok(());
    }
    let mut msg = String::new();
    msg.push_str(&String::from_utf8_lossy(&out.stdout));
    msg.push_str(&String::from_utf8_lossy(&out.stderr));
    Err(format!(
        "Step script failed with exit={}: {}",
        super::exit_code(out.status),
        msg.trim()
    ))
}

fn wait_step_outputs(
    workdir: &Path,
    step_text: &str,
    step: &PipelineStep,
    opts: &PipelineOptions,
    hook: &mut impl PipelineHook,
    ignore_error_logs: &HashSet<PathBuf>,
) -> Result<(), String> {
    let stall_timeout = resolve_no_progress_timeout(step, opts);

    if step.items.is_empty() {
        let total = 1usize;
        let start = Instant::now();
        let poll = Duration::from_secs_f64(opts.poll_sec.max(0.5));
        let mut last_progress = Instant::now();
        let mut last_done = 0usize;
        let mut spinner_tick = 0usize;

        loop {
            let done = if all_outputs_ready(&step.outputs) {
                total
            } else {
                0
            };
            if done > last_done {
                last_done = done;
                last_progress = Instant::now();
            }
            if done >= total {
                break;
            }

            if let Some(timeout) = stall_timeout {
                if done < total && last_progress.elapsed() >= timeout {
                    return Err(format!(
                        "{step_text} has no progress for {} (done {done}/{total}).",
                        super::format_elapsed(timeout)
                    ));
                }
            }

            render_step_progress(step_text, done, total, start.elapsed(), spinner_tick)?;
            spinner_tick = spinner_tick.wrapping_add(1);
            thread::sleep(poll);
        }
        clear_progress_line_if_tty()?;
        super::print_success_line(&format!(
            "{} ...Finished [{}]",
            step_text,
            super::format_elapsed(start.elapsed())
        ));
        return Ok(());
    }

    let total = step.items.len();
    let start = Instant::now();
    let mut last_error_scan = Instant::now();
    let poll = Duration::from_secs_f64(opts.poll_sec.max(0.5));
    let mut pending: Vec<usize> = Vec::new();
    let mut done = 0usize;
    let mut last_progress = Instant::now();
    let mut spinner_tick = 0usize;

    for (idx, item) in step.items.iter().enumerate() {
        if all_outputs_ready(&item.outputs) {
            done += 1;
        } else {
            pending.push(idx);
        }
    }

    loop {
        if !pending.is_empty() {
            let mut next_pending: Vec<usize> = Vec::with_capacity(pending.len());
            for idx in &pending {
                let item = &step.items[*idx];
                if all_outputs_ready(&item.outputs) {
                    done += 1;
                    last_progress = Instant::now();
                    hook.on_item_completed(step, item)?;
                } else {
                    next_pending.push(*idx);
                }
            }
            pending = next_pending;
        }

        if done >= total {
            break;
        }

        if opts.detect_failed_logs && matches!(opts.scheduler, Scheduler::Csub) {
            let scan_interval = Duration::from_secs_f64((opts.poll_sec * 3.0).max(5.0));
            if last_error_scan.elapsed() >= scan_interval {
                last_error_scan = Instant::now();
                let failed =
                    find_failed_item_logs_for_pending(workdir, step, &pending, 3, 4, ignore_error_logs)?;
                if !failed.is_empty() {
                    return Err(format!(
                        "{step_text} detected failed subtasks. Example stderr:\n{}",
                        failed
                            .iter()
                            .map(|x| format!("- {x}"))
                            .collect::<Vec<String>>()
                            .join("\n")
                    ));
                }
            }
        }
        if let Some(timeout) = stall_timeout {
            if done < total && last_progress.elapsed() >= timeout {
                return Err(format!(
                    "{step_text} has no progress for {} (done {done}/{total}).",
                    super::format_elapsed(timeout)
                ));
            }
        }

        render_step_progress(step_text, done, total, start.elapsed(), spinner_tick)?;
        spinner_tick = spinner_tick.wrapping_add(1);
        thread::sleep(poll);
    }
    clear_progress_line_if_tty()?;
    super::print_success_line(&format!(
        "{} ...Finished [{}]",
        step_text,
        super::format_elapsed(start.elapsed())
    ));
    Ok(())
}

fn render_step_progress(
    step_text: &str,
    done: usize,
    total: usize,
    elapsed: Duration,
    spinner_tick: usize,
) -> Result<(), String> {
    if !io::stdout().is_terminal() {
        return Ok(());
    }
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let idx = spinner_tick % frames.len();
    let line = format!(
        "\r{} {} [{}/{}] [{}]",
        frames[idx],
        step_text,
        done,
        total,
        super::format_elapsed(elapsed)
    );
    if super::supports_color() {
        print!("{}", super::style_green(&line));
    } else {
        print!("{line}");
    }
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush progress output: {e}"))?;
    Ok(())
}

fn clear_progress_line_if_tty() -> Result<(), String> {
    if !io::stdout().is_terminal() {
        return Ok(());
    }
    print!("\r\x1b[2K");
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush progress output: {e}"))?;
    Ok(())
}

fn all_paths_exist(paths: &[PathBuf]) -> bool {
    paths.iter().all(|p| p.exists())
}

pub(crate) fn all_outputs_ready(paths: &[PathBuf]) -> bool {
    paths.iter().all(|p| is_output_ready(p))
}

pub(crate) fn is_output_ready(path: &Path) -> bool {
    if !path.exists() {
        return false;
    }
    if path.is_dir() {
        return true;
    }
    let md = match fs::metadata(path) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if !md.is_file() {
        return false;
    }
    if requires_nonempty_output(path) && md.len() == 0 {
        return false;
    }
    std::fs::File::open(path).is_ok()
}

fn requires_nonempty_output(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|x| x.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if name.ends_with(".finished") || name.ends_with(".done") || name.ends_with(".ok") {
        return false;
    }
    true
}

fn resolve_no_progress_timeout(step: &PipelineStep, opts: &PipelineOptions) -> Option<Duration> {
    if let Some(sec) = opts.no_progress_timeout_sec {
        if sec == 0 {
            return None;
        }
        return Some(Duration::from_secs(sec));
    }
    if matches!(opts.scheduler, Scheduler::Nohup) && opts.nohup_max_jobs <= 1 {
        return None;
    }
    let eta_sec = step.eta_minutes.unwrap_or(0).saturating_mul(120);
    let sec = eta_sec.max(4 * 3600);
    Some(Duration::from_secs(sec))
}

fn set_script_executable(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(path)
            .map_err(|e| format!("Failed to stat {}: {e}", path.display()))?
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms)
            .map_err(|e| format!("Failed to chmod {}: {e}", path.display()))?;
    }
    Ok(())
}

fn find_failed_item_logs_for_pending(
    workdir: &Path,
    step: &PipelineStep,
    pending_idx: &[usize],
    max_items: usize,
    max_lines: usize,
    ignore_error_logs: &HashSet<PathBuf>,
) -> Result<Vec<String>, String> {
    let log_dir = workdir.join("log");
    if !log_dir.exists() {
        return Ok(Vec::new());
    }

    let mut snippets: Vec<String> = Vec::new();
    for idx in pending_idx {
        let Some(item) = step.items.get(*idx) else {
            continue;
        };
        if all_outputs_ready(&item.outputs) {
            continue;
        }
        let safe_id = safe_job_label(&item.id);
        let mut err_files: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(&log_dir)
            .map_err(|e| format!("Failed to read {}: {e}", log_dir.display()))?
        {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
                continue;
            };
            if (name == format!("{safe_id}.e")
                || (name.starts_with(&(safe_id.clone() + ".")) && name.ends_with(".e")))
                && fs::metadata(&path).map(|m| m.len()).unwrap_or(0) > 0
            {
                if ignore_error_logs.contains(&path) {
                    continue;
                }
                err_files.push(path);
            }
        }
        err_files.sort();

        for ef in err_files {
            let text = fs::read_to_string(&ef).unwrap_or_default();
            if text.trim().is_empty() {
                continue;
            }
            let fatal_lines: Vec<String> = text
                .lines()
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty() && is_likely_fatal_stderr_line(x))
                .collect();
            if fatal_lines.is_empty() {
                continue;
            }
            let head = fatal_lines
                .into_iter()
                .take(max_lines)
                .collect::<Vec<String>>()
                .join(" | ");
            snippets.push(format!("{}: {}", item.id, head));
            if snippets.len() >= max_items {
                return Ok(snippets);
            }
        }
    }
    Ok(snippets)
}

fn collect_existing_item_error_logs(
    workdir: &Path,
    step: &PipelineStep,
) -> Result<HashSet<PathBuf>, String> {
    let mut ignore: HashSet<PathBuf> = HashSet::new();
    let log_dir = workdir.join("log");
    if !log_dir.exists() || step.items.is_empty() {
        return Ok(ignore);
    }

    let mut safe_ids: Vec<String> = step.items.iter().map(|x| safe_job_label(&x.id)).collect();
    safe_ids.sort();
    safe_ids.dedup();

    for entry in
        fs::read_dir(&log_dir).map_err(|e| format!("Failed to read {}: {e}", log_dir.display()))?
    {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if fs::metadata(&path).map(|m| m.len()).unwrap_or(0) == 0 {
            continue;
        }
        let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
            continue;
        };
        if !name.ends_with(".e") {
            continue;
        }
        for safe_id in &safe_ids {
            if name == format!("{safe_id}.e")
                || (name.starts_with(&(safe_id.clone() + ".")) && name.ends_with(".e"))
            {
                ignore.insert(path.clone());
                break;
            }
        }
    }
    Ok(ignore)
}

fn is_likely_fatal_stderr_line(line: &str) -> bool {
    let s = line.trim().to_ascii_lowercase();
    if s.is_empty() {
        return false;
    }
    if is_benign_stderr_line(&s) {
        return false;
    }
    let keys = [
        "traceback (most recent call last)",
        "runtimeerror",
        "exception",
        "fatal",
        "error",
        "failed",
        "no such file or directory",
        "command not found",
        "segmentation fault",
        "exited with exit code",
        "killed",
    ];
    keys.iter().any(|k| s.contains(k))
}

fn is_benign_stderr_line(line: &str) -> bool {
    let s = line.trim().to_ascii_lowercase();
    if s.is_empty() {
        return true;
    }
    s.contains("no version information available (required by")
}

pub(crate) fn safe_job_label(job: &str) -> String {
    let s = job.to_string();
    let out: String = s
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() {
        "job".to_string()
    } else {
        out
    }
}
