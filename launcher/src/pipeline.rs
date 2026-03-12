use std::collections::HashSet;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(unix)]
use std::sync::Once;
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
    pub step_index_offset: usize,
    pub display_total_steps: Option<usize>,
    pub emit_completion_line: bool,
    pub emit_progress_line: bool,
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
            step_index_offset: 0,
            display_total_steps: None,
            emit_completion_line: true,
            emit_progress_line: true,
        }
    }
}

pub(crate) trait PipelineHook {
    fn on_item_completed(&mut self, _step: &PipelineStep, _item: &StepItem) -> Result<(), String> {
        Ok(())
    }
}

static INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);
#[cfg(unix)]
static INSTALL_SIGNAL_HANDLER_ONCE: Once = Once::new();

#[cfg(unix)]
unsafe extern "C" {
    fn signal(signum: i32, handler: usize) -> usize;
}

#[cfg(unix)]
extern "C" fn janusx_signal_handler(_sig: i32) {
    INTERRUPT_REQUESTED.store(true, Ordering::SeqCst);
}

fn setup_interrupt_trap() {
    #[cfg(unix)]
    {
        const SIGINT: i32 = 2;
        const SIGTERM: i32 = 15;
        INSTALL_SIGNAL_HANDLER_ONCE.call_once(|| unsafe {
            let _ = signal(SIGINT, janusx_signal_handler as usize);
            let _ = signal(SIGTERM, janusx_signal_handler as usize);
        });
    }
    INTERRUPT_REQUESTED.store(false, Ordering::SeqCst);
}

fn interrupt_requested() -> bool {
    INTERRUPT_REQUESTED.load(Ordering::SeqCst)
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
    setup_interrupt_trap();
    fs::create_dir_all(workdir.join("tmp"))
        .map_err(|e| format!("Failed to create {}: {e}", workdir.join("tmp").display()))?;
    fs::create_dir_all(workdir.join("log"))
        .map_err(|e| format!("Failed to create {}: {e}", workdir.join("log").display()))?;

    for (idx, step) in steps.iter().enumerate() {
        let step_no = idx + opts.step_index_offset;
        let total_steps = opts.display_total_steps.unwrap_or(steps.len());
        let step_text = format_step_label(step_no, total_steps, step);
        if interrupt_requested() {
            return Err("Interrupted by signal.".to_string());
        }
        if opts.skip_if_outputs_exist && all_outputs_ready(&step.outputs) {
            clear_ready_item_submission_markers(workdir, step);
            if opts.emit_completion_line {
                super::print_success_line(&format!("{step_text} ...Skipped"));
            }
            continue;
        }
        if opts.skip_if_outputs_exist && !step.items.is_empty() {
            let pending_idx = pending_item_indices(step);
            if pending_idx.is_empty() {
                clear_ready_item_submission_markers(workdir, step);
                if opts.emit_completion_line {
                    super::print_success_line(&format!("{step_text} ...Skipped"));
                }
                continue;
            }
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
        let existing_error_logs = if opts.detect_failed_logs && !step.items.is_empty() {
            collect_existing_item_error_logs(workdir, step)?
        } else {
            HashSet::new()
        };

        let mut commands_to_submit: Vec<String> = step.commands.clone();
        let mut skip_submit = false;
        if matches!(opts.scheduler, Scheduler::Csub)
            && opts.detect_failed_logs
            && !step.items.is_empty()
            && step.commands.len() == step.items.len()
        {
            let pending_idx = pending_item_indices(step);
            if !pending_idx.is_empty() {
                if let Ok(running_idx) =
                    find_running_csub_items_for_pending(workdir, step, &pending_idx)
                {
                    if !running_idx.is_empty() {
                        let submit_idx: Vec<usize> = pending_idx
                            .iter()
                            .copied()
                            .filter(|i| !running_idx.contains(i))
                            .collect();
                        if submit_idx.is_empty() {
                            skip_submit = true;
                            println!(
                                "{}",
                                super::style_yellow(&format!(
                                    "{step_text} detected existing running csub jobs; skip re-submit and continue waiting."
                                ))
                            );
                        } else if submit_idx.len() < pending_idx.len() {
                            println!(
                                "{}",
                                super::style_yellow(&format!(
                                    "{step_text} detected existing running csub jobs; only submit missing subtasks ({}/{}).",
                                    submit_idx.len(),
                                    pending_idx.len()
                                ))
                            );
                            commands_to_submit = submit_idx
                                .iter()
                                .map(|i| step.commands[*i].clone())
                                .collect();
                        }
                    }
                }
            }
        }

        let step_exec_start = Instant::now();
        let mut submitted_this_round = false;
        let mut script_child: Option<Child> = None;
        if !skip_submit {
            submitted_this_round = true;
            let script = build_step_script_from_commands(&commands_to_submit, opts);
            fs::write(&sh, script).map_err(|e| format!("Failed to write {}: {e}", sh.display()))?;
            set_script_executable(&sh)?;
            if matches!(opts.scheduler, Scheduler::Nohup) {
                script_child = Some(start_step_script_async(workdir, &sh)?);
            } else if let Err(e) = run_step_script(workdir, &sh, opts) {
                if interrupt_requested() {
                    return handle_interrupt_for_step(
                        workdir,
                        &step_text,
                        step,
                        opts,
                        &pending_item_indices(step),
                    );
                }
                if opts.detect_failed_logs && !step.items.is_empty() {
                    let pending_now = pending_item_indices(step);
                    if let Ok(failed) = find_failed_item_logs_for_pending(
                        workdir,
                        step,
                        &pending_now,
                        3,
                        4,
                        &existing_error_logs,
                    ) {
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
                return Err(e);
            }
        }
        if interrupt_requested() {
            terminate_step_script(&mut script_child);
            return handle_interrupt_for_step(
                workdir,
                &step_text,
                step,
                opts,
                &pending_item_indices(step),
            );
        }
        let pre_wait_elapsed = step_exec_start.elapsed();
        wait_step_outputs(
            workdir,
            &step_text,
            step,
            opts,
            hook,
            &existing_error_logs,
            submitted_this_round,
            pre_wait_elapsed,
            script_child,
        )?;
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

fn build_step_script_from_commands(commands: &[String], opts: &PipelineOptions) -> String {
    if commands.is_empty() {
        return "echo 'empty step'\n".to_string();
    }
    match opts.scheduler {
        Scheduler::Csub => {
            let mut text = commands.join("\n");
            if !text.ends_with('\n') {
                text.push('\n');
            }
            text
        }
        Scheduler::Nohup => {
            let mut preamble: Vec<String> = vec![
                "JANUSX_CHILD_PIDS=\"\"".to_string(),
                "janusx_cleanup_children() {".to_string(),
                "  for pid in $JANUSX_CHILD_PIDS; do".to_string(),
                "    if kill -0 \"$pid\" 2>/dev/null; then".to_string(),
                "      kill \"$pid\" 2>/dev/null || true".to_string(),
                "    fi".to_string(),
                "  done".to_string(),
                "  wait 2>/dev/null || true".to_string(),
                "}".to_string(),
                "trap 'janusx_cleanup_children; exit 130' INT TERM".to_string(),
            ];
            if opts.nohup_max_jobs > 0 {
                let mut parts = vec![format!("MAX_JOBS={}", opts.nohup_max_jobs)];
                parts.append(&mut preamble);
                for line in commands {
                    parts.push(
                        "while [ \"$(jobs -pr | wc -l)\" -ge \"$MAX_JOBS\" ]; do sleep 2; done"
                            .to_string(),
                    );
                    parts.push(format!("{line} &"));
                    parts.push("JANUSX_CHILD_PIDS=\"$JANUSX_CHILD_PIDS $!\"".to_string());
                }
                parts.push("wait".to_string());
                parts.push("rc=$?".to_string());
                parts.push("trap - INT TERM".to_string());
                parts.push("exit $rc".to_string());
                parts.join("\n") + "\n"
            } else {
                let mut lines: Vec<String> = Vec::new();
                lines.append(&mut preamble);
                for line in commands {
                    lines.push(format!("{line} &"));
                    lines.push("JANUSX_CHILD_PIDS=\"$JANUSX_CHILD_PIDS $!\"".to_string());
                }
                lines.push("wait".to_string());
                lines.push("rc=$?".to_string());
                lines.push("trap - INT TERM".to_string());
                lines.push("exit $rc".to_string());
                lines.join("\n") + "\n"
            }
        }
    }
}

fn pending_item_indices(step: &PipelineStep) -> Vec<usize> {
    let mut pending: Vec<usize> = Vec::new();
    for (idx, item) in step.items.iter().enumerate() {
        if !all_outputs_ready(&item.outputs) {
            pending.push(idx);
        }
    }
    pending
}

fn submission_marker_path(workdir: &Path, safe_job: &str) -> PathBuf {
    workdir.join("log").join(format!("{safe_job}.submitted"))
}

fn item_submission_marker_path(workdir: &Path, item_id: &str) -> PathBuf {
    submission_marker_path(workdir, &safe_job_label(item_id))
}

fn read_submission_marker_job_id(marker: &Path) -> Option<String> {
    let raw = fs::read_to_string(marker).ok()?;
    let job_id = raw.trim();
    if job_id.is_empty() || !job_id.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(job_id.to_string())
}

fn clear_item_submission_marker(workdir: &Path, item_id: &str) {
    let marker = item_submission_marker_path(workdir, item_id);
    if marker.exists() {
        let _ = fs::remove_file(marker);
    }
}

fn clear_ready_item_submission_markers(workdir: &Path, step: &PipelineStep) {
    for item in &step.items {
        if all_outputs_ready(&item.outputs) {
            clear_item_submission_marker(workdir, &item.id);
        }
    }
}

pub(crate) fn infer_first_incomplete_step(steps: &[PipelineStep]) -> usize {
    for (idx, step) in steps.iter().enumerate() {
        if !is_step_outputs_ready(step) {
            return idx;
        }
    }
    steps.len()
}

fn is_step_outputs_ready(step: &PipelineStep) -> bool {
    if all_outputs_ready(&step.outputs) {
        return true;
    }
    if !step.items.is_empty() {
        return step
            .items
            .iter()
            .all(|item| all_outputs_ready(&item.outputs));
    }
    false
}

#[derive(Clone, Debug)]
struct CsubJobInfo {
    job_id: String,
}

fn find_running_csub_items_for_pending(
    workdir: &Path,
    step: &PipelineStep,
    pending_idx: &[usize],
) -> Result<HashSet<usize>, String> {
    let running_jobs = query_cjobs_active_jobs()?;
    if running_jobs.is_empty() {
        return Ok(HashSet::new());
    }

    let mut out: HashSet<usize> = HashSet::new();
    for idx in pending_idx {
        let Some(item) = step.items.get(*idx) else {
            continue;
        };
        let marker = item_submission_marker_path(workdir, &item.id);
        let Some(job_id) = read_submission_marker_job_id(&marker) else {
            if marker.exists() {
                let _ = fs::remove_file(marker);
            }
            continue;
        };
        if running_jobs.iter().any(|job| job.job_id == job_id) {
            out.insert(*idx);
        } else {
            let _ = fs::remove_file(marker);
        }
    }
    Ok(out)
}

fn query_cjobs_active_jobs() -> Result<Vec<CsubJobInfo>, String> {
    let out = Command::new("cjobs")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run cjobs: {e}"))?;
    if !out.status.success() {
        let mut msg = String::new();
        msg.push_str(&String::from_utf8_lossy(&out.stdout));
        msg.push_str(&String::from_utf8_lossy(&out.stderr));
        return Err(format!(
            "cjobs exited with code {}: {}",
            super::exit_code(out.status),
            msg.trim()
        ));
    }

    let mut jobs: Vec<CsubJobInfo> = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let raw = line.trim();
        if raw.is_empty() || raw.starts_with("JOBID") {
            continue;
        }
        let cols: Vec<&str> = raw.split_whitespace().collect();
        if cols.len() < 7 {
            continue;
        }
        let stat = cols[2].to_ascii_uppercase();
        let active = matches!(stat.as_str(), "RUN" | "PEND" | "PSUSP" | "USUSP" | "SSUSP");
        if !active {
            continue;
        }
        let job_id = cols[0].trim().to_string();
        let job_name = cols[6].trim();
        if job_id.is_empty() || job_name.is_empty() {
            continue;
        }
        let key = (job_id.clone(), job_name.to_string());
        if seen.insert(key) {
            jobs.push(CsubJobInfo { job_id });
        }
    }
    Ok(jobs)
}

fn collect_csub_job_ids_for_pending(
    workdir: &Path,
    step: &PipelineStep,
    pending_idx: &[usize],
) -> Result<Vec<String>, String> {
    if pending_idx.is_empty() {
        return Ok(Vec::new());
    }
    let running_jobs = query_cjobs_active_jobs()?;
    if running_jobs.is_empty() {
        return Ok(Vec::new());
    }
    let mut ids: HashSet<String> = HashSet::new();
    for idx in pending_idx {
        let Some(item) = step.items.get(*idx) else {
            continue;
        };
        let marker = item_submission_marker_path(workdir, &item.id);
        let Some(job_id) = read_submission_marker_job_id(&marker) else {
            if marker.exists() {
                let _ = fs::remove_file(marker);
            }
            continue;
        };
        if running_jobs.iter().any(|job| job.job_id == job_id) {
            ids.insert(job_id);
        } else if marker.exists() {
            let _ = fs::remove_file(marker);
        }
    }
    let mut out: Vec<String> = ids.into_iter().collect();
    out.sort();
    Ok(out)
}

fn ckill_job_ids(job_ids: &[String]) -> Result<(), String> {
    if job_ids.is_empty() {
        return Ok(());
    }
    let mut errors: Vec<String> = Vec::new();
    for job_id in job_ids {
        let out = Command::new("ckill")
            .arg(job_id)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();
        match out {
            Ok(v) => {
                if !v.status.success() {
                    let mut msg = String::new();
                    msg.push_str(&String::from_utf8_lossy(&v.stdout));
                    msg.push_str(&String::from_utf8_lossy(&v.stderr));
                    errors.push(format!(
                        "JOBID {}: exit={} {}",
                        job_id,
                        super::exit_code(v.status),
                        msg.trim()
                    ));
                }
            }
            Err(e) => {
                errors.push(format!("JOBID {}: {}", job_id, e));
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join(" | "))
    }
}

fn handle_interrupt_for_step(
    workdir: &Path,
    step_text: &str,
    step: &PipelineStep,
    opts: &PipelineOptions,
    pending_idx: &[usize],
) -> Result<(), String> {
    if !matches!(opts.scheduler, Scheduler::Csub) {
        return Err("Interrupted by signal.".to_string());
    }
    let job_ids = collect_csub_job_ids_for_pending(workdir, step, pending_idx)?;
    if job_ids.is_empty() {
        return Err("Interrupted by signal.".to_string());
    }
    match ckill_job_ids(&job_ids) {
        Ok(()) => Err(format!(
            "{step_text} interrupted. Cancelled csub jobs: {}",
            job_ids.join(", ")
        )),
        Err(e) => Err(format!(
            "{step_text} interrupted. Failed to cancel some csub jobs: {e}"
        )),
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

fn start_step_script_async(workdir: &Path, sh: &Path) -> Result<Child, String> {
    Command::new("bash")
        .arg(sh)
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to start {}: {e}", sh.display()))
}

fn poll_script_exit(script_child: &mut Option<Child>) -> Result<Option<ExitStatus>, String> {
    let status = match script_child.as_mut() {
        Some(child) => child
            .try_wait()
            .map_err(|e| format!("Failed to poll step script status: {e}"))?,
        None => return Ok(None),
    };
    if let Some(st) = status {
        *script_child = None;
        return Ok(Some(st));
    }
    Ok(None)
}

fn terminate_step_script(script_child: &mut Option<Child>) {
    let Some(mut child) = script_child.take() else {
        return;
    };
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(child.id().to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        thread::sleep(Duration::from_millis(250));
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn wait_step_outputs(
    workdir: &Path,
    step_text: &str,
    step: &PipelineStep,
    opts: &PipelineOptions,
    hook: &mut impl PipelineHook,
    ignore_error_logs: &HashSet<PathBuf>,
    submitted_this_round: bool,
    pre_wait_elapsed: Duration,
    script_child: Option<Child>,
) -> Result<(), String> {
    let mut script_child = script_child;
    let stall_timeout = resolve_no_progress_timeout(step, opts);

    if step.items.is_empty() {
        let total = 1usize;
        let start = Instant::now();
        let mut last_progress = Instant::now();
        let mut last_done = 0usize;

        loop {
            if interrupt_requested() {
                clear_progress_line_if_tty()?;
                terminate_step_script(&mut script_child);
                return handle_interrupt_for_step(workdir, step_text, step, opts, &[]);
            }
            let done = if all_outputs_ready(&step.outputs) {
                total
            } else {
                0
            };
            if let Some(status) = poll_script_exit(&mut script_child)? {
                if !status.success() {
                    clear_progress_line_if_tty()?;
                    return Err(format!(
                        "{step_text} failed with exit={}.",
                        super::exit_code(status)
                    ));
                }
                if done < total {
                    clear_progress_line_if_tty()?;
                    let miss = step
                        .outputs
                        .iter()
                        .filter(|p| !is_output_ready(p))
                        .take(5)
                        .map(|p| p.display().to_string())
                        .collect::<Vec<String>>()
                        .join(", ");
                    return Err(format!(
                        "{step_text} finished but outputs are incomplete: {miss}"
                    ));
                }
            }
            if done >= total {
                if script_child.is_some() {
                    if opts.emit_progress_line {
                        render_step_progress(
                            step_text,
                            done,
                            total,
                            pre_wait_elapsed + start.elapsed(),
                            0,
                        )?;
                    }
                    thread::sleep(super::spinner_refresh_interval(
                        pre_wait_elapsed + start.elapsed(),
                    ));
                    continue;
                }
                let elapsed = pre_wait_elapsed + start.elapsed();
                if opts.emit_completion_line {
                    if opts.skip_if_outputs_exist && !submitted_this_round {
                        super::print_success_line(&format!("{step_text} ...Skipped"));
                    } else {
                        super::print_success_line(&format!(
                            "{} ...Finished [{}]",
                            step_text,
                            super::format_elapsed(elapsed)
                        ));
                    }
                } else if opts.emit_progress_line {
                    clear_progress_line_if_tty()?;
                }
                return Ok(());
            }
            if done > last_done {
                last_done = done;
                last_progress = Instant::now();
            }

            if let Some(timeout) = stall_timeout {
                if done < total && last_progress.elapsed() >= timeout {
                    clear_progress_line_if_tty()?;
                    return Err(format!(
                        "{step_text} has no progress for {} (done {done}/{total}).",
                        super::format_elapsed(timeout)
                    ));
                }
            }

            if opts.emit_progress_line {
                render_step_progress(
                    step_text,
                    done,
                    total,
                    pre_wait_elapsed + start.elapsed(),
                    0,
                )?;
            }
            thread::sleep(super::spinner_refresh_interval(
                pre_wait_elapsed + start.elapsed(),
            ));
        }
    }

    let total = step.items.len();
    let start = Instant::now();
    let mut last_error_scan = Instant::now();
    let mut pending: Vec<usize> = Vec::new();
    let mut done = 0usize;
    let mut last_progress = Instant::now();

    for (idx, item) in step.items.iter().enumerate() {
        if all_outputs_ready(&item.outputs) {
            done += 1;
        } else {
            pending.push(idx);
        }
    }
    if done >= total && script_child.is_none() {
        clear_ready_item_submission_markers(workdir, step);
        if opts.emit_completion_line {
            if opts.skip_if_outputs_exist && !submitted_this_round {
                super::print_success_line(&format!("{step_text} ...Skipped"));
            } else {
                super::print_success_line(&format!(
                    "{} ...Finished [{}]",
                    step_text,
                    super::format_elapsed(pre_wait_elapsed)
                ));
            }
        } else if opts.emit_progress_line {
            clear_progress_line_if_tty()?;
        }
        return Ok(());
    }

    loop {
        if interrupt_requested() {
            clear_progress_line_if_tty()?;
            terminate_step_script(&mut script_child);
            return handle_interrupt_for_step(workdir, step_text, step, opts, &pending);
        }
        if !pending.is_empty() {
            let mut next_pending: Vec<usize> = Vec::with_capacity(pending.len());
            for idx in &pending {
                let item = &step.items[*idx];
                if all_outputs_ready(&item.outputs) {
                    done += 1;
                    last_progress = Instant::now();
                    clear_item_submission_marker(workdir, &item.id);
                    hook.on_item_completed(step, item)?;
                } else {
                    next_pending.push(*idx);
                }
            }
            pending = next_pending;
        }

        if let Some(status) = poll_script_exit(&mut script_child)? {
            if !status.success() {
                clear_progress_line_if_tty()?;
                return Err(format!(
                    "{step_text} failed with exit={}.",
                    super::exit_code(status)
                ));
            }
            if done < total {
                clear_progress_line_if_tty()?;
                if opts.detect_failed_logs {
                    let failed = find_failed_item_logs_for_pending(
                        workdir,
                        step,
                        &pending,
                        3,
                        4,
                        ignore_error_logs,
                    )?;
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
                let miss = pending
                    .iter()
                    .take(5)
                    .filter_map(|idx| step.items.get(*idx))
                    .map(|x| x.id.clone())
                    .collect::<Vec<String>>()
                    .join(", ");
                return Err(format!(
                    "{step_text} finished but outputs are incomplete for: {miss}"
                ));
            }
        }

        if done >= total && script_child.is_none() {
            break;
        }

        if opts.detect_failed_logs {
            let scan_interval = Duration::from_secs_f64((opts.poll_sec * 3.0).max(2.0));
            if last_error_scan.elapsed() >= scan_interval {
                last_error_scan = Instant::now();
                let failed = find_failed_item_logs_for_pending(
                    workdir,
                    step,
                    &pending,
                    3,
                    4,
                    ignore_error_logs,
                )?;
                if !failed.is_empty() {
                    clear_progress_line_if_tty()?;
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
                clear_progress_line_if_tty()?;
                return Err(format!(
                    "{step_text} has no progress for {} (done {done}/{total}).",
                    super::format_elapsed(timeout)
                ));
            }
        }

        if opts.emit_progress_line {
            render_step_progress(
                step_text,
                done,
                total,
                pre_wait_elapsed + start.elapsed(),
                0,
            )?;
        }
        thread::sleep(super::spinner_refresh_interval(
            pre_wait_elapsed + start.elapsed(),
        ));
    }
    if opts.emit_progress_line {
        clear_progress_line_if_tty()?;
    }
    clear_ready_item_submission_markers(workdir, step);
    if opts.emit_completion_line {
        super::print_success_line(&format!(
            "{} ...Finished [{}]",
            step_text,
            super::format_elapsed(pre_wait_elapsed + start.elapsed())
        ));
    }
    Ok(())
}

fn render_step_progress(
    step_text: &str,
    done: usize,
    total: usize,
    elapsed: Duration,
    _spinner_tick: usize,
) -> Result<(), String> {
    if !io::stdout().is_terminal() {
        return Ok(());
    }
    let line = format!(
        "\r{} {} [{}/{}] [{}]",
        super::spinner_frame_for_elapsed(elapsed),
        step_text,
        done,
        total,
        super::format_elapsed_live(elapsed)
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
    if is_csub_failure_marker_line(&s) {
        return false;
    }
    let keys = [
        "traceback (most recent call last)",
        "runtimeerror",
        "exception",
        "fatal",
        "error",
        "failed",
        "daemon is not running",
        "no such file or directory",
        "command not found",
        "permission denied",
        "segmentation fault",
        "exited with exit code",
        "killed",
        "terminated",
        "cancelled",
        "canceled",
        "sigterm",
        "sigkill",
        "signal",
        "out of memory",
        "oom",
    ];
    keys.iter().any(|k| s.contains(k))
}

fn is_csub_failure_marker_line(s: &str) -> bool {
    // csub frequently emits this line when a task stderr is produced.
    // For pending items this is a strong failure signal even when detailed
    // stderr lines are missing or delayed.
    s.starts_with("job ") && s.contains(" stderr output")
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
