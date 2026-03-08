use super::pipeline::{
    all_outputs_ready, run_pipeline_with_hook, PipelineOptions, PipelineStep, Scheduler, StepItem,
};
mod cmd;
mod state;
use cmd::{
    cmd_bam2gvcf, cmd_beagle_impute, cmd_bwamem, cmd_cgvcf, cmd_concat_imputed_vcfs, cmd_fastp,
    cmd_filter_imputed_by_maf_and_missing, cmd_gvcf2snp_table, cmd_markdup,
    cmd_snpvcf_to_gt_and_missing, sh_quote,
    wrap_scheduler_cmd,
};
use state::{
    inspect_work_state_basic_params, inspect_work_state_params, WorkStateParamStatus,
    WorkStateTracker,
};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, TryRecvError};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

const REQUIRED_DLC_TOOLS: [&str; 9] = [
    "fastp",
    "samtools",
    "sambamba",
    "gatk",
    "bcftools",
    "tabix",
    "bgzip",
    "plink",
    "beagle",
];
type ToolProbe = (String, bool, bool, String);
const FASTQ2VCF_TOTAL_STEPS: usize = 10;
const FASTQ_SUFFIXES: [&str; 4] = [".fastq.gz", ".fq.gz", ".fastq", ".fq"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadKind {
    R1,
    R2,
}

#[derive(Clone, Debug)]
struct FastqPair {
    r1: Option<PathBuf>,
    r2: Option<PathBuf>,
}

impl FastqPair {
    fn new() -> Self {
        Self { r1: None, r2: None }
    }

    fn complete(self) -> Result<(PathBuf, PathBuf), String> {
        let Some(r1) = self.r1 else {
            return Err("missing R1".to_string());
        };
        let Some(r2) = self.r2 else {
            return Err("missing R2".to_string());
        };
        Ok((r1, r2))
    }
}

#[derive(Clone, Debug)]
struct ParsedArgs {
    reference: PathBuf,
    fastq_dir: PathBuf,
    workdir: PathBuf,
}

#[derive(Clone, Debug)]
struct FaiData {
    order: Vec<String>,
    lens: BTreeMap<String, u64>,
}

#[derive(Clone, Debug)]
struct ReferenceIndexTask {
    reference: PathBuf,
    workdir: PathBuf,
    safe_job: String,
    ignore_error_logs: HashSet<PathBuf>,
    is_csub: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReferenceWaitStatus {
    Ready,
    Pending,
}

pub(crate) fn run_fastq2vcf_module(args: &[String]) -> Result<i32, String> {
    if args.iter().any(|x| matches!(x.as_str(), "-h" | "--help")) {
        print_fastq2vcf_help();
        return Ok(0);
    }

    let parsed = parse_fastq2vcf_args(args)?;
    let (reference, fastq_dir, workdir) = validate_and_resolve_inputs(&parsed)?;

    fs::create_dir_all(workdir.join("log")).map_err(|e| {
        format!(
            "Failed to create log directory {}: {e}",
            workdir.join("log").display()
        )
    })?;
    fs::create_dir_all(workdir.join("tmp")).map_err(|e| {
        format!(
            "Failed to create tmp directory {}: {e}",
            workdir.join("tmp").display()
        )
    })?;
    let reference = prepare_reference_for_pipeline(&reference, &workdir)?;

    let (backend, backend_reason) = detect_backend();
    println!(
        "{}",
        super::style_green(&format!("Selected backend: {backend}"))
    );
    if !backend_reason.is_empty() {
        println!("{}", backend_reason);
    }

    let jx_cmd = "jx";
    let (toolchain, aligner_cmd) = probe_dlc_toolchain()?;
    print_toolchain_line(&toolchain);
    let missing_wrappers = missing_tools_from_probe(&toolchain);
    if !missing_wrappers.is_empty() {
        return Err(format!(
            "Missing DLC tool wrappers: {}. Please run `jx -update dlc`.",
            missing_wrappers
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        ));
    }

    let fastq_files = collect_fastq_files(&fastq_dir)?;
    if fastq_files.is_empty() {
        return Err(format!("No FASTQ files found in {}", fastq_dir.display()));
    }
    let samples = classify_fastq_pairs(&fastq_files)?;
    if samples.is_empty() {
        return Err(format!(
            "Detected 0 paired samples from {} FASTQ files.\n{}",
            fastq_files.len(),
            recognized_fastq_pairing_hint()
        ));
    }

    println!(
        "{}",
        super::style_green(&format!(
            "Detected {} paired samples from {} FASTQ files.",
            samples.len(),
            fastq_files.len()
        ))
    );

    let singularity = jx_cmd.to_string();
    let mut opts = PipelineOptions::default();
    opts.scheduler = if backend == "csub" {
        Scheduler::Csub
    } else {
        Scheduler::Nohup
    };
    opts.nohup_max_jobs = 1;
    opts.skip_if_outputs_exist = true;
    opts.poll_sec = 0.5;
    opts.detect_failed_logs = true;

    let state_path = workdir.join(".work.json");
    let mut param_mismatch_confirmed = false;
    let basic_param_status = inspect_work_state_basic_params(
        &state_path,
        &reference,
        &samples,
        &backend,
        opts.nohup_max_jobs,
        &singularity,
    )?;
    match basic_param_status {
        WorkStateParamStatus::Mismatch => {
            println!(
                "{}",
                super::style_yellow(&format!(
                    "Warning: detected existing {} with different run parameters.",
                    state_path.display()
                ))
            );
            let proceed = prompt_yes_no(
                "Continue with current parameters in the same workdir? [y/N]: ",
                false,
            )?;
            if !proceed {
                return Err("Cancelled by user due to parameter mismatch.".to_string());
            }
            param_mismatch_confirmed = true;
        }
        WorkStateParamStatus::Match | WorkStateParamStatus::NotFound => {}
    }
    let mut resumed_line_printed = false;
    if matches!(basic_param_status, WorkStateParamStatus::Match) && state_path.exists() {
        println!("Resuming pipeline from {}", state_path.display());
        resumed_line_printed = true;
    }

    let index_task = start_reference_indexing(
        &reference,
        &workdir,
        &backend,
        jx_cmd,
        "indexREF",
        &aligner_cmd,
    )?;

    let fastp_step = build_fastp_step(&samples, &workdir, &backend, &singularity)?;
    let fastp_skipped_by_outputs = opts.skip_if_outputs_exist && all_outputs_ready(&fastp_step.outputs);
    if let Some(task) = index_task.clone() {
        if fastp_skipped_by_outputs {
            wait_reference_indexes(&task)?;
            super::print_success_line("Step 1/10: fastp (~15m) ...Skipped");
        } else {
            let mut fastp_opts = opts.clone();
            fastp_opts.step_index_offset = 0;
            fastp_opts.display_total_steps = Some(FASTQ2VCF_TOTAL_STEPS);
            fastp_opts.emit_completion_line = false;
            fastp_opts.emit_progress_line = false;
            let (fastp_elapsed, index_elapsed) = run_fastp_and_index_parallel(
                &workdir,
                &fastp_step,
                &fastp_opts,
                &task,
            )?;
            super::print_success_line(&format!(
                "Reference index check ...Finished [{}]",
                super::format_elapsed_live(index_elapsed)
            ));
            super::print_success_line(&format!(
                "Step 1/10: fastp (~15m) ...Finished [{}]",
                super::format_elapsed_live(fastp_elapsed)
            ));
        }
    } else {
        let fastp_start = Instant::now();
        let mut fastp_opts = opts.clone();
        fastp_opts.step_index_offset = 0;
        fastp_opts.display_total_steps = Some(FASTQ2VCF_TOTAL_STEPS);
        fastp_opts.emit_completion_line = false;
        let mut noop_hook = |_step: &PipelineStep, _item: &StepItem| Ok(());
        run_pipeline_with_hook(
            &workdir,
            std::slice::from_ref(&fastp_step),
            &fastp_opts,
            &mut noop_hook,
        )?;
        let fastp_elapsed = fastp_start.elapsed();
        if fastp_skipped_by_outputs {
            super::print_success_line("Step 1/10: fastp (~15m) ...Skipped");
        } else {
            super::print_success_line(&format!(
                "Step 1/10: fastp (~15m) ...Finished [{}]",
                super::format_elapsed_live(fastp_elapsed)
            ));
        }
    }

    let fai_data = read_fai_data_from_reference(&reference)?;
    let chroms = fai_data.order.clone();

    let legacy_metadata_path = workdir.join(".fastq2vcf.metadata.json");
    if legacy_metadata_path.exists() {
        let _ = fs::remove_file(&legacy_metadata_path);
    }
    let steps = build_fastq2vcf_steps(
        &reference,
        &samples,
        &chroms,
        &fai_data.lens,
        &workdir,
        &backend,
        &aligner_cmd,
        &singularity,
        true,
    )?;

    let run_params_json = build_run_params_json(
        &reference,
        &samples,
        &chroms,
        &backend,
        opts.nohup_max_jobs,
        &singularity,
    );
    if !param_mismatch_confirmed {
        match inspect_work_state_params(&state_path, &run_params_json)? {
            WorkStateParamStatus::Mismatch => {
                println!(
                    "{}",
                    super::style_yellow(&format!(
                        "Warning: detected existing {} with different run parameters.",
                        state_path.display()
                    ))
                );
                let proceed = prompt_yes_no(
                    "Continue with current parameters in the same workdir? [y/N]: ",
                    false,
                )?;
                if !proceed {
                    return Err("Cancelled by user due to parameter mismatch.".to_string());
                }
            }
            WorkStateParamStatus::Match | WorkStateParamStatus::NotFound => {}
        }
    }
    let input_files = build_input_files(&reference, &samples);
    let (mut state_tracker, resumed) =
        WorkStateTracker::init_or_resume(&state_path, &run_params_json, &input_files, &steps)?;
    if resumed && !resumed_line_printed {
        println!("Resuming pipeline from {}", state_path.display());
    }
    let summary = state_tracker.summary();
    if summary.total_items > 0 && summary.done_items >= summary.total_items {
        println!("All FASTQ2VCF tasks already completed (same parameters).");
        println!(
            "Re-checking all {} steps with rich progress...",
            steps.len()
        );
    }
    state_tracker.mark_running()?;

    let home = super::runtime_home()?;
    let _python = super::ensure_runtime(false)?;
    let _ = super::maybe_auto_warmup(&home)?;
    let pipeline_result = {
        let mut item_hook =
            |step: &PipelineStep, item: &StepItem| state_tracker.mark_item_done(step, item);
        let mut main_opts = opts.clone();
        main_opts.step_index_offset = 1;
        main_opts.display_total_steps = Some(FASTQ2VCF_TOTAL_STEPS);
        if steps.len() <= 1 {
            Ok(())
        } else {
            run_pipeline_with_hook(&workdir, &steps[1..], &main_opts, &mut item_hook)
        }
    };
    match pipeline_result {
        Ok(()) => {
            state_tracker.mark_completed()?;
        }
        Err(err) => {
            if let Err(mark_err) = state_tracker.mark_failed(&err) {
                return Err(format!(
                    "{err}\nAdditionally failed to update .work.json: {mark_err}"
                ));
            }
            return Err(err);
        }
    }

    let bulk_names = infer_bulks(samples.keys().cloned().collect::<Vec<String>>());
    let latest_tsv = latest_snp_table(&workdir);
    let tsv_text = match latest_tsv {
        Some(p) => p.to_string_lossy().to_string(),
        None => {
            if let Some(last) = chroms.last() {
                workdir
                    .join("4.merge")
                    .join(format!("Merge.{last}.SNP.tsv"))
                    .to_string_lossy()
                    .to_string()
            } else {
                workdir
                    .join("4.merge")
                    .join("Merge.<chrom>.SNP.tsv")
                    .to_string_lossy()
                    .to_string()
            }
        }
    };
    let postbsa_glob = workdir
        .join("4.merge")
        .join("Merge.*.SNP.tsv")
        .to_string_lossy()
        .to_string();
    println!();
    println!(
        "{}",
        super::style_green(&format!(
            "Detected bulks: {}",
            if bulk_names.is_empty() {
                "None".to_string()
            } else {
                bulk_names.join(", ")
            }
        ))
    );
    println!(
        "{}",
        super::style_green(&format!("Latest SNP table: {tsv_text}"))
    );
    println!("Next step suggestion:");
    println!("  jx postbsa -file {postbsa_glob} -b1 [bulk1] -b2 [bulk2]");

    Ok(0)
}

fn print_fastq2vcf_help() {
    let width = super::help_line_width();
    print!("{} ", super::style_orange("Usage:"));
    print!("{}", style_gray("jx fastq2vcf"));
    print!(" [");
    print!("{}", super::style_blue("-h"));
    print!("] ");
    print!("{}", super::style_blue("-r"));
    print!(" ");
    print!("{}", super::style_green("REFERENCE"));
    print!(" ");
    print!("{}", super::style_blue("-i"));
    print!(" ");
    print!("{}", super::style_green("FASTQ_DIR"));
    print!(" ");
    print!("{}", super::style_blue("-w"));
    print!(" ");
    println!("{}", super::style_green("WORKDIR"));
    println!();
    println!("{}", super::style_orange("Required Arguments:"));
    print_colored_help_entry(
        2,
        Some(("-r", "--reference", "REFERENCE")),
        "Reference genome FASTA file.",
        42,
        width,
    );
    print_colored_help_entry(
        2,
        Some(("-i", "--fastq-dir", "FASTQ_DIR")),
        "Directory containing FASTQ files.",
        42,
        width,
    );
    print_colored_help_entry(
        2,
        Some(("-w", "--workdir", "WORKDIR")),
        "Working directory for outputs.",
        42,
        width,
    );
    println!();
    println!("{}", super::style_orange("Optional Arguments:"));
    print_colored_help_entry(2, None, "Show this help message and exit.", 42, width);
    println!();
    println!("{}", super::style_orange("Examples:"));
    println!("  jx fastq2vcf -r ref.fa -i fastq_dir -w workdir");
    println!("  jx -update dlc");
    println!();
    println!("{}", super::style_orange("Citation:"));
    println!("  https://github.com/FJingxian/JanusX/");
}

fn style_gray(text: &str) -> String {
    if super::supports_color() {
        format!("\x1b[90m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

fn print_colored_help_entry(
    indent: usize,
    required: Option<(&str, &str, &str)>,
    desc: &str,
    key_width: usize,
    total_width: usize,
) {
    let lead = " ".repeat(indent);
    let min_desc_width = 20usize;

    let (key_plain, key_colored) = match required {
        Some((short, long, meta)) => (
            format!("{short} {meta}, {long} {meta}"),
            format!(
                "{} {}, {} {}",
                super::style_blue(short),
                super::style_green(meta),
                super::style_blue(long),
                super::style_green(meta),
            ),
        ),
        None => (
            "-h, --help".to_string(),
            format!(
                "{}, {}",
                super::style_blue("-h"),
                super::style_blue("--help")
            ),
        ),
    };

    let min_total = indent + key_width + 2 + min_desc_width;
    if total_width <= min_total {
        println!("{lead}{key_colored}");
        for line in super::wrap_help_text(
            desc,
            total_width.saturating_sub(indent + 2).max(min_desc_width),
        ) {
            println!("{lead}  {}", super::style_white(&line));
        }
        return;
    }

    let desc_width = total_width - indent - key_width - 2;
    let wrapped = super::wrap_help_text(desc, desc_width);
    if wrapped.is_empty() {
        println!("{lead}{key_colored}");
        return;
    }

    let pad_len = key_width.saturating_sub(key_plain.chars().count());
    let key_with_pad = format!("{key_colored}{}", " ".repeat(pad_len));
    println!("{lead}{key_with_pad}  {}", super::style_white(&wrapped[0]));
    let pad = " ".repeat(key_width);
    for line in wrapped.iter().skip(1) {
        println!("{lead}{pad}  {}", super::style_white(line));
    }
}

fn parse_fastq2vcf_args(args: &[String]) -> Result<ParsedArgs, String> {
    let mut reference: Option<String> = None;
    let mut fastq_dir: Option<String> = None;
    let mut workdir: Option<String> = None;
    let mut i = 0usize;

    while i < args.len() {
        let token = args[i].as_str();
        if token == "-r" || token == "--reference" {
            i += 1;
            if i >= args.len() {
                return Err("Option `-r/--reference` requires a value.".to_string());
            }
            reference = Some(args[i].clone());
            i += 1;
            continue;
        }
        if token == "-i" || token == "--fastq-dir" {
            i += 1;
            if i >= args.len() {
                return Err("Option `-i/--fastq-dir` requires a value.".to_string());
            }
            fastq_dir = Some(args[i].clone());
            i += 1;
            continue;
        }
        if token == "-w" || token == "--workdir" {
            i += 1;
            if i >= args.len() {
                return Err("Option `-w/--workdir` requires a value.".to_string());
            }
            workdir = Some(args[i].clone());
            i += 1;
            continue;
        }
        if let Some(v) = token.strip_prefix("--reference=") {
            reference = Some(v.to_string());
            i += 1;
            continue;
        }
        if let Some(v) = token.strip_prefix("--fastq-dir=") {
            fastq_dir = Some(v.to_string());
            i += 1;
            continue;
        }
        if let Some(v) = token.strip_prefix("--workdir=") {
            workdir = Some(v.to_string());
            i += 1;
            continue;
        }
        return Err(format!(
            "Unknown option: {token}\nUse `jx fastq2vcf -h` for help."
        ));
    }

    let (Some(reference), Some(fastq_dir), Some(workdir)) = (reference, fastq_dir, workdir) else {
        return Err(
            "`-r/--reference`, `-i/--fastq-dir`, and `-w/--workdir` are required.".to_string(),
        );
    };

    Ok(ParsedArgs {
        reference: PathBuf::from(reference),
        fastq_dir: PathBuf::from(fastq_dir),
        workdir: PathBuf::from(workdir),
    })
}

fn validate_and_resolve_inputs(args: &ParsedArgs) -> Result<(PathBuf, PathBuf, PathBuf), String> {
    let reference = absolutize_path(&args.reference)?;
    if !reference.exists() || !reference.is_file() {
        return Err(format!("Reference file not found: {}", reference.display()));
    }

    let fastq_dir = absolutize_path(&args.fastq_dir)?;
    if !fastq_dir.exists() || !fastq_dir.is_dir() {
        return Err(format!(
            "FASTQ directory not found: {}",
            fastq_dir.display()
        ));
    }
    let workdir = absolutize_path(&args.workdir)?;

    Ok((reference, fastq_dir, workdir))
}

fn prompt_yes_no(prompt: &str, default_yes: bool) -> Result<bool, String> {
    if !io::stdin().is_terminal() {
        return Ok(default_yes);
    }
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

fn prepare_reference_for_pipeline(reference: &Path, workdir: &Path) -> Result<PathBuf, String> {
    let reference = absolutize_path(reference)?;
    if !is_gzip_reference_path(&reference) {
        println!(
            "{}",
            super::style_green(&format!("Reference path: {}", reference.display()))
        );
        return Ok(reference);
    }
    let Some(gzip_bin) = find_in_path("gzip") else {
        return Err(
            "Reference is gzip-compressed but `gzip` command is not found in PATH.".to_string(),
        );
    };
    let ref_dir = workdir.join(".reference");
    fs::create_dir_all(&ref_dir)
        .map_err(|e| format!("Failed to create {}: {e}", ref_dir.display()))?;

    let out_name = decompressed_reference_name(&reference);
    let out_path = ref_dir.join(out_name);
    let needs_refresh = reference_decompress_needed(&reference, &out_path)?;
    if needs_refresh {
        let tmp_path = out_path.with_extension(format!(
            "{}.tmp",
            out_path
                .extension()
                .and_then(|x| x.to_str())
                .unwrap_or_default()
        ));
        let cmdline = format!(
            "{} -dc {} > {}",
            sh_quote(&gzip_bin.to_string_lossy()),
            sh_quote(&reference.to_string_lossy()),
            sh_quote(&tmp_path.to_string_lossy()),
        );
        run_shell_capture(workdir, &cmdline, "reference gunzip")?;
        if out_path.exists() {
            let _ = fs::remove_file(&out_path);
        }
        fs::rename(&tmp_path, &out_path).map_err(|e| {
            format!(
                "Failed to finalize decompressed reference {} -> {}: {e}",
                tmp_path.display(),
                out_path.display()
            )
        })?;
    }
    check_readable_nonempty(&out_path)?;
    println!(
        "{}",
        super::style_yellow(&format!(
            "Unzipped reference path: {}",
            out_path.display()
        ))
    );
    Ok(out_path)
}

fn is_gzip_reference_path(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|x| x.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    name.ends_with(".gz")
}

fn decompressed_reference_name(reference: &Path) -> String {
    let name = reference
        .file_name()
        .and_then(|x| x.to_str())
        .unwrap_or("reference.fa.gz");
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".gz") && name.len() > 3 {
        return name[..name.len() - 3].to_string();
    }
    "reference.fa".to_string()
}

fn reference_decompress_needed(src_gz: &Path, dst_plain: &Path) -> Result<bool, String> {
    if !dst_plain.exists() {
        return Ok(true);
    }
    let dst_md =
        fs::metadata(dst_plain).map_err(|e| format!("Failed to stat {}: {e}", dst_plain.display()))?;
    if !dst_md.is_file() || dst_md.len() == 0 {
        return Ok(true);
    }
    let src_md =
        fs::metadata(src_gz).map_err(|e| format!("Failed to stat {}: {e}", src_gz.display()))?;
    let src_m = src_md.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let dst_m = dst_md.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    Ok(dst_m < src_m)
}

fn detect_backend() -> (String, String) {
    let Some(csub_bin) = find_in_path("csub") else {
        return ("nohup".to_string(), String::new());
    };
    let mut probe = Command::new(&csub_bin);
    probe
        .arg("-h")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if probe.status().is_ok() {
        return (
            "csub".to_string(),
            format!("Backend probe: detected csub at {}", csub_bin.display()),
        );
    }
    (
        "nohup".to_string(),
        "Backend probe: csub probe failed. nohup runs single-task mode.".to_string(),
    )
}

fn probe_dlc_toolchain() -> Result<(Vec<ToolProbe>, String), String> {
    let jx_bin = env::current_exe().map_err(|e| format!("Failed to locate jx binary: {e}"))?;
    let mut out: Vec<ToolProbe> = Vec::new();

    for tool in REQUIRED_DLC_TOOLS {
        out.push((
            tool.to_string(),
            probe_dlc_tool_wrapper(&jx_bin, tool),
            true,
            format!("jx {tool}"),
        ));
    }

    let (aligner_display, aligner_ok, aligner_cmd, aligner_missing_hint) =
        probe_aligner_tool(&jx_bin);
    out.push((
        aligner_display,
        aligner_ok,
        true,
        format!("jx {aligner_missing_hint}"),
    ));

    if !aligner_ok {
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out.dedup_by(|a, b| a.0 == b.0);
        return Ok((out, aligner_cmd));
    }

    out.sort_by(|a, b| a.0.cmp(&b.0));
    out.dedup_by(|a, b| a.0 == b.0);
    Ok((out, aligner_cmd))
}

fn probe_aligner_tool(jx_bin: &Path) -> (String, bool, String, String) {
    let mem2_ok = probe_dlc_tool_wrapper(jx_bin, "bwa-mem2");
    let bwa_ok = probe_dlc_tool_wrapper(jx_bin, "bwa");
    if mem2_ok {
        let display = if !bwa_ok && bwa_mem2_is_bwa_alias(jx_bin) {
            "bwa".to_string()
        } else {
            "bwa-mem2".to_string()
        };
        return (
            display,
            true,
            "jx bwa-mem2".to_string(),
            "bwa-mem2 or bwa".to_string(),
        );
    }
    if bwa_ok {
        return (
            "bwa".to_string(),
            true,
            "jx bwa".to_string(),
            "bwa-mem2 or bwa".to_string(),
        );
    }
    if let Some(mem2_bin) = find_in_path("bwa-mem2") {
        return (
            "bwa-mem2".to_string(),
            true,
            sh_quote(&mem2_bin.to_string_lossy()),
            "bwa-mem2 or bwa".to_string(),
        );
    }
    if let Some(bwa_bin) = find_in_path("bwa") {
        return (
            "bwa".to_string(),
            true,
            sh_quote(&bwa_bin.to_string_lossy()),
            "bwa-mem2 or bwa".to_string(),
        );
    }
    (
        "bwa|bwa-mem2".to_string(),
        false,
        "jx bwa-mem2".to_string(),
        "bwa-mem2 or bwa".to_string(),
    )
}

fn bwa_mem2_is_bwa_alias(jx_bin: &Path) -> bool {
    let out_cmd = Command::new(jx_bin)
        .arg("bwa-mem2")
        .arg("mem")
        .arg("-h")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    let out = match out_cmd {
        Ok(v) => v,
        Err(_) => return false,
    };
    let mut merged = String::new();
    merged.push_str(&String::from_utf8_lossy(&out.stdout));
    merged.push('\n');
    merged.push_str(&String::from_utf8_lossy(&out.stderr));
    if wrapper_missing_by_output(&merged) {
        return false;
    }
    let msg = merged.to_ascii_lowercase();
    if msg.contains("bwa-mem2") {
        return false;
    }
    msg.contains("program: bwa")
        || msg.contains("usage: bwa ")
        || msg.contains("burrows-wheeler transformation")
}

fn probe_dlc_tool_wrapper(jx_bin: &Path, tool: &str) -> bool {
    let out_cmd = Command::new(jx_bin)
        .arg(tool)
        .arg("-h")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    let out_cmd = match out_cmd {
        Ok(v) => v,
        Err(_) => return false,
    };
    if out_cmd.status.success() {
        return true;
    }
    let mut merged = String::new();
    merged.push_str(&String::from_utf8_lossy(&out_cmd.stdout));
    merged.push('\n');
    merged.push_str(&String::from_utf8_lossy(&out_cmd.stderr));
    !wrapper_missing_by_output(&merged)
}

fn print_toolchain_line(toolchain: &[ToolProbe]) {
    let mut segs = Vec::with_capacity(toolchain.len());
    for (tool, ok, _, _) in toolchain {
        if *ok {
            segs.push(super::style_green(tool));
        } else {
            segs.push(super::style_yellow(tool));
        }
    }
    println!("{}", segs.join(" "));
}

fn missing_tools_from_probe(toolchain: &[ToolProbe]) -> Vec<String> {
    let mut missing = toolchain
        .iter()
        .filter_map(
            |(_, ok, required, missing_hint)| {
                if *required && !*ok {
                    Some(missing_hint.clone())
                } else {
                    None
                }
            },
        )
        .collect::<Vec<String>>();
    missing.sort();
    missing.dedup();
    missing
}

fn wrapper_missing_by_output(text: &str) -> bool {
    let msg = text.to_ascii_lowercase();
    if msg.is_empty() {
        return false;
    }
    msg.contains("please run `jx -update dlc`")
        || msg.contains("dlc runtime is not ready")
        || msg.contains("dlc runtime does not contain")
        || msg.contains("dlc env runtime is missing")
        || msg.contains("unsupported dlc tool")
        || msg.contains("unknown module:")
}

fn start_reference_indexing(
    reference: &Path,
    workdir: &Path,
    backend: &str,
    jx_prefix: &str,
    job_name: &str,
    aligner_cmd: &str,
) -> Result<Option<ReferenceIndexTask>, String> {
    let reference = absolutize_path(reference)?;
    fs::create_dir_all(workdir.join("log")).map_err(|e| {
        format!(
            "Failed to create log directory {}: {e}",
            workdir.join("log").display()
        )
    })?;
    if validate_reference_index_integrity(&reference).is_ok() {
        return Ok(None);
    }

    let dict_path = reference_dict_path(&reference);

    let idx_cmd = format!(
        "{} samtools faidx {} && {} index {} && if [ -f {} ]; then rm -f {}; fi && {} gatk CreateSequenceDictionary -R {} -O {}",
        jx_prefix,
        sh_quote(&reference.to_string_lossy()),
        aligner_cmd,
        sh_quote(&reference.to_string_lossy()),
        sh_quote(&dict_path.to_string_lossy()),
        sh_quote(&dict_path.to_string_lossy()),
        jx_prefix,
        sh_quote(&reference.to_string_lossy()),
        sh_quote(&dict_path.to_string_lossy()),
    );
    let safe_job = crate::pipeline::safe_job_label(job_name);
    let ignore_error_logs = collect_existing_step_error_logs(workdir, &safe_job)?;
    let wrapped = wrap_reference_submit_cmd(&idx_cmd, &safe_job, backend);
    run_shell_capture(workdir, &wrapped, "indexing reference submit")?;
    Ok(Some(ReferenceIndexTask {
        reference,
        workdir: workdir.to_path_buf(),
        safe_job,
        ignore_error_logs,
        is_csub: backend == "csub",
    }))
}

fn wrap_reference_submit_cmd(cmd: &str, safe_job: &str, backend: &str) -> String {
    let quoted_cmd = sh_quote(cmd);
    if backend == "csub" {
        return format!(
            "csub -J {} -o ./log/{}.%J.o -e ./log/{}.%J.e -q c01 -n 1 {}",
            safe_job, safe_job, safe_job, quoted_cmd
        );
    }
    format!(
        "nohup bash -lc {} > ./log/{}.o 2> ./log/{}.e &",
        quoted_cmd, safe_job, safe_job
    )
}

fn wait_reference_indexes(task: &ReferenceIndexTask) -> Result<(), String> {
    match wait_for_reference_index_ready(
        task,
        "Reference index check failed, Rebuilding indexes...",
        Duration::from_millis(500),
        None,
        true,
    )? {
        ReferenceWaitStatus::Ready | ReferenceWaitStatus::Pending => Ok(()),
    }
}

fn wait_reference_indexes_silent(task: &ReferenceIndexTask) -> Result<(), String> {
    match wait_for_reference_index_ready(
        task,
        "Reference index check failed, Rebuilding indexes...",
        Duration::from_millis(500),
        None,
        false,
    )? {
        ReferenceWaitStatus::Ready | ReferenceWaitStatus::Pending => Ok(()),
    }
}

fn reference_sidecar(reference: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{}", reference.to_string_lossy(), suffix))
}

fn reference_dict_path(reference: &Path) -> PathBuf {
    let mut s = reference.to_string_lossy().to_string();
    let lower = s.to_ascii_lowercase();
    if lower.ends_with(".fasta.gz") {
        s.truncate(s.len().saturating_sub(".fasta.gz".len()));
    } else if lower.ends_with(".fa.gz") {
        s.truncate(s.len().saturating_sub(".fa.gz".len()));
    } else if lower.ends_with(".fasta") {
        s.truncate(s.len().saturating_sub(".fasta".len()));
    } else if lower.ends_with(".fa") {
        s.truncate(s.len().saturating_sub(".fa".len()));
    }
    PathBuf::from(format!("{s}.dict"))
}

fn read_fai_data_from_reference(reference: &Path) -> Result<FaiData, String> {
    let reference = absolutize_path(reference)?;
    let fai = reference_sidecar(&reference, ".fai");
    parse_fai_file(&fai)
}

fn validate_reference_index_integrity(reference: &Path) -> Result<(), String> {
    let fai = reference_sidecar(reference, ".fai");
    let dict = reference_dict_path(reference);

    let fai_data = parse_fai_file(&fai)?;
    validate_bwa_or_bwa_mem2_indexes(reference)?;
    let dict_map = parse_dict_file(&dict)?;

    let mut mismatch: Vec<String> = Vec::new();
    for (chrom, fai_len) in &fai_data.lens {
        let Some(dict_len) = dict_map.get(chrom) else {
            mismatch.push(format!("missing in .dict: {chrom}"));
            continue;
        };
        if *dict_len != *fai_len {
            mismatch.push(format!(
                "length mismatch for {chrom} (.fai={fai_len}, .dict={dict_len})"
            ));
        }
    }
    for chrom in dict_map.keys() {
        if !fai_data.lens.contains_key(chrom) {
            mismatch.push(format!("extra contig in .dict: {chrom}"));
        }
    }
    if !mismatch.is_empty() {
        let show = mismatch
            .into_iter()
            .take(8)
            .collect::<Vec<String>>()
            .join("; ");
        return Err(format!(
            "reference index is inconsistent between {} and {}: {}",
            fai.display(),
            dict.display(),
            show
        ));
    }
    Ok(())
}

fn validate_bwa_or_bwa_mem2_indexes(reference: &Path) -> Result<(), String> {
    let mut mem2_missing: Vec<String> = Vec::new();
    for p in bwa_mem2_index_paths(reference) {
        if let Err(e) = check_readable_nonempty(&p) {
            mem2_missing.push(e);
        }
    }
    if mem2_missing.is_empty() {
        return Ok(());
    }
    let mut bwa_missing: Vec<String> = Vec::new();
    for p in bwa_index_paths(reference) {
        if let Err(e) = check_readable_nonempty(&p) {
            bwa_missing.push(e);
        }
    }
    if bwa_missing.is_empty() {
        return Ok(());
    }
    Err(format!(
        "BWA index not ready (need either bwa-mem2 or bwa files). bwa-mem2 missing: {}; bwa missing: {}",
        mem2_missing.join(" | "),
        bwa_missing.join(" | ")
    ))
}

fn bwa_mem2_index_paths(reference: &Path) -> Vec<PathBuf> {
    [".0123", ".amb", ".ann", ".bwt.2bit.64", ".pac"]
        .iter()
        .map(|suffix| reference_sidecar(reference, suffix))
        .collect()
}

fn bwa_index_paths(reference: &Path) -> Vec<PathBuf> {
    [".amb", ".ann", ".bwt", ".pac", ".sa"]
        .iter()
        .map(|suffix| reference_sidecar(reference, suffix))
        .collect()
}

fn check_readable_nonempty(path: &Path) -> Result<(), String> {
    if !path.exists() || !path.is_file() {
        return Err(format!("index file missing: {}", path.display()));
    }
    let md = fs::metadata(path).map_err(|e| format!("Failed to stat {}: {e}", path.display()))?;
    if md.len() == 0 {
        return Err(format!("index file is empty: {}", path.display()));
    }
    File::open(path).map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    Ok(())
}

fn parse_fai_file(fai: &Path) -> Result<FaiData, String> {
    check_readable_nonempty(fai)?;
    let file = File::open(fai)
        .map_err(|e| format!("Failed to open FASTA index {}: {e}", fai.display()))?;
    let reader = BufReader::new(file);
    let mut order = Vec::new();
    let mut lens: BTreeMap<String, u64> = BTreeMap::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read {}: {e}", fai.display()))?;
        let raw = line.trim();
        if raw.is_empty() {
            continue;
        }
        let cols: Vec<&str> = raw.split('\t').collect();
        if cols.len() < 2 {
            return Err(format!("Invalid .fai row in {}: {}", fai.display(), raw));
        }
        let chrom = cols[0].trim();
        if chrom.is_empty() {
            return Err(format!(
                "Invalid .fai row with empty contig in {}",
                fai.display()
            ));
        }
        let len: u64 = cols[1].trim().parse().map_err(|_| {
            format!(
                "Invalid contig length in .fai {} for {}: {}",
                fai.display(),
                chrom,
                cols[1].trim()
            )
        })?;
        if lens.contains_key(chrom) {
            return Err(format!(
                "Duplicate contig in .fai {}: {}",
                fai.display(),
                chrom
            ));
        }
        order.push(chrom.to_string());
        lens.insert(chrom.to_string(), len);
    }
    if order.is_empty() {
        return Err(format!("Empty FASTA index: {}", fai.display()));
    }
    Ok(FaiData { order, lens })
}

fn parse_dict_file(dict: &Path) -> Result<BTreeMap<String, u64>, String> {
    check_readable_nonempty(dict)?;
    let file = File::open(dict)
        .map_err(|e| format!("Failed to open sequence dict {}: {e}", dict.display()))?;
    let reader = BufReader::new(file);
    let mut out: BTreeMap<String, u64> = BTreeMap::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read {}: {e}", dict.display()))?;
        let raw = line.trim();
        if !raw.starts_with("@SQ") {
            continue;
        }
        let mut sn: Option<String> = None;
        let mut ln: Option<u64> = None;
        for field in raw.split('\t').skip(1) {
            if let Some(v) = field.strip_prefix("SN:") {
                let s = v.trim().to_string();
                if !s.is_empty() {
                    sn = Some(s);
                }
                continue;
            }
            if let Some(v) = field.strip_prefix("LN:") {
                let n = v
                    .trim()
                    .parse::<u64>()
                    .map_err(|_| format!("Invalid LN field in {}: {}", dict.display(), raw))?;
                ln = Some(n);
            }
        }
        let Some(name) = sn else {
            return Err(format!("Missing SN field in {}: {}", dict.display(), raw));
        };
        let Some(length) = ln else {
            return Err(format!("Missing LN field in {}: {}", dict.display(), raw));
        };
        if out.insert(name.clone(), length).is_some() {
            return Err(format!("Duplicate SN in {}: {}", dict.display(), name));
        }
    }
    if out.is_empty() {
        return Err(format!(
            "No @SQ entries found in sequence dict {}",
            dict.display()
        ));
    }
    Ok(out)
}

fn collect_fastq_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let root = absolutize_path(root)?;
    let mut stack = vec![root];
    let mut out = Vec::new();
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|e| format!("Failed to read directory {}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
            let path = entry.path();
            let ftype = entry
                .file_type()
                .map_err(|e| format!("Failed to read file type {}: {e}", path.display()))?;
            if ftype.is_dir() {
                stack.push(path);
                continue;
            }
            if !ftype.is_file() {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|x| x.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if FASTQ_SUFFIXES.iter().any(|s| name.ends_with(s)) {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn classify_fastq_pairs(files: &[PathBuf]) -> Result<BTreeMap<String, (PathBuf, PathBuf)>, String> {
    let mut grouped: BTreeMap<String, FastqPair> = BTreeMap::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    for path in files {
        let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
            skipped.push(path.to_string_lossy().to_string());
            continue;
        };
        let Some(stem) = strip_fastq_suffix(name) else {
            skipped.push(path.to_string_lossy().to_string());
            continue;
        };
        let tokens = tokenize_name(&stem);
        let Some((read_idx, read_kind)) = detect_read_token(&tokens) else {
            skipped.push(path.to_string_lossy().to_string());
            continue;
        };
        let sample_key = build_sample_key(&tokens, read_idx, &stem);
        if sample_key.is_empty() {
            skipped.push(path.to_string_lossy().to_string());
            continue;
        }
        let entry = grouped
            .entry(sample_key.clone())
            .or_insert_with(FastqPair::new);
        match read_kind {
            ReadKind::R1 => {
                if entry.r1.is_some() {
                    errors.push(format!(
                        "Duplicate R1 for sample `{}`: {}",
                        sample_key,
                        path.display()
                    ));
                } else {
                    entry.r1 = Some(path.clone());
                }
            }
            ReadKind::R2 => {
                if entry.r2.is_some() {
                    errors.push(format!(
                        "Duplicate R2 for sample `{}`: {}",
                        sample_key,
                        path.display()
                    ));
                } else {
                    entry.r2 = Some(path.clone());
                }
            }
        }
    }

    let mut complete: BTreeMap<String, (PathBuf, PathBuf)> = BTreeMap::new();
    for (sample, pair) in grouped {
        match pair.complete() {
            Ok((r1, r2)) => {
                complete.insert(sample, (r1, r2));
            }
            Err(detail) => {
                errors.push(format!("Sample `{sample}` {detail}."));
            }
        }
    }

    if !errors.is_empty() {
        return Err(format!("FASTQ pairing failed:\n- {}", errors.join("\n- ")));
    }

    if !skipped.is_empty() {
        eprintln!(
            "{}",
            super::style_yellow(&format!(
                "Warning: skipped {} FASTQ file(s) due to unrecognized R1/R2 naming.",
                skipped.len()
            ))
        );
    }
    Ok(complete)
}

fn recognized_fastq_pairing_hint() -> String {
    let suffix_hint = FASTQ_SUFFIXES.join(", ");
    let read_hint = [
        "_R1/_R2",
        ".R1/.R2",
        "_READ1/_READ2",
        "_1/_2",
        ".1/.2",
        "_R1_001/_R2_001",
    ]
    .join(", ");
    format!(
        "No valid paired FASTQ files were detected.\nRecognized FASTQ suffixes: {suffix_hint}\nRecognized R1/R2 naming tokens: {read_hint}\nExamples: sample_R1.fastq.gz + sample_R2.fastq.gz; sample.1.fq.gz + sample.2.fq.gz"
    )
}

fn strip_fastq_suffix(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    for suffix in FASTQ_SUFFIXES {
        if lower.ends_with(suffix) {
            let keep = name.len().saturating_sub(suffix.len());
            return Some(name[..keep].to_string());
        }
    }
    None
}

fn tokenize_name(stem: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in stem.chars() {
        if matches!(ch, '_' | '-' | '.') {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            continue;
        }
        current.push(ch);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn detect_read_token(tokens: &[String]) -> Option<(usize, ReadKind)> {
    for (idx, tok) in tokens.iter().enumerate().rev() {
        let upper = tok.to_ascii_uppercase();
        if upper == "R1" || upper == "READ1" {
            return Some((idx, ReadKind::R1));
        }
        if upper == "R2" || upper == "READ2" {
            return Some((idx, ReadKind::R2));
        }
        if upper.starts_with("R1")
            && upper.len() > 2
            && upper[2..].chars().all(|c| c.is_ascii_digit())
        {
            return Some((idx, ReadKind::R1));
        }
        if upper.starts_with("R2")
            && upper.len() > 2
            && upper[2..].chars().all(|c| c.is_ascii_digit())
        {
            return Some((idx, ReadKind::R2));
        }
    }
    for (idx, tok) in tokens.iter().enumerate().rev() {
        if tok == "1" {
            return Some((idx, ReadKind::R1));
        }
        if tok == "2" {
            return Some((idx, ReadKind::R2));
        }
    }
    None
}

fn build_sample_key(tokens: &[String], read_idx: usize, stem: &str) -> String {
    let mut kept = Vec::new();
    for (idx, token) in tokens.iter().enumerate() {
        if idx != read_idx {
            kept.push(token.to_string());
        }
    }
    if let Some(last) = kept.last() {
        if last == "001" || last == "0001" {
            kept.pop();
        }
    }
    if kept.is_empty() {
        return sanitize_sample_key(stem);
    }
    sanitize_sample_key(&kept.join("_"))
}

fn sanitize_sample_key(raw: &str) -> String {
    let mut out = String::new();
    let mut prev_underscore = false;
    for ch in raw.chars() {
        let normalized = if ch.is_ascii_alphanumeric() || ch == '_' {
            ch
        } else if matches!(ch, '-' | '.' | ' ') {
            '_'
        } else {
            ch
        };
        if normalized == '_' {
            if prev_underscore {
                continue;
            }
            prev_underscore = true;
            out.push('_');
        } else {
            prev_underscore = false;
            out.push(normalized);
        }
    }
    out.trim_matches('_').to_string()
}

fn build_run_params_json(
    reference: &Path,
    samples: &BTreeMap<String, (PathBuf, PathBuf)>,
    chroms: &[String],
    scheduler: &str,
    nohup_max_jobs: usize,
    singularity: &str,
) -> String {
    let mut json = String::new();
    json.push('{');
    json.push_str("\"chrom\":[");
    for (idx, chrom) in chroms.iter().enumerate() {
        if idx > 0 {
            json.push(',');
        }
        json.push_str(&json_string(chrom));
    }
    json.push(']');
    json.push_str(",\"nohup_max_jobs\":");
    json.push_str(&nohup_max_jobs.to_string());
    json.push_str(",\"reference\":");
    json.push_str(&json_string(&reference.to_string_lossy()));
    json.push_str(",\"samples\":{");
    let mut first = true;
    for (sample, (r1, r2)) in samples {
        if !first {
            json.push(',');
        }
        first = false;
        json.push_str(&json_string(sample));
        json.push(':');
        json.push('[');
        json.push_str(&json_string(&r1.to_string_lossy()));
        json.push(',');
        json.push_str(&json_string(&r2.to_string_lossy()));
        json.push(']');
    }
    json.push('}');
    json.push_str(",\"scheduler\":");
    json.push_str(&json_string(scheduler));
    json.push_str(",\"singularity\":");
    json.push_str(&json_string(singularity));
    json.push('}');
    json
}

fn build_input_files(
    reference: &Path,
    samples: &BTreeMap<String, (PathBuf, PathBuf)>,
) -> Vec<PathBuf> {
    let mut files = vec![reference.to_path_buf()];
    for (r1, r2) in samples.values() {
        files.push(r1.clone());
        files.push(r2.clone());
    }
    dedup_paths(files)
}

fn infer_bulks(samples: Vec<String>) -> Vec<String> {
    let mut keys: Vec<String> = samples
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    keys.sort();
    keys.dedup();
    if keys.is_empty() {
        return Vec::new();
    }
    let prefixes: Vec<String> = keys
        .iter()
        .map(|s| {
            if let Some(pos) = s.find('_') {
                s[..pos].to_string()
            } else {
                s.to_string()
            }
        })
        .collect();
    let mut uniq = prefixes.clone();
    uniq.sort();
    uniq.dedup();
    if uniq.len() >= 2 {
        return uniq;
    }
    keys
}

fn latest_snp_table(workdir: &Path) -> Option<PathBuf> {
    let merge_dir = workdir.join("4.merge");
    if !merge_dir.exists() {
        return None;
    }
    let entries = fs::read_dir(&merge_dir).ok()?;
    let mut tables: Vec<PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
            continue;
        };
        if name.starts_with("Merge.") && name.ends_with(".SNP.tsv") {
            tables.push(path);
        }
    }
    if tables.is_empty() {
        return None;
    }
    tables.sort_by_key(|p| {
        p.metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    });
    tables.pop()
}

fn build_fastp_step(
    samples: &BTreeMap<String, (PathBuf, PathBuf)>,
    workdir: &Path,
    backend: &str,
    singularity: &str,
) -> Result<PipelineStep, String> {
    let cleanfolder = workdir.join("1.clean");
    fs::create_dir_all(&cleanfolder)
        .map_err(|e| format!("Failed to create {}: {e}", cleanfolder.display()))?;

    let sample_names: Vec<String> = samples.keys().cloned().collect();
    let mut step_cmds = Vec::new();
    let mut step_inputs = Vec::new();
    let mut step_outputs = Vec::new();
    let mut step_items = Vec::new();
    for sample in &sample_names {
        let Some((fq1, fq2)) = samples.get(sample) else {
            return Err(format!("Sample missing FASTQ pair in metadata: {sample}"));
        };
        step_inputs.push(fq1.clone());
        step_inputs.push(fq2.clone());
        let out_r1 = cleanfolder.join(format!("{sample}.R1.clean.fastq.gz"));
        let out_r2 = cleanfolder.join(format!("{sample}.R2.clean.fastq.gz"));
        let out_html = cleanfolder.join(format!("{sample}.html"));
        let out_json = cleanfolder.join(format!("{sample}.json"));
        let raw = cmd_fastp(sample, fq1, fq2, &cleanfolder, 16, singularity);
        step_cmds.push(wrap_scheduler_cmd(
            &raw,
            &format!("fastp.{sample}"),
            16,
            backend,
        ));
        let outs = vec![out_r1, out_r2, out_html, out_json];
        step_outputs.extend(outs.clone());
        step_items.push(StepItem {
            id: format!("fastp.{sample}"),
            outputs: outs,
        });
    }
    Ok(PipelineStep {
        id: "step1_fastp".to_string(),
        name: "fastp".to_string(),
        eta_minutes: Some(15),
        commands: step_cmds,
        inputs: dedup_paths(step_inputs),
        outputs: dedup_paths(step_outputs),
        items: step_items,
    })
}

fn build_fastq2vcf_steps(
    reference: &Path,
    samples: &BTreeMap<String, (PathBuf, PathBuf)>,
    chroms: &[String],
    chrom_lens: &BTreeMap<String, u64>,
    workdir: &Path,
    backend: &str,
    aligner_cmd: &str,
    singularity: &str,
    include_fastp_step: bool,
) -> Result<Vec<PipelineStep>, String> {
    let cleanfolder = workdir.join("1.clean");
    let mappingfolder = workdir.join("2.mapping");
    let gvcffolder = workdir.join("3.gvcf");
    let mergefolder = workdir.join("4.merge");
    let imputefolder = workdir.join("5.impute");
    let genotypefolder = workdir.join("6.genotype");

    for d in [
        &cleanfolder,
        &mappingfolder,
        &gvcffolder,
        &mergefolder,
        &imputefolder,
        &genotypefolder,
    ] {
        fs::create_dir_all(d).map_err(|e| format!("Failed to create {}: {e}", d.display()))?;
    }

    let sample_names: Vec<String> = samples.keys().cloned().collect();
    let dynamic_eta = estimate_step5_to_step10_eta_minutes(chroms, chrom_lens, sample_names.len());
    let mut steps: Vec<PipelineStep> = Vec::new();

    if include_fastp_step {
        steps.push(build_fastp_step(samples, workdir, backend, singularity)?);
    }

    let mut step2_cmds = Vec::new();
    let mut step2_inputs = Vec::new();
    let mut step2_outputs = Vec::new();
    let mut step2_items = Vec::new();
    for sample in &sample_names {
        let r1 = cleanfolder.join(format!("{sample}.R1.clean.fastq.gz"));
        let r2 = cleanfolder.join(format!("{sample}.R2.clean.fastq.gz"));
        step2_inputs.push(r1.clone());
        step2_inputs.push(r2.clone());
        let out_bam = mappingfolder.join(format!("{sample}.sorted.bam"));
        let out_done = mappingfolder.join(format!("{sample}.sorted.bam.finished"));
        let raw = cmd_bwamem(
            reference,
            sample,
            &r1,
            &r2,
            &mappingfolder,
            64,
            aligner_cmd,
            singularity,
        );
        step2_cmds.push(wrap_scheduler_cmd(
            &raw,
            &format!("bwamem.{sample}"),
            64,
            backend,
        ));
        let outs = vec![out_bam, out_done];
        step2_outputs.extend(outs.clone());
        step2_items.push(StepItem {
            id: format!("bwamem.{sample}"),
            outputs: outs,
        });
    }
    steps.push(PipelineStep {
        id: "step2_bwamem".to_string(),
        name: "bwamem".to_string(),
        eta_minutes: Some(255),
        commands: step2_cmds,
        inputs: dedup_paths(step2_inputs),
        outputs: dedup_paths(step2_outputs),
        items: step2_items,
    });

    let mut step3_cmds = Vec::new();
    let mut step3_inputs = Vec::new();
    let mut step3_outputs = Vec::new();
    let mut step3_items = Vec::new();
    let markdup_threads = 8usize;
    for sample in &sample_names {
        let bam = mappingfolder.join(format!("{sample}.sorted.bam"));
        step3_inputs.push(bam.clone());
        let out_md_bam = mappingfolder.join(format!("{sample}.Markdup.bam"));
        let out_md_bai = mappingfolder.join(format!("{sample}.Markdup.bam.bai"));
        let out_md_metrics = mappingfolder.join(format!("{sample}.Markdup.metrics.txt"));
        let raw = cmd_markdup(sample, &bam, &mappingfolder, markdup_threads, 200, singularity);
        step3_cmds.push(wrap_scheduler_cmd(
            &raw,
            &format!("markdup.{sample}"),
            markdup_threads,
            backend,
        ));
        let outs = vec![out_md_bam, out_md_bai, out_md_metrics];
        step3_outputs.extend(outs.clone());
        step3_items.push(StepItem {
            id: format!("markdup.{sample}"),
            outputs: outs,
        });
    }
    steps.push(PipelineStep {
        id: "step3_markdup".to_string(),
        name: "markdup".to_string(),
        eta_minutes: Some(30),
        commands: step3_cmds,
        inputs: dedup_paths(step3_inputs),
        outputs: dedup_paths(step3_outputs),
        items: step3_items,
    });

    let mut step4_cmds = Vec::new();
    let mut step4_inputs = Vec::new();
    let mut step4_outputs = Vec::new();
    let mut step4_items = Vec::new();
    for sample in &sample_names {
        let md_bam = mappingfolder.join(format!("{sample}.Markdup.bam"));
        step4_inputs.push(md_bam.clone());
        for chrom in chroms {
            let out_g = gvcffolder.join(format!("{sample}.{chrom}.g.vcf.gz"));
            let out_tbi = gvcffolder.join(format!("{sample}.{chrom}.g.vcf.gz.tbi"));
            let raw = cmd_bam2gvcf(
                reference,
                sample,
                &md_bam,
                chrom,
                &gvcffolder,
                2,
                singularity,
            );
            step4_cmds.push(wrap_scheduler_cmd(
                &raw,
                &format!("bam2gvcf.{sample}.{chrom}"),
                2,
                backend,
            ));
            let outs = vec![out_g, out_tbi];
            step4_outputs.extend(outs.clone());
            step4_items.push(StepItem {
                id: format!("bam2gvcf.{sample}.{chrom}"),
                outputs: outs,
            });
        }
    }
    steps.push(PipelineStep {
        id: "step4_bam2gvcf".to_string(),
        name: "bam2gvcf".to_string(),
        eta_minutes: Some(451),
        commands: step4_cmds,
        inputs: dedup_paths(step4_inputs),
        outputs: dedup_paths(step4_outputs),
        items: step4_items,
    });

    let mut step5_cmds = Vec::new();
    let mut step5_inputs = Vec::new();
    let mut step5_outputs = Vec::new();
    let mut step5_items = Vec::new();
    for chrom in chroms {
        let mut gvcfs: Vec<PathBuf> = Vec::new();
        for sample in &sample_names {
            let p = gvcffolder.join(format!("{sample}.{chrom}.g.vcf.gz"));
            step5_inputs.push(p.clone());
            gvcfs.push(p);
        }
        let out_gendb = mergefolder.join(format!("Merge.{chrom}.gendb"));
        let out_gendb_done = mergefolder.join(format!("Merge.{chrom}.gendb.done"));
        let raw = cmd_cgvcf(reference, chrom, &gvcfs, &mergefolder, 2, singularity);
        step5_cmds.push(wrap_scheduler_cmd(
            &raw,
            &format!("cgvcf.{chrom}"),
            2,
            backend,
        ));
        let outs = vec![out_gendb, out_gendb_done];
        step5_outputs.extend(outs.clone());
        step5_items.push(StepItem {
            id: format!("cgvcf.{chrom}"),
            outputs: outs,
        });
    }
    steps.push(PipelineStep {
        id: "step5_cgvcf".to_string(),
        name: "cgvcf".to_string(),
        eta_minutes: Some(dynamic_eta.step5_cgvcf),
        commands: step5_cmds,
        inputs: dedup_paths(step5_inputs),
        outputs: dedup_paths(step5_outputs),
        items: step5_items,
    });

    let mut step6_cmds = Vec::new();
    let mut step6_inputs = Vec::new();
    let mut step6_outputs = Vec::new();
    let mut step6_items = Vec::new();
    for chrom in chroms {
        let in_gendb = mergefolder.join(format!("Merge.{chrom}.gendb"));
        let in_done = mergefolder.join(format!("Merge.{chrom}.gendb.done"));
        step6_inputs.push(in_gendb);
        step6_inputs.push(in_done);
        let out_snp_f = mergefolder.join(format!("Merge.{chrom}.SNP.filtered.vcf.gz"));
        let out_snp_f_tbi = mergefolder.join(format!("Merge.{chrom}.SNP.filtered.vcf.gz.tbi"));
        let out_snp_tsv = mergefolder.join(format!("Merge.{chrom}.SNP.tsv"));
        let cmd6 = cmd_gvcf2snp_table(reference, chrom, &mergefolder, 2, 50, singularity);
        step6_cmds.push(wrap_scheduler_cmd(
            &cmd6,
            &format!("gvcf2vcf.{chrom}"),
            2,
            backend,
        ));
        let outs = vec![out_snp_f, out_snp_f_tbi, out_snp_tsv];
        step6_outputs.extend(outs.clone());
        step6_items.push(StepItem {
            id: format!("gvcf2vcf.{chrom}"),
            outputs: outs,
        });
    }
    steps.push(PipelineStep {
        id: "step6_gvcf2vcf".to_string(),
        name: "gvcf2vcf".to_string(),
        eta_minutes: Some(dynamic_eta.step6_gvcf2vcf),
        commands: step6_cmds,
        inputs: dedup_paths(step6_inputs),
        outputs: dedup_paths(step6_outputs),
        items: step6_items,
    });

    let mut step7_cmds = Vec::new();
    let mut step7_inputs = Vec::new();
    let mut step7_outputs = Vec::new();
    let mut step7_items = Vec::new();
    for chrom in chroms {
        let in_vcf = mergefolder.join(format!("Merge.{chrom}.SNP.filtered.vcf.gz"));
        step7_inputs.push(in_vcf);
        let out_gt = imputefolder.join(format!("Merge.{chrom}.SNP.GT.vcf.gz"));
        let out_gt_tbi = imputefolder.join(format!("Merge.{chrom}.SNP.GT.vcf.gz.tbi"));
        let out_lmiss = imputefolder.join(format!("Merge.{chrom}.SNP.GT.lmiss"));
        let out_imiss = imputefolder.join(format!("Merge.{chrom}.SNP.GT.imiss"));
        let raw = cmd_snpvcf_to_gt_and_missing(
            chrom,
            &mergefolder,
            &imputefolder,
            5,
            20,
            2,
            2,
            singularity,
        );
        step7_cmds.push(wrap_scheduler_cmd(
            &raw,
            &format!("gtprep.{chrom}"),
            2,
            backend,
        ));
        let outs = vec![out_gt, out_gt_tbi, out_lmiss, out_imiss];
        step7_outputs.extend(outs.clone());
        step7_items.push(StepItem {
            id: format!("gtprep.{chrom}"),
            outputs: outs,
        });
    }
    steps.push(PipelineStep {
        id: "step7_gtprep".to_string(),
        name: "gtprep".to_string(),
        eta_minutes: Some(dynamic_eta.step7_gtprep),
        commands: step7_cmds,
        inputs: dedup_paths(step7_inputs),
        outputs: dedup_paths(step7_outputs),
        items: step7_items,
    });

    let mut step8_cmds = Vec::new();
    let mut step8_inputs = Vec::new();
    let mut step8_outputs = Vec::new();
    let mut step8_items = Vec::new();
    let impute_threads_max = 32usize;
    for chrom in chroms {
        let in_gt = imputefolder.join(format!("Merge.{chrom}.SNP.GT.vcf.gz"));
        step8_inputs.push(in_gt);
        let out_imp = imputefolder.join(format!("Merge.{chrom}.SNP.GT.imp.vcf.gz"));
        let out_imp_tbi = imputefolder.join(format!("Merge.{chrom}.SNP.GT.imp.vcf.gz.tbi"));
        let out_imp_ok = imputefolder.join(format!("Merge.{chrom}.SNP.GT.imp.ok"));
        let chrom_threads = dynamic_impute_threads(chrom, chrom_lens, impute_threads_max);
        let raw = cmd_beagle_impute(chrom, &imputefolder, chrom_threads, singularity);
        step8_cmds.push(wrap_scheduler_cmd(
            &raw,
            &format!("beagle.{chrom}"),
            chrom_threads,
            backend,
        ));
        let outs = vec![out_imp, out_imp_tbi, out_imp_ok];
        step8_outputs.extend(outs.clone());
        step8_items.push(StepItem {
            id: format!("beagle.{chrom}"),
            outputs: outs,
        });
    }
    steps.push(PipelineStep {
        id: "step8_impute".to_string(),
        name: "impute".to_string(),
        eta_minutes: Some(dynamic_eta.step8_impute),
        commands: step8_cmds,
        inputs: dedup_paths(step8_inputs),
        outputs: dedup_paths(step8_outputs),
        items: step8_items,
    });

    let mut step9_cmds = Vec::new();
    let mut step9_inputs = Vec::new();
    let mut step9_outputs = Vec::new();
    let mut step9_items = Vec::new();
    for chrom in chroms {
        step9_inputs.push(imputefolder.join(format!("Merge.{chrom}.SNP.GT.imp.vcf.gz")));
        step9_inputs.push(imputefolder.join(format!("Merge.{chrom}.SNP.GT.imp.ok")));
        step9_inputs.push(imputefolder.join(format!("Merge.{chrom}.SNP.GT.lmiss")));
        step9_inputs.push(imputefolder.join(format!("Merge.{chrom}.SNP.GT.imiss")));
        let out_vcf = imputefolder.join(format!("Merge.{chrom}.SNP.GT.imp.maf0.02.miss0.2.vcf.gz"));
        let out_tbi = imputefolder.join(format!(
            "Merge.{chrom}.SNP.GT.imp.maf0.02.miss0.2.vcf.gz.tbi"
        ));
        let raw = cmd_filter_imputed_by_maf_and_missing(
            chrom,
            chroms,
            &imputefolder,
            0.02,
            0.2,
            2,
            singularity,
        );
        step9_cmds.push(wrap_scheduler_cmd(
            &raw,
            &format!("impfilter.{chrom}"),
            2,
            backend,
        ));
        let outs = vec![out_vcf, out_tbi];
        step9_outputs.extend(outs.clone());
        step9_items.push(StepItem {
            id: format!("impfilter.{chrom}"),
            outputs: outs,
        });
    }
    steps.push(PipelineStep {
        id: "step9_impfilter".to_string(),
        name: "impfilter".to_string(),
        eta_minutes: Some(dynamic_eta.step9_impfilter),
        commands: step9_cmds,
        inputs: dedup_paths(step9_inputs),
        outputs: dedup_paths(step9_outputs),
        items: step9_items,
    });

    let mut step10_inputs = Vec::new();
    for chrom in chroms {
        step10_inputs
            .push(imputefolder.join(format!("Merge.{chrom}.SNP.GT.imp.maf0.02.miss0.2.vcf.gz")));
    }
    let out_merge_vcf = genotypefolder.join("Merge.SNP.GT.imp.maf0.02.miss0.2.vcf.gz");
    let out_merge_tbi = genotypefolder.join("Merge.SNP.GT.imp.maf0.02.miss0.2.vcf.gz.tbi");
    let raw10 = cmd_concat_imputed_vcfs(chroms, &imputefolder, &genotypefolder, 2, singularity);
    let step10_cmds = vec![wrap_scheduler_cmd(&raw10, "mergevcf.all", 2, backend)];
    let step10_outputs = vec![out_merge_vcf, out_merge_tbi];
    let step10_items = vec![StepItem {
        id: "mergevcf.all".to_string(),
        outputs: step10_outputs.clone(),
    }];
    steps.push(PipelineStep {
        id: "step10_mergevcf".to_string(),
        name: "mergevcf".to_string(),
        eta_minutes: Some(dynamic_eta.step10_mergevcf),
        commands: step10_cmds,
        inputs: dedup_paths(step10_inputs),
        outputs: dedup_paths(step10_outputs),
        items: step10_items,
    });

    Ok(steps)
}

fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut seen = BTreeMap::<String, ()>::new();
    for p in paths {
        let key = p.to_string_lossy().to_string();
        if seen.contains_key(&key) {
            continue;
        }
        seen.insert(key, ());
        out.push(p);
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct DynamicEtaStep5To10 {
    step5_cgvcf: u64,
    step6_gvcf2vcf: u64,
    step7_gtprep: u64,
    step8_impute: u64,
    step9_impfilter: u64,
    step10_mergevcf: u64,
}

fn estimate_step5_to_step10_eta_minutes(
    chroms: &[String],
    chrom_lens: &BTreeMap<String, u64>,
    sample_count: usize,
) -> DynamicEtaStep5To10 {
    // Use reference locus scale as "site count" proxy before VCF is generated.
    let locus_count: f64 = chroms
        .iter()
        .map(|c| chrom_lens.get(c).copied().unwrap_or(0) as f64)
        .sum();
    let site_factor = (locus_count / 500_000_000.0).clamp(0.35, 10.0);
    let site_factor_sqrt = site_factor.sqrt();

    let samples = sample_count.max(1) as f64;
    let pop_factor_linear = (samples / 2.0).clamp(0.7, 20.0);
    let pop_factor_sqrt = pop_factor_linear.sqrt();

    let chrom_factor = ((chroms.len().max(1) as f64) / 10.0).clamp(0.4, 4.0);

    DynamicEtaStep5To10 {
        step5_cgvcf: scale_eta_minutes(
            36.0 * site_factor_sqrt * pop_factor_sqrt,
            8,
            12 * 60,
        ),
        step6_gvcf2vcf: scale_eta_minutes(
            32.0 * site_factor_sqrt * (0.6 + 0.4 * pop_factor_sqrt),
            6,
            10 * 60,
        ),
        step7_gtprep: scale_eta_minutes(
            40.0 * site_factor_sqrt * (0.5 + 0.5 * pop_factor_sqrt),
            8,
            12 * 60,
        ),
        step8_impute: scale_eta_minutes(
            120.0 * site_factor_sqrt * (0.7 + 0.6 * pop_factor_sqrt),
            20,
            24 * 60,
        ),
        step9_impfilter: scale_eta_minutes(
            24.0 * site_factor_sqrt * (0.7 + 0.3 * pop_factor_sqrt),
            5,
            6 * 60,
        ),
        step10_mergevcf: scale_eta_minutes(
            8.0 * chrom_factor * site_factor_sqrt * (0.8 + 0.2 * pop_factor_sqrt),
            2,
            4 * 60,
        ),
    }
}

fn scale_eta_minutes(value: f64, min_minutes: u64, max_minutes: u64) -> u64 {
    let raw = if value.is_finite() { value.round() } else { min_minutes as f64 };
    raw.max(min_minutes as f64).min(max_minutes as f64) as u64
}

fn dynamic_impute_threads(chrom: &str, chrom_lens: &BTreeMap<String, u64>, max_threads: usize) -> usize {
    let max_threads = max_threads.max(1);
    if max_threads <= 4 {
        return max_threads;
    }
    let Some(max_len) = chrom_lens.values().copied().max() else {
        return max_threads;
    };
    if max_len == 0 {
        return max_threads;
    }
    let len = chrom_lens.get(chrom).copied().unwrap_or(max_len);
    let ratio = (len as f64) / (max_len as f64);
    let t = if ratio >= 0.70 {
        32
    } else if ratio >= 0.40 {
        16
    } else if ratio >= 0.20 {
        8
    } else {
        4
    };
    t.min(max_threads).max(1)
}

fn run_shell_capture(cwd: &Path, cmdline: &str, desc: &str) -> Result<(), String> {
    let mut cmd = shell_cmd(cmdline);
    cmd.current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let out = cmd
        .output()
        .map_err(|e| format!("Failed to run command for {desc}: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let mut msg = String::new();
    msg.push_str(&String::from_utf8_lossy(&out.stdout));
    msg.push_str(&String::from_utf8_lossy(&out.stderr));
    Err(format!(
        "{desc} failed with exit={}.\n{}",
        super::exit_code(out.status),
        msg.trim()
    ))
}

fn wait_for_reference_index_ready(
    task: &ReferenceIndexTask,
    desc: &str,
    _poll: Duration,
    soft_wait: Option<Duration>,
    emit_ui: bool,
) -> Result<ReferenceWaitStatus, String> {
    if validate_reference_index_integrity(&task.reference).is_ok() {
        return Ok(ReferenceWaitStatus::Ready);
    }
    let start = Instant::now();
    let max_wait_non_csub = Duration::from_secs(30 * 60);
    let max_wait_csub_hard = Duration::from_secs(72 * 60 * 60);
    let csub_inactive_timeout = Duration::from_secs(30 * 60);
    let mut csub_last_active = Instant::now();
    let mut csub_probe_issue: Option<String> = None;
    let is_tty = emit_ui && io::stdout().is_terminal();
    loop {
        match validate_reference_index_integrity(&task.reference) {
            Ok(()) => break,
            Err(_) => {}
        }
        if let Some(err) = find_failed_step_error_snippet(task)? {
            if is_tty {
                print!("\r\x1b[2K");
                io::stdout()
                    .flush()
                    .map_err(|e| format!("Failed to flush progress output: {e}"))?;
            }
            return Err(format!("{desc} failed: {err}"));
        }
        if let Some(limit) = soft_wait {
            if start.elapsed() >= limit {
                if is_tty {
                    print!("\r\x1b[2K");
                    io::stdout()
                        .flush()
                        .map_err(|e| format!("Failed to flush progress output: {e}"))?;
                }
                return Ok(ReferenceWaitStatus::Pending);
            }
        }
        if task.is_csub {
            match csub_job_is_active(&task.safe_job) {
                Ok(true) => {
                    csub_last_active = Instant::now();
                    csub_probe_issue = None;
                }
                Ok(false) => {}
                Err(e) => {
                    csub_probe_issue = Some(e);
                    // Do not treat cjobs probe failure as immediate inactivity.
                    csub_last_active = Instant::now();
                }
            }
            if start.elapsed() >= max_wait_csub_hard {
                if is_tty {
                    print!("\r\x1b[2K");
                    io::stdout()
                        .flush()
                        .map_err(|e| format!("Failed to flush progress output: {e}"))?;
                }
                let probe_tip = csub_probe_issue
                    .as_ref()
                    .map(|e| format!(" Last cjobs probe error: {e}"))
                    .unwrap_or_default();
                return Err(format!(
                    "{desc} timeout after {} (hard limit) with csub mode. Please check {} and queue status.{probe_tip}",
                    super::format_elapsed(max_wait_csub_hard),
                    task.workdir.join("log").display()
                ));
            }
            if csub_last_active.elapsed() >= csub_inactive_timeout {
                if is_tty {
                    print!("\r\x1b[2K");
                    io::stdout()
                        .flush()
                        .map_err(|e| format!("Failed to flush progress output: {e}"))?;
                }
                return Err(format!(
                    "{desc} timeout: no active csub job named `{}` for {} and indexes are still incomplete. Check {}.",
                    task.safe_job,
                    super::format_elapsed(csub_inactive_timeout),
                    task.workdir.join("log").display()
                ));
            }
        } else if start.elapsed() >= max_wait_non_csub {
            if is_tty {
                print!("\r\x1b[2K");
                io::stdout()
                    .flush()
                    .map_err(|e| format!("Failed to flush progress output: {e}"))?;
            }
            return Err(format!(
                "{desc} timeout after {}. Please check log files under {}.",
                super::format_elapsed(max_wait_non_csub),
                task.workdir.join("log").display()
            ));
        }
        if is_tty {
            let elapsed = start.elapsed();
            let line = format!(
                "\r{} {} [{}]",
                super::spinner_frame_for_elapsed(elapsed),
                desc,
                super::format_elapsed_live(elapsed)
            );
            if super::supports_color() {
                print!("{}", super::style_yellow(&line));
            } else {
                print!("{line}");
            }
            io::stdout()
                .flush()
                .map_err(|e| format!("Failed to flush progress output: {e}"))?;
        }
        thread::sleep(super::spinner_refresh_interval(start.elapsed()));
    }
    if is_tty {
        print!("\r\x1b[2K");
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush progress output: {e}"))?;
    }
    if emit_ui {
        super::print_success_line(&format!(
            "{} ...Finished [{}]",
            "Reference index check",
            super::format_elapsed_live(start.elapsed())
        ));
    }
    Ok(ReferenceWaitStatus::Ready)
}

fn run_fastp_and_index_parallel(
    workdir: &Path,
    fastp_step: &PipelineStep,
    fastp_opts: &PipelineOptions,
    index_task: &ReferenceIndexTask,
) -> Result<(Duration, Duration), String> {
    let workdir_c = workdir.to_path_buf();
    let fastp_step_c = fastp_step.clone();
    let fastp_opts_c = fastp_opts.clone();
    let index_task_c = index_task.clone();

    let (tx_fastp, rx_fastp) = mpsc::channel::<Result<(), String>>();
    let (tx_index, rx_index) = mpsc::channel::<Result<(), String>>();

    let fastp_start = Instant::now();
    let index_start = Instant::now();

    thread::spawn(move || {
        let mut noop_hook = |_step: &PipelineStep, _item: &StepItem| Ok(());
        let res = run_pipeline_with_hook(
            &workdir_c,
            std::slice::from_ref(&fastp_step_c),
            &fastp_opts_c,
            &mut noop_hook,
        );
        let _ = tx_fastp.send(res);
    });

    thread::spawn(move || {
        let res = wait_reference_indexes_silent(&index_task_c);
        let _ = tx_index.send(res);
    });

    let mut fastp_done: Option<Result<(), String>> = None;
    let mut index_done: Option<Result<(), String>> = None;
    let mut fastp_elapsed = Duration::from_secs(0);
    let mut index_elapsed = Duration::from_secs(0);
    let is_tty = io::stdout().is_terminal();
    let mut rendered = false;

    if is_tty {
        // Reserve exactly two lines for deterministic in-place refresh:
        // line 1 = reference index, line 2 = fastp step.
        print!("\n\n");
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to initialize parallel progress block: {e}"))?;
    }

    loop {
        if fastp_done.is_none() {
            match rx_fastp.try_recv() {
                Ok(v) => {
                    fastp_elapsed = fastp_start.elapsed();
                    fastp_done = Some(v);
                }
                Err(TryRecvError::Disconnected) => {
                    fastp_elapsed = fastp_start.elapsed();
                    fastp_done = Some(Err("fastp worker disconnected unexpectedly".to_string()));
                }
                Err(TryRecvError::Empty) => {}
            }
        }
        if index_done.is_none() {
            match rx_index.try_recv() {
                Ok(v) => {
                    index_elapsed = index_start.elapsed();
                    index_done = Some(v);
                }
                Err(TryRecvError::Disconnected) => {
                    index_elapsed = index_start.elapsed();
                    index_done = Some(Err("reference index worker disconnected unexpectedly".to_string()));
                }
                Err(TryRecvError::Empty) => {}
            }
        }

        if is_tty {
            // Always move to the beginning of the reserved two-line block.
            print!("\x1b[2F");

            let idx_elapsed_now = if index_done.is_some() {
                index_elapsed
            } else {
                index_start.elapsed()
            };
            let idx_line = if index_done.is_some() {
                format!(
                    "✔︎ Reference index check ...Finished [{}]",
                    super::format_elapsed_live(idx_elapsed_now)
                )
            } else {
                format!(
                    "{} Reference index check failed, Rebuilding indexes... [{}]",
                    super::spinner_frame_for_elapsed(idx_elapsed_now),
                    super::format_elapsed_live(idx_elapsed_now)
                )
            };
            print!("\x1b[2K\r{}\n", super::style_yellow(&idx_line));

            let fp_elapsed_now = if fastp_done.is_some() {
                fastp_elapsed
            } else {
                fastp_start.elapsed()
            };
            let fp_line = if fastp_done.is_some() {
                format!(
                    "✔︎ Step 1/10: fastp (~15m) ...Finished [{}]",
                    super::format_elapsed_live(fp_elapsed_now)
                )
            } else {
                format!(
                    "{} Step 1/10: fastp (~15m) [{}]",
                    super::spinner_frame_for_elapsed(fp_elapsed_now),
                    super::format_elapsed_live(fp_elapsed_now)
                )
            };
            print!("\x1b[2K\r{}\n", super::style_green(&fp_line));
            io::stdout()
                .flush()
                .map_err(|e| format!("Failed to flush parallel progress output: {e}"))?;
            rendered = true;
        }

        if fastp_done.is_some() && index_done.is_some() {
            break;
        }
        let tick_elapsed = if index_done.is_none() {
            index_start.elapsed()
        } else {
            fastp_start.elapsed()
        };
        thread::sleep(super::spinner_refresh_interval(tick_elapsed));
    }

    if is_tty && rendered {
        // Delete the reserved two-line block so no blank lines remain.
        // Cursor is at the line below block; move to block start then delete 2 lines.
        print!("\x1b[2F\x1b[2M\r");
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to clear parallel progress block: {e}"))?;
    }

    if let Some(Err(e)) = index_done {
        return Err(e);
    }
    if let Some(Err(e)) = fastp_done {
        return Err(e);
    }
    Ok((fastp_elapsed, index_elapsed))
}

fn csub_job_is_active(safe_job: &str) -> Result<bool, String> {
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
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let raw = line.trim();
        if raw.is_empty() || raw.starts_with("JOBID") {
            continue;
        }
        let cols: Vec<&str> = raw.split_whitespace().collect();
        if cols.len() < 7 {
            continue;
        }
        let stat = cols[2].trim().to_ascii_uppercase();
        let active = matches!(
            stat.as_str(),
            "RUN" | "PEND" | "PSUSP" | "USUSP" | "SSUSP"
        );
        if !active {
            continue;
        }
        if csub_job_name_matches_safe(cols[6].trim(), safe_job) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn csub_job_name_matches_safe(job_name: &str, safe_job: &str) -> bool {
    let j = job_name.trim();
    if j.is_empty() {
        return false;
    }
    if j == safe_job {
        return true;
    }
    let j1 = j.trim_start_matches('*');
    if !j1.is_empty() && (j1 == safe_job || safe_job.ends_with(j1)) {
        return true;
    }
    let j2 = j.trim_matches('*');
    !j2.is_empty() && (j2 == safe_job || safe_job.ends_with(j2))
}

fn collect_existing_step_error_logs(
    workdir: &Path,
    safe_job: &str,
) -> Result<HashSet<PathBuf>, String> {
    let mut ignore: HashSet<PathBuf> = HashSet::new();
    let log_dir = workdir.join("log");
    if !log_dir.exists() {
        return Ok(ignore);
    }
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
        if name == format!("{safe_job}.e")
            || (name.starts_with(&(safe_job.to_string() + ".")) && name.ends_with(".e"))
        {
            ignore.insert(path);
        }
    }
    Ok(ignore)
}

fn find_failed_step_error_snippet(task: &ReferenceIndexTask) -> Result<Option<String>, String> {
    let log_dir = task.workdir.join("log");
    if !log_dir.exists() {
        return Ok(None);
    }
    let mut err_files: Vec<PathBuf> = Vec::new();
    for entry in
        fs::read_dir(&log_dir).map_err(|e| format!("Failed to read {}: {e}", log_dir.display()))?
    {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
            continue;
        };
        if (name == format!("{}.e", task.safe_job)
            || (name.starts_with(&(task.safe_job.clone() + ".")) && name.ends_with(".e")))
            && fs::metadata(&path).map(|m| m.len()).unwrap_or(0) > 0
            && !task.ignore_error_logs.contains(&path)
        {
            err_files.push(path);
        }
    }
    err_files.sort();
    for ef in err_files {
        let text = fs::read_to_string(&ef).unwrap_or_default();
        if text.trim().is_empty() {
            continue;
        }
        let fatal = text
            .lines()
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty() && is_likely_fatal_stderr_line(x))
            .take(4)
            .collect::<Vec<String>>();
        if !fatal.is_empty() {
            return Ok(Some(fatal.join(" | ")));
        }
    }
    Ok(None)
}

fn is_likely_fatal_stderr_line(line: &str) -> bool {
    let s = line.trim().to_ascii_lowercase();
    if s.is_empty() {
        return false;
    }
    if s.contains("no version information available (required by") {
        return false;
    }
    if s.starts_with("job ") && s.contains(" stderr output") {
        return true;
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
        "killed",
        "terminated",
        "cancelled",
        "canceled",
        "sigterm",
        "sigkill",
        "out of memory",
        "oom",
    ];
    keys.iter().any(|k| s.contains(k))
}

fn shell_cmd(cmdline: &str) -> Command {
    #[cfg(windows)]
    {
        let mut cmd = Command::new("cmd");
        cmd.arg("/C").arg(cmdline);
        cmd
    }
    #[cfg(not(windows))]
    {
        let mut cmd = Command::new("bash");
        cmd.arg("-lc").arg(cmdline);
        cmd
    }
}

fn json_string(raw: &str) -> String {
    let mut out = String::new();
    out.push('"');
    for ch in raw.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\u{20}' => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn absolutize_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    let cwd = env::current_dir().map_err(|e| format!("Failed to get current directory: {e}"))?;
    Ok(cwd.join(path))
}

fn find_in_path(bin: &str) -> Option<PathBuf> {
    let candidate = Path::new(bin);
    if candidate.components().count() > 1 {
        if candidate.exists() && candidate.is_file() {
            return Some(candidate.to_path_buf());
        }
        return None;
    }
    let paths = env::var_os("PATH")?;
    for dir in env::split_paths(&paths) {
        let full = dir.join(bin);
        if full.is_file() {
            return Some(full);
        }
        #[cfg(windows)]
        {
            let exe = dir.join(format!("{bin}.exe"));
            if exe.is_file() {
                return Some(exe);
            }
            let bat = dir.join(format!("{bin}.bat"));
            if bat.is_file() {
                return Some(bat);
            }
            let cmd = dir.join(format!("{bin}.cmd"));
            if cmd.is_file() {
                return Some(cmd);
            }
        }
    }
    None
}
