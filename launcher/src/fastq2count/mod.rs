use super::pipeline::{run_pipeline_with_hook, PipelineOptions, PipelineStep, Scheduler, StepItem};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

mod cmd;
use cmd::{
    cmd_fastp, cmd_featurecounts_and_metrics, cmd_hisat2_align_sort_index, cmd_hisat2_index,
    wrap_scheduler_cmd,
};

const FASTQ2COUNT_TOTAL_STEPS: usize = 4;
const FASTQ_SUFFIXES: [&str; 4] = [".fastq.gz", ".fq.gz", ".fastq", ".fq"];

type ToolProbe = (String, bool, bool, String);

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
    annotation: PathBuf,
    input_dir: PathBuf,
    workdir: PathBuf,
    from_step: Option<usize>,
    to_step: Option<usize>,
    strandness: String,
    threads: usize,
    feature_type: String,
    gene_attr: String,
}

pub(crate) fn run_fastq2count_module(args: &[String]) -> Result<i32, String> {
    if args.iter().any(|x| matches!(x.as_str(), "-h" | "--help")) {
        print_fastq2count_help();
        return Ok(0);
    }

    let parsed = parse_fastq2count_args(args)?;
    let (from_step, to_step) = normalize_step_range(parsed.from_step, parsed.to_step)?;
    let (reference, annotation, input_dir, workdir) = validate_and_resolve_inputs(&parsed)?;

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

    println!(
        "{}",
        super::style_green(&format!("Reference path: {}", reference.display()))
    );
    println!(
        "{}",
        super::style_green(&format!("Annotation path: {}", annotation.display()))
    );

    let (backend, backend_reason) = detect_backend();
    println!(
        "{}",
        super::style_green(&format!("Selected backend: {backend}"))
    );
    if !backend_reason.is_empty() {
        println!("{}", backend_reason);
    }
    println!(
        "{}",
        super::style_green(&format!(
            "Pipeline step range: {} -> {} (inclusive)",
            from_step, to_step
        ))
    );

    let (toolchain, use_jx_fastp) = probe_toolchain()?;
    print_toolchain_line(&toolchain);
    let missing = missing_tools_from_probe(&toolchain);
    if !missing.is_empty() {
        return Err(format!("Missing required tools: {}", missing.join(", ")));
    }

    let sample_pairs = resolve_sample_pairs(from_step, &input_dir)?;
    if sample_pairs.is_empty() {
        return Err(format!(
            "Detected 0 paired samples.\n{}",
            recognized_fastq_pairing_hint()
        ));
    }
    println!(
        "{}",
        super::style_green(&format!("Detected {} paired samples.", sample_pairs.len(),))
    );

    let sample_names: Vec<String> = sample_pairs.keys().cloned().collect();
    let metrics_script = ensure_count_metrics_script(&workdir)?;

    let steps = build_fastq2count_steps(
        &reference,
        &annotation,
        &sample_pairs,
        &sample_names,
        &workdir,
        &backend,
        &parsed.strandness,
        parsed.threads,
        &parsed.feature_type,
        &parsed.gene_attr,
        &metrics_script,
        use_jx_fastp,
    )?;

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
    opts.step_index_offset = from_step.saturating_sub(1);
    opts.display_total_steps = Some(FASTQ2COUNT_TOTAL_STEPS);

    let mut noop_hook = |_step: &PipelineStep, _item: &StepItem| Ok(());
    run_pipeline_with_hook(
        &workdir,
        &steps[(from_step - 1)..to_step],
        &opts,
        &mut noop_hook,
    )?;

    if to_step < FASTQ2COUNT_TOTAL_STEPS {
        println!(
            "{}",
            super::style_yellow(&format!("Stopped at step {} by -to-step option.", to_step))
        );
        return Ok(0);
    }

    let out_counts = workdir.join("4.count").join("gene_counts.txt");
    let out_fpkm = workdir.join("4.count").join("gene_counts.fpkm.tsv");
    let out_tpm = workdir.join("4.count").join("gene_counts.tpm.tsv");
    println!("{}", super::style_green("FASTQ2COUNT completed."));
    println!(
        "{}",
        super::style_green(&format!("Count table: {}", out_counts.display()))
    );
    println!(
        "{}",
        super::style_green(&format!("FPKM table: {}", out_fpkm.display()))
    );
    println!(
        "{}",
        super::style_green(&format!("TPM table: {}", out_tpm.display()))
    );

    Ok(0)
}

fn parse_fastq2count_args(args: &[String]) -> Result<ParsedArgs, String> {
    let mut reference: Option<String> = None;
    let mut annotation: Option<String> = None;
    let mut input_dir: Option<String> = None;
    let mut workdir: Option<String> = None;
    let mut from_step: Option<usize> = None;
    let mut to_step: Option<usize> = None;
    let mut strandness: Option<String> = None;
    let mut threads: Option<usize> = None;
    let mut feature_type: Option<String> = None;
    let mut gene_attr: Option<String> = None;
    let mut i = 0usize;

    while i < args.len() {
        let token = args[i].as_str();
        match token {
            "-r" | "--reference" => {
                i += 1;
                if i >= args.len() {
                    return Err("Option `-r/--reference` requires a value.".to_string());
                }
                reference = Some(args[i].clone());
            }
            "-a" | "--annotation" => {
                i += 1;
                if i >= args.len() {
                    return Err("Option `-a/--annotation` requires a value.".to_string());
                }
                annotation = Some(args[i].clone());
            }
            "-i" | "--in" | "--fastq-dir" => {
                i += 1;
                if i >= args.len() {
                    return Err("Option `-i/--in` requires a value.".to_string());
                }
                input_dir = Some(args[i].clone());
            }
            "-w" | "--workdir" => {
                i += 1;
                if i >= args.len() {
                    return Err("Option `-w/--workdir` requires a value.".to_string());
                }
                workdir = Some(args[i].clone());
            }
            "-from-step" | "--from-step" => {
                i += 1;
                if i >= args.len() {
                    return Err("Option `-from-step/--from-step` requires a value.".to_string());
                }
                let raw = args[i].trim();
                let v: usize = raw.parse().map_err(|_| {
                    format!("Invalid value for -from-step/--from-step: {raw} (expect integer)")
                })?;
                from_step = Some(v);
            }
            "-to-step" | "--to-step" => {
                i += 1;
                if i >= args.len() {
                    return Err("Option `-to-step/--to-step` requires a value.".to_string());
                }
                let raw = args[i].trim();
                let v: usize = raw.parse().map_err(|_| {
                    format!("Invalid value for -to-step/--to-step: {raw} (expect integer)")
                })?;
                to_step = Some(v);
            }
            "-strandness" | "--strandness" => {
                i += 1;
                if i >= args.len() {
                    return Err("Option `-strandness/--strandness` requires a value.".to_string());
                }
                strandness = Some(args[i].clone());
            }
            "-t" | "--threads" => {
                i += 1;
                if i >= args.len() {
                    return Err("Option `-t/--threads` requires a value.".to_string());
                }
                let raw = args[i].trim();
                let v: usize = raw.parse().map_err(|_| {
                    format!("Invalid value for -t/--threads: {raw} (expect integer)")
                })?;
                threads = Some(v.max(1));
            }
            "-feature-type" | "--feature-type" => {
                i += 1;
                if i >= args.len() {
                    return Err(
                        "Option `-feature-type/--feature-type` requires a value.".to_string()
                    );
                }
                feature_type = Some(args[i].clone());
            }
            "-gene-attr" | "--gene-attr" => {
                i += 1;
                if i >= args.len() {
                    return Err("Option `-gene-attr/--gene-attr` requires a value.".to_string());
                }
                gene_attr = Some(args[i].clone());
            }
            _ => {
                if let Some(v) = token.strip_prefix("--reference=") {
                    reference = Some(v.to_string());
                } else if let Some(v) = token.strip_prefix("--annotation=") {
                    annotation = Some(v.to_string());
                } else if let Some(v) = token.strip_prefix("--in=") {
                    input_dir = Some(v.to_string());
                } else if let Some(v) = token.strip_prefix("--fastq-dir=") {
                    input_dir = Some(v.to_string());
                } else if let Some(v) = token.strip_prefix("--workdir=") {
                    workdir = Some(v.to_string());
                } else if let Some(v) = token.strip_prefix("--from-step=") {
                    from_step = Some(v.trim().parse().map_err(|_| {
                        format!(
                            "Invalid value for --from-step: {} (expect integer)",
                            v.trim()
                        )
                    })?);
                } else if let Some(v) = token.strip_prefix("--to-step=") {
                    to_step = Some(v.trim().parse().map_err(|_| {
                        format!("Invalid value for --to-step: {} (expect integer)", v.trim())
                    })?);
                } else if let Some(v) = token.strip_prefix("--strandness=") {
                    strandness = Some(v.to_string());
                } else if let Some(v) = token.strip_prefix("--threads=") {
                    let parsed = v.trim().parse::<usize>().map_err(|_| {
                        format!("Invalid value for --threads: {} (expect integer)", v.trim())
                    })?;
                    threads = Some(parsed.max(1));
                } else if let Some(v) = token.strip_prefix("--feature-type=") {
                    feature_type = Some(v.to_string());
                } else if let Some(v) = token.strip_prefix("--gene-attr=") {
                    gene_attr = Some(v.to_string());
                } else {
                    return Err(format!(
                        "Unknown option: {token}\nUse `jx fastq2count -h` for help."
                    ));
                }
            }
        }
        i += 1;
    }

    let (Some(reference), Some(annotation), Some(input_dir), Some(workdir)) =
        (reference, annotation, input_dir, workdir)
    else {
        return Err(
            "`-r/--reference`, `-a/--annotation`, `-i/--in`, and `-w/--workdir` are required."
                .to_string(),
        );
    };

    Ok(ParsedArgs {
        reference: PathBuf::from(reference),
        annotation: PathBuf::from(annotation),
        input_dir: PathBuf::from(input_dir),
        workdir: PathBuf::from(workdir),
        from_step,
        to_step,
        strandness: strandness.unwrap_or_else(|| "FR".to_string()),
        threads: threads.unwrap_or(16),
        feature_type: feature_type.unwrap_or_else(|| "exon".to_string()),
        gene_attr: gene_attr.unwrap_or_else(|| "gene_id".to_string()),
    })
}

fn normalize_step_range(
    from_step: Option<usize>,
    to_step: Option<usize>,
) -> Result<(usize, usize), String> {
    let from = from_step.unwrap_or(1);
    let to = to_step.unwrap_or(FASTQ2COUNT_TOTAL_STEPS);
    if from == 0 || from > FASTQ2COUNT_TOTAL_STEPS {
        return Err(format!(
            "`-from-step/--from-step` must be in [1..{}].",
            FASTQ2COUNT_TOTAL_STEPS
        ));
    }
    if to == 0 || to > FASTQ2COUNT_TOTAL_STEPS {
        return Err(format!(
            "`-to-step/--to-step` must be in [1..{}].",
            FASTQ2COUNT_TOTAL_STEPS
        ));
    }
    if to < from {
        return Err(format!(
            "Invalid step range: from-step ({from}) is greater than to-step ({to})."
        ));
    }
    Ok((from, to))
}

fn validate_and_resolve_inputs(
    args: &ParsedArgs,
) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf), String> {
    let reference = absolutize_path(&args.reference)?;
    if !reference.exists() || !reference.is_file() {
        return Err(format!("Reference file not found: {}", reference.display()));
    }

    let annotation = absolutize_path(&args.annotation)?;
    if !annotation.exists() || !annotation.is_file() {
        return Err(format!(
            "Annotation file not found: {}",
            annotation.display()
        ));
    }

    let input_dir = absolutize_path(&args.input_dir)?;
    if !input_dir.exists() || !input_dir.is_dir() {
        return Err(format!(
            "Input directory not found: {}",
            input_dir.display()
        ));
    }

    let workdir = absolutize_path(&args.workdir)?;
    Ok((reference, annotation, input_dir, workdir))
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

fn probe_toolchain() -> Result<(Vec<ToolProbe>, bool), String> {
    let jx_bin = env::current_exe().map_err(|e| format!("Failed to locate jx binary: {e}"))?;
    let mut out: Vec<ToolProbe> = Vec::new();

    let fastp_via_jx = probe_jx_tool_wrapper(&jx_bin, "fastp");
    let fastp_ok = fastp_via_jx || find_in_path("fastp").is_some();
    out.push((
        "fastp".to_string(),
        fastp_ok,
        true,
        "jx fastp or fastp".to_string(),
    ));

    for tool in [
        "hisat2-build",
        "hisat2",
        "samtools",
        "featureCounts",
        "python3",
    ] {
        out.push((
            tool.to_string(),
            find_in_path(tool).is_some(),
            true,
            tool.to_string(),
        ));
    }

    out.sort_by(|a, b| a.0.cmp(&b.0));
    out.dedup_by(|a, b| a.0 == b.0);
    Ok((out, fastp_via_jx))
}

fn probe_jx_tool_wrapper(jx_bin: &Path, tool: &str) -> bool {
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
        .filter_map(|(_, ok, required, hint)| {
            if *required && !*ok {
                Some(hint.clone())
            } else {
                None
            }
        })
        .collect::<Vec<String>>();
    missing.sort();
    missing.dedup();
    missing
}

fn resolve_sample_pairs(
    from_step: usize,
    input_dir: &Path,
) -> Result<BTreeMap<String, (PathBuf, PathBuf)>, String> {
    if from_step == 4 {
        let samples = infer_samples_from_bam(input_dir)?;
        return Ok(samples);
    }

    let fastq_files = collect_fastq_files(input_dir)?;
    if fastq_files.is_empty() {
        return Err(format!("No FASTQ files found in {}", input_dir.display()));
    }
    classify_fastq_pairs(&fastq_files)
}

fn infer_samples_from_bam(
    mapping_dir: &Path,
) -> Result<BTreeMap<String, (PathBuf, PathBuf)>, String> {
    if !mapping_dir.exists() {
        return Ok(BTreeMap::new());
    }
    let mut out: BTreeMap<String, (PathBuf, PathBuf)> = BTreeMap::new();
    for entry in fs::read_dir(mapping_dir)
        .map_err(|e| format!("Failed to read directory {}: {e}", mapping_dir.display()))?
    {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
            continue;
        };
        if name.ends_with(".bam") && !name.ends_with(".bam.bai") {
            let sample = name.trim_end_matches(".bam").trim();
            if sample.is_empty() {
                continue;
            }
            let p1 = PathBuf::from(format!("{sample}.R1.clean.fastq.gz"));
            let p2 = PathBuf::from(format!("{sample}.R2.clean.fastq.gz"));
            out.insert(sample.to_string(), (p1, p2));
        }
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
    while matches!(kept.last(), Some(x) if x.eq_ignore_ascii_case("clean")) {
        kept.pop();
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
        "Recognized FASTQ suffixes: {suffix_hint}\nRecognized R1/R2 naming tokens: {read_hint}\nExamples: sample_R1.fastq.gz + sample_R2.fastq.gz; sample.1.fq.gz + sample.2.fq.gz"
    )
}

fn build_fastq2count_steps(
    reference: &Path,
    annotation: &Path,
    sample_pairs: &BTreeMap<String, (PathBuf, PathBuf)>,
    sample_names: &[String],
    workdir: &Path,
    backend: &str,
    strandness: &str,
    threads: usize,
    feature_type: &str,
    gene_attr: &str,
    metrics_script: &Path,
    use_jx_fastp: bool,
) -> Result<Vec<PipelineStep>, String> {
    let clean_dir = workdir.join("1.clean");
    let clean_qc_dir = clean_dir.join("qc");
    let index_dir = workdir.join("2.index");
    let mapping_dir = workdir.join("3.mapping");
    let count_dir = workdir.join("4.count");

    for d in [
        &clean_dir,
        &clean_qc_dir,
        &index_dir,
        &mapping_dir,
        &count_dir,
    ] {
        fs::create_dir_all(d).map_err(|e| format!("Failed to create {}: {e}", d.display()))?;
    }

    let mut steps: Vec<PipelineStep> = Vec::new();

    // Step 1: fastp
    let mut step1_cmds = Vec::new();
    let mut step1_inputs = Vec::new();
    let mut step1_outputs = Vec::new();
    let mut step1_items = Vec::new();
    let fastp_prefix = if use_jx_fastp { "jx" } else { "" };
    for sample in sample_names {
        let Some((fq1, fq2)) = sample_pairs.get(sample) else {
            return Err(format!("Sample missing FASTQ pair: {sample}"));
        };
        step1_inputs.push(fq1.clone());
        step1_inputs.push(fq2.clone());
        let out_r1 = clean_dir.join(format!("{sample}.R1.clean.fastq.gz"));
        let out_r2 = clean_dir.join(format!("{sample}.R2.clean.fastq.gz"));
        let out_html = clean_qc_dir.join(format!("{sample}.html"));
        let out_json = clean_qc_dir.join(format!("{sample}.json"));
        let raw = cmd_fastp(
            sample,
            fq1,
            fq2,
            &clean_dir,
            &clean_qc_dir,
            threads.min(16),
            fastp_prefix,
        );
        step1_cmds.push(wrap_scheduler_cmd(
            &raw,
            &format!("fastp.{sample}"),
            threads.min(16),
            backend,
        ));
        let outs = vec![out_r1, out_r2, out_html, out_json];
        step1_outputs.extend(outs.clone());
        step1_items.push(StepItem {
            id: format!("fastp.{sample}"),
            outputs: outs,
        });
    }
    steps.push(PipelineStep {
        id: "step1_fastp".to_string(),
        name: "fastp".to_string(),
        eta_minutes: Some(15),
        commands: step1_cmds,
        inputs: dedup_paths(step1_inputs),
        outputs: dedup_paths(step1_outputs),
        items: step1_items,
    });

    // Step 2: hisat2-build
    let index_done = index_dir.join("reference.index.ok");
    let index_ss = index_dir.join("reference.ss");
    let index_exon = index_dir.join("reference.exon");
    let step2_raw = cmd_hisat2_index(reference, annotation, &index_dir, threads.min(20), "");
    let step2_cmds = vec![wrap_scheduler_cmd(
        &step2_raw,
        "hisat2.index",
        threads.min(20),
        backend,
    )];
    let step2_outputs = vec![index_done.clone(), index_ss, index_exon];
    let step2_items = vec![StepItem {
        id: "hisat2.index".to_string(),
        outputs: step2_outputs.clone(),
    }];
    steps.push(PipelineStep {
        id: "step2_hisat2_index".to_string(),
        name: "hisat2-index".to_string(),
        eta_minutes: Some(30),
        commands: step2_cmds,
        inputs: vec![reference.to_path_buf(), annotation.to_path_buf()],
        outputs: step2_outputs,
        items: step2_items,
    });

    // Step 3: hisat2 + samtools sort/index
    let mut step3_cmds = Vec::new();
    let mut step3_inputs = vec![index_done.clone()];
    let mut step3_outputs = Vec::new();
    let mut step3_items = Vec::new();
    let align_threads = threads.clamp(1, 32);
    let sort_threads = ((threads / 2).max(1)).clamp(1, 16);
    for sample in sample_names {
        let in_r1 = clean_dir.join(format!("{sample}.R1.clean.fastq.gz"));
        let in_r2 = clean_dir.join(format!("{sample}.R2.clean.fastq.gz"));
        step3_inputs.push(in_r1.clone());
        step3_inputs.push(in_r2.clone());
        let out_bam = mapping_dir.join(format!("{sample}.bam"));
        let out_bai = mapping_dir.join(format!("{sample}.bam.bai"));
        let out_log = mapping_dir.join(format!("{sample}.hisat2.log"));
        let raw = cmd_hisat2_align_sort_index(
            sample,
            &index_dir,
            &clean_dir,
            &mapping_dir,
            align_threads,
            sort_threads,
            Some(strandness),
            "",
        );
        step3_cmds.push(wrap_scheduler_cmd(
            &raw,
            &format!("hisat2.{sample}"),
            align_threads,
            backend,
        ));
        let outs = vec![out_bam, out_bai, out_log];
        step3_outputs.extend(outs.clone());
        step3_items.push(StepItem {
            id: format!("hisat2.{sample}"),
            outputs: outs,
        });
    }
    steps.push(PipelineStep {
        id: "step3_hisat2_align".to_string(),
        name: "hisat2-align".to_string(),
        eta_minutes: Some(120),
        commands: step3_cmds,
        inputs: dedup_paths(step3_inputs),
        outputs: dedup_paths(step3_outputs),
        items: step3_items,
    });

    // Step 4: featureCounts + FPKM/TPM
    let out_counts = count_dir.join("gene_counts.txt");
    let out_fpkm = count_dir.join("gene_counts.fpkm.tsv");
    let out_tpm = count_dir.join("gene_counts.tpm.tsv");
    let mut step4_inputs = vec![annotation.to_path_buf(), metrics_script.to_path_buf()];
    for sample in sample_names {
        step4_inputs.push(mapping_dir.join(format!("{sample}.bam")));
    }
    let step4_raw = cmd_featurecounts_and_metrics(
        annotation,
        feature_type,
        gene_attr,
        sample_names,
        &mapping_dir,
        &count_dir,
        metrics_script,
        threads.clamp(1, 32),
        "",
    );
    let step4_cmds = vec![wrap_scheduler_cmd(
        &step4_raw,
        "featurecounts.all",
        threads.clamp(1, 32),
        backend,
    )];
    let step4_outputs = vec![out_counts, out_fpkm, out_tpm];
    let step4_items = vec![StepItem {
        id: "featurecounts.all".to_string(),
        outputs: step4_outputs.clone(),
    }];
    steps.push(PipelineStep {
        id: "step4_featurecounts".to_string(),
        name: "featurecounts".to_string(),
        eta_minutes: Some(25),
        commands: step4_cmds,
        inputs: dedup_paths(step4_inputs),
        outputs: step4_outputs,
        items: step4_items,
    });

    Ok(steps)
}

fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut seen = HashSet::<String>::new();
    for p in paths {
        let key = p.to_string_lossy().to_string();
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        out.push(p);
    }
    out
}

fn ensure_count_metrics_script(workdir: &Path) -> Result<PathBuf, String> {
    let script_path = workdir.join("tmp").join("fastq2count_metrics.py");
    let script = r###"#!/usr/bin/env python3
import csv
import math
import sys
from typing import List


def _f(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: fastq2count_metrics.py <gene_counts.txt> <out.fpkm.tsv> <out.tpm.tsv>", file=sys.stderr)
        return 2

    in_path, out_fpkm, out_tpm = sys.argv[1], sys.argv[2], sys.argv[3]
    rows: List[List[str]] = []
    header = None
    with open(in_path, "r", encoding="utf-8", newline="") as fh:
        for raw in fh:
            if raw.startswith("#"):
                continue
            line = raw.rstrip("\n\r")
            if not line:
                continue
            cols = line.split("\t")
            if header is None:
                header = cols
                continue
            rows.append(cols)

    if header is None or len(header) < 7:
        print("Invalid featureCounts table: expect >=7 columns", file=sys.stderr)
        return 1

    m = len(header)
    sample_idx = list(range(6, m))
    lib = [0.0 for _ in sample_idx]
    lengths: List[float] = []
    counts: List[List[float]] = []

    for r in rows:
        if len(r) < m:
            r.extend(["0"] * (m - len(r)))
        ln = _f(r[5])
        if not math.isfinite(ln) or ln <= 0:
            ln = 1.0
        lengths.append(ln)
        row_counts = []
        for j, idx in enumerate(sample_idx):
            c = _f(r[idx])
            if not math.isfinite(c) or c < 0:
                c = 0.0
            row_counts.append(c)
            lib[j] += c
        counts.append(row_counts)

    rpk_sum = [0.0 for _ in sample_idx]
    for i in range(len(rows)):
        ln = lengths[i]
        for j in range(len(sample_idx)):
            rpk_sum[j] += counts[i][j] / (ln / 1000.0)

    with open(out_fpkm, "w", encoding="utf-8", newline="") as f_fpkm, open(out_tpm, "w", encoding="utf-8", newline="") as f_tpm:
        w_fpkm = csv.writer(f_fpkm, delimiter="\t", lineterminator="\n")
        w_tpm = csv.writer(f_tpm, delimiter="\t", lineterminator="\n")
        w_fpkm.writerow(header)
        w_tpm.writerow(header)

        for i, r in enumerate(rows):
            meta = r[:6]
            ln = lengths[i]
            fpkm_row = []
            tpm_row = []
            for j in range(len(sample_idx)):
                c = counts[i][j]
                fpkm = 0.0
                if lib[j] > 0 and ln > 0:
                    fpkm = c * 1e9 / (ln * lib[j])
                tpm = 0.0
                denom = rpk_sum[j]
                if denom > 0 and ln > 0:
                    tpm = (c / (ln / 1000.0)) * 1e6 / denom
                fpkm_row.append(f"{fpkm:.6f}")
                tpm_row.append(f"{tpm:.6f}")
            w_fpkm.writerow(meta + fpkm_row)
            w_tpm.writerow(meta + tpm_row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"###;
    fs::write(&script_path, script)
        .map_err(|e| format!("Failed to write {}: {e}", script_path.display()))?;
    Ok(script_path)
}

fn print_fastq2count_help() {
    let width = super::help_line_width();
    print!("{} ", super::style_orange("Usage:"));
    print!("{} ", style_gray("jx fastq2count"));
    print!("[");
    print!("{}", super::style_blue("-h"));
    print!("] ");
    print!("{} ", super::style_blue("-r"));
    print!("{} ", super::style_green("REFERENCE"));
    print!("{} ", super::style_blue("-a"));
    print!("{} ", super::style_green("ANNOTATION"));
    print!("{} ", super::style_blue("-i"));
    print!("{} ", super::style_green("INPUT_DIR"));
    print!(" ");
    print!("[");
    print!("{} ", super::style_blue("-from-step"));
    print!("{}", super::style_green("STEP"));
    print!("] ");
    print!("[");
    print!("{} ", super::style_blue("-to-step"));
    print!("{}", super::style_green("STEP"));
    print!("] ");
    print!("{} ", super::style_blue("-w"));
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
        Some(("-a", "--annotation", "ANNOTATION")),
        "Gene annotation GTF/GFF file for index augmentation and featureCounts.",
        42,
        width,
    );
    print_colored_help_entry(
        2,
        Some(("-i", "--in", "INPUT_DIR")),
        "Input directory for current -from-step. This parameter is required.",
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
    print_colored_help_entry(
        2,
        Some(("-from-step", "--from-step", "STEP")),
        "Start from step number (inclusive, 1-4).",
        42,
        width,
    );
    print_colored_help_entry(
        2,
        Some(("-to-step", "--to-step", "STEP")),
        "Stop after step number (inclusive, default: last step).",
        42,
        width,
    );
    print_colored_help_entry(
        2,
        Some(("-strandness", "--strandness", "STRAND")),
        "RNA strandness passed to hisat2 `--rna-strandness` (default: FR). Use `none` to disable.",
        42,
        width,
    );
    print_colored_help_entry(
        2,
        Some(("-t", "--threads", "THREADS")),
        "Worker threads for fastp/hisat2/samtools/featureCounts (default: 16).",
        42,
        width,
    );
    print_colored_help_entry(
        2,
        Some(("-feature-type", "--feature-type", "TYPE")),
        "featureCounts -t value (default: exon).",
        42,
        width,
    );
    print_colored_help_entry(
        2,
        Some(("-gene-attr", "--gene-attr", "ATTR")),
        "featureCounts -g value (default: gene_id).",
        42,
        width,
    );
    println!();

    println!("{}", super::style_orange("Pipeline Steps:"));
    print_fastq2count_step_entry(
        "Step 1",
        "fastp: input paired FASTQ/FASTQ.GZ. Supports .fastq/.fq with optional .gz and common R1/R2 tokens. Outputs 1.clean/{sample}.R1.clean.fastq.gz, 1.clean/{sample}.R2.clean.fastq.gz, and QC reports in 1.clean/qc/.",
        width,
    );
    print_fastq2count_step_entry(
        "Step 2",
        "hisat2-index: build HISAT2 reference index in 2.index/. If splice/exon extraction scripts are available, generates reference.ss and reference.exon then builds index with --ss/--exon; otherwise falls back to plain hisat2-build.",
        width,
    );
    print_fastq2count_step_entry(
        "Step 3",
        "hisat2-align: align cleaned paired reads and stream to samtools sort/index. Outputs 3.mapping/{sample}.bam, .bam.bai, and 3.mapping/{sample}.hisat2.log.",
        width,
    );
    print_fastq2count_step_entry(
        "Step 4",
        "featurecounts: quantify read counts with featureCounts and derive FPKM/TPM. Outputs 4.count/gene_counts.txt, 4.count/gene_counts.fpkm.tsv, and 4.count/gene_counts.tpm.tsv.",
        width,
    );
}

fn print_colored_help_entry(
    indent: usize,
    flags: Option<(&str, &str, &str)>,
    desc: &str,
    key_width: usize,
    total_width: usize,
) {
    let lead = " ".repeat(indent);
    let min_desc_width = 20usize;
    let min_total = indent + key_width + 2 + min_desc_width;

    let key_colored = if let Some((short, long, value)) = flags {
        if value.is_empty() {
            format!("{}, {}", super::style_blue(short), super::style_blue(long))
        } else {
            format!(
                "{}, {} {}",
                super::style_blue(short),
                super::style_blue(long),
                super::style_green(value)
            )
        }
    } else {
        format!(
            "{}, {}",
            super::style_blue("-h"),
            super::style_blue("--help")
        )
    };

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

    let key_len = if let Some((short, long, value)) = flags {
        if value.is_empty() {
            format!("{short}, {long}").chars().count()
        } else {
            format!("{short}, {long} {value}").chars().count()
        }
    } else {
        "-h, --help".chars().count()
    };
    let pad_len = key_width.saturating_sub(key_len);
    let key_with_pad = format!("{key_colored}{}", " ".repeat(pad_len));

    println!("{lead}{key_with_pad}  {}", super::style_white(&wrapped[0]));
    let pad = " ".repeat(key_width);
    for line in wrapped.iter().skip(1) {
        println!("{lead}{pad}  {}", super::style_white(line));
    }
}

fn print_fastq2count_step_entry(step: &str, desc: &str, total_width: usize) {
    let indent = 2usize;
    let key_width = 10usize;
    let lead = " ".repeat(indent);
    let min_desc_width = 20usize;
    let min_total = indent + key_width + 2 + min_desc_width;
    let key_colored = super::style_blue(step);
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
    let pad_len = key_width.saturating_sub(step.chars().count());
    let key_with_pad = format!("{key_colored}{}", " ".repeat(pad_len));
    println!("{lead}{key_with_pad}  {}", super::style_white(&wrapped[0]));
    let pad = " ".repeat(key_width);
    for line in wrapped.iter().skip(1) {
        println!("{lead}{pad}  {}", super::style_white(line));
    }
}

fn style_gray(text: &str) -> String {
    if super::supports_color() {
        format!("\x1b[90m{text}\x1b[0m")
    } else {
        text.to_string()
    }
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
