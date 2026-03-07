use super::pipeline::{run_pipeline_with_hook, PipelineOptions, PipelineStep, Scheduler, StepItem};
mod cmd;
mod state;
use cmd::{
    cmd_bam2gvcf, cmd_beagle_impute, cmd_bwamem, cmd_cgvcf, cmd_concat_imputed_vcfs, cmd_fastp,
    cmd_filter_imputed_by_maf_and_missing, cmd_filtersnp, cmd_gvcf2vcf, cmd_markdup,
    cmd_selectfiltersnp, cmd_snpvcf_to_gt_and_missing, cmd_vcf2snpvcf, cmd_vcf2table, sh_quote,
    wrap_scheduler_cmd,
};
use state::WorkStateTracker;
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

const REQUIRED_DLC_TOOLS: [&str; 8] = [
    "fastp", "bwa", "samtools", "gatk", "bcftools", "tabix", "plink", "beagle",
];
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

    let (backend, backend_reason) = detect_backend();
    println!(
        "{}",
        super::style_green(&format!("Selected backend: {backend}"))
    );
    if !backend_reason.is_empty() {
        println!("{}", backend_reason);
    }

    let jx_cmd = "jx";
    let missing_wrappers = missing_dlc_tool_wrappers()?;
    if !missing_wrappers.is_empty() {
        return Err(format!(
            "Missing DLC tool wrappers: {}. Please run `jx -update dlc`.",
            missing_wrappers
                .iter()
                .map(|x| format!("jx {x}"))
                .collect::<Vec<String>>()
                .join(", ")
        ));
    }

    ensure_reference_indexes(&reference, &workdir, &backend, jx_cmd, "indexREF")?;
    let chroms = read_chroms_from_fai(&reference)?;
    let fastq_files = collect_fastq_files(&fastq_dir)?;
    if fastq_files.is_empty() {
        return Err(format!("No FASTQ files found in {}", fastq_dir.display()));
    }
    let samples = classify_fastq_pairs(&fastq_files)?;
    if samples.is_empty() {
        return Err("No valid paired FASTQ files were detected.".to_string());
    }

    println!(
        "{}",
        super::style_green(&format!(
            "Detected {} paired samples from {} FASTQ files.",
            samples.len(),
            fastq_files.len()
        ))
    );

    let metadata_json = build_metadata_json(&reference, &chroms, &samples);
    let metadata_path = workdir.join(".fastq2vcf.metadata.json");
    fs::write(&metadata_path, metadata_json).map_err(|e| {
        format!(
            "Failed to write metadata file {}: {e}",
            metadata_path.display()
        )
    })?;
    let singularity = jx_cmd.to_string();
    let steps = build_fastq2vcf_steps(
        &reference,
        &samples,
        &chroms,
        &workdir,
        &backend,
        &singularity,
    )?;
    let mut opts = PipelineOptions::default();
    opts.scheduler = if backend == "csub" {
        Scheduler::Csub
    } else {
        Scheduler::Nohup
    };
    opts.nohup_max_jobs = 1;
    opts.skip_if_outputs_exist = true;
    opts.poll_sec = 2.0;
    opts.detect_failed_logs = true;

    let state_path = workdir.join(".work.json");
    let run_params_json = build_run_params_json(
        &reference,
        &samples,
        &chroms,
        &backend,
        opts.nohup_max_jobs,
        &singularity,
    );
    let input_files = build_input_files(&reference, &samples);
    let (mut state_tracker, resumed) =
        WorkStateTracker::init_or_resume(&state_path, &run_params_json, &input_files, &steps)?;
    if resumed {
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
        run_pipeline_with_hook(&workdir, &steps, &opts, &mut item_hook)
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
    println!("  jx postbsa -file {tsv_text} -b1 [bulk1] -b2 [bulk2]");
    if bulk_names.len() >= 2 {
        println!(
            "  Example: jx postbsa -file {tsv_text} -b1 {} -b2 {}",
            bulk_names[0], bulk_names[1]
        );
    }

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

fn detect_backend() -> (String, String) {
    let Some(csub_bin) = find_in_path("csub") else {
        return (
            "nohup".to_string(),
            "Backend probe: csub not found in PATH.".to_string(),
        );
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

fn missing_dlc_tool_wrappers() -> Result<Vec<String>, String> {
    let jx_bin = env::current_exe().map_err(|e| format!("Failed to locate jx binary: {e}"))?;
    let mut missing = Vec::new();

    for tool in REQUIRED_DLC_TOOLS {
        let out = Command::new(&jx_bin)
            .arg(tool)
            .arg("-h")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();
        let out = match out {
            Ok(v) => v,
            Err(_) => {
                missing.push(tool.to_string());
                continue;
            }
        };
        if out.status.success() {
            continue;
        }
        let mut merged = String::new();
        merged.push_str(&String::from_utf8_lossy(&out.stdout));
        merged.push('\n');
        merged.push_str(&String::from_utf8_lossy(&out.stderr));
        if wrapper_missing_by_output(&merged) {
            missing.push(tool.to_string());
        }
    }

    missing.sort();
    missing.dedup();
    Ok(missing)
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

fn ensure_reference_indexes(
    reference: &Path,
    workdir: &Path,
    backend: &str,
    jx_prefix: &str,
    job_name: &str,
) -> Result<(), String> {
    let reference = absolutize_path(reference)?;
    if validate_reference_index_integrity(&reference).is_ok() {
        return Ok(());
    }
    let initial_err = validate_reference_index_integrity(&reference)
        .err()
        .unwrap_or_else(|| "unknown index validation error".to_string());
    eprintln!(
        "{}",
        super::style_yellow(&format!(
            "Reference index check failed: {}. Rebuilding indexes...",
            initial_err
        ))
    );

    let fai = reference_sidecar(&reference, ".fai");
    let dict = reference_dict_path(&reference);
    let bwa_files = bwa_index_paths(&reference);
    let mut wait_files = vec![fai, dict];
    wait_files.extend(bwa_files);
    let dict_path = wait_files[1].clone();

    let idx_cmd = format!(
        "{} samtools faidx {} && {} bwa index {} && if [ -f {} ]; then rm -f {}; fi && {} gatk CreateSequenceDictionary -R {} -O {}",
        jx_prefix,
        sh_quote(&reference.to_string_lossy()),
        jx_prefix,
        sh_quote(&reference.to_string_lossy()),
        sh_quote(&dict_path.to_string_lossy()),
        sh_quote(&dict_path.to_string_lossy()),
        jx_prefix,
        sh_quote(&reference.to_string_lossy()),
        sh_quote(&dict_path.to_string_lossy()),
    );
    let wrapped = wrap_scheduler_cmd(&idx_cmd, job_name, 1, backend);
    run_shell_with_spinner(workdir, &wrapped, "Indexing reference")?;
    wait_for_paths(
        &wait_files,
        "Waiting for reference index files",
        Duration::from_secs(2),
    )?;
    validate_reference_index_integrity(&reference)?;
    Ok(())
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

fn read_chroms_from_fai(reference: &Path) -> Result<Vec<String>, String> {
    let reference = absolutize_path(reference)?;
    let fai = reference_sidecar(&reference, ".fai");
    let parsed = parse_fai_file(&fai)?;
    Ok(parsed.order)
}

fn validate_reference_index_integrity(reference: &Path) -> Result<(), String> {
    let fai = reference_sidecar(reference, ".fai");
    let dict = reference_dict_path(reference);

    let fai_data = parse_fai_file(&fai)?;
    validate_bwa_indexes(reference)?;
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

fn validate_bwa_indexes(reference: &Path) -> Result<(), String> {
    for p in bwa_index_paths(reference) {
        check_readable_nonempty(&p)?;
    }
    Ok(())
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

fn build_metadata_json(
    reference: &Path,
    chroms: &[String],
    samples: &BTreeMap<String, (PathBuf, PathBuf)>,
) -> String {
    let mut json = String::new();
    json.push('{');
    json.push_str("\"reference\":");
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
    json.push_str(",\"chrom\":[");
    for (idx, chrom) in chroms.iter().enumerate() {
        if idx > 0 {
            json.push(',');
        }
        json.push_str(&json_string(chrom));
    }
    json.push(']');
    json.push('}');
    json
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

fn build_fastq2vcf_steps(
    reference: &Path,
    samples: &BTreeMap<String, (PathBuf, PathBuf)>,
    chroms: &[String],
    workdir: &Path,
    backend: &str,
    singularity: &str,
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
    let mut steps: Vec<PipelineStep> = Vec::new();

    let mut step1_cmds = Vec::new();
    let mut step1_inputs = Vec::new();
    let mut step1_outputs = Vec::new();
    let mut step1_items = Vec::new();
    for sample in &sample_names {
        let Some((fq1, fq2)) = samples.get(sample) else {
            return Err(format!("Sample missing FASTQ pair in metadata: {sample}"));
        };
        step1_inputs.push(fq1.clone());
        step1_inputs.push(fq2.clone());
        let out_r1 = cleanfolder.join(format!("{sample}.R1.clean.fastq.gz"));
        let out_r2 = cleanfolder.join(format!("{sample}.R2.clean.fastq.gz"));
        let out_html = cleanfolder.join(format!("{sample}.html"));
        let out_json = cleanfolder.join(format!("{sample}.json"));
        let raw = cmd_fastp(sample, fq1, fq2, &cleanfolder, 16, singularity);
        step1_cmds.push(wrap_scheduler_cmd(
            &raw,
            &format!("fastp.{sample}"),
            16,
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
        let raw = cmd_bwamem(reference, sample, &r1, &r2, &mappingfolder, 64, singularity);
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
    for sample in &sample_names {
        let bam = mappingfolder.join(format!("{sample}.sorted.bam"));
        step3_inputs.push(bam.clone());
        let out_md_bam = mappingfolder.join(format!("{sample}.Markdup.bam"));
        let out_md_metrics = mappingfolder.join(format!("{sample}.Markdup.metrics.txt"));
        let raw = cmd_markdup(sample, &bam, &mappingfolder, 16, 200, singularity);
        step3_cmds.push(wrap_scheduler_cmd(
            &raw,
            &format!("markdup.{sample}"),
            16,
            backend,
        ));
        let outs = vec![out_md_bam, out_md_metrics];
        step3_outputs.extend(outs.clone());
        step3_items.push(StepItem {
            id: format!("markdup.{sample}"),
            outputs: outs,
        });
    }
    steps.push(PipelineStep {
        id: "step3_markdup".to_string(),
        name: "markdup".to_string(),
        eta_minutes: Some(73),
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
        let out_g = mergefolder.join(format!("Merge.{chrom}.g.vcf.gz"));
        let out_tbi = mergefolder.join(format!("Merge.{chrom}.g.vcf.gz.tbi"));
        let raw = cmd_cgvcf(reference, chrom, &gvcfs, &mergefolder, singularity);
        step5_cmds.push(wrap_scheduler_cmd(
            &raw,
            &format!("cgvcf.{chrom}"),
            1,
            backend,
        ));
        let outs = vec![out_g, out_tbi];
        step5_outputs.extend(outs.clone());
        step5_items.push(StepItem {
            id: format!("cgvcf.{chrom}"),
            outputs: outs,
        });
    }
    steps.push(PipelineStep {
        id: "step5_cgvcf".to_string(),
        name: "cgvcf".to_string(),
        eta_minutes: Some(36),
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
        let in_g = mergefolder.join(format!("Merge.{chrom}.g.vcf.gz"));
        step6_inputs.push(in_g);
        let out_snp_f = mergefolder.join(format!("Merge.{chrom}.SNP.filtered.vcf.gz"));
        let out_snp_f_tbi = mergefolder.join(format!("Merge.{chrom}.SNP.filtered.vcf.gz.tbi"));
        let out_snp_tsv = mergefolder.join(format!("Merge.{chrom}.SNP.tsv"));
        let cmd6 = [
            cmd_gvcf2vcf(reference, chrom, &mergefolder, 1, 50, singularity),
            cmd_vcf2snpvcf(reference, chrom, &mergefolder, singularity),
            cmd_filtersnp(reference, chrom, &mergefolder, singularity),
            cmd_selectfiltersnp(reference, chrom, &mergefolder, singularity),
            cmd_vcf2table(reference, chrom, &mergefolder, singularity),
        ]
        .join(" && ");
        step6_cmds.push(wrap_scheduler_cmd(
            &cmd6,
            &format!("gvcf2vcf.{chrom}"),
            1,
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
        eta_minutes: Some(32),
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
        eta_minutes: Some(40),
        commands: step7_cmds,
        inputs: dedup_paths(step7_inputs),
        outputs: dedup_paths(step7_outputs),
        items: step7_items,
    });

    let mut step8_cmds = Vec::new();
    let mut step8_inputs = Vec::new();
    let mut step8_outputs = Vec::new();
    let mut step8_items = Vec::new();
    for chrom in chroms {
        let in_gt = imputefolder.join(format!("Merge.{chrom}.SNP.GT.vcf.gz"));
        step8_inputs.push(in_gt);
        let out_imp = imputefolder.join(format!("Merge.{chrom}.SNP.GT.imp.vcf.gz"));
        let out_imp_tbi = imputefolder.join(format!("Merge.{chrom}.SNP.GT.imp.vcf.gz.tbi"));
        let raw = cmd_beagle_impute(chrom, &imputefolder, 2, singularity);
        step8_cmds.push(wrap_scheduler_cmd(
            &raw,
            &format!("beagle.{chrom}"),
            2,
            backend,
        ));
        let outs = vec![out_imp, out_imp_tbi];
        step8_outputs.extend(outs.clone());
        step8_items.push(StepItem {
            id: format!("beagle.{chrom}"),
            outputs: outs,
        });
    }
    steps.push(PipelineStep {
        id: "step8_impute".to_string(),
        name: "impute".to_string(),
        eta_minutes: Some(120),
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
        eta_minutes: Some(24),
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
        eta_minutes: Some(8),
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

fn run_shell_with_spinner(cwd: &Path, cmdline: &str, desc: &str) -> Result<(), String> {
    let mut cmd = shell_cmd(cmdline);
    cmd.current_dir(cwd)
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
    Err(format!(
        "{desc} failed with exit={}.\n{}",
        super::exit_code(out.status),
        msg.trim()
    ))
}

fn wait_for_paths(paths: &[PathBuf], desc: &str, poll: Duration) -> Result<(), String> {
    if paths.iter().all(|p| p.exists()) {
        return Ok(());
    }
    let start = Instant::now();
    let is_tty = io::stdout().is_terminal();
    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let mut i = 0usize;

    while !paths.iter().all(|p| p.exists()) {
        if is_tty {
            let line = format!(
                "\r{} {} [{}]",
                frames[i % frames.len()],
                desc,
                super::format_elapsed(start.elapsed())
            );
            if super::supports_color() {
                print!("{}", super::style_green(&line));
            } else {
                print!("{line}");
            }
            io::stdout()
                .flush()
                .map_err(|e| format!("Failed to flush progress output: {e}"))?;
            i += 1;
        }
        thread::sleep(poll);
    }
    if is_tty {
        print!("\r\x1b[2K");
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush progress output: {e}"))?;
    }
    super::print_success_line(&format!(
        "{desc}[{}]",
        super::format_elapsed(start.elapsed())
    ));
    Ok(())
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
