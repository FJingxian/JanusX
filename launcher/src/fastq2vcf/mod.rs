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
    let home = super::runtime_home()?;
    let python = super::ensure_runtime(false)?;
    let _ = super::maybe_auto_warmup(&home)?;
    let status = run_python_fastq2vcf(&python, &metadata_path, &workdir, &backend, 1, jx_cmd)?;
    if status != 0 {
        return Ok(status);
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
    let fai = reference_sidecar(&reference, ".fai");
    let ann = reference_sidecar(&reference, ".ann");
    let dict = reference_dict_path(&reference);
    if fai.exists() && ann.exists() && dict.exists() {
        return Ok(());
    }

    let idx_cmd = format!(
        "{} samtools faidx {} && {} bwa index {} && {} gatk CreateSequenceDictionary -R {} -O {}",
        jx_prefix,
        sh_quote(&reference.to_string_lossy()),
        jx_prefix,
        sh_quote(&reference.to_string_lossy()),
        jx_prefix,
        sh_quote(&reference.to_string_lossy()),
        sh_quote(&dict.to_string_lossy()),
    );
    let wrapped = wrap_scheduler_cmd(&idx_cmd, job_name, 1, backend);
    run_shell_with_spinner(workdir, &wrapped, "Indexing reference")?;
    wait_for_paths(
        &[fai, ann, dict],
        "Waiting for reference index files",
        Duration::from_secs(2),
    )?;
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
    let file = File::open(&fai)
        .map_err(|e| format!("Failed to open FASTA index {}: {e}", fai.display()))?;
    let reader = BufReader::new(file);
    let mut chroms = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read {}: {e}", fai.display()))?;
        let mut parts = line.split('\t');
        let Some(chrom) = parts.next() else {
            continue;
        };
        let c = chrom.trim();
        if !c.is_empty() {
            chroms.push(c.to_string());
        }
    }
    if chroms.is_empty() {
        return Err(format!("Empty FASTA index: {}", fai.display()));
    }
    Ok(chroms)
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

fn run_python_fastq2vcf(
    python: &Path,
    metadata_path: &Path,
    workdir: &Path,
    backend: &str,
    nohup_max_jobs: i32,
    singularity: &str,
) -> Result<i32, String> {
    let pycode = r#"
import pathlib
import sys
from janusx.pipeline.fastq2vcf import fastq2vcf

metadata_path = pathlib.Path(sys.argv[1]).resolve()
metadata = metadata_path.read_text(encoding="utf-8")
import json
metadata = json.loads(metadata)
workdir = pathlib.Path(sys.argv[2]).resolve()
backend = str(sys.argv[3])
nohup_max_jobs = int(sys.argv[4])
singularity = str(sys.argv[5])

fastq2vcf(
    metadata=metadata,
    workdir=workdir,
    backbend=backend,
    nohup_max_jobs=nohup_max_jobs,
    singularity=singularity,
)
"#;
    let status = Command::new(python)
        .arg("-c")
        .arg(pycode)
        .arg(metadata_path.to_string_lossy().to_string())
        .arg(workdir.to_string_lossy().to_string())
        .arg(backend)
        .arg(nohup_max_jobs.to_string())
        .arg(singularity)
        .current_dir(workdir)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("Failed to run fastq2vcf pipeline: {e}"))?;
    Ok(super::exit_code(status))
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

fn wrap_scheduler_cmd(cmd: &str, job: &str, threads: usize, backend: &str) -> String {
    if backend == "csub" {
        let safe_job = safe_job_label(job);
        return format!(
            "csub -J {} -o ./log/{}.%J.o -e ./log/{}.%J.e -q c01 -n {} \"{}\"",
            job, safe_job, safe_job, threads, cmd
        );
    }
    let safe_cmd = cmd.replace('"', "\\\"");
    format!(
        "nohup bash -c \"{}\" > ./log/{}.o 2> ./log/{}.e",
        safe_cmd, job, job
    )
}

fn safe_job_label(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "job".to_string()
    } else {
        out
    }
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

fn sh_quote(raw: &str) -> String {
    let escaped = raw.replace('\'', "'\"'\"'");
    format!("'{escaped}'")
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
