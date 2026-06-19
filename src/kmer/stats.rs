use crate::kmer::format::SampleEntry;
use crate::kmer::inputs::{inspect_kmc_inputs, resolve_samples};
use crate::kmer::kbin_stats::{
    inspect_kbin_compare, planned_kbin_outputs, scan_kbin_compare, write_kbin_compare_outputs,
};
use crate::kmer::stage1_bucket::{run_stage1, Stage1Config};
use crate::kmer::stage2_stats::{run_stage2_stats, Stage2StatsConfig, VennPatternCount};
use anyhow::{bail, Context, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PairMode {
    Union,
    Intersection,
    Both,
}

impl PairMode {
    fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "union" => Ok(Self::Union),
            "intersection" => Ok(Self::Intersection),
            "both" => Ok(Self::Both),
            other => bail!("unsupported pair mode `{other}`; expected union/intersection/both"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StatsMode {
    Pair(PairMode),
    Venn,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KstatsMode {
    Kmc(StatsMode),
    KbinCompare,
}

#[derive(Debug)]
struct KstatsArgs {
    pub db_inputs: Vec<String>,
    pub sample_ids: Option<Vec<String>>,
    pub kbin_prefix: Option<PathBuf>,
    pub compare_groups: Option<Vec<String>>,
    pub out_dir: PathBuf,
    pub prefix: String,
    pub mode: KstatsMode,
    pub threads: usize,
    pub memory_mb: usize,
    pub tmp_dir: Option<PathBuf>,
    pub max_run_size_mb: usize,
    pub bucket_bits: u8,
    pub batch_size: usize,
    pub min_count: u32,
    pub keep_tmp: bool,
    pub force: bool,
    pub log_path: Option<PathBuf>,
    pub progress_callback: Option<Arc<Py<PyAny>>>,
}

#[derive(Clone, Debug)]
struct KstatsSummary {
    pub pair_intersection_path: Option<PathBuf>,
    pub pair_union_path: Option<PathBuf>,
    pub venn_path: Option<PathBuf>,
    pub compare_groups_path: Option<PathBuf>,
    pub compare_patterns_path: Option<PathBuf>,
    pub compare_pairs_path: Option<PathBuf>,
    pub n_samples: u64,
    pub k: u32,
    pub matrix_bytes: u64,
    pub using_threads: usize,
    pub n_groups: Option<u64>,
    pub backend: String,
    pub scan_mode: Option<String>,
}

#[pyfunction(
    name = "kstats_run",
    signature = (
        db_inputs,
        sample_ids = None,
        kbin = None,
        compare_groups = None,
        out = ".",
        prefix = "kstats",
        pair = None,
        venn = false,
        thread = 8,
        memory = 2048,
        tmp_dir = None,
        max_run_size = 2048,
        bucket_bits = 10,
        batch_size = 65_536,
        min_count = 1,
        keep_tmp = false,
        force = false,
        log_path = None,
        progress_callback = None
    )
)]
pub fn kstats_run_py(
    py: Python<'_>,
    db_inputs: Vec<String>,
    sample_ids: Option<Vec<String>>,
    kbin: Option<String>,
    compare_groups: Option<Vec<String>>,
    out: &str,
    prefix: &str,
    pair: Option<&str>,
    venn: bool,
    thread: usize,
    memory: usize,
    tmp_dir: Option<String>,
    max_run_size: usize,
    bucket_bits: u8,
    batch_size: usize,
    min_count: u32,
    keep_tmp: bool,
    force: bool,
    log_path: Option<String>,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<Py<PyDict>> {
    let mode = parse_cli_mode(kbin.as_deref(), compare_groups.as_ref(), pair, venn)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let args = KstatsArgs {
        db_inputs,
        sample_ids,
        kbin_prefix: kbin.map(PathBuf::from),
        compare_groups,
        out_dir: PathBuf::from(out),
        prefix: prefix.to_string(),
        mode,
        threads: thread.max(1),
        memory_mb: memory.max(1),
        tmp_dir: tmp_dir.map(PathBuf::from),
        max_run_size_mb: max_run_size.max(1),
        bucket_bits,
        batch_size: batch_size.max(1),
        min_count: min_count.max(1),
        keep_tmp,
        force,
        log_path: log_path.map(PathBuf::from),
        progress_callback: progress_callback.map(Arc::new),
    };

    let summary = py
        .detach(|| run_kstats(args))
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    let out = PyDict::new(py);
    out.set_item(
        "pair_intersection",
        summary
            .pair_intersection_path
            .as_ref()
            .map(|x| x.to_string_lossy().into_owned()),
    )?;
    out.set_item(
        "pair_union",
        summary
            .pair_union_path
            .as_ref()
            .map(|x| x.to_string_lossy().into_owned()),
    )?;
    out.set_item(
        "venn",
        summary
            .venn_path
            .as_ref()
            .map(|x| x.to_string_lossy().into_owned()),
    )?;
    out.set_item(
        "compare_groups",
        summary
            .compare_groups_path
            .as_ref()
            .map(|x| x.to_string_lossy().into_owned()),
    )?;
    out.set_item(
        "compare_patterns",
        summary
            .compare_patterns_path
            .as_ref()
            .map(|x| x.to_string_lossy().into_owned()),
    )?;
    out.set_item(
        "compare_pairs",
        summary
            .compare_pairs_path
            .as_ref()
            .map(|x| x.to_string_lossy().into_owned()),
    )?;
    out.set_item("n_samples", summary.n_samples)?;
    out.set_item("k", summary.k)?;
    out.set_item("matrix_bytes", summary.matrix_bytes)?;
    out.set_item("using_threads", summary.using_threads)?;
    out.set_item("n_groups", summary.n_groups)?;
    out.set_item("backend", summary.backend)?;
    out.set_item("scan_mode", summary.scan_mode)?;
    Ok(out.unbind())
}

fn run_kstats(args: KstatsArgs) -> Result<KstatsSummary> {
    if args.prefix.trim().is_empty() {
        bail!("prefix cannot be empty");
    }
    match args.mode {
        KstatsMode::Kmc(mode) => run_kstats_kmc(args, mode),
        KstatsMode::KbinCompare => run_kstats_kbin(args),
    }
}

fn parse_mode(pair: Option<&str>, venn: bool) -> Result<StatsMode> {
    match (pair, venn) {
        (Some(_), true) => bail!("`-pair` and `-venn` are mutually exclusive"),
        (None, false) => bail!("one of `-pair` or `-venn` is required"),
        (Some(raw), false) => Ok(StatsMode::Pair(PairMode::parse(raw)?)),
        (None, true) => Ok(StatsMode::Venn),
    }
}

fn parse_cli_mode(
    kbin: Option<&str>,
    compare_groups: Option<&Vec<String>>,
    pair: Option<&str>,
    venn: bool,
) -> Result<KstatsMode> {
    if kbin.is_some() {
        if pair.is_some() || venn {
            bail!("`-pair`/`-venn` are not used with `-kbin`; use `-compare`");
        }
        let compare_len = compare_groups.map(|x| x.len()).unwrap_or(0);
        if compare_len < 2 {
            bail!("`-kbin` mode requires at least 2 `-compare` groups");
        }
        return Ok(KstatsMode::KbinCompare);
    }
    Ok(KstatsMode::Kmc(parse_mode(pair, venn)?))
}

fn run_kstats_kmc(args: KstatsArgs, mode: StatsMode) -> Result<KstatsSummary> {
    if args.db_inputs.is_empty() {
        bail!("no input KMC database provided");
    }
    if args.batch_size == 0 {
        bail!("batch_size must be > 0");
    }
    if args.bucket_bits == 0 {
        bail!("bucket_bits must be > 0");
    }

    let samples = resolve_samples(&args.db_inputs, args.sample_ids.as_ref())?;
    if samples.len() < 2 {
        bail!("kstats requires at least 2 KMC databases");
    }

    let inspect = inspect_kmc_inputs(&samples)?;
    let out_dir = args.out_dir.clone();
    let tmp_dir = args
        .tmp_dir
        .clone()
        .unwrap_or_else(|| out_dir.join(format!("{}.tmp", args.prefix)));
    let matrix_len = lower_tri_len(samples.len())?;
    let matrix_bytes = u64::try_from(matrix_len)
        .ok()
        .and_then(|x| x.checked_mul(8))
        .ok_or_else(|| anyhow::anyhow!("pairwise matrix byte size overflow"))?;

    let log_path = args.log_path.as_deref();
    prepare_output_space(&out_dir, &tmp_dir, &args.prefix, mode, args.force)?;
    let max_records_per_flush =
        max_records_per_flush(args.memory_mb, args.threads, args.max_run_size_mb)?;

    emit_run_line(
        log_path,
        &format!(
            "Rust kstats: samples={}, k={}, backend=bucket-parallel, threads={}, memory={}MB, bucket_bits={}, batch_size={}, matrix_bytes={}",
            samples.len(),
            inspect.k,
            args.threads,
            args.memory_mb,
            args.bucket_bits,
            args.batch_size,
            matrix_bytes
        ),
    )?;

    let progress_callback = args.progress_callback.as_deref();
    let stage1_progress = build_progress_callback(args.progress_callback.as_ref(), 1);
    let stage2_progress = build_progress_callback(args.progress_callback.as_ref(), 2);
    let stage3_progress = build_progress_callback(args.progress_callback.as_ref(), 3);

    emit_run_line(log_path, "Stage 1/3: KMC stream -> bucketed sorted runs")?;
    emit_progress_callback(
        progress_callback,
        1,
        0,
        inspect.total_records.max(1) as usize,
    )?;
    run_stage1(&Stage1Config {
        tmp_dir: &tmp_dir,
        samples: &samples,
        k: inspect.k,
        min_count: args.min_count,
        batch_size: args.batch_size,
        bucket_bits: args.bucket_bits,
        max_records_per_flush,
        threads: args.threads,
        progress_callback: stage1_progress.clone(),
        progress_total: inspect.total_records.max(1),
    })?;

    emit_run_line(log_path, "Stage 2/3: bucket merge -> pairwise counts")?;
    let stage2_total = 1u64 << args.bucket_bits;
    emit_progress_callback(progress_callback, 2, 0, stage2_total.max(1) as usize)?;
    let stage2 = run_stage2_stats(&Stage2StatsConfig {
        tmp_dir: &tmp_dir,
        n_samples: samples.len(),
        bucket_bits: args.bucket_bits,
        threads: args.threads,
        collect_patterns: matches!(mode, StatsMode::Venn) && samples.len() > 2,
        progress_callback: stage2_progress.clone(),
        progress_total: stage2_total.max(1),
    })?;

    emit_run_line(log_path, "Stage 3/3: write statistics tables")?;
    let stage3_total = output_row_total(mode, samples.len(), stage2.venn_patterns.as_ref());
    emit_progress_callback(progress_callback, 3, 0, stage3_total.max(1) as usize)?;
    let mut stage3_done = 0u64;
    let summary = write_outputs(
        &out_dir,
        &args.prefix,
        mode,
        inspect.k,
        &samples,
        &stage2.pair_counts,
        stage2.venn_patterns.as_deref(),
        stage3_progress.as_ref(),
        &mut stage3_done,
        stage3_total.max(1),
        matrix_bytes,
        args.threads,
    )?;

    if !args.keep_tmp {
        fs::remove_dir_all(&tmp_dir)
            .with_context(|| format!("failed to remove tmp dir: {}", tmp_dir.display()))?;
    }
    Ok(summary)
}

fn run_kstats_kbin(args: KstatsArgs) -> Result<KstatsSummary> {
    let kbin_prefix = args
        .kbin_prefix
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing `-kbin` prefix"))?;
    let compare_groups = args
        .compare_groups
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing `-compare` groups"))?;
    let out_dir = args.out_dir.clone();
    let log_path = args.log_path.as_deref();

    emit_run_line(log_path, "Stage 1/3: load kmerge bitmatrix metadata")?;
    emit_progress_callback(args.progress_callback.as_deref(), 1, 0, 1)?;
    let plan = inspect_kbin_compare(kbin_prefix, compare_groups)?;
    emit_progress_callback(args.progress_callback.as_deref(), 1, 1, 1)?;

    prepare_kbin_output_space(&out_dir, &args.prefix, args.force)?;
    emit_run_line(
        log_path,
        &format!(
            "Bitmatrix files: meta={} | idv={} | bkmer={} | bsite={}",
            plan.meta_path.display(),
            plan.idv_path.display(),
            plan.bkmer_path.display(),
            plan.bsite_path.display()
        ),
    )?;
    emit_run_line(
        log_path,
        &format!(
            "Rust kstats: samples={}, groups={}, k={}, backend=bitmatrix-parallel, threads={}, bytes_per_col={}, scan_mode_hint={}, matrix_bytes={}",
            plan.n_samples,
            plan.groups.len(),
            plan.k,
            args.threads,
            plan.bytes_per_col,
            plan.scan_mode,
            plan.n_kmers.saturating_mul(plan.bytes_per_col as u64)
        ),
    )?;

    let progress_stage2 = build_progress_callback(args.progress_callback.as_ref(), 2);
    let progress_stage3 = build_progress_callback(args.progress_callback.as_ref(), 3);
    emit_run_line(
        log_path,
        "Stage 2/3: parallel .bsite scan -> group patterns",
    )?;
    emit_progress_callback(
        args.progress_callback.as_deref(),
        2,
        0,
        usize::try_from(plan.n_kmers).unwrap_or(usize::MAX),
    )?;
    let scan = scan_kbin_compare(&plan, args.threads, progress_stage2.clone())?;

    emit_run_line(log_path, "Stage 3/3: write comparison tables")?;
    let compare_outputs = write_kbin_compare_outputs(
        &out_dir,
        &args.prefix,
        &plan,
        &scan,
        progress_stage3.clone(),
    )?;

    Ok(KstatsSummary {
        pair_intersection_path: None,
        pair_union_path: None,
        venn_path: None,
        compare_groups_path: Some(compare_outputs.groups_path),
        compare_patterns_path: Some(compare_outputs.patterns_path),
        compare_pairs_path: Some(compare_outputs.pairs_path),
        n_samples: plan.n_samples as u64,
        k: plan.k,
        matrix_bytes: scan.scan_bytes,
        using_threads: args.threads,
        n_groups: Some(plan.groups.len() as u64),
        backend: "bitmatrix-parallel".to_string(),
        scan_mode: Some(scan.scan_mode),
    })
}

fn prepare_output_space(
    out_dir: &Path,
    tmp_dir: &Path,
    prefix: &str,
    mode: StatsMode,
    force: bool,
) -> Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create output dir: {}", out_dir.display()))?;
    for path in planned_outputs(out_dir, prefix, mode) {
        if path.exists() && !force {
            bail!(
                "output file already exists: {} (use --force to overwrite)",
                path.display()
            );
        }
    }
    if tmp_dir.exists() {
        if force {
            fs::remove_dir_all(tmp_dir).with_context(|| {
                format!(
                    "failed to clear tmp dir before rerun: {}",
                    tmp_dir.display()
                )
            })?;
        } else if tmp_dir.read_dir()?.next().is_some() {
            bail!(
                "tmp dir already exists and is not empty: {} (use --force to overwrite)",
                tmp_dir.display()
            );
        }
    }
    fs::create_dir_all(tmp_dir)
        .with_context(|| format!("failed to create tmp dir: {}", tmp_dir.display()))?;
    Ok(())
}

fn prepare_kbin_output_space(out_dir: &Path, prefix: &str, force: bool) -> Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create output dir: {}", out_dir.display()))?;
    for path in planned_kbin_outputs(out_dir, prefix) {
        if path.exists() && !force {
            bail!(
                "output file already exists: {} (use --force to overwrite)",
                path.display()
            );
        }
    }
    Ok(())
}

fn planned_outputs(out_dir: &Path, prefix: &str, mode: StatsMode) -> Vec<PathBuf> {
    match mode {
        StatsMode::Pair(PairMode::Union) => {
            vec![out_dir.join(format!("{prefix}.pair.union.tsv"))]
        }
        StatsMode::Pair(PairMode::Intersection) => {
            vec![out_dir.join(format!("{prefix}.pair.intersection.tsv"))]
        }
        StatsMode::Pair(PairMode::Both) => vec![
            out_dir.join(format!("{prefix}.pair.intersection.tsv")),
            out_dir.join(format!("{prefix}.pair.union.tsv")),
        ],
        StatsMode::Venn => vec![out_dir.join(format!("{prefix}.venn.tsv"))],
    }
}

fn output_row_total(
    mode: StatsMode,
    n_samples: usize,
    venn_patterns: Option<&Vec<VennPatternCount>>,
) -> u64 {
    match mode {
        StatsMode::Pair(PairMode::Union) | StatsMode::Pair(PairMode::Intersection) => {
            n_samples as u64
        }
        StatsMode::Pair(PairMode::Both) => (n_samples as u64).saturating_mul(2),
        StatsMode::Venn => {
            if n_samples <= 2 {
                1
            } else {
                venn_patterns
                    .map(|rows| rows.len().max(1) as u64)
                    .unwrap_or(1)
            }
        }
    }
}

fn write_outputs(
    out_dir: &Path,
    prefix: &str,
    mode: StatsMode,
    k: u32,
    samples: &[SampleEntry],
    pair_counts: &[u64],
    venn_patterns: Option<&[VennPatternCount]>,
    progress_callback: Option<&ProgressFnStage>,
    progress_done: &mut u64,
    progress_total: u64,
    matrix_bytes: u64,
    using_threads: usize,
) -> Result<KstatsSummary> {
    let mut pair_intersection_path = None::<PathBuf>;
    let mut pair_union_path = None::<PathBuf>;
    let mut venn_path = None::<PathBuf>;

    match mode {
        StatsMode::Pair(PairMode::Intersection) => {
            let path = out_dir.join(format!("{prefix}.pair.intersection.tsv"));
            write_pair_matrix(
                &path,
                PairMode::Intersection,
                samples,
                pair_counts,
                progress_callback,
                progress_done,
                progress_total,
            )?;
            pair_intersection_path = Some(path);
        }
        StatsMode::Pair(PairMode::Union) => {
            let path = out_dir.join(format!("{prefix}.pair.union.tsv"));
            write_pair_matrix(
                &path,
                PairMode::Union,
                samples,
                pair_counts,
                progress_callback,
                progress_done,
                progress_total,
            )?;
            pair_union_path = Some(path);
        }
        StatsMode::Pair(PairMode::Both) => {
            let inter_path = out_dir.join(format!("{prefix}.pair.intersection.tsv"));
            write_pair_matrix(
                &inter_path,
                PairMode::Intersection,
                samples,
                pair_counts,
                progress_callback,
                progress_done,
                progress_total,
            )?;
            pair_intersection_path = Some(inter_path);

            let union_path = out_dir.join(format!("{prefix}.pair.union.tsv"));
            write_pair_matrix(
                &union_path,
                PairMode::Union,
                samples,
                pair_counts,
                progress_callback,
                progress_done,
                progress_total,
            )?;
            pair_union_path = Some(union_path);
        }
        StatsMode::Venn => {
            let path = out_dir.join(format!("{prefix}.venn.tsv"));
            if samples.len() <= 2 {
                write_venn_table_pair(
                    &path,
                    samples,
                    pair_counts,
                    progress_callback,
                    progress_done,
                    progress_total,
                )?;
            } else {
                write_venn_table_multi(
                    &path,
                    samples,
                    venn_patterns.unwrap_or(&[]),
                    progress_callback,
                    progress_done,
                    progress_total,
                )?;
            }
            venn_path = Some(path);
        }
    }

    Ok(KstatsSummary {
        pair_intersection_path,
        pair_union_path,
        venn_path,
        compare_groups_path: None,
        compare_patterns_path: None,
        compare_pairs_path: None,
        n_samples: samples.len() as u64,
        k,
        matrix_bytes,
        using_threads,
        n_groups: None,
        backend: "bucket-parallel".to_string(),
        scan_mode: None,
    })
}

fn write_pair_matrix(
    path: &Path,
    mode: PairMode,
    samples: &[SampleEntry],
    pair_counts: &[u64],
    progress_callback: Option<&ProgressFnStage>,
    progress_done: &mut u64,
    progress_total: u64,
) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create pairwise stats file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(b"sample")?;
    for sample in samples {
        writer.write_all(b"\t")?;
        writer.write_all(sample.sample_id.as_bytes())?;
    }
    writer.write_all(b"\n")?;

    for i in 0..samples.len() {
        writer.write_all(samples[i].sample_id.as_bytes())?;
        for j in 0..samples.len() {
            writer.write_all(b"\t")?;
            if j > i {
                writer.write_all(b".")?;
                continue;
            }
            let value = match mode {
                PairMode::Intersection => pair_counts[tri_index(i, j)],
                PairMode::Union => {
                    let inter = pair_counts[tri_index(i, j)];
                    let uniq_i = pair_counts[tri_index(i, i)];
                    let uniq_j = pair_counts[tri_index(j, j)];
                    uniq_i.saturating_add(uniq_j).saturating_sub(inter)
                }
                PairMode::Both => unreachable!("pair both should dispatch per-file"),
            };
            write!(writer, "{value}")?;
        }
        writer.write_all(b"\n")?;
        emit_stage2_tick(progress_callback, progress_done, progress_total, 1)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_venn_table_pair(
    path: &Path,
    samples: &[SampleEntry],
    pair_counts: &[u64],
    progress_callback: Option<&ProgressFnStage>,
    progress_done: &mut u64,
    progress_total: u64,
) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create venn stats file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    let inter = pair_counts[tri_index(1, 0)];
    let unique_a = pair_counts[tri_index(0, 0)];
    let unique_b = pair_counts[tri_index(1, 1)];
    let union = unique_a.saturating_add(unique_b).saturating_sub(inter);
    let only_a = unique_a.saturating_sub(inter);
    let only_b = unique_b.saturating_sub(inter);
    let jaccard = if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    };
    writer.write_all(
        b"sample_a\tsample_b\tunique_a\tunique_b\tintersection\tunion\tonly_a\tonly_b\tboth\tjaccard\n",
    )?;
    writeln!(
        writer,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.10}",
        samples[0].sample_id,
        samples[1].sample_id,
        unique_a,
        unique_b,
        inter,
        union,
        only_a,
        only_b,
        inter,
        jaccard
    )?;
    writer.flush()?;
    emit_stage2_tick(progress_callback, progress_done, progress_total, 1)?;
    Ok(())
}

fn write_venn_table_multi(
    path: &Path,
    samples: &[SampleEntry],
    venn_patterns: &[VennPatternCount],
    progress_callback: Option<&ProgressFnStage>,
    progress_done: &mut u64,
    progress_total: u64,
) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create venn stats file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(b"pattern_id\tcount\tn_present\tpattern_bits\tsample_ids")?;
    for sample in samples {
        writer.write_all(b"\t")?;
        writer.write_all(sample.sample_id.as_bytes())?;
    }
    writer.write_all(b"\n")?;

    for (idx, row) in venn_patterns.iter().enumerate() {
        let pattern_bits = render_pattern_bits(&row.bit_words, samples.len());
        let sample_ids = render_pattern_sample_ids(&row.bit_words, samples);
        let n_present = bit_count(&row.bit_words);
        write!(
            writer,
            "{}\t{}\t{}\t{}\t{}",
            idx + 1,
            row.count,
            n_present,
            pattern_bits,
            sample_ids
        )?;
        for sample_idx in 0..samples.len() {
            let present = if bit_is_set(&row.bit_words, sample_idx) {
                1
            } else {
                0
            };
            write!(writer, "\t{present}")?;
        }
        writer.write_all(b"\n")?;
        emit_stage2_tick(progress_callback, progress_done, progress_total, 1)?;
    }
    writer.flush()?;
    Ok(())
}

type ProgressFnStage = Arc<dyn Fn(u64, u64) -> Result<()> + Send + Sync + 'static>;

fn lower_tri_len(n: usize) -> Result<usize> {
    n.checked_mul(n + 1)
        .and_then(|x| x.checked_div(2))
        .ok_or_else(|| anyhow::anyhow!("pairwise matrix size overflow"))
}

fn tri_index(i: usize, j: usize) -> usize {
    let (hi, lo) = if i >= j { (i, j) } else { (j, i) };
    hi * (hi + 1) / 2 + lo
}

fn bit_count(words: &[u64]) -> u32 {
    words.iter().map(|word| word.count_ones()).sum::<u32>()
}

fn bit_is_set(words: &[u64], idx: usize) -> bool {
    let word_idx = idx / 64;
    let bit_idx = idx % 64;
    words
        .get(word_idx)
        .map(|word| ((word >> bit_idx) & 1) == 1)
        .unwrap_or(false)
}

fn render_pattern_bits(words: &[u64], n_samples: usize) -> String {
    let mut out = String::with_capacity(n_samples);
    for idx in 0..n_samples {
        out.push(if bit_is_set(words, idx) { '1' } else { '0' });
    }
    out
}

fn render_pattern_sample_ids(words: &[u64], samples: &[SampleEntry]) -> String {
    let mut out = Vec::new();
    for (idx, sample) in samples.iter().enumerate() {
        if bit_is_set(words, idx) {
            out.push(sample.sample_id.as_str());
        }
    }
    out.join(",")
}

fn max_records_per_flush(
    memory_mb: usize,
    threads: usize,
    max_run_size_mb: usize,
) -> Result<usize> {
    let memory_bytes = memory_mb
        .checked_mul(1024)
        .and_then(|x| x.checked_mul(1024))
        .ok_or_else(|| anyhow::anyhow!("memory size overflow"))?;
    let max_run_bytes = max_run_size_mb
        .checked_mul(1024)
        .and_then(|x| x.checked_mul(1024))
        .ok_or_else(|| anyhow::anyhow!("max-run-size overflow"))?;
    let per_worker = (memory_bytes / threads.max(1)).max(16 * 1024 * 1024);
    let usable_by_memory = (per_worker / 2).max(16 * 1024);
    let usable_bytes = usable_by_memory.min(max_run_bytes.max(16 * 1024));
    Ok((usable_bytes / 16).max(1))
}

fn emit_stage2_tick(
    callback: Option<&ProgressFnStage>,
    done: &mut u64,
    total: u64,
    step: u64,
) -> Result<()> {
    *done = done.saturating_add(step).min(total.max(1));
    if let Some(cb) = callback {
        cb(*done, total.max(1))?;
    }
    Ok(())
}

fn emit_run_line(log_path: Option<&Path>, line: &str) -> Result<()> {
    println!("{line}");
    if let Some(path) = log_path {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("failed to open kstats log file: {}", path.display()))?;
        writeln!(file, "{line}")
            .with_context(|| format!("failed to write kstats log file: {}", path.display()))?;
    }
    Ok(())
}

fn build_progress_callback(cb: Option<&Arc<Py<PyAny>>>, stage: usize) -> Option<ProgressFnStage> {
    let cb = Arc::clone(cb?);
    Some(Arc::new(move |done: u64, total: u64| -> Result<()> {
        emit_progress_callback(Some(cb.as_ref()), stage, done as usize, total as usize)
    }))
}

fn emit_progress_callback(
    cb: Option<&Py<PyAny>>,
    stage: usize,
    done: usize,
    total: usize,
) -> Result<()> {
    let total_use = total.max(1);
    let done_use = done.min(total_use);
    if let Some(cb) = cb {
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (stage, done_use, total_use))?;
            Ok(())
        })
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::tri_index;

    #[test]
    fn lower_triangle_index_is_stable() {
        assert_eq!(tri_index(0, 0), 0);
        assert_eq!(tri_index(1, 0), 1);
        assert_eq!(tri_index(1, 1), 2);
        assert_eq!(tri_index(2, 0), 3);
        assert_eq!(tri_index(2, 1), 4);
        assert_eq!(tri_index(2, 2), 5);
        assert_eq!(tri_index(0, 2), 3);
        assert_eq!(tri_index(1, 2), 4);
    }
}
