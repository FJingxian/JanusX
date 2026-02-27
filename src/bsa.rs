use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

const CHR_INTERVAL_RATIO: f64 = 1.0 / 50.0;
const EPS: f64 = 1e-10;

#[derive(Clone, Debug)]
struct Config {
    input: String,
    bulk1: String,
    bulk2: String,
    out_prefix: String,
    min_dp: i64,
    min_gq: i64,
    total_dp_min: i64,
    total_dp_max: i64,
    ref_allele_freq: f64,
    depth_difference: i64,
    window_mb: f64,
    step_mb: f64,
    ed_power: i32,
}

#[derive(Clone, Debug)]
struct ColumnIndex {
    chrom: usize,
    pos: usize,
    bulk1_dp: usize,
    bulk1_ad: usize,
    bulk1_gq: usize,
    bulk2_dp: usize,
    bulk2_ad: usize,
    bulk2_gq: usize,
}

#[derive(Clone, Debug)]
struct RawRecord {
    pos_raw: f64,
    bulk1_snp: f64,
    bulk2_snp: f64,
    delta: f64,
    ed: f64,
    g: f64,
}

#[derive(Clone, Debug)]
struct ChromData {
    chr: String,
    records: Vec<RawRecord>,
}

#[derive(Clone, Debug)]
struct ChromSmoothOutput {
    order_idx: usize,
    lines: Vec<String>,
}

#[derive(Clone, Debug)]
enum ChromSortValue {
    Numeric(f64),
    Text(String),
}

fn run(config: Config) -> Result<(String, String), String> {
    if config.window_mb <= 0.0 || config.step_mb <= 0.0 {
        return Err("window_mb and step_mb must be positive".to_string());
    }

    let input = File::open(&config.input)
        .map_err(|err| format!("failed to open input {}: {err}", config.input))?;
    let mut reader = BufReader::new(input);

    let mut header_line = String::new();
    let read_bytes = reader
        .read_line(&mut header_line)
        .map_err(|err| format!("failed to read header: {err}"))?;
    if read_bytes == 0 {
        return Err("input file is empty".to_string());
    }

    let header = split_tsv_line(&header_line);
    let columns = resolve_columns(&header, &config)?;

    let mut chrom_lookup: HashMap<String, usize> = HashMap::new();
    let mut chrom_data: Vec<ChromData> = Vec::new();

    let mut line = String::new();
    while reader
        .read_line(&mut line)
        .map_err(|err| format!("failed to read input line: {err}"))?
        > 0
    {
        let line_buf = std::mem::take(&mut line);
        let fields = split_tsv_line(&line_buf);

        if !columns_available(&fields, &columns) {
            continue;
        }

        let bulk1_dp = parse_i64(fields[columns.bulk1_dp]).unwrap_or(0);
        let bulk2_dp = parse_i64(fields[columns.bulk2_dp]).unwrap_or(0);
        let bulk1_gq = parse_i64(fields[columns.bulk1_gq]).unwrap_or(0);
        let bulk2_gq = parse_i64(fields[columns.bulk2_gq]).unwrap_or(0);

        if bulk1_dp < config.min_dp || bulk2_dp < config.min_dp {
            continue;
        }
        if bulk1_gq < config.min_gq || bulk2_gq < config.min_gq {
            continue;
        }

        let total_dp = bulk1_dp + bulk2_dp;
        if total_dp < config.total_dp_min || total_dp > config.total_dp_max {
            continue;
        }

        let dp_diff = (bulk1_dp - bulk2_dp).abs();
        if dp_diff > config.depth_difference {
            continue;
        }

        let bulk1_ad = parse_ad(fields[columns.bulk1_ad]);
        let bulk2_ad = parse_ad(fields[columns.bulk2_ad]);
        if bulk1_dp <= 0 || bulk2_dp <= 0 {
            continue;
        }

        let bulk1_snp = bulk1_ad as f64 / bulk1_dp as f64;
        let bulk2_snp = bulk2_ad as f64 / bulk2_dp as f64;
        let low_low = bulk1_snp < config.ref_allele_freq && bulk2_snp < config.ref_allele_freq;
        let high_high =
            bulk1_snp > 1.0 - config.ref_allele_freq && bulk2_snp > 1.0 - config.ref_allele_freq;
        if low_low || high_high {
            continue;
        }

        let chrom = match clean_chr_value(fields[columns.chrom]) {
            Some(value) => value,
            None => continue,
        };
        let pos_raw = match parse_f64(fields[columns.pos]) {
            Some(value) if value.is_finite() => value,
            _ => continue,
        };

        let delta = bulk2_snp - bulk1_snp;
        let ed = (2.0 * delta * delta).sqrt();
        let g = g_statistic(
            (bulk1_dp - bulk1_ad).max(0) as f64,
            bulk1_ad as f64,
            (bulk2_dp - bulk2_ad).max(0) as f64,
            bulk2_ad as f64,
        );

        let record = RawRecord {
            pos_raw,
            bulk1_snp,
            bulk2_snp,
            delta,
            ed,
            g,
        };

        let chr_index = *chrom_lookup.entry(chrom.clone()).or_insert_with(|| {
            chrom_data.push(ChromData {
                chr: chrom.clone(),
                records: Vec::new(),
            });
            chrom_data.len() - 1
        });
        chrom_data[chr_index].records.push(record);
    }

    let mut chrom_order: Vec<usize> = (0..chrom_data.len()).collect();
    chrom_order.sort_by(|left, right| compare_chr(&chrom_data[*left].chr, &chrom_data[*right].chr));

    chrom_data.par_iter_mut().for_each(|chromosome| {
        chromosome.records.par_sort_unstable_by(|left, right| {
            left.pos_raw
                .partial_cmp(&right.pos_raw)
                .unwrap_or(Ordering::Equal)
        });
    });

    let total_loc: f64 = chrom_order
        .iter()
        .filter_map(|idx| chrom_data[*idx].records.last().map(|record| record.pos_raw))
        .sum();
    let chr_interval = (total_loc * CHR_INTERVAL_RATIO).floor();

    let raw_path = PathBuf::from(format!("{}_raw.tsv", config.out_prefix));
    let smooth_path = PathBuf::from(format!("{}_smooth.tsv", config.out_prefix));

    let raw_file = File::create(&raw_path)
        .map_err(|err| format!("failed to create {}: {err}", raw_path.display()))?;
    let smooth_file = File::create(&smooth_path)
        .map_err(|err| format!("failed to create {}: {err}", smooth_path.display()))?;
    let mut raw_writer = BufWriter::new(raw_file);
    let mut smooth_writer = BufWriter::new(smooth_file);

    let delta_col = format!("Delta.SNPindex({}-{})", config.bulk2, config.bulk1);
    writeln!(
        raw_writer,
        "chr\tpos\tpos_raw\t{}.SNPindex\t{}.SNPindex\t{}\tED\tG",
        config.bulk1, config.bulk2, delta_col
    )
    .map_err(|err| format!("failed to write raw header: {err}"))?;
    writeln!(
        smooth_writer,
        "chr\tpos\tpos_raw\t{}.SNPindex\t{}.SNPindex\t{}\tED_power\tGprime",
        config.bulk1, config.bulk2, delta_col
    )
    .map_err(|err| format!("failed to write smooth header: {err}"))?;

    let window_size = config.window_mb * 1_000_000.0;
    let half_window = window_size / 2.0;
    let step_size = config.step_mb * 1_000_000.0;
    let min_window_snps = ((window_size * 1e-4) as usize).max(5);

    let mut offsets_by_order = vec![0.0_f64; chrom_order.len()];
    let mut running_offset = 0.0_f64;
    for (order_idx, idx) in chrom_order.iter().enumerate() {
        offsets_by_order[order_idx] = running_offset;
        if let Some(chr_max) = chrom_data[*idx].records.last().map(|record| record.pos_raw) {
            running_offset += chr_max + chr_interval;
        }
    }

    for (order_idx, idx) in chrom_order.iter().enumerate() {
        let chromosome = &chrom_data[*idx];
        if chromosome.records.is_empty() {
            continue;
        }

        let offset = offsets_by_order[order_idx];
        for record in &chromosome.records {
            writeln!(
                raw_writer,
                "{}\t{:.4}\t{:.4}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                chromosome.chr,
                record.pos_raw + offset,
                record.pos_raw,
                record.bulk1_snp,
                record.bulk2_snp,
                record.delta,
                record.ed,
                record.g,
            )
            .map_err(|err| format!("failed to write raw data: {err}"))?;
        }
    }

    let mut smooth_outputs: Vec<ChromSmoothOutput> = chrom_order
        .par_iter()
        .enumerate()
        .map(|(order_idx, idx)| {
            build_smooth_output(
                order_idx,
                &chrom_data[*idx],
                offsets_by_order[order_idx],
                window_size,
                half_window,
                step_size,
                min_window_snps,
                config.ed_power,
            )
        })
        .collect();
    smooth_outputs.sort_by_key(|output| output.order_idx);
    for output in smooth_outputs {
        for line in output.lines {
            writeln!(smooth_writer, "{line}")
                .map_err(|err| format!("failed to write smooth data: {err}"))?;
        }
    }

    raw_writer
        .flush()
        .map_err(|err| format!("failed to flush raw writer: {err}"))?;
    smooth_writer
        .flush()
        .map_err(|err| format!("failed to flush smooth writer: {err}"))?;

    Ok((
        raw_path.to_string_lossy().to_string(),
        smooth_path.to_string_lossy().to_string(),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    input_path,
    bulk1,
    bulk2,
    out_prefix,
    min_dp,
    min_gq,
    total_dp_min,
    total_dp_max,
    ref_allele_freq,
    depth_difference,
    window_mb,
    step_mb,
    ed_power = 4
))]
pub fn preprocess_bsa(
    py: Python<'_>,
    input_path: String,
    bulk1: String,
    bulk2: String,
    out_prefix: String,
    min_dp: i64,
    min_gq: i64,
    total_dp_min: i64,
    total_dp_max: i64,
    ref_allele_freq: f64,
    depth_difference: i64,
    window_mb: f64,
    step_mb: f64,
    ed_power: i32,
) -> PyResult<(String, String)> {
    let config = Config {
        input: input_path,
        bulk1,
        bulk2,
        out_prefix,
        min_dp,
        min_gq,
        total_dp_min,
        total_dp_max,
        ref_allele_freq,
        depth_difference,
        window_mb,
        step_mb,
        ed_power,
    };

    py.detach(|| run(config)).map_err(PyRuntimeError::new_err)
}

fn build_smooth_output(
    order_idx: usize,
    chromosome: &ChromData,
    offset: f64,
    window_size: f64,
    half_window: f64,
    step_size: f64,
    min_window_snps: usize,
    ed_power: i32,
) -> ChromSmoothOutput {
    if chromosome.records.is_empty() {
        return ChromSmoothOutput {
            order_idx,
            lines: Vec::new(),
        };
    }

    let chr_max = chromosome.records.last().unwrap().pos_raw;
    let positions: Vec<f64> = chromosome
        .records
        .iter()
        .map(|record| record.pos_raw)
        .collect();
    if chr_max - positions[0] < window_size {
        return ChromSmoothOutput {
            order_idx,
            lines: Vec::new(),
        };
    }

    let bulk1_values: Vec<f64> = chromosome
        .records
        .iter()
        .map(|record| record.bulk1_snp)
        .collect();
    let bulk2_values: Vec<f64> = chromosome
        .records
        .iter()
        .map(|record| record.bulk2_snp)
        .collect();
    let delta_values: Vec<f64> = chromosome
        .records
        .iter()
        .map(|record| record.delta)
        .collect();
    let ed_power_values: Vec<f64> = chromosome
        .records
        .iter()
        .map(|record| record.ed.powi(ed_power))
        .collect();
    let g_values: Vec<f64> = chromosome.records.iter().map(|record| record.g).collect();
    let prefix_bulk1 = prefix_sums(&bulk1_values);
    let prefix_bulk2 = prefix_sums(&bulk2_values);
    let prefix_delta = prefix_sums(&delta_values);
    let prefix_ed_power = prefix_sums(&ed_power_values);

    let mut lines = Vec::new();
    let mut center = positions[0] + step_size;
    while center < chr_max {
        let left = lower_bound(&positions, center - half_window);
        let right = upper_bound(&positions, center + half_window);
        let count = right.saturating_sub(left);

        if count >= min_window_snps {
            let bulk1_mean = range_mean(&prefix_bulk1, left, right);
            let bulk2_mean = range_mean(&prefix_bulk2, left, right);
            let delta_mean = range_mean(&prefix_delta, left, right);
            let ed_power_mean = range_mean(&prefix_ed_power, left, right);
            let gprime = tricube_gprime(&positions, &g_values, left, right, center, half_window);
            lines.push(format!(
                "{}\t{:.4}\t{:.4}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                chromosome.chr,
                center + offset,
                center,
                bulk1_mean,
                bulk2_mean,
                delta_mean,
                ed_power_mean,
                gprime,
            ));
        }

        center += step_size;
    }

    ChromSmoothOutput { order_idx, lines }
}

fn split_tsv_line(line: &str) -> Vec<&str> {
    line.trim_end_matches(['\n', '\r']).split('\t').collect()
}

fn resolve_columns(header: &[&str], config: &Config) -> Result<ColumnIndex, String> {
    let mut header_map: HashMap<&str, usize> = HashMap::new();
    for (idx, name) in header.iter().enumerate() {
        header_map.insert(*name, idx);
    }

    let get = |name: String| {
        header_map
            .get(name.as_str())
            .copied()
            .ok_or_else(|| format!("missing column: {name}"))
    };

    Ok(ColumnIndex {
        chrom: header_map
            .get("CHROM")
            .copied()
            .ok_or_else(|| "missing column: CHROM".to_string())?,
        pos: header_map
            .get("POS")
            .copied()
            .ok_or_else(|| "missing column: POS".to_string())?,
        bulk1_dp: get(format!("{}.DP", config.bulk1))?,
        bulk1_ad: get(format!("{}.AD", config.bulk1))?,
        bulk1_gq: get(format!("{}.GQ", config.bulk1))?,
        bulk2_dp: get(format!("{}.DP", config.bulk2))?,
        bulk2_ad: get(format!("{}.AD", config.bulk2))?,
        bulk2_gq: get(format!("{}.GQ", config.bulk2))?,
    })
}

fn columns_available(fields: &[&str], columns: &ColumnIndex) -> bool {
    let max_idx = [
        columns.chrom,
        columns.pos,
        columns.bulk1_dp,
        columns.bulk1_ad,
        columns.bulk1_gq,
        columns.bulk2_dp,
        columns.bulk2_ad,
        columns.bulk2_gq,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);

    fields.len() > max_idx
}

fn parse_i64(value: &str) -> Option<i64> {
    value.trim().parse::<i64>().ok()
}

fn parse_f64(value: &str) -> Option<f64> {
    value.trim().parse::<f64>().ok()
}

fn parse_ad(value: &str) -> i64 {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return 0;
    }
    match trimmed.rsplit(',').next() {
        Some(last) => parse_i64(last).unwrap_or(0),
        None => 0,
    }
}

fn clean_chr_value(value: &str) -> Option<String> {
    let upper = value.trim().to_uppercase();
    let cleaned = upper
        .strip_prefix("CHR")
        .unwrap_or(&upper)
        .trim()
        .to_string();
    if cleaned.is_empty() {
        return None;
    }
    match cleaned.as_str() {
        "NAN" | "NA" | "NULL" => None,
        _ => {
            if let Ok(parsed) = cleaned.parse::<f64>() {
                return Some((parsed as i64).to_string());
            }
            Some(cleaned)
        }
    }
}

fn chrom_sort_value(chr: &str) -> ChromSortValue {
    match chr.parse::<f64>() {
        Ok(num) => ChromSortValue::Numeric(num),
        Err(_) => ChromSortValue::Text(chr.to_string()),
    }
}

fn compare_chr(left: &str, right: &str) -> Ordering {
    match (chrom_sort_value(left), chrom_sort_value(right)) {
        (ChromSortValue::Numeric(a), ChromSortValue::Numeric(b)) => {
            a.partial_cmp(&b).unwrap_or(Ordering::Equal)
        }
        (ChromSortValue::Numeric(_), ChromSortValue::Text(_)) => Ordering::Less,
        (ChromSortValue::Text(_), ChromSortValue::Numeric(_)) => Ordering::Greater,
        (ChromSortValue::Text(a), ChromSortValue::Text(b)) => a.cmp(&b),
    }
}

fn g_statistic(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let observed = [[a, b], [c, d]];
    let total = a + b + c + d;
    if total <= 0.0 {
        return f64::NAN;
    }

    let row_sums = [a + b, c + d];
    let col_sums = [a + c, b + d];
    let mut g_value = 0.0;
    for row in 0..2 {
        for col in 0..2 {
            let obs = observed[row][col].max(EPS);
            let exp = (row_sums[row] * col_sums[col] / (total + EPS)).max(EPS);
            g_value += obs * (obs / exp).ln();
        }
    }
    2.0 * g_value
}

fn prefix_sums(values: &[f64]) -> Vec<f64> {
    let mut prefix = Vec::with_capacity(values.len() + 1);
    prefix.push(0.0);
    let mut running = 0.0;
    for value in values {
        running += *value;
        prefix.push(running);
    }
    prefix
}

fn range_mean(prefix: &[f64], left: usize, right: usize) -> f64 {
    let count = right.saturating_sub(left);
    if count == 0 {
        return f64::NAN;
    }
    (prefix[right] - prefix[left]) / count as f64
}

fn lower_bound(values: &[f64], target: f64) -> usize {
    let mut left = 0;
    let mut right = values.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if values[mid] < target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn upper_bound(values: &[f64], target: f64) -> usize {
    let mut left = 0;
    let mut right = values.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if values[mid] <= target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn tricube_gprime(
    positions: &[f64],
    g_values: &[f64],
    left: usize,
    right: usize,
    center: f64,
    half_window: f64,
) -> f64 {
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for idx in left..right {
        let distance = ((positions[idx] - center).abs()) / half_window;
        if distance > 1.0 {
            continue;
        }
        let weight = (1.0 - distance.powi(3)).powi(3);
        weighted_sum += weight * g_values[idx];
        weight_sum += weight;
    }
    if weight_sum > 0.0 {
        weighted_sum / weight_sum
    } else {
        f64::NAN
    }
}
