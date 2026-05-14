use crate::beam::{beam_search_and_binary_mcc, beam_search_and_continuous_abs_corr, BeamAndResult};
use crate::gfcore::{process_snp_row, BedSnpIter, HmpSnpIter, SiteInfo, TxtSnpIter, VcfSnpIter};
use crate::gfreader::build_sample_selection;
use crate::score::{score_binary_mcc_packed, score_cont_corr_packed};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const BIN01_MAGIC: &[u8; 8] = b"JXBIN001";
const BIN01_HEADER_LEN: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldInputKind {
    Auto,
    Bfile,
    Vcf,
    Hmp,
    File,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldBinMode {
    Bin,
    Mbin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldResponse {
    Binary,
    Continuous,
}

#[derive(Clone)]
struct EncodedRow {
    site: SiteInfo,
    bits: Vec<u8>,
}

enum GarfieldInputReader {
    Bed(BedSnpIter),
    Vcf(VcfSnpIter),
    Hmp(HmpSnpIter),
    Txt(TxtSnpIter),
}

impl GarfieldInputReader {
    fn sample_ids(&self) -> &[String] {
        match self {
            GarfieldInputReader::Bed(it) => &it.samples,
            GarfieldInputReader::Vcf(it) => &it.samples,
            GarfieldInputReader::Hmp(it) => &it.samples,
            GarfieldInputReader::Txt(it) => &it.samples,
        }
    }
}

#[derive(Clone, Debug)]
struct GarfieldWindow {
    chrom: String,
    bp_start: i32,
    bp_end: i32,
    indices: Vec<usize>,
}

#[derive(Clone, Debug)]
struct ConstrainedBeamNode {
    selected: Vec<usize>,
    combined: Vec<u64>,
    score: f64,
    last_index: usize,
}

#[inline]
fn parse_response(response: &str) -> Result<GarfieldResponse, String> {
    let t = response.trim().to_ascii_lowercase();
    match t.as_str() {
        "binary" | "bin" | "b" => Ok(GarfieldResponse::Binary),
        "continuous" | "cont" | "c" => Ok(GarfieldResponse::Continuous),
        _ => Err("response must be 'binary' or 'continuous'".to_string()),
    }
}

#[inline]
fn parse_input_kind(input_kind: &str) -> Result<GarfieldInputKind, String> {
    let t = input_kind.trim().to_ascii_lowercase();
    match t.as_str() {
        "" | "auto" => Ok(GarfieldInputKind::Auto),
        "bfile" | "plink" | "bed" => Ok(GarfieldInputKind::Bfile),
        "vcf" => Ok(GarfieldInputKind::Vcf),
        "hmp" | "hapmap" => Ok(GarfieldInputKind::Hmp),
        "file" | "txt" | "npy" | "bin" => Ok(GarfieldInputKind::File),
        _ => Err("input_kind must be one of: auto, bfile, vcf, hmp, file".to_string()),
    }
}

#[inline]
fn parse_bin_mode(mode: &str) -> Result<GarfieldBinMode, String> {
    let t = mode.trim().to_ascii_lowercase();
    match t.as_str() {
        "bin" | "bin02" => Ok(GarfieldBinMode::Bin),
        "mbin" => Ok(GarfieldBinMode::Mbin),
        _ => Err("mode must be one of: bin, mbin".to_string()),
    }
}

#[inline]
fn words_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(64).max(1)
}

#[inline]
fn tail_mask(n_samples: usize) -> Option<u64> {
    let rem = n_samples & 63;
    if rem == 0 {
        None
    } else {
        Some((1u64 << rem) - 1u64)
    }
}

#[inline]
fn apply_tail_mask(bits: &mut [u64], mask: Option<u64>) {
    if let Some(m) = mask {
        if let Some(last) = bits.last_mut() {
            *last &= m;
        }
    }
}

#[inline]
fn row_prefix<'a>(
    bits_flat: &'a [u64],
    row_words: usize,
    row_idx: usize,
    needed_words: usize,
) -> &'a [u64] {
    let st = row_idx * row_words;
    &bits_flat[st..st + needed_words]
}

#[inline]
fn score_key(s: f64) -> f64 {
    if s.is_nan() {
        f64::NEG_INFINITY
    } else {
        s
    }
}

#[inline]
fn constrained_node_better(a: &ConstrainedBeamNode, b: &ConstrainedBeamNode) -> bool {
    let sa = score_key(a.score);
    let sb = score_key(b.score);
    if sa > sb {
        return true;
    }
    if sa < sb {
        return false;
    }
    if a.selected.len() < b.selected.len() {
        return true;
    }
    if a.selected.len() > b.selected.len() {
        return false;
    }
    a.selected < b.selected
}

#[inline]
fn sort_truncate_nodes(mut nodes: Vec<ConstrainedBeamNode>, k: usize) -> Vec<ConstrainedBeamNode> {
    nodes.sort_by(|a, b| {
        let sa = score_key(a.score);
        let sb = score_key(b.score);
        match sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal) {
            std::cmp::Ordering::Equal => match a.selected.len().cmp(&b.selected.len()) {
                std::cmp::Ordering::Equal => a.selected.cmp(&b.selected),
                other => other,
            },
            other => other,
        }
    });
    if nodes.len() > k {
        nodes.truncate(k);
    }
    nodes
}

#[inline]
fn validate_constrained_inputs(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
    group_ids: &[usize],
    ctx: &str,
) -> Result<usize, String> {
    if n_rows == 0 {
        return Err(format!("{ctx}: n_rows must be > 0"));
    }
    if row_words == 0 {
        return Err(format!("{ctx}: row_words must be > 0"));
    }
    if n_samples == 0 {
        return Err(format!("{ctx}: n_samples must be > 0"));
    }
    if max_pick == 0 {
        return Err(format!("{ctx}: max_pick must be > 0"));
    }
    if beam_width == 0 {
        return Err(format!("{ctx}: beam_width must be > 0"));
    }
    if group_ids.len() != n_rows {
        return Err(format!(
            "{ctx}: group_ids length mismatch: {} vs n_rows={}",
            group_ids.len(),
            n_rows
        ));
    }
    let needed_words = words_for_samples(n_samples);
    if row_words < needed_words {
        return Err(format!(
            "{ctx}: row_words={} smaller than required {}",
            row_words, needed_words
        ));
    }
    let total_words = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: n_rows*row_words overflow"))?;
    if bits_flat.len() < total_words {
        return Err(format!(
            "{ctx}: bits length={} smaller than required {}",
            bits_flat.len(),
            total_words
        ));
    }
    Ok(needed_words)
}

fn beam_search_and_with_group_exclusion<F>(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
    group_ids: &[usize],
    score_fn: F,
) -> Result<BeamAndResult, String>
where
    F: Fn(&[u64]) -> f64 + Sync,
{
    let ctx = "garfield::beam_search_and_with_group_exclusion";
    let needed_words = validate_constrained_inputs(
        bits_flat, row_words, n_rows, n_samples, max_pick, beam_width, group_ids, ctx,
    )?;

    let mask = tail_mask(n_samples);
    let max_depth = max_pick.min(n_rows);
    let layer_cap = beam_width.min(n_rows);

    let mut beam: Vec<ConstrainedBeamNode> = Vec::with_capacity(layer_cap);
    for i in 0..n_rows {
        let mut combined = row_prefix(bits_flat, row_words, i, needed_words).to_vec();
        apply_tail_mask(&mut combined, mask);
        let score = score_fn(&combined);
        beam.push(ConstrainedBeamNode {
            selected: vec![i],
            combined,
            score,
            last_index: i,
        });
    }
    beam = sort_truncate_nodes(beam, layer_cap);
    if beam.is_empty() {
        return Err(format!("{ctx}: no candidates"));
    }

    let mut best = beam[0].clone();
    for _depth in 2..=max_depth {
        let mut next: Vec<ConstrainedBeamNode> = Vec::new();
        for node in beam.iter() {
            for cand in (node.last_index + 1)..n_rows {
                let cg = group_ids[cand];
                if node.selected.iter().any(|&s| group_ids[s] == cg) {
                    continue;
                }
                let row = row_prefix(bits_flat, row_words, cand, needed_words);
                let mut combined = node.combined.clone();
                for (a, &b) in combined.iter_mut().zip(row.iter()) {
                    *a &= b;
                }
                apply_tail_mask(&mut combined, mask);
                let score = score_fn(&combined);
                let mut selected = node.selected.clone();
                selected.push(cand);
                next.push(ConstrainedBeamNode {
                    selected,
                    combined,
                    score,
                    last_index: cand,
                });
            }
        }
        if next.is_empty() {
            break;
        }
        beam = sort_truncate_nodes(next, layer_cap);
        if constrained_node_better(&beam[0], &best) {
            best = beam[0].clone();
        }
    }

    Ok(BeamAndResult {
        selected_indices: best.selected,
        score: best.score,
        combined_bits: best.combined,
    })
}

fn parse_bin01_header(bytes: &[u8], ctx: &str) -> Result<(usize, usize, usize, usize), String> {
    if bytes.len() < BIN01_HEADER_LEN {
        return Err(format!("{ctx}: BIN file too small"));
    }
    if &bytes[0..8] != BIN01_MAGIC {
        return Err(format!("{ctx}: invalid BIN magic (expected JXBIN001)"));
    }

    let n_rows_u64 = u64::from_le_bytes(
        bytes[8..16]
            .try_into()
            .map_err(|_| format!("{ctx}: malformed BIN header n_rows"))?,
    );
    let n_samples_u64 = u64::from_le_bytes(
        bytes[16..24]
            .try_into()
            .map_err(|_| format!("{ctx}: malformed BIN header n_samples"))?,
    );

    let n_rows = usize::try_from(n_rows_u64)
        .map_err(|_| format!("{ctx}: n_rows too large for this platform"))?;
    let n_samples = usize::try_from(n_samples_u64)
        .map_err(|_| format!("{ctx}: n_samples too large for this platform"))?;
    if n_samples == 0 {
        return Err(format!("{ctx}: n_samples is zero"));
    }

    let row_bytes = n_samples.div_ceil(8);
    let data_bytes = n_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| format!("{ctx}: BIN payload size overflow"))?;
    let expected = BIN01_HEADER_LEN
        .checked_add(data_bytes)
        .ok_or_else(|| format!("{ctx}: BIN file size overflow"))?;
    if bytes.len() < expected {
        return Err(format!(
            "{ctx}: BIN payload truncated (have {}, need at least {})",
            bytes.len(),
            expected
        ));
    }

    Ok((n_rows, n_samples, row_bytes, BIN01_HEADER_LEN))
}

fn load_bin01_as_u64_words(path: &str) -> Result<(Vec<u64>, usize, usize, usize), String> {
    let ctx = "garfield::load_bin01_as_u64_words";
    let bytes = fs::read(path).map_err(|e| format!("{ctx}: failed to read {path}: {e}"))?;
    let (n_rows, n_samples, row_bytes, data_offset) = parse_bin01_header(&bytes, ctx)?;

    let row_words = words_for_samples(n_samples);
    let total_words = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: n_rows * row_words overflow"))?;
    let mut bits_flat = vec![0u64; total_words];

    let mask = tail_mask(n_samples);
    for r in 0..n_rows {
        let src_start = data_offset
            .checked_add(
                r.checked_mul(row_bytes)
                    .ok_or_else(|| format!("{ctx}: row byte offset overflow"))?,
            )
            .ok_or_else(|| format!("{ctx}: row byte offset overflow"))?;
        let src_end = src_start
            .checked_add(row_bytes)
            .ok_or_else(|| format!("{ctx}: row byte end overflow"))?;
        let src = &bytes[src_start..src_end];

        let dst_start = r
            .checked_mul(row_words)
            .ok_or_else(|| format!("{ctx}: row word offset overflow"))?;
        let dst = &mut bits_flat[dst_start..dst_start + row_words];

        for w in 0..row_words {
            let b0 = w * 8;
            if b0 >= row_bytes {
                break;
            }
            let b1 = (b0 + 8).min(row_bytes);
            let mut buf = [0u8; 8];
            let n = b1 - b0;
            buf[..n].copy_from_slice(&src[b0..b1]);
            dst[w] = u64::from_le_bytes(buf);
        }
        apply_tail_mask(dst, mask);
    }
    Ok((bits_flat, row_words, n_rows, n_samples))
}

fn load_bin01_selected_rows_as_u64_words(
    path: &str,
    row_indices: &[usize],
) -> Result<(Vec<u64>, usize, usize, usize), String> {
    let ctx = "garfield::load_bin01_selected_rows_as_u64_words";
    if row_indices.is_empty() {
        return Err(format!("{ctx}: row_indices is empty"));
    }
    let bytes = fs::read(path).map_err(|e| format!("{ctx}: failed to read {path}: {e}"))?;
    let (n_rows_all, n_samples, row_bytes, data_offset) = parse_bin01_header(&bytes, ctx)?;

    let row_words = words_for_samples(n_samples);
    let n_rows = row_indices.len();
    let total_words = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: n_rows * row_words overflow"))?;
    let mut bits_flat = vec![0u64; total_words];

    let mask = tail_mask(n_samples);
    for (ri, &src_row_idx) in row_indices.iter().enumerate() {
        if src_row_idx >= n_rows_all {
            return Err(format!(
                "{ctx}: row index out of range: {} (n_rows={})",
                src_row_idx, n_rows_all
            ));
        }
        let src_start = data_offset
            .checked_add(
                src_row_idx
                    .checked_mul(row_bytes)
                    .ok_or_else(|| format!("{ctx}: row byte offset overflow"))?,
            )
            .ok_or_else(|| format!("{ctx}: row byte offset overflow"))?;
        let src_end = src_start
            .checked_add(row_bytes)
            .ok_or_else(|| format!("{ctx}: row byte end overflow"))?;
        let src = &bytes[src_start..src_end];

        let dst_start = ri
            .checked_mul(row_words)
            .ok_or_else(|| format!("{ctx}: row word offset overflow"))?;
        let dst = &mut bits_flat[dst_start..dst_start + row_words];

        for w in 0..row_words {
            let b0 = w * 8;
            if b0 >= row_bytes {
                break;
            }
            let b1 = (b0 + 8).min(row_bytes);
            let mut buf = [0u8; 8];
            let n = b1 - b0;
            buf[..n].copy_from_slice(&src[b0..b1]);
            dst[w] = u64::from_le_bytes(buf);
        }
        apply_tail_mask(dst, mask);
    }
    Ok((bits_flat, row_words, n_rows, n_samples))
}

#[inline]
fn append_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut os: OsString = prefix.as_os_str().to_os_string();
    os.push(suffix);
    PathBuf::from(os)
}

fn bin_prefix(path: &str) -> PathBuf {
    let p = Path::new(path);
    let mut out = p.to_path_buf();
    let is_bin = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("bin"))
        .unwrap_or(false);
    if is_bin {
        out.set_extension("");
    }
    out
}

fn discover_site_sidecar(path: &str) -> Option<PathBuf> {
    let prefix = bin_prefix(path);
    let candidates = [
        append_suffix(&prefix, ".bsite"),
        append_suffix(&prefix, ".site"),
        append_suffix(&prefix, ".site.tsv"),
        append_suffix(&prefix, ".site.txt"),
        append_suffix(&prefix, ".site.csv"),
        append_suffix(&prefix, ".sites.tsv"),
        append_suffix(&prefix, ".sites.txt"),
        append_suffix(&prefix, ".sites.csv"),
        append_suffix(&prefix, ".bin.site"),
        append_suffix(&prefix, ".bim"),
    ];
    for cand in candidates {
        if cand.exists() {
            return Some(cand);
        }
    }
    None
}

fn discover_id_sidecar(path: &str) -> Option<PathBuf> {
    let prefix = bin_prefix(path);
    let candidates = [
        append_suffix(&prefix, ".bin.id"),
        append_suffix(&prefix, ".id"),
        append_suffix(&prefix, ".fam"),
    ];
    for cand in candidates {
        if cand.exists() {
            return Some(cand);
        }
    }
    None
}

fn copy_site_sidecar(src_bin: &str, dst_bin: &str) -> Result<(), String> {
    let Some(src) = discover_site_sidecar(src_bin) else {
        return Ok(());
    };
    let src_name = src
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| "invalid sidecar filename".to_string())?;
    let src_prefix = bin_prefix(src_bin);
    let dst_prefix = bin_prefix(dst_bin);
    let src_prefix_name = src_prefix
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| "invalid src prefix".to_string())?;

    let suffix = src_name
        .strip_prefix(src_prefix_name)
        .unwrap_or("")
        .to_string();
    if suffix.is_empty() {
        return Ok(());
    }
    let dst = append_suffix(&dst_prefix, &suffix);
    fs::copy(&src, &dst)
        .map_err(|e| format!("copy {} -> {}: {e}", src.display(), dst.display()))?;
    Ok(())
}

fn copy_id_sidecar(src_bin: &str, dst_bin: &str) -> Result<(), String> {
    let Some(src) = discover_id_sidecar(src_bin) else {
        return Ok(());
    };
    let src_name = src
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| "invalid id-sidecar filename".to_string())?;
    let src_prefix = bin_prefix(src_bin);
    let dst_prefix = bin_prefix(dst_bin);
    let src_prefix_name = src_prefix
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| "invalid src prefix".to_string())?;

    let suffix = src_name
        .strip_prefix(src_prefix_name)
        .unwrap_or("")
        .to_string();
    if suffix.is_empty() {
        return Ok(());
    }
    let dst = append_suffix(&dst_prefix, &suffix);
    fs::copy(&src, &dst)
        .map_err(|e| format!("copy {} -> {}: {e}", src.display(), dst.display()))?;
    Ok(())
}

fn read_sample_ids_from_sidecar(path: &Path) -> Result<Vec<String>, String> {
    let fr = BufReader::new(File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?);
    let is_fam = path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| x.eq_ignore_ascii_case("fam"))
        .unwrap_or(false);
    let mut out: Vec<String> = Vec::new();
    for (ln, line) in fr.lines().enumerate() {
        let raw = line.map_err(|e| format!("read {}:{}: {e}", path.display(), ln + 1))?;
        let s = raw.trim();
        if s.is_empty() {
            continue;
        }
        if is_fam {
            let toks: Vec<&str> = s.split_whitespace().collect();
            if toks.len() < 2 {
                return Err(format!(
                    "{}:{} malformed FAM row (need >=2 columns)",
                    path.display(),
                    ln + 1
                ));
            }
            out.push(toks[1].to_string());
        } else {
            let toks: Vec<&str> = s.split_whitespace().collect();
            if toks.is_empty() {
                continue;
            }
            out.push(toks[0].to_string());
        }
    }
    Ok(out)
}

fn write_subset_id_sidecar(
    src_bin: &str,
    dst_bin: &str,
    sample_indices: &[usize],
) -> Result<bool, String> {
    let Some(src_id_path) = discover_id_sidecar(src_bin) else {
        return Ok(false);
    };
    let src_ids = read_sample_ids_from_sidecar(&src_id_path)?;
    let dst_id_path = out_bin_id_path(dst_bin);
    if let Some(parent) = dst_id_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let mut fw = BufWriter::new(
        File::create(&dst_id_path).map_err(|e| format!("create {}: {e}", dst_id_path.display()))?,
    );
    for &si in sample_indices.iter() {
        if si >= src_ids.len() {
            return Err(format!(
                "sample index out of range while subsetting IDs: {} >= {}",
                si,
                src_ids.len()
            ));
        }
        fw.write_all(src_ids[si].as_bytes())
            .map_err(|e| format!("write {}: {e}", dst_id_path.display()))?;
        fw.write_all(b"\n")
            .map_err(|e| format!("write {}: {e}", dst_id_path.display()))?;
    }
    fw.flush()
        .map_err(|e| format!("flush {}: {e}", dst_id_path.display()))?;
    Ok(true)
}

#[inline]
fn normalize_plink_prefix(path_or_prefix: &str) -> String {
    let s = path_or_prefix.trim();
    let low = s.to_ascii_lowercase();
    if low.ends_with(".bed") || low.ends_with(".bim") || low.ends_with(".fam") {
        s[..s.len() - 4].to_string()
    } else {
        s.to_string()
    }
}

fn make_input_reader(
    input_path: &str,
    input_kind: GarfieldInputKind,
) -> Result<GarfieldInputReader, String> {
    let p = input_path.trim();
    if p.is_empty() {
        return Err("input_path must not be empty".to_string());
    }
    let low = p.to_ascii_lowercase();
    match input_kind {
        GarfieldInputKind::Bfile => {
            let prefix = normalize_plink_prefix(p);
            let it = BedSnpIter::new_with_fill(&prefix, 0.0, 1.0, false, false, 0.02)?;
            Ok(GarfieldInputReader::Bed(it))
        }
        GarfieldInputKind::Vcf => {
            let it = VcfSnpIter::new_with_fill(p, 0.0, 1.0, false, false, 0.02)?;
            Ok(GarfieldInputReader::Vcf(it))
        }
        GarfieldInputKind::Hmp => {
            let it = HmpSnpIter::new_with_fill(p, 0.0, 1.0, false, false, 0.02)?;
            Ok(GarfieldInputReader::Hmp(it))
        }
        GarfieldInputKind::File => {
            let it = TxtSnpIter::new(p, None)?;
            Ok(GarfieldInputReader::Txt(it))
        }
        GarfieldInputKind::Auto => {
            if low.ends_with(".vcf") || low.ends_with(".vcf.gz") {
                let it = VcfSnpIter::new_with_fill(p, 0.0, 1.0, false, false, 0.02)?;
                return Ok(GarfieldInputReader::Vcf(it));
            }
            if low.ends_with(".hmp") || low.ends_with(".hmp.gz") {
                let it = HmpSnpIter::new_with_fill(p, 0.0, 1.0, false, false, 0.02)?;
                return Ok(GarfieldInputReader::Hmp(it));
            }
            let prefix = normalize_plink_prefix(p);
            let bed = format!("{prefix}.bed");
            let bim = format!("{prefix}.bim");
            let fam = format!("{prefix}.fam");
            if Path::new(&bed).exists() && Path::new(&bim).exists() && Path::new(&fam).exists() {
                let it = BedSnpIter::new_with_fill(&prefix, 0.0, 1.0, false, false, 0.02)?;
                return Ok(GarfieldInputReader::Bed(it));
            }
            let it = TxtSnpIter::new(p, None)?;
            Ok(GarfieldInputReader::Txt(it))
        }
    }
}

#[inline]
fn row_bytes_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(8).max(1)
}

#[inline]
fn normalize_genotype3(v: f32) -> Option<u8> {
    if !v.is_finite() || v < 0.0 {
        return None;
    }
    let r = v.round();
    if r <= 0.0 {
        Some(0)
    } else if r >= 2.0 {
        Some(2)
    } else {
        Some(1)
    }
}

fn write_bin01_header(file: &mut File, n_rows: u64, n_samples: usize) -> Result<(), String> {
    file.seek(SeekFrom::Start(0))
        .map_err(|e| format!("seek BIN header: {e}"))?;
    file.write_all(BIN01_MAGIC)
        .map_err(|e| format!("write BIN magic: {e}"))?;
    file.write_all(&n_rows.to_le_bytes())
        .map_err(|e| format!("write BIN n_rows: {e}"))?;
    file.write_all(&(n_samples as u64).to_le_bytes())
        .map_err(|e| format!("write BIN n_samples: {e}"))?;
    file.write_all(&(0u64).to_le_bytes())
        .map_err(|e| format!("write BIN reserved: {e}"))?;
    Ok(())
}

#[inline]
fn bin_output_prefix(out_bin_path: &str) -> PathBuf {
    let p = Path::new(out_bin_path);
    let mut out = p.to_path_buf();
    let is_bin = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("bin"))
        .unwrap_or(false);
    if is_bin {
        out.set_extension("");
    }
    out
}

#[inline]
fn out_bin_id_path(out_bin_path: &str) -> PathBuf {
    append_suffix(&bin_output_prefix(out_bin_path), ".bin.id")
}

#[inline]
fn out_bin_site_path(out_bin_path: &str) -> PathBuf {
    append_suffix(&bin_output_prefix(out_bin_path), ".bin.site")
}

fn write_sample_id_sidecar(out_bin_path: &str, sample_ids: &[String]) -> Result<(), String> {
    let id_path = out_bin_id_path(out_bin_path);
    if let Some(parent) = id_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create parent dir {}: {e}", parent.display()))?;
    }
    let mut fw = BufWriter::new(
        File::create(&id_path).map_err(|e| format!("create {}: {e}", id_path.display()))?,
    );
    for sid in sample_ids.iter() {
        fw.write_all(sid.as_bytes())
            .map_err(|e| format!("write {}: {e}", id_path.display()))?;
        fw.write_all(b"\n")
            .map_err(|e| format!("write {}: {e}", id_path.display()))?;
    }
    fw.flush()
        .map_err(|e| format!("flush {}: {e}", id_path.display()))?;
    Ok(())
}

fn row_select_by_indices(row: Vec<f32>, sample_indices: &[usize]) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(sample_indices.len());
    for &idx in sample_indices.iter() {
        if idx >= row.len() {
            return Err(format!(
                "sample index out of range while slicing row: {} >= {}",
                idx,
                row.len()
            ));
        }
        out.push(row[idx]);
    }
    Ok(out)
}

fn mbin_site_variants(site: &SiteInfo) -> [SiteInfo; 3] {
    let dom = SiteInfo {
        chrom: site.chrom.clone(),
        pos: site.pos,
        ref_allele: site.ref_allele.clone(),
        alt_allele: format!("{}|DOM", site.alt_allele),
    };
    let rec = SiteInfo {
        chrom: site.chrom.clone(),
        pos: site.pos,
        ref_allele: site.ref_allele.clone(),
        alt_allele: format!("{}|REC", site.alt_allele),
    };
    let het = SiteInfo {
        chrom: site.chrom.clone(),
        pos: site.pos,
        ref_allele: site.ref_allele.clone(),
        alt_allele: format!("{}|HET", site.alt_allele),
    };
    [dom, rec, het]
}

fn encode_row_to_bits(
    mut row: Vec<f32>,
    mut site: SiteInfo,
    mode: GarfieldBinMode,
    row_bytes: usize,
    maf: f32,
    geno: f32,
    impute: bool,
    het: f32,
) -> Option<Vec<EncodedRow>> {
    let keep = process_snp_row(
        &mut row,
        &mut site.ref_allele,
        &mut site.alt_allele,
        maf,
        geno,
        impute,
        false,
        het,
    );
    if !keep {
        return None;
    }
    if matches!(mode, GarfieldBinMode::Bin) {
        let mut non_missing = 0usize;
        let mut het_count = 0usize;
        for &v in row.iter() {
            if let Some(g) = normalize_genotype3(v) {
                non_missing += 1;
                if g == 1 {
                    het_count += 1;
                }
            }
        }
        if non_missing == 0 {
            return None;
        }
        let het_rate = het_count as f32 / non_missing as f32;
        if het_rate > het {
            return None;
        }
    }

    match mode {
        GarfieldBinMode::Bin => {
            let mut c0 = 0usize;
            let mut c2 = 0usize;
            for &v in row.iter() {
                match normalize_genotype3(v) {
                    Some(0) => c0 += 1,
                    Some(2) => c2 += 1,
                    _ => {}
                }
            }
            let mode02 = if c2 > c0 { 2u8 } else { 0u8 };
            let mut bits = vec![0u8; row_bytes];
            for (i, &v) in row.iter().enumerate() {
                let g = normalize_genotype3(v).unwrap_or(mode02);
                let is_one = if g == 0 {
                    false
                } else if g == 2 {
                    true
                } else {
                    mode02 == 2
                };
                if is_one {
                    bits[i >> 3] |= 1u8 << (i & 7);
                }
            }
            site.alt_allele = format!("{}|BIN", site.alt_allele);
            Some(vec![EncodedRow { site, bits }])
        }
        GarfieldBinMode::Mbin => {
            let mut dom = vec![0u8; row_bytes];
            let mut rec = vec![0u8; row_bytes];
            let mut het_bits = vec![0u8; row_bytes];
            for (i, &v) in row.iter().enumerate() {
                let Some(g) = normalize_genotype3(v) else {
                    continue;
                };
                let mask = 1u8 << (i & 7);
                if g > 0 {
                    dom[i >> 3] |= mask;
                }
                if g == 2 {
                    rec[i >> 3] |= mask;
                }
                if g == 1 {
                    het_bits[i >> 3] |= mask;
                }
            }
            let [s_dom, s_rec, s_het] = mbin_site_variants(&site);
            Some(vec![
                EncodedRow {
                    site: s_dom,
                    bits: dom,
                },
                EncodedRow {
                    site: s_rec,
                    bits: rec,
                },
                EncodedRow {
                    site: s_het,
                    bits: het_bits,
                },
            ])
        }
    }
}

fn encode_batch_rows(
    batch: Vec<(Vec<f32>, SiteInfo)>,
    mode: GarfieldBinMode,
    row_bytes: usize,
    maf: f32,
    geno: f32,
    impute: bool,
    het: f32,
    pool: Option<&rayon::ThreadPool>,
) -> Vec<Option<Vec<EncodedRow>>> {
    if let Some(p) = pool {
        p.install(|| {
            batch
                .into_par_iter()
                .map(|(row, site)| {
                    encode_row_to_bits(row, site, mode, row_bytes, maf, geno, impute, het)
                })
                .collect::<Vec<_>>()
        })
    } else {
        batch
            .into_iter()
            .map(|(row, site)| {
                encode_row_to_bits(row, site, mode, row_bytes, maf, geno, impute, het)
            })
            .collect::<Vec<_>>()
    }
}

struct GarfieldBinWriter {
    file: File,
    site_fw: BufWriter<File>,
    row_bytes: usize,
    n_rows_written: usize,
}

impl GarfieldBinWriter {
    fn new(out_bin_path: &str, n_samples: usize) -> Result<Self, String> {
        let out_path = Path::new(out_bin_path);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create parent dir {}: {e}", parent.display()))?;
        }
        let mut file =
            File::create(out_path).map_err(|e| format!("create {}: {e}", out_path.display()))?;
        write_bin01_header(&mut file, 0, n_samples)?;

        let site_path = out_bin_site_path(out_bin_path);
        if let Some(parent) = site_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create parent dir {}: {e}", parent.display()))?;
        }
        let site_fw = BufWriter::new(
            File::create(&site_path).map_err(|e| format!("create {}: {e}", site_path.display()))?,
        );
        Ok(Self {
            file,
            site_fw,
            row_bytes: row_bytes_for_samples(n_samples),
            n_rows_written: 0,
        })
    }

    fn write_rows(&mut self, rows: &[EncodedRow]) -> Result<(), String> {
        for row in rows.iter() {
            if row.bits.len() != self.row_bytes {
                return Err(format!(
                    "encoded row byte mismatch: got {}, expected {}",
                    row.bits.len(),
                    self.row_bytes
                ));
            }
            self.file
                .write_all(&row.bits)
                .map_err(|e| format!("write BIN row: {e}"))?;
            writeln!(
                self.site_fw,
                "{}\t{}\t{}\t{}",
                row.site.chrom, row.site.pos, row.site.ref_allele, row.site.alt_allele
            )
            .map_err(|e| format!("write BIN site row: {e}"))?;
            self.n_rows_written = self.n_rows_written.saturating_add(1);
        }
        Ok(())
    }

    fn finish(mut self, n_samples: usize) -> Result<usize, String> {
        self.site_fw
            .flush()
            .map_err(|e| format!("flush BIN site sidecar: {e}"))?;
        self.file
            .flush()
            .map_err(|e| format!("flush BIN file: {e}"))?;
        let n_rows_u64 =
            u64::try_from(self.n_rows_written).map_err(|_| "n_rows overflows u64".to_string())?;
        write_bin01_header(&mut self.file, n_rows_u64, n_samples)?;
        self.file
            .flush()
            .map_err(|e| format!("flush BIN header rewrite: {e}"))?;
        Ok(self.n_rows_written)
    }
}

#[inline]
fn normalize_chrom(chrom: &str) -> String {
    let s = chrom.trim();
    if s.len() > 2 && (s.ends_with("_1") || s.ends_with("_2")) {
        return s[..s.len() - 2].to_string();
    }
    if s.len() > 1 && (s.ends_with('-') || s.ends_with('+')) {
        return s[..s.len() - 1].to_string();
    }
    s.to_string()
}

fn build_chrom_index(sites: &[SiteInfo], n_rows: usize) -> HashMap<String, Vec<(i32, usize)>> {
    let mut idx: HashMap<String, Vec<(i32, usize)>> = HashMap::new();
    for (i, s) in sites.iter().take(n_rows).enumerate() {
        idx.entry(normalize_chrom(&s.chrom))
            .or_default()
            .push((s.pos, i));
    }
    for v in idx.values_mut() {
        v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    }
    idx
}

fn interval_indices(
    chrom_idx: &HashMap<String, Vec<(i32, usize)>>,
    chrom: &str,
    start: i32,
    end: i32,
) -> Vec<usize> {
    let key = normalize_chrom(chrom);
    let Some(v) = chrom_idx.get(&key) else {
        return Vec::new();
    };
    if v.is_empty() {
        return Vec::new();
    }
    let (lo, hi) = if start <= end {
        (start, end)
    } else {
        (end, start)
    };
    let mut out: Vec<usize> = Vec::new();
    for (pos, idx) in v.iter() {
        if *pos < lo {
            continue;
        }
        if *pos > hi {
            break;
        }
        out.push(*idx);
    }
    out
}

fn gather_rows_by_indices(
    bits_flat: &[u64],
    row_words: usize,
    row_indices: &[usize],
) -> Result<Vec<u64>, String> {
    let ctx = "garfield::gather_rows_by_indices";
    if row_indices.is_empty() {
        return Ok(Vec::new());
    }
    let n_rows_all = bits_flat.len() / row_words;
    let mut out = vec![0u64; row_indices.len() * row_words];
    for (dst_r, &src_r) in row_indices.iter().enumerate() {
        if src_r >= n_rows_all {
            return Err(format!(
                "{ctx}: row index out of range: {} >= {}",
                src_r, n_rows_all
            ));
        }
        let src_st = src_r * row_words;
        let dst_st = dst_r * row_words;
        out[dst_st..dst_st + row_words].copy_from_slice(&bits_flat[src_st..src_st + row_words]);
    }
    Ok(out)
}

fn build_feature_group_ids(sites: &[SiteInfo], n_rows: usize) -> Vec<usize> {
    let mut map: HashMap<(String, i32), usize> = HashMap::new();
    let mut out = vec![0usize; n_rows];
    for (i, site) in sites.iter().take(n_rows).enumerate() {
        let key = (normalize_chrom(&site.chrom), site.pos);
        let gid = if let Some(g) = map.get(&key) {
            *g
        } else {
            let g = map.len();
            map.insert(key, g);
            g
        };
        out[i] = gid;
    }
    out
}

fn run_beam_with_feature_exclusion(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    response: GarfieldResponse,
    y_train_f64: &[f64],
    y_train_bin: Option<&[u8]>,
    max_pick: usize,
    beam_width: usize,
    local_group_ids: Option<&[usize]>,
) -> Result<BeamAndResult, String> {
    if let Some(groups) = local_group_ids {
        return match response {
            GarfieldResponse::Binary => {
                let yb = y_train_bin.ok_or_else(|| "binary y is not prepared".to_string())?;
                beam_search_and_with_group_exclusion(
                    bits_flat,
                    row_words,
                    n_rows,
                    n_samples,
                    max_pick,
                    beam_width,
                    groups,
                    |combined| score_binary_mcc_packed(yb, combined, n_samples),
                )
            }
            GarfieldResponse::Continuous => beam_search_and_with_group_exclusion(
                bits_flat,
                row_words,
                n_rows,
                n_samples,
                max_pick,
                beam_width,
                groups,
                |combined| score_cont_corr_packed(y_train_f64, combined, n_samples).abs(),
            ),
        };
    }

    match response {
        GarfieldResponse::Binary => {
            let yb = y_train_bin.ok_or_else(|| "binary y is not prepared".to_string())?;
            beam_search_and_binary_mcc(
                yb, bits_flat, row_words, n_rows, n_samples, max_pick, beam_width,
            )
        }
        GarfieldResponse::Continuous => beam_search_and_continuous_abs_corr(
            y_train_f64,
            bits_flat,
            row_words,
            n_rows,
            n_samples,
            max_pick,
            beam_width,
        ),
    }
}

fn build_windows_from_sites(
    sites: &[SiteInfo],
    n_rows: usize,
    extension: usize,
    step: usize,
) -> Vec<GarfieldWindow> {
    if n_rows == 0 || sites.is_empty() || extension == 0 || step == 0 {
        return vec![GarfieldWindow {
            chrom: "ALL".to_string(),
            bp_start: 0,
            bp_end: 0,
            indices: (0..n_rows).collect(),
        }];
    }

    let mut groups: HashMap<String, Vec<(i32, usize)>> = HashMap::new();
    let mut chrom_order: Vec<String> = Vec::new();
    for (idx, site) in sites.iter().enumerate().take(n_rows) {
        let chrom = normalize_chrom(&site.chrom);
        if !groups.contains_key(&chrom) {
            groups.insert(chrom.clone(), Vec::new());
            chrom_order.push(chrom.clone());
        }
        if let Some(v) = groups.get_mut(&chrom) {
            v.push((site.pos, idx));
        }
    }
    if groups.is_empty() {
        return vec![GarfieldWindow {
            chrom: "ALL".to_string(),
            bp_start: 0,
            bp_end: 0,
            indices: (0..n_rows).collect(),
        }];
    }

    let mut windows: Vec<GarfieldWindow> = Vec::new();
    for chrom in chrom_order {
        let Some(mut pairs) = groups.remove(&chrom) else {
            continue;
        };
        if pairs.is_empty() {
            continue;
        }
        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let n = pairs.len();
        let bps: Vec<i32> = pairs.iter().map(|(bp, _)| *bp).collect();
        let idxs: Vec<usize> = pairs.iter().map(|(_, i)| *i).collect();
        let min_bp = bps[0];
        let max_bp = bps[n - 1];
        let mut l = 0usize;
        let mut r = 0usize;
        let mut center = min_bp;
        let mut prev_sig: Option<(usize, usize, usize)> = None;

        loop {
            let left_i64 = (center as i64)
                .saturating_sub(extension as i64)
                .max(min_bp as i64);
            let right_i64 = (center as i64)
                .saturating_add(extension as i64)
                .min(max_bp as i64);

            while l < n && (bps[l] as i64) < left_i64 {
                l += 1;
            }
            if r < l {
                r = l;
            }
            while r < n && (bps[r] as i64) <= right_i64 {
                r += 1;
            }

            if r > l {
                let chunk = idxs[l..r].to_vec();
                let sig = (chunk[0], chunk[chunk.len() - 1], chunk.len());
                if prev_sig.map_or(true, |s| s != sig) {
                    windows.push(GarfieldWindow {
                        chrom: chrom.clone(),
                        bp_start: left_i64 as i32,
                        bp_end: right_i64 as i32,
                        indices: chunk,
                    });
                    prev_sig = Some(sig);
                }
            }

            if center >= max_bp {
                break;
            }
            center = (center as i64).saturating_add(step as i64) as i32;
            if center <= min_bp && step > 0 {
                break;
            }
        }

        let has_chrom_window = windows.last().map(|w| w.chrom == chrom).unwrap_or(false);
        if !has_chrom_window {
            windows.push(GarfieldWindow {
                chrom: chrom.clone(),
                bp_start: min_bp,
                bp_end: max_bp.saturating_add(1),
                indices: idxs,
            });
        }
    }

    if windows.is_empty() {
        vec![GarfieldWindow {
            chrom: "ALL".to_string(),
            bp_start: 0,
            bp_end: 0,
            indices: (0..n_rows).collect(),
        }]
    } else {
        windows
    }
}

fn read_y_f64(arr: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    match arr.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => arr.as_array().iter().copied().collect(),
    }
}

fn validate_binary_y(y: &[f64]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(y.len());
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("binary y contains non-finite value at index {}", i));
        }
        if v == 0.0 {
            out.push(0u8);
        } else if v == 1.0 {
            out.push(1u8);
        } else {
            return Err(format!(
                "binary response requires y in {{0,1}}; got y[{}]={}",
                i, v
            ));
        }
    }
    Ok(out)
}

fn validate_continuous_y(y: &[f64]) -> Result<(), String> {
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!(
                "continuous y contains non-finite value at index {}",
                i
            ));
        }
    }
    Ok(())
}

#[pyfunction(name = "garfield_prepare_input_bin")]
#[pyo3(signature = (
    input_path,
    out_bin_path,
    input_kind="auto",
    mode="bin",
    threads=0,
    maf=0.0,
    geno=1.0,
    impute=false,
    het=0.05,
    sample_ids=None,
    sample_indices=None
))]
pub fn garfield_prepare_input_bin_py(
    input_path: String,
    out_bin_path: String,
    input_kind: &str,
    mode: &str,
    threads: usize,
    maf: f64,
    geno: f64,
    impute: bool,
    het: f64,
    sample_ids: Option<Vec<String>>,
    sample_indices: Option<Vec<usize>>,
) -> PyResult<(usize, usize, usize, usize)> {
    if !(0.0..=0.5).contains(&maf) {
        return Err(PyValueError::new_err("maf must be within [0, 0.5]"));
    }
    if !(0.0..=1.0).contains(&geno) {
        return Err(PyValueError::new_err("geno must be within [0, 1]"));
    }
    if !(0.0..=1.0).contains(&het) {
        return Err(PyValueError::new_err("het must be within [0, 1]"));
    }

    let input_kind = parse_input_kind(input_kind).map_err(PyValueError::new_err)?;
    let mode = parse_bin_mode(mode).map_err(PyValueError::new_err)?;
    let mut reader = make_input_reader(&input_path, input_kind).map_err(PyRuntimeError::new_err)?;

    let (selected_indices, selected_ids) =
        build_sample_selection(reader.sample_ids(), sample_ids, sample_indices)
            .map_err(PyRuntimeError::new_err)?;
    if selected_indices.is_empty() {
        return Err(PyValueError::new_err("selected sample set is empty"));
    }
    write_sample_id_sidecar(&out_bin_path, &selected_ids).map_err(PyRuntimeError::new_err)?;

    let n_samples = selected_indices.len();
    let identity_selection = selected_indices
        .iter()
        .enumerate()
        .all(|(i, &idx)| i == idx);
    let row_bytes = row_bytes_for_samples(n_samples);
    let mut writer =
        GarfieldBinWriter::new(&out_bin_path, n_samples).map_err(PyRuntimeError::new_err)?;

    let available_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1usize);
    let n_threads = if threads == 0 {
        available_threads
    } else {
        threads.min(available_threads).max(1)
    };
    let pool = if n_threads > 1 {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("build thread pool: {e}")))?,
        )
    } else {
        None
    };
    let pool_ref = pool.as_ref();

    let maf_f32 = maf as f32;
    let geno_f32 = geno as f32;
    let het_f32 = het as f32;

    let mut n_sites_seen = 0usize;
    let mut n_sites_written = 0usize;
    let mut n_rows_written = 0usize;

    match &mut reader {
        GarfieldInputReader::Bed(bed) => {
            if n_threads > 1 {
                let n_sites = bed.sites.len();
                let target_bytes = 16usize * 1024 * 1024;
                let mut chunk_size = target_bytes / n_samples.max(1);
                if chunk_size == 0 {
                    chunk_size = 1;
                }
                if chunk_size > 4096 {
                    chunk_size = 4096;
                }
                for start in (0..n_sites).step_by(chunk_size) {
                    let end = (start + chunk_size).min(n_sites);
                    let rows_res: Vec<Result<Option<Vec<EncodedRow>>, String>> = pool_ref
                        .expect("thread pool exists when n_threads>1")
                        .install(|| {
                            (start..end)
                                .into_par_iter()
                                .map(|idx| {
                                    let (row, site) = if identity_selection {
                                        bed.get_snp_row_raw(idx).ok_or_else(|| {
                                            format!("BED decode failed at row {}", idx)
                                        })?
                                    } else {
                                        bed.get_snp_row_selected_raw(idx, &selected_indices)
                                            .ok_or_else(|| {
                                                format!("BED decode with sample selection failed at row {}", idx)
                                            })?
                                    };
                                    Ok(encode_row_to_bits(
                                        row,
                                        site,
                                        mode,
                                        row_bytes,
                                        maf_f32,
                                        geno_f32,
                                        impute,
                                        het_f32,
                                    ))
                                })
                                .collect()
                        });

                    for item in rows_res.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        let maybe_rows = item.map_err(PyRuntimeError::new_err)?;
                        if let Some(rows) = maybe_rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows.len());
                            writer.write_rows(&rows).map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            } else {
                while let Some((row, site)) = if identity_selection {
                    bed.next_snp_raw()
                } else {
                    bed.next_snp_selected_raw(&selected_indices)
                } {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows) = encode_row_to_bits(
                        row, site, mode, row_bytes, maf_f32, geno_f32, impute, het_f32,
                    ) {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows.len());
                        writer.write_rows(&rows).map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
        GarfieldInputReader::Vcf(vcf) => {
            let mut batch: Vec<(Vec<f32>, SiteInfo)> = Vec::with_capacity(1024);
            loop {
                let Some((row_raw, site)) = vcf.next_snp_raw() else {
                    break;
                };
                let row = if identity_selection {
                    row_raw
                } else {
                    row_select_by_indices(row_raw, &selected_indices)
                        .map_err(PyRuntimeError::new_err)?
                };
                batch.push((row, site));
                if batch.len() >= 1024 {
                    let results = encode_batch_rows(
                        std::mem::take(&mut batch),
                        mode,
                        row_bytes,
                        maf_f32,
                        geno_f32,
                        impute,
                        het_f32,
                        pool_ref,
                    );
                    for rows in results.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        if let Some(rows_kept) = rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                            writer
                                .write_rows(&rows_kept)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            }
            if !batch.is_empty() {
                let results = encode_batch_rows(
                    std::mem::take(&mut batch),
                    mode,
                    row_bytes,
                    maf_f32,
                    geno_f32,
                    impute,
                    het_f32,
                    pool_ref,
                );
                for rows in results.into_iter() {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows_kept) = rows {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                        writer
                            .write_rows(&rows_kept)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
        GarfieldInputReader::Hmp(hmp) => {
            let mut batch: Vec<(Vec<f32>, SiteInfo)> = Vec::with_capacity(1024);
            loop {
                let Some((row_raw, site)) = hmp.next_snp_raw() else {
                    break;
                };
                let row = if identity_selection {
                    row_raw
                } else {
                    row_select_by_indices(row_raw, &selected_indices)
                        .map_err(PyRuntimeError::new_err)?
                };
                batch.push((row, site));
                if batch.len() >= 1024 {
                    let results = encode_batch_rows(
                        std::mem::take(&mut batch),
                        mode,
                        row_bytes,
                        maf_f32,
                        geno_f32,
                        impute,
                        het_f32,
                        pool_ref,
                    );
                    for rows in results.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        if let Some(rows_kept) = rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                            writer
                                .write_rows(&rows_kept)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            }
            if !batch.is_empty() {
                let results = encode_batch_rows(
                    std::mem::take(&mut batch),
                    mode,
                    row_bytes,
                    maf_f32,
                    geno_f32,
                    impute,
                    het_f32,
                    pool_ref,
                );
                for rows in results.into_iter() {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows_kept) = rows {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                        writer
                            .write_rows(&rows_kept)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
        GarfieldInputReader::Txt(txt) => {
            let mut batch: Vec<(Vec<f32>, SiteInfo)> = Vec::with_capacity(1024);
            while let Some((row_raw, site)) = txt.next_snp() {
                let row = if identity_selection {
                    row_raw
                } else {
                    row_select_by_indices(row_raw, &selected_indices)
                        .map_err(PyRuntimeError::new_err)?
                };
                batch.push((row, site));
                if batch.len() >= 1024 {
                    let results = encode_batch_rows(
                        std::mem::take(&mut batch),
                        mode,
                        row_bytes,
                        maf_f32,
                        geno_f32,
                        impute,
                        het_f32,
                        pool_ref,
                    );
                    for rows in results.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        if let Some(rows_kept) = rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                            writer
                                .write_rows(&rows_kept)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            }
            if !batch.is_empty() {
                let results = encode_batch_rows(
                    std::mem::take(&mut batch),
                    mode,
                    row_bytes,
                    maf_f32,
                    geno_f32,
                    impute,
                    het_f32,
                    pool_ref,
                );
                for rows in results.into_iter() {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows_kept) = rows {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                        writer
                            .write_rows(&rows_kept)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
    }

    let header_rows = writer.finish(n_samples).map_err(PyRuntimeError::new_err)?;
    if header_rows != n_rows_written {
        return Err(PyRuntimeError::new_err(format!(
            "internal row count mismatch: header_rows={}, tracked_rows={}",
            header_rows, n_rows_written
        )));
    }

    Ok((n_sites_seen, n_sites_written, n_rows_written, n_samples))
}

#[pyfunction(name = "garfield_subset_bin_samples")]
pub fn garfield_subset_bin_samples_py(
    bin_path: String,
    out_bin_path: String,
    sample_indices: Vec<usize>,
) -> PyResult<()> {
    let ctx = "garfield_subset_bin_samples";
    if sample_indices.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{ctx}: sample_indices is empty"
        )));
    }
    let bytes = fs::read(&bin_path).map_err(|e| PyRuntimeError::new_err(format!("{ctx}: {e}")))?;
    let (n_rows, n_samples, row_bytes, data_offset) =
        parse_bin01_header(&bytes, ctx).map_err(PyRuntimeError::new_err)?;
    for &si in sample_indices.iter() {
        if si >= n_samples {
            return Err(PyValueError::new_err(format!(
                "{ctx}: sample index out of range {} >= {}",
                si, n_samples
            )));
        }
    }

    let n_out = sample_indices.len();
    let row_bytes_out = n_out.div_ceil(8);
    let payload_len = n_rows
        .checked_mul(row_bytes_out)
        .ok_or_else(|| PyRuntimeError::new_err(format!("{ctx}: payload overflow")))?;
    let mut out = Vec::<u8>::with_capacity(BIN01_HEADER_LEN + payload_len);
    out.extend_from_slice(BIN01_MAGIC);
    out.extend_from_slice(&(n_rows as u64).to_le_bytes());
    out.extend_from_slice(&(n_out as u64).to_le_bytes());
    out.extend_from_slice(&(0u64).to_le_bytes());

    for r in 0..n_rows {
        let src_start =
            data_offset
                .checked_add(r.checked_mul(row_bytes).ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{ctx}: row offset overflow"))
                })?)
                .ok_or_else(|| PyRuntimeError::new_err(format!("{ctx}: row offset overflow")))?;
        let src_end = src_start
            .checked_add(row_bytes)
            .ok_or_else(|| PyRuntimeError::new_err(format!("{ctx}: row end overflow")))?;
        let src = &bytes[src_start..src_end];

        let mut row = vec![0u8; row_bytes_out];
        for (dst_i, &src_i) in sample_indices.iter().enumerate() {
            let src_b = src[src_i >> 3];
            let bit = (src_b >> (src_i & 7)) & 1u8;
            if bit != 0 {
                row[dst_i >> 3] |= 1u8 << (dst_i & 7);
            }
        }
        out.extend_from_slice(&row);
    }

    let mut fw = File::create(&out_bin_path)
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: create {out_bin_path}: {e}")))?;
    fw.write_all(&out)
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: write {out_bin_path}: {e}")))?;
    fw.flush()
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: flush {out_bin_path}: {e}")))?;

    copy_site_sidecar(&bin_path, &out_bin_path).map_err(PyRuntimeError::new_err)?;
    let wrote_subset_ids = write_subset_id_sidecar(&bin_path, &out_bin_path, &sample_indices)
        .map_err(PyRuntimeError::new_err)?;
    if !wrote_subset_ids {
        copy_id_sidecar(&bin_path, &out_bin_path).map_err(PyRuntimeError::new_err)?;
    }
    Ok(())
}

#[pyfunction(name = "garfield_scan_groups_bin")]
#[pyo3(signature = (
    bin_path,
    y_train,
    groups,
    response="continuous",
    max_pick=3,
    beam_width=5,
    enforce_feature_exclusion=true
))]
pub fn garfield_scan_groups_bin_py(
    bin_path: String,
    y_train: PyReadonlyArray1<'_, f64>,
    groups: Vec<Vec<(String, i32, i32)>>,
    response: &str,
    max_pick: usize,
    beam_width: usize,
    enforce_feature_exclusion: bool,
) -> PyResult<Vec<(usize, usize, f64, Vec<usize>)>> {
    if groups.is_empty() {
        return Ok(Vec::new());
    }
    let resp = parse_response(response).map_err(PyValueError::new_err)?;
    let y_vec = read_y_f64(&y_train);
    let y_bin_owned = if resp == GarfieldResponse::Binary {
        Some(validate_binary_y(&y_vec).map_err(PyValueError::new_err)?)
    } else {
        validate_continuous_y(&y_vec).map_err(PyValueError::new_err)?;
        None
    };
    let y_bin = y_bin_owned.as_deref();

    let (bits_flat_all, row_words, n_rows_all, n_samples) =
        load_bin01_as_u64_words(&bin_path).map_err(PyRuntimeError::new_err)?;
    if y_vec.len() < n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_scan_groups_bin: y length={} smaller than n_samples={}",
            y_vec.len(),
            n_samples
        )));
    }

    let sites = TxtSnpIter::new(&bin_path, None)
        .map_err(PyRuntimeError::new_err)?
        .sites;
    let chrom_idx = build_chrom_index(&sites, n_rows_all);
    let feature_group_ids_all = if enforce_feature_exclusion {
        Some(build_feature_group_ids(&sites, n_rows_all))
    } else {
        None
    };

    let mut out: Vec<(usize, usize, f64, Vec<usize>)> = Vec::with_capacity(groups.len());
    for (gi, group) in groups.iter().enumerate() {
        let mut idx_all: Vec<usize> = Vec::new();
        for (chrom, start, end) in group.iter() {
            let mut iv = interval_indices(&chrom_idx, chrom, *start, *end);
            idx_all.append(&mut iv);
        }
        if idx_all.is_empty() {
            out.push((gi, 0usize, f64::NEG_INFINITY, Vec::new()));
            continue;
        }
        idx_all.sort_unstable();
        idx_all.dedup();

        let n_rows = idx_all.len();
        let mut bits_flat = vec![0u64; n_rows * row_words];
        for (ri, &src_idx) in idx_all.iter().enumerate() {
            let src_start = src_idx * row_words;
            let dst_start = ri * row_words;
            bits_flat[dst_start..dst_start + row_words]
                .copy_from_slice(&bits_flat_all[src_start..src_start + row_words]);
        }

        let local_group_ids = if let Some(global_ids) = feature_group_ids_all.as_ref() {
            Some(
                idx_all
                    .iter()
                    .map(|&gi| global_ids[gi])
                    .collect::<Vec<usize>>(),
            )
        } else {
            None
        };
        let res = run_beam_with_feature_exclusion(
            &bits_flat,
            row_words,
            n_rows,
            n_samples,
            resp,
            y_vec.as_slice(),
            y_bin,
            max_pick,
            beam_width,
            local_group_ids.as_deref(),
        )
        .map_err(PyRuntimeError::new_err)?;
        let selected_global = res
            .selected_indices
            .iter()
            .map(|&local_idx| idx_all[local_idx])
            .collect::<Vec<_>>();
        out.push((gi, idx_all.len(), res.score, selected_global));
    }
    Ok(out)
}

#[pyfunction(name = "garfield_scan_windows_bin")]
#[pyo3(signature = (
    bin_path,
    y_train,
    response="continuous",
    max_pick=3,
    beam_width=5,
    extension=50000,
    step=None,
    enforce_feature_exclusion=true
))]
pub fn garfield_scan_windows_bin_py(
    bin_path: String,
    y_train: PyReadonlyArray1<'_, f64>,
    response: &str,
    max_pick: usize,
    beam_width: usize,
    extension: usize,
    step: Option<usize>,
    enforce_feature_exclusion: bool,
) -> PyResult<Vec<(usize, String, i32, i32, usize, f64, Vec<usize>)>> {
    let resp = parse_response(response).map_err(PyValueError::new_err)?;
    if extension == 0 {
        return Err(PyValueError::new_err(
            "garfield_scan_windows_bin: extension must be > 0",
        ));
    }
    let step_v = step.unwrap_or((extension / 2).max(1));
    if step_v == 0 {
        return Err(PyValueError::new_err(
            "garfield_scan_windows_bin: step must be > 0",
        ));
    }

    let y_vec = read_y_f64(&y_train);
    let y_bin_owned = if resp == GarfieldResponse::Binary {
        Some(validate_binary_y(&y_vec).map_err(PyValueError::new_err)?)
    } else {
        validate_continuous_y(&y_vec).map_err(PyValueError::new_err)?;
        None
    };
    let y_bin = y_bin_owned.as_deref();

    let (bits_flat_all, row_words, n_rows_all, n_samples) =
        load_bin01_as_u64_words(&bin_path).map_err(PyRuntimeError::new_err)?;
    if y_vec.len() < n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_scan_windows_bin: y length={} smaller than n_samples={}",
            y_vec.len(),
            n_samples
        )));
    }

    let sites = TxtSnpIter::new(&bin_path, None)
        .map_err(PyRuntimeError::new_err)?
        .sites;
    let windows = build_windows_from_sites(&sites, n_rows_all, extension, step_v);
    if windows.is_empty() {
        return Ok(Vec::new());
    }
    let feature_group_ids_all = if enforce_feature_exclusion {
        Some(build_feature_group_ids(&sites, n_rows_all))
    } else {
        None
    };

    let mut out: Vec<(usize, String, i32, i32, usize, f64, Vec<usize>)> =
        Vec::with_capacity(windows.len());
    for (wi0, win) in windows.iter().enumerate() {
        if win.indices.is_empty() {
            continue;
        }
        let n_rows = win.indices.len();
        let sub = if n_rows == n_rows_all {
            bits_flat_all.clone()
        } else {
            gather_rows_by_indices(&bits_flat_all, row_words, &win.indices)
                .map_err(PyRuntimeError::new_err)?
        };

        let local_group_ids = if let Some(global_ids) = feature_group_ids_all.as_ref() {
            Some(
                win.indices
                    .iter()
                    .map(|&gi| global_ids[gi])
                    .collect::<Vec<usize>>(),
            )
        } else {
            None
        };

        let res = run_beam_with_feature_exclusion(
            &sub,
            row_words,
            n_rows,
            n_samples,
            resp,
            y_vec.as_slice(),
            y_bin,
            max_pick,
            beam_width,
            local_group_ids.as_deref(),
        )
        .map_err(PyRuntimeError::new_err)?;

        let selected_global = res
            .selected_indices
            .iter()
            .map(|&local_idx| win.indices[local_idx])
            .collect::<Vec<_>>();
        out.push((
            wi0 + 1,
            win.chrom.clone(),
            win.bp_start,
            win.bp_end,
            n_rows,
            res.score,
            selected_global,
        ));
    }
    Ok(out)
}

#[pyfunction(name = "garfield_eval_rule_bin")]
#[pyo3(signature = (bin_path, y, snp_indices, response="continuous"))]
pub fn garfield_eval_rule_bin_py(
    bin_path: String,
    y: PyReadonlyArray1<'_, f64>,
    snp_indices: Vec<usize>,
    response: &str,
) -> PyResult<(f64, usize, Vec<u8>)> {
    let resp = parse_response(response).map_err(PyValueError::new_err)?;
    if snp_indices.is_empty() {
        return Ok((f64::NEG_INFINITY, 0usize, Vec::new()));
    }
    let y_vec = read_y_f64(&y);
    let y_bin = if resp == GarfieldResponse::Binary {
        Some(validate_binary_y(&y_vec).map_err(PyValueError::new_err)?)
    } else {
        validate_continuous_y(&y_vec).map_err(PyValueError::new_err)?;
        None
    };

    let (bits_flat, row_words, n_rows, n_samples) =
        load_bin01_selected_rows_as_u64_words(&bin_path, &snp_indices)
            .map_err(PyRuntimeError::new_err)?;
    if n_rows == 0 {
        return Ok((f64::NEG_INFINITY, 0usize, Vec::new()));
    }
    if y_vec.len() < n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_eval_rule_bin: y length={} smaller than n_samples={}",
            y_vec.len(),
            n_samples
        )));
    }

    let needed_words = words_for_samples(n_samples);
    let mut combined = bits_flat[0..needed_words].to_vec();
    for r in 1..n_rows {
        let st = r * row_words;
        let row = &bits_flat[st..st + needed_words];
        for (a, &b) in combined.iter_mut().zip(row.iter()) {
            *a &= b;
        }
    }
    apply_tail_mask(&mut combined, tail_mask(n_samples));

    let score = match resp {
        GarfieldResponse::Binary => {
            let yb = y_bin.as_ref().expect("binary y prepared");
            score_binary_mcc_packed(yb.as_slice(), &combined, n_samples)
        }
        GarfieldResponse::Continuous => {
            score_cont_corr_packed(y_vec.as_slice(), &combined, n_samples).abs()
        }
    };
    let support = combined
        .iter()
        .map(|w| w.count_ones() as usize)
        .sum::<usize>();
    let mut xcombine = vec![0u8; n_samples];
    for i in 0..n_samples {
        let bit = (combined[i >> 6] >> (i & 63)) & 1u64;
        xcombine[i] = if bit != 0 { 1u8 } else { 0u8 };
    }
    Ok((score, support, xcombine))
}
