// src/gfcore.rs
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use flate2::read::MultiGzDecoder;
use memmap2::{Mmap, MmapOptions};

const BED_HEADER_LEN: usize = 3;

#[inline]
fn system_page_size() -> usize {
    #[cfg(unix)]
    {
        let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if ps > 0 {
            ps as usize
        } else {
            4096
        }
    }
    #[cfg(not(unix))]
    {
        4096
    }
}

// ---------------------------
// Variant metadata
// ---------------------------
#[derive(Clone, Debug)]
pub struct SiteInfo {
    pub chrom: String,
    pub pos: i32,
    pub ref_allele: String,
    pub alt_allele: String,
}

// ---------------------------
// PLINK helpers
// ---------------------------
pub fn read_fam(prefix: &str) -> Result<Vec<String>, String> {
    let fam_path = format!("{prefix}.fam");
    let file = File::open(&fam_path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);

    let mut samples = Vec::new();
    for line in reader.lines() {
        let l = line.map_err(|e| e.to_string())?;
        let mut it = l.split_whitespace();
        it.next(); // FID
        if let Some(iid) = it.next() {
            samples.push(iid.to_string());
        } else {
            return Err(format!("Malformed FAM line: {l}"));
        }
    }
    Ok(samples)
}

pub fn read_bim(prefix: &str) -> Result<Vec<SiteInfo>, String> {
    let bim_path = PathBuf::from(format!("{prefix}.bim"));
    read_bim_file(&bim_path)
}

// ---------------------------
// VCF open helper
// ---------------------------
pub fn open_text_maybe_gz(path: &Path) -> Result<Box<dyn BufRead + Send + Sync>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    if path.extension().map(|e| e == "gz").unwrap_or(false) {
        Ok(Box::new(BufReader::new(MultiGzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

// ---------------------------
// SNP row processing
// ---------------------------
pub fn process_snp_row(
    row: &mut [f32],
    ref_allele: &mut String,
    alt_allele: &mut String,
    maf_threshold: f32,
    max_missing_rate: f32,
    fill_missing: bool,
    apply_het_filter: bool,
    het_threshold: f32,
) -> bool {
    let mut alt_sum: f64 = 0.0;
    let mut non_missing: i64 = 0;
    let mut het_count: i64 = 0;

    for &g in row.iter() {
        if g >= 0.0 {
            alt_sum += g as f64;
            non_missing += 1;
            if apply_het_filter && (g - 1.0).abs() < 1e-6 {
                het_count += 1;
            }
        }
    }

    let n_samples = row.len() as f64;
    if n_samples == 0.0 {
        return false;
    }

    let missing_rate = 1.0 - (non_missing as f64 / n_samples);
    if missing_rate > max_missing_rate as f64 {
        return false;
    }

    if non_missing == 0 {
        if maf_threshold > 0.0 {
            return false;
        } else {
            if fill_missing {
                row.fill(0.0);
            }
            return true;
        }
    }

    if apply_het_filter {
        let het_rate = het_count as f64 / non_missing as f64;
        let low = het_threshold as f64;
        let high = 1.0 - low;
        if het_rate < low || het_rate > high {
            return false;
        }
    }

    let mut alt_freq = alt_sum / (2.0 * non_missing as f64);

    if alt_freq > 0.5 {
        for g in row.iter_mut() {
            if *g >= 0.0 {
                *g = 2.0 - *g;
            }
        }
        std::mem::swap(ref_allele, alt_allele);
        alt_sum = 2.0 * non_missing as f64 - alt_sum;
        alt_freq = alt_sum / (2.0 * non_missing as f64);
    }

    let maf = alt_freq.min(1.0 - alt_freq);
    if maf < maf_threshold as f64 {
        return false;
    }

    if fill_missing {
        let mean_g = alt_sum / non_missing as f64;
        let imputed: f32 = mean_g as f32;
        for g in row.iter_mut() {
            if *g < 0.0 {
                *g = imputed;
            }
        }
    }

    true
}

fn parse_delimiter_char(delimiter: Option<&str>) -> Result<Option<char>, String> {
    match delimiter {
        None => Ok(None),
        Some(s) => {
            let d = s.trim();
            if d.is_empty() {
                return Ok(None);
            }
            if d == "\\t" {
                return Ok(Some('\t'));
            }
            let mut chars = d.chars();
            let ch = chars
                .next()
                .ok_or_else(|| "delimiter is empty".to_string())?;
            if chars.next().is_some() {
                return Err(format!(
                    "delimiter must be a single character or \"\\\\t\", got: {d}"
                ));
            }
            Ok(Some(ch))
        }
    }
}

fn parse_txt_numeric_token(tok: &str) -> Result<f32, String> {
    let t = tok.trim();
    if t.is_empty() {
        return Err("empty token".to_string());
    }
    let up = t.to_ascii_uppercase();
    // Treat common textual missing-value markers as internal missing code.
    if matches!(up.as_str(), "NA" | "NAN" | "NULL" | "." | "-") {
        return Ok(-9.0);
    }
    t.parse::<f32>()
        .map_err(|_| format!("invalid float token: {tok}"))
}

fn split_line_tokens<'a>(line: &'a str, delimiter: Option<char>) -> Vec<&'a str> {
    if let Some(delim) = delimiter {
        if delim.is_ascii_whitespace() {
            line.split_whitespace().collect()
        } else {
            line.split(|c: char| c == delim || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .collect()
        }
    } else {
        line.split(|c: char| c.is_whitespace() || c == ',' || c == ';')
            .filter(|s| !s.is_empty())
            .collect()
    }
}

struct TxtPaths {
    prefix: PathBuf,
    txt_path: Option<PathBuf>,
    npy_path: PathBuf,
    id_path: PathBuf,
    src_site_path: Option<PathBuf>,
    src_bim_path: Option<PathBuf>,
    cache_bim_path: PathBuf,
}

fn cache_prefix_path(prefix: &Path) -> PathBuf {
    let name = prefix.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if name.starts_with('~') {
        return prefix.to_path_buf();
    }
    let mut cached = prefix.to_path_buf();
    cached.set_file_name(format!("~{name}"));
    cached
}

fn strip_cache_prefix(prefix: &Path) -> Option<PathBuf> {
    let name = prefix.file_name()?.to_str()?;
    if !name.starts_with('~') {
        return None;
    }
    let mut original = prefix.to_path_buf();
    original.set_file_name(&name[1..]);
    Some(original)
}

fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut os: OsString = path.as_os_str().to_os_string();
    os.push(suffix);
    PathBuf::from(os)
}

fn find_txt_with_prefix(prefix: &Path) -> Option<PathBuf> {
    let txt = append_suffix(prefix, ".txt");
    if txt.exists() {
        return Some(txt);
    }
    let tsv = append_suffix(prefix, ".tsv");
    if tsv.exists() {
        return Some(tsv);
    }
    let csv = append_suffix(prefix, ".csv");
    if csv.exists() {
        return Some(csv);
    }
    None
}

fn find_id_with_prefix(prefix: &Path) -> Option<PathBuf> {
    let id = append_suffix(prefix, ".id");
    if id.exists() {
        return Some(id);
    }
    None
}

fn find_site_with_prefix(prefix: &Path) -> Option<PathBuf> {
    let candidates = [
        append_suffix(prefix, ".site"),
        append_suffix(prefix, ".site.tsv"),
        append_suffix(prefix, ".site.txt"),
        append_suffix(prefix, ".site.csv"),
        append_suffix(prefix, ".sites.tsv"),
        append_suffix(prefix, ".sites.txt"),
        append_suffix(prefix, ".sites.csv"),
    ];
    for cand in candidates {
        if cand.exists() {
            return Some(cand);
        }
    }
    None
}

fn find_bim_with_prefix(prefix: &Path) -> Option<PathBuf> {
    let bim = append_suffix(prefix, ".bim");
    if bim.exists() {
        return Some(bim);
    }
    None
}

fn resolve_txt_paths(path_or_prefix: &str) -> Result<TxtPaths, String> {
    let input = Path::new(path_or_prefix);
    let ext = input
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());

    if let Some(ext) = ext.as_deref() {
        if matches!(ext, "txt" | "tsv" | "csv") {
            if !input.exists() {
                return Err(format!("text matrix file not found: {}", input.display()));
            }
            let mut prefix = input.with_extension("");
            if let Some(p) = strip_cache_prefix(&prefix) {
                prefix = p;
            }
            let cache_prefix = cache_prefix_path(&prefix);
            let npy_path = append_suffix(&cache_prefix, ".npy");
            let id_path = find_id_with_prefix(&prefix)
                .or_else(|| find_id_with_prefix(&cache_prefix))
                .unwrap_or_else(|| append_suffix(&prefix, ".id"));
            let src_site_path = find_site_with_prefix(&prefix);
            let src_bim_path = find_bim_with_prefix(&prefix);
            let cache_bim_path = append_suffix(&cache_prefix, ".bim");
            return Ok(TxtPaths {
                prefix,
                txt_path: Some(input.to_path_buf()),
                npy_path,
                id_path,
                src_site_path,
                src_bim_path,
                cache_bim_path,
            });
        }
        if ext == "npy" {
            if !input.exists() {
                return Err(format!("NPY file not found: {}", input.display()));
            }
            let cache_prefix = input.with_extension("");
            let prefix = strip_cache_prefix(&cache_prefix).unwrap_or_else(|| cache_prefix.clone());
            let txt_path =
                find_txt_with_prefix(&prefix).or_else(|| find_txt_with_prefix(&cache_prefix));
            let id_path = find_id_with_prefix(&prefix)
                .or_else(|| find_id_with_prefix(&cache_prefix))
                .unwrap_or_else(|| append_suffix(&prefix, ".id"));
            let src_site_path =
                find_site_with_prefix(&prefix).or_else(|| find_site_with_prefix(&cache_prefix));
            let src_bim_path = find_bim_with_prefix(&prefix);
            let cache_bim_path = append_suffix(&cache_prefix_path(&prefix), ".bim");
            return Ok(TxtPaths {
                prefix,
                txt_path,
                npy_path: input.to_path_buf(),
                id_path,
                src_site_path,
                src_bim_path,
                cache_bim_path,
            });
        }
    }

    if input.exists() && input.is_file() {
        // Unknown extension: treat as text matrix path, cache to "<path>.npy".
        let prefix = strip_cache_prefix(input).unwrap_or_else(|| input.to_path_buf());
        let cache_prefix = cache_prefix_path(&prefix);
        let npy_path = append_suffix(&cache_prefix, ".npy");
        let id_path = find_id_with_prefix(&prefix)
            .or_else(|| find_id_with_prefix(&cache_prefix))
            .unwrap_or_else(|| append_suffix(&prefix, ".id"));
        let src_site_path =
            find_site_with_prefix(&prefix).or_else(|| find_site_with_prefix(&cache_prefix));
        let src_bim_path = find_bim_with_prefix(&prefix);
        let cache_bim_path = append_suffix(&cache_prefix, ".bim");
        return Ok(TxtPaths {
            prefix,
            txt_path: Some(input.to_path_buf()),
            npy_path,
            id_path,
            src_site_path,
            src_bim_path,
            cache_bim_path,
        });
    }

    let raw_prefix = input.to_path_buf();
    let prefix = strip_cache_prefix(&raw_prefix).unwrap_or(raw_prefix.clone());
    let txt_path = find_txt_with_prefix(&prefix).or_else(|| find_txt_with_prefix(&raw_prefix));
    let cache_prefix = cache_prefix_path(&prefix);
    let npy_path = append_suffix(&cache_prefix, ".npy");
    let id_path = find_id_with_prefix(&prefix)
        .or_else(|| find_id_with_prefix(&raw_prefix))
        .or_else(|| find_id_with_prefix(&cache_prefix))
        .unwrap_or_else(|| append_suffix(&prefix, ".id"));
    let src_site_path = find_site_with_prefix(&prefix)
        .or_else(|| find_site_with_prefix(&raw_prefix))
        .or_else(|| find_site_with_prefix(&cache_prefix));
    let src_bim_path = find_bim_with_prefix(&prefix);
    let cache_bim_path = append_suffix(&cache_prefix, ".bim");
    if !npy_path.exists() && txt_path.is_none() {
        return Err(format!("text matrix not found: {path_or_prefix}"));
    }
    Ok(TxtPaths {
        prefix,
        txt_path,
        npy_path,
        id_path,
        src_site_path,
        src_bim_path,
        cache_bim_path,
    })
}

fn read_id_file(path: &Path) -> Result<Vec<String>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (line_no, line) in reader.lines().enumerate() {
        let l = line.map_err(|e| format!("{}:{}: {}", path.display(), line_no + 1, e))?;
        let trimmed = l.trim();
        if trimmed.is_empty() {
            continue;
        }
        let toks = split_line_tokens(trimmed, None);
        if toks.len() != 1 {
            return Err(format!(
                "{}:{}: expected 1 sample ID per line",
                path.display(),
                line_no + 1
            ));
        }
        let sid = toks[0].to_string();
        if !seen.insert(sid.clone()) {
            return Err(format!("duplicate sample ID in {}: {sid}", path.display()));
        }
        samples.push(sid);
    }
    if samples.is_empty() {
        return Err(format!("no sample IDs found in {}", path.display()));
    }
    Ok(samples)
}

fn infer_text_delimiter(line: &str) -> Option<char> {
    if line.contains(',') && !line.contains('\t') {
        Some(',')
    } else if line.contains('\t') {
        Some('\t')
    } else {
        None
    }
}

fn read_site_file(path: &Path) -> Result<Vec<SiteInfo>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);

    let mut sites = Vec::new();
    let mut delimiter: Option<char> = None;
    let mut header_ready = false;
    let mut idx_chr = 0usize;
    let mut idx_pos = 1usize;
    let mut idx_ref = 2usize;
    let mut idx_alt = 3usize;

    for (line_no, line) in reader.lines().enumerate() {
        let l = line.map_err(|e| format!("{}:{}: {}", path.display(), line_no + 1, e))?;
        let trimmed = l.trim();
        if trimmed.is_empty() {
            continue;
        }
        if delimiter.is_none() {
            delimiter = infer_text_delimiter(trimmed);
        }
        let toks = split_line_tokens(trimmed, delimiter);
        if toks.is_empty() {
            continue;
        }
        if !header_ready {
            header_ready = true;
            let low: Vec<String> = toks.iter().map(|x| x.trim().to_ascii_lowercase()).collect();
            let pick = |cands: &[&str]| -> Option<usize> {
                low.iter().position(|v| cands.iter().any(|cand| v == cand))
            };
            if let (Some(chr_i), Some(pos_i), Some(ref_i), Some(alt_i)) = (
                pick(&["#chrom", "chrom", "chr", "chromosome"]),
                pick(&["pos", "bp", "position", "ps"]),
                pick(&["ref", "a0", "ref_allele"]),
                pick(&["alt", "a1", "alt_allele"]),
            ) {
                idx_chr = chr_i;
                idx_pos = pos_i;
                idx_ref = ref_i;
                idx_alt = alt_i;
                continue;
            }
        }
        if toks.len() <= idx_alt {
            return Err(format!(
                "Malformed site line at {}:{}: {trimmed}",
                path.display(),
                line_no + 1
            ));
        }
        let chrom = toks[idx_chr].to_string();
        let pos: i32 = toks[idx_pos].parse().map_err(|_| {
            format!(
                "invalid position at {}:{} -> {}",
                path.display(),
                line_no + 1,
                toks[idx_pos]
            )
        })?;
        sites.push(SiteInfo {
            chrom,
            pos,
            ref_allele: toks[idx_ref].to_string(),
            alt_allele: toks[idx_alt].to_string(),
        });
    }
    if sites.is_empty() {
        return Err(format!("no site rows found in {}", path.display()));
    }
    Ok(sites)
}

fn default_sites(n_rows: usize) -> Vec<SiteInfo> {
    let mut sites = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let pos = i32::try_from(i + 1).unwrap_or(i32::MAX);
        sites.push(SiteInfo {
            chrom: "N".to_string(),
            pos,
            ref_allele: "N".to_string(),
            alt_allele: "N".to_string(),
        });
    }
    sites
}

fn read_bim_file(path: &Path) -> Result<Vec<SiteInfo>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);

    let mut sites = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let l = line.map_err(|e| format!("{}:{}: {}", path.display(), line_no + 1, e))?;
        let cols: Vec<&str> = l.split_whitespace().collect();
        if cols.len() < 6 {
            return Err(format!(
                "Malformed BIM line at {}:{}: {l}",
                path.display(),
                line_no + 1
            ));
        }
        let chrom = cols[0].to_string();
        let pos: i32 = cols[3].parse().unwrap_or(0);
        let a1 = cols[4].to_string();
        let a2 = cols[5].to_string();

        sites.push(SiteInfo {
            chrom,
            pos,
            ref_allele: a1,
            alt_allele: a2,
        });
    }
    Ok(sites)
}

fn write_bim_file(path: &Path, sites: &[SiteInfo]) -> Result<(), String> {
    let file = File::create(path).map_err(|e| e.to_string())?;
    let mut writer = BufWriter::new(file);
    for s in sites.iter() {
        let sid = format!("{}_{}", s.chrom, s.pos);
        writeln!(
            writer,
            "{}\t{}\t0\t{}\t{}\t{}",
            s.chrom, sid, s.pos, s.ref_allele, s.alt_allele
        )
        .map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|e| e.to_string())
}

fn resolve_txt_sites(paths: &TxtPaths, n_snps: usize) -> Result<Vec<SiteInfo>, String> {
    if let Some(src_site) = paths.src_site_path.as_ref() {
        let sites = read_site_file(src_site)?;
        if sites.len() != n_snps {
            return Err(format!(
                "site row count mismatch: {} has {} rows, expected {}",
                src_site.display(),
                sites.len(),
                n_snps
            ));
        }
        write_bim_file(&paths.cache_bim_path, &sites)?;
        return Ok(sites);
    }

    if let Some(src_bim) = paths.src_bim_path.as_ref() {
        let sites = read_bim_file(src_bim)?;
        if sites.len() != n_snps {
            return Err(format!(
                "bim row count mismatch: {} has {} rows, expected {}",
                src_bim.display(),
                sites.len(),
                n_snps
            ));
        }
        write_bim_file(&paths.cache_bim_path, &sites)?;
        return Ok(sites);
    }

    if paths.cache_bim_path.exists() {
        let sites = read_bim_file(&paths.cache_bim_path)?;
        if sites.len() != n_snps {
            return Err(format!(
                "cached bim row count mismatch: {} has {} rows, expected {}",
                paths.cache_bim_path.display(),
                sites.len(),
                n_snps
            ));
        }
        return Ok(sites);
    }

    let sites = default_sites(n_snps);
    write_bim_file(&paths.cache_bim_path, &sites)?;
    Ok(sites)
}

fn purge_stale_txt_cache(paths: &TxtPaths) -> Result<(), String> {
    // Only compare freshness when the original text matrix is available.
    if paths.txt_path.is_none() {
        return Ok(());
    }

    let mut source_files: Vec<&Path> = Vec::with_capacity(4);
    if let Some(txt_path) = paths.txt_path.as_ref() {
        source_files.push(txt_path.as_path());
    }
    source_files.push(paths.id_path.as_path());
    if let Some(src_site_path) = paths.src_site_path.as_ref() {
        source_files.push(src_site_path.as_path());
    }
    if let Some(src_bim_path) = paths.src_bim_path.as_ref() {
        source_files.push(src_bim_path.as_path());
    }
    if source_files.is_empty() {
        return Ok(());
    }

    let cache_files = [paths.npy_path.as_path(), paths.cache_bim_path.as_path()];
    let existing_cache_files: Vec<&Path> =
        cache_files.iter().copied().filter(|p| p.exists()).collect();
    if existing_cache_files.is_empty() {
        return Ok(());
    }

    let newest_source = source_files
        .iter()
        .filter_map(|p| fs::metadata(p).and_then(|m| m.modified()).ok())
        .max();
    let oldest_cache = existing_cache_files
        .iter()
        .filter_map(|p| fs::metadata(p).and_then(|m| m.modified()).ok())
        .min();

    if matches!((newest_source, oldest_cache), (Some(src), Some(cache)) if cache < src) {
        eprintln!(
            "warning: detected stale TXT cache for {} (cache older than genotype files); removing and rebuilding.",
            paths.prefix.display()
        );
        for path in cache_files {
            if path.exists() {
                fs::remove_file(path).map_err(|e| {
                    format!(
                        "failed to remove stale cache file {}: {}",
                        path.display(),
                        e
                    )
                })?;
            }
        }
    }

    Ok(())
}

fn write_npy_f32_header(w: &mut File, rows: usize, cols: usize) -> Result<usize, String> {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    let preamble_len = 10usize; // magic(6)+version(2)+header_len(2)
    let pad_len = (16 - ((preamble_len + header.len() + 1) % 16)) % 16;
    if pad_len > 0 {
        header.push_str(&" ".repeat(pad_len));
    }
    header.push('\n');

    let header_len = header.len();
    if header_len > u16::MAX as usize {
        return Err("NPY header too large for v1.0".into());
    }

    w.write_all(b"\x93NUMPY").map_err(|e| e.to_string())?;
    w.write_all(&[1u8, 0u8]).map_err(|e| e.to_string())?;
    w.write_all(&(header_len as u16).to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(header.as_bytes()).map_err(|e| e.to_string())?;

    Ok(preamble_len + header_len)
}

fn parse_npy_shape(header: &str) -> Result<(usize, usize), String> {
    let shape_key_pos = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| "NPY header missing shape field".to_string())?;
    let after = &header[shape_key_pos..];
    let open = after
        .find('(')
        .ok_or_else(|| "NPY header has malformed shape tuple".to_string())?;
    let close = after[open + 1..]
        .find(')')
        .ok_or_else(|| "NPY header has malformed shape tuple".to_string())?;
    let inside = &after[open + 1..open + 1 + close];

    let dims: Vec<usize> = inside
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| format!("invalid NPY shape dimension: {s}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    match dims.as_slice() {
        [rows] => Ok((*rows, 1)),
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(format!("unsupported NPY shape rank: {:?}", dims)),
    }
}

fn parse_npy_f32_header(bytes: &[u8]) -> Result<(usize, usize, usize), String> {
    if bytes.len() < 10 {
        return Err("NPY file too small".into());
    }
    if &bytes[0..6] != b"\x93NUMPY" {
        return Err("invalid NPY magic".into());
    }

    let major = bytes[6];
    let minor = bytes[7];
    let (header_len, header_start) = match major {
        1 => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10usize)
        }
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("NPY file too small for v2/v3 header".into());
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12usize)
        }
        _ => return Err(format!("unsupported NPY version: {major}.{minor}")),
    };
    let header_end = header_start
        .checked_add(header_len)
        .ok_or_else(|| "NPY header overflow".to_string())?;
    if header_end > bytes.len() {
        return Err("NPY header exceeds file size".into());
    }

    let header =
        std::str::from_utf8(&bytes[header_start..header_end]).map_err(|e| e.to_string())?;
    if !header.contains("descr': '<f4'")
        && !header.contains("descr': '|f4'")
        && !header.contains("descr\": \"<f4\"")
        && !header.contains("descr\": \"|f4\"")
    {
        return Err("NPY dtype is not float32".into());
    }
    if header.contains("fortran_order': True") || header.contains("fortran_order\": true") {
        return Err("fortran_order=True NPY is not supported".into());
    }

    let (rows, cols) = parse_npy_shape(header)?;
    let data_offset = header_end;
    let data_bytes = rows
        .checked_mul(cols)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| "NPY data size overflow".to_string())?;
    let expected_end = data_offset
        .checked_add(data_bytes)
        .ok_or_else(|| "NPY file size overflow".to_string())?;
    if expected_end > bytes.len() {
        return Err("NPY data truncated".into());
    }
    Ok((rows, cols, data_offset))
}

fn convert_text_matrix_to_npy(
    txt_path: &Path,
    npy_path: &Path,
    delimiter: Option<char>,
    expected_cols: usize,
) -> Result<(usize, usize), String> {
    let input = File::open(txt_path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(input);

    let raw_tmp_path = PathBuf::from(format!("{}.rawtmp", npy_path.to_string_lossy()));
    let raw_tmp = File::create(&raw_tmp_path).map_err(|e| e.to_string())?;
    let mut raw_writer = BufWriter::new(raw_tmp);

    let mut n_rows: usize = 0;

    for (line_no, line) in reader.lines().enumerate() {
        let l = line.map_err(|e| format!("{}:{}: {}", txt_path.display(), line_no + 1, e))?;
        let trimmed = l.trim();
        if trimmed.is_empty() {
            continue;
        }
        let toks = split_line_tokens(trimmed, delimiter);
        if toks.is_empty() {
            continue;
        }

        if toks.len() != expected_cols {
            return Err(format!(
                "column count mismatch at {}:{}: expected {}, got {}",
                txt_path.display(),
                line_no + 1,
                expected_cols,
                toks.len()
            ));
        }

        for tok in toks {
            let v: f32 = parse_txt_numeric_token(tok).map_err(|_| {
                format!(
                    "invalid float at {}:{} -> {}",
                    txt_path.display(),
                    line_no + 1,
                    tok
                )
            })?;
            raw_writer
                .write_all(&v.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
        n_rows += 1;
    }
    raw_writer.flush().map_err(|e| e.to_string())?;

    if n_rows == 0 {
        let _ = fs::remove_file(&raw_tmp_path);
        return Err(format!(
            "no numeric matrix rows found in {}",
            txt_path.display()
        ));
    }

    let mut npy_file = File::create(npy_path).map_err(|e| e.to_string())?;
    write_npy_f32_header(&mut npy_file, n_rows, expected_cols)?;
    let mut raw_reader = File::open(&raw_tmp_path).map_err(|e| e.to_string())?;
    std::io::copy(&mut raw_reader, &mut npy_file).map_err(|e| e.to_string())?;
    npy_file.sync_all().map_err(|e| e.to_string())?;
    let _ = fs::remove_file(&raw_tmp_path);
    Ok((n_rows, expected_cols))
}

fn ensure_cached_text_npy(
    txt_path: &Path,
    npy_path: &Path,
    delimiter: Option<char>,
    expected_cols: usize,
) -> Result<(usize, usize), String> {
    let txt_mtime = fs::metadata(txt_path).and_then(|m| m.modified()).ok();
    let npy_mtime = fs::metadata(npy_path).and_then(|m| m.modified()).ok();
    let use_cache =
        npy_path.exists() && matches!((txt_mtime, npy_mtime), (Some(t), Some(n)) if n >= t);

    if use_cache {
        let file = File::open(npy_path).map_err(|e| e.to_string())?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
        if let Ok((rows, cols, _)) = parse_npy_f32_header(&mmap[..]) {
            return Ok((rows, cols));
        }
    }

    convert_text_matrix_to_npy(txt_path, npy_path, delimiter, expected_cols)
}

// ======================================================================
// BED SNP iterator (single SNP each time): returns Vec<f32> (len n)
// ======================================================================
pub struct BedSnpIter {
    #[allow(dead_code)]
    pub prefix: String,
    pub samples: Vec<String>,
    pub sites: Vec<SiteInfo>,
    mmap: Mmap,
    mmap_offset: usize,
    window_snps: Option<usize>,
    window_start_snp: usize,
    window_len_snps: usize,
    bed_len: usize,
    bed_file: Option<File>,
    n_samples: usize,
    n_snps: usize,
    bytes_per_snp: usize,
    cur: usize,
    #[allow(dead_code)]
    maf: f32,
    #[allow(dead_code)]
    miss: f32,
    #[allow(dead_code)]
    fill_missing: bool,
    #[allow(dead_code)]
    apply_het_filter: bool,
    #[allow(dead_code)]
    het_threshold: f32,
}

impl BedSnpIter {
    pub fn new_with_fill(
        prefix: &str,
        maf: f32,
        miss: f32,
        fill_missing: bool,
        apply_het_filter: bool,
        het_threshold: f32,
    ) -> Result<Self, String> {
        let samples = read_fam(prefix)?;
        let sites = read_bim(prefix)?;
        let n_samples = samples.len();
        let n_snps = sites.len();

        let bed_path = format!("{prefix}.bed");
        let file = File::open(&bed_path).map_err(|e| e.to_string())?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
        let bed_len = mmap.len();

        if bed_len < BED_HEADER_LEN {
            return Err("BED too small".into());
        }
        if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
            return Err("Only SNP-major BED supported".into());
        }

        let bytes_per_snp = (n_samples + 3) / 4;

        Ok(Self {
            prefix: prefix.to_string(),
            samples,
            sites,
            mmap,
            mmap_offset: 0,
            window_snps: None,
            window_start_snp: 0,
            window_len_snps: n_snps,
            bed_len,
            bed_file: None,
            n_samples,
            n_snps,
            bytes_per_snp,
            cur: 0,
            maf,
            miss,
            fill_missing,
            apply_het_filter,
            het_threshold,
        })
    }

    pub fn new_with_fill_window(
        prefix: &str,
        maf: f32,
        miss: f32,
        fill_missing: bool,
        apply_het_filter: bool,
        het_threshold: f32,
        mmap_window_mb: usize,
    ) -> Result<Self, String> {
        if mmap_window_mb == 0 {
            return Err("mmap_window_mb must be > 0".into());
        }
        let samples = read_fam(prefix)?;
        let sites = read_bim(prefix)?;
        let n_samples = samples.len();
        let n_snps = sites.len();

        let bed_path = format!("{prefix}.bed");
        let mut file = File::open(&bed_path).map_err(|e| e.to_string())?;
        let bed_len = file.metadata().map_err(|e| e.to_string())?.len() as usize;

        let mut header = [0u8; BED_HEADER_LEN];
        file.read_exact(&mut header).map_err(|e| e.to_string())?;
        if header[0] != 0x6C || header[1] != 0x1B || header[2] != 0x01 {
            return Err("Only SNP-major BED supported".into());
        }

        let bytes_per_snp = (n_samples + 3) / 4;
        let window_bytes = mmap_window_mb.saturating_mul(1024 * 1024);
        let window_snps = std::cmp::max(1, window_bytes / bytes_per_snp);
        let (mmap, mmap_offset, window_len_snps) =
            Self::map_window(&file, bed_len, 0, window_snps, bytes_per_snp)?;

        Ok(Self {
            prefix: prefix.to_string(),
            samples,
            sites,
            mmap,
            mmap_offset,
            window_snps: Some(window_snps),
            window_start_snp: 0,
            window_len_snps,
            bed_len,
            bed_file: Some(file),
            n_samples,
            n_snps,
            bytes_per_snp,
            cur: 0,
            maf,
            miss,
            fill_missing,
            apply_het_filter,
            het_threshold,
        })
    }

    fn map_window(
        file: &File,
        bed_len: usize,
        start_snp: usize,
        window_snps: usize,
        bytes_per_snp: usize,
    ) -> Result<(Mmap, usize, usize), String> {
        let data_len = bed_len
            .checked_sub(BED_HEADER_LEN)
            .ok_or_else(|| "BED too small".to_string())?;
        let max_snps = data_len / bytes_per_snp;
        if start_snp >= max_snps {
            return Err("window start SNP out of range".into());
        }
        let remaining_snps = max_snps - start_snp;
        let map_snps = remaining_snps.min(window_snps);

        let desired_offset = BED_HEADER_LEN + start_snp * bytes_per_snp;
        let page_size = system_page_size();
        let aligned_offset = desired_offset / page_size * page_size;
        let leading = desired_offset - aligned_offset;
        let desired_len = leading + map_snps * bytes_per_snp;
        let max_len = bed_len - aligned_offset;
        let map_len = desired_len.min(max_len);

        let mmap = unsafe {
            MmapOptions::new()
                .offset(aligned_offset as u64)
                .len(map_len)
                .map(file)
                .map_err(|e| e.to_string())?
        };
        Ok((mmap, aligned_offset, map_snps))
    }

    fn remap_window(&mut self, start_snp: usize) -> Result<(), String> {
        let window_snps = self
            .window_snps
            .ok_or_else(|| "window_snps not set".to_string())?;
        let file = self
            .bed_file
            .as_ref()
            .ok_or_else(|| "bed_file not set".to_string())?;
        let (mmap, mmap_offset, window_len_snps) = Self::map_window(
            file,
            self.bed_len,
            start_snp,
            window_snps,
            self.bytes_per_snp,
        )?;
        self.mmap = mmap;
        self.mmap_offset = mmap_offset;
        self.window_start_snp = start_snp;
        self.window_len_snps = window_len_snps;
        Ok(())
    }

    fn snp_bytes(&self, snp_idx: usize) -> Option<&[u8]> {
        let offset = BED_HEADER_LEN + snp_idx * self.bytes_per_snp;
        if let Some(_window_snps) = self.window_snps {
            let window_end = self.window_start_snp + self.window_len_snps;
            if snp_idx < self.window_start_snp || snp_idx >= window_end {
                return None;
            }
        }
        let rel = offset.checked_sub(self.mmap_offset)?;
        let end = rel + self.bytes_per_snp;
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[rel..end])
    }

    fn decode_snp_row_raw(&self, snp_idx: usize) -> Option<(Vec<f32>, SiteInfo)> {
        if snp_idx >= self.n_snps {
            return None;
        }
        if self.window_snps.is_some() {
            panic!("windowed mmap does not support random SNP access");
        }

        let snp_bytes = self.snp_bytes(snp_idx)?;

        let mut row: Vec<f32> = vec![-9.0; self.n_samples];

        for (byte_idx, byte) in snp_bytes.iter().enumerate() {
            for within in 0..4 {
                let samp_idx = byte_idx * 4 + within;
                if samp_idx >= self.n_samples {
                    break;
                }
                let code = (byte >> (within * 2)) & 0b11;
                row[samp_idx] = match code {
                    0b00 => 0.0,
                    0b10 => 1.0,
                    0b11 => 2.0,
                    0b01 => -9.0,
                    _ => -9.0,
                };
            }
        }

        let site = self.sites[snp_idx].clone();
        Some((row, site))
    }

    /// 随机访问解码某个 SNP（用于并行）
    pub fn get_snp_row_raw(&self, snp_idx: usize) -> Option<(Vec<f32>, SiteInfo)> {
        self.decode_snp_row_raw(snp_idx)
    }

    /// 随机访问并按迭代器参数过滤 SNP
    #[allow(dead_code)]
    pub fn get_snp_row(&self, snp_idx: usize) -> Option<(Vec<f32>, SiteInfo)> {
        let (mut row, mut site) = self.decode_snp_row_raw(snp_idx)?;
        let keep = crate::gfcore::process_snp_row(
            &mut row,
            &mut site.ref_allele,
            &mut site.alt_allele,
            self.maf,
            self.miss,
            self.fill_missing,
            self.apply_het_filter,
            self.het_threshold,
        );
        if keep {
            Some((row, site))
        } else {
            None
        }
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn next_snp_raw(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        while self.cur < self.n_snps {
            let snp_idx = self.cur;
            self.cur += 1;

            if self.window_snps.is_some() {
                let window_end = self.window_start_snp + self.window_len_snps;
                if snp_idx < self.window_start_snp || snp_idx >= window_end {
                    if let Err(e) = self.remap_window(snp_idx) {
                        panic!("failed to remap BED window: {e}");
                    }
                }
            }

            let snp_bytes = match self.snp_bytes(snp_idx) {
                Some(bytes) => bytes,
                None => return None,
            };

            let mut row: Vec<f32> = vec![-9.0; self.n_samples];

            for (byte_idx, byte) in snp_bytes.iter().enumerate() {
                for within in 0..4 {
                    let samp_idx = byte_idx * 4 + within;
                    if samp_idx >= self.n_samples {
                        break;
                    }
                    let code = (byte >> (within * 2)) & 0b11;
                    row[samp_idx] = match code {
                        0b00 => 0.0,
                        0b10 => 1.0,
                        0b11 => 2.0,
                        0b01 => -9.0,
                        _ => -9.0,
                    };
                }
            }

            let site = self.sites[snp_idx].clone();
            return Some((row, site));
        }
        None
    }

    #[allow(dead_code)]
    pub fn next_snp(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        loop {
            let (mut row, mut site) = self.next_snp_raw()?;
            let keep = process_snp_row(
                &mut row,
                &mut site.ref_allele,
                &mut site.alt_allele,
                self.maf,
                self.miss,
                self.fill_missing,
                self.apply_het_filter,
                self.het_threshold,
            );
            if keep {
                return Some((row, site));
            }
        }
    }
}

// ======================================================================
// VCF SNP iterator (single SNP each time)
// ======================================================================
pub struct VcfSnpIter {
    pub samples: Vec<String>,
    reader: Box<dyn BufRead + Send + Sync + 'static>,
    #[allow(dead_code)]
    maf: f32,
    #[allow(dead_code)]
    miss: f32,
    #[allow(dead_code)]
    fill_missing: bool,
    #[allow(dead_code)]
    apply_het_filter: bool,
    #[allow(dead_code)]
    het_threshold: f32,
    finished: bool,
}

impl VcfSnpIter {
    pub fn new_with_fill(
        path: &str,
        maf: f32,
        miss: f32,
        fill_missing: bool,
        apply_het_filter: bool,
        het_threshold: f32,
    ) -> Result<Self, String> {
        let p = Path::new(path);
        let mut reader = open_text_maybe_gz(p)?;

        // parse header to get samples
        let mut header_line = String::new();
        let samples: Vec<String>;
        loop {
            header_line.clear();
            let n = reader
                .read_line(&mut header_line)
                .map_err(|e| e.to_string())?;
            if n == 0 {
                return Err("No #CHROM header found in VCF".into());
            }
            if header_line.starts_with("#CHROM") {
                let parts: Vec<_> = header_line.trim_end().split('\t').collect();
                if parts.len() < 10 {
                    return Err("#CHROM header too short".into());
                }
                samples = parts[9..].iter().map(|s| s.to_string()).collect();
                break;
            }
        }

        Ok(Self {
            samples,
            reader,
            maf,
            miss,
            fill_missing,
            apply_het_filter,
            het_threshold,
            finished: false,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn next_snp_raw(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        if self.finished {
            return None;
        }

        let mut line = String::new();
        loop {
            line.clear();
            let n = self.reader.read_line(&mut line).ok()?;
            if n == 0 {
                self.finished = true;
                return None;
            }
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }

            let parts: Vec<_> = line.trim_end().split('\t').collect();
            if parts.len() < 10 {
                continue;
            }

            let format = parts[8];
            if !format.split(':').any(|f| f == "GT") {
                continue;
            }

            let site = SiteInfo {
                chrom: parts[0].to_string(),
                pos: parts[1].parse().unwrap_or(0),
                ref_allele: parts[3].to_string(),
                alt_allele: parts[4].to_string(),
            };

            let mut row: Vec<f32> = Vec::with_capacity(self.samples.len());
            for s in 9..parts.len() {
                let gt = parts[s].split(':').next().unwrap_or(".");
                let g = match gt {
                    "0/0" | "0|0" => 0.0,
                    "0/1" | "1/0" | "0|1" | "1|0" => 1.0,
                    "1/1" | "1|1" => 2.0,
                    "./." | ".|." => -9.0,
                    _ => -9.0,
                };
                row.push(g);
            }
            return Some((row, site));
        }
    }

    #[allow(dead_code)]
    pub fn next_snp(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        loop {
            let (mut row, mut site) = self.next_snp_raw()?;
            let keep = process_snp_row(
                &mut row,
                &mut site.ref_allele,
                &mut site.alt_allele,
                self.maf,
                self.miss,
                self.fill_missing,
                self.apply_het_filter,
                self.het_threshold,
            );
            if keep {
                return Some((row, site));
            }
        }
    }
}

// ======================================================================
// HMP SNP iterator (single SNP each time)
// Supports HapMap text files (.hmp / .hmp.gz).
// ======================================================================

#[inline]
fn is_dna_base(c: char) -> bool {
    matches!(c, 'A' | 'C' | 'G' | 'T')
}

#[inline]
fn default_alt_for_ref(r: char) -> char {
    match r {
        'A' => 'C',
        'C' => 'A',
        'G' => 'A',
        'T' => 'A',
        _ => 'A',
    }
}

#[inline]
fn iupac_to_pair(c: char) -> Option<(char, char)> {
    match c {
        'R' => Some(('A', 'G')),
        'Y' => Some(('C', 'T')),
        'S' => Some(('G', 'C')),
        'W' => Some(('A', 'T')),
        'K' => Some(('G', 'T')),
        'M' => Some(('A', 'C')),
        _ => None,
    }
}

fn split_hmp_fields(line: &str) -> Vec<&str> {
    let tab_cols: Vec<&str> = line.trim_end().split('\t').collect();
    if tab_cols.len() >= 12 {
        return tab_cols;
    }
    line.split_whitespace().collect()
}

fn parse_hmp_alleles_field(field: &str) -> Option<(char, char)> {
    let up = field.trim().to_ascii_uppercase();
    if up.is_empty() {
        return None;
    }
    let mut bases: Vec<char> = Vec::new();
    for part in up.split(|c: char| c == '/' || c == '|' || c == ',' || c == ';') {
        if part.is_empty() {
            continue;
        }
        let b = part.chars().find(|c| is_dna_base(*c));
        if let Some(x) = b {
            if !bases.contains(&x) {
                bases.push(x);
            }
        }
    }
    if bases.len() >= 2 {
        return Some((bases[0], bases[1]));
    }
    let letters: Vec<char> = up.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if letters.len() == 2 && is_dna_base(letters[0]) && is_dna_base(letters[1]) {
        if letters[0] != letters[1] {
            return Some((letters[0], letters[1]));
        }
        return Some((letters[0], default_alt_for_ref(letters[0])));
    }
    if letters.len() == 1 && is_dna_base(letters[0]) {
        return Some((letters[0], default_alt_for_ref(letters[0])));
    }
    None
}

fn parse_hmp_gt_token(token: &str) -> Option<(char, char)> {
    let up = token.trim().to_ascii_uppercase();
    if up.is_empty() {
        return None;
    }
    if matches!(up.as_str(), "." | "./." | ".|." | "-" | "--" | "N" | "NN" | "NA") {
        return None;
    }

    if up.contains('/') || up.contains('|') {
        let mut alleles: Vec<char> = Vec::with_capacity(2);
        for part in up.split(|c: char| c == '/' || c == '|') {
            if part.is_empty() {
                continue;
            }
            if let Some(c) = part.chars().find(|x| is_dna_base(*x)) {
                alleles.push(c);
                if alleles.len() >= 2 {
                    break;
                }
            }
        }
        if alleles.len() >= 2 {
            return Some((alleles[0], alleles[1]));
        }
        return None;
    }

    let letters: Vec<char> = up.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if letters.len() == 1 {
        let c = letters[0];
        if is_dna_base(c) {
            return Some((c, c));
        }
        if let Some((a, b)) = iupac_to_pair(c) {
            return Some((a, b));
        }
        return None;
    }
    if letters.len() >= 2 && is_dna_base(letters[0]) && is_dna_base(letters[1]) {
        return Some((letters[0], letters[1]));
    }
    None
}

fn infer_hmp_ref_alt_from_samples(sample_fields: &[&str]) -> (char, char) {
    let mut seen: Vec<char> = Vec::new();
    for tok in sample_fields.iter() {
        if let Some((a, b)) = parse_hmp_gt_token(tok) {
            for x in [a, b] {
                if is_dna_base(x) && !seen.contains(&x) {
                    seen.push(x);
                    if seen.len() >= 2 {
                        return (seen[0], seen[1]);
                    }
                }
            }
        }
    }
    if seen.len() == 1 {
        return (seen[0], default_alt_for_ref(seen[0]));
    }
    ('A', 'C')
}

pub struct HmpSnpIter {
    pub samples: Vec<String>,
    reader: Box<dyn BufRead + Send + Sync + 'static>,
    #[allow(dead_code)]
    maf: f32,
    #[allow(dead_code)]
    miss: f32,
    #[allow(dead_code)]
    fill_missing: bool,
    #[allow(dead_code)]
    apply_het_filter: bool,
    #[allow(dead_code)]
    het_threshold: f32,
    finished: bool,
}

impl HmpSnpIter {
    pub fn new_with_fill(
        path: &str,
        maf: f32,
        miss: f32,
        fill_missing: bool,
        apply_het_filter: bool,
        het_threshold: f32,
    ) -> Result<Self, String> {
        let p = Path::new(path);
        let mut reader = open_text_maybe_gz(p)?;
        let mut header_line = String::new();
        let samples: Vec<String>;

        loop {
            header_line.clear();
            let n = reader
                .read_line(&mut header_line)
                .map_err(|e| e.to_string())?;
            if n == 0 {
                return Err("No HapMap header found".into());
            }
            let trimmed = header_line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts = split_hmp_fields(trimmed);
            if parts.len() < 12 {
                return Err("Malformed HapMap header: expected >=12 columns".into());
            }
            samples = parts[11..].iter().map(|s| s.to_string()).collect();
            break;
        }

        Ok(Self {
            samples,
            reader,
            maf,
            miss,
            fill_missing,
            apply_het_filter,
            het_threshold,
            finished: false,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn next_snp_raw(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        if self.finished {
            return None;
        }

        let mut line = String::new();
        loop {
            line.clear();
            let n = self.reader.read_line(&mut line).ok()?;
            if n == 0 {
                self.finished = true;
                return None;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts = split_hmp_fields(trimmed);
            if parts.len() < 11 {
                continue;
            }

            let chrom = parts[2].to_string();
            let pos: i32 = parts[3].parse().unwrap_or(0);
            let sample_fields = if parts.len() > 11 { &parts[11..] } else { &[][..] };

            let (ref_c, alt_c) = if let Some((r, a)) = parse_hmp_alleles_field(parts[1]) {
                if r != a {
                    (r, a)
                } else {
                    let (rr, aa) = infer_hmp_ref_alt_from_samples(sample_fields);
                    (rr, aa)
                }
            } else {
                infer_hmp_ref_alt_from_samples(sample_fields)
            };

            let ref_a = ref_c.to_string();
            let alt_a = alt_c.to_string();

            let mut row: Vec<f32> = Vec::with_capacity(self.samples.len());
            for i in 0..self.samples.len() {
                let tok = if i < sample_fields.len() {
                    sample_fields[i]
                } else {
                    "N"
                };
                let g = if let Some((a1, a2)) = parse_hmp_gt_token(tok) {
                    if (a1 == ref_c || a1 == alt_c) && (a2 == ref_c || a2 == alt_c) {
                        let mut d = 0.0f32;
                        if a1 == alt_c {
                            d += 1.0;
                        }
                        if a2 == alt_c {
                            d += 1.0;
                        }
                        d
                    } else {
                        -9.0
                    }
                } else {
                    -9.0
                };
                row.push(g);
            }

            let site = SiteInfo {
                chrom,
                pos,
                ref_allele: ref_a,
                alt_allele: alt_a,
            };
            return Some((row, site));
        }
    }

    #[allow(dead_code)]
    pub fn next_snp(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        loop {
            let (mut row, mut site) = self.next_snp_raw()?;
            let keep = process_snp_row(
                &mut row,
                &mut site.ref_allele,
                &mut site.alt_allele,
                self.maf,
                self.miss,
                self.fill_missing,
                self.apply_het_filter,
                self.het_threshold,
            );
            if keep {
                return Some((row, site));
            }
        }
    }
}

// ======================================================================
// TXT float32 matrix iterator:
// - read numeric text matrix in chunks
// - convert/cache as .npy (float32, C-order)
// - mmap .npy for row-wise streaming
// ======================================================================
pub struct TxtSnpIter {
    #[allow(dead_code)]
    pub path: String,
    #[allow(dead_code)]
    pub npy_path: String,
    pub samples: Vec<String>,
    pub sites: Vec<SiteInfo>,
    mmap: Mmap,
    data_offset: usize,
    n_samples: usize,
    n_snps: usize,
    row_bytes: usize,
    cur: usize,
}

impl TxtSnpIter {
    pub fn new(path: &str, delimiter: Option<&str>) -> Result<Self, String> {
        let paths = resolve_txt_paths(path)?;

        let delimiter = parse_delimiter_char(delimiter)?;
        let samples = read_id_file(&paths.id_path)?;
        let n_samples = samples.len();
        if n_samples == 0 {
            return Err("sample IDs are empty".into());
        }

        purge_stale_txt_cache(&paths)?;

        let (n_snps, n_cols) = if let Some(txt_path) = paths.txt_path.as_ref() {
            ensure_cached_text_npy(txt_path, &paths.npy_path, delimiter, n_samples)?
        } else {
            let file = File::open(&paths.npy_path).map_err(|e| e.to_string())?;
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
            let (rows, cols, _) = parse_npy_f32_header(&mmap[..])?;
            (rows, cols)
        };
        if n_cols != n_samples {
            return Err(format!(
                "sample ID count mismatch: ids={}, matrix columns={}",
                n_samples, n_cols
            ));
        }
        let sites = resolve_txt_sites(&paths, n_snps)?;

        let file = File::open(&paths.npy_path).map_err(|e| e.to_string())?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
        let (rows, cols, data_offset) = parse_npy_f32_header(&mmap[..])?;
        if rows != n_snps || cols != n_samples {
            return Err(format!(
                "cached NPY shape mismatch for {}: expected ({}, {}), got ({}, {})",
                paths.npy_path.display(),
                n_snps,
                n_samples,
                rows,
                cols
            ));
        }
        let row_bytes = n_samples
            .checked_mul(4)
            .ok_or_else(|| "row byte size overflow".to_string())?;

        Ok(Self {
            path: paths.prefix.to_string_lossy().to_string(),
            npy_path: paths.npy_path.to_string_lossy().to_string(),
            samples,
            sites,
            mmap,
            data_offset,
            n_samples,
            n_snps,
            row_bytes,
            cur: 0,
        })
    }

    fn row_bytes(&self, snp_idx: usize) -> Option<&[u8]> {
        if snp_idx >= self.n_snps {
            return None;
        }
        let start = self
            .data_offset
            .checked_add(snp_idx.checked_mul(self.row_bytes)?)?;
        let end = start.checked_add(self.row_bytes)?;
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[start..end])
    }

    pub fn get_snp_row(&self, snp_idx: usize) -> Option<(Vec<f32>, SiteInfo)> {
        let bytes = self.row_bytes(snp_idx)?;
        let mut row = Vec::with_capacity(self.n_samples);
        for chunk in bytes.chunks_exact(4) {
            row.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        if row.len() != self.n_samples {
            return None;
        }
        Some((row, self.sites[snp_idx].clone()))
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn next_snp(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        if self.cur >= self.n_snps {
            return None;
        }
        let idx = self.cur;
        self.cur += 1;
        self.get_snp_row(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::{read_bim, TxtSnpIter};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_dir(prefix: &str) -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_{prefix}_{stamp}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn txt_iter_defaults_work() {
        let dir = make_temp_dir("txt_default");
        let txt_path = dir.join("geno.txt");

        fs::write(&txt_path, "s1 s2 s3\n0.1 -0.2 0.3\n-0.4 0.5 -0.6\n").unwrap();

        let mut it = TxtSnpIter::new(txt_path.to_str().unwrap(), None).unwrap();
        assert_eq!(it.n_samples(), 3);
        assert_eq!(it.sites.len(), 2);
        assert_eq!(it.sites[0].chrom, "N");
        assert_eq!(it.sites[0].pos, 1);
        assert_eq!(it.sites[1].chrom, "N");
        assert_eq!(it.sites[1].pos, 2);

        let (row0, site0) = it.next_snp().unwrap();
        assert_eq!(site0.pos, 1);
        assert!((row0[0] - 0.1).abs() < 1e-6);
        assert!((row0[1] + 0.2).abs() < 1e-6);
        assert!((row0[2] - 0.3).abs() < 1e-6);

        let (row1, site1) = it.next_snp().unwrap();
        assert_eq!(site1.pos, 2);
        assert!((row1[0] + 0.4).abs() < 1e-6);
        assert!((row1[1] - 0.5).abs() < 1e-6);
        assert!((row1[2] + 0.6).abs() < 1e-6);
        assert!(it.next_snp().is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn txt_iter_reads_header_delim() {
        let dir = make_temp_dir("txt_header_delim");
        let txt_path = dir.join("geno.txt");

        fs::write(&txt_path, "a,b,c\n1,2,3\n4,5,6\n").unwrap();

        let it = TxtSnpIter::new(txt_path.to_str().unwrap(), Some(",")).unwrap();
        assert_eq!(it.n_samples(), 3);
        assert_eq!(it.sites.len(), 2);
        assert_eq!(it.samples[0], "a");
        assert_eq!(it.samples[1], "b");
        assert_eq!(it.samples[2], "c");
        assert_eq!(it.sites[0].chrom, "N");
        assert_eq!(it.sites[0].pos, 1);
        assert_eq!(it.sites[0].ref_allele, "N");
        assert_eq!(it.sites[0].alt_allele, "N");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn txt_iter_prefers_source_bim_and_writes_cache_bim() {
        let dir = make_temp_dir("txt_bim_cache");
        let txt_path = dir.join("g.txt");
        let bim_path = dir.join("g.bim");
        let cache_prefix = dir.join("~g");
        let cache_bim_path = dir.join("~g.bim");

        fs::write(&txt_path, "s1 s2\n0.1 0.2\n0.3 0.4\n").unwrap();
        fs::write(&bim_path, "2 rsA 0 101 A G\n5 rsB 0 202 C T\n").unwrap();

        let it = TxtSnpIter::new(txt_path.to_str().unwrap(), None).unwrap();
        assert_eq!(it.sites.len(), 2);
        assert_eq!(it.sites[0].chrom, "2");
        assert_eq!(it.sites[0].pos, 101);
        assert_eq!(it.sites[0].ref_allele, "A");
        assert_eq!(it.sites[0].alt_allele, "G");
        assert_eq!(it.sites[1].chrom, "5");
        assert_eq!(it.sites[1].pos, 202);
        assert_eq!(it.sites[1].ref_allele, "C");
        assert_eq!(it.sites[1].alt_allele, "T");

        assert!(cache_bim_path.exists());
        let cached_sites = read_bim(cache_prefix.to_str().unwrap()).unwrap();
        assert_eq!(cached_sites.len(), 2);
        assert_eq!(cached_sites[0].chrom, "2");
        assert_eq!(cached_sites[0].pos, 101);
        assert_eq!(cached_sites[0].ref_allele, "A");
        assert_eq!(cached_sites[0].alt_allele, "G");

        let _ = fs::remove_dir_all(&dir);
    }
}
