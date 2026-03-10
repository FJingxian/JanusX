use flate2::read::MultiGzDecoder;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Clone, Copy)]
enum SepKind {
    Tab,
    Comma,
    Whitespace,
}

fn detect_sep(line: &str) -> SepKind {
    if line.contains('\t') {
        SepKind::Tab
    } else if line.contains(',') {
        SepKind::Comma
    } else {
        SepKind::Whitespace
    }
}

fn split_fields(line: &str, sep: SepKind) -> Vec<&str> {
    match sep {
        SepKind::Tab => line.split('\t').collect(),
        SepKind::Comma => line.split(',').collect(),
        SepKind::Whitespace => line.split_whitespace().collect(),
    }
}

fn normalize_header_name(text: &str) -> String {
    text.trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim_start_matches('\u{feff}')
        .to_ascii_lowercase()
}

fn pick_column_index(header_norm: &[String], candidates: &[String]) -> Option<usize> {
    for cand in candidates {
        let cand_norm = cand.to_ascii_lowercase();
        if let Some((idx, _)) = header_norm
            .iter()
            .enumerate()
            .find(|(_, name)| name.as_str() == cand_norm.as_str())
        {
            return Some(idx);
        }
    }
    None
}

fn open_reader(path: &Path) -> Result<Box<dyn BufRead>, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    if path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| x.eq_ignore_ascii_case("gz"))
        .unwrap_or(false)
    {
        let decoder = MultiGzDecoder::new(file);
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

#[pyfunction]
#[pyo3(signature = (path, chrom_candidates, pos_candidates, p_candidates))]
pub fn load_gwas_triplet_fast(
    path: String,
    chrom_candidates: Vec<String>,
    pos_candidates: Vec<String>,
    p_candidates: Vec<String>,
) -> PyResult<(Vec<String>, Vec<f64>, Vec<f64>, String, String, String)> {
    let path_obj = Path::new(&path);
    let mut reader = open_reader(path_obj).map_err(PyRuntimeError::new_err)?;

    let mut sep = SepKind::Whitespace;
    let mut chr_idx: usize = 0;
    let mut pos_idx: usize = 0;
    let mut p_idx: usize = 0;
    let mut found_header = false;
    let mut chosen_chr = String::new();
    let mut chosen_pos = String::new();
    let mut chosen_p = String::new();
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).map_err(|err| {
            PyRuntimeError::new_err(format!("failed to read {}: {err}", path_obj.display()))
        })?;
        if n == 0 {
            break;
        }

        let raw = line.trim_end_matches(['\n', '\r']);
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with("##") {
            continue;
        }

        let this_sep = detect_sep(trimmed);
        let fields = split_fields(trimmed, this_sep);
        if fields.len() < 3 {
            continue;
        }
        let fields_clean: Vec<String> = fields.iter().map(|x| x.trim().to_string()).collect();
        let header_norm: Vec<String> = fields_clean
            .iter()
            .map(|x| normalize_header_name(x))
            .collect();

        let c1 = pick_column_index(&header_norm, &chrom_candidates);
        let c2 = pick_column_index(&header_norm, &pos_candidates);
        let c3 = pick_column_index(&header_norm, &p_candidates);
        if let (Some(i_chr), Some(i_pos), Some(i_p)) = (c1, c2, c3) {
            sep = this_sep;
            chr_idx = i_chr;
            pos_idx = i_pos;
            p_idx = i_p;
            chosen_chr = fields_clean[chr_idx].clone();
            chosen_pos = fields_clean[pos_idx].clone();
            chosen_p = fields_clean[p_idx].clone();
            found_header = true;
            break;
        }
    }

    if !found_header {
        return Err(PyRuntimeError::new_err(
            "Cannot detect required columns (chrom/pos/pvalue) from header.",
        ));
    }

    let max_idx = chr_idx.max(pos_idx).max(p_idx);
    let mut chrom_vals: Vec<String> = Vec::with_capacity(1 << 20);
    let mut pos_vals: Vec<f64> = Vec::with_capacity(1 << 20);
    let mut p_vals: Vec<f64> = Vec::with_capacity(1 << 20);

    loop {
        line.clear();
        let n = reader.read_line(&mut line).map_err(|err| {
            PyRuntimeError::new_err(format!("failed to read {}: {err}", path_obj.display()))
        })?;
        if n == 0 {
            break;
        }
        let raw = line.trim_end_matches(['\n', '\r']);
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields = split_fields(trimmed, sep);
        if fields.len() <= max_idx {
            continue;
        }
        let chrom = fields[chr_idx].trim();
        if chrom.is_empty() {
            continue;
        }

        let pos = match fields[pos_idx].trim().parse::<f64>() {
            Ok(v) if v.is_finite() => v,
            _ => continue,
        };
        let p = match fields[p_idx].trim().parse::<f64>() {
            Ok(v) if v.is_finite() && v > 0.0 && v <= 1.0 => v,
            _ => continue,
        };
        chrom_vals.push(chrom.to_string());
        pos_vals.push(pos);
        p_vals.push(p);
    }

    if chrom_vals.is_empty() {
        return Err(PyRuntimeError::new_err(
            "No valid SNP rows after filtering (pos/pvalue).",
        ));
    }

    Ok((
        chrom_vals, pos_vals, p_vals, chosen_chr, chosen_pos, chosen_p,
    ))
}
