use crate::kmer::ffi::kmc_reader::KmcReader;
use crate::kmer::format::SampleEntry;
use anyhow::{bail, Context, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct KmcInputInspect {
    pub k: u32,
    pub total_records: u64,
    pub sample_total_kmers: Vec<u64>,
    pub all_canonical: bool,
}

pub fn resolve_samples(
    db_inputs: &[String],
    sample_ids_override: Option<&Vec<String>>,
) -> Result<Vec<SampleEntry>> {
    let mut raw_pairs = if db_inputs.len() == 1 && looks_like_db_list_file(&db_inputs[0]) {
        parse_db_list_file(&db_inputs[0])?
    } else {
        expand_db_inputs(db_inputs)?
    };

    if let Some(overrides) = sample_ids_override {
        if overrides.len() != raw_pairs.len() {
            bail!(
                "sample-id count mismatch: got {}, expected {}",
                overrides.len(),
                raw_pairs.len()
            );
        }
        for (idx, sample_id) in overrides.iter().enumerate() {
            let sid = sample_id.trim();
            if sid.is_empty() {
                bail!("sample-id cannot contain empty values");
            }
            raw_pairs[idx].0 = sid.to_string();
        }
    }

    let mut samples = Vec::with_capacity(raw_pairs.len());
    for (idx, (sample_id, prefix)) in raw_pairs.into_iter().enumerate() {
        validate_prefix(&prefix)?;
        samples.push(SampleEntry {
            index: idx as u32,
            sample_id,
            kmc_prefix: prefix,
        });
    }
    Ok(samples)
}

pub fn inspect_kmc_inputs(samples: &[SampleEntry]) -> Result<KmcInputInspect> {
    let mut common_k = None::<u32>;
    let mut total_records = 0u64;
    let mut sample_total_kmers = Vec::with_capacity(samples.len());
    let mut all_canonical = true;

    for sample in samples {
        let reader = KmcReader::open(&sample.kmc_prefix, 0)
            .with_context(|| format!("failed to inspect KMC db: {}", sample.kmc_prefix))?;
        total_records = total_records.saturating_add(reader.info().total_kmers);
        sample_total_kmers.push(reader.info().total_kmers);
        all_canonical &= reader.info().both_strands;
        let k = reader.info().kmer_length;
        if k == 0 {
            bail!("invalid KMC k-mer length for {}", sample.kmc_prefix);
        }
        if k > 31 {
            bail!("current Rust k-mer backend only supports k <= 31, got {k}");
        }
        if let Some(prev) = common_k {
            if prev != k {
                bail!(
                    "all KMC databases must share one k-mer length, got {} and {}",
                    prev,
                    k
                );
            }
        } else {
            common_k = Some(k);
        }
    }

    Ok(KmcInputInspect {
        k: common_k
            .ok_or_else(|| anyhow::anyhow!("no samples available after resolving KMC inputs"))?,
        total_records,
        sample_total_kmers,
        all_canonical,
    })
}

#[pyfunction(
    name = "kmer_resolve_inputs",
    signature = (db_inputs, sample_ids = None)
)]
pub fn kmer_resolve_inputs_py(
    py: Python<'_>,
    db_inputs: Vec<String>,
    sample_ids: Option<Vec<String>>,
) -> PyResult<Py<PyDict>> {
    let samples = resolve_samples(&db_inputs, sample_ids.as_ref())
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let out = PyDict::new(py);
    out.set_item("n_samples", samples.len())?;
    out.set_item(
        "sample_ids",
        samples
            .iter()
            .map(|sample| sample.sample_id.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "prefixes",
        samples
            .iter()
            .map(|sample| sample.kmc_prefix.clone())
            .collect::<Vec<_>>(),
    )?;
    Ok(out.unbind())
}

fn parse_db_list_file(path: &str) -> Result<Vec<(String, String)>> {
    let file =
        File::open(path).with_context(|| format!("failed to open KMC db list file: {path}"))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    let mut seen = HashSet::<String>::new();
    for line in reader.lines() {
        let line = line?;
        let text = line.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let fields = text.split_whitespace().collect::<Vec<_>>();
        let expanded = if fields.len() == 1 {
            expand_raw_input(fields[0])?
        } else {
            expand_named_input(fields[0], fields[1])?
        };
        for (sample_id, prefix_raw) in expanded {
            if seen.insert(prefix_raw.clone()) {
                out.push((sample_id, prefix_raw));
            }
        }
    }
    if out.is_empty() {
        bail!("KMC db list file is empty: {path}");
    }
    Ok(out)
}

fn expand_db_inputs(db_inputs: &[String]) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    let mut seen = HashSet::<String>::new();
    for raw in db_inputs {
        for (sample_id, prefix) in expand_cli_input(raw)? {
            if seen.insert(prefix.clone()) {
                out.push((sample_id, prefix));
            }
        }
    }
    if out.is_empty() {
        bail!("no valid KMC databases were resolved from input arguments");
    }
    Ok(out)
}

fn expand_cli_input(raw: &str) -> Result<Vec<(String, String)>> {
    let text = raw.trim();
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let prefix = normalize_prefix(text)?;
    if contains_glob_meta(&prefix) {
        return expand_raw_input(text);
    }

    let path = Path::new(text);
    if path.is_dir() {
        return expand_directory_input(path);
    }

    if has_complete_kmc_db(&prefix) {
        return Ok(vec![(default_sample_id(&prefix), prefix)]);
    }

    if path.exists() {
        return Ok(Vec::new());
    }

    bail!(
        "KMC database not complete for prefix `{prefix}`. Expected {prefix}.kmc_pre and {prefix}.kmc_suf"
    )
}

fn expand_directory_input(path: &Path) -> Result<Vec<(String, String)>> {
    let mut prefixes = Vec::new();
    let mut seen = HashSet::<String>::new();
    for entry in fs::read_dir(path)
        .with_context(|| format!("failed to read KMC input directory: {}", path.display()))?
    {
        let entry = entry?;
        let entry_path = entry.path();
        let entry_text = entry_path.to_string_lossy().into_owned();
        let prefix = normalize_prefix(&entry_text)?;
        if has_complete_kmc_db(&prefix) && seen.insert(prefix.clone()) {
            prefixes.push((default_sample_id(&prefix), prefix));
        }
    }
    prefixes.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(prefixes)
}

fn expand_raw_input(raw: &str) -> Result<Vec<(String, String)>> {
    let prefixes = expand_prefix_candidates(raw)?;
    Ok(prefixes
        .into_iter()
        .map(|prefix| (default_sample_id(&prefix), prefix))
        .collect::<Vec<_>>())
}

fn expand_named_input(sample_id: &str, raw: &str) -> Result<Vec<(String, String)>> {
    let sid = sample_id.trim();
    if sid.is_empty() {
        bail!("sample-id cannot be empty in KMC db list file");
    }
    let prefixes = expand_prefix_candidates(raw)?;
    if prefixes.len() > 1 {
        bail!(
            "sample-id `{sid}` maps to multiple KMC databases via pattern `{raw}`; please expand them explicitly"
        );
    }
    Ok(prefixes
        .into_iter()
        .map(|prefix| (sid.to_string(), prefix))
        .collect::<Vec<_>>())
}

fn expand_prefix_candidates(raw: &str) -> Result<Vec<String>> {
    let prefix = normalize_prefix(raw)?;
    if !contains_glob_meta(&prefix) {
        return Ok(vec![prefix]);
    }

    let matched_paths = expand_glob_paths(&prefix)?;
    let mut seen = HashSet::<String>::new();
    let mut prefixes = Vec::new();
    for path in matched_paths {
        let path_text = path.to_string_lossy().into_owned();
        let candidate = normalize_prefix(&path_text)?;
        if has_complete_kmc_db(&candidate) && seen.insert(candidate.clone()) {
            prefixes.push(candidate);
        }
    }
    prefixes.sort();
    if prefixes.is_empty() {
        bail!("glob input `{raw}` did not match any complete KMC database");
    }
    Ok(prefixes)
}

fn contains_glob_meta(text: &str) -> bool {
    text.chars().any(|ch| matches!(ch, '*' | '?'))
}

fn expand_glob_paths(pattern: &str) -> Result<Vec<PathBuf>> {
    let root = if pattern.starts_with('/') {
        PathBuf::from("/")
    } else {
        PathBuf::from(".")
    };
    let mut current = vec![root];
    for component in pattern.split('/') {
        if component.is_empty() {
            continue;
        }
        if component == "." {
            continue;
        }
        if component == ".." {
            current = current
                .into_iter()
                .map(|path| path.parent().unwrap_or(Path::new("/")).to_path_buf())
                .collect::<Vec<_>>();
            continue;
        }

        if contains_glob_meta(component) {
            let mut next = Vec::new();
            for base in &current {
                let dir = if base.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    base.as_path()
                };
                if !dir.is_dir() {
                    continue;
                }
                for entry in fs::read_dir(dir)
                    .with_context(|| format!("failed to read directory for glob expansion: {}", dir.display()))?
                {
                    let entry = entry?;
                    let file_name = entry.file_name();
                    let Some(name) = file_name.to_str() else {
                        continue;
                    };
                    if glob_component_matches(component, name) {
                        next.push(dir.join(name));
                    }
                }
            }
            current = next;
        } else {
            current = current
                .into_iter()
                .map(|path| path.join(component))
                .collect::<Vec<_>>();
        }
    }
    current.sort();
    current.dedup();
    Ok(current)
}

fn glob_component_matches(pattern: &str, text: &str) -> bool {
    let p = pattern.as_bytes();
    let t = text.as_bytes();
    let mut dp = vec![vec![false; t.len() + 1]; p.len() + 1];
    dp[0][0] = true;
    for i in 0..p.len() {
        match p[i] {
            b'*' => {
                dp[i + 1][0] = dp[i][0];
                for j in 0..t.len() {
                    dp[i + 1][j + 1] = dp[i][j + 1] || dp[i + 1][j];
                }
            }
            b'?' => {
                for j in 0..t.len() {
                    if dp[i][j] {
                        dp[i + 1][j + 1] = true;
                    }
                }
            }
            byte => {
                for j in 0..t.len() {
                    if dp[i][j] && byte == t[j] {
                        dp[i + 1][j + 1] = true;
                    }
                }
            }
        }
    }
    dp[p.len()][t.len()]
}

fn looks_like_db_list_file(raw: &str) -> bool {
    let raw = raw.trim();
    if raw.is_empty() {
        return false;
    }
    let lower = raw.to_ascii_lowercase();
    if lower.ends_with(".kmc_pre") || lower.ends_with(".kmc_suf") {
        return false;
    }
    Path::new(raw).is_file()
}

fn normalize_prefix(path_or_prefix: &str) -> Result<String> {
    let value = path_or_prefix.trim();
    if value.is_empty() {
        bail!("KMC prefix cannot be empty");
    }
    let lower = value.to_ascii_lowercase();
    if lower.ends_with(".kmc_pre") {
        return Ok(value[..value.len() - ".kmc_pre".len()].to_string());
    }
    if lower.ends_with(".kmc_suf") {
        return Ok(value[..value.len() - ".kmc_suf".len()].to_string());
    }
    Ok(value.to_string())
}

fn validate_prefix(prefix: &str) -> Result<()> {
    let pre = PathBuf::from(format!("{prefix}.kmc_pre"));
    let suf = PathBuf::from(format!("{prefix}.kmc_suf"));
    if !pre.is_file() || !suf.is_file() {
        bail!(
            "KMC database not complete for prefix `{prefix}`. Expected {} and {}",
            pre.display(),
            suf.display()
        );
    }
    Ok(())
}

fn has_complete_kmc_db(prefix: &str) -> bool {
    let pre = PathBuf::from(format!("{prefix}.kmc_pre"));
    let suf = PathBuf::from(format!("{prefix}.kmc_suf"));
    pre.is_file() && suf.is_file()
}

fn default_sample_id(prefix: &str) -> String {
    let name = Path::new(prefix)
        .file_name()
        .and_then(|x| x.to_str())
        .unwrap_or("sample");
    let mapped = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    let trimmed = mapped.trim_matches(|ch| matches!(ch, '.' | '_' | '-'));
    if trimmed.is_empty() {
        "sample".to_string()
    } else {
        trimmed.to_string()
    }
}
