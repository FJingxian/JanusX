use crate::kmer::ffi::kmc_reader::KmcReader;
use crate::kmer::format::SampleEntry;
use anyhow::{bail, Context, Result};
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
        db_inputs
            .iter()
            .map(|raw| {
                let prefix = normalize_prefix(raw)?;
                Ok((default_sample_id(&prefix), prefix))
            })
            .collect::<Result<Vec<_>>>()?
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

fn parse_db_list_file(path: &str) -> Result<Vec<(String, String)>> {
    let file =
        File::open(path).with_context(|| format!("failed to open KMC db list file: {path}"))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let text = line.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let fields = text.split_whitespace().collect::<Vec<_>>();
        let (sample_id, prefix_raw) = if fields.len() == 1 {
            let prefix = normalize_prefix(fields[0])?;
            (default_sample_id(&prefix), prefix)
        } else {
            (fields[0].to_string(), normalize_prefix(fields[1])?)
        };
        out.push((sample_id, prefix_raw));
    }
    if out.is_empty() {
        bail!("KMC db list file is empty: {path}");
    }
    Ok(out)
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
