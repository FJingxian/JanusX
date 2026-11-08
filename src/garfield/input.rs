use super::*;

pub(super) fn load_bin01_sites(path: &str, n_rows: usize) -> Result<Option<Vec<SiteInfo>>, String> {
    let Some(site_path) = discover_site_sidecar(path) else {
        return Ok(None);
    };
    let sites = crate::gfcore::read_site_file(&site_path)?;
    if sites.len() < n_rows {
        return Err(format!(
            "BIN sidecar site count mismatch: {} < n_rows={}",
            sites.len(),
            n_rows
        ));
    }
    Ok(Some(sites))
}

pub(super) fn compute_bin01_group_ids(sites: Option<&[SiteInfo]>, n_rows: usize) -> Vec<u64> {
    let Some(sites) = sites else {
        return (0..n_rows).map(|i| i as u64).collect();
    };
    build_feature_group_ids(sites, n_rows)
        .into_iter()
        .map(|g| g as u64)
        .collect()
}

pub(super) fn load_bin01_packed_owned(
    path: &str,
) -> Result<(Vec<u8>, Vec<u64>, usize, usize, usize), String> {
    let ctx = "garfield::load_bin01_packed";
    let resolved = resolve_bin01_path(path)?;
    let resolved_str = resolved.to_string_lossy().to_string();
    let (packed, n_rows, n_samples, row_bytes) =
        load_bin01_packed_payload_owned(&resolved_str, ctx)?;
    let sites = load_bin01_sites(&resolved_str, n_rows)?;
    if let Some(sites) = sites.as_deref() {
        if sites.iter().any(|site| {
            let suffix = site
                .alt_allele
                .rsplit('|')
                .next()
                .unwrap_or_default()
                .trim();
            ["DOM", "REC", "HET"]
                .iter()
                .any(|mode| suffix.eq_ignore_ascii_case(mode))
        }) {
            return Err(
                "GARFIELD DOM/REC/HET feature rows are retired and cannot be loaded as BIN input"
                    .to_string(),
            );
        }
    }
    let group_ids = compute_bin01_group_ids(sites.as_deref(), n_rows);
    Ok((packed, group_ids, n_samples, n_rows, row_bytes))
}

pub(super) fn copy_site_sidecar(src_bin: &str, dst_bin: &str) -> Result<(), String> {
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

pub(super) fn copy_id_sidecar(src_bin: &str, dst_bin: &str) -> Result<(), String> {
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

pub(super) fn read_sample_ids_from_sidecar(path: &Path) -> Result<Vec<String>, String> {
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

pub(super) fn write_subset_id_sidecar(
    src_bin: &str,
    dst_bin: &str,
    sample_indices: &[usize],
) -> Result<bool, String> {
    let Some(src_id_path) = discover_id_sidecar(src_bin) else {
        return Ok(false);
    };
    let src_ids = read_sample_ids_from_sidecar(&src_id_path)?;
    let dst_id_path = bin01_id_sidecar_path(dst_bin);
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
pub(super) fn normalize_plink_prefix(path_or_prefix: &str) -> String {
    let s = path_or_prefix.trim();
    let low = s.to_ascii_lowercase();
    for suffix in [".bed", ".bim", ".fam", ".pgen", ".pvar", ".psam"] {
        if low.ends_with(suffix) {
            return s[..s.len() - suffix.len()].to_string();
        }
    }
    s.to_string()
}

pub(super) fn make_input_reader(
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
pub(super) fn row_bytes_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(8).max(1)
}

#[inline]
pub(super) fn normalize_genotype3(v: f32) -> Option<u8> {
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

pub(super) fn write_sample_id_sidecar(
    out_bin_path: &str,
    sample_ids: &[String],
) -> Result<(), String> {
    let id_path = bin01_id_sidecar_path(out_bin_path);
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

pub(super) fn row_select_by_indices(
    row: Vec<f32>,
    sample_indices: &[usize],
) -> Result<Vec<f32>, String> {
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

pub(super) fn encode_row_to_bits(
    row: Vec<f32>,
    mut site: SiteInfo,
    mode: GarfieldBinMode,
    row_bytes: usize,
    maf: f32,
    geno: f32,
    impute: bool,
    het: f32,
) -> Option<Vec<EncodedRow>> {
    match mode {
        GarfieldBinMode::Bin => {
            let _ = (impute, het);
            let mut c0 = 0usize;
            let mut c2 = 0usize;
            let mut logic_missing = 0usize;
            let n_samples = row.len();
            for &v in row.iter() {
                match normalize_genotype3(v) {
                    Some(0) => c0 += 1,
                    Some(2) => c2 += 1,
                    _ => logic_missing += 1,
                }
            }
            if n_samples == 0 {
                return None;
            }
            let missing_rate = (logic_missing as f32) / (n_samples as f32);
            if missing_rate > geno {
                return None;
            }
            let usable_homo = n_samples.saturating_sub(logic_missing);
            if usable_homo == 0 {
                return None;
            }
            let alt_freq = (c2 as f32) / (usable_homo as f32);
            let maf_obs = alt_freq.min(1.0_f32 - alt_freq).max(0.0_f32);
            if maf_obs < maf {
                return None;
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
    }
}

pub(super) fn encode_batch_rows(
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
