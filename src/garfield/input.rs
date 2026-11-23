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

#[inline]
pub(super) fn normalize_plink_prefix(path_or_prefix: &str) -> String {
    let s = path_or_prefix.trim();
    let low = s.to_ascii_lowercase();
    for suffix in [
        ".bed",
        ".bim",
        ".fam",
        ".pgen",
        ".pvar.zst",
        ".pvar",
        ".psam",
    ] {
        if low.ends_with(suffix) {
            return s[..s.len() - suffix.len()].to_string();
        }
    }
    s.to_string()
}

#[cfg(test)]
#[inline]
pub(super) fn row_bytes_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(8).max(1)
}

#[cfg(test)]
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

#[cfg(test)]
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
