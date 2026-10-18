/// Resolve the four-cell support floor on the samples actually scored.
/// Production Max6 uses one aligned binary matrix per phenotype, so all pairs
/// share this effective sample count. Resolve once before observed/null scans.
pub(super) fn resolve_min_cell(
    count: usize,
    fraction: f64,
    n_effective: usize,
) -> Result<usize, String> {
    if !fraction.is_finite() || !(0.0..=0.25).contains(&fraction) {
        return Err("min_cell_fraction must be finite and between 0 and 0.25".to_string());
    }
    Ok(count.max((fraction * n_effective as f64).ceil() as usize))
}
