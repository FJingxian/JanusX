#[derive(Clone, Debug)]
pub(crate) struct ActivePathState<Aux = ()> {
    pub(crate) active_rows: Vec<usize>,
    pub(crate) active_mask: Vec<bool>,
    pub(crate) active_diag: Vec<f32>,
    pub(crate) active_weights: Vec<f32>,
    pub(crate) active_beta: Vec<f32>,
    pub(crate) active_dense: Option<Vec<f32>>,
    pub(crate) residual: Vec<f32>,
    pub(crate) aux: Aux,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ActivePathSolveConfig {
    pub(crate) lambda: f32,
    pub(crate) active_cap: usize,
    pub(crate) max_outer: usize,
    pub(crate) max_sweeps: usize,
    pub(crate) cd_tol: f32,
    pub(crate) kkt_tol: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ActivePathSolveStats {
    pub(crate) converged: bool,
    pub(crate) outer_iters: usize,
    pub(crate) sweeps_last: usize,
    pub(crate) total_sweeps: usize,
    pub(crate) max_update_last: f32,
    pub(crate) total_violators: usize,
    pub(crate) total_seed_rows: usize,
    pub(crate) total_added_rows: usize,
    pub(crate) peak_active: usize,
}

pub(crate) fn validate_active_path_state<Aux, F>(
    state: &ActivePathState<Aux>,
    n_features: usize,
    n_samples: usize,
    validate_aux: F,
) -> Result<(), String>
where
    F: FnOnce(&Aux) -> Result<(), String>,
{
    if state.active_mask.len() != n_features {
        return Err(format!(
            "active_mask length mismatch: got {}, expected {}",
            state.active_mask.len(),
            n_features
        ));
    }
    if state.residual.len() != n_samples {
        return Err(format!(
            "residual length mismatch: got {}, expected {}",
            state.residual.len(),
            n_samples
        ));
    }
    let active_len = state.active_rows.len();
    if state.active_diag.len() != active_len
        || state.active_weights.len() != active_len
        || state.active_beta.len() != active_len
    {
        return Err("active warm state vector length mismatch".to_string());
    }
    if let Some(active_dense) = state.active_dense.as_ref() {
        let expected = active_len.saturating_mul(n_samples);
        if active_dense.len() != expected {
            return Err(format!(
                "active_dense length mismatch: got {}, expected {}",
                active_dense.len(),
                expected
            ));
        }
    }
    for &row_idx in &state.active_rows {
        if row_idx >= n_features {
            return Err(format!(
                "active row index out of bounds: {row_idx} >= {n_features}"
            ));
        }
        if !state.active_mask[row_idx] {
            return Err(format!("active_mask missing active row {row_idx}"));
        }
    }
    validate_aux(&state.aux)
}

pub(crate) fn run_active_kkt_path<
    Aux,
    RestoreDense,
    SeedRows,
    DenseSweep,
    StreamSweep,
    ScanKkt,
    ExtendRows,
>(
    mut state: ActivePathState<Aux>,
    cfg: ActivePathSolveConfig,
    mut restore_dense: RestoreDense,
    mut seed_rows: SeedRows,
    mut dense_sweep: DenseSweep,
    mut stream_sweep: StreamSweep,
    mut scan_kkt: ScanKkt,
    mut extend_rows: ExtendRows,
) -> Result<(ActivePathState<Aux>, ActivePathSolveStats), String>
where
    RestoreDense: FnMut(&[usize]) -> Result<Vec<f32>, String>,
    SeedRows: FnMut(&mut ActivePathState<Aux>) -> Result<usize, String>,
    DenseSweep: FnMut(&mut ActivePathState<Aux>, f32) -> Result<f32, String>,
    StreamSweep: FnMut(&mut ActivePathState<Aux>, f32, usize) -> Result<f32, String>,
    ScanKkt: FnMut(&mut ActivePathState<Aux>, f32, f32) -> Result<Vec<usize>, String>,
    ExtendRows: FnMut(&mut ActivePathState<Aux>, &[usize], usize) -> Result<usize, String>,
{
    if state.active_dense.is_none()
        && !state.active_rows.is_empty()
        && state.active_rows.len() <= cfg.active_cap
    {
        state.active_dense = Some(restore_dense(&state.active_rows)?);
    }

    let mut stats = ActivePathSolveStats {
        peak_active: state.active_rows.len(),
        ..ActivePathSolveStats::default()
    };

    let seeded = seed_rows(&mut state)?;
    stats.total_seed_rows = seeded;
    stats.peak_active = stats.peak_active.max(state.active_rows.len());

    for outer in 0..cfg.max_outer.max(1) {
        stats.sweeps_last = 0usize;
        stats.peak_active = stats.peak_active.max(state.active_rows.len());
        if !state.active_rows.is_empty() {
            for _ in 0..cfg.max_sweeps.max(1) {
                let max_update = if state.active_dense.is_some() {
                    dense_sweep(&mut state, cfg.lambda)?
                } else {
                    stream_sweep(&mut state, cfg.lambda, cfg.active_cap)?
                };
                stats.sweeps_last += 1;
                stats.total_sweeps += 1;
                stats.max_update_last = max_update;
                if max_update <= cfg.cd_tol {
                    break;
                }
            }
        } else {
            stats.max_update_last = 0.0_f32;
        }

        let violators = scan_kkt(&mut state, cfg.lambda, cfg.kkt_tol)?;
        stats.outer_iters = outer + 1;
        if violators.is_empty() {
            stats.converged = true;
            break;
        }
        stats.total_violators += violators.len();
        let added = extend_rows(&mut state, &violators, cfg.active_cap)?;
        stats.total_added_rows += added;
        stats.peak_active = stats.peak_active.max(state.active_rows.len());
    }

    Ok((state, stats))
}
