#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
use crate::blas::{
    cblas_dgemm_dispatch, cblas_dtrsm_dispatch, CblasInt, CBLAS_DIAG_NON_UNIT, CBLAS_LEFT,
    CBLAS_LOWER, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS,
};
use crate::ct::packed::FilledPair;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ConditionMode {
    AddDom,
    Genotype,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InferenceKind {
    LinearModel,
    FixedCovariance,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FitStatus {
    Testable,
    UnidentifiablePhase,
    InsufficientDegreesOfFreedom,
    ZeroResidualVariance,
}

#[derive(Clone, Debug)]
pub(crate) struct PairDesign {
    pub(crate) background: Vec<f64>,
    pub(crate) phase: Vec<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct WhitenedPairDesign {
    pub(crate) background_columns: Vec<Vec<f64>>,
    pub(crate) phase: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WhiteningStageTimings {
    pub(crate) total: Duration,
    pub(crate) rhs_columns_total: usize,
    pub(crate) rhs_columns_unique: usize,
    pub(crate) matrix_assembly: Duration,
    pub(crate) column_deduplication: Duration,
    pub(crate) row_matrix_copy: Duration,
    pub(crate) solve_or_scale: Duration,
    pub(crate) column_unpack: Duration,
    pub(crate) duplicate_restore_and_pair_pack: Duration,
}

impl WhiteningStageTimings {
    pub(crate) fn add_assign(&mut self, other: Self) {
        self.total += other.total;
        self.rhs_columns_total += other.rhs_columns_total;
        self.rhs_columns_unique += other.rhs_columns_unique;
        self.matrix_assembly += other.matrix_assembly;
        self.column_deduplication += other.column_deduplication;
        self.row_matrix_copy += other.row_matrix_copy;
        self.solve_or_scale += other.solve_or_scale;
        self.column_unpack += other.column_unpack;
        self.duplicate_restore_and_pair_pack += other.duplicate_restore_and_pair_pack;
    }

    pub(crate) fn share(self, divisor: usize) -> Self {
        let divisor = u32::try_from(divisor.max(1)).unwrap_or(u32::MAX);
        Self {
            total: self.total / divisor,
            rhs_columns_total: 0,
            rhs_columns_unique: 0,
            matrix_assembly: self.matrix_assembly / divisor,
            column_deduplication: self.column_deduplication / divisor,
            row_matrix_copy: self.row_matrix_copy / divisor,
            solve_or_scale: self.solve_or_scale / divisor,
            column_unpack: self.column_unpack / divisor,
            duplicate_restore_and_pair_pack: self.duplicate_restore_and_pair_pack / divisor,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ConditionalFit {
    pub(crate) status: FitStatus,
    pub(crate) beta_z: f64,
    pub(crate) se_z: f64,
    pub(crate) beta_cis_minus_trans: f64,
    pub(crate) se_cis_minus_trans: f64,
    pub(crate) beta_shared: f64,
    pub(crate) se_shared: f64,
    pub(crate) beta_cis: f64,
    pub(crate) se_cis: f64,
    pub(crate) beta_trans: f64,
    pub(crate) se_trans: f64,
    pub(crate) beta_a1: f64,
    pub(crate) beta_d1: f64,
    pub(crate) beta_a2: f64,
    pub(crate) beta_d2: f64,
    pub(crate) shared_effect_identifiable: bool,
    pub(crate) statistic: f64,
    pub(crate) pwald1: f64,
    pub(crate) phase_wald_statistic: f64,
    pub(crate) pwald2: f64,
    pub(crate) fixed_effect_rank: usize,
    pub(crate) residual_df: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CovarianceMetric {
    Identity { n: usize },
    Diagonal { n: usize, standard_deviation: f64 },
    Cholesky { n: usize, lower: Vec<f64> },
}

impl CovarianceMetric {
    pub(crate) fn identity(n: usize) -> Self {
        Self::Identity { n }
    }

    pub(crate) fn from_dense(matrix: &[f64], n: usize) -> Result<Self, String> {
        if n == 0 || matrix.len() != n.saturating_mul(n) {
            return Err("covariance matrix shape mismatch or empty matrix".to_string());
        }
        validate_symmetric(matrix, n, "covariance matrix")?;
        let mut lower = matrix.to_vec();
        let cholesky_profile = std::env::var_os("JX_CT_PROFILE").map(|_| Instant::now());
        let cholesky = cholesky_relative(&mut lower, n);
        if let Some(started) = cholesky_profile {
            eprintln!(
                "CT_PROFILE covariance_cholesky_seconds={:.6}",
                started.elapsed().as_secs_f64()
            );
        }
        cholesky.ok_or_else(|| {
            "covariance matrix is not positive definite at its numerical scale".to_string()
        })?;
        Ok(Self::Cholesky { n, lower })
    }

    pub(crate) fn from_components(
        grm: &[f64],
        n: usize,
        sigma_g2: f64,
        sigma_e2: f64,
    ) -> Result<Self, String> {
        if n == 0 || grm.len() != n.saturating_mul(n) {
            return Err("GRM shape mismatch or empty matrix".to_string());
        }
        if !(sigma_g2.is_finite() && sigma_g2 >= 0.0) || !(sigma_e2.is_finite() && sigma_e2 > 0.0) {
            return Err(
                "FvLMM variances must be finite, sigma_g2 >= 0 and sigma_e2 > 0".to_string(),
            );
        }
        validate_symmetric(grm, n, "GRM")?;
        if sigma_g2 == 0.0 {
            return Ok(Self::Diagonal {
                n,
                standard_deviation: sigma_e2.sqrt(),
            });
        }
        let mut covariance = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..n {
                covariance[row * n + col] = sigma_g2 * grm[row * n + col];
            }
            covariance[row * n + row] += sigma_e2;
        }
        Self::from_dense(&covariance, n)
    }

    pub(crate) fn n(&self) -> usize {
        match self {
            Self::Identity { n } | Self::Diagonal { n, .. } | Self::Cholesky { n, .. } => *n,
        }
    }

    pub(crate) fn has_cholesky_factor(&self) -> bool {
        matches!(self, Self::Cholesky { .. })
    }

    pub(crate) fn whiten(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        if values.len() != self.n() {
            return Err(format!(
                "covariance whitening vector length mismatch: got {}, expected {}",
                values.len(),
                self.n()
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("covariance whitening input contains non-finite values".to_string());
        }
        match self {
            Self::Identity { .. } => Ok(values.to_vec()),
            Self::Diagonal {
                standard_deviation, ..
            } => Ok(values
                .iter()
                .map(|value| value / standard_deviation)
                .collect()),
            Self::Cholesky { n, lower } => {
                let mut out = values.to_vec();
                for row in 0..*n {
                    let mut value = out[row];
                    for col in 0..row {
                        value -= lower[row * n + col] * out[col];
                    }
                    out[row] = value / lower[row * n + row];
                }
                Ok(out)
            }
        }
    }

    /// Whiten a row-major matrix by solving all columns against the same
    /// Cholesky factor in one pass, preserving the vector solve semantics for
    /// every right-hand side.
    pub(crate) fn whiten_row_major_columns(
        &self,
        values: &[f64],
        n_columns: usize,
    ) -> Result<Vec<Vec<f64>>, String> {
        self.whiten_row_major_columns_profiled(values, n_columns, false)
            .map(|(columns, _)| columns)
    }

    fn whiten_row_major_columns_profiled(
        &self,
        values: &[f64],
        n_columns: usize,
        profile: bool,
    ) -> Result<(Vec<Vec<f64>>, WhiteningStageTimings), String> {
        let mut timings = WhiteningStageTimings::default();
        let n = self.n();
        if n_columns == 0 || values.len() != n.saturating_mul(n_columns) {
            return Err("covariance whitening matrix shape mismatch or empty matrix".to_string());
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("covariance whitening input contains non-finite values".to_string());
        }

        let copy_started = profile.then(Instant::now);
        let mut row_major = values.to_vec();
        if let Some(started) = copy_started {
            timings.row_matrix_copy = started.elapsed();
        }
        let solve_started = profile.then(Instant::now);
        match self {
            Self::Identity { .. } => {}
            Self::Diagonal {
                standard_deviation, ..
            } => {
                for value in &mut row_major {
                    *value /= standard_deviation;
                }
            }
            Self::Cholesky { lower, .. } => {
                #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
                unsafe {
                    cblas_dtrsm_dispatch(
                        CBLAS_ROW_MAJOR,
                        CBLAS_LEFT,
                        CBLAS_LOWER,
                        CBLAS_NO_TRANS,
                        CBLAS_DIAG_NON_UNIT,
                        n as CblasInt,
                        n_columns as CblasInt,
                        1.0,
                        lower.as_ptr(),
                        n as CblasInt,
                        row_major.as_mut_ptr(),
                        n_columns as CblasInt,
                    );
                }
                #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
                {
                    for row in 0..n {
                        let row_start = row * n_columns;
                        for col in 0..row {
                            let factor = lower[row * n + col];
                            let previous_start = col * n_columns;
                            for column in 0..n_columns {
                                row_major[row_start + column] -=
                                    factor * row_major[previous_start + column];
                            }
                        }
                        let diagonal = lower[row * n + row];
                        for column in 0..n_columns {
                            row_major[row_start + column] /= diagonal;
                        }
                    }
                }
            }
        }
        if let Some(started) = solve_started {
            timings.solve_or_scale = started.elapsed();
        }

        let unpack_started = profile.then(Instant::now);
        let mut columns = vec![vec![0.0; n]; n_columns];
        for row in 0..n {
            for column in 0..n_columns {
                columns[column][row] = row_major[row * n_columns + column];
            }
        }
        if let Some(started) = unpack_started {
            timings.column_unpack = started.elapsed();
        }
        Ok((columns, timings))
    }

    pub(crate) fn solve_lower_transpose(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        if values.len() != self.n() {
            return Err(format!(
                "covariance triangular solve length mismatch: got {}, expected {}",
                values.len(),
                self.n()
            ));
        }
        match self {
            Self::Identity { .. } => Ok(values.to_vec()),
            Self::Diagonal {
                standard_deviation, ..
            } => Ok(values
                .iter()
                .map(|value| value / standard_deviation)
                .collect()),
            Self::Cholesky { n, lower } => {
                let mut out = values.to_vec();
                for offset in 0..*n {
                    let row = n - 1 - offset;
                    let mut value = out[row];
                    for col in (row + 1)..*n {
                        value -= lower[col * n + row] * out[col];
                    }
                    out[row] = value / lower[row * n + row];
                }
                Ok(out)
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn whiten_pair_design_batch(
    designs: &[PairDesign],
    q_b: usize,
    metric: &CovarianceMetric,
) -> Result<Vec<WhitenedPairDesign>, String> {
    whiten_pair_design_batch_profiled(designs, q_b, metric, false).map(|(designs, _)| designs)
}

pub(crate) fn whiten_pair_design_batch_profiled(
    designs: &[PairDesign],
    q_b: usize,
    metric: &CovarianceMetric,
    profile: bool,
) -> Result<(Vec<WhitenedPairDesign>, WhiteningStageTimings), String> {
    let total_started = profile.then(Instant::now);
    let mut timings = WhiteningStageTimings::default();
    if q_b == 0 {
        return Err("pair design batch must have at least one background column".to_string());
    }
    if designs.is_empty() {
        return Ok((Vec::new(), timings));
    }

    let n = metric.n();
    let expected_background_len = n
        .checked_mul(q_b)
        .ok_or_else(|| "pair design batch dimensions overflow".to_string())?;
    for design in designs {
        if design.background.len() != expected_background_len || design.phase.len() != n {
            return Err("pair design batch contains inconsistent vector dimensions".to_string());
        }
    }

    let columns_per_pair = q_b
        .checked_add(1)
        .ok_or_else(|| "pair design batch dimensions overflow".to_string())?;
    let n_columns = designs
        .len()
        .checked_mul(columns_per_pair)
        .ok_or_else(|| "pair design batch dimensions overflow".to_string())?;
    if profile {
        timings.rhs_columns_total = n_columns;
    }
    let matrix_len = n
        .checked_mul(n_columns)
        .ok_or_else(|| "pair design batch dimensions overflow".to_string())?;
    let assembly_started = profile.then(Instant::now);
    let mut combined = Vec::with_capacity(matrix_len);
    for row in 0..n {
        let row_start = row * q_b;
        for design in designs {
            combined.extend_from_slice(&design.background[row_start..row_start + q_b]);
            combined.push(design.phase[row]);
        }
    }
    if let Some(started) = assembly_started {
        timings.matrix_assembly = started.elapsed();
    }

    let deduplicate_started = profile.then(Instant::now);
    let (combined, column_mapping) = if metric.has_cholesky_factor() && n > 0 {
        deduplicate_row_major_columns(combined, n, n_columns)?
    } else {
        (combined, (0..n_columns).collect())
    };
    if let Some(started) = deduplicate_started {
        timings.column_deduplication = started.elapsed();
    }
    let n_unique_columns = if n == 0 {
        n_columns
    } else {
        combined.len() / n
    };
    if profile {
        timings.rhs_columns_unique = n_unique_columns;
    }
    let (unique_columns, solve_timings) =
        metric.whiten_row_major_columns_profiled(&combined, n_unique_columns, profile)?;
    timings.row_matrix_copy = solve_timings.row_matrix_copy;
    timings.solve_or_scale = solve_timings.solve_or_scale;
    timings.column_unpack = solve_timings.column_unpack;

    let restore_started = profile.then(Instant::now);
    let whitened_columns = if n_unique_columns == n_columns {
        unique_columns
    } else {
        column_mapping
            .iter()
            .map(|&column| unique_columns[column].clone())
            .collect()
    };
    let mut columns = whitened_columns.into_iter();
    let mut whitened = Vec::with_capacity(designs.len());
    for _ in designs {
        let background_columns = (0..q_b)
            .map(|_| columns.next().expect("pair background column is present"))
            .collect();
        let phase = columns.next().expect("pair phase column is present");
        whitened.push(WhitenedPairDesign {
            background_columns,
            phase,
        });
    }
    if let Some(started) = restore_started {
        timings.duplicate_restore_and_pair_pack = started.elapsed();
    }
    if let Some(started) = total_started {
        timings.total = started.elapsed();
    }
    Ok((whitened, timings))
}

/// Collapse bitwise-identical RHS columns before the expensive Cholesky solve.
/// The returned mapping restores the original column order after whitening.
fn deduplicate_row_major_columns(
    values: Vec<f64>,
    n: usize,
    n_columns: usize,
) -> Result<(Vec<f64>, Vec<usize>), String> {
    let expected_len = n
        .checked_mul(n_columns)
        .ok_or_else(|| "covariance whitening matrix dimensions overflow".to_string())?;
    if n == 0 || n_columns == 0 || values.len() != expected_len {
        return Err("covariance whitening matrix shape mismatch or empty matrix".to_string());
    }

    let packed = pack_discrete_row_major_columns(&values, n, n_columns);
    let mut hashes = vec![0xcbf29ce484222325_u64; n_columns];
    if let Some((packed_values, words_per_column)) = &packed {
        for column in 0..n_columns {
            let mut hash = 0xcbf29ce484222325_u64;
            for word in 0..*words_per_column {
                hash =
                    (hash ^ packed_values[word * n_columns + column]).wrapping_mul(0x100000001b3);
            }
            hashes[column] = hash;
        }
    } else {
        for row in 0..n {
            let row_start = row * n_columns;
            for column in 0..n_columns {
                hashes[column] = (hashes[column] ^ values[row_start + column].to_bits())
                    .wrapping_mul(0x100000001b3);
            }
        }
    }

    let mut unique_source_columns: Vec<usize> = Vec::with_capacity(n_columns);
    let mut mapping: Vec<usize> = Vec::with_capacity(n_columns);
    let mut unique_by_hash: HashMap<u64, Vec<usize>> = HashMap::with_capacity(n_columns);
    for column in 0..n_columns {
        let mut matched = None;
        if let Some(candidates) = unique_by_hash.get(&hashes[column]) {
            'candidate: for &unique_index in candidates {
                let source_column = unique_source_columns[unique_index];
                if let Some((packed_values, words_per_column)) = &packed {
                    for word in 0..*words_per_column {
                        if packed_values[word * n_columns + column]
                            != packed_values[word * n_columns + source_column]
                        {
                            continue 'candidate;
                        }
                    }
                } else {
                    for row in 0..n {
                        let left = values[row * n_columns + column].to_bits();
                        let right = values[row * n_columns + source_column].to_bits();
                        if left != right {
                            continue 'candidate;
                        }
                    }
                }
                matched = Some(unique_index);
                break;
            }
        }
        if let Some(unique_index) = matched {
            mapping.push(unique_index);
        } else {
            unique_source_columns.push(column);
            let unique_index = unique_source_columns.len() - 1;
            mapping.push(unique_index);
            unique_by_hash
                .entry(hashes[column])
                .or_default()
                .push(unique_index);
        }
    }

    if unique_source_columns.len() == n_columns {
        return Ok((values, mapping));
    }

    let compact_len = n
        .checked_mul(unique_source_columns.len())
        .ok_or_else(|| "deduplicated whitening matrix dimensions overflow".to_string())?;
    let mut compact = Vec::with_capacity(compact_len);
    for row in 0..n {
        let row_start = row * n_columns;
        for &column in &unique_source_columns {
            compact.push(values[row_start + column]);
        }
    }
    Ok((compact, mapping))
}

/// Pack CT design columns, whose values are integer dosages/phase codes, into
/// two bits per sample. This makes content hashing and exact duplicate checks
/// cheaper while retaining a full bitwise equality check before deduplication.
fn pack_discrete_row_major_columns(
    values: &[f64],
    n: usize,
    n_columns: usize,
) -> Option<(Vec<u64>, usize)> {
    if n_columns == 0 || values.len() != n.checked_mul(n_columns)? {
        return None;
    }
    let words_per_column = n.saturating_add(31) / 32;
    let packed_len = words_per_column.checked_mul(n_columns)?;
    let mut packed = vec![0_u64; packed_len];
    for row in 0..n {
        let word = row / 32;
        let shift = (row % 32) * 2;
        let row_start = row * n_columns;
        let packed_start = word * n_columns;
        for column in 0..n_columns {
            let code = match values[row_start + column].to_bits() {
                bits if bits == 0.0_f64.to_bits() => 0,
                bits if bits == 1.0_f64.to_bits() => 1,
                bits if bits == 2.0_f64.to_bits() => 2,
                bits if bits == (-1.0_f64).to_bits() => 3,
                _ => return None,
            };
            packed[packed_start + column] |= code << shift;
        }
    }
    Some((packed, words_per_column))
}

fn validate_symmetric(matrix: &[f64], n: usize, label: &str) -> Result<(), String> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} contains non-finite values"));
    }
    let scale = matrix
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let tolerance = 1.0e-10 * scale.max(f64::MIN_POSITIVE);
    for row in 0..n {
        for col in 0..n {
            let value = matrix[row * n + col];
            let opposite = matrix[col * n + row];
            if (value - opposite).abs() > tolerance {
                return Err(format!("{label} must be symmetric at ({row}, {col})"));
            }
        }
    }
    Ok(())
}

fn cholesky_relative(matrix: &mut [f64], n: usize) -> Option<()> {
    let scale = (0..n)
        .map(|i| matrix[i * n + i].abs())
        .fold(0.0_f64, f64::max);
    let threshold = scale * f64::EPSILON * (n as f64).max(1.0) * 8.0;
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    if n >= CHOLESKY_BLOCK_SIZE {
        return cholesky_blocked(matrix, n, threshold);
    }
    cholesky_unblocked(matrix, n, threshold)
}

const CHOLESKY_BLOCK_SIZE: usize = 128;

fn cholesky_unblocked(matrix: &mut [f64], n: usize, threshold: f64) -> Option<()> {
    for row in 0..n {
        for col in 0..=row {
            let mut value = matrix[row * n + col];
            for inner in 0..col {
                value -= matrix[row * n + inner] * matrix[col * n + inner];
            }
            if row == col {
                if !value.is_finite() || value <= threshold {
                    return None;
                }
                matrix[row * n + col] = value.sqrt();
            } else {
                matrix[row * n + col] = value / matrix[col * n + col];
            }
        }
        for col in (row + 1)..n {
            matrix[row * n + col] = 0.0;
        }
    }
    Some(())
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
fn cholesky_blocked(matrix: &mut [f64], n: usize, threshold: f64) -> Option<()> {
    if n == 0 || matrix.len() != n.checked_mul(n)? {
        return None;
    }
    let n_blas = CblasInt::try_from(n).ok()?;
    let mut panel_rhs = Vec::new();

    for block_start in (0..n).step_by(CHOLESKY_BLOCK_SIZE) {
        let block_end = (block_start + CHOLESKY_BLOCK_SIZE).min(n);
        let block_size = block_end - block_start;
        cholesky_diagonal_block(matrix, n, block_start, block_end, threshold)?;

        let below = n - block_end;
        if below == 0 {
            continue;
        }
        let block_blas = CblasInt::try_from(block_size).ok()?;
        let below_blas = CblasInt::try_from(below).ok()?;
        panel_rhs.resize(block_size.checked_mul(below)?, 0.0);
        for row_offset in 0..below {
            let row = block_end + row_offset;
            for col_offset in 0..block_size {
                panel_rhs[col_offset * below + row_offset] =
                    matrix[row * n + block_start + col_offset];
            }
        }

        // Solve L_kk * panel_rhs = A_ik^T, then transpose the result back
        // into the lower panel A_ik = A_ik * L_kk^-T.
        unsafe {
            cblas_dtrsm_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_LEFT,
                CBLAS_LOWER,
                CBLAS_NO_TRANS,
                CBLAS_DIAG_NON_UNIT,
                block_blas,
                below_blas,
                1.0,
                matrix.as_ptr().add(block_start * n + block_start),
                n_blas,
                panel_rhs.as_mut_ptr(),
                below_blas,
            );
        }
        for row_offset in 0..below {
            let row = block_end + row_offset;
            for col_offset in 0..block_size {
                matrix[row * n + block_start + col_offset] =
                    panel_rhs[col_offset * below + row_offset];
            }
        }

        for row_start in (block_end..n).step_by(CHOLESKY_BLOCK_SIZE) {
            let rows = (row_start + CHOLESKY_BLOCK_SIZE).min(n) - row_start;
            let rows_blas = CblasInt::try_from(rows).ok()?;
            for col_start in (block_end..=row_start).step_by(CHOLESKY_BLOCK_SIZE) {
                let cols = (col_start + CHOLESKY_BLOCK_SIZE).min(n) - col_start;
                let cols_blas = CblasInt::try_from(cols).ok()?;
                // The input panel occupies columns before block_end, while
                // the output tile starts at or after block_end; these regions
                // are disjoint even when row_start == col_start.
                unsafe {
                    cblas_dgemm_dispatch(
                        CBLAS_ROW_MAJOR,
                        CBLAS_NO_TRANS,
                        CBLAS_TRANS,
                        rows_blas,
                        cols_blas,
                        block_blas,
                        -1.0,
                        matrix.as_ptr().add(row_start * n + block_start),
                        n_blas,
                        matrix.as_ptr().add(col_start * n + block_start),
                        n_blas,
                        1.0,
                        matrix.as_mut_ptr().add(row_start * n + col_start),
                        n_blas,
                    );
                }
            }
        }
    }
    Some(())
}

fn cholesky_diagonal_block(
    matrix: &mut [f64],
    n: usize,
    block_start: usize,
    block_end: usize,
    threshold: f64,
) -> Option<()> {
    for row in block_start..block_end {
        for col in block_start..=row {
            let mut value = matrix[row * n + col];
            for inner in block_start..col {
                value -= matrix[row * n + inner] * matrix[col * n + inner];
            }
            if row == col {
                if !value.is_finite() || value <= threshold {
                    return None;
                }
                matrix[row * n + col] = value.sqrt();
            } else {
                matrix[row * n + col] = value / matrix[col * n + col];
            }
        }
        for col in (row + 1)..n {
            matrix[row * n + col] = 0.0;
        }
    }
    Some(())
}

pub(crate) fn build_background(
    filled: &FilledPair,
    condition: ConditionMode,
) -> Result<(Vec<f64>, usize), String> {
    let n = filled.z.len();
    if n == 0 || filled.g_i.len() != n || filled.g_j.len() != n || filled.imputed.len() != n {
        return Err("filled pair vector lengths are inconsistent or empty".to_string());
    }
    let n_columns = match condition {
        ConditionMode::AddDom => 5,
        ConditionMode::Genotype => 8,
    };
    let mut out = Vec::with_capacity(n * n_columns);
    for row in 0..n {
        let gi_raw = filled.g_i[row];
        let gj_raw = filled.g_j[row];
        if gi_raw > 2 || gj_raw > 2 || !(-1..=1).contains(&filled.z[row]) {
            return Err(format!("invalid filled pair state at sample {row}"));
        }
        let gi = gi_raw as f64;
        let gj = gj_raw as f64;
        let hi = if gi_raw == 1 { 1.0 } else { 0.0 };
        let hj = if gj_raw == 1 { 1.0 } else { 0.0 };
        out.extend([gi, gj, hi, hj]);
        if condition == ConditionMode::Genotype {
            out.extend([gi * gj, gi * hj, hi * gj]);
        }
        // H is last in both modes. Together, eta*H + theta*Z allows
        // independent cis/trans deviations from the remaining background.
        out.push(hi * hj);
    }
    Ok((out, n_columns))
}

/// Trait-level work shared by every selected pair: whiten the response and
/// fixed-effect design once, then reuse them in each conditional fit.
pub(crate) struct PreparedConditionalFit {
    n: usize,
    y_w: Vec<f64>,
    c_w: Vec<Vec<f64>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ConditionalFitTimings {
    pub(crate) joint_design_whiten: Duration,
    pub(crate) nuisance_projection: Duration,
    pub(crate) phase_test: Duration,
    pub(crate) main_effects: Duration,
    pub(crate) phase_effects: Duration,
}

pub(crate) fn prepare_conditional_fit(
    y: &[f64],
    c: &[f64],
    q_c: usize,
    metric: &CovarianceMetric,
) -> Result<PreparedConditionalFit, String> {
    let n = y.len();
    if n == 0 || metric.n() != n || q_c == 0 {
        return Err("conditional model has an empty sample set or invalid dimensions".to_string());
    }
    if c.len() != n.saturating_mul(q_c) {
        return Err("conditional model design shape mismatch".to_string());
    }
    if y.iter().chain(c).any(|value| !value.is_finite()) {
        return Err("conditional model input contains non-finite values".to_string());
    }

    let y_w = metric.whiten(y)?;
    let c_w = centered_design_columns(c, n, q_c)
        .into_iter()
        .map(|column| metric.whiten(&column))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PreparedConditionalFit { n, y_w, c_w })
}

#[cfg(test)]
pub(crate) fn fit_conditional(
    y: &[f64],
    c: &[f64],
    q_c: usize,
    b: &[f64],
    q_b: usize,
    z: &[f64],
    metric: &CovarianceMetric,
    inference: InferenceKind,
    shared_background_column: Option<usize>,
) -> Result<ConditionalFit, String> {
    let prepared = prepare_conditional_fit(y, c, q_c, metric)?;
    fit_conditional_prepared(
        &prepared,
        b,
        q_b,
        z,
        metric,
        inference,
        shared_background_column,
    )
}

pub(crate) fn fit_conditional_prepared(
    prepared: &PreparedConditionalFit,
    b: &[f64],
    q_b: usize,
    z: &[f64],
    metric: &CovarianceMetric,
    inference: InferenceKind,
    shared_background_column: Option<usize>,
) -> Result<ConditionalFit, String> {
    fit_conditional_prepared_inner(
        prepared,
        b,
        q_b,
        z,
        metric,
        inference,
        shared_background_column,
        false,
        &mut ConditionalFitTimings::default(),
    )
}

pub(crate) fn fit_conditional_prepared_profiled(
    prepared: &PreparedConditionalFit,
    b: &[f64],
    q_b: usize,
    z: &[f64],
    metric: &CovarianceMetric,
    inference: InferenceKind,
    shared_background_column: Option<usize>,
    timings: &mut ConditionalFitTimings,
) -> Result<ConditionalFit, String> {
    fit_conditional_prepared_inner(
        prepared,
        b,
        q_b,
        z,
        metric,
        inference,
        shared_background_column,
        true,
        timings,
    )
}

pub(crate) fn fit_conditional_prepared_whitened(
    prepared: &PreparedConditionalFit,
    design: WhitenedPairDesign,
    q_b: usize,
    inference: InferenceKind,
    shared_background_column: Option<usize>,
) -> Result<ConditionalFit, String> {
    fit_conditional_prepared_whitened_inner(
        prepared,
        design.background_columns,
        q_b,
        design.phase,
        inference,
        shared_background_column,
        false,
        &mut ConditionalFitTimings::default(),
    )
}

pub(crate) fn fit_conditional_prepared_whitened_profiled(
    prepared: &PreparedConditionalFit,
    design: WhitenedPairDesign,
    q_b: usize,
    inference: InferenceKind,
    shared_background_column: Option<usize>,
    timings: &mut ConditionalFitTimings,
) -> Result<ConditionalFit, String> {
    fit_conditional_prepared_whitened_inner(
        prepared,
        design.background_columns,
        q_b,
        design.phase,
        inference,
        shared_background_column,
        true,
        timings,
    )
}

#[allow(clippy::too_many_arguments)]
fn fit_conditional_prepared_inner(
    prepared: &PreparedConditionalFit,
    b: &[f64],
    q_b: usize,
    z: &[f64],
    metric: &CovarianceMetric,
    inference: InferenceKind,
    shared_background_column: Option<usize>,
    profile: bool,
    timings: &mut ConditionalFitTimings,
) -> Result<ConditionalFit, String> {
    let n = prepared.n;
    if n == 0 || metric.n() != n || q_b == 0 {
        return Err("conditional model has an empty sample set or invalid dimensions".to_string());
    }
    if b.len() != n.saturating_mul(q_b) || z.len() != n {
        return Err("conditional model design shape mismatch".to_string());
    }
    if shared_background_column.is_some_and(|index| index >= q_b) {
        return Err("shared-effect background column index is out of bounds".to_string());
    }
    if b.iter().chain(z).any(|value| !value.is_finite()) {
        return Err("conditional model input contains non-finite values".to_string());
    }

    let joint_design_whiten_started = profile.then(Instant::now);
    let n_design_columns = q_b + 1;
    let mut pair_design = Vec::with_capacity(n.saturating_mul(n_design_columns));
    for row in 0..n {
        pair_design.extend_from_slice(&b[row * q_b..(row + 1) * q_b]);
        pair_design.push(z[row]);
    }
    let mut whitened_design = metric.whiten_row_major_columns(&pair_design, n_design_columns)?;
    let z_w = whitened_design
        .pop()
        .expect("joint pair design includes one phase column");
    let b_w = whitened_design;
    if let Some(started) = joint_design_whiten_started {
        timings.joint_design_whiten += started.elapsed();
    }
    fit_conditional_prepared_whitened_inner(
        prepared,
        b_w,
        q_b,
        z_w,
        inference,
        shared_background_column,
        profile,
        timings,
    )
}

#[allow(clippy::too_many_arguments)]
fn fit_conditional_prepared_whitened_inner(
    prepared: &PreparedConditionalFit,
    b_w: Vec<Vec<f64>>,
    q_b: usize,
    z_w: Vec<f64>,
    inference: InferenceKind,
    shared_background_column: Option<usize>,
    profile: bool,
    timings: &mut ConditionalFitTimings,
) -> Result<ConditionalFit, String> {
    let n = prepared.n;
    if n == 0 || q_b == 0 {
        return Err("conditional model has an empty sample set or invalid dimensions".to_string());
    }
    if z_w.len() != n || b_w.len() != q_b || b_w.iter().any(|column| column.len() != n) {
        return Err("whitened conditional design shape mismatch".to_string());
    }
    if shared_background_column.is_some_and(|index| index >= q_b) {
        return Err("shared-effect background column index is out of bounds".to_string());
    }
    if b_w
        .iter()
        .flatten()
        .chain(&z_w)
        .any(|value| !value.is_finite())
    {
        return Err("whitened conditional design contains non-finite values".to_string());
    }

    let y_w = &prepared.y_w;
    let c_w = &prepared.c_w;
    let nuisance_projection_started = profile.then(Instant::now);
    let shared_column = shared_background_column.map(|index| b_w[index].clone());
    let shared_norm2 = shared_column.as_ref().map(|column| dot(column, column));
    let mut nuisance_columns = c_w.clone();
    nuisance_columns.extend(
        b_w.iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != shared_background_column)
            .map(|(_, column)| column.clone()),
    );
    let (base_basis, base_factor) = orthonormal_basis_with_factor(nuisance_columns);
    let mut projection_values = vec![y_w.to_vec(), z_w.clone()];
    if let Some(column) = shared_column {
        projection_values.push(column);
    }
    project_off_many(&mut projection_values, &base_basis);
    let mut projected_values = projection_values.into_iter();
    let mut y_after_base = projected_values
        .next()
        .expect("joint projection includes the response");
    let mut z_after_base = projected_values
        .next()
        .expect("joint projection includes the phase vector");
    let shared_residual = shared_norm2.map(|_| {
        projected_values
            .next()
            .expect("joint projection includes the shared background column")
    });
    // Orthogonalize H after the other background columns. Center the response
    // against that same base first, so roundoff from a large intercept cannot
    // leak into the separate cis/trans effects.
    let shared_projection = shared_norm2
        .zip(shared_residual)
        .and_then(|(norm2, residual)| {
            let information = dot(&residual, &residual);
            if !information.is_finite() || information <= norm2 * 1.0e-24 {
                return None;
            }
            let eta_without_z = dot(&residual, &y_after_base) / information;
            let z_loading = dot(&residual, &z_after_base) / information;
            let mut unit = residual;
            let scale = information.sqrt();
            for value in &mut unit {
                *value /= scale;
            }
            Some((information, eta_without_z, z_loading, unit))
        });
    let (y_after_nuisance, z_after) = if let Some((_, _, _, h_unit)) = shared_projection.as_ref() {
        let y_loading = dot(&y_after_base, h_unit);
        let z_loading = dot(&z_after_base, h_unit);
        for (value, &h) in y_after_base.iter_mut().zip(h_unit) {
            *value -= y_loading * h;
        }
        for (value, &h) in z_after_base.iter_mut().zip(h_unit) {
            *value -= z_loading * h;
        }
        (y_after_base, z_after_base)
    } else {
        (y_after_base, z_after_base)
    };
    let rank = base_basis.len() + usize::from(shared_projection.is_some()) + 1;
    if let Some(started) = nuisance_projection_started {
        timings.nuisance_projection += started.elapsed();
    }

    let phase_test_started = profile.then(Instant::now);
    let z_norm = dot(&z_w, &z_w).sqrt();
    let phase_information = dot(&z_after, &z_after);
    if !phase_information.is_finite() || phase_information <= (z_norm * 1.0e-12).powi(2) {
        if let Some(started) = phase_test_started {
            timings.phase_test += started.elapsed();
        }
        return Ok(non_testable(FitStatus::UnidentifiablePhase, rank - 1, 0));
    }
    let residual_df = n.saturating_sub(rank);
    if residual_df == 0 {
        if let Some(started) = phase_test_started {
            timings.phase_test += started.elapsed();
        }
        return Ok(non_testable(
            FitStatus::InsufficientDegreesOfFreedom,
            rank,
            residual_df,
        ));
    }

    let score = dot(&z_after, &y_after_nuisance);
    let beta_z = score / phase_information;
    let mut full_residual = y_after_nuisance;
    for (residual, &phase) in full_residual.iter_mut().zip(&z_after) {
        *residual -= beta_z * phase;
    }
    let rss = dot(&full_residual, &full_residual);
    let se_z = match inference {
        InferenceKind::LinearModel => {
            if !(rss.is_finite() && rss > 0.0) {
                if let Some(started) = phase_test_started {
                    timings.phase_test += started.elapsed();
                }
                return Ok(non_testable(
                    FitStatus::ZeroResidualVariance,
                    rank,
                    residual_df,
                ));
            }
            (rss / residual_df as f64 / phase_information).sqrt()
        }
        InferenceKind::FixedCovariance => (1.0 / phase_information).sqrt(),
    };
    if !(beta_z.is_finite() && se_z.is_finite() && se_z > 0.0) {
        if let Some(started) = phase_test_started {
            timings.phase_test += started.elapsed();
        }
        return Ok(non_testable(
            FitStatus::UnidentifiablePhase,
            rank,
            residual_df,
        ));
    }
    let statistic = match inference {
        InferenceKind::LinearModel => (beta_z / se_z).powi(2),
        InferenceKind::FixedCovariance => beta_z * beta_z / (se_z * se_z),
    };
    let pwald1 = match inference {
        InferenceKind::LinearModel => {
            crate::linalg::student_t_p_two_sided(statistic.sqrt(), residual_df as i32)
        }
        InferenceKind::FixedCovariance => crate::linalg::chi2_sf_df1(statistic),
    };
    if let Some(started) = phase_test_started {
        timings.phase_test += started.elapsed();
    }
    let mut beta_shared = f64::NAN;
    let mut se_shared = f64::NAN;
    let mut beta_cis = f64::NAN;
    let mut se_cis = f64::NAN;
    let mut beta_trans = f64::NAN;
    let mut se_trans = f64::NAN;
    let shared_effect_identifiable = shared_projection.is_some();
    let mut phase_wald_statistic = f64::NAN;
    let mut pwald2 = f64::NAN;
    let phase_effects_started = profile.then(Instant::now);
    if let Some((information_h, eta_without_z, z_loading, _)) = shared_projection {
        let scale = match inference {
            InferenceKind::LinearModel => rss / residual_df as f64,
            InferenceKind::FixedCovariance => 1.0,
        };
        let var_eta_without_z = scale / information_h;
        let var_z = se_z * se_z;
        phase_wald_statistic = eta_without_z.powi(2) / var_eta_without_z + beta_z.powi(2) / var_z;
        pwald2 = match inference {
            InferenceKind::LinearModel => (residual_df as f64
                / (residual_df as f64 + phase_wald_statistic))
                .powf(residual_df as f64 / 2.0),
            InferenceKind::FixedCovariance => crate::linalg::chi2_sf(phase_wald_statistic, 2.0),
        };
        beta_shared = eta_without_z - z_loading * beta_z;
        se_shared = (scale / information_h + z_loading.powi(2) * var_z).sqrt();
        beta_cis = beta_shared + beta_z;
        beta_trans = beta_shared - beta_z;
        // These include Cov(eta,theta) = -z_loading * Var(theta).
        se_cis = (scale / information_h + (z_loading - 1.0).powi(2) * var_z).sqrt();
        se_trans = (scale / information_h + (z_loading + 1.0).powi(2) * var_z).sqrt();
    }
    if let Some(started) = phase_effects_started {
        timings.phase_effects += started.elapsed();
    }
    let mut beta_a1 = f64::NAN;
    let mut beta_d1 = f64::NAN;
    let mut beta_a2 = f64::NAN;
    let mut beta_d2 = f64::NAN;
    let main_effects_started = profile.then(Instant::now);
    if b_w.len() >= 4 {
        let target_indices = (c_w.len()..c_w.len() + 4).collect::<Vec<_>>();
        let basis_coefficients = shared_background_column
            .filter(|&shared_index| shared_index >= 4)
            .filter(|_| shared_effect_identifiable && beta_shared.is_finite())
            .and_then(|shared_index| {
                let factor = base_factor.as_ref()?;
                coefficients_from_mgs_factor_with_extra_terms(
                    &base_basis,
                    factor,
                    y_w,
                    &[
                        (b_w[shared_index].as_slice(), beta_shared),
                        (z_w.as_slice(), beta_z),
                    ],
                    &target_indices,
                )
            });
        let coefficients = basis_coefficients.unwrap_or_else(|| {
            let mut full_design = Vec::with_capacity(c_w.len() + b_w.len() + 1);
            full_design.extend(c_w.iter().map(Vec::as_slice));
            full_design.extend(b_w.iter().map(Vec::as_slice));
            full_design.push(z_w.as_slice());
            design_coefficients_for_targets(&full_design, y_w, &target_indices)
        });
        beta_a1 = coefficients[0];
        beta_a2 = coefficients[1];
        beta_d1 = coefficients[2];
        beta_d2 = coefficients[3];
    }
    if let Some(started) = main_effects_started {
        timings.main_effects += started.elapsed();
    }
    Ok(ConditionalFit {
        status: FitStatus::Testable,
        beta_z,
        se_z,
        beta_cis_minus_trans: 2.0 * beta_z,
        se_cis_minus_trans: 2.0 * se_z,
        beta_shared,
        se_shared,
        beta_cis,
        se_cis,
        beta_trans,
        se_trans,
        beta_a1,
        beta_d1,
        beta_a2,
        beta_d2,
        shared_effect_identifiable,
        statistic,
        pwald1,
        phase_wald_statistic,
        pwald2,
        fixed_effect_rank: rank,
        residual_df,
    })
}

fn non_testable(status: FitStatus, fixed_effect_rank: usize, residual_df: usize) -> ConditionalFit {
    ConditionalFit {
        status,
        beta_z: f64::NAN,
        se_z: f64::NAN,
        beta_cis_minus_trans: f64::NAN,
        se_cis_minus_trans: f64::NAN,
        beta_shared: f64::NAN,
        se_shared: f64::NAN,
        beta_cis: f64::NAN,
        se_cis: f64::NAN,
        beta_trans: f64::NAN,
        se_trans: f64::NAN,
        beta_a1: f64::NAN,
        beta_d1: f64::NAN,
        beta_a2: f64::NAN,
        beta_d2: f64::NAN,
        shared_effect_identifiable: false,
        statistic: f64::NAN,
        pwald1: f64::NAN,
        phase_wald_statistic: f64::NAN,
        pwald2: f64::NAN,
        fixed_effect_rank,
        residual_df,
    }
}

fn partial_regression_coefficient(target: &[f64], response: &[f64], others: Vec<Vec<f64>>) -> f64 {
    if target.len() != response.len()
        || target
            .iter()
            .chain(response)
            .any(|value| !value.is_finite())
    {
        return f64::NAN;
    }
    let target_norm2 = dot(target, target);
    if !(target_norm2.is_finite() && target_norm2 > 0.0) {
        return f64::NAN;
    }
    let basis = orthonormal_basis(others);
    let target_residual = project_off(target, &basis);
    let information = dot(&target_residual, &target_residual);
    if !(information.is_finite() && information > target_norm2 * 1.0e-24) {
        return f64::NAN;
    }
    let response_residual = project_off(response, &basis);
    let beta = dot(&target_residual, &response_residual) / information;
    if beta.is_finite() {
        beta
    } else {
        f64::NAN
    }
}

fn design_coefficients_for_targets<C: AsRef<[f64]>>(
    design: &[C],
    response: &[f64],
    targets: &[usize],
) -> Vec<f64> {
    if let Some(coefficients) = pivoted_qr_coefficients(design, response) {
        return targets
            .iter()
            .map(|&target| coefficients.get(target).copied().unwrap_or(f64::NAN))
            .collect();
    }
    targets
        .iter()
        .map(|&target| {
            let Some(column) = design.get(target) else {
                return f64::NAN;
            };
            let others = design
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != target)
                .map(|(_, column)| column.as_ref().to_vec())
                .collect::<Vec<_>>();
            partial_regression_coefficient(column.as_ref(), response, others)
        })
        .collect()
}

#[cfg(test)]
fn coefficients_from_basis_with_extra_terms(
    basis: &[Vec<f64>],
    design: &[&[f64]],
    response: &[f64],
    extra_terms: &[(&[f64], f64)],
    targets: &[usize],
) -> Option<Vec<f64>> {
    let n = response.len();
    if n == 0
        || basis.is_empty()
        || basis.len() != design.len()
        || response.iter().any(|value| !value.is_finite())
        || basis
            .iter()
            .any(|column| column.len() != n || column.iter().any(|value| !value.is_finite()))
        || design
            .iter()
            .any(|column| column.len() != n || column.iter().any(|value| !value.is_finite()))
        || extra_terms.iter().any(|(column, coefficient)| {
            column.len() != n
                || column.iter().any(|value| !value.is_finite())
                || !coefficient.is_finite()
        })
    {
        return None;
    }

    // The basis spans the nuisance design. Projecting each original column
    // and the response into those coordinates gives a small, equivalent
    // least-squares system without refactorizing all sample-sized columns.
    let coordinate_design = design
        .iter()
        .map(|column| {
            basis
                .iter()
                .map(|basis_column| dot(basis_column, column))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let coordinate_response = basis
        .iter()
        .map(|basis_column| {
            extra_terms.iter().fold(
                dot(basis_column, response),
                |value, (column, coefficient)| value - coefficient * dot(basis_column, column),
            )
        })
        .collect::<Vec<_>>();
    let coefficients = pivoted_qr_coefficients(&coordinate_design, &coordinate_response)?;
    Some(
        targets
            .iter()
            .map(|&target| coefficients.get(target).copied().unwrap_or(f64::NAN))
            .collect(),
    )
}

fn coefficients_from_mgs_factor_with_extra_terms(
    basis: &[Vec<f64>],
    factor: &PivotedMgsFactor,
    response: &[f64],
    extra_terms: &[(&[f64], f64)],
    targets: &[usize],
) -> Option<Vec<f64>> {
    let n = response.len();
    let n_columns = factor.pivot_order.len();
    if n == 0
        || n_columns == 0
        || basis.len() != n_columns
        || factor.source_norms.len() != n_columns
        || factor.r_by_source.len() != n_columns
        || factor.r_by_source.iter().any(|row| row.len() != n_columns)
        || factor
            .source_norms
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        || basis
            .iter()
            .any(|column| column.len() != n || column.iter().any(|value| !value.is_finite()))
        || response.iter().any(|value| !value.is_finite())
        || extra_terms.iter().any(|(column, coefficient)| {
            column.len() != n
                || column.iter().any(|value| !value.is_finite())
                || !coefficient.is_finite()
        })
    {
        return None;
    }

    // The MGS pass already formed R while orthogonalizing the original
    // nuisance columns. Reuse it to back-solve instead of rebuilding QᵀA and
    // factoring that small matrix for every candidate.
    let coordinate_response = basis
        .iter()
        .map(|basis_column| {
            extra_terms.iter().fold(
                dot(basis_column, response),
                |value, (column, coefficient)| value - coefficient * dot(basis_column, column),
            )
        })
        .collect::<Vec<_>>();
    let mut pivoted_coefficients = vec![0.0; n_columns];
    for row in (0..n_columns).rev() {
        let source_column = factor.pivot_order[row];
        if source_column >= n_columns {
            return None;
        }
        let mut value = coordinate_response[row];
        for column in (row + 1)..n_columns {
            let later_source = factor.pivot_order[column];
            if later_source >= n_columns {
                return None;
            }
            value -= factor.r_by_source[row][later_source] * pivoted_coefficients[column];
        }
        let diagonal = factor.r_by_source[row][source_column];
        if !(diagonal.is_finite() && diagonal > 0.0) {
            return None;
        }
        pivoted_coefficients[row] = value / diagonal;
        if !pivoted_coefficients[row].is_finite() {
            return None;
        }
    }

    let mut coefficients = vec![f64::NAN; n_columns];
    for (pivoted_index, &source_column) in factor.pivot_order.iter().enumerate() {
        if source_column >= n_columns {
            return None;
        }
        coefficients[source_column] =
            pivoted_coefficients[pivoted_index] / factor.source_norms[source_column];
    }
    Some(
        targets
            .iter()
            .map(|&target| coefficients.get(target).copied().unwrap_or(f64::NAN))
            .collect(),
    )
}

/// Full-rank, column-pivoted modified Gram-Schmidt solve. Scaling the source
/// columns before pivoting makes rank decisions independent of predictor units;
/// rank-deficient designs return `None` so callers can preserve their per-term
/// estimability behavior with a fallback.
fn pivoted_qr_coefficients<C: AsRef<[f64]>>(columns: &[C], response: &[f64]) -> Option<Vec<f64>> {
    if columns.is_empty()
        || response.is_empty()
        || response.iter().any(|value| !value.is_finite())
        || columns.iter().any(|column| {
            let column = column.as_ref();
            column.len() != response.len() || column.iter().any(|value| !value.is_finite())
        })
        || columns.len() > response.len()
    {
        return None;
    }
    let n_columns = columns.len();
    let mut residual_columns = Vec::with_capacity(n_columns);
    let mut source_norms = Vec::with_capacity(n_columns);
    for column in columns {
        let column = column.as_ref();
        let scale = column
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        if !(scale.is_finite() && scale > 0.0) {
            return None;
        }
        let mut normalized = column.iter().map(|value| value / scale).collect::<Vec<_>>();
        let norm = dot(&normalized, &normalized).sqrt();
        let source_norm = scale * norm;
        if !(norm.is_finite() && norm > 0.0 && source_norm.is_finite() && source_norm > 0.0) {
            return None;
        }
        for value in &mut normalized {
            *value /= norm;
        }
        residual_columns.push(normalized);
        source_norms.push(source_norm);
    }

    let mut permutation = (0..n_columns).collect::<Vec<_>>();
    let mut upper = vec![vec![0.0; n_columns]; n_columns];
    let mut basis = Vec::with_capacity(n_columns);
    const RANK_CUTOFF: f64 = 1.0e-12;
    for pivot_column in 0..n_columns {
        let (best_index, best_norm2) = (pivot_column..n_columns)
            .map(|index| {
                (
                    index,
                    dot(&residual_columns[index], &residual_columns[index]),
                )
            })
            .max_by(|left, right| left.1.total_cmp(&right.1))?;
        let best_norm = best_norm2.max(0.0).sqrt();
        if !(best_norm.is_finite() && best_norm > RANK_CUTOFF) {
            return None;
        }
        residual_columns.swap(pivot_column, best_index);
        source_norms.swap(pivot_column, best_index);
        permutation.swap(pivot_column, best_index);
        for row in 0..pivot_column {
            upper[row].swap(pivot_column, best_index);
        }

        let mut q = residual_columns[pivot_column].clone();
        for value in &mut q {
            *value /= best_norm;
        }
        upper[pivot_column][pivot_column] = best_norm;
        basis.push(q);
        let q = basis.last().expect("pivoted QR appends its basis column");
        for column in (pivot_column + 1)..n_columns {
            let mut coefficient = 0.0;
            for _ in 0..2 {
                let correction = dot(q, &residual_columns[column]);
                coefficient += correction;
                for (value, &basis_value) in residual_columns[column].iter_mut().zip(q) {
                    *value -= correction * basis_value;
                }
            }
            upper[pivot_column][column] = coefficient;
        }
    }

    let mut pivoted_coefficients = vec![0.0; n_columns];
    for row in (0..n_columns).rev() {
        let mut value = dot(&basis[row], response);
        for column in (row + 1)..n_columns {
            value -= upper[row][column] * pivoted_coefficients[column];
        }
        let diagonal = upper[row][row];
        if !(diagonal.is_finite() && diagonal > 0.0) {
            return None;
        }
        pivoted_coefficients[row] = value / diagonal;
    }

    let mut coefficients = vec![0.0; n_columns];
    for pivoted_index in 0..n_columns {
        let coefficient = pivoted_coefficients[pivoted_index] / source_norms[pivoted_index];
        if !coefficient.is_finite() {
            return None;
        }
        coefficients[permutation[pivoted_index]] = coefficient;
    }
    Some(coefficients)
}

/// Return column vectors from a row-major design and center every column after
/// the first. CT's first fixed-effect column is the intercept, so this affine
/// transformation preserves the design span while avoiding large offsets.
pub(crate) fn centered_design_columns(matrix: &[f64], n: usize, n_columns: usize) -> Vec<Vec<f64>> {
    (0..n_columns)
        .map(|column| {
            let mut values = (0..n)
                .map(|row| matrix[row * n_columns + column])
                .collect::<Vec<_>>();
            if column > 0 && !values.is_empty() {
                let origin = values[0];
                for value in &mut values {
                    *value -= origin;
                }
                let scale = values
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0_f64, f64::max);
                if scale > 0.0 && scale.is_finite() {
                    let mean_offset = scale * values.iter().map(|value| value / scale).sum::<f64>()
                        / values.len() as f64;
                    for value in &mut values {
                        *value -= mean_offset;
                    }
                }
            }
            values
        })
        .collect()
}

pub(crate) fn orthonormal_basis(residual_columns: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    orthonormal_basis_impl(residual_columns, false).0
}

#[derive(Debug)]
struct PivotedMgsFactor {
    // R is indexed by original source columns; pivot_order maps triangular
    // solve positions back to those source columns.
    r_by_source: Vec<Vec<f64>>,
    source_norms: Vec<f64>,
    pivot_order: Vec<usize>,
}

fn orthonormal_basis_with_factor(
    residual_columns: Vec<Vec<f64>>,
) -> (Vec<Vec<f64>>, Option<PivotedMgsFactor>) {
    orthonormal_basis_impl(residual_columns, true)
}

fn orthonormal_basis_impl(
    mut residual_columns: Vec<Vec<f64>>,
    retain_factor: bool,
) -> (Vec<Vec<f64>>, Option<PivotedMgsFactor>) {
    if residual_columns.is_empty() {
        return (Vec::new(), None);
    }
    let n_columns = residual_columns.len();
    let mut source_indices = (0..n_columns).collect::<Vec<_>>();
    let mut source_norms = retain_factor.then(|| vec![f64::NAN; n_columns]);
    let mut r_by_source = retain_factor.then(|| vec![vec![0.0; n_columns]; n_columns]);
    let mut pivot_order = Vec::with_capacity(n_columns);

    // Scale every source column before pivoting. Rank is a property of the
    // design span, so the decision must not depend on a covariate's units.
    for (source_index, column) in residual_columns.iter_mut().enumerate() {
        let scale = column
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        if !(scale.is_finite() && scale > 0.0) {
            column.fill(0.0);
            continue;
        }
        for value in column.iter_mut() {
            *value /= scale;
        }
        let norm = dot(column, column).sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            column.fill(0.0);
            continue;
        }
        if let Some(source_norms) = source_norms.as_mut() {
            source_norms[source_index] = scale * norm;
        }
        for value in column.iter_mut() {
            *value /= norm;
        }
    }

    let cutoff = 1.0e-12;
    let mut basis = Vec::with_capacity(n_columns);
    loop {
        let best = residual_columns
            .iter()
            .enumerate()
            .map(|(index, column)| (index, dot(column, column)))
            .max_by(|left, right| left.1.total_cmp(&right.1));
        let Some((best_index, best_norm2)) = best else {
            break;
        };
        let best_norm = best_norm2.max(0.0).sqrt();
        if !best_norm.is_finite() || best_norm <= cutoff {
            break;
        }
        let mut column = residual_columns.swap_remove(best_index);
        let source_index = source_indices.swap_remove(best_index);
        let basis_index = basis.len();
        pivot_order.push(source_index);
        for value in column.iter_mut() {
            *value /= best_norm;
        }
        if let Some(r_by_source) = r_by_source.as_mut() {
            r_by_source[basis_index][source_index] = best_norm;
        }
        basis.push(column);
        let last = basis.last().expect("basis has the appended column");
        for (other_index, other) in residual_columns.iter_mut().enumerate() {
            let mut coefficient = 0.0;
            for _ in 0..2 {
                let correction = dot(other, last);
                coefficient += correction;
                for (value, &q) in other.iter_mut().zip(last) {
                    *value -= correction * q;
                }
            }
            if let Some(r_by_source) = r_by_source.as_mut() {
                r_by_source[basis_index][source_indices[other_index]] = coefficient;
            }
        }
    }

    let factor = match (source_norms, r_by_source) {
        (Some(source_norms), Some(r_by_source)) if basis.len() == n_columns => {
            Some(PivotedMgsFactor {
                r_by_source,
                source_norms,
                pivot_order,
            })
        }
        _ => None,
    };
    (basis, factor)
}

pub(crate) fn project_off(values: &[f64], basis: &[Vec<f64>]) -> Vec<f64> {
    let mut out = values.to_vec();
    for _ in 0..2 {
        for column in basis {
            let coefficient = dot(&out, column);
            for (value, &q) in out.iter_mut().zip(column) {
                *value -= coefficient * q;
            }
        }
    }
    out
}

fn project_off_many(values: &mut [Vec<f64>], basis: &[Vec<f64>]) {
    if values.is_empty() || basis.is_empty() {
        return;
    }
    let n = values[0].len();
    debug_assert!(values.iter().all(|column| column.len() == n));
    debug_assert!(basis.iter().all(|column| column.len() == n));
    // `orthonormal_basis` has already reorthogonalized its columns. Compute
    // all Qᵀx coefficients from the same original x, then apply x - Q(Qᵀx)
    // in one pass. This is the standard orthogonal projector and avoids a
    // second full traversal over every sample vector.
    let mut coefficients = vec![vec![0.0; basis.len()]; values.len()];
    for (basis_index, basis_column) in basis.iter().enumerate() {
        for (value_index, values_column) in values.iter().enumerate() {
            coefficients[value_index][basis_index] = dot(values_column, basis_column);
        }
    }
    for (value_index, values_column) in values.iter_mut().enumerate() {
        for (coefficient, basis_column) in coefficients[value_index].iter().zip(basis) {
            for (value, &basis_value) in values_column.iter_mut().zip(basis_column) {
                *value -= coefficient * basis_value;
            }
        }
    }
}

#[inline]
fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    use super::cholesky_blocked;
    use super::{
        build_background, fit_conditional, fit_conditional_prepared,
        fit_conditional_prepared_whitened, partial_regression_coefficient, pivoted_qr_coefficients,
        prepare_conditional_fit, whiten_pair_design_batch, whiten_pair_design_batch_profiled,
        ConditionMode, CovarianceMetric, FitStatus, InferenceKind, PairDesign,
    };
    use crate::ct::packed::FilledPair;
    use nalgebra::DMatrix;

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    #[test]
    fn blocked_cholesky_recovers_known_lower_factor_across_tiles() {
        let n = 257;
        let mut expected = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..=row {
                expected[row * n + col] = if row == col {
                    1.4 + (row % 7) as f64 * 0.03
                } else {
                    ((row * 37 + col * 19) % 29) as f64 * 0.0005 - 0.007
                };
            }
        }
        let mut matrix = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..=row {
                let value = (0..=col)
                    .map(|inner| expected[row * n + inner] * expected[col * n + inner])
                    .sum::<f64>();
                matrix[row * n + col] = value;
                matrix[col * n + row] = value;
            }
        }
        let scale = (0..n)
            .map(|i| matrix[i * n + i].abs())
            .fold(0.0_f64, f64::max);
        let threshold = scale * f64::EPSILON * n as f64 * 8.0;

        cholesky_blocked(&mut matrix, n, threshold)
            .expect("constructed matrix must be positive definite");
        for row in 0..n {
            for col in 0..=row {
                assert!(
                    (matrix[row * n + col] - expected[row * n + col]).abs() < 2.0e-11,
                    "factor mismatch at ({row}, {col}): {} vs {}",
                    matrix[row * n + col],
                    expected[row * n + col]
                );
            }
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    #[test]
    fn blocked_cholesky_keeps_relative_pivot_rejection() {
        let n = 129;
        let threshold = f64::EPSILON * n as f64 * 8.0;
        let mut near_singular = vec![0.0; n * n];
        for i in 0..n {
            near_singular[i * n + i] = 1.0;
        }
        near_singular[(n - 1) * n + (n - 1)] = threshold * 0.5;
        assert!(cholesky_blocked(&mut near_singular, n, threshold).is_none());

        let mut indefinite = vec![0.0; n * n];
        for i in 0..n {
            indefinite[i * n + i] = 1.0;
        }
        indefinite[(n - 1) * n + (n - 1)] = -0.25;
        assert!(cholesky_blocked(&mut indefinite, n, threshold).is_none());
    }

    #[test]
    fn pivoted_qr_matches_independent_partial_regression_coefficients() {
        let n = 40;
        let columns = vec![
            vec![1.0; n],
            (0..n).map(|i| i as f64).collect(),
            (0..n).map(|i| (i * i) as f64).collect(),
            (0..n).map(|i| ((i * 13 % 17) as f64) / 17.0).collect(),
            (0..n).map(|i| ((i as f64) * 0.37).sin()).collect(),
        ];
        let response = (0..n)
            .map(|i| {
                0.75 + 0.2 * columns[1][i] - 0.04 * columns[2][i] + 1.1 * columns[3][i]
                    - 0.6 * columns[4][i]
            })
            .collect::<Vec<_>>();
        let observed = pivoted_qr_coefficients(&columns, &response)
            .expect("full-rank design must produce coefficients");
        assert_eq!(observed.len(), columns.len());
        for target in 0..columns.len() {
            let others = columns
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != target)
                .map(|(_, column)| column.clone())
                .collect();
            let reference = partial_regression_coefficient(&columns[target], &response, others);
            assert!(
                (observed[target] - reference).abs() < 1.0e-9,
                "target={target}: pivoted QR={}, partial regression={reference}",
                observed[target]
            );
        }
    }

    #[test]
    fn target_coefficients_accept_borrowed_design_columns() {
        let n = 40;
        let columns = vec![
            vec![1.0; n],
            (0..n).map(|i| i as f64).collect(),
            (0..n).map(|i| (i * i) as f64).collect(),
            (0..n).map(|i| ((i * 13 % 17) as f64) / 17.0).collect(),
            (0..n).map(|i| ((i as f64) * 0.37).sin()).collect(),
        ];
        let response = (0..n)
            .map(|i| {
                0.75 + 0.2 * columns[1][i] - 0.04 * columns[2][i] + 1.1 * columns[3][i]
                    - 0.6 * columns[4][i]
            })
            .collect::<Vec<_>>();
        let expected = pivoted_qr_coefficients(&columns, &response).unwrap();
        let borrowed = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let observed = super::design_coefficients_for_targets(&borrowed, &response, &[1, 3, 4]);

        for (target, actual) in [1, 3, 4].into_iter().zip(observed) {
            assert!(
                (actual - expected[target]).abs() < 1.0e-10,
                "target={target}: borrowed={actual}, owned={}",
                expected[target]
            );
        }
    }

    #[test]
    fn borrowed_target_coefficients_keep_partial_regression_fallback() {
        let n = 32;
        let x = (0..n).map(|i| i as f64 - 12.0).collect::<Vec<_>>();
        let columns = vec![
            vec![1.0; n],
            x.clone(),
            x,
            (0..n).map(|i| ((i as f64) * 0.41).sin()).collect(),
        ];
        let response = (0..n)
            .map(|i| 0.8 + 2.0 * columns[1][i] + 0.5 * columns[3][i])
            .collect::<Vec<_>>();
        let borrowed = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let targets = [1, 2, 3];
        let observed = super::design_coefficients_for_targets(&borrowed, &response, &targets);

        for (&target, actual) in targets.iter().zip(observed) {
            let others = borrowed
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != target)
                .map(|(_, column)| column.to_vec())
                .collect();
            let expected =
                super::partial_regression_coefficient(borrowed[target], &response, others);
            assert!(
                (actual.is_nan() && expected.is_nan()) || (actual - expected).abs() < 1.0e-10,
                "target={target}: borrowed={actual}, fallback={expected}"
            );
        }
    }

    #[test]
    fn nuisance_basis_coordinate_coefficients_match_full_design_qr() {
        let n = 48;
        let nuisance = vec![
            vec![1.0; n],
            (0..n).map(|i| ((i * 7 % 19) as f64 - 9.0) / 9.0).collect(),
            (0..n)
                .map(|i| ((i * i % 37) as f64 - 18.0) / 18.0)
                .collect(),
            (0..n).map(|i| ((i as f64) * 0.31).sin()).collect(),
        ];
        let shared = (0..n)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let phase = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let response = (0..n)
            .map(|i| {
                0.75 + 0.2 * nuisance[1][i] - 0.04 * nuisance[2][i]
                    + 1.1 * nuisance[3][i]
                    + 0.35 * shared[i]
                    - 0.6 * phase[i]
            })
            .collect::<Vec<_>>();
        let basis = super::orthonormal_basis(nuisance.clone());
        let nuisance_refs = nuisance.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let full_design = nuisance_refs
            .iter()
            .copied()
            .chain([shared.as_slice(), phase.as_slice()])
            .collect::<Vec<_>>();
        let targets = [1, 2, 3];
        let expected = super::design_coefficients_for_targets(&full_design, &response, &targets);
        let observed = super::coefficients_from_basis_with_extra_terms(
            &basis,
            &nuisance_refs,
            &response,
            &[(shared.as_slice(), 0.35), (phase.as_slice(), -0.6)],
            &targets,
        )
        .expect("full-rank nuisance basis must solve the projected system");

        for (target, (actual, expected)) in
            targets.into_iter().zip(observed.into_iter().zip(expected))
        {
            assert!(
                (actual - expected).abs() < 1.0e-9,
                "target={target}: basis={actual}, full QR={expected}"
            );
        }
    }

    #[test]
    fn nuisance_mgs_factor_coefficients_match_full_design_qr() {
        let n = 72;
        let nuisance = vec![
            vec![1.0; n],
            (0..n)
                .map(|i| ((i * 7 % 31) as f64 - 15.0) / 15.0)
                .collect(),
            (0..n).map(|i| ((i as f64) * 0.43).sin()).collect(),
            (0..n).map(|i| ((i as f64) * 0.17).cos()).collect(),
            (0..n)
                .map(|i| if i % 3 == 0 { -0.8 } else { 0.4 })
                .collect(),
        ];
        let shared = (0..n)
            .map(|i| if i % 5 < 2 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let phase = (0..n)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect::<Vec<_>>();
        let response = (0..n)
            .map(|i| {
                0.75 + 0.2 * nuisance[1][i] - 0.04 * nuisance[2][i]
                    + 1.1 * nuisance[3][i]
                    + 0.3 * nuisance[4][i]
                    + 0.35 * shared[i]
                    - 0.6 * phase[i]
            })
            .collect::<Vec<_>>();
        let nuisance_refs = nuisance.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let full_design = nuisance_refs
            .iter()
            .copied()
            .chain([shared.as_slice(), phase.as_slice()])
            .collect::<Vec<_>>();
        let targets = [1, 2, 3, 4];
        let expected = super::design_coefficients_for_targets(&full_design, &response, &targets);
        let (basis, factor) = super::orthonormal_basis_with_factor(nuisance.clone());
        let factor = factor.expect("full-rank nuisance design must have an MGS factor");
        let observed = super::coefficients_from_mgs_factor_with_extra_terms(
            &basis,
            &factor,
            &response,
            &[(shared.as_slice(), 0.35), (phase.as_slice(), -0.6)],
            &targets,
        )
        .expect("full-rank nuisance design must solve from its reusable QR factor");

        for (target, (actual, expected)) in
            targets.into_iter().zip(observed.into_iter().zip(expected))
        {
            assert!(
                (actual - expected).abs() < 1.0e-9,
                "target={target}: MGS factor={actual}, full QR={expected}"
            );
        }
    }

    #[test]
    fn nuisance_basis_coordinate_coefficients_reject_rank_deficient_bases() {
        let n = 16;
        let x = (0..n).map(|i| i as f64).collect::<Vec<_>>();
        let design = vec![vec![1.0; n], x.clone(), x];
        let basis = super::orthonormal_basis(design.clone());
        let refs = design.iter().map(Vec::as_slice).collect::<Vec<_>>();
        assert!(super::coefficients_from_basis_with_extra_terms(
            &basis,
            &refs,
            &vec![0.0; n],
            &[],
            &[1]
        )
        .is_none());
    }

    #[test]
    fn multi_vector_projection_matches_reorthogonalized_projection() {
        let n = 67;
        let basis = super::orthonormal_basis(vec![
            vec![1.0; n],
            (0..n)
                .map(|i| ((i * 11 % 23) as f64 - 11.0) / 11.0)
                .collect(),
            (0..n).map(|i| ((i as f64) * 0.27).sin()).collect(),
        ]);
        let inputs = (0..3)
            .map(|column| {
                (0..n)
                    .map(|row| ((row * 7 + column * 5) % 31) as f64 - 15.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let expected = inputs
            .iter()
            .map(|values| super::project_off(values, &basis))
            .collect::<Vec<_>>();
        let input_norms = inputs
            .iter()
            .map(|values| super::dot(values, values).sqrt().max(1.0))
            .collect::<Vec<_>>();
        let mut observed = inputs;
        super::project_off_many(&mut observed, &basis);

        for ((actual, expected), input_norm) in observed.iter().zip(&expected).zip(input_norms) {
            let difference_norm = actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(difference_norm <= 1.0e-12 * input_norm);
            for basis_column in &basis {
                assert!(super::dot(actual, basis_column).abs() <= 1.0e-12 * input_norm);
            }
        }
    }

    fn synthetic_design() -> (Vec<f64>, usize, Vec<f64>, usize, Vec<f64>, Vec<f64>) {
        let n = 24;
        let mut c = Vec::with_capacity(n * 2);
        let mut b = Vec::with_capacity(n * 4);
        let mut z = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let x = ((i * 7 % 19) as f64 - 9.0) / 9.0;
            let g_i = (i * 5 % 3) as f64;
            let g_j = ((i * 7 + 1) % 4 % 3) as f64;
            let h_i = if g_i == 1.0 { 1.0 } else { 0.0 };
            let h_j = if g_j == 1.0 { 1.0 } else { 0.0 };
            let phase = if i % 2 == 0 { 1.0 } else { -1.0 };
            let noise = ((i * 13 % 17) as f64 - 8.0) / 30.0;
            c.extend([1.0, x]);
            b.extend([g_i, g_j, h_i, h_j]);
            z.push(phase);
            y.push(
                0.7 + 0.23 * x + 0.18 * g_i - 0.11 * g_j
                    + 0.14 * h_i
                    + 0.06 * h_j
                    + 0.62 * phase
                    + noise,
            );
        }
        (c, 2, b, 4, z, y)
    }

    #[test]
    fn batched_pair_design_whitening_matches_individual_solves() {
        let n = 9;
        let q_b = 3;
        let mut lower = vec![0.0; n * n];
        for row in 0..n {
            lower[row * n + row] = 1.4 + row as f64 * 0.1;
            for col in 0..row {
                lower[row * n + col] = (row + col + 1) as f64 * 0.015;
            }
        }
        let mut covariance = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..n {
                for inner in 0..=row.min(col) {
                    covariance[row * n + col] += lower[row * n + inner] * lower[col * n + inner];
                }
            }
        }
        let metrics = [
            CovarianceMetric::identity(n),
            CovarianceMetric::Diagonal {
                n,
                standard_deviation: 1.7,
            },
            CovarianceMetric::from_dense(&covariance, n).unwrap(),
        ];
        let designs = (0..5)
            .map(|pair| {
                let mut background = Vec::with_capacity(n * q_b);
                let mut phase = Vec::with_capacity(n);
                for row in 0..n {
                    for col in 0..q_b {
                        background.push(((row * 11 + pair * 7 + col * 3) % 17) as f64 - 8.0);
                    }
                    phase.push(if (row + pair) % 2 == 0 { 1.0 } else { -1.0 });
                }
                PairDesign { background, phase }
            })
            .collect::<Vec<_>>();

        for metric in &metrics {
            let batched = whiten_pair_design_batch(&designs, q_b, metric).unwrap();
            assert_eq!(batched.len(), designs.len());
            let (profiled, timing) =
                whiten_pair_design_batch_profiled(&designs, q_b, metric, true).unwrap();
            assert_eq!(profiled.len(), batched.len());
            for (expected, observed) in batched.iter().zip(&profiled) {
                assert_eq!(expected.background_columns, observed.background_columns);
                assert_eq!(expected.phase, observed.phase);
            }
            let accounted = timing.matrix_assembly
                + timing.column_deduplication
                + timing.row_matrix_copy
                + timing.solve_or_scale
                + timing.column_unpack
                + timing.duplicate_restore_and_pair_pack;
            assert!(timing.total >= accounted);
            assert_eq!(timing.rhs_columns_total, designs.len() * (q_b + 1));
            assert!(timing.rhs_columns_unique <= timing.rhs_columns_total);
            if metric.has_cholesky_factor() {
                assert!(timing.solve_or_scale > std::time::Duration::ZERO);
            }
            let c = (0..n)
                .flat_map(|row| [1.0, ((row * 5 % 13) as f64 - 6.0) / 6.0])
                .collect::<Vec<_>>();
            let y = (0..n)
                .map(|row| ((row * 7 % 19) as f64 - 9.0) / 8.0)
                .collect::<Vec<_>>();
            let prepared = prepare_conditional_fit(&y, &c, 2, metric).unwrap();
            for (input, output) in designs.iter().zip(&batched) {
                for column in 0..q_b {
                    let raw = (0..n)
                        .map(|row| input.background[row * q_b + column])
                        .collect::<Vec<_>>();
                    let expected = metric.whiten(&raw).unwrap();
                    assert!(output.background_columns[column]
                        .iter()
                        .zip(expected)
                        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
                }
                let expected_phase = metric.whiten(&input.phase).unwrap();
                assert!(output
                    .phase
                    .iter()
                    .zip(expected_phase)
                    .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));

                let scalar_fit = fit_conditional_prepared(
                    &prepared,
                    &input.background,
                    q_b,
                    &input.phase,
                    metric,
                    InferenceKind::FixedCovariance,
                    Some(q_b - 1),
                )
                .unwrap();
                let batched_fit = fit_conditional_prepared_whitened(
                    &prepared,
                    output.clone(),
                    q_b,
                    InferenceKind::FixedCovariance,
                    Some(q_b - 1),
                )
                .unwrap();
                assert_eq!(scalar_fit.status, batched_fit.status);
                assert_eq!(
                    scalar_fit.shared_effect_identifiable,
                    batched_fit.shared_effect_identifiable
                );
                assert_eq!(scalar_fit.fixed_effect_rank, batched_fit.fixed_effect_rank);
                assert_eq!(scalar_fit.residual_df, batched_fit.residual_df);
                for (name, scalar, batch) in [
                    ("beta_z", scalar_fit.beta_z, batched_fit.beta_z),
                    ("se_z", scalar_fit.se_z, batched_fit.se_z),
                    ("beta_cis", scalar_fit.beta_cis, batched_fit.beta_cis),
                    ("se_cis", scalar_fit.se_cis, batched_fit.se_cis),
                    ("beta_trans", scalar_fit.beta_trans, batched_fit.beta_trans),
                    ("se_trans", scalar_fit.se_trans, batched_fit.se_trans),
                    ("beta_a1", scalar_fit.beta_a1, batched_fit.beta_a1),
                    ("beta_d1", scalar_fit.beta_d1, batched_fit.beta_d1),
                    ("beta_a2", scalar_fit.beta_a2, batched_fit.beta_a2),
                    ("beta_d2", scalar_fit.beta_d2, batched_fit.beta_d2),
                    ("pwald1", scalar_fit.pwald1, batched_fit.pwald1),
                    ("pwald2", scalar_fit.pwald2, batched_fit.pwald2),
                ] {
                    let equal = (scalar.is_nan() && batch.is_nan())
                        || (scalar - batch).abs()
                            <= 1.0e-10 * scalar.abs().max(batch.abs()).max(1.0);
                    assert!(equal, "{name}: scalar={scalar}, batch={batch}");
                }
            }
        }
    }

    #[test]
    fn row_major_column_dedup_preserves_first_seen_order_and_mapping() {
        let n = 3;
        let n_columns = 5;
        let matrix = vec![
            1.0, 4.0, 1.0, 7.0, 4.0, // row 0
            2.0, 5.0, 2.0, 8.0, 5.0, // row 1
            3.0, 6.0, 3.0, 9.0, 6.0, // row 2
        ];

        let (unique, mapping) = super::deduplicate_row_major_columns(matrix, n, n_columns).unwrap();
        assert_eq!(mapping, vec![0, 1, 0, 2, 1]);
        assert_eq!(unique, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn packed_column_fingerprints_keep_tail_values_and_exact_duplicate_mapping() {
        let n = 37;
        let n_columns = 4;
        let mut matrix = Vec::with_capacity(n * n_columns);
        for row in 0..n {
            let value = (row % 3) as f64;
            matrix.push(value);
            matrix.push(value);
            matrix.push(if row + 1 == n { 2.0 } else { value });
            matrix.push(if row % 2 == 0 { -1.0 } else { 1.0 });
        }

        let (packed, words_per_column) =
            super::pack_discrete_row_major_columns(&matrix, n, n_columns).unwrap();
        assert_eq!(words_per_column, 2);
        for word in 0..words_per_column {
            assert_eq!(packed[word * n_columns], packed[word * n_columns + 1]);
        }
        assert_ne!(
            packed[(words_per_column - 1) * n_columns],
            packed[(words_per_column - 1) * n_columns + 2]
        );

        let (unique, mapping) = super::deduplicate_row_major_columns(matrix, n, n_columns).unwrap();
        assert_eq!(mapping, vec![0, 0, 1, 2]);
        assert_eq!(unique.len(), n * 3);
    }

    #[test]
    fn duplicate_pair_columns_whiten_to_the_same_restored_outputs() {
        let n = 11;
        let q_b = 2;
        let base = PairDesign {
            background: (0..n)
                .flat_map(|row| [((row * 3 % 7) as f64) - 2.0, ((row * 5 % 9) as f64) - 4.0])
                .collect(),
            phase: (0..n)
                .map(|row| if row % 3 == 0 { 1.0 } else { -1.0 })
                .collect(),
        };
        let other = PairDesign {
            background: (0..n)
                .flat_map(|row| [((row * 7 % 11) as f64) - 5.0, base.background[row * q_b]])
                .collect(),
            phase: base
                .background
                .chunks_exact(q_b)
                .map(|row| row[1])
                .collect(),
        };
        let designs = vec![base.clone(), other, base.clone(), base.clone()];

        let mut lower = vec![0.0; n * n];
        for row in 0..n {
            lower[row * n + row] = 1.2 + row as f64 * 0.04;
            for col in 0..row {
                lower[row * n + col] = ((row * 5 + col * 3) % 13) as f64 * 0.01;
            }
        }
        let mut covariance = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..n {
                for inner in 0..=row.min(col) {
                    covariance[row * n + col] += lower[row * n + inner] * lower[col * n + inner];
                }
            }
        }
        let metric = CovarianceMetric::from_dense(&covariance, n).unwrap();
        let whitened = whiten_pair_design_batch(&designs, q_b, &metric).unwrap();

        for (input, output) in designs.iter().zip(&whitened) {
            for column in 0..q_b {
                let raw = (0..n)
                    .map(|row| input.background[row * q_b + column])
                    .collect::<Vec<_>>();
                let expected = metric.whiten(&raw).unwrap();
                assert!(output.background_columns[column]
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
            }
            assert!(output
                .phase
                .iter()
                .zip(metric.whiten(&input.phase).unwrap())
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
        }
        assert_eq!(
            whitened[0].background_columns,
            whitened[2].background_columns
        );
        assert_eq!(whitened[0].phase, whitened[2].phase);
        assert_eq!(
            whitened[0].background_columns,
            whitened[3].background_columns
        );
        assert_eq!(whitened[0].phase, whitened[3].phase);
    }

    fn dense_ols_oracle(
        c: &[f64],
        q_c: usize,
        b: &[f64],
        q_b: usize,
        z: &[f64],
        y: &[f64],
    ) -> (f64, f64, usize) {
        let n = y.len();
        let p = q_c + q_b + 1;
        let mut x = Vec::with_capacity(n * p);
        for row in 0..n {
            x.extend_from_slice(&c[row * q_c..(row + 1) * q_c]);
            x.extend_from_slice(&b[row * q_b..(row + 1) * q_b]);
            x.push(z[row]);
        }
        let xm = DMatrix::from_row_slice(n, p, &x);
        let ym = DMatrix::from_column_slice(n, 1, y);
        let xtx = xm.transpose() * &xm;
        let xtx_inv = xtx.try_inverse().expect("oracle design is full rank");
        let beta = &xtx_inv * xm.transpose() * ym;
        let resid = DMatrix::from_column_slice(n, 1, y) - xm * &beta;
        let rss = (resid.transpose() * resid)[(0, 0)];
        let df = n - p;
        let se = (rss / df as f64 * xtx_inv[(p - 1, p - 1)]).sqrt();
        (beta[(p - 1, 0)], se, df)
    }

    #[test]
    fn lm_add_dom_matches_full_ols_with_rank_and_residual_df() {
        let (c, q_c, b, q_b, z, y) = synthetic_design();
        let (beta, se, df) = dense_ols_oracle(&c, q_c, &b, q_b, &z, &y);
        let fit = fit_conditional(
            &y,
            &c,
            q_c,
            &b,
            q_b,
            &z,
            &CovarianceMetric::identity(y.len()),
            InferenceKind::LinearModel,
            None,
        )
        .unwrap();
        assert_eq!(fit.status, FitStatus::Testable);
        assert_eq!(fit.fixed_effect_rank, q_c + q_b + 1);
        assert_eq!(fit.residual_df, df);
        assert!((fit.beta_z - beta).abs() < 1e-11);
        assert!((fit.se_z - se).abs() < 1e-11);
        assert!((fit.beta_cis_minus_trans - 2.0 * beta).abs() < 1e-11);
        assert!((fit.se_cis_minus_trans - 2.0 * se).abs() < 1e-11);
    }

    fn asymmetric_phase_data(cis_effect: f64, trans_effect: f64) -> (FilledPair, Vec<f64>) {
        let mut filled = FilledPair {
            z: Vec::new(),
            g_i: Vec::new(),
            g_j: Vec::new(),
            imputed: Vec::new(),
            n_imputed: 0,
        };
        let mut y = Vec::new();
        for gi in 0..=2 {
            for gj in 0..=2 {
                let phases: &[i8] = if gi == 1 && gj == 1 { &[1, -1] } else { &[0] };
                for &phase in phases {
                    let repeats = if phase == 1 { 25 } else { 5 };
                    for repeat in 0..repeats {
                        let hi = f64::from(gi == 1);
                        let hj = f64::from(gj == 1);
                        let effect = match phase {
                            1 => cis_effect,
                            -1 => trans_effect,
                            _ => 0.0,
                        };
                        y.push(
                            0.7 + 0.18 * gi as f64 - 0.11 * gj as f64
                                + 0.14 * hi
                                + 0.06 * hj
                                + effect
                                + 0.02 * (repeat as f64 - (repeats - 1) as f64 / 2.0),
                        );
                        filled.z.push(phase);
                        filled.g_i.push(gi);
                        filled.g_j.push(gj);
                        filled.imputed.push(false);
                    }
                }
            }
        }
        (filled, y)
    }

    #[test]
    fn add_dom_phase_contrast_allows_unequal_same_direction_effects() {
        for (cis, trans) in [(3.0, 1.0), (2.0, 2.0)] {
            let (filled, y) = asymmetric_phase_data(cis, trans);
            let (b, q_b) = build_background(&filled, ConditionMode::AddDom).unwrap();
            let fit = fit_conditional(
                &y,
                &vec![1.0; y.len()],
                1,
                &b,
                q_b,
                &filled.z.iter().map(|&z| z as f64).collect::<Vec<_>>(),
                &CovarianceMetric::identity(y.len()),
                InferenceKind::LinearModel,
                Some(q_b - 1),
            )
            .unwrap();
            assert_eq!(fit.status, FitStatus::Testable);
            assert!(
                (fit.beta_cis_minus_trans - (cis - trans)).abs() < 1e-10,
                "cis={cis}, trans={trans}, fitted difference={}",
                fit.beta_cis_minus_trans
            );
        }
    }

    #[test]
    fn separate_phase_effects_and_standard_errors_match_dummy_ols_and_gls() {
        let (filled, y) = asymmetric_phase_data(3.0, 1.0);
        let n = y.len();
        for condition in [ConditionMode::AddDom, ConditionMode::Genotype] {
            let (b, q_b) = build_background(&filled, condition).unwrap();
            for inference in [InferenceKind::LinearModel, InferenceKind::FixedCovariance] {
                let mut v = DMatrix::<f64>::identity(n, n);
                if inference == InferenceKind::FixedCovariance {
                    for i in 0..n {
                        for j in 0..n {
                            let ai = ((i * 7 % 17) as f64 - 8.0) / 9.0;
                            let aj = ((j * 7 % 17) as f64 - 8.0) / 9.0;
                            v[(i, j)] = if i == j { 1.7 } else { 0.0 };
                            v[(i, j)] += 0.6 * ai * aj;
                        }
                    }
                }
                let metric_values = (0..n)
                    .flat_map(|i| {
                        let v = &v;
                        (0..n).map(move |j| v[(i, j)])
                    })
                    .collect::<Vec<_>>();
                let metric = CovarianceMetric::from_dense(&metric_values, n).unwrap();
                let z = filled.z.iter().map(|&z| z as f64).collect::<Vec<_>>();
                let p = q_b + 2;
                let mut x = Vec::new();
                for row in 0..n {
                    x.push(1.0);
                    x.extend_from_slice(&b[row * q_b..row * q_b + q_b - 1]);
                    x.push(f64::from(filled.z[row] == 1));
                    x.push(f64::from(filled.z[row] == -1));
                }
                let xm = DMatrix::from_row_slice(n, p, &x);
                let ym = DMatrix::from_column_slice(n, 1, &y);
                let precision = v.try_inverse().unwrap();
                let mut covariance = (xm.transpose() * &precision * &xm).try_inverse().unwrap();
                let beta = &covariance * xm.transpose() * &precision * &ym;
                if inference == InferenceKind::LinearModel {
                    let residual = &ym - &xm * &beta;
                    covariance *= (residual.transpose() * residual)[(0, 0)] / (n - p) as f64;
                }
                let fit = fit_conditional(
                    &y,
                    &vec![1.0; n],
                    1,
                    &b,
                    q_b,
                    &z,
                    &metric,
                    inference,
                    Some(q_b - 1),
                )
                .unwrap();
                let prepared = prepare_conditional_fit(&y, &vec![1.0; n], 1, &metric).unwrap();
                let prepared_fit = fit_conditional_prepared(
                    &prepared,
                    &b,
                    q_b,
                    &z,
                    &metric,
                    inference,
                    Some(q_b - 1),
                )
                .unwrap();
                assert_eq!(prepared_fit, fit);
                assert_eq!(fit.status, FitStatus::Testable);
                assert!(fit.shared_effect_identifiable);
                assert_eq!(fit.fixed_effect_rank, p);
                assert_eq!(fit.residual_df, n - p);
                assert!((fit.beta_cis - beta[(p - 2, 0)]).abs() < 1e-10);
                assert!((fit.beta_trans - beta[(p - 1, 0)]).abs() < 1e-10);
                assert!((fit.se_cis - covariance[(p - 2, p - 2)].sqrt()).abs() < 1e-10);
                assert!((fit.se_trans - covariance[(p - 1, p - 1)].sqrt()).abs() < 1e-10);
                let variance_difference = covariance[(p - 2, p - 2)] + covariance[(p - 1, p - 1)]
                    - 2.0 * covariance[(p - 2, p - 1)];
                assert!((fit.se_cis_minus_trans - variance_difference.sqrt()).abs() < 1e-10);
                assert!((fit.beta_shared - (fit.beta_cis + fit.beta_trans) / 2.0).abs() < 1e-12);
                let variance_shared = (covariance[(p - 2, p - 2)]
                    + covariance[(p - 1, p - 1)]
                    + 2.0 * covariance[(p - 2, p - 1)])
                    / 4.0;
                assert!((fit.se_shared - variance_shared.sqrt()).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn all_double_heterozygotes_identify_contrast_but_not_individual_deviations() {
        let n = 20;
        let filled = FilledPair {
            z: (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect(),
            g_i: vec![1; n],
            g_j: vec![1; n],
            imputed: vec![false; n],
            n_imputed: 0,
        };
        let y = filled
            .z
            .iter()
            .enumerate()
            .map(|(i, &z)| 5.0 + z as f64 + 0.01 * (i / 2) as f64)
            .collect::<Vec<_>>();
        for condition in [ConditionMode::AddDom, ConditionMode::Genotype] {
            let (b, q_b) = build_background(&filled, condition).unwrap();
            let fit = fit_conditional(
                &y,
                &vec![1.0; n],
                1,
                &b,
                q_b,
                &filled.z.iter().map(|&z| z as f64).collect::<Vec<_>>(),
                &CovarianceMetric::identity(n),
                InferenceKind::LinearModel,
                Some(q_b - 1),
            )
            .unwrap();
            assert_eq!(fit.status, FitStatus::Testable);
            assert_eq!(fit.fixed_effect_rank, 2);
            assert!((fit.beta_cis_minus_trans - 2.0).abs() < 1e-10);
            assert!(!fit.shared_effect_identifiable);
            assert!(fit.beta_cis.is_nan() && fit.beta_trans.is_nan());
            assert!(fit.se_cis.is_nan() && fit.se_trans.is_nan());
        }
    }

    #[test]
    fn condition_modes_build_add_dom_and_full_unphased_genotype_backgrounds() {
        let filled = FilledPair {
            z: vec![1, -1, 0, 0],
            g_i: vec![0, 1, 2, 1],
            g_j: vec![1, 1, 2, 0],
            imputed: vec![false; 4],
            n_imputed: 0,
        };
        let (add_dom, q_add_dom) = build_background(&filled, ConditionMode::AddDom).unwrap();
        assert_eq!(q_add_dom, 5);
        assert_eq!(
            add_dom,
            vec![
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 1.0, 0.0, 0.0
            ]
        );

        let (genotype, q_genotype) = build_background(&filled, ConditionMode::Genotype).unwrap();
        assert_eq!(q_genotype, 8);
        for row in 0..filled.z.len() {
            let base = row * q_genotype;
            let gi = filled.g_i[row] as f64;
            let gj = filled.g_j[row] as f64;
            let hi = if filled.g_i[row] == 1 { 1.0 } else { 0.0 };
            let hj = if filled.g_j[row] == 1 { 1.0 } else { 0.0 };
            assert_eq!(
                &genotype[base..base + 8],
                &[gi, gj, hi, hj, gi * gj, gi * hj, hi * gj, hi * hj]
            );
        }
    }

    #[test]
    fn fixed_covariance_gls_matches_dense_oracle_with_nonunit_absolute_scale() {
        let (c, q_c, b, q_b, z, y) = synthetic_design();
        let n = y.len();
        let mut v = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let ai = ((i * 3 % 11) as f64 - 5.0) / 8.0;
                let aj = ((j * 3 % 11) as f64 - 5.0) / 8.0;
                v[i * n + j] = if i == j { 2.4 } else { 0.0 };
                v[i * n + j] += 1.7 * ai * aj;
            }
        }
        let vm = DMatrix::from_row_slice(n, n, &v);
        let vinv = vm
            .clone()
            .try_inverse()
            .expect("oracle V is positive definite");
        let p = q_c + q_b + 1;
        let mut x = Vec::with_capacity(n * p);
        for row in 0..n {
            x.extend_from_slice(&c[row * q_c..(row + 1) * q_c]);
            x.extend_from_slice(&b[row * q_b..(row + 1) * q_b]);
            x.push(z[row]);
        }
        let xm = DMatrix::from_row_slice(n, p, &x);
        let ym = DMatrix::from_column_slice(n, 1, &y);
        let xtvix = xm.transpose() * &vinv * &xm;
        let covariance = xtvix.try_inverse().expect("oracle GLS design is full rank");
        let beta = &covariance * xm.transpose() * &vinv * ym;
        let fit = fit_conditional(
            &y,
            &c,
            q_c,
            &b,
            q_b,
            &z,
            &CovarianceMetric::from_dense(&v, n).unwrap(),
            InferenceKind::FixedCovariance,
            None,
        )
        .unwrap();
        assert_eq!(fit.status, FitStatus::Testable);
        assert!((fit.beta_z - beta[(p - 1, 0)]).abs() < 1e-10);
        assert!((fit.se_z - covariance[(p - 1, p - 1)].sqrt()).abs() < 1e-10);
        assert!((fit.statistic - fit.beta_z * fit.beta_z / (fit.se_z * fit.se_z)).abs() < 1e-12);
    }

    #[test]
    fn zero_genetic_variance_gls_retains_nonunit_residual_variance_scale() {
        let (c, q_c, b, q_b, z, y) = synthetic_design();
        let n = y.len();
        let p = q_c + q_b + 1;
        let mut x = Vec::with_capacity(n * p);
        for row in 0..n {
            x.extend_from_slice(&c[row * q_c..(row + 1) * q_c]);
            x.extend_from_slice(&b[row * q_b..(row + 1) * q_b]);
            x.push(z[row]);
        }
        let xm = DMatrix::from_row_slice(n, p, &x);
        let covariance = (xm.transpose() * &xm)
            .try_inverse()
            .expect("oracle design is full rank");
        let mut grm = vec![0.0; n * n];
        for i in 0..n {
            grm[i * n + i] = 1.0;
        }
        let fit = fit_conditional(
            &y,
            &c,
            q_c,
            &b,
            q_b,
            &z,
            &CovarianceMetric::from_components(&grm, n, 0.0, 4.0).unwrap(),
            InferenceKind::FixedCovariance,
            None,
        )
        .unwrap();
        assert_eq!(fit.status, FitStatus::Testable);
        assert!((fit.se_z - 2.0 * covariance[(p - 1, p - 1)].sqrt()).abs() < 1e-10);
    }

    #[test]
    fn phase_term_collinear_with_background_is_not_testable_without_ridge() {
        let y = [1.0, 0.0, 1.0, 0.0, 2.0, 1.0];
        let c = [1.0; 6];
        let b = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let z = b;
        let fit = fit_conditional(
            &y,
            &c,
            1,
            &b,
            1,
            &z,
            &CovarianceMetric::identity(y.len()),
            InferenceKind::LinearModel,
            None,
        )
        .unwrap();
        assert_eq!(fit.status, FitStatus::UnidentifiablePhase);
        assert!(fit.beta_z.is_nan());
        assert!(fit.pwald1.is_nan());
    }

    #[test]
    fn duplicate_background_columns_do_not_gain_rank_from_projection_roundoff() {
        let n = 20;
        let c = vec![1.0; n];
        let b = vec![1.0; n * 4];
        let z = (0..n)
            .map(|row| if row % 2 == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let y = (0..n)
            .map(|row| {
                let noise = match row % 4 {
                    0 => 0.1,
                    1 => -0.1,
                    2 => -0.1,
                    _ => 0.1,
                };
                0.4 + 0.6 * z[row] + noise
            })
            .collect::<Vec<_>>();

        let fit = fit_conditional(
            &y,
            &c,
            1,
            &b,
            4,
            &z,
            &CovarianceMetric::identity(n),
            InferenceKind::LinearModel,
            None,
        )
        .unwrap();

        assert_eq!(fit.status, FitStatus::Testable);
        assert_eq!(fit.fixed_effect_rank, 2);
        assert_eq!(fit.residual_df, n - 2);
        assert!((fit.beta_z - 0.6).abs() < 1.0e-12);
    }

    #[test]
    fn covariate_offset_does_not_change_conditional_fit_or_rank() {
        let n = 200;
        let mut c_centered = Vec::with_capacity(n * 2);
        let mut c_shifted = Vec::with_capacity(n * 2);
        let mut b = Vec::with_capacity(n * 4);
        let mut z = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for row in 0..n {
            let x = ((row * 17 % 101) as f64 - 50.0) / 37.0;
            let phase = if row % 2 == 0 { 1.0 } else { -1.0 };
            let noise = ((row * 13 % 19) as f64 - 9.0) / 80.0;
            c_centered.extend([1.0, x]);
            c_shifted.extend([1.0, x + 1.0e7]);
            b.extend([1.0; 4]);
            z.push(phase);
            y.push(10.0 + 4.0 * x + 0.3 * phase + noise);
        }

        let metric = CovarianceMetric::identity(n);
        let centered = fit_conditional(
            &y,
            &c_centered,
            2,
            &b,
            4,
            &z,
            &metric,
            InferenceKind::LinearModel,
            None,
        )
        .unwrap();
        let shifted = fit_conditional(
            &y,
            &c_shifted,
            2,
            &b,
            4,
            &z,
            &metric,
            InferenceKind::LinearModel,
            None,
        )
        .unwrap();

        assert_eq!(centered.fixed_effect_rank, 3);
        assert_eq!(shifted.fixed_effect_rank, centered.fixed_effect_rank);
        assert_eq!(shifted.residual_df, centered.residual_df);
        assert!((shifted.beta_z - centered.beta_z).abs() < 1.0e-10);
        assert!((shifted.se_z - centered.se_z).abs() < 1.0e-10);
        assert!((shifted.pwald1 - centered.pwald1).abs() < 1.0e-9);
    }

    #[test]
    fn separate_phase_effects_are_stable_with_a_large_phenotype_baseline() {
        let n = 12;
        let c = vec![1.0; n];
        let mut h = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let noise = [
            0.05, -0.05, 0.05, -0.05, 0.04, -0.04, 0.04, -0.04, 0.1, -0.1, 0.1, -0.1,
        ];
        for (row, noise) in noise.into_iter().enumerate() {
            let shared = f64::from(row < 8);
            let phase = match row {
                0..=3 => 1.0,
                4..=7 => -1.0,
                _ => 0.0,
            };
            h.push(shared);
            z.push(phase);
            y.push(1.0e10 + 2.0 * shared + 0.75 * phase + noise);
        }

        let fit = fit_conditional(
            &y,
            &c,
            1,
            &h,
            1,
            &z,
            &CovarianceMetric::identity(n),
            InferenceKind::LinearModel,
            Some(0),
        )
        .unwrap();

        assert_eq!(fit.status, FitStatus::Testable);
        assert!((fit.beta_cis - 2.75).abs() < 1.0e-7, "{}", fit.beta_cis);
        assert!((fit.beta_trans - 1.25).abs() < 1.0e-7, "{}", fit.beta_trans);
    }
}
