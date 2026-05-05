use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseKind {
    Binary,
    Continuous,
}

pub fn parse_response(response: &str) -> Result<ResponseKind, String> {
    let t = response.trim().to_ascii_lowercase();
    match t.as_str() {
        "binary" | "bin" | "b" => Ok(ResponseKind::Binary),
        "continuous" | "cont" | "c" => Ok(ResponseKind::Continuous),
        _ => Err("response must be 'binary' or 'continuous'".to_string()),
    }
}

pub fn readonly_f64_to_vec(arr: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    match arr.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => arr.as_array().iter().copied().collect(),
    }
}

pub fn matrix_i8_to_binary_rows(
    x: PyReadonlyArray2<'_, i8>,
) -> Result<(Vec<Vec<u8>>, usize, usize), String> {
    let view = x.as_array();
    if view.ndim() != 2 {
        return Err("x must be a 2D matrix with shape (n_snps, n_samples)".to_string());
    }
    let n_snps = view.shape()[0];
    let n_samples = view.shape()[1];
    if n_snps == 0 || n_samples == 0 {
        return Err("x must be non-empty in both dimensions".to_string());
    }
    let mut out = vec![vec![0u8; n_samples]; n_snps];
    for r in 0..n_snps {
        for c in 0..n_samples {
            // GARFIELD binary literal semantics: carrier if dosage > 0.
            out[r][c] = if view[[r, c]] > 0 { 1 } else { 0 };
        }
    }
    Ok((out, n_snps, n_samples))
}

pub fn validate_xy(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
) -> Result<(usize, usize), String> {
    if x_rows.is_empty() {
        return Err("x must contain at least one SNP row".to_string());
    }
    let n_snps = x_rows.len();
    let n_samples = x_rows[0].len();
    if n_samples == 0 {
        return Err("x must contain at least one sample".to_string());
    }
    if y.len() != n_samples {
        return Err(format!(
            "y length ({}) must match x n_samples ({})",
            y.len(),
            n_samples
        ));
    }
    for (i, row) in x_rows.iter().enumerate() {
        if row.len() != n_samples {
            return Err(format!(
                "x row {} length ({}) != n_samples ({})",
                i,
                row.len(),
                n_samples
            ));
        }
    }
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("y contains non-finite value at index {}", i));
        }
    }
    if response == ResponseKind::Binary {
        for (i, &v) in y.iter().enumerate() {
            if !(v == 0.0 || v == 1.0) {
                return Err(format!(
                    "binary response requires y in {{0,1}}; got y[{}]={}",
                    i, v
                ));
            }
        }
    }
    Ok((n_snps, n_samples))
}

pub fn topk_indices(scores: &[f64], k: usize) -> Vec<usize> {
    if k == 0 || scores.is_empty() {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        let sa = if scores[a].is_finite() {
            scores[a]
        } else {
            f64::NEG_INFINITY
        };
        let sb = if scores[b].is_finite() {
            scores[b]
        } else {
            f64::NEG_INFINITY
        };
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(k.min(idx.len()));
    idx
}
