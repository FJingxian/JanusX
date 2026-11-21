#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseKind {
    Binary,
    Continuous,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportanceKind {
    Imp,
    Permutation,
}

pub fn parse_importance(kind: &str) -> Result<ImportanceKind, String> {
    let t = kind.trim().to_ascii_lowercase();
    match t.as_str() {
        "" | "imp" | "importance" | "gain" | "split" => Ok(ImportanceKind::Imp),
        "perm" | "permutation" | "permutation_importance" => Ok(ImportanceKind::Permutation),
        _ => Err("importance must be 'imp' or 'permutation'".to_string()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermutationScoring {
    Auto,
    R2,
    NegMse,
    Accuracy,
    NegLogLoss,
}

pub fn parse_permutation_scoring(scoring: &str) -> Result<PermutationScoring, String> {
    let t = scoring.trim().to_ascii_lowercase();
    match t.as_str() {
        "" | "auto" => Ok(PermutationScoring::Auto),
        "r2" => Ok(PermutationScoring::R2),
        "neg_mse" | "neg-mse" | "mse" => Ok(PermutationScoring::NegMse),
        "accuracy" | "acc" => Ok(PermutationScoring::Accuracy),
        "neg_logloss" | "neg-logloss" | "logloss" => Ok(PermutationScoring::NegLogLoss),
        _ => Err(
            "permutation_scoring must be one of: auto, r2, neg_mse, accuracy, neg_logloss"
                .to_string(),
        ),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PermutationConfig {
    pub n_repeats: usize,
    pub scoring: PermutationScoring,
    pub seed: u64,
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

#[inline]
pub fn normalize_importance(scores: &[f64]) -> Vec<f64> {
    let sum_scores: f64 = scores.iter().sum();
    if sum_scores.is_finite() && sum_scores > 0.0 {
        scores.iter().map(|v| *v / sum_scores).collect()
    } else {
        scores.to_vec()
    }
}

#[inline]
fn resolve_scoring(response: ResponseKind, scoring: PermutationScoring) -> PermutationScoring {
    match scoring {
        PermutationScoring::Auto => match response {
            ResponseKind::Binary => PermutationScoring::Accuracy,
            ResponseKind::Continuous => PermutationScoring::R2,
        },
        other => other,
    }
}

#[inline]
fn clip_prob(p: f64) -> f64 {
    p.clamp(1e-12, 1.0 - 1e-12)
}

pub fn score_predictions(
    y: &[f64],
    pred: &[f64],
    response: ResponseKind,
    scoring: PermutationScoring,
) -> f64 {
    if y.len() != pred.len() || y.is_empty() {
        return f64::NAN;
    }
    let resolved = resolve_scoring(response, scoring);
    match resolved {
        PermutationScoring::R2 => {
            let mean_y = y.iter().sum::<f64>() / (y.len() as f64);
            let mut ss_tot = 0.0f64;
            let mut ss_res = 0.0f64;
            for i in 0..y.len() {
                let dy = y[i] - mean_y;
                let dr = y[i] - pred[i];
                ss_tot += dy * dy;
                ss_res += dr * dr;
            }
            if ss_tot <= 0.0 {
                0.0
            } else {
                1.0 - ss_res / ss_tot
            }
        }
        PermutationScoring::NegMse => {
            let mut sse = 0.0f64;
            for i in 0..y.len() {
                let d = y[i] - pred[i];
                sse += d * d;
            }
            -(sse / (y.len() as f64))
        }
        PermutationScoring::Accuracy => {
            let mut correct = 0usize;
            for i in 0..y.len() {
                let cls = if pred[i] >= 0.5 { 1.0 } else { 0.0 };
                if cls == y[i] {
                    correct += 1;
                }
            }
            correct as f64 / (y.len() as f64)
        }
        PermutationScoring::NegLogLoss => {
            let mut loss = 0.0f64;
            for i in 0..y.len() {
                let p = clip_prob(pred[i]);
                loss += -(y[i] * p.ln() + (1.0 - y[i]) * (1.0 - p).ln());
            }
            -(loss / (y.len() as f64))
        }
        PermutationScoring::Auto => unreachable!(),
    }
}
