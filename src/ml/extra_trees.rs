use crate::ml::common::ResponseKind;
use rand::rngs::StdRng;
use rand::seq::index::sample as sample_index;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct ExtraTreesConfig {
    pub n_estimators: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
    pub bootstrap: bool,
    pub feature_subsample: f64,
    pub seed: u64,
    pub allow_parallel: bool,
}

#[derive(Clone, Copy, Debug)]
struct NodeStats {
    n: usize,
    sum: f64,
    pos: usize,
}

#[inline]
fn make_node_stats(samples: &[usize], y: &[f64], response: ResponseKind) -> NodeStats {
    let mut sum = 0.0f64;
    let mut pos = 0usize;
    for &i in samples {
        let v = y[i];
        sum += v;
        if response == ResponseKind::Binary && v > 0.5 {
            pos += 1;
        }
    }
    NodeStats {
        n: samples.len(),
        sum,
        pos,
    }
}

#[inline]
fn gini_n(stats: NodeStats) -> f64 {
    if stats.n == 0 {
        return 0.0;
    }
    let p = stats.pos as f64;
    let n = stats.n as f64;
    let q = n - p;
    if p <= 0.0 || q <= 0.0 {
        0.0
    } else {
        2.0 * p * q / n
    }
}

fn compute_mtry(n_features: usize, feature_subsample: f64) -> usize {
    if n_features <= 1 {
        return n_features;
    }
    if feature_subsample.is_finite() && feature_subsample > 0.0 {
        if feature_subsample < 1.0 {
            return ((feature_subsample * (n_features as f64)).round() as usize)
                .clamp(1, n_features);
        }
        return (feature_subsample.round() as usize).clamp(1, n_features);
    }
    // Default ET/RF-style sqrt(m).
    ((n_features as f64).sqrt().round() as usize).clamp(1, n_features)
}

fn split_gain_for_feature(
    feat_row: &[u8],
    samples: &[usize],
    y: &[f64],
    response: ResponseKind,
    min_leaf: usize,
    parent: NodeStats,
) -> Option<f64> {
    let mut left_n = 0usize;
    let mut left_sum = 0.0f64;
    let mut left_pos = 0usize;

    for &si in samples {
        if feat_row[si] == 0 {
            left_n += 1;
            let v = y[si];
            match response {
                ResponseKind::Continuous => {
                    left_sum += v;
                }
                ResponseKind::Binary => {
                    if v > 0.5 {
                        left_pos += 1;
                    }
                }
            }
        }
    }

    let right_n = parent.n.saturating_sub(left_n);
    if left_n < min_leaf || right_n < min_leaf {
        return None;
    }

    let left = NodeStats {
        n: left_n,
        sum: left_sum,
        pos: left_pos,
    };
    let right = NodeStats {
        n: right_n,
        sum: parent.sum - left_sum,
        pos: parent.pos.saturating_sub(left_pos),
    };

    let gain = match response {
        // For binary 0/1 features, the SSE reduction has a closed form that only
        // needs group counts and group sums; this is algebraically equivalent to
        // the original variance-based split score but avoids per-candidate sumsq work.
        ResponseKind::Continuous => {
            let parent_term = (parent.sum * parent.sum) / (parent.n as f64);
            let left_term = (left.sum * left.sum) / (left.n as f64);
            let right_term = (right.sum * right.sum) / (right.n as f64);
            left_term + right_term - parent_term
        }
        ResponseKind::Binary => gini_n(parent) - gini_n(left) - gini_n(right),
    };
    if gain.is_finite() && gain > 0.0 {
        Some(gain)
    } else {
        None
    }
}

fn split_samples_by_feature(feat_row: &[u8], samples: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut left = Vec::<usize>::with_capacity(samples.len());
    let mut right = Vec::<usize>::with_capacity(samples.len());
    for &si in samples {
        if feat_row[si] == 0 {
            left.push(si);
        } else {
            right.push(si);
        }
    }
    (left, right)
}

fn grow_tree(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    cfg: ExtraTreesConfig,
    samples: &[usize],
    depth: usize,
    rng: &mut StdRng,
    importance: &mut [f64],
    mtry: usize,
    feature_group_ids: Option<&[usize]>,
    used_groups: &[usize],
) {
    if samples.len() < cfg.min_samples_split {
        return;
    }
    if depth >= cfg.max_depth {
        return;
    }

    let parent = make_node_stats(samples, y, response);
    if parent.n < cfg.min_samples_split {
        return;
    }

    let n_features = x_rows.len();
    let feat_candidates: Vec<usize> = if mtry >= n_features {
        (0..n_features).collect()
    } else {
        sample_index(rng, n_features, mtry).into_vec()
    };
    if feat_candidates.is_empty() {
        return;
    }

    let mut best_feat: Option<usize> = None;
    let mut best_gain = f64::NEG_INFINITY;
    for feat in feat_candidates {
        if let Some(group_ids) = feature_group_ids {
            if used_groups.iter().any(|&gid| gid == group_ids[feat]) {
                continue;
            }
        }
        if let Some(gain) = split_gain_for_feature(
            &x_rows[feat],
            samples,
            y,
            response,
            cfg.min_samples_leaf,
            parent,
        ) {
            if gain > best_gain {
                best_gain = gain;
                best_feat = Some(feat);
            }
        }
    }

    let Some(feat) = best_feat else {
        return;
    };
    if !best_gain.is_finite() || best_gain <= 0.0 {
        return;
    }
    importance[feat] += best_gain;

    let mut next_used = used_groups.to_vec();
    if let Some(group_ids) = feature_group_ids {
        next_used.push(group_ids[feat]);
    }

    let (left, right) = split_samples_by_feature(&x_rows[feat], samples);
    if left.len() < cfg.min_samples_leaf || right.len() < cfg.min_samples_leaf {
        return;
    }

    grow_tree(
        x_rows,
        y,
        response,
        cfg,
        &left,
        depth + 1,
        rng,
        importance,
        mtry,
        feature_group_ids,
        next_used.as_slice(),
    );
    grow_tree(
        x_rows,
        y,
        response,
        cfg,
        &right,
        depth + 1,
        rng,
        importance,
        mtry,
        feature_group_ids,
        next_used.as_slice(),
    );
}

fn bootstrap_indices(n_samples: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        out.push(rng.random_range(0..n_samples));
    }
    out
}

#[allow(dead_code)]
pub fn feature_scores_extra_trees(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    cfg: ExtraTreesConfig,
) -> Vec<f64> {
    feature_scores_extra_trees_grouped(x_rows, y, response, cfg, None)
}

pub fn feature_scores_extra_trees_grouped(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    cfg: ExtraTreesConfig,
    feature_group_ids: Option<&[usize]>,
) -> Vec<f64> {
    let n_features = x_rows.len();
    if n_features == 0 {
        return Vec::new();
    }
    let n_samples = y.len();
    if n_samples == 0 {
        return vec![0.0; n_features];
    }

    let n_estimators = cfg.n_estimators.max(1);
    let max_depth = cfg.max_depth.max(1);
    let min_leaf = cfg.min_samples_leaf.max(1);
    let min_split = cfg.min_samples_split.max(min_leaf * 2).max(2);
    let cfg_use = ExtraTreesConfig {
        n_estimators,
        max_depth,
        min_samples_leaf: min_leaf,
        min_samples_split: min_split,
        bootstrap: cfg.bootstrap,
        feature_subsample: cfg.feature_subsample,
        seed: cfg.seed,
        allow_parallel: cfg.allow_parallel,
    };
    let mtry = compute_mtry(n_features, cfg_use.feature_subsample);

    let merged = if cfg_use.allow_parallel {
        (0..n_estimators)
            .into_par_iter()
            .map(|t| {
                let mut imp = vec![0.0f64; n_features];
                let mut rng = StdRng::seed_from_u64(
                    cfg_use.seed ^ ((t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                );
                let root_samples: Vec<usize> = if cfg_use.bootstrap {
                    bootstrap_indices(n_samples, &mut rng)
                } else {
                    (0..n_samples).collect()
                };
                grow_tree(
                    x_rows,
                    y,
                    response,
                    cfg_use,
                    &root_samples,
                    0,
                    &mut rng,
                    &mut imp,
                    mtry,
                    feature_group_ids,
                    &[],
                );
                imp
            })
            .reduce(
                || vec![0.0f64; n_features],
                |mut a, b| {
                    for i in 0..a.len() {
                        a[i] += b[i];
                    }
                    a
                },
            )
    } else {
        let mut merged = vec![0.0f64; n_features];
        for t in 0..n_estimators {
            let mut imp = vec![0.0f64; n_features];
            let mut rng = StdRng::seed_from_u64(
                cfg_use.seed ^ ((t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let root_samples: Vec<usize> = if cfg_use.bootstrap {
                bootstrap_indices(n_samples, &mut rng)
            } else {
                (0..n_samples).collect()
            };
            grow_tree(
                x_rows,
                y,
                response,
                cfg_use,
                &root_samples,
                0,
                &mut rng,
                &mut imp,
                mtry,
                feature_group_ids,
                &[],
            );
            for i in 0..n_features {
                merged[i] += imp[i];
            }
        }
        merged
    };

    let sum_imp: f64 = merged.iter().sum();
    if sum_imp.is_finite() && sum_imp > 0.0 {
        merged.iter().map(|v| *v / sum_imp).collect()
    } else {
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_split_gain_matches_sse_reduction() {
        let feat = vec![0u8, 0, 1, 1];
        let y = vec![5.0, 4.0, -1.0, -2.0];
        let samples = vec![0usize, 1, 2, 3];
        let parent = make_node_stats(&samples, &y, ResponseKind::Continuous);
        let gain = split_gain_for_feature(
            feat.as_slice(),
            samples.as_slice(),
            y.as_slice(),
            ResponseKind::Continuous,
            1,
            parent,
        )
        .unwrap();
        let old_gain = {
            let y = y.as_slice();
            let mean_parent = y.iter().copied().sum::<f64>() / (y.len() as f64);
            let mean_left = 9.0 / 2.0;
            let mean_right = -3.0 / 2.0;
            let parent_sse = y
                .iter()
                .map(|v| (v - mean_parent) * (v - mean_parent))
                .sum::<f64>();
            let left_sse = [5.0, 4.0]
                .iter()
                .map(|v| (v - mean_left) * (v - mean_left))
                .sum::<f64>();
            let right_sse = [-1.0, -2.0]
                .iter()
                .map(|v| (v - mean_right) * (v - mean_right))
                .sum::<f64>();
            parent_sse - left_sse - right_sse
        };
        assert!((gain - old_gain).abs() < 1e-12);
        assert!((gain - 36.0).abs() < 1e-12);
    }
}
