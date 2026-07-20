use crate::ml::common::{normalize_importance, score_predictions, PermutationConfig, ResponseKind};
use crate::ml::extra_trees::ExtraTreesConfig;
use rand::rngs::StdRng;
use rand::seq::index::sample as sample_index;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
struct NodeStats {
    n: usize,
    sum: f64,
    pos: usize,
}

#[derive(Clone, Debug)]
enum RfNode {
    Leaf(f64),
    Split {
        feat: usize,
        threshold: u8,
        left: Box<RfNode>,
        right: Box<RfNode>,
    },
}

#[derive(Clone, Debug)]
struct RandomForestModel {
    trees: Vec<RfNode>,
}

#[inline]
fn leaf_value(samples: &[usize], y: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples.iter().map(|&i| y[i]).sum::<f64>();
    sum / (samples.len() as f64)
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

#[inline]
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
    ((n_features as f64).sqrt().round() as usize).clamp(1, n_features)
}

#[derive(Clone, Copy, Debug)]
struct SplitCandidate {
    threshold: u8,
    gain: f64,
}

#[inline]
fn accumulate_node_stats(stats: &mut NodeStats, y_val: f64, response: ResponseKind) {
    stats.n += 1;
    match response {
        ResponseKind::Continuous => {
            stats.sum += y_val;
        }
        ResponseKind::Binary => {
            if y_val > 0.5 {
                stats.pos += 1;
            }
        }
    }
}

#[inline]
fn split_gain_from_stats(
    parent: NodeStats,
    left: NodeStats,
    right: NodeStats,
    response: ResponseKind,
) -> Option<f64> {
    let gain = match response {
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

fn split_gain_for_feature(
    feat_row: &[u8],
    samples: &[usize],
    y: &[f64],
    response: ResponseKind,
    min_leaf: usize,
    parent: NodeStats,
) -> Option<SplitCandidate> {
    let mut left_le0 = NodeStats {
        n: 0,
        sum: 0.0,
        pos: 0,
    };
    let mut left_le1 = NodeStats {
        n: 0,
        sum: 0.0,
        pos: 0,
    };

    for &si in samples {
        let v = y[si];
        match feat_row[si] {
            0 => {
                accumulate_node_stats(&mut left_le0, v, response);
                accumulate_node_stats(&mut left_le1, v, response);
            }
            1 => {
                accumulate_node_stats(&mut left_le1, v, response);
            }
            _ => {}
        }
    }

    let mut best = None::<SplitCandidate>;
    for (threshold, left) in [(0u8, left_le0), (1u8, left_le1)] {
        let right_n = parent.n.saturating_sub(left.n);
        if left.n < min_leaf || right_n < min_leaf {
            continue;
        }
        let right = NodeStats {
            n: right_n,
            sum: parent.sum - left.sum,
            pos: parent.pos.saturating_sub(left.pos),
        };
        if let Some(gain) = split_gain_from_stats(parent, left, right, response) {
            let cand = SplitCandidate { threshold, gain };
            if best.map(|cur| cand.gain > cur.gain).unwrap_or(true) {
                best = Some(cand);
            }
        }
    }
    best
}

fn split_samples_by_feature(
    feat_row: &[u8],
    samples: &[usize],
    threshold: u8,
) -> (Vec<usize>, Vec<usize>) {
    let mut left = Vec::<usize>::with_capacity(samples.len());
    let mut right = Vec::<usize>::with_capacity(samples.len());
    for &si in samples {
        if feat_row[si] <= threshold {
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
) -> RfNode {
    let leaf = RfNode::Leaf(leaf_value(samples, y));
    if samples.len() < cfg.min_samples_split || depth >= cfg.max_depth {
        return leaf;
    }

    let parent = make_node_stats(samples, y, response);
    if parent.n < cfg.min_samples_split {
        return leaf;
    }

    let n_features = x_rows.len();
    let feat_candidates: Vec<usize> = if mtry >= n_features {
        (0..n_features).collect()
    } else {
        sample_index(rng, n_features, mtry).into_vec()
    };
    if feat_candidates.is_empty() {
        return leaf;
    }

    let mut best_feat: Option<usize> = None;
    let mut best_threshold = 0u8;
    let mut best_gain = f64::NEG_INFINITY;
    for feat in feat_candidates {
        if let Some(group_ids) = feature_group_ids {
            if used_groups.iter().any(|&gid| gid == group_ids[feat]) {
                continue;
            }
        }
        if let Some(split) = split_gain_for_feature(
            &x_rows[feat],
            samples,
            y,
            response,
            cfg.min_samples_leaf,
            parent,
        ) {
            if split.gain > best_gain {
                best_gain = split.gain;
                best_feat = Some(feat);
                best_threshold = split.threshold;
            }
        }
    }

    let Some(feat) = best_feat else {
        return leaf;
    };
    if !best_gain.is_finite() || best_gain <= 0.0 {
        return leaf;
    }

    let (left_samples, right_samples) =
        split_samples_by_feature(&x_rows[feat], samples, best_threshold);
    if left_samples.len() < cfg.min_samples_leaf || right_samples.len() < cfg.min_samples_leaf {
        return leaf;
    }

    importance[feat] += best_gain;
    let mut next_used = used_groups.to_vec();
    if let Some(group_ids) = feature_group_ids {
        next_used.push(group_ids[feat]);
    }
    RfNode::Split {
        feat,
        threshold: best_threshold,
        left: Box::new(grow_tree(
            x_rows,
            y,
            response,
            cfg,
            &left_samples,
            depth + 1,
            rng,
            importance,
            mtry,
            feature_group_ids,
            next_used.as_slice(),
        )),
        right: Box::new(grow_tree(
            x_rows,
            y,
            response,
            cfg,
            &right_samples,
            depth + 1,
            rng,
            importance,
            mtry,
            feature_group_ids,
            next_used.as_slice(),
        )),
    }
}

fn bootstrap_indices(n_samples: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        out.push(rng.random_range(0..n_samples));
    }
    out
}

fn fit_random_forest(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    cfg: ExtraTreesConfig,
    feature_group_ids: Option<&[usize]>,
) -> (RandomForestModel, Vec<f64>) {
    let n_features = x_rows.len();
    let n_samples = y.len();
    if n_features == 0 || n_samples == 0 {
        return (
            RandomForestModel { trees: Vec::new() },
            vec![0.0; n_features],
        );
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

    let parts = if cfg_use.allow_parallel {
        (0..n_estimators)
            .into_par_iter()
            .map(|t| {
                let mut importance = vec![0.0f64; n_features];
                let mut rng = StdRng::seed_from_u64(
                    cfg_use.seed ^ ((t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                );
                let root_samples = if cfg_use.bootstrap {
                    bootstrap_indices(n_samples, &mut rng)
                } else {
                    (0..n_samples).collect()
                };
                let tree = grow_tree(
                    x_rows,
                    y,
                    response,
                    cfg_use,
                    &root_samples,
                    0,
                    &mut rng,
                    &mut importance,
                    mtry,
                    feature_group_ids,
                    &[],
                );
                (tree, importance)
            })
            .collect::<Vec<_>>()
    } else {
        let mut out = Vec::with_capacity(n_estimators);
        for t in 0..n_estimators {
            let mut importance = vec![0.0f64; n_features];
            let mut rng = StdRng::seed_from_u64(
                cfg_use.seed ^ ((t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let root_samples = if cfg_use.bootstrap {
                bootstrap_indices(n_samples, &mut rng)
            } else {
                (0..n_samples).collect()
            };
            let tree = grow_tree(
                x_rows,
                y,
                response,
                cfg_use,
                &root_samples,
                0,
                &mut rng,
                &mut importance,
                mtry,
                feature_group_ids,
                &[],
            );
            out.push((tree, importance));
        }
        out
    };

    let mut trees = Vec::with_capacity(parts.len());
    let mut importance = vec![0.0f64; n_features];
    for (tree, imp) in parts {
        trees.push(tree);
        for i in 0..n_features {
            importance[i] += imp[i];
        }
    }
    (
        RandomForestModel { trees },
        normalize_importance(&importance),
    )
}

fn predict_tree(
    node: &RfNode,
    x_rows: &[Vec<u8>],
    sample_idx: usize,
    override_feat: Option<usize>,
    override_row: Option<&[u8]>,
) -> f64 {
    match node {
        RfNode::Leaf(v) => *v,
        RfNode::Split {
            feat,
            threshold,
            left,
            right,
        } => {
            let bit = if override_feat == Some(*feat) {
                override_row
                    .map(|r| r[sample_idx])
                    .unwrap_or(x_rows[*feat][sample_idx])
            } else {
                x_rows[*feat][sample_idx]
            };
            if bit <= *threshold {
                predict_tree(left, x_rows, sample_idx, override_feat, override_row)
            } else {
                predict_tree(right, x_rows, sample_idx, override_feat, override_row)
            }
        }
    }
}

fn predict_forest(
    model: &RandomForestModel,
    x_rows: &[Vec<u8>],
    override_feat: Option<usize>,
    override_row: Option<&[u8]>,
) -> Vec<f64> {
    if model.trees.is_empty() || x_rows.is_empty() {
        return Vec::new();
    }
    let n_samples = x_rows[0].len();
    let inv_trees = 1.0f64 / (model.trees.len() as f64);
    let mut pred = vec![0.0f64; n_samples];
    for tree in &model.trees {
        for (i, pi) in pred.iter_mut().enumerate() {
            *pi += predict_tree(tree, x_rows, i, override_feat, override_row) * inv_trees;
        }
    }
    pred
}

fn permutation_importance(
    model: &RandomForestModel,
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    perm_cfg: PermutationConfig,
    allow_parallel: bool,
) -> Vec<f64> {
    if x_rows.is_empty() || y.is_empty() {
        return vec![0.0; x_rows.len()];
    }
    let baseline_pred = predict_forest(model, x_rows, None, None);
    let baseline_score = score_predictions(y, &baseline_pred, response, perm_cfg.scoring);
    let repeats = perm_cfg.n_repeats.max(1);

    if allow_parallel {
        (0..x_rows.len())
            .into_par_iter()
            .map(|feat| {
                let mut rng = StdRng::seed_from_u64(
                    perm_cfg.seed ^ ((feat as u64).wrapping_mul(0xD6E8_FD93_5A4D_94A5)),
                );
                let mut drops = 0.0f64;
                for _ in 0..repeats {
                    let mut perm_row = x_rows[feat].clone();
                    perm_row.shuffle(&mut rng);
                    let perm_pred = predict_forest(model, x_rows, Some(feat), Some(&perm_row));
                    let perm_score = score_predictions(y, &perm_pred, response, perm_cfg.scoring);
                    drops += baseline_score - perm_score;
                }
                drops / (repeats as f64)
            })
            .collect()
    } else {
        let mut out = Vec::with_capacity(x_rows.len());
        for feat in 0..x_rows.len() {
            let mut rng = StdRng::seed_from_u64(
                perm_cfg.seed ^ ((feat as u64).wrapping_mul(0xD6E8_FD93_5A4D_94A5)),
            );
            let mut drops = 0.0f64;
            for _ in 0..repeats {
                let mut perm_row = x_rows[feat].clone();
                perm_row.shuffle(&mut rng);
                let perm_pred = predict_forest(model, x_rows, Some(feat), Some(&perm_row));
                let perm_score = score_predictions(y, &perm_pred, response, perm_cfg.scoring);
                drops += baseline_score - perm_score;
            }
            out.push(drops / (repeats as f64));
        }
        out
    }
}

#[allow(dead_code)]
pub fn feature_scores_random_forest(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    cfg: ExtraTreesConfig,
) -> Vec<f64> {
    feature_scores_random_forest_grouped(x_rows, y, response, cfg, None)
}

pub fn feature_scores_random_forest_grouped(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    cfg: ExtraTreesConfig,
    feature_group_ids: Option<&[usize]>,
) -> Vec<f64> {
    let (_model, importance) = fit_random_forest(x_rows, y, response, cfg, feature_group_ids);
    importance
}

#[allow(dead_code)]
pub fn feature_scores_random_forest_permutation(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    cfg: ExtraTreesConfig,
    perm_cfg: PermutationConfig,
) -> Vec<f64> {
    feature_scores_random_forest_permutation_grouped(x_rows, y, response, cfg, perm_cfg, None)
}

pub fn feature_scores_random_forest_permutation_grouped(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    cfg: ExtraTreesConfig,
    perm_cfg: PermutationConfig,
    feature_group_ids: Option<&[usize]>,
) -> Vec<f64> {
    let (model, _importance) = fit_random_forest(x_rows, y, response, cfg, feature_group_ids);
    permutation_importance(&model, x_rows, y, response, perm_cfg, cfg.allow_parallel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::common::PermutationScoring;
    use rand::Rng;
    use rand_distr::{Distribution, StandardNormal};

    fn toy_x() -> Vec<Vec<u8>> {
        vec![
            vec![0, 0, 1, 1, 0, 1, 0, 1],
            vec![0, 1, 0, 1, 0, 1, 0, 1],
            vec![0, 0, 0, 1, 1, 1, 0, 1],
            vec![1, 1, 0, 0, 1, 0, 1, 0],
        ]
    }

    fn cfg() -> ExtraTreesConfig {
        ExtraTreesConfig {
            n_estimators: 16,
            max_depth: 3,
            min_samples_leaf: 1,
            min_samples_split: 2,
            bootstrap: true,
            feature_subsample: 0.0,
            seed: 7,
            allow_parallel: true,
        }
    }

    fn random_continuous_dataset(seed: u64) -> (Vec<Vec<u8>>, Vec<f64>, usize) {
        let n_features = 64usize;
        let n_samples = 256usize;
        let lead_feat = 7usize;
        let minor_feat_a = 13usize;
        let minor_feat_b = 21usize;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = vec![vec![0u8; n_samples]; n_features];
        for row in &mut x {
            let maf = rng.random_range(0.2..0.8);
            for v in row.iter_mut() {
                *v = if rng.random::<f64>() < maf { 1 } else { 0 };
            }
        }
        let mut y = vec![0.0_f64; n_samples];
        for (i, yi) in y.iter_mut().enumerate() {
            let noise: f64 = StandardNormal.sample(&mut rng);
            *yi = 4.0 * (x[lead_feat][i] as f64) + 1.25 * (x[minor_feat_a][i] as f64)
                - 0.75 * (x[minor_feat_b][i] as f64)
                + 0.15 * noise;
        }
        (x, y, lead_feat)
    }

    fn argmax(values: &[f64]) -> usize {
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (i, &v) in values.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        best_idx
    }

    #[test]
    fn rf_continuous_scores_are_finite() {
        let x = toy_x();
        let y = vec![0.0, 0.2, 0.9, 1.1, 0.1, 1.0, 0.0, 1.2];
        let imp = feature_scores_random_forest(&x, &y, ResponseKind::Continuous, cfg());
        let perm = feature_scores_random_forest_permutation(
            &x,
            &y,
            ResponseKind::Continuous,
            cfg(),
            PermutationConfig {
                n_repeats: 2,
                scoring: PermutationScoring::R2,
                seed: 11,
            },
        );
        assert_eq!(imp.len(), x.len());
        assert_eq!(perm.len(), x.len());
        assert!(imp.iter().all(|v| v.is_finite()));
        assert!(perm.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rf_binary_scores_are_finite() {
        let x = toy_x();
        let y = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let imp = feature_scores_random_forest(&x, &y, ResponseKind::Binary, cfg());
        let perm = feature_scores_random_forest_permutation(
            &x,
            &y,
            ResponseKind::Binary,
            cfg(),
            PermutationConfig {
                n_repeats: 2,
                scoring: PermutationScoring::Accuracy,
                seed: 13,
            },
        );
        assert_eq!(imp.len(), x.len());
        assert_eq!(perm.len(), x.len());
        assert!(imp.iter().all(|v| v.is_finite()));
        assert!(perm.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rf_random_dataset_recovers_lead_feature() {
        let (x, y, lead_feat) = random_continuous_dataset(20260517);
        let imp = feature_scores_random_forest(&x, &y, ResponseKind::Continuous, cfg());
        let perm = feature_scores_random_forest_permutation(
            &x,
            &y,
            ResponseKind::Continuous,
            cfg(),
            PermutationConfig {
                n_repeats: 3,
                scoring: PermutationScoring::R2,
                seed: 29,
            },
        );
        assert_eq!(argmax(&imp), lead_feat);
        assert_eq!(argmax(&perm), lead_feat);
    }

    #[test]
    fn rf_continuous_split_gain_matches_sse_reduction() {
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
        assert_eq!(gain.threshold, 0);
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
        assert!((gain.gain - old_gain).abs() < 1e-12);
        assert!((gain.gain - 36.0).abs() < 1e-12);
    }

    #[test]
    fn rf_dosage_split_can_choose_threshold_one() {
        let feat = vec![0u8, 1, 2, 2];
        let y = vec![3.0, 2.0, -3.0, -4.0];
        let samples = vec![0usize, 1, 2, 3];
        let parent = make_node_stats(&samples, &y, ResponseKind::Continuous);
        let split = split_gain_for_feature(
            feat.as_slice(),
            samples.as_slice(),
            y.as_slice(),
            ResponseKind::Continuous,
            1,
            parent,
        )
        .unwrap();
        assert_eq!(split.threshold, 1);
        assert!(split.gain > 0.0);
    }
}
