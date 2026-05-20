use crate::ml::common::{normalize_importance, score_predictions, PermutationConfig, ResponseKind};
use crate::ml::extra_trees::ExtraTreesConfig;
use rand::rngs::StdRng;
use rand::seq::index::sample as sample_index;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;

const GBDT_LEARNING_RATE: f64 = 0.05;
const GBDT_SUBSAMPLE: f64 = 0.7;

#[derive(Clone, Copy, Debug)]
struct RegStats {
    n: usize,
    sum: f64,
    sumsq: f64,
}

#[derive(Clone, Debug)]
enum GbdtNode {
    Leaf(f64),
    Split {
        feat: usize,
        left: Box<GbdtNode>,
        right: Box<GbdtNode>,
    },
}

#[derive(Clone, Debug)]
struct GbdtModel {
    init: f64,
    learning_rate: f64,
    response: ResponseKind,
    trees: Vec<GbdtNode>,
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[inline]
fn logit(p: f64) -> f64 {
    let q = p.clamp(1e-6, 1.0 - 1e-6);
    (q / (1.0 - q)).ln()
}

#[inline]
fn reg_leaf_value(samples: &[usize], target: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples.iter().map(|&i| target[i]).sum::<f64>();
    sum / (samples.len() as f64)
}

#[inline]
fn make_reg_stats(samples: &[usize], target: &[f64]) -> RegStats {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &i in samples {
        let v = target[i];
        sum += v;
        sumsq += v * v;
    }
    RegStats {
        n: samples.len(),
        sum,
        sumsq,
    }
}

#[inline]
fn sse(stats: RegStats) -> f64 {
    if stats.n == 0 {
        return 0.0;
    }
    stats.sumsq - (stats.sum * stats.sum) / (stats.n as f64)
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

fn split_gain_for_feature(
    feat_row: &[u8],
    samples: &[usize],
    target: &[f64],
    min_leaf: usize,
    parent: RegStats,
) -> Option<f64> {
    let mut left_n = 0usize;
    let mut left_sum = 0.0f64;
    let mut left_sumsq = 0.0f64;

    for &si in samples {
        if feat_row[si] == 0 {
            left_n += 1;
            let v = target[si];
            left_sum += v;
            left_sumsq += v * v;
        }
    }

    let right_n = parent.n.saturating_sub(left_n);
    if left_n < min_leaf || right_n < min_leaf {
        return None;
    }

    let left = RegStats {
        n: left_n,
        sum: left_sum,
        sumsq: left_sumsq,
    };
    let right = RegStats {
        n: right_n,
        sum: parent.sum - left_sum,
        sumsq: parent.sumsq - left_sumsq,
    };
    let gain = sse(parent) - sse(left) - sse(right);
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
    target: &[f64],
    cfg: ExtraTreesConfig,
    samples: &[usize],
    depth: usize,
    rng: &mut StdRng,
    importance: &mut [f64],
    mtry: usize,
) -> GbdtNode {
    let leaf = GbdtNode::Leaf(reg_leaf_value(samples, target));
    if samples.len() < cfg.min_samples_split || depth >= cfg.max_depth {
        return leaf;
    }
    let parent = make_reg_stats(samples, target);
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
    let mut best_gain = f64::NEG_INFINITY;
    for feat in feat_candidates {
        if let Some(gain) =
            split_gain_for_feature(&x_rows[feat], samples, target, cfg.min_samples_leaf, parent)
        {
            if gain > best_gain {
                best_gain = gain;
                best_feat = Some(feat);
            }
        }
    }

    let Some(feat) = best_feat else {
        return leaf;
    };
    if !best_gain.is_finite() || best_gain <= 0.0 {
        return leaf;
    }

    let (left_samples, right_samples) = split_samples_by_feature(&x_rows[feat], samples);
    if left_samples.len() < cfg.min_samples_leaf || right_samples.len() < cfg.min_samples_leaf {
        return leaf;
    }

    importance[feat] += best_gain;
    GbdtNode::Split {
        feat,
        left: Box::new(grow_tree(
            x_rows,
            target,
            cfg,
            &left_samples,
            depth + 1,
            rng,
            importance,
            mtry,
        )),
        right: Box::new(grow_tree(
            x_rows,
            target,
            cfg,
            &right_samples,
            depth + 1,
            rng,
            importance,
            mtry,
        )),
    }
}

fn subsample_indices(n_samples: usize, frac: f64, rng: &mut StdRng) -> Vec<usize> {
    let want = ((frac * (n_samples as f64)).round() as usize).clamp(1, n_samples);
    let mut idx: Vec<usize> = (0..n_samples).collect();
    idx.shuffle(rng);
    idx.truncate(want);
    idx
}

fn predict_tree(
    node: &GbdtNode,
    x_rows: &[Vec<u8>],
    sample_idx: usize,
    override_feat: Option<usize>,
    override_row: Option<&[u8]>,
) -> f64 {
    match node {
        GbdtNode::Leaf(v) => *v,
        GbdtNode::Split { feat, left, right } => {
            let bit = if override_feat == Some(*feat) {
                override_row
                    .map(|r| r[sample_idx])
                    .unwrap_or(x_rows[*feat][sample_idx])
            } else {
                x_rows[*feat][sample_idx]
            };
            if bit == 0 {
                predict_tree(left, x_rows, sample_idx, override_feat, override_row)
            } else {
                predict_tree(right, x_rows, sample_idx, override_feat, override_row)
            }
        }
    }
}

fn predict_model(
    model: &GbdtModel,
    x_rows: &[Vec<u8>],
    override_feat: Option<usize>,
    override_row: Option<&[u8]>,
) -> Vec<f64> {
    if x_rows.is_empty() {
        return Vec::new();
    }
    let n_samples = x_rows[0].len();
    let mut raw = vec![model.init; n_samples];
    for tree in &model.trees {
        for (i, ri) in raw.iter_mut().enumerate() {
            *ri += model.learning_rate * predict_tree(tree, x_rows, i, override_feat, override_row);
        }
    }
    match model.response {
        ResponseKind::Continuous => raw,
        ResponseKind::Binary => raw.into_iter().map(sigmoid).collect(),
    }
}

fn fit_gbdt(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    cfg: ExtraTreesConfig,
) -> (GbdtModel, Vec<f64>) {
    let n_features = x_rows.len();
    let n_samples = y.len();
    if n_features == 0 || n_samples == 0 {
        return (
            GbdtModel {
                init: 0.0,
                learning_rate: GBDT_LEARNING_RATE,
                response,
                trees: Vec::new(),
            },
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

    let init = match response {
        ResponseKind::Continuous => y.iter().sum::<f64>() / (n_samples as f64),
        ResponseKind::Binary => logit(y.iter().sum::<f64>() / (n_samples as f64)),
    };
    let mut raw_pred = vec![init; n_samples];
    let mut trees = Vec::<GbdtNode>::with_capacity(n_estimators);
    let mut importance = vec![0.0f64; n_features];
    let subsample = if cfg_use.bootstrap {
        GBDT_SUBSAMPLE
    } else {
        1.0
    };

    for t in 0..n_estimators {
        let residual = match response {
            ResponseKind::Continuous => raw_pred
                .iter()
                .enumerate()
                .map(|(i, &p)| y[i] - p)
                .collect::<Vec<_>>(),
            ResponseKind::Binary => raw_pred
                .iter()
                .enumerate()
                .map(|(i, &f)| y[i] - sigmoid(f))
                .collect::<Vec<_>>(),
        };

        let mut rng =
            StdRng::seed_from_u64(cfg_use.seed ^ ((t as u64).wrapping_mul(0xA24B_AED4_963E_E407)));
        let samples = if subsample < 0.999 {
            subsample_indices(n_samples, subsample, &mut rng)
        } else {
            (0..n_samples).collect::<Vec<_>>()
        };
        let tree = grow_tree(
            x_rows,
            &residual,
            cfg_use,
            &samples,
            0,
            &mut rng,
            &mut importance,
            mtry,
        );
        for (i, pi) in raw_pred.iter_mut().enumerate() {
            *pi += GBDT_LEARNING_RATE * predict_tree(&tree, x_rows, i, None, None);
        }
        trees.push(tree);
    }

    (
        GbdtModel {
            init,
            learning_rate: GBDT_LEARNING_RATE,
            response,
            trees,
        },
        normalize_importance(&importance),
    )
}

fn permutation_importance(
    model: &GbdtModel,
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    perm_cfg: PermutationConfig,
    allow_parallel: bool,
) -> Vec<f64> {
    if x_rows.is_empty() || y.is_empty() {
        return vec![0.0; x_rows.len()];
    }
    let baseline_pred = predict_model(model, x_rows, None, None);
    let baseline_score = score_predictions(y, &baseline_pred, response, perm_cfg.scoring);
    let repeats = perm_cfg.n_repeats.max(1);

    if allow_parallel {
        (0..x_rows.len())
            .into_par_iter()
            .map(|feat| {
                let mut rng = StdRng::seed_from_u64(
                    perm_cfg.seed ^ ((feat as u64).wrapping_mul(0x94D0_49BB_1331_11EB)),
                );
                let mut drops = 0.0f64;
                for _ in 0..repeats {
                    let mut perm_row = x_rows[feat].clone();
                    perm_row.shuffle(&mut rng);
                    let perm_pred = predict_model(model, x_rows, Some(feat), Some(&perm_row));
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
                perm_cfg.seed ^ ((feat as u64).wrapping_mul(0x94D0_49BB_1331_11EB)),
            );
            let mut drops = 0.0f64;
            for _ in 0..repeats {
                let mut perm_row = x_rows[feat].clone();
                perm_row.shuffle(&mut rng);
                let perm_pred = predict_model(model, x_rows, Some(feat), Some(&perm_row));
                let perm_score = score_predictions(y, &perm_pred, response, perm_cfg.scoring);
                drops += baseline_score - perm_score;
            }
            out.push(drops / (repeats as f64));
        }
        out
    }
}

pub fn feature_scores_gbdt(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    cfg: ExtraTreesConfig,
) -> Vec<f64> {
    let (_model, importance) = fit_gbdt(x_rows, y, response, cfg);
    importance
}

pub fn feature_scores_gbdt_permutation(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    cfg: ExtraTreesConfig,
    perm_cfg: PermutationConfig,
) -> Vec<f64> {
    let (model, _importance) = fit_gbdt(x_rows, y, response, cfg);
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
            n_estimators: 20,
            max_depth: 3,
            min_samples_leaf: 1,
            min_samples_split: 2,
            bootstrap: true,
            feature_subsample: 0.0,
            seed: 17,
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
    fn gbdt_continuous_scores_are_finite() {
        let x = toy_x();
        let y = vec![0.0, 0.2, 0.9, 1.1, 0.1, 1.0, 0.0, 1.2];
        let imp = feature_scores_gbdt(&x, &y, ResponseKind::Continuous, cfg());
        let perm = feature_scores_gbdt_permutation(
            &x,
            &y,
            ResponseKind::Continuous,
            cfg(),
            PermutationConfig {
                n_repeats: 2,
                scoring: PermutationScoring::R2,
                seed: 19,
            },
        );
        assert_eq!(imp.len(), x.len());
        assert_eq!(perm.len(), x.len());
        assert!(imp.iter().all(|v| v.is_finite()));
        assert!(perm.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gbdt_binary_scores_are_finite() {
        let x = toy_x();
        let y = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let imp = feature_scores_gbdt(&x, &y, ResponseKind::Binary, cfg());
        let perm = feature_scores_gbdt_permutation(
            &x,
            &y,
            ResponseKind::Binary,
            cfg(),
            PermutationConfig {
                n_repeats: 2,
                scoring: PermutationScoring::Accuracy,
                seed: 23,
            },
        );
        assert_eq!(imp.len(), x.len());
        assert_eq!(perm.len(), x.len());
        assert!(imp.iter().all(|v| v.is_finite()));
        assert!(perm.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gbdt_random_dataset_recovers_lead_feature() {
        let (x, y, lead_feat) = random_continuous_dataset(20260517);
        let imp = feature_scores_gbdt(&x, &y, ResponseKind::Continuous, cfg());
        let perm = feature_scores_gbdt_permutation(
            &x,
            &y,
            ResponseKind::Continuous,
            cfg(),
            PermutationConfig {
                n_repeats: 3,
                scoring: PermutationScoring::R2,
                seed: 31,
            },
        );
        assert_eq!(argmax(&imp), lead_feat);
        assert_eq!(argmax(&perm), lead_feat);
    }
}
