use crate::ml::common::{ImportanceKind, PermutationConfig, ResponseKind};
use crate::ml::extra_trees::{feature_scores_extra_trees_grouped, ExtraTreesConfig};
use crate::ml::gbdt::{
    feature_scores_gbdt_grouped, feature_scores_gbdt_permutation_grouped,
};
use crate::ml::rf::{
    feature_scores_random_forest_grouped, feature_scores_random_forest_permutation_grouped,
};
use crate::ml::univariate::{
    feature_scores_abs_corr_binary_x, feature_scores_abs_mcc_binary_x,
    feature_scores_abs_mean_diff_binary_x, feature_scores_fisher_binary_x,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlEngine {
    Auto,
    ExtraTrees,
    RandomForest,
    Gbdt,
    Gbdt2,
    Corr,
    Mcc,
    MeanDiff,
    Fisher,
    LightGbm,
}

pub fn parse_ml_engine(method: &str) -> Result<MlEngine, String> {
    let m = method.trim().to_ascii_lowercase();
    match m.as_str() {
        "" | "auto" => Ok(MlEngine::Auto),
        "et" | "extratrees" | "extra_trees" => Ok(MlEngine::ExtraTrees),
        "rf" | "random_forest" | "randomforest" => Ok(MlEngine::RandomForest),
        "gbdt" | "gradient_boosting" | "gradientboosting" => Ok(MlEngine::Gbdt),
        "gbdt2" | "gradient_boosting2" | "gradientboosting2" => Ok(MlEngine::Gbdt2),
        "corr" | "pearson" => Ok(MlEngine::Corr),
        "mcc" => Ok(MlEngine::Mcc),
        "mean_diff" | "meandiff" | "mean-diff" => Ok(MlEngine::MeanDiff),
        "fisher" | "fisher_score" | "fisherscore" => Ok(MlEngine::Fisher),
        "lgbm" | "lightgbm" => Ok(MlEngine::LightGbm),
        _ => Err(format!(
            "unsupported ML engine: {method}. supported: auto, et, rf, gbdt, gbdt2, corr, mcc, mean_diff, fisher, lightgbm"
        )),
    }
}

pub fn compute_feature_scores(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    engine: MlEngine,
    cfg: ExtraTreesConfig,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
) -> Result<Vec<f64>, String> {
    compute_feature_scores_grouped(
        x_rows,
        y,
        response,
        engine,
        cfg,
        importance,
        perm_cfg,
        None,
    )
}

pub fn compute_feature_scores_grouped(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    engine: MlEngine,
    cfg: ExtraTreesConfig,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    feature_group_ids: Option<&[usize]>,
) -> Result<Vec<f64>, String> {
    if let Some(group_ids) = feature_group_ids {
        if group_ids.len() != x_rows.len() {
            return Err(format!(
                "feature_group_ids length mismatch: got {}, expected {} features",
                group_ids.len(),
                x_rows.len()
            ));
        }
    }

    let resolved = match engine {
        MlEngine::Auto => match importance {
            ImportanceKind::Imp => MlEngine::ExtraTrees,
            ImportanceKind::Permutation => MlEngine::RandomForest,
        },
        other => other,
    };

    let scores = match (resolved, importance) {
        (MlEngine::ExtraTrees, ImportanceKind::Imp) => {
            feature_scores_extra_trees_grouped(x_rows, y, response, cfg, feature_group_ids)
        }
        (MlEngine::ExtraTrees, ImportanceKind::Permutation) => {
            return Err(
                "permutation importance is not implemented for extra_trees; use rf or gbdt"
                    .to_string(),
            );
        }
        (MlEngine::RandomForest, ImportanceKind::Imp) => {
            feature_scores_random_forest_grouped(x_rows, y, response, cfg, feature_group_ids)
        }
        (MlEngine::RandomForest, ImportanceKind::Permutation) => {
            feature_scores_random_forest_permutation_grouped(
                x_rows,
                y,
                response,
                cfg,
                perm_cfg,
                feature_group_ids,
            )
        }
        (MlEngine::Gbdt, ImportanceKind::Imp)
        | (MlEngine::Gbdt2, ImportanceKind::Imp)
        | (MlEngine::LightGbm, ImportanceKind::Imp) => {
            feature_scores_gbdt_grouped(x_rows, y, response, cfg, feature_group_ids)
        }
        (MlEngine::Gbdt, ImportanceKind::Permutation)
        | (MlEngine::Gbdt2, ImportanceKind::Permutation)
        | (MlEngine::LightGbm, ImportanceKind::Permutation) => {
            feature_scores_gbdt_permutation_grouped(
                x_rows,
                y,
                response,
                cfg,
                perm_cfg,
                feature_group_ids,
            )
        }
        (MlEngine::Corr, ImportanceKind::Imp) | (MlEngine::Corr, ImportanceKind::Permutation) => {
            feature_scores_abs_corr_binary_x(x_rows, y)
        }
        (MlEngine::Mcc, ImportanceKind::Imp) | (MlEngine::Mcc, ImportanceKind::Permutation) => {
            feature_scores_abs_mcc_binary_x(x_rows, y)
        }
        (MlEngine::MeanDiff, ImportanceKind::Imp)
        | (MlEngine::MeanDiff, ImportanceKind::Permutation) => {
            feature_scores_abs_mean_diff_binary_x(x_rows, y)
        }
        (MlEngine::Fisher, ImportanceKind::Imp)
        | (MlEngine::Fisher, ImportanceKind::Permutation) => {
            feature_scores_fisher_binary_x(x_rows, y, response)
        }
        (MlEngine::Auto, _) => unreachable!(),
    };
    Ok(scores)
}
