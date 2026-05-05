use crate::ml::common::ResponseKind;
use crate::ml::extra_trees::{feature_scores_extra_trees, ExtraTreesConfig};
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
        "corr" | "pearson" => Ok(MlEngine::Corr),
        "mcc" => Ok(MlEngine::Mcc),
        "mean_diff" | "meandiff" | "mean-diff" => Ok(MlEngine::MeanDiff),
        "fisher" | "fisher_score" | "fisherscore" => Ok(MlEngine::Fisher),
        "lgbm" | "lightgbm" => Ok(MlEngine::LightGbm),
        _ => Err(format!(
            "unsupported ML engine: {method}. supported: auto, et, rf, gbdt, corr, mcc, mean_diff, fisher, lightgbm"
        )),
    }
}

pub fn compute_feature_scores(
    x_rows: &[Vec<u8>],
    y: &[f64],
    response: ResponseKind,
    engine: MlEngine,
    cfg: ExtraTreesConfig,
) -> Result<Vec<f64>, String> {
    let resolved = match engine {
        MlEngine::Auto => MlEngine::ExtraTrees,
        other => other,
    };

    let scores = match resolved {
        MlEngine::ExtraTrees | MlEngine::RandomForest | MlEngine::Gbdt => {
            feature_scores_extra_trees(x_rows, y, response, cfg)
        }
        MlEngine::Corr => feature_scores_abs_corr_binary_x(x_rows, y),
        MlEngine::Mcc => feature_scores_abs_mcc_binary_x(x_rows, y),
        MlEngine::MeanDiff => feature_scores_abs_mean_diff_binary_x(x_rows, y),
        MlEngine::Fisher => feature_scores_fisher_binary_x(x_rows, y, response),
        MlEngine::LightGbm => {
            return Err(
                "lightgbm engine is not yet implemented in Rust. please use: et/rf/gbdt/corr/mcc/mean_diff/fisher"
                    .to_string(),
            );
        }
        MlEngine::Auto => unreachable!(),
    };
    Ok(scores)
}
