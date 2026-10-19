use super::adapter::{normalize_binary_rows, ranked_site_proposals, validate_context};
use super::types::{
    ChannelEvidence, ChannelProposalBatch, ProposalChannel, ProposalChannelId, ProposalContext,
};
use crate::ml::common::{ImportanceKind, PermutationConfig, PermutationScoring, ResponseKind};
use crate::ml::engine::compute_feature_scores_grouped;
use crate::ml::extra_trees::ExtraTreesConfig;

#[derive(Clone, Copy, Debug)]
pub(crate) struct RfChannel {
    top_k: usize,
    config: ExtraTreesConfig,
}

impl RfChannel {
    pub(crate) const fn new(top_k: usize, config: ExtraTreesConfig) -> Self {
        Self { top_k, config }
    }
}

impl Default for RfChannel {
    fn default() -> Self {
        Self::new(
            64,
            ExtraTreesConfig {
                n_estimators: 100,
                max_depth: 5,
                min_samples_leaf: 1,
                min_samples_split: 2,
                bootstrap: true,
                feature_subsample: 0.0,
                seed: 42,
                allow_parallel: true,
            },
        )
    }
}

impl ProposalChannel for RfChannel {
    fn id(&self) -> ProposalChannelId {
        ProposalChannelId::Rf
    }

    fn propose(&self, context: &ProposalContext<'_>) -> Result<ChannelProposalBatch, String> {
        validate_context(context)?;
        let rows = normalize_binary_rows(context.genotype_rows)?;
        let mut config = self.config;
        config.seed ^= context.replicate_seed;
        let scores = compute_feature_scores_grouped(
            rows.as_slice(),
            context.residual,
            ResponseKind::Continuous,
            crate::ml::engine::MlEngine::RandomForest,
            config,
            ImportanceKind::Imp,
            PermutationConfig {
                n_repeats: 0,
                scoring: PermutationScoring::Auto,
                seed: config.seed,
            },
            None,
        )?;
        let site_proposals = ranked_site_proposals(
            context.marker_ids,
            scores.as_slice(),
            self.id(),
            self.top_k,
            |importance| ChannelEvidence::Rf {
                importance,
                tree_prevalence: None,
                leaf_mass: None,
                supporting_trees: None,
                min_depth: None,
            },
        );
        Ok(ChannelProposalBatch {
            channel: self.id(),
            window_id: context.window_id.to_string(),
            site_proposals,
            rule_proposals: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garfield::proposal::types::ProposalContext;
    use crate::ml::extra_trees::ExtraTreesConfig;

    fn context<'a>(
        marker_ids: &'a [usize],
        genotype_rows: &'a [Vec<u8>],
        residual: &'a [f64],
        maf: &'a [f64],
    ) -> ProposalContext<'a> {
        ProposalContext {
            window_id: "rf-test",
            marker_ids,
            genotype_rows,
            residual,
            maf,
            ld_r2: 0.8,
            max_order: 3,
            replicate_seed: 23,
        }
    }

    fn config() -> ExtraTreesConfig {
        ExtraTreesConfig {
            n_estimators: 8,
            max_depth: 3,
            min_samples_leaf: 1,
            min_samples_split: 2,
            bootstrap: true,
            feature_subsample: 0.0,
            seed: 23,
            allow_parallel: false,
        }
    }

    #[test]
    fn rf_fixed_seed_is_reproducible() {
        let marker_ids = [101, 205, 309];
        let genotype_rows = vec![
            vec![0_u8, 1, 0, 1, 0, 1, 0, 1],
            vec![0, 0, 1, 1, 0, 0, 1, 1],
            vec![1, 0, 1, 0, 1, 0, 1, 0],
        ];
        let residual = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let maf = [0.5, 0.5, 0.5];
        let context = context(&marker_ids, &genotype_rows, &residual, &maf);
        let first = RfChannel::new(2, config()).propose(&context).unwrap();
        let second = RfChannel::new(2, config()).propose(&context).unwrap();
        assert_eq!(first.site_proposals, second.site_proposals);
    }
}
