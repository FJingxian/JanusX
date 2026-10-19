use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum ProposalChannelId {
    Corr,
    Rf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub(crate) struct SourceMask(u8);

impl SourceMask {
    pub(crate) const CORR: Self = Self(1 << 0);
    pub(crate) const RF: Self = Self(1 << 1);

    #[cfg(test)]
    pub(crate) const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    #[cfg(test)]
    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl From<ProposalChannelId> for SourceMask {
    fn from(channel: ProposalChannelId) -> Self {
        match channel {
            ProposalChannelId::Corr => Self::CORR,
            ProposalChannelId::Rf => Self::RF,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct SiteSetKey(Vec<usize>);

impl SiteSetKey {
    #[cfg(test)]
    pub(crate) fn new<const N: usize>(ids: [usize; N]) -> Self {
        Self::from_iter(ids)
    }

    #[cfg(test)]
    pub(crate) fn from_iter<I>(ids: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        let mut ids = ids.into_iter().collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        Self(ids)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ChannelEvidence {
    Corr {
        score: f64,
    },
    Rf {
        importance: f64,
        tree_prevalence: Option<f64>,
        leaf_mass: Option<f64>,
        supporting_trees: Option<usize>,
        min_depth: Option<usize>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SiteProposal {
    pub(crate) marker_id: usize,
    pub(crate) channel: ProposalChannelId,
    pub(crate) rank: usize,
    pub(crate) percentile: Option<f64>,
    pub(crate) evidence: ChannelEvidence,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuleProposal {
    pub(crate) sites: SiteSetKey,
    pub(crate) channel: ProposalChannelId,
    pub(crate) rank: usize,
    pub(crate) percentile: Option<f64>,
    pub(crate) evidence: ChannelEvidence,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ChannelProposalBatch {
    pub(crate) channel: ProposalChannelId,
    pub(crate) window_id: String,
    pub(crate) site_proposals: Vec<SiteProposal>,
    pub(crate) rule_proposals: Vec<RuleProposal>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PairSeedReason {
    ExactRank,
    #[cfg(test)]
    ProposalRescue,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PairSeed {
    pub(crate) marker_ids: [usize; 2],
    pub(crate) reason: PairSeedReason,
    pub(crate) sources: SourceMask,
    pub(crate) source_rank: usize,
    pub(crate) exact_score: f64,
}

impl PairSeed {
    pub(crate) fn new(
        marker_ids: [usize; 2],
        reason: PairSeedReason,
        sources: SourceMask,
        source_rank: usize,
    ) -> Self {
        let marker_ids = if marker_ids[0] <= marker_ids[1] {
            marker_ids
        } else {
            [marker_ids[1], marker_ids[0]]
        };
        Self {
            marker_ids,
            reason,
            sources,
            source_rank,
            exact_score: f64::NAN,
        }
    }
}

pub(crate) struct ProposalContext<'a> {
    pub(crate) window_id: &'a str,
    pub(crate) marker_ids: &'a [usize],
    pub(crate) genotype_rows: &'a [Vec<u8>],
    pub(crate) residual: &'a [f64],
    pub(crate) maf: &'a [f64],
    pub(crate) ld_r2: f64,
    pub(crate) max_order: usize,
    pub(crate) replicate_seed: u64,
}

pub(crate) trait ProposalChannel: Send + Sync {
    fn id(&self) -> ProposalChannelId;
    fn propose(&self, context: &ProposalContext<'_>) -> Result<ChannelProposalBatch, String>;
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ClusterChannelEvidence {
    pub(crate) best_rank: usize,
    pub(crate) best_percentile: Option<f64>,
    pub(crate) n_sources: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LdCluster {
    pub(crate) cluster_id: usize,
    pub(crate) representative_marker_id: usize,
    pub(crate) members: Vec<usize>,
    pub(crate) channel_evidence: BTreeMap<ProposalChannelId, ClusterChannelEvidence>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MergedSitePool {
    pub(crate) representatives: Vec<usize>,
    pub(crate) clusters: Vec<LdCluster>,
    pub(crate) proxy_to_representative: BTreeMap<usize, usize>,
    pub(crate) k_pre: usize,
    pub(crate) k_site: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_context<'a>(
        marker_ids: &'a [usize],
        genotype_rows: &'a [Vec<u8>],
        residual: &'a [f64],
        maf: &'a [f64],
        seed: u64,
    ) -> ProposalContext<'a> {
        ProposalContext {
            window_id: "window-1",
            marker_ids,
            genotype_rows,
            residual,
            maf,
            ld_r2: 0.8,
            max_order: 3,
            replicate_seed: seed,
        }
    }

    #[test]
    fn dedup_key_is_independent_of_source_and_order() {
        let a = SiteSetKey::new([17, 3, 17]);
        let b = SiteSetKey::new([3, 17]);
        assert_eq!(a, b);
    }

    #[test]
    fn pair_seed_keeps_reason_and_source() {
        let seed = PairSeed::new([3, 17], PairSeedReason::ExactRank, SourceMask::RF, 4);
        assert_eq!(seed.reason, PairSeedReason::ExactRank);
        assert!(seed.sources.contains(SourceMask::RF));
    }

    #[test]
    fn proposal_context_preserves_aligned_inputs() {
        let marker_ids = [11, 17];
        let genotype_rows = vec![vec![0_u8, 1], vec![1, 0]];
        let residual = [1.0, -1.0];
        let maf = [0.5, 0.5];
        let context = test_context(&marker_ids, &genotype_rows, &residual, &maf, 9);
        assert_eq!(context.marker_ids, &marker_ids);
        assert_eq!(context.genotype_rows.len(), 2);
        assert_eq!(context.replicate_seed, 9);
    }
}
