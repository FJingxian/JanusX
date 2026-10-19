use std::collections::BTreeMap;

use super::config::{ProposalConfig, ProposalMode};
use super::corr::CorrChannel;
use super::merge::merge_and_select_site_pool;
use super::rf::RfChannel;
use super::types::{
    MergedSitePool, ProposalChannel, ProposalChannelId, ProposalContext, SiteProposal,
};

fn dosage_maf(rows: &[Vec<u8>]) -> Vec<f64> {
    rows.iter()
        .map(|row| {
            if row.is_empty() {
                0.0
            } else {
                let allele_frequency = row.iter().map(|&value| f64::from(value)).sum::<f64>()
                    / (2.0 * row.len() as f64);
                allele_frequency.min(1.0 - allele_frequency).clamp(0.0, 0.5)
            }
        })
        .collect()
}

/// Build a Corr+RF pool when Corr scores were computed from packed bitplanes
/// and RF is intentionally restricted to a Corr-ranked shortlist.  The RF
/// channel keeps its normal local ranking/evidence semantics; only the input
/// feature matrix is reduced before tree fitting.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_corr_rf_site_pool_from_ranked_corr(
    config: ProposalConfig,
    window_id: &str,
    marker_ids: &[usize],
    corr_scores: &[f64],
    rf_marker_ids: &[usize],
    rf_genotype_rows: &[Vec<u8>],
    residual: &[f64],
    ld_r2: f64,
    replicate_seed: u64,
) -> Result<MergedSitePool, String> {
    if !matches!(config.mode, ProposalMode::CorrRf) {
        return Err("packed Corr+RF proposal pool requires CorrRf mode".to_string());
    }
    if marker_ids.len() != corr_scores.len() {
        return Err(format!(
            "packed Corr score length mismatch: {} markers vs {} scores",
            marker_ids.len(),
            corr_scores.len()
        ));
    }
    if rf_marker_ids.len() != rf_genotype_rows.len() {
        return Err(format!(
            "RF shortlist marker/row length mismatch: {} vs {}",
            rf_marker_ids.len(),
            rf_genotype_rows.len()
        ));
    }
    let corr_proposals = CorrChannel::ranked_proposals(marker_ids, corr_scores, config.k_site);
    let rf_maf = dosage_maf(rf_genotype_rows);
    let context = ProposalContext {
        window_id,
        marker_ids: rf_marker_ids,
        genotype_rows: rf_genotype_rows,
        residual,
        maf: rf_maf.as_slice(),
        ld_r2,
        max_order: config.max_order,
        replicate_seed,
    };
    let rf_proposals = RfChannel::default().propose(&context)?.site_proposals;
    let mut proposals =
        Vec::<SiteProposal>::with_capacity(corr_proposals.len().saturating_add(rf_proposals.len()));
    proposals.extend(corr_proposals);
    proposals.extend(rf_proposals);
    let marker_rows = rf_marker_ids
        .iter()
        .copied()
        .zip(rf_genotype_rows.iter().cloned())
        .collect::<BTreeMap<_, _>>();
    // Corr's top-k must be contained in the RF shortlist.  Keep the check
    // explicit so a future shortlist policy cannot silently drop an LD row.
    for proposal in proposals.iter() {
        if !marker_rows.contains_key(&proposal.marker_id) {
            return Err(format!(
                "RF shortlist does not contain Corr proposal marker {}",
                proposal.marker_id
            ));
        }
    }
    // The map is intentionally mutable above to make the ownership contract
    // obvious: clustering receives rows for every union proposal and no
    // dense full-window matrix is retained after this function returns.
    merge_and_select_site_pool(
        proposals,
        &marker_rows,
        ld_r2,
        default_channel_quotas(config.mode, config.k_site),
        config.k_site,
    )
}

/// Execute proposal channels and return the final LD-representative site pool.
/// This module owns channel dispatch so the scan code sees one stable contract.
pub(crate) fn build_proposal_site_pool(
    config: ProposalConfig,
    window_id: &str,
    marker_ids: &[usize],
    genotype_rows: &[Vec<u8>],
    residual: &[f64],
    maf: &[f64],
    ld_r2: f64,
    replicate_seed: u64,
) -> Result<MergedSitePool, String> {
    if matches!(config.mode, ProposalMode::Off) {
        return Err("proposal site pool requested while proposal mode is off".to_string());
    }
    let context = ProposalContext {
        window_id,
        marker_ids,
        genotype_rows,
        residual,
        maf,
        ld_r2,
        max_order: config.max_order,
        replicate_seed,
    };
    let mut proposals = Vec::<SiteProposal>::new();
    match config.mode {
        ProposalMode::Off => unreachable!("checked above"),
        ProposalMode::Corr => {
            proposals.extend(CorrChannel::default().propose(&context)?.site_proposals)
        }
        ProposalMode::Rf => {
            proposals.extend(RfChannel::default().propose(&context)?.site_proposals)
        }
        ProposalMode::CorrRf => {
            proposals.extend(CorrChannel::default().propose(&context)?.site_proposals);
            proposals.extend(RfChannel::default().propose(&context)?.site_proposals);
        }
    }
    let marker_rows = marker_ids
        .iter()
        .copied()
        .zip(genotype_rows.iter().cloned())
        .collect::<BTreeMap<_, _>>();
    merge_and_select_site_pool(
        proposals,
        &marker_rows,
        ld_r2,
        default_channel_quotas(config.mode, config.k_site),
        config.k_site,
    )
}

fn default_channel_quotas(mode: ProposalMode, k_site: usize) -> super::merge::ChannelQuota {
    let channels = match mode {
        ProposalMode::Off => Vec::new(),
        ProposalMode::Corr => vec![(ProposalChannelId::Corr, k_site)],
        ProposalMode::Rf => vec![(ProposalChannelId::Rf, k_site)],
        ProposalMode::CorrRf => vec![
            (ProposalChannelId::Corr, k_site.div_ceil(2)),
            (ProposalChannelId::Rf, k_site / 2),
        ],
    };
    super::merge::ChannelQuota::from_entries(channels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_mode_is_not_accidentally_a_proposal_run() {
        let result = build_proposal_site_pool(
            ProposalConfig::default(),
            "test",
            &[0],
            &[vec![0, 1]],
            &[0.0, 1.0],
            &[0.5],
            0.8,
            1,
        );
        assert!(result.is_err());
    }
}
