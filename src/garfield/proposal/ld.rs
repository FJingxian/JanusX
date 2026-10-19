use std::collections::BTreeMap;

use super::types::{ClusterChannelEvidence, LdCluster, ProposalChannelId, SiteProposal};

pub(crate) fn channel_priority(channel: ProposalChannelId) -> usize {
    match channel {
        ProposalChannelId::Corr => 0,
        ProposalChannelId::Rf => 1,
    }
}

fn binary_r2(lhs: &[u8], rhs: &[u8]) -> Result<f64, String> {
    if lhs.is_empty() || rhs.is_empty() || lhs.len() != rhs.len() {
        return Err("LD rows must be non-empty and have equal length".to_string());
    }
    let n = lhs.len() as f64;
    let mean_l = lhs.iter().map(|&value| f64::from(value > 0)).sum::<f64>() / n;
    let mean_r = rhs.iter().map(|&value| f64::from(value > 0)).sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut variance_l = 0.0;
    let mut variance_r = 0.0;
    for (&value_l, &value_r) in lhs.iter().zip(rhs) {
        let centered_l = f64::from(value_l > 0) - mean_l;
        let centered_r = f64::from(value_r > 0) - mean_r;
        covariance += centered_l * centered_r;
        variance_l += centered_l * centered_l;
        variance_r += centered_r * centered_r;
    }
    if variance_l == 0.0 || variance_r == 0.0 {
        return Ok(
            if lhs
                .iter()
                .map(|&value| value > 0)
                .eq(rhs.iter().map(|&value| value > 0))
            {
                1.0
            } else {
                0.0
            },
        );
    }
    Ok((covariance * covariance / (variance_l * variance_r)).clamp(0.0, 1.0))
}

fn find(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

fn proposal_channel_evidence(
    proposals: &[&SiteProposal],
) -> BTreeMap<ProposalChannelId, ClusterChannelEvidence> {
    let mut grouped: BTreeMap<ProposalChannelId, Vec<&SiteProposal>> = BTreeMap::new();
    for proposal in proposals {
        grouped.entry(proposal.channel).or_default().push(*proposal);
    }
    grouped
        .into_iter()
        .map(|(channel, channel_proposals)| {
            let best_rank = channel_proposals
                .iter()
                .map(|proposal| proposal.rank)
                .min()
                .unwrap_or(usize::MAX);
            let best_percentile = channel_proposals
                .iter()
                .filter_map(|proposal| proposal.percentile)
                .filter(|value| value.is_finite())
                .max_by(|lhs, rhs| lhs.total_cmp(rhs));
            (
                channel,
                ClusterChannelEvidence {
                    best_rank,
                    best_percentile,
                    n_sources: channel_proposals.len(),
                },
            )
        })
        .collect()
}

fn representative_for_cluster<'a>(
    members: &[usize],
    proposals_by_marker: &'a BTreeMap<usize, Vec<&'a SiteProposal>>,
) -> usize {
    members
        .iter()
        .copied()
        .min_by(|lhs, rhs| {
            let lhs_key = proposals_by_marker
                .get(lhs)
                .and_then(|proposals| {
                    proposals
                        .iter()
                        .min_by_key(|proposal| (proposal.rank, channel_priority(proposal.channel)))
                        .map(|proposal| (proposal.rank, channel_priority(proposal.channel), *lhs))
                })
                .unwrap_or((usize::MAX, usize::MAX, *lhs));
            let rhs_key = proposals_by_marker
                .get(rhs)
                .and_then(|proposals| {
                    proposals
                        .iter()
                        .min_by_key(|proposal| (proposal.rank, channel_priority(proposal.channel)))
                        .map(|proposal| (proposal.rank, channel_priority(proposal.channel), *rhs))
                })
                .unwrap_or((usize::MAX, usize::MAX, *rhs));
            lhs_key.cmp(&rhs_key)
        })
        .expect("LD cluster cannot be empty")
}

/// Build transitive LD credit-set clusters from channel site proposals.
///
/// The input rows are keyed by global marker ID.  Every proposal marker must
/// have a row so that a missing alignment cannot silently create a singleton
/// cluster.  A cluster is connected when its members are linked by pairwise
/// binary r² at or above `ld_r2`; the transitive closure is intentional and
/// matches credit-set semantics.
pub(crate) fn cluster_site_proposals(
    proposals: &[SiteProposal],
    marker_rows: &BTreeMap<usize, Vec<u8>>,
    ld_r2: f64,
) -> Result<Vec<LdCluster>, String> {
    if !ld_r2.is_finite() || !(0.0..=1.0).contains(&ld_r2) {
        return Err(format!("ld_r2 must be in [0,1], got {ld_r2}"));
    }
    if proposals.is_empty() {
        return Ok(Vec::new());
    }

    let mut proposals_by_marker: BTreeMap<usize, Vec<&SiteProposal>> = BTreeMap::new();
    for proposal in proposals {
        if !marker_rows.contains_key(&proposal.marker_id) {
            return Err(format!(
                "missing genotype row for proposal marker {}",
                proposal.marker_id
            ));
        }
        proposals_by_marker
            .entry(proposal.marker_id)
            .or_default()
            .push(proposal);
    }
    let marker_ids = proposals_by_marker.keys().copied().collect::<Vec<_>>();
    let mut parent = (0..marker_ids.len()).collect::<Vec<_>>();
    for lhs_idx in 0..marker_ids.len() {
        for rhs_idx in (lhs_idx + 1)..marker_ids.len() {
            let lhs = marker_rows
                .get(&marker_ids[lhs_idx])
                .expect("validated marker row");
            let rhs = marker_rows
                .get(&marker_ids[rhs_idx])
                .expect("validated marker row");
            if binary_r2(lhs, rhs)? >= ld_r2 {
                let lhs_root = find(&mut parent, lhs_idx);
                let rhs_root = find(&mut parent, rhs_idx);
                if lhs_root != rhs_root {
                    parent[rhs_root] = lhs_root;
                }
            }
        }
    }

    let mut members_by_root: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, marker_id) in marker_ids.iter().copied().enumerate() {
        let root = find(&mut parent, index);
        members_by_root.entry(root).or_default().push(marker_id);
    }

    members_by_root
        .into_values()
        .enumerate()
        .map(|(cluster_id, mut members)| {
            members.sort_unstable();
            let proposals_in_cluster = members
                .iter()
                .flat_map(|marker_id| {
                    proposals_by_marker
                        .get(marker_id)
                        .into_iter()
                        .flatten()
                        .copied()
                })
                .collect::<Vec<_>>();
            let representative_marker_id =
                representative_for_cluster(&members, &proposals_by_marker);
            Ok(LdCluster {
                cluster_id,
                representative_marker_id,
                members,
                channel_evidence: proposal_channel_evidence(&proposals_in_cluster),
            })
        })
        .collect()
}

pub(crate) fn cluster_proxy_map(clusters: &[LdCluster]) -> BTreeMap<usize, usize> {
    clusters
        .iter()
        .flat_map(|cluster| {
            cluster
                .members
                .iter()
                .copied()
                .map(|marker_id| (marker_id, cluster.representative_marker_id))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garfield::proposal::types::{ChannelEvidence, ProposalChannelId, SiteProposal};

    fn site_proposals_for_rows(ids: [usize; 2], scores: [f64; 2]) -> Vec<SiteProposal> {
        ids.into_iter()
            .zip(scores)
            .enumerate()
            .map(|(idx, (marker_id, score))| SiteProposal {
                marker_id,
                channel: ProposalChannelId::Rf,
                rank: idx + 1,
                percentile: Some(1.0 - idx as f64 / 2.0),
                evidence: ChannelEvidence::Rf {
                    importance: score,
                    tree_prevalence: None,
                    leaf_mass: None,
                    supporting_trees: None,
                    min_depth: None,
                },
            })
            .collect()
    }

    fn binary_rows_for_high_ld_pair() -> BTreeMap<usize, Vec<u8>> {
        BTreeMap::from([(11, vec![0, 1, 0, 1, 0, 1]), (17, vec![0, 1, 0, 1, 0, 1])])
    }

    #[test]
    fn high_ld_members_share_one_credit_set() {
        let proposals = site_proposals_for_rows([11, 17], [0.9, 0.8]);
        let clusters =
            cluster_site_proposals(&proposals, &binary_rows_for_high_ld_pair(), 0.8).unwrap();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members, vec![11, 17]);
    }
}
