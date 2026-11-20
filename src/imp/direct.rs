//! Per-window eligibility checks for direct phased-target imputation.

use super::types::{PhaseSetId, PhaseWindow, PhasedRow, PhasedWindow};
use crate::imp_reader::PhaseProvenance;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectFallbackReason {
    MissingGenotype,
    UnphasedHeterozygote,
    PhaseSetMismatch,
    PgenNeedsGlobalAssumption,
    UnsupportedCall,
    ForcedNormal,
}

impl DirectFallbackReason {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::MissingGenotype => "missing",
            Self::UnphasedHeterozygote => "unphased-het",
            Self::PhaseSetMismatch => "phase-set",
            Self::PgenNeedsGlobalAssumption => "pgen-global-phase",
            Self::UnsupportedCall => "unsupported",
            Self::ForcedNormal => "forced-normal",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectEligibility {
    Eligible,
    Fallback(DirectFallbackReason),
}

pub(crate) fn classify(
    window: &PhaseWindow,
    provenance: PhaseProvenance,
    assume_global_phase: bool,
) -> DirectEligibility {
    for row in &window.rows {
        if row.meta.alt_alleles.len() != 1 {
            return DirectEligibility::Fallback(DirectFallbackReason::UnsupportedCall);
        }
        for sample in 0..row.calls.len() {
            let call = row.calls.get(sample).unwrap_or_else(|| unreachable!());
            if !call.is_complete() {
                return DirectEligibility::Fallback(DirectFallbackReason::MissingGenotype);
            }
            if call.alleles.into_iter().flatten().any(|allele| allele > 1) {
                return DirectEligibility::Fallback(DirectFallbackReason::UnsupportedCall);
            }
            if call.is_heterozygous() && !call.phased {
                return DirectEligibility::Fallback(DirectFallbackReason::UnphasedHeterozygote);
            }
        }
    }

    if provenance == PhaseProvenance::Pgen && !assume_global_phase {
        return DirectEligibility::Fallback(DirectFallbackReason::PgenNeedsGlobalAssumption);
    }
    if !matches!(provenance, PhaseProvenance::Vcf | PhaseProvenance::Pgen) {
        return DirectEligibility::Fallback(DirectFallbackReason::UnsupportedCall);
    }

    let n_samples = window.rows.first().map_or(0, |row| row.calls.len());
    let has_track = window
        .rows
        .iter()
        .any(|row| row.calls.has_phase_set_track());
    if has_track {
        for sample in 0..n_samples {
            let mut explicit: Option<PhaseSetId> = None;
            let mut implicit = false;
            for row in &window.rows {
                let call = row.calls.get(sample).unwrap_or_else(|| unreachable!());
                if !call.is_heterozygous() {
                    continue;
                }
                match row.calls.phase_set(sample) {
                    Some(phase_set) => {
                        if explicit.is_some_and(|current| current != phase_set) {
                            return DirectEligibility::Fallback(
                                DirectFallbackReason::PhaseSetMismatch,
                            );
                        }
                        explicit = Some(phase_set);
                    }
                    None => implicit = true,
                }
            }
            if implicit && explicit.is_some() {
                return DirectEligibility::Fallback(DirectFallbackReason::PhaseSetMismatch);
            }
        }
    }
    DirectEligibility::Eligible
}

pub(crate) fn phased_window_from_calls(window: &PhaseWindow) -> Result<PhasedWindow, String> {
    let rows = window
        .rows
        .iter()
        .map(|row| {
            if row
                .calls
                .iter()
                .any(|call| !call.is_complete() || (call.is_heterozygous() && !call.phased))
            {
                return Err(format!(
                    "direct phased-target window contains an ineligible call at {}:{}",
                    row.meta.chrom, row.meta.pos
                ));
            }
            Ok(PhasedRow {
                meta: row.meta.clone(),
                calls: row.calls.clone(),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(PhasedWindow {
        chrom: window.chrom.clone(),
        index: window.index,
        rows,
        emit_start: window.emit_start,
        emit_end: window.emit_end,
    })
}

#[cfg(test)]
mod tests {
    use super::{classify, DirectEligibility, DirectFallbackReason};
    use crate::imp::types::{
        DiploidCall, DiploidCalls, PhaseSetId, PhaseWindow, VariantMeta, VariantRow,
    };
    use crate::imp_reader::PhaseProvenance;

    fn row(position: u64, call: DiploidCall, phase_set: Option<PhaseSetId>) -> VariantRow {
        let mut calls = DiploidCalls::from_calls(vec![call]);
        if phase_set.is_some() {
            calls.set_phase_set(0, phase_set).unwrap();
        }
        VariantRow {
            meta: VariantMeta {
                chrom: "1".to_owned(),
                pos: position,
                id: format!("v{position}"),
                ref_allele: "A".to_owned(),
                alt_alleles: vec!["G".to_owned()],
            },
            calls,
        }
    }

    fn window(rows: Vec<VariantRow>) -> PhaseWindow {
        let end = rows.len();
        let cm_positions = (0..end).map(|index| index as f64).collect();
        PhaseWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            cm_positions,
            incoming_overlap_len: 0,
            emit_start: 0,
            emit_end: end,
        }
    }

    fn phased_het() -> DiploidCall {
        DiploidCall {
            alleles: [Some(0), Some(1)],
            phased: true,
        }
    }

    #[test]
    fn complete_phased_vcf_without_ps_is_eligible() {
        let window = window(vec![row(100, phased_het(), None)]);
        assert_eq!(
            classify(&window, PhaseProvenance::Vcf, false),
            DirectEligibility::Eligible
        );
    }

    #[test]
    fn consistent_explicit_ps_is_eligible() {
        let phase_set = Some(PhaseSetId::new(9));
        let window = window(vec![
            row(100, phased_het(), phase_set),
            row(200, phased_het(), phase_set),
        ]);
        assert_eq!(
            classify(&window, PhaseProvenance::Vcf, false),
            DirectEligibility::Eligible
        );
    }

    #[test]
    fn changed_ps_inside_window_falls_back() {
        let window = window(vec![
            row(100, phased_het(), Some(PhaseSetId::new(9))),
            row(200, phased_het(), Some(PhaseSetId::new(10))),
        ]);
        assert_eq!(
            classify(&window, PhaseProvenance::Vcf, false),
            DirectEligibility::Fallback(DirectFallbackReason::PhaseSetMismatch)
        );
    }

    #[test]
    fn unphased_heterozygote_and_missing_call_have_distinct_reasons() {
        let unphased = window(vec![row(
            100,
            DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: false,
            },
            None,
        )]);
        assert_eq!(
            classify(&unphased, PhaseProvenance::Vcf, false),
            DirectEligibility::Fallback(DirectFallbackReason::UnphasedHeterozygote)
        );

        let missing = window(vec![row(100, DiploidCall::missing(true), None)]);
        assert_eq!(
            classify(&missing, PhaseProvenance::Vcf, false),
            DirectEligibility::Fallback(DirectFallbackReason::MissingGenotype)
        );
    }

    #[test]
    fn pgen_requires_assume_global_phase() {
        let window = window(vec![row(100, phased_het(), None)]);
        assert_eq!(
            classify(&window, PhaseProvenance::Pgen, false),
            DirectEligibility::Fallback(DirectFallbackReason::PgenNeedsGlobalAssumption)
        );
        assert_eq!(
            classify(&window, PhaseProvenance::Pgen, true),
            DirectEligibility::Eligible
        );
    }
}
