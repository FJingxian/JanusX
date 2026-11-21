//! Reference-driven window bounds and midpoint ownership.
//!
//! Beagle schedules windows and overlap splices on reference markers, then
//! projects those bounds onto the typed target markers. Target density must
//! not move the reference window or discard its incoming HMM context.

use super::{types, window, ReferenceWindowStream};

type ReferenceRecord = (types::VariantRow, f64, Option<u64>);

pub(super) struct ScheduledWindow {
    pub target: types::PhaseWindow,
    pub reference: (types::PhasedWindow, Vec<f64>, Vec<Option<u64>>),
    pub terminal: bool,
    pub outgoing_phase_carry_len: usize,
    pub previous_emitted_pos: Option<u64>,
    pub reference_emit_end_pos: u64,
}

#[derive(Default)]
pub(super) struct ReferenceWindowSchedule {
    target_pending: Option<(types::VariantRow, f64)>,
    target_carry: Vec<(types::VariantRow, f64)>,
    reference_carry: Vec<ReferenceRecord>,
    incoming_phase_carry_len: usize,
    previous_emitted_pos: Option<u64>,
    index: u64,
}

impl ReferenceWindowSchedule {
    pub(super) fn next_window<I, F>(
        &mut self,
        targets: &mut window::WindowCursor<I>,
        references: &mut ReferenceWindowStream,
        config: &types::ImpConfig,
        admission: &mut F,
    ) -> Result<Option<ScheduledWindow>, String>
    where
        I: Iterator<Item = Result<types::VariantRow, String>>,
        F: FnMut(usize, &types::VariantRow) -> Result<(), String>,
    {
        if self.target_pending.is_none() {
            self.target_pending = targets.next_row_for_reference()?;
        }
        if self.target_carry.is_empty() && self.target_pending.is_none() {
            return Ok(None);
        }
        let first = self
            .target_carry
            .first()
            .or(self.target_pending.as_ref())
            .unwrap();
        let chrom = first.0.meta.chrom.clone();
        let first_target_pos = first.0.meta.pos;
        let output_start = config
            .region
            .as_ref()
            .filter(|region| region.contains_chromosome(&chrom))
            .and_then(|region| region.start)
            .unwrap_or(0);
        let output_end = config
            .region
            .as_ref()
            .filter(|region| region.contains_chromosome(&chrom))
            .and_then(|region| region.end)
            .unwrap_or(u64::MAX);
        let mut reference_rows = std::mem::take(&mut self.reference_carry);
        let overlap_end = reference_rows.len();
        if overlap_end == 0 {
            self.previous_emitted_pos = None;
        }
        // The compression partition is updated for skipped reference records
        // too; seeking into a region must not restart SeqCoder3 at its edge.
        loop {
            let Some(record) = references.read_next()? else {
                return Err(format!(
                    "reference input has no remaining markers for chromosome {chrom}"
                ));
            };
            if record.0.meta.chrom != chrom {
                continue;
            }
            // Region context may start far into a chromosome. Do not create
            // empty reference windows before the first retained target row.
            if overlap_end == 0 && record.0.meta.pos < output_start {
                continue;
            }
            if overlap_end == 0 && record.0.meta.pos > output_end {
                references.pending = Some(record);
                break;
            }
            references.pending = Some(record);
            break;
        }
        let next_reference_cm = references.pending.as_ref().unwrap().1;
        let step = if overlap_end == 0 {
            config.window as f32
        } else {
            config.window as f32 - config.overlap as f32
        };
        let mut end_pos = references
            .map
            .base_pos_at(&chrom, next_reference_cm + f64::from(step))?;
        end_pos = end_pos.min(output_end.saturating_add(1));
        // Preserve the full output prefix even when its first typed anchor is
        // beyond one ordinary window. Reference-only prefix records must not
        // disappear simply because the target panel is sparse.
        if overlap_end == 0 {
            end_pos = end_pos.max(first_target_pos.saturating_add(1));
        }
        while reference_rows.len() < config.window_markers {
            let Some(record) = references.read_next()? else {
                break;
            };
            if record.0.meta.chrom != chrom
                || record.0.meta.pos >= end_pos
                || record.0.meta.pos > output_end
            {
                references.pending = Some(record);
                break;
            }
            reference_rows.push(record);
        }
        if reference_rows.is_empty() {
            return Err(format!("reference window on chromosome {chrom} is empty"));
        }
        let capped = reference_rows.len() == config.window_markers;
        if references.pending.is_none() {
            references.pending = references.read_next()?;
        }
        let terminal_reference = references
            .pending
            .as_ref()
            .map_or(true, |row| row.0.meta.chrom != chrom);
        let target_end_pos = if terminal_reference {
            u64::MAX
        } else if capped {
            reference_rows.last().unwrap().0.meta.pos.saturating_add(1)
        } else {
            end_pos
        };
        let mut target_rows = std::mem::take(&mut self.target_carry);
        for (row, _) in &target_rows {
            admission(target_rows.len(), row)?;
        }
        loop {
            let next = match self.target_pending.take() {
                Some(row) => Some(row),
                None => targets.next_row_for_reference()?,
            };
            let Some(record) = next else { break };
            if record.0.meta.chrom != chrom
                || (!terminal_reference && record.0.meta.pos >= target_end_pos)
            {
                self.target_pending = Some(record);
                break;
            }
            target_rows.push(record);
            admission(target_rows.len(), &target_rows.last().unwrap().0)?;
            if target_rows.len() > config.window_markers {
                return Err("target/reference window exceeds --window-markers".to_owned());
            }
        }
        if target_rows.is_empty() {
            return Err(format!(
                "target and reference have no target records in the window on chromosome {chrom}"
            ));
        }
        let terminal_target = self
            .target_pending
            .as_ref()
            .map_or(true, |row| row.0.meta.chrom != chrom);
        let terminal = terminal_target || terminal_reference;
        // Preserve JanusX's complete reference endpoint output contract.
        if terminal_target && !terminal_reference {
            let context_end = config
                .region
                .as_ref()
                .filter(|region| region.contains_chromosome(&chrom))
                .and_then(|region| region.end)
                .map_or(u64::MAX, |end| {
                    end.max(target_rows.last().unwrap().0.meta.pos)
                });
            loop {
                let Some(record) = references.read_next()? else {
                    break;
                };
                if record.0.meta.chrom != chrom || record.0.meta.pos > context_end {
                    references.pending = Some(record);
                    break;
                }
                if reference_rows.len() >= config.window_markers {
                    return Err("terminal reference window exceeds --window-markers".to_owned());
                }
                reference_rows.push(record);
            }
        }
        let overlap_start = if terminal {
            reference_rows.len()
        } else {
            let boundary_cm = references.map.cm_at(&chrom, end_pos.saturating_sub(1))?
                - f64::from(config.overlap as f32);
            let boundary_pos = references.map.base_pos_at(&chrom, boundary_cm)?;
            let low = reference_rows
                .len()
                .saturating_sub((config.window_markers >> 2).max(1));
            let insertion =
                low + reference_rows[low..].partition_point(|r| r.0.meta.pos < boundary_pos);
            if insertion < reference_rows.len()
                && reference_rows[insertion].0.meta.pos == boundary_pos
            {
                insertion
            } else {
                insertion.saturating_sub(1)
            }
        };
        if !terminal && overlap_start == 0 && overlap_end == reference_rows.len() {
            return Err("reference window overlap does not advance; reduce --overlap".to_owned());
        }
        let previous_splice = overlap_end / 2;
        let next_splice = (reference_rows.len() + overlap_start) / 2;
        let before_reference = |index: usize| -> usize {
            reference_rows.get(index).map_or(target_rows.len(), |r| {
                target_rows.partition_point(|t| t.0.meta.pos < r.0.meta.pos)
            })
        };
        let emit_start = if overlap_end == 0 {
            0
        } else {
            before_reference(previous_splice)
        };
        let emit_end = before_reference(next_splice);
        let target_overlap_start = before_reference(overlap_start);
        let outgoing_phase_carry_len = emit_end.saturating_sub(target_overlap_start);
        self.target_carry = target_rows[target_overlap_start..].to_vec();
        self.reference_carry = reference_rows[overlap_start..].to_vec();
        let previous_emitted_pos = self.previous_emitted_pos;
        let reference_emit_end_pos = reference_rows[next_splice.saturating_sub(1)].0.meta.pos;
        self.previous_emitted_pos = Some(reference_emit_end_pos);
        let (rows, cm_positions) = target_rows.into_iter().unzip();
        let target = types::PhaseWindow {
            chrom: chrom.clone(),
            index: self.index,
            rows,
            cm_positions,
            incoming_overlap_len: self.incoming_phase_carry_len,
            emit_start,
            emit_end,
        };
        self.incoming_phase_carry_len = outgoing_phase_carry_len;
        let mut rows = Vec::with_capacity(reference_rows.len());
        let mut positions = Vec::with_capacity(reference_rows.len());
        let mut groups = Vec::with_capacity(reference_rows.len());
        for (row, cm, group) in reference_rows {
            rows.push(types::PhasedRow {
                meta: row.meta,
                calls: row.calls,
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            });
            positions.push(cm);
            groups.push(group);
        }
        let reference = types::PhasedWindow {
            chrom,
            index: self.index,
            emit_start: previous_splice,
            emit_end: next_splice,
            rows,
        };
        self.index += 1;
        Ok(Some(ScheduledWindow {
            target,
            reference: (reference, positions, groups),
            terminal,
            outgoing_phase_carry_len,
            previous_emitted_pos,
            reference_emit_end_pos,
        }))
    }
}
