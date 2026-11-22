//! Genetic-map interpolation and bounded overlapping-window iteration.

use super::types::{
    normalize_region_chromosome, ImpConfig, PhaseWindow, PhasedRow, PhasedWindow, RegionFilter,
    VariantRow,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub(crate) struct GeneticMap {
    by_chromosome: HashMap<String, Vec<(u64, f64)>>,
    use_fallback: bool,
}

impl GeneticMap {
    pub(crate) fn from_plink(path: &Path) -> Result<Self, String> {
        let file =
            File::open(path).map_err(|e| format!("open genetic map {}: {e}", path.display()))?;
        let reader = BufReader::new(file);
        let mut by_chromosome: HashMap<String, Vec<(u64, f64)>> = HashMap::new();
        for (line_idx, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| format!("{}:{}: {e}", path.display(), line_idx + 1))?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let fields = trimmed.split_whitespace().collect::<Vec<_>>();
            if fields.len() < 4 {
                // A single textual column header is accepted; data rows must
                // follow the PLINK CHR SNP CM BP layout.
                if line_idx == 0 && fields.iter().any(|value| value.eq_ignore_ascii_case("chr")) {
                    continue;
                }
                return Err(format!(
                    "{}:{}: expected PLINK map columns CHR SNP CM BP",
                    path.display(),
                    line_idx + 1
                ));
            }
            let Ok(cm) = fields[2].parse::<f64>() else {
                if by_chromosome.is_empty() {
                    continue;
                }
                return Err(format!(
                    "{}:{}: invalid genetic position {}",
                    path.display(),
                    line_idx + 1,
                    fields[2]
                ));
            };
            let pos = fields[3].parse::<u64>().map_err(|_| {
                format!(
                    "{}:{}: invalid physical position {}",
                    path.display(),
                    line_idx + 1,
                    fields[3]
                )
            })?;
            if pos == 0 || !cm.is_finite() || cm < 0.0 {
                return Err(format!(
                    "{}:{}: map positions must be positive bp and finite nonnegative cM",
                    path.display(),
                    line_idx + 1
                ));
            }
            let points = by_chromosome.entry(fields[0].to_owned()).or_default();
            if let Some(&(previous_pos, previous_cm)) = points.last() {
                if pos < previous_pos || cm < previous_cm {
                    return Err(format!(
                        "{}:{}: map positions must be nondecreasing within each chromosome",
                        path.display(),
                        line_idx + 1
                    ));
                }
            }
            points.push((pos, cm));
        }
        if by_chromosome.is_empty() {
            return Err(format!(
                "{}: genetic map contains no data rows",
                path.display()
            ));
        }
        Ok(Self {
            by_chromosome,
            use_fallback: false,
        })
    }

    pub(crate) fn default_1cm_per_mb() -> Self {
        Self {
            by_chromosome: HashMap::new(),
            use_fallback: true,
        }
    }

    pub(crate) fn cm_at(&self, chrom: &str, pos: u64) -> Result<f64, String> {
        let Some(points) = self.by_chromosome.get(chrom) else {
            if self.use_fallback {
                return Ok(pos as f64 / 1_000_000.0);
            }
            return Err(format!("genetic map has no chromosome {chrom}"));
        };
        if points.len() == 1 {
            return Ok(points[0].1 + (pos as f64 - points[0].0 as f64) / 1_000_000.0);
        }
        let upper = points.partition_point(|(bp, _)| *bp < pos);
        let (left, right) = if upper == 0 {
            let min_cm = points[0].1 + 5.0;
            let right = points.partition_point(|(_, map_cm)| *map_cm < min_cm);
            (points[0], points[right.min(points.len() - 1)])
        } else if upper >= points.len() {
            let last = points.len() - 1;
            let min_cm = points[last].1 - 5.0;
            let lower = points.partition_point(|(_, map_cm)| *map_cm < min_cm);
            let left = if lower < points.len() && points[lower].1 == min_cm {
                lower
            } else {
                lower.saturating_sub(1)
            }
            .min(last - 1);
            (points[left], points[last])
        } else if points[upper].0 == pos {
            return Ok(points[upper].1);
        } else {
            (points[upper - 1], points[upper])
        };
        if left.0 == right.0 {
            return Ok(left.1);
        }
        let fraction = (pos as f64 - left.0 as f64) / (right.0 as f64 - left.0 as f64);
        Ok(left.1 + fraction * (right.1 - left.1))
    }

    /// Convert a genetic position to the nearest base position, matching
    /// Beagle's `GeneticMap.basePos` boundary conversion.
    pub(crate) fn base_pos_at(&self, chrom: &str, cm: f64) -> Result<u64, String> {
        if !cm.is_finite() {
            return Err("genetic map boundary must be finite".to_owned());
        }
        let rounded = if self.use_fallback {
            cm * 1_000_000.0
        } else {
            let points = self
                .by_chromosome
                .get(chrom)
                .ok_or_else(|| format!("genetic map has no chromosome {chrom}"))?;
            if points.is_empty() {
                return Err(format!(
                    "genetic map has no positions for chromosome {chrom}"
                ));
            }
            if points.len() == 1 {
                points[0].0 as f64 + (cm - points[0].1) * 1_000_000.0
            } else {
                let last = points.len() - 1;
                let upper = points.partition_point(|(_, map_cm)| *map_cm < cm);
                if upper < points.len() && points[upper].1 == cm {
                    return Ok(points[upper].0);
                }
                let (left, right) = if cm < points[0].1 {
                    let threshold = points[0].1 + 5.0;
                    let upper = points.partition_point(|(_, map_cm)| *map_cm < threshold);
                    (0, upper.min(last))
                } else if cm > points[last].1 {
                    let threshold = points[last].1 - 5.0;
                    let upper = points.partition_point(|(_, map_cm)| *map_cm < threshold);
                    (upper.saturating_sub(1).min(last - 1), last)
                } else {
                    (upper.saturating_sub(1).min(last - 1), upper.min(last))
                };
                let (left_bp, left_cm) = points[left];
                let (right_bp, right_cm) = points[right];
                if left_cm == right_cm {
                    left_bp as f64
                } else {
                    left_bp as f64
                        + ((cm - left_cm) / (right_cm - left_cm))
                            * (right_bp as f64 - left_bp as f64)
                }
            }
        };
        if !rounded.is_finite() || rounded < 0.0 || rounded > u64::MAX as f64 {
            return Err(format!(
                "genetic position {cm} cM has no valid base position"
            ));
        }
        Ok(rounded.round() as u64)
    }
}

/// Stateful window reader. Only the previous overlap, the active window, and
/// one pending row are retained; rows outside this range are never revisited.
pub(crate) struct WindowCursor<I>
where
    I: Iterator<Item = Result<VariantRow, String>>,
{
    rows: I,
    map: GeneticMap,
    window_cm: f64,
    overlap_cm: f64,
    window_markers: usize,
    pending: Option<(VariantRow, f64)>,
    carry: Vec<(VariantRow, f64)>,
    outgoing_phase_carry_len: usize,
    window_index: u64,
    last_chromosome: Option<String>,
    last_pos: u64,
    last_cm: f64,
    exhausted: bool,
    seen_chromosomes: HashMap<String, ()>,
    region_filter: Option<RegionFilter>,
    seen_region_chroms: HashMap<String, ()>,
    last_region_chrom: Option<String>,
    last_window_region_interval: Option<(String, usize)>,
}

impl<I> WindowCursor<I>
where
    I: Iterator<Item = Result<VariantRow, String>>,
{
    pub(crate) fn new(rows: I, map: GeneticMap, config: &ImpConfig) -> Self {
        Self {
            rows,
            map,
            window_cm: config.window,
            overlap_cm: config.overlap,
            window_markers: config.window_markers,
            pending: None,
            carry: Vec::new(),
            outgoing_phase_carry_len: 0,
            window_index: 0,
            last_chromosome: None,
            last_pos: 0,
            last_cm: f64::NEG_INFINITY,
            exhausted: false,
            seen_chromosomes: HashMap::new(),
            region_filter: config.region_filter(),
            seen_region_chroms: HashMap::new(),
            last_region_chrom: None,
            last_window_region_interval: None,
        }
    }

    pub(crate) fn next_window(&mut self) -> Result<Option<PhaseWindow>, String> {
        self.next_window_with_admission(&mut |_, _| Ok(()))
    }

    pub(crate) fn next_row_for_reference(&mut self) -> Result<Option<(VariantRow, f64)>, String> {
        self.read_next()
    }

    /// Number of rows at the start of the outgoing row overlap whose phase
    /// must be fixed in the next window, following Beagle's midpoint splice.
    pub(crate) fn outgoing_phase_carry_len(&self) -> usize {
        self.outgoing_phase_carry_len
    }

    /// Returns whether the window just returned by `next_window` is the last
    /// one on its chromosome, including a chromosome before the input ends.
    pub(crate) fn is_terminal_window(&self, chrom: &str) -> bool {
        self.carry.is_empty()
            && (self.exhausted
                || self.pending.as_ref().is_some_and(|(row, _)| {
                    row.meta.chrom != chrom
                        || self.region_filter.as_ref().is_some_and(|filter| {
                            let pending_key = filter
                                .interval_for(&row.meta.chrom, row.meta.pos)
                                .map(|(idx, _, _)| {
                                    (normalize_region_chromosome(&row.meta.chrom), idx)
                                });
                            pending_key != self.last_window_region_interval
                        })
                }))
    }

    pub(crate) fn next_window_with_admission<F>(
        &mut self,
        admission: &mut F,
    ) -> Result<Option<PhaseWindow>, String>
    where
        F: FnMut(usize, &VariantRow) -> Result<(), String>,
    {
        if self.exhausted && self.carry.is_empty() && self.pending.is_none() {
            return Ok(None);
        }
        let mut window_rows = std::mem::take(&mut self.carry);
        let retained_overlap_rows = window_rows.len();
        let incoming_phase_carry_len = self.outgoing_phase_carry_len;
        if retained_overlap_rows > 0 && self.pending.is_none() && !self.exhausted {
            self.pending = self.read_next()?;
            if self.pending.is_none() {
                self.exhausted = true;
            }
        }
        let chrom = if let Some((row, cm)) = self.pending.take() {
            if window_rows.is_empty() {
                window_rows.push((row, cm));
            } else {
                self.pending = Some((row, cm));
            }
            window_rows[0].0.meta.chrom.clone()
        } else if let Some(first) = window_rows.first() {
            first.0.meta.chrom.clone()
        } else {
            let Some((row, cm)) = self.read_next()? else {
                self.exhausted = true;
                return Ok(None);
            };
            let chrom = row.meta.chrom.clone();
            window_rows.push((row, cm));
            chrom
        };
        if window_rows.is_empty() {
            return Ok(None);
        }
        for (row, _) in &window_rows {
            admission(window_rows.len(), row)?;
        }
        let window_region_interval = self.region_filter.as_ref().and_then(|filter| {
            let first = &window_rows[0].0.meta;
            filter
                .interval_for(&first.chrom, first.pos)
                .map(|(idx, _, _)| (normalize_region_chromosome(&first.chrom), idx))
        });
        let start_cm = window_rows[0].1;
        let end_cm = if retained_overlap_rows > 0 {
            self.pending
                .as_ref()
                .filter(|(row, _)| row.meta.chrom == chrom)
                .map(|(_, cm)| *cm + self.window_cm - self.overlap_cm)
                .unwrap_or(start_cm + self.window_cm)
        } else {
            start_cm + self.window_cm
        };
        let end_pos = self.map.base_pos_at(&chrom, end_cm)?;
        let mut hit_chromosome_boundary = false;
        let mut hit_cm_boundary = false;
        while window_rows.len() < self.window_markers {
            if self.exhausted {
                break;
            }
            let next = self.read_next()?;
            let Some((row, cm)) = next else {
                self.exhausted = true;
                break;
            };
            let same_region_interval = match (self.region_filter.as_ref(), &window_region_interval) {
                (Some(filter), Some(expected)) => filter
                    .interval_for(&row.meta.chrom, row.meta.pos)
                    .is_some_and(|(idx, _, _)| {
                        expected.0 == normalize_region_chromosome(&row.meta.chrom)
                            && expected.1 == idx
                    }),
                _ => true,
            };
            if row.meta.chrom != chrom || !same_region_interval {
                self.pending = Some((row, cm));
                hit_chromosome_boundary = true;
                break;
            }
            if row.meta.pos >= end_pos {
                self.pending = Some((row, cm));
                hit_cm_boundary = true;
                break;
            }
            window_rows.push((row, cm));
            admission(window_rows.len(), &window_rows.last().unwrap().0)?;
        }
        let marker_cap_reached = window_rows.len() >= self.window_markers;
        let no_new_rows_after_overlap = window_rows.len() == retained_overlap_rows;
        let terminal = self.exhausted
            || hit_chromosome_boundary
            || (hit_cm_boundary && no_new_rows_after_overlap);
        let (carry_start, emit_end, outgoing_phase_carry_len) = if terminal {
            (window_rows.len(), window_rows.len(), 0)
        } else {
            let overlap_start_cm =
                (window_rows.last().unwrap().1 - self.overlap_cm).max(window_rows[0].1);
            let overlap_start_pos = self.map.base_pos_at(&chrom, overlap_start_cm)?;
            let low = window_rows
                .len()
                .saturating_sub((self.window_markers >> 2).max(1));
            let insertion = low
                + window_rows[low..].partition_point(|(row, _)| row.meta.pos < overlap_start_pos);
            let candidate = if insertion < window_rows.len()
                && window_rows[insertion].0.meta.pos == overlap_start_pos
            {
                insertion
            } else {
                insertion.saturating_sub(1)
            };
            let candidate_pos = window_rows[candidate].0.meta.pos;
            let overlap_start =
                window_rows.partition_point(|(row, _)| row.meta.pos < candidate_pos);
            // Beagle divides the overlap in half: the preceding window emits
            // through this midpoint, while only the prefix from overlap_start
            // to the midpoint is fixed as phase carry in the next window.
            let splice = (window_rows.len() + overlap_start) >> 1;
            (overlap_start, splice, splice - overlap_start)
        };
        let (rows, cm_positions): (Vec<_>, Vec<_>) = window_rows.into_iter().unzip();
        let mut retained = rows[carry_start..]
            .iter()
            .cloned()
            .zip(cm_positions[carry_start..].iter().copied())
            .collect::<Vec<_>>();
        let mut outgoing_phase_carry_len = outgoing_phase_carry_len;
        if marker_cap_reached && retained.len() >= self.window_markers {
            let removed = retained.len() - (self.window_markers - 1);
            retained.drain(..removed);
            outgoing_phase_carry_len = outgoing_phase_carry_len.saturating_sub(removed);
        }
        self.carry = retained;
        self.outgoing_phase_carry_len = outgoing_phase_carry_len;
        self.last_window_region_interval = window_region_interval;
        let index = self.window_index;
        self.window_index = self.window_index.saturating_add(1);
        Ok(Some(PhaseWindow {
            chrom,
            index,
            rows,
            cm_positions,
            incoming_overlap_len: incoming_phase_carry_len,
            emit_start: incoming_phase_carry_len,
            emit_end,
        }))
    }

    fn read_next(&mut self) -> Result<Option<(VariantRow, f64)>, String> {
        loop {
            let next = if let Some(pending) = self.pending.take() {
                Some(pending)
            } else {
                let Some(row) = self.rows.next() else {
                    return Ok(None);
                };
                Some((row?, 0.0))
            };
            let Some((row, pending_cm)) = next else {
                return Ok(None);
            };
            let chrom = row.meta.chrom.clone();
            if let Some(previous_chrom) = self.last_chromosome.as_ref() {
                if chrom == *previous_chrom {
                    if row.meta.pos < self.last_pos {
                        return Err(format!(
                            "input positions are not sorted: {}:{} follows {}:{}",
                            chrom, row.meta.pos, previous_chrom, self.last_pos
                        ));
                    }
                } else {
                    self.seen_chromosomes.insert(previous_chrom.clone(), ());
                    if self.seen_chromosomes.contains_key(&chrom) {
                        return Err(format!(
                            "input chromosome {chrom} reappears after another chromosome"
                        ));
                    }
                    self.last_cm = f64::NEG_INFINITY;
                }
            }
            let cm = if pending_cm != 0.0 {
                pending_cm
            } else {
                self.map.cm_at(&chrom, row.meta.pos)?
            };
            if cm + f64::EPSILON < self.last_cm {
                return Err(format!(
                    "genetic positions decrease at {}:{} ({} cM after {} cM)",
                    chrom, row.meta.pos, cm, self.last_cm
                ));
            }
            self.last_chromosome = Some(chrom.clone());
            self.last_pos = row.meta.pos;
            self.last_cm = cm;

            let Some(region_filter) = self.region_filter.as_ref() else {
                return Ok(Some((row, cm)));
            };
            let chrom_key = normalize_region_chromosome(&chrom);
            if !region_filter.contains_chromosome(&chrom) {
                if let Some(prev_region_chrom) = self.last_region_chrom.take() {
                    self.seen_region_chroms.insert(prev_region_chrom, ());
                }
                if self.seen_region_chroms.len() >= region_filter.chrom_count() {
                    return Ok(None);
                }
                continue;
            }
            if self
                .last_region_chrom
                .as_deref()
                .is_some_and(|prev| prev != chrom_key)
            {
                if let Some(prev_region_chrom) = self.last_region_chrom.take() {
                    self.seen_region_chroms.insert(prev_region_chrom, ());
                }
            }
            self.last_region_chrom = Some(chrom_key.clone());
            let Some((_, start, end)) = region_filter.interval_for(&chrom, row.meta.pos) else {
                if region_filter.is_past_chrom_bounds(&chrom, row.meta.pos) {
                    self.seen_region_chroms.insert(chrom_key, ());
                    self.last_region_chrom = None;
                    if self.seen_region_chroms.len() >= region_filter.chrom_count() {
                        return Ok(None);
                    }
                }
                continue;
            };
            // An explicit -bimrange is a hard analysis boundary, as Beagle's
            // chrom=CHROM:START-END argument is. Expanding the boundary by the
            // window length can join distant disjoint benchmark windows into
            // one chromosome-wide HMM.
            if let Some(start_bp) = start {
                let context_start_cm = self.map.cm_at(&chrom, start_bp)?;
                if cm + f64::EPSILON < context_start_cm {
                    continue;
                }
            }
            if let Some(end_bp) = end {
                let context_end_cm = self.map.cm_at(&chrom, end_bp)?;
                if cm - f64::EPSILON > context_end_cm {
                    continue;
                }
            }
            return Ok(Some((row, cm)));
        }
    }
}

pub(crate) fn cursor_from_reader<I>(rows: I, config: &ImpConfig) -> Result<WindowCursor<I>, String>
where
    I: Iterator<Item = Result<VariantRow, String>>,
{
    let map = map_from_config(config)?;
    Ok(WindowCursor::new(rows, map, config))
}

pub(crate) fn map_from_config(config: &ImpConfig) -> Result<GeneticMap, String> {
    match config.map.as_deref() {
        Some(path) => GeneticMap::from_plink(path),
        None => Ok(GeneticMap::default_1cm_per_mb()),
    }
}

pub(crate) fn reference_carry_rows(
    reference_window: &PhasedWindow,
    cm_positions: &[f64],
    start_cm: Option<f64>,
) -> Result<Vec<(PhasedRow, f64)>, String> {
    if reference_window.rows.len() != cm_positions.len() {
        return Err(format!(
            "reference carry has {} rows but {} genetic positions",
            reference_window.rows.len(),
            cm_positions.len()
        ));
    }
    let Some(start_cm) = start_cm else {
        return Ok(Vec::new());
    };
    Ok(reference_window
        .rows
        .iter()
        .cloned()
        .zip(cm_positions.iter().copied())
        .filter(|(_, cm)| *cm >= start_cm)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{reference_carry_rows, GeneticMap, WindowCursor};
    use crate::imp::types::{
        DiploidCall, DiploidCalls, ImpConfig, PhasedRow, PhasedWindow, RegionSpec, VariantMeta,
        VariantRow,
    };
    use std::collections::HashMap;

    fn row(position: u64) -> VariantRow {
        row_with_id(position, format!("v{position}"))
    }

    fn row_with_id(position: u64, id: String) -> VariantRow {
        VariantRow {
            meta: VariantMeta {
                chrom: "1".to_owned(),
                pos: position,
                id,
                ref_allele: "A".to_owned(),
                alt_alleles: vec!["C".to_owned()],
            },
            calls: DiploidCalls::from_calls(vec![DiploidCall {
                alleles: [Some(0), Some(0)],
                phased: false,
            }]),
        }
    }

    #[test]
    fn next_window_starts_at_marker_at_or_before_overlap_boundary() {
        let positions = [
            1_000_000, 4_000_000, 8_000_000, 9_000_000, 10_500_000, 12_000_000,
        ];
        let rows = positions.into_iter().map(|pos| Ok(row(pos)));
        let config = ImpConfig {
            window: 10.0,
            overlap: 3.0,
            window_markers: 100,
            ..ImpConfig::default()
        };
        let mut cursor = WindowCursor::new(rows, GeneticMap::default_1cm_per_mb(), &config);

        let first = cursor.next_window().unwrap().unwrap();
        assert_eq!(cursor.outgoing_phase_carry_len(), 2);
        let second = cursor.next_window().unwrap().unwrap();

        // Beagle emits through the midpoint of the overlap. The next window
        // starts at the overlap anchor but emits only after its incoming
        // midpoint splice.
        assert_eq!(first.emit_end, 3);
        assert_eq!(first.rows[first.emit_end].meta.pos, 9_000_000);
        assert_eq!(second.rows[0].meta.pos, 4_000_000);
        assert_eq!(second.emit_start, 2);
        assert_eq!(second.incoming_overlap_len, 2);
    }

    #[test]
    fn next_window_end_is_anchored_to_first_unread_marker() {
        let positions = [
            1_000_000, 4_000_000, 8_000_000, 9_000_000, 10_500_000, 12_000_000, 17_000_000,
            18_500_000, 20_000_000,
        ];
        let rows = positions.into_iter().map(|pos| Ok(row(pos)));
        let config = ImpConfig {
            window: 10.0,
            overlap: 3.0,
            window_markers: 100,
            ..ImpConfig::default()
        };
        let mut cursor = WindowCursor::new(rows, GeneticMap::default_1cm_per_mb(), &config);

        cursor.next_window().unwrap().unwrap();
        let second = cursor.next_window().unwrap().unwrap();

        assert_eq!(second.rows.last().unwrap().meta.pos, 18_500_000);
    }

    #[test]
    fn first_window_clamps_overlap_boundary_before_chromosome_start() {
        let rows = [100, 200, 300].into_iter().map(|pos| Ok(row(pos)));
        let config = ImpConfig {
            window_markers: 2,
            ..ImpConfig::default()
        };
        let mut cursor = WindowCursor::new(rows, GeneticMap::default_1cm_per_mb(), &config);

        assert!(cursor.next_window().unwrap().is_some());
    }

    #[test]
    fn marker_cap_preserves_same_position_rows_and_emits_each_once() {
        let rows = (0..4).map(|index| Ok(row_with_id(3_000_000, format!("v{index}"))));
        let config = ImpConfig {
            window_markers: 3,
            ..ImpConfig::default()
        };
        let mut cursor = WindowCursor::new(rows, GeneticMap::default_1cm_per_mb(), &config);
        let mut emitted = Vec::new();
        while let Some(window) = cursor.next_window().unwrap() {
            emitted.extend(
                window.rows[window.emit_start..window.emit_end]
                    .iter()
                    .map(|row| row.meta.id.clone()),
            );
        }

        assert_eq!(emitted, ["v0", "v1", "v2", "v3"]);
    }

    #[test]
    fn reference_carry_moves_rows_at_or_after_next_window_start() {
        let rows = (0..3)
            .map(|marker| PhasedRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker + 1,
                    id: format!("v{marker}"),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::empty(1),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            })
            .collect();
        let window = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            emit_start: 0,
            emit_end: 3,
        };
        let carry = reference_carry_rows(&window, &[0.1, 0.2, 0.3], Some(0.2)).unwrap();
        assert_eq!(
            carry
                .iter()
                .map(|(row, cm)| (row.meta.pos, *cm))
                .collect::<Vec<_>>(),
            vec![(2, 0.2), (3, 0.3)]
        );
    }

    #[test]
    fn region_cursor_uses_the_requested_range_as_a_hard_hmm_boundary() {
        let positions = [
            1_000_000, 4_000_000, 6_000_000, 8_000_000, 9_000_000, 10_000_000, 12_000_000,
            13_000_000, 15_000_000, 18_000_000,
        ];
        let rows = positions.into_iter().map(|pos| Ok(row(pos)));
        let config = ImpConfig {
            window: 2.0,
            overlap: 0.5,
            window_markers: 100,
            region: Some(RegionSpec {
                chrom: "1".to_owned(),
                start: Some(8_000_000),
                end: Some(10_000_000),
            }),
            ..ImpConfig::default()
        };
        let mut cursor = WindowCursor::new(rows, GeneticMap::default_1cm_per_mb(), &config);
        let mut observed = Vec::new();
        while let Some(window) = cursor.next_window().unwrap() {
            observed.extend(window.rows.iter().map(|row| row.meta.pos));
        }
        observed.sort_unstable();
        observed.dedup();

        assert!(!observed.contains(&6_000_000), "{observed:?}");
        assert!(!observed.contains(&1_000_000), "{observed:?}");
        assert!(observed.contains(&8_000_000), "{observed:?}");
        assert!(observed.contains(&9_000_000), "{observed:?}");
        assert!(observed.contains(&10_000_000), "{observed:?}");
        assert!(!observed.contains(&12_000_000), "{observed:?}");
        assert!(!observed.contains(&13_000_000), "{observed:?}");
        assert!(!observed.contains(&15_000_000), "{observed:?}");
    }

    #[test]
    fn large_window_does_not_join_distant_disjoint_benchmark_windows() {
        let positions = [
            10_000_000,
            11_000_000,
            12_000_000,
            140_000_000,
            141_000_000,
            142_000_000,
        ];
        let rows = positions.into_iter().map(|pos| Ok(row(pos)));
        let config = ImpConfig {
            window: 1_000.0,
            overlap: 1.0,
            window_markers: 100,
            region: Some(RegionSpec {
                chrom: "1".to_owned(),
                start: Some(10_000_000),
                end: Some(12_000_000),
            }),
            ..ImpConfig::default()
        };
        let mut cursor = WindowCursor::new(rows, GeneticMap::default_1cm_per_mb(), &config);
        let mut observed = Vec::new();
        while let Some(window) = cursor.next_window().unwrap() {
            observed.extend(window.rows.iter().map(|row| row.meta.pos));
        }
        observed.sort_unstable();
        observed.dedup();
        assert_eq!(observed, vec![10_000_000, 11_000_000, 12_000_000]);
    }

    #[test]
    fn physical_bimrange_boundary_wins_over_flat_genetic_map() {
        let positions = [8_000_000, 9_000_000, 10_000_000, 10_500_000, 12_000_000];
        let rows = positions.into_iter().map(|pos| Ok(row(pos)));
        let config = ImpConfig {
            window: 2.0,
            overlap: 0.5,
            window_markers: 100,
            region: Some(RegionSpec {
                chrom: "1".to_owned(),
                start: Some(8_000_000),
                end: Some(10_000_000),
            }),
            ..ImpConfig::default()
        };
        let map = GeneticMap {
            by_chromosome: HashMap::from([(
                "1".to_owned(),
                vec![(8_000_000, 1.0), (12_000_000, 1.0)],
            )]),
            use_fallback: false,
        };
        let mut cursor = WindowCursor::new(rows, map, &config);
        let mut observed = Vec::new();
        while let Some(window) = cursor.next_window().unwrap() {
            observed.extend(window.rows.iter().map(|row| row.meta.pos));
        }
        observed.sort_unstable();
        observed.dedup();
        assert_eq!(observed, vec![8_000_000, 9_000_000, 10_000_000]);
    }

    #[test]
    fn region_start_and_end_are_inclusive() {
        let region = RegionSpec {
            chrom: "chr1".to_owned(),
            start: Some(10),
            end: Some(20),
        };
        assert!(!region.contains("1", 9));
        assert!(region.contains("1", 10));
        assert!(region.contains("Chr1", 20));
        assert!(!region.contains("1", 21));
        assert!(!region.contains("2", 15));
    }

    #[test]
    fn multi_region_cursor_separates_disjoint_intervals_and_multiple_chroms() {
        let make_row = |chrom: &str, pos: u64| VariantRow {
            meta: VariantMeta {
                chrom: chrom.to_owned(),
                pos,
                id: format!("{chrom}_{pos}"),
                ref_allele: "A".to_owned(),
                alt_alleles: vec!["C".to_owned()],
            },
            calls: DiploidCalls::from_calls(vec![DiploidCall {
                alleles: [Some(0), Some(0)],
                phased: false,
            }]),
        };
        let rows = [
            make_row("1", 1_000_000),
            make_row("1", 2_000_000),
            make_row("1", 5_000_000),
            make_row("1", 10_000_000),
            make_row("1", 11_000_000),
            make_row("2", 3_000_000),
            make_row("3", 4_000_000),
            make_row("3", 5_000_000),
        ]
        .into_iter()
        .map(Ok);
        let config = ImpConfig {
            window: 1_000.0,
            overlap: 1.0,
            window_markers: 100,
            regions: vec![
                RegionSpec {
                    chrom: "1".to_owned(),
                    start: Some(1_000_000),
                    end: Some(2_000_000),
                },
                RegionSpec {
                    chrom: "1".to_owned(),
                    start: Some(10_000_000),
                    end: Some(11_000_000),
                },
                RegionSpec {
                    chrom: "chr3".to_owned(),
                    start: None,
                    end: None,
                },
            ],
            ..ImpConfig::default()
        };
        let mut cursor = WindowCursor::new(rows, GeneticMap::default_1cm_per_mb(), &config);
        let mut windows = Vec::new();
        while let Some(window) = cursor.next_window().unwrap() {
            windows.push(
                window
                    .rows
                    .iter()
                    .map(|r| (r.meta.chrom.clone(), r.meta.pos))
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(
            windows,
            vec![
                vec![("1".to_owned(), 1_000_000), ("1".to_owned(), 2_000_000)],
                vec![("1".to_owned(), 10_000_000), ("1".to_owned(), 11_000_000)],
                vec![("3".to_owned(), 4_000_000), ("3".to_owned(), 5_000_000)],
            ]
        );
    }
}
