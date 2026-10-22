use crate::ct::input::{GenotypeCall, PhaseSet, Site, SiteSource};
use crate::gfcore::{HmpSnpIter, SiteInfo};
use std::collections::{HashMap, HashSet};

pub(crate) struct HmpSource {
    iter: HmpSnpIter,
    sample_ids: Vec<String>,
    selected_columns: Vec<usize>,
    next_index: usize,
    previous_chrom: Option<String>,
    previous_pos: i64,
    seen_chromosomes: HashSet<String>,
}

impl HmpSource {
    pub(crate) fn open(path: &str, selected_ids: &[String]) -> Result<Self, String> {
        let iter = HmpSnpIter::new_with_fill(path, 0.0, 1.0, false, false, 1.0)?;
        let mut id_to_column = HashMap::with_capacity(iter.samples.len());
        for (index, id) in iter.samples.iter().enumerate() {
            if id.trim().is_empty() {
                return Err(format!(
                    "empty sample ID in HMP header at sample column {}",
                    index + 12
                ));
            }
            if id_to_column.insert(id.clone(), index).is_some() {
                return Err(format!("duplicate HMP sample ID '{id}'"));
            }
        }
        let requested = if selected_ids.is_empty() {
            iter.samples.clone()
        } else {
            selected_ids.to_vec()
        };
        let mut requested_seen = HashSet::with_capacity(requested.len());
        let mut sample_ids = Vec::with_capacity(requested.len());
        let mut selected_columns = Vec::with_capacity(requested.len());
        for id in requested {
            if !requested_seen.insert(id.clone()) {
                return Err(format!("duplicate selected sample ID '{id}'"));
            }
            let column = id_to_column
                .get(&id)
                .copied()
                .ok_or_else(|| format!("selected sample ID '{id}' is absent from HMP"))?;
            sample_ids.push(id);
            selected_columns.push(column);
        }
        if sample_ids.is_empty() {
            return Err("HMP contains no selected samples".to_string());
        }
        Ok(Self {
            iter,
            sample_ids,
            selected_columns,
            next_index: 0,
            previous_chrom: None,
            previous_pos: 0,
            seen_chromosomes: HashSet::new(),
        })
    }

    fn validate_order(&mut self, site: &SiteInfo) -> Result<(), String> {
        let chrom = site.chrom.trim();
        let pos = site.pos as i64;
        if chrom.is_empty() || pos <= 0 {
            return Err(format!(
                "invalid HMP site coordinate '{}:{}'",
                site.chrom, site.pos
            ));
        }
        match self.previous_chrom.as_deref() {
            None => {
                self.seen_chromosomes.insert(chrom.to_string());
            }
            Some(previous) if previous == chrom => {
                if pos < self.previous_pos {
                    return Err(format!(
                        "HMP positions must be nondecreasing within chromosome {chrom}: {pos} after {}",
                        self.previous_pos
                    ));
                }
            }
            Some(_) => {
                if !self.seen_chromosomes.insert(chrom.to_string()) {
                    return Err(format!(
                        "HMP chromosome '{chrom}' occurs in more than one block"
                    ));
                }
            }
        }
        self.previous_chrom = Some(chrom.to_string());
        self.previous_pos = pos;
        Ok(())
    }
}

impl SiteSource for HmpSource {
    fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    fn next_site(&mut self) -> Result<Option<Site>, String> {
        loop {
            let Some((row, info)) = self.iter.next_snp_raw() else {
                return Ok(None);
            };
            self.validate_order(&info)?;
            let allele0 = info.ref_allele.to_ascii_uppercase();
            let allele1 = info.alt_allele.to_ascii_uppercase();
            if !is_simple_snp(&allele0, &allele1) {
                continue;
            }
            let index = self.next_index;
            self.next_index += 1;
            let mut calls = Vec::with_capacity(self.selected_columns.len());
            for &column in &self.selected_columns {
                let dosage = row.get(column).copied().and_then(parse_dosage);
                calls.push(GenotypeCall {
                    dosage,
                    alt_on_hap0: None,
                    phase_set: PhaseSet::Unknown,
                });
            }
            return Ok(Some(Site {
                index,
                chrom: info.chrom,
                pos: info.pos as i64,
                id: info.snp,
                allele0,
                allele1,
                calls,
            }));
        }
    }
}

fn parse_dosage(value: f32) -> Option<u8> {
    if !value.is_finite() {
        return None;
    }
    match value as u8 {
        dosage @ 0..=2 if value == dosage as f32 => Some(dosage),
        _ => None,
    }
}

fn is_simple_snp(allele0: &str, allele1: &str) -> bool {
    let is_base = |allele: &str| matches!(allele, "A" | "C" | "G" | "T");
    is_base(allele0) && is_base(allele1) && allele0 != allele1
}
