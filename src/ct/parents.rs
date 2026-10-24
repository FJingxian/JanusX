use crate::ct::input::{GenotypeCall, PhaseSet, ScannedSite, Site, SiteSource};
use crate::ct::packed::{PackedGenotypes, PackedGenotypesBuilder};
use crate::gfcore::{read_bim, read_fam, SiteInfo};
use crate::gload::{read_plink_bed_header_info, BedMmapMatrix, GenotypeMatrix};
use csv::ReaderBuilder;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default)]
pub(crate) struct ParentBedStats {
    pub(crate) father_markers: usize,
    pub(crate) mother_markers: usize,
    pub(crate) matched_markers: usize,
    pub(crate) unmatched_father_markers: usize,
    pub(crate) unmatched_mother_markers: usize,
    pub(crate) allele_conflicts: usize,
    pub(crate) non_simple_markers: usize,
    pub(crate) child_calls_from_heterozygous_parent: usize,
    pub(crate) child_calls_from_missing_parent: usize,
}

#[derive(Clone, Debug)]
struct ParentMarker {
    source_index: usize,
    info: SiteInfo,
    allele0: String,
    allele1: String,
}

#[derive(Clone, Debug)]
struct CommonMarker {
    father: ParentMarker,
    mother: ParentMarker,
    mother_alleles_to_canonical: [u8; 2],
}

#[derive(Clone, Debug)]
struct ChildParents {
    father_index: usize,
    mother_index: usize,
}

#[derive(Clone, Copy)]
struct ParentBedCodeIndex {
    byte_index: usize,
    shift: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ParentLink {
    pub(crate) child_id: String,
    pub(crate) father_id: String,
    pub(crate) mother_id: String,
}

pub(crate) struct ParentBedSource {
    father_bed: BedMmapMatrix,
    mother_bed: BedMmapMatrix,
    sample_ids: Vec<String>,
    father_parent_indices: Vec<ParentBedCodeIndex>,
    mother_parent_indices: Vec<ParentBedCodeIndex>,
    father_codes: Vec<u8>,
    mother_codes: Vec<u8>,
    children: Vec<ChildParents>,
    markers: Vec<CommonMarker>,
    next_marker: usize,
    stats: ParentBedStats,
}

#[derive(Clone, Debug)]
struct ParentBfileEntry {
    father_prefix: String,
    mother_prefix: String,
}

/// Sequentially reads chromosome-specific parental BFILEs as one logical
/// site stream. Only the current chromosome is memory mapped and decoded.
pub(crate) struct ParentBedListSource {
    entries: Vec<ParentBfileEntry>,
    cross_path: String,
    sample_ids: Vec<String>,
    current: Option<ParentBedSource>,
    next_entry: usize,
    site_index_offset: usize,
    completed_stats: ParentBedStats,
}

pub(crate) struct ParentDerivedSource<S> {
    source: S,
    sample_ids: Vec<String>,
    children: Vec<ChildParents>,
}

impl<S: SiteSource> ParentDerivedSource<S> {
    pub(crate) fn new(source: S, links: Vec<ParentLink>) -> Result<Self, String> {
        let source_index = id_to_index(source.sample_ids(), "parent genotype source")?;
        let mut sample_ids = Vec::with_capacity(links.len());
        let mut children = Vec::with_capacity(links.len());
        for link in links {
            let father_index = source_index.get(&link.father_id).copied().ok_or_else(|| {
                format!(
                    "parent ID '{}' is absent from genotype input",
                    link.father_id
                )
            })?;
            let mother_index = source_index.get(&link.mother_id).copied().ok_or_else(|| {
                format!(
                    "parent ID '{}' is absent from genotype input",
                    link.mother_id
                )
            })?;
            sample_ids.push(link.child_id.clone());
            children.push(ChildParents {
                father_index,
                mother_index,
            });
        }
        Ok(Self {
            source,
            sample_ids,
            children,
        })
    }

    fn next_site_impl(
        &mut self,
        materialize_calls: bool,
    ) -> Result<Option<(Site, Option<PackedGenotypes>)>, String> {
        let Some(mut site) = self.source.next_site()? else {
            return Ok(None);
        };
        let parent_calls = std::mem::take(&mut site.calls);
        let mut calls = materialize_calls.then(|| Vec::with_capacity(self.children.len()));
        let mut packed_builder =
            (!materialize_calls).then(|| PackedGenotypesBuilder::new(self.children.len()));
        for child in &self.children {
            let father_alt = homozygous_alt(parent_calls[child.father_index].dosage);
            let mother_alt = homozygous_alt(parent_calls[child.mother_index].dosage);
            let (dosage, alt_on_hap0, phase_set) = match (father_alt, mother_alt) {
                (Some(father_alt), Some(mother_alt)) => {
                    let dosage = father_alt + mother_alt;
                    (
                        Some(dosage),
                        (dosage == 1).then_some(father_alt == 1),
                        if dosage == 1 {
                            PhaseSet::Implicit
                        } else {
                            PhaseSet::Unknown
                        },
                    )
                }
                _ => (None, None, PhaseSet::Unknown),
            };
            if let Some(calls) = calls.as_mut() {
                calls.push(GenotypeCall {
                    dosage,
                    alt_on_hap0,
                    phase_set: phase_set.clone(),
                });
            }
            if let Some(builder) = packed_builder.as_mut() {
                builder.push(dosage, alt_on_hap0, &phase_set);
            }
        }
        site.calls = calls.unwrap_or_default();
        let packed = packed_builder.map(PackedGenotypesBuilder::finish);
        Ok(Some((site, packed)))
    }

    fn next_scanned_site_impl(&mut self) -> Result<Option<ScannedSite>, String> {
        self.next_site_impl(false).map(|site| {
            site.map(|(site, packed_calls)| ScannedSite {
                site,
                packed_calls: Some(packed_calls.expect("packed parent decode builds bitsets")),
            })
        })
    }
}

impl<S: SiteSource> SiteSource for ParentDerivedSource<S> {
    fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    fn next_site(&mut self) -> Result<Option<Site>, String> {
        self.next_site_impl(true)
            .map(|site| site.map(|(site, _)| site))
    }

    fn next_scanned_site(&mut self) -> Result<Option<ScannedSite>, String> {
        self.next_scanned_site_impl()
    }
}

pub(crate) fn parent_sample_ids(links: &[ParentLink]) -> Vec<String> {
    let mut seen = HashSet::with_capacity(links.len() * 2);
    let mut ids = Vec::with_capacity(links.len() * 2);
    for link in links {
        for id in [&link.father_id, &link.mother_id] {
            if seen.insert(id.as_str()) {
                ids.push(id.clone());
            }
        }
    }
    ids
}

fn homozygous_alt(dosage: Option<u8>) -> Option<u8> {
    match dosage {
        Some(0) => Some(0),
        Some(2) => Some(1),
        _ => None,
    }
}

impl ParentBedSource {
    pub(crate) fn open(
        father_prefix: &str,
        mother_prefix: &str,
        cross_path: &str,
        selected_child_ids: &[String],
    ) -> Result<Self, String> {
        let father_bed = BedMmapMatrix::open(father_prefix)?;
        let mother_bed = BedMmapMatrix::open(mother_prefix)?;
        let father_prefix = father_bed.bed_prefix().to_string();
        let mother_prefix = mother_bed.bed_prefix().to_string();
        let father_ids = unique_fam_iids(&father_prefix)?;
        let mother_ids = unique_fam_iids(&mother_prefix)?;
        if father_ids.len() != father_bed.n_samples_full()
            || mother_ids.len() != mother_bed.n_samples_full()
        {
            return Err("FAM IDs do not match BED sample dimensions".to_string());
        }
        validate_bed_bim_dimensions(&father_prefix, father_ids.len(), father_bed.n_snps_total())?;
        validate_bed_bim_dimensions(&mother_prefix, mother_ids.len(), mother_bed.n_snps_total())?;

        let father_sites = read_bim(&father_prefix)?;
        let mother_sites = read_bim(&mother_prefix)?;
        let father_markers = index_bim_sites(father_sites, &father_prefix)?;
        let mother_markers = index_bim_sites(mother_sites, &mother_prefix)?;
        validate_father_marker_order(&father_markers)?;
        let father_ids_by_index = id_to_index(&father_ids, "father")?;
        let mother_ids_by_index = id_to_index(&mother_ids, "mother")?;
        let links = read_parent_map(cross_path, selected_child_ids)?;
        let mut children = Vec::with_capacity(links.len());
        let mut sample_ids = Vec::with_capacity(links.len());
        for link in links {
            let father_index = father_ids_by_index
                .get(&link.father_id)
                .copied()
                .ok_or_else(|| {
                    format!("father ID '{}' is absent from father FAM", link.father_id)
                })?;
            let mother_index = mother_ids_by_index
                .get(&link.mother_id)
                .copied()
                .ok_or_else(|| {
                    format!("mother ID '{}' is absent from mother FAM", link.mother_id)
                })?;
            sample_ids.push(link.child_id.clone());
            children.push(ChildParents {
                father_index,
                mother_index,
            });
        }
        let (father_indices, mother_indices) = compact_parent_indices(&mut children);
        let father_parent_indices = parent_bed_code_indices(father_indices);
        let mother_parent_indices = parent_bed_code_indices(mother_indices);
        let father_codes = vec![0; father_parent_indices.len()];
        let mother_codes = vec![0; mother_parent_indices.len()];

        let mut stats = ParentBedStats {
            father_markers: father_markers.len(),
            mother_markers: mother_markers.len(),
            ..ParentBedStats::default()
        };
        let father_coordinates = father_markers
            .iter()
            .map(|marker| (marker.info.chrom.clone(), marker.info.pos))
            .collect::<HashSet<_>>();
        let mother_by_coordinate = mother_markers
            .iter()
            .enumerate()
            .map(|(index, marker)| ((marker.info.chrom.clone(), marker.info.pos), index))
            .collect::<HashMap<_, _>>();
        let mut markers = Vec::new();
        for father in father_markers {
            let Some(mother_index) =
                mother_by_coordinate.get(&(father.info.chrom.clone(), father.info.pos))
            else {
                stats.unmatched_father_markers += 1;
                continue;
            };
            let mother = &mother_markers[*mother_index];
            if !is_simple_biallelic_snp(&father.allele0, &father.allele1)
                || !is_simple_biallelic_snp(&mother.allele0, &mother.allele1)
            {
                stats.non_simple_markers += 1;
                continue;
            }
            if !same_allele_set(&father, mother) {
                stats.allele_conflicts += 1;
                continue;
            }
            let mother_alleles_to_canonical = [
                canonical_allele_index(&mother.allele0, &father.allele0, &father.allele1)
                    .ok_or_else(|| {
                        "mother BIM allele disappeared during harmonization".to_string()
                    })?,
                canonical_allele_index(&mother.allele1, &father.allele0, &father.allele1)
                    .ok_or_else(|| {
                        "mother BIM allele disappeared during harmonization".to_string()
                    })?,
            ];
            markers.push(CommonMarker {
                father,
                mother: mother.clone(),
                mother_alleles_to_canonical,
            });
            stats.matched_markers += 1;
        }
        stats.unmatched_mother_markers = mother_markers
            .iter()
            .filter(|marker| {
                !father_coordinates.contains(&(marker.info.chrom.clone(), marker.info.pos))
            })
            .count();
        Ok(Self {
            father_bed,
            mother_bed,
            sample_ids,
            father_parent_indices,
            mother_parent_indices,
            father_codes,
            mother_codes,
            children,
            markers,
            next_marker: 0,
            stats,
        })
    }

    pub(crate) fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    pub(crate) fn stats(&self) -> &ParentBedStats {
        &self.stats
    }

    pub(crate) fn next_site(&mut self) -> Result<Option<Site>, String> {
        self.next_site_impl(true)
            .map(|site| site.map(|(site, _)| site))
    }

    fn next_scanned_site_impl(&mut self) -> Result<Option<ScannedSite>, String> {
        self.next_site_impl(false).map(|scanned| {
            scanned.map(|(site, packed_calls)| ScannedSite {
                site,
                packed_calls: Some(packed_calls.expect("packed-only parent decode builds bitsets")),
            })
        })
    }

    fn next_site_impl(
        &mut self,
        materialize_calls: bool,
    ) -> Result<Option<(Site, Option<crate::ct::packed::PackedGenotypes>)>, String> {
        let Some(marker) = self.markers.get(self.next_marker) else {
            return Ok(None);
        };
        let source_index = marker.father.source_index;
        let father_row = self.father_bed.source_row_bytes(source_index);
        let mother_row = self.mother_bed.source_row_bytes(marker.mother.source_index);
        decode_parent_codes_into(
            father_row,
            &self.father_parent_indices,
            &mut self.father_codes,
        );
        decode_parent_codes_into(
            mother_row,
            &self.mother_parent_indices,
            &mut self.mother_codes,
        );
        let mut calls = Vec::new();
        let packed_calls;
        if materialize_calls {
            calls.reserve(self.children.len());
            for child in &self.children {
                let father_code = self.father_codes[child.father_index];
                let mother_code = self.mother_codes[child.mother_index];
                if father_code <= 1 && mother_code <= 1 {
                    let father_alt = father_code;
                    let mother_alt = marker.mother_alleles_to_canonical[mother_code as usize];
                    let dosage = father_alt + mother_alt;
                    calls.push(GenotypeCall {
                        dosage: Some(dosage),
                        alt_on_hap0: (dosage == 1).then_some(father_alt == 1),
                        phase_set: if dosage == 1 {
                            PhaseSet::Implicit
                        } else {
                            PhaseSet::Unknown
                        },
                    });
                } else {
                    record_parent_missing(&mut self.stats, father_code, mother_code);
                    calls.push(GenotypeCall {
                        dosage: None,
                        alt_on_hap0: None,
                        phase_set: PhaseSet::Unknown,
                    });
                }
            }
            packed_calls = None;
        } else {
            let mut builder = PackedGenotypesBuilder::new(self.children.len());
            for child in &self.children {
                let father_code = self.father_codes[child.father_index];
                let mother_code = self.mother_codes[child.mother_index];
                if father_code <= 1 && mother_code <= 1 {
                    let father_alt = father_code;
                    let mother_alt = marker.mother_alleles_to_canonical[mother_code as usize];
                    builder.push_parent_call(Some(father_alt + mother_alt), father_alt == 1);
                } else {
                    record_parent_missing(&mut self.stats, father_code, mother_code);
                    builder.push_parent_call(None, false);
                }
            }
            packed_calls = Some(builder.finish());
        }

        let site = Site {
            index: source_index,
            chrom: marker.father.info.chrom.clone(),
            pos: marker.father.info.pos as i64,
            id: marker.father.info.snp.clone(),
            allele0: marker.father.allele0.clone(),
            allele1: marker.father.allele1.clone(),
            calls,
        };
        self.next_marker += 1;
        Ok(Some((site, packed_calls)))
    }
}

impl ParentBedListSource {
    pub(crate) fn open(
        manifest_path: &str,
        cross_path: &str,
        selected_child_ids: &[String],
    ) -> Result<Self, String> {
        let entries = read_parent_bfile_manifest(manifest_path)?;
        let first = entries
            .first()
            .ok_or_else(|| "parent BFILE manifest contains no chromosome entries".to_string())?;
        let current = ParentBedSource::open(
            &first.father_prefix,
            &first.mother_prefix,
            cross_path,
            selected_child_ids,
        )?;
        let sample_ids = current.sample_ids().to_vec();
        Ok(Self {
            entries,
            cross_path: cross_path.to_string(),
            sample_ids,
            current: Some(current),
            next_entry: 1,
            site_index_offset: 0,
            completed_stats: ParentBedStats::default(),
        })
    }

    pub(crate) fn stats(&self) -> ParentBedStats {
        let mut stats = self.completed_stats.clone();
        if let Some(current) = &self.current {
            add_parent_bed_stats(&mut stats, current.stats());
        }
        stats
    }

    pub(crate) fn source_prefixes(&self) -> Vec<(String, String)> {
        self.entries
            .iter()
            .map(|entry| (entry.father_prefix.clone(), entry.mother_prefix.clone()))
            .collect()
    }

    fn open_next_entry(&mut self) -> Result<bool, String> {
        let Some(entry) = self.entries.get(self.next_entry) else {
            return Ok(false);
        };
        let source = ParentBedSource::open(
            &entry.father_prefix,
            &entry.mother_prefix,
            &self.cross_path,
            &self.sample_ids,
        )?;
        self.current = Some(source);
        self.next_entry += 1;
        Ok(true)
    }

    fn finish_current_entry(&mut self) -> Result<(), String> {
        let Some(current) = self.current.take() else {
            return Ok(());
        };
        self.site_index_offset = self
            .site_index_offset
            .checked_add(current.stats().father_markers)
            .ok_or_else(|| "parent BFILE global marker index overflow".to_string())?;
        add_parent_bed_stats(&mut self.completed_stats, current.stats());
        Ok(())
    }

    fn next_site_impl(&mut self) -> Result<Option<Site>, String> {
        loop {
            if self.current.is_none() && !self.open_next_entry()? {
                return Ok(None);
            }
            let site = self
                .current
                .as_mut()
                .expect("parent BFILE entry is open")
                .next_site()?;
            if let Some(mut site) = site {
                site.index = site
                    .index
                    .checked_add(self.site_index_offset)
                    .ok_or_else(|| "parent BFILE global marker index overflow".to_string())?;
                return Ok(Some(site));
            }
            self.finish_current_entry()?;
        }
    }

    fn next_scanned_site_impl(&mut self) -> Result<Option<ScannedSite>, String> {
        loop {
            if self.current.is_none() && !self.open_next_entry()? {
                return Ok(None);
            }
            let scanned = self
                .current
                .as_mut()
                .expect("parent BFILE entry is open")
                .next_scanned_site()?;
            if let Some(mut scanned) = scanned {
                scanned.site.index = scanned
                    .site
                    .index
                    .checked_add(self.site_index_offset)
                    .ok_or_else(|| "parent BFILE global marker index overflow".to_string())?;
                return Ok(Some(scanned));
            }
            self.finish_current_entry()?;
        }
    }
}

impl SiteSource for ParentBedListSource {
    fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    fn prefetch_site_bytes_hint(&self) -> usize {
        PackedGenotypes::estimated_storage_bytes_for_samples(self.sample_ids.len())
            .saturating_add(512)
    }

    fn next_site(&mut self) -> Result<Option<Site>, String> {
        self.next_site_impl()
    }

    fn next_scanned_site(&mut self) -> Result<Option<ScannedSite>, String> {
        self.next_scanned_site_impl()
    }
}

fn add_parent_bed_stats(target: &mut ParentBedStats, source: &ParentBedStats) {
    target.father_markers += source.father_markers;
    target.mother_markers += source.mother_markers;
    target.matched_markers += source.matched_markers;
    target.unmatched_father_markers += source.unmatched_father_markers;
    target.unmatched_mother_markers += source.unmatched_mother_markers;
    target.allele_conflicts += source.allele_conflicts;
    target.non_simple_markers += source.non_simple_markers;
    target.child_calls_from_heterozygous_parent += source.child_calls_from_heterozygous_parent;
    target.child_calls_from_missing_parent += source.child_calls_from_missing_parent;
}

fn read_parent_bfile_manifest(path: &str) -> Result<Vec<ParentBfileEntry>, String> {
    let manifest = Path::new(path);
    let base_dir = manifest
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .flexible(true)
        .comment(Some(b'#'))
        .from_path(manifest)
        .map_err(|error| format!("open parent BFILE manifest {}: {error}", manifest.display()))?;
    let mut entries = Vec::new();
    for (row_index, record) in reader.records().enumerate() {
        let record = record.map_err(|error| {
            format!("read parent BFILE manifest {}: {error}", manifest.display())
        })?;
        let fields = record.iter().map(str::trim).collect::<Vec<_>>();
        if fields.is_empty() || fields.iter().all(|field| field.is_empty()) {
            continue;
        }
        if row_index == 0
            && fields.len() >= 2
            && matches!(
                fields[0].to_ascii_lowercase().as_str(),
                "father" | "father_prefix"
            )
            && matches!(
                fields[1].to_ascii_lowercase().as_str(),
                "mother" | "mother_prefix"
            )
        {
            continue;
        }
        if !(1..=2).contains(&fields.len()) || fields[0].is_empty() {
            return Err(format!(
                "parent BFILE manifest row {} must contain one or two nonempty BFILE prefixes",
                row_index + 1
            ));
        }
        let resolve = |value: &str| {
            let value = PathBuf::from(value);
            if value.is_absolute() {
                value
            } else {
                base_dir.map_or(value.clone(), |base| base.join(value))
            }
        };
        let father = resolve(fields[0]).to_string_lossy().into_owned();
        let mother = if fields.get(1).is_some_and(|value| !value.is_empty()) {
            resolve(fields[1]).to_string_lossy().into_owned()
        } else if fields.len() == 1 {
            father.clone()
        } else {
            return Err(format!(
                "parent BFILE manifest row {} has an empty mother prefix",
                row_index + 1
            ));
        };
        entries.push(ParentBfileEntry {
            father_prefix: father,
            mother_prefix: mother,
        });
    }
    if entries.is_empty() {
        return Err("parent BFILE manifest contains no chromosome entries".to_string());
    }
    Ok(entries)
}

impl SiteSource for ParentBedSource {
    fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    fn prefetch_site_bytes_hint(&self) -> usize {
        PackedGenotypes::estimated_storage_bytes_for_samples(self.sample_ids.len())
            .saturating_add(512)
    }

    fn next_site(&mut self) -> Result<Option<Site>, String> {
        ParentBedSource::next_site(self)
    }

    fn next_scanned_site(&mut self) -> Result<Option<ScannedSite>, String> {
        ParentBedSource::next_scanned_site_impl(self)
    }
}

fn parent_bed_code_indices(sample_indices: Vec<usize>) -> Vec<ParentBedCodeIndex> {
    sample_indices
        .into_iter()
        .map(|sample_index| ParentBedCodeIndex {
            byte_index: sample_index / 4,
            shift: (2 * (sample_index % 4)) as u32,
        })
        .collect()
}

#[inline]
fn decode_parent_codes_into(row: &[u8], indices: &[ParentBedCodeIndex], decoded: &mut [u8]) {
    const BED_CODE_TO_DOSAGE: [u8; 4] = [0, 3, 2, 1];
    debug_assert_eq!(indices.len(), decoded.len());
    for (decoded_code, index) in decoded.iter_mut().zip(indices) {
        let bed_code = (row[index.byte_index] >> index.shift) & 0b11;
        *decoded_code = BED_CODE_TO_DOSAGE[bed_code as usize];
    }
}

#[inline]
fn record_parent_missing(stats: &mut ParentBedStats, father_code: u8, mother_code: u8) {
    if father_code == 2 || mother_code == 2 {
        stats.child_calls_from_heterozygous_parent += 1;
    } else {
        stats.child_calls_from_missing_parent += 1;
    }
}

fn unique_fam_iids(prefix: &str) -> Result<Vec<String>, String> {
    let ids = read_fam(prefix)?;
    let mut seen = HashSet::with_capacity(ids.len());
    for id in &ids {
        if !seen.insert(id.clone()) {
            return Err(format!("duplicate IID '{id}' in FAM {prefix}.fam"));
        }
    }
    Ok(ids)
}

fn id_to_index(ids: &[String], parent_role: &str) -> Result<HashMap<String, usize>, String> {
    let mut out = HashMap::with_capacity(ids.len());
    for (index, id) in ids.iter().enumerate() {
        if out.insert(id.clone(), index).is_some() {
            return Err(format!("duplicate {parent_role} IID '{id}'"));
        }
    }
    Ok(out)
}

fn compact_parent_indices(children: &mut [ChildParents]) -> (Vec<usize>, Vec<usize>) {
    fn compact(
        index: usize,
        lookup: &mut HashMap<usize, usize>,
        indices: &mut Vec<usize>,
    ) -> usize {
        if let Some(&dense) = lookup.get(&index) {
            return dense;
        }
        let dense = indices.len();
        indices.push(index);
        lookup.insert(index, dense);
        dense
    }

    let mut father_lookup = HashMap::with_capacity(children.len());
    let mut mother_lookup = HashMap::with_capacity(children.len());
    let mut father_indices = Vec::new();
    let mut mother_indices = Vec::new();
    for child in children {
        child.father_index = compact(child.father_index, &mut father_lookup, &mut father_indices);
        child.mother_index = compact(child.mother_index, &mut mother_lookup, &mut mother_indices);
    }
    (father_indices, mother_indices)
}

fn validate_bed_bim_dimensions(
    prefix: &str,
    n_samples: usize,
    n_snps: usize,
) -> Result<(), String> {
    let bed_path = PathBuf::from(format!("{prefix}.bed"));
    let shape = read_plink_bed_header_info(&bed_path, n_samples)?;
    if shape.n_snps_total != n_snps {
        return Err(format!(
            "BED/BIM marker count mismatch for {prefix}: BED has {}, BIM has {n_snps}",
            shape.n_snps_total
        ));
    }
    Ok(())
}

fn index_bim_sites(sites: Vec<SiteInfo>, prefix: &str) -> Result<Vec<ParentMarker>, String> {
    let mut coordinates = HashSet::with_capacity(sites.len());
    let mut ids = HashSet::with_capacity(sites.len());
    let mut markers = Vec::with_capacity(sites.len());
    for (source_index, info) in sites.into_iter().enumerate() {
        if info.pos <= 0 {
            return Err(format!(
                "nonpositive BIM position in {prefix}.bim row {}",
                source_index + 1
            ));
        }
        let coordinate = (info.chrom.clone(), info.pos);
        if !coordinates.insert(coordinate) {
            return Err(format!(
                "duplicate BIM coordinate {}:{} in {prefix}.bim",
                info.chrom, info.pos
            ));
        }
        let id = if info.snp.is_empty() || info.snp == "." {
            format!("{}_{}", info.chrom, info.pos)
        } else {
            info.snp.clone()
        };
        if !ids.insert(id.clone()) {
            return Err(format!("duplicate BIM variant ID '{id}' in {prefix}.bim"));
        }
        let allele0 = info.ref_allele.to_ascii_uppercase();
        let allele1 = info.alt_allele.to_ascii_uppercase();
        markers.push(ParentMarker {
            source_index,
            info: SiteInfo { snp: id, ..info },
            allele0,
            allele1,
        });
    }
    Ok(markers)
}

fn same_allele_set(father: &ParentMarker, mother: &ParentMarker) -> bool {
    (father.allele0 == mother.allele0 && father.allele1 == mother.allele1)
        || (father.allele0 == mother.allele1 && father.allele1 == mother.allele0)
}

fn is_simple_biallelic_snp(a0: &str, a1: &str) -> bool {
    fn is_base(allele: &str) -> bool {
        matches!(allele, "A" | "C" | "G" | "T")
    }
    is_base(a0) && is_base(a1) && a0 != a1
}

fn canonical_allele_index(allele: &str, allele0: &str, allele1: &str) -> Option<u8> {
    if allele == allele0 {
        Some(0)
    } else if allele == allele1 {
        Some(1)
    } else {
        None
    }
}

fn validate_father_marker_order(markers: &[ParentMarker]) -> Result<(), String> {
    let mut previous_chrom: Option<&str> = None;
    let mut previous_pos = 0i32;
    let mut seen = HashSet::new();
    for marker in markers {
        let chrom = marker.info.chrom.as_str();
        if previous_chrom.is_some_and(|previous| previous != chrom) {
            if !seen.insert(chrom.to_string()) {
                return Err(format!(
                    "father BIM chromosome '{chrom}' occurs in more than one block"
                ));
            }
        } else if previous_chrom.is_none() {
            seen.insert(chrom.to_string());
        } else if marker.info.pos < previous_pos {
            return Err(format!(
                "father BIM positions must be nondecreasing within chromosome {chrom}"
            ));
        }
        previous_chrom = Some(chrom);
        previous_pos = marker.info.pos;
    }
    Ok(())
}

pub(crate) fn read_parent_map(
    path: &str,
    selected_child_ids: &[String],
) -> Result<Vec<ParentLink>, String> {
    let file = File::open(path).map_err(|e| format!("open cross table {path}: {e}"))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    let mut child_lookup = HashMap::new();
    let mut header_pending = true;
    for (line_index, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("read {path}:{}: {e}", line_index + 1))?;
        let line = line.trim().trim_start_matches('\u{feff}');
        if line.is_empty() {
            continue;
        }
        let (mapping_line, comment) = if let Some(comment) = line.strip_prefix('#') {
            (comment, true)
        } else {
            (line, false)
        };
        let fields = parent_map_fields(mapping_line)
            .map_err(|error| format!("malformed parent map row {}: {error}", line_index + 1))?;
        if comment {
            if is_parent_map_header(&fields) {
                header_pending = false;
            }
            continue;
        }
        if header_pending && is_parent_map_header(&fields) {
            header_pending = false;
            continue;
        }
        header_pending = false;
        if fields.len() != 3 || fields.iter().any(|value| value.is_empty()) {
            return Err(format!(
                "malformed parent map row {}: {line}",
                line_index + 1
            ));
        }
        let child_id = fields[0].to_string();
        if child_lookup.insert(child_id.clone(), rows.len()).is_some() {
            return Err(format!("duplicate child ID '{child_id}' in parent map"));
        }
        rows.push(ParentLink {
            child_id,
            father_id: fields[1].to_string(),
            mother_id: fields[2].to_string(),
        });
    }
    if rows.is_empty() {
        return Err(format!("parent map {path} contains no child rows"));
    }
    if selected_child_ids.is_empty() {
        return Ok(rows);
    }
    let mut seen = HashSet::with_capacity(selected_child_ids.len());
    let mut selected = Vec::with_capacity(selected_child_ids.len());
    for id in selected_child_ids {
        if !seen.insert(id.as_str()) {
            return Err(format!("duplicate selected child ID '{id}'"));
        }
        let index = child_lookup.get(id.as_str()).copied().ok_or_else(|| {
            format!("selected child ID '{id}' is absent from cross table (parent map)")
        })?;
        selected.push(rows[index].clone());
    }
    Ok(selected)
}

fn parent_map_fields(line: &str) -> Result<Vec<String>, csv::Error> {
    if line.contains(',') {
        let mut reader = ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_reader(line.as_bytes());
        match reader.records().next() {
            Some(record) => Ok(record?
                .iter()
                .map(|field| field.trim().to_string())
                .collect()),
            None => Ok(Vec::new()),
        }
    } else {
        Ok(line.split_whitespace().map(str::to_string).collect())
    }
}

fn is_parent_map_header(fields: &[String]) -> bool {
    if fields.len() != 3 {
        return false;
    }
    let child = fields[0].trim_start_matches('#').to_ascii_lowercase();
    let father = fields[1].to_ascii_lowercase();
    let mother = fields[2].to_ascii_lowercase();
    matches!(
        child.as_str(),
        "child"
            | "child_id"
            | "offspring"
            | "offspring_id"
            | "progeny"
            | "progeny_id"
            | "f1"
            | "f1_id"
            | "f1id"
            | "f1name"
            | "f1_name"
    ) && matches!(father.as_str(), "father" | "father_id" | "sire" | "p1")
        && matches!(mother.as_str(), "mother" | "mother_id" | "dam" | "p2")
}

#[cfg(test)]
mod tests {
    use super::{
        read_parent_map, ParentBedListSource, ParentBedSource, ParentDerivedSource, ParentLink,
    };
    use crate::ct::hmp::HmpSource;
    use crate::ct::input::SiteSource;
    use crate::ct::packed::count_pair;
    use crate::ct::packed::PackedGenotypes;
    use crate::ct::vcf::VcfSource;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

    fn temp_prefix(name: &str) -> PathBuf {
        let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "janusx-ct-parent-{}-{id}-{name}",
            std::process::id()
        ))
    }

    fn write_bfile(
        prefix: &Path,
        fam_ids: &[&str],
        variants: &[(&str, &str, i64, &str, &str, &[u8])],
    ) {
        let mut fam = String::new();
        for id in fam_ids {
            fam.push_str(&format!("FAM\t{id}\t0\t0\t1\t-9\n"));
        }
        fs::write(prefix.with_extension("fam"), fam).unwrap();

        let mut bim = String::new();
        let mut bed = vec![0x6c, 0x1b, 0x01];
        for (chrom, id, pos, a0, a1, calls) in variants {
            assert_eq!(calls.len(), fam_ids.len());
            bim.push_str(&format!("{chrom}\t{id}\t0\t{pos}\t{a0}\t{a1}\n"));
            let mut row = vec![0u8; fam_ids.len().div_ceil(4)];
            for (sample, code) in calls.iter().enumerate() {
                row[sample / 4] |= code << (2 * (sample % 4));
            }
            bed.extend(row);
        }
        fs::write(prefix.with_extension("bim"), bim).unwrap();
        fs::write(prefix.with_extension("bed"), bed).unwrap();
    }

    fn write_cross(contents: &str) -> PathBuf {
        let path = temp_prefix("cross.tsv");
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn parent_map_accepts_common_delimiters_and_f1name_p1_p2_headers() {
        for (index, contents) in [
            "f1name,p1,p2\nc1,father,mother\n",
            "f1name\tp1\tp2\nc1\tfather\tmother\n",
            "f1name p1 p2\nc1 father mother\n",
        ]
        .into_iter()
        .enumerate()
        {
            let path = write_cross(contents);
            let links = read_parent_map(path.to_str().unwrap(), &[]).unwrap();
            assert_eq!(
                links,
                vec![ParentLink {
                    child_id: "c1".to_string(),
                    father_id: "father".to_string(),
                    mother_id: "mother".to_string(),
                }],
                "mapping format {index}"
            );
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn unphased_parent_vcf_derives_child_phase_from_homozygous_parent_origins() {
        let path = temp_prefix("parents.vcf");
        fs::write(
            &path,
            concat!(
                "##fileformat=VCFv4.3\n",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tfather\tmother\n",
                "1\t10\trs1\tA\tG\t.\tPASS\t.\tGT\t0/0\t1/1\n",
                "1\t20\trs2\tC\tT\t.\tPASS\t.\tGT\t1/1\t0/0\n",
                "1\t30\trs3\tG\tA\t.\tPASS\t.\tGT\t0/1\t0/0\n",
            ),
        )
        .unwrap();
        let links = vec![ParentLink {
            child_id: "child".to_string(),
            father_id: "father".to_string(),
            mother_id: "mother".to_string(),
        }];
        let source =
            VcfSource::open(path.to_str().unwrap(), &["father".into(), "mother".into()]).unwrap();
        let mut source = ParentDerivedSource::new(source, links).unwrap();
        assert_eq!(source.sample_ids(), &["child"]);
        let first = source.next_site().unwrap().unwrap();
        let second = source.next_site().unwrap().unwrap();
        let third = source.next_site().unwrap().unwrap();
        assert_eq!(first.calls[0].dosage, Some(1));
        assert_eq!(first.calls[0].alt_on_hap0, Some(false));
        assert_eq!(second.calls[0].dosage, Some(1));
        assert_eq!(second.calls[0].alt_on_hap0, Some(true));
        let pair = count_pair(&first.calls, &second.calls).unwrap();
        assert_eq!(pair.trans_observed(), 1);
        assert_eq!(pair.cis_observed(), 0);
        assert_eq!(third.calls[0].dosage, None);
        assert_eq!(
            third.calls[0].phase_set,
            crate::ct::input::PhaseSet::Unknown
        );
        assert!(source.next_site().unwrap().is_none());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn single_bfile_parent_map_selects_both_parents_from_one_fam() {
        let bfile = temp_prefix("family");
        write_bfile(
            &bfile,
            &["father", "mother"],
            &[
                ("1", "rs1", 10, "A", "G", &[0, 3]),
                ("1", "rs2", 20, "C", "T", &[3, 0]),
            ],
        );
        let cross = write_cross("f1name,p1,p2\nchild,father,mother\n");
        let mut source = ParentBedSource::open(
            bfile.to_str().unwrap(),
            bfile.to_str().unwrap(),
            cross.to_str().unwrap(),
            &["child".into()],
        )
        .unwrap();
        let first = source.next_site().unwrap().unwrap();
        let second = source.next_site().unwrap().unwrap();
        assert_eq!(source.sample_ids(), &["child"]);
        assert_eq!(first.calls[0].alt_on_hap0, Some(false));
        assert_eq!(second.calls[0].alt_on_hap0, Some(true));
        assert_eq!(
            count_pair(&first.calls, &second.calls)
                .unwrap()
                .trans_observed(),
            1
        );
        for extension in ["bed", "bim", "fam"] {
            fs::remove_file(bfile.with_extension(extension)).unwrap();
        }
        fs::remove_file(cross).unwrap();
    }

    #[test]
    fn chromosome_bfile_list_streams_sites_with_one_global_marker_index() {
        let father1 = temp_prefix("chr1_father");
        let mother1 = temp_prefix("chr1_mother");
        let father2 = temp_prefix("chr2_father");
        let mother2 = temp_prefix("chr2_mother");
        write_bfile(&father1, &["F1"], &[("1", "chr1_snp", 10, "A", "G", &[0])]);
        write_bfile(&mother1, &["M1"], &[("1", "chr1_snp", 10, "A", "G", &[3])]);
        write_bfile(&father2, &["F1"], &[("2", "chr2_snp", 20, "C", "T", &[3])]);
        write_bfile(&mother2, &["M1"], &[("2", "chr2_snp", 20, "C", "T", &[0])]);
        let cross = write_cross("child_id\tfather_id\tmother_id\nkid\tF1\tM1\n");
        let manifest = temp_prefix("chromosomes.tsv");
        let manifest_text = format!(
            "father_prefix\tmother_prefix\n{}\t{}\n{}\t{}\n",
            father1.display(),
            mother1.display(),
            father2.display(),
            mother2.display(),
        );
        fs::write(&manifest, manifest_text).unwrap();

        let mut source = ParentBedListSource::open(
            manifest.to_str().unwrap(),
            cross.to_str().unwrap(),
            &["kid".into()],
        )
        .unwrap();
        assert_eq!(source.sample_ids(), &["kid"]);
        let first = source.next_scanned_site().unwrap().unwrap();
        let second = source.next_scanned_site().unwrap().unwrap();
        assert_eq!(
            (first.site.chrom.as_str(), first.site.pos, first.site.index),
            ("1", 10, 0)
        );
        assert_eq!(
            (
                second.site.chrom.as_str(),
                second.site.pos,
                second.site.index
            ),
            ("2", 20, 1)
        );
        assert!(source.next_scanned_site().unwrap().is_none());
        assert_eq!(source.stats().matched_markers, 2);

        for prefix in [&father1, &mother1, &father2, &mother2] {
            for ext in ["bed", "bim", "fam"] {
                fs::remove_file(prefix.with_extension(ext)).unwrap();
            }
        }
        fs::remove_file(cross).unwrap();
        fs::remove_file(manifest).unwrap();
    }

    #[test]
    fn hmp_parent_calls_are_streamed_and_derived_in_parent_order() {
        let path = temp_prefix("parents.hmp");
        fs::write(
            &path,
            concat!(
                "rs#\talleles\tchrom\tpos\tstrand\tassembly#\tcenter\tprotLSID\tassayLSID\tpanelLSID\tQCcode\tfather\tmother\n",
                "rs1\tA/G\t1\t10\t+\tasm\tc\tp\ta\tpanel\tQC\tAA\tGG\n",
                "rs2\tC/T\t1\t20\t+\tasm\tc\tp\ta\tpanel\tQC\tTT\tCC\n",
            ),
        )
        .unwrap();
        let links = vec![ParentLink {
            child_id: "child".to_string(),
            father_id: "father".to_string(),
            mother_id: "mother".to_string(),
        }];
        let source =
            HmpSource::open(path.to_str().unwrap(), &["father".into(), "mother".into()]).unwrap();
        let mut source = ParentDerivedSource::new(source, links).unwrap();
        let first = source.next_site().unwrap().unwrap();
        let second = source.next_site().unwrap().unwrap();
        assert_eq!(first.calls[0].dosage, Some(1));
        assert_eq!(first.calls[0].alt_on_hap0, Some(false));
        assert_eq!(second.calls[0].dosage, Some(1));
        assert_eq!(second.calls[0].alt_on_hap0, Some(true));
        let pair = count_pair(&first.calls, &second.calls).unwrap();
        assert_eq!(pair.trans_observed(), 1);
        assert!(source.next_site().unwrap().is_none());
        fs::remove_file(path).unwrap();
    }

    fn read_two_sites(
        source: &mut ParentBedSource,
    ) -> (crate::ct::input::Site, crate::ct::input::Site) {
        let a = source.next_site().unwrap().unwrap();
        let b = source.next_site().unwrap().unwrap();
        assert!(source.next_site().unwrap().is_none());
        (a, b)
    }

    #[test]
    fn matches_parents_by_iid_and_marker_coordinates_and_normalizes_swapped_maternal_alleles() {
        let father = temp_prefix("fathers");
        let mother = temp_prefix("mothers");
        write_bfile(
            &father,
            &["F2", "F1"],
            &[
                ("1", "rs10", 10, "A", "G", &[3, 3]),
                ("1", "rs20", 20, "C", "T", &[0, 3]),
            ],
        );
        write_bfile(
            &mother,
            &["M2", "M1"],
            &[
                ("1", "rs20_m", 20, "T", "C", &[0, 3]),
                ("1", "rs10_m", 10, "G", "A", &[3, 3]),
            ],
        );
        let cross = write_cross(
            "child_id\tfather_id\tmother_id\nchild_trans\tF2\tM2\nchild_reuse\tF1\tM1\nchild_cis\tF1\tM1\n",
        );
        let selected = vec![
            "child_cis".into(),
            "child_trans".into(),
            "child_reuse".into(),
        ];
        let mut source = ParentBedSource::open(
            father.to_str().unwrap(),
            mother.to_str().unwrap(),
            cross.to_str().unwrap(),
            &selected,
        )
        .unwrap();
        assert_eq!(source.sample_ids(), selected);
        assert_eq!(source.father_parent_indices.len(), 2);
        assert_eq!(source.mother_parent_indices.len(), 2);
        let (site10, site20) = read_two_sites(&mut source);
        assert_eq!(site10.allele0, "A");
        assert_eq!(site10.allele1, "G");
        assert_eq!(site20.allele0, "C");
        assert_eq!(site20.allele1, "T");
        let counts = count_pair(&site10.calls, &site20.calls).unwrap();
        assert_eq!(counts.cis_observed(), 2);
        assert_eq!(counts.trans_observed(), 1);
        assert_eq!(site10.calls[0].alt_on_hap0, Some(true));
        assert_eq!(site20.calls[1].alt_on_hap0, Some(false));

        let mut packed_source = ParentBedSource::open(
            father.to_str().unwrap(),
            mother.to_str().unwrap(),
            cross.to_str().unwrap(),
            &selected,
        )
        .unwrap();
        for scalar_site in [&site10, &site20] {
            let scanned = packed_source.next_scanned_site().unwrap().unwrap();
            assert!(scanned.site.calls.is_empty());
            assert_eq!(scanned.site.index, scalar_site.index);
            assert_eq!(scanned.site.chrom, scalar_site.chrom);
            assert_eq!(scanned.site.pos, scalar_site.pos);
            assert_eq!(
                scanned.packed_calls.unwrap(),
                PackedGenotypes::from_calls(&scalar_site.calls)
            );
        }
        assert!(packed_source.next_scanned_site().unwrap().is_none());

        let mut vcf_text = String::from(
            "##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tchild_cis\tchild_trans\tchild_reuse\n",
        );
        vcf_text.push_str("1\t10\trs10\tA\tG\t.\tPASS\t.\tGT\t1|0\t1|0\t1|0\n");
        vcf_text.push_str("1\t20\trs20\tC\tT\t.\tPASS\t.\tGT\t1|0\t0|1\t1|0\n");
        let vcf_path = temp_prefix("equivalent.vcf");
        fs::write(&vcf_path, vcf_text).unwrap();
        let mut vcf = VcfSource::open(vcf_path.to_str().unwrap(), &selected).unwrap();
        assert_eq!(vcf.next_site().unwrap().unwrap(), site10);
        assert_eq!(vcf.next_site().unwrap().unwrap(), site20);
        fs::remove_file(vcf_path).unwrap();

        for prefix in [&father, &mother] {
            for ext in ["bed", "bim", "fam"] {
                fs::remove_file(prefix.with_extension(ext)).unwrap();
            }
        }
        fs::remove_file(cross).unwrap();
    }

    #[test]
    fn heterozygous_or_missing_parent_calls_make_the_child_site_unknown() {
        let father = temp_prefix("uncertain_fathers");
        let mother = temp_prefix("uncertain_mothers");
        write_bfile(
            &father,
            &["F1"],
            &[
                ("1", "rs1", 1, "A", "G", &[2]),
                ("1", "rs2", 2, "C", "T", &[1]),
            ],
        );
        write_bfile(
            &mother,
            &["M1"],
            &[
                ("1", "rs1m", 1, "A", "G", &[0]),
                ("1", "rs2m", 2, "C", "T", &[3]),
            ],
        );
        let cross = write_cross("child_id father_id mother_id\nkid F1 M1\n");
        let mut source = ParentBedSource::open(
            father.to_str().unwrap(),
            mother.to_str().unwrap(),
            cross.to_str().unwrap(),
            &[],
        )
        .unwrap();
        let first = source.next_site().unwrap().unwrap();
        let second = source.next_site().unwrap().unwrap();
        assert_eq!(first.calls[0].dosage, None);
        assert_eq!(second.calls[0].dosage, None);
        assert!(source.next_site().unwrap().is_none());
        for prefix in [&father, &mother] {
            for ext in ["bed", "bim", "fam"] {
                fs::remove_file(prefix.with_extension(ext)).unwrap();
            }
        }
        fs::remove_file(cross).unwrap();
    }

    #[test]
    fn coordinate_allele_conflicts_are_counted_and_not_matched() {
        let father = temp_prefix("conflict_fathers");
        let mother = temp_prefix("conflict_mothers");
        write_bfile(&father, &["F1"], &[("1", "rs1", 1, "A", "G", &[0])]);
        write_bfile(&mother, &["M1"], &[("1", "rs1m", 1, "A", "T", &[0])]);
        let cross = write_cross("child_id father_id mother_id\nkid F1 M1\n");
        let mut source = ParentBedSource::open(
            father.to_str().unwrap(),
            mother.to_str().unwrap(),
            cross.to_str().unwrap(),
            &[],
        )
        .unwrap();
        assert!(source.next_site().unwrap().is_none());
        assert_eq!(source.stats().allele_conflicts, 1);
        fs::remove_file(father.with_extension("bed")).unwrap();
        fs::remove_file(father.with_extension("bim")).unwrap();
        fs::remove_file(father.with_extension("fam")).unwrap();
        fs::remove_file(mother.with_extension("bed")).unwrap();
        fs::remove_file(mother.with_extension("bim")).unwrap();
        fs::remove_file(mother.with_extension("fam")).unwrap();
        fs::remove_file(cross).unwrap();
    }

    #[test]
    fn duplicate_bim_coordinates_and_duplicate_or_unknown_cross_ids_are_errors() {
        let father = temp_prefix("bad_fathers");
        let mother = temp_prefix("bad_mothers");
        write_bfile(
            &father,
            &["F1"],
            &[
                ("1", "rs1", 1, "A", "G", &[0]),
                ("1", "rs1b", 1, "A", "G", &[0]),
            ],
        );
        write_bfile(&mother, &["M1"], &[("1", "rs1", 1, "A", "G", &[0])]);
        let cross = write_cross("child_id father_id mother_id\nkid F1 M1\n");
        assert!(ParentBedSource::open(
            father.to_str().unwrap(),
            mother.to_str().unwrap(),
            cross.to_str().unwrap(),
            &[],
        )
        .err()
        .unwrap()
        .contains("duplicate BIM coordinate"));
        fs::remove_file(father.with_extension("bed")).unwrap();
        fs::remove_file(father.with_extension("bim")).unwrap();
        fs::remove_file(father.with_extension("fam")).unwrap();

        write_bfile(&father, &["F1"], &[("1", "rs1", 1, "A", "G", &[0])]);
        fs::write(&cross, "child_id father_id mother_id\nkid absent M1\n").unwrap();
        assert!(ParentBedSource::open(
            father.to_str().unwrap(),
            mother.to_str().unwrap(),
            cross.to_str().unwrap(),
            &[],
        )
        .is_err());
        fs::write(&cross, "child_id father_id mother_id\nkid F1 M1\n").unwrap();
        assert!(ParentBedSource::open(
            father.to_str().unwrap(),
            mother.to_str().unwrap(),
            cross.to_str().unwrap(),
            &["absent_child".into()],
        )
        .err()
        .unwrap()
        .contains("absent from cross table"));
        fs::write(
            &cross,
            "child_id father_id mother_id\nkid F1 M1\nkid F1 M1\n",
        )
        .unwrap();
        assert!(ParentBedSource::open(
            father.to_str().unwrap(),
            mother.to_str().unwrap(),
            cross.to_str().unwrap(),
            &[],
        )
        .err()
        .unwrap()
        .contains("duplicate child ID"));
        for prefix in [&father, &mother] {
            for ext in ["bed", "bim", "fam"] {
                fs::remove_file(prefix.with_extension(ext)).unwrap();
            }
        }
        fs::remove_file(cross).unwrap();
    }
}
