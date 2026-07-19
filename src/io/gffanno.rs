use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::gfcore::open_text_maybe_gz;

const GFF_CACHE_VERSION: u32 = 3;
const DESC_GENE_FLANK_BP_DEFAULT: i64 = 2_000;
const DESC_INTERGENIC: &str = "Intergenic;NA;NA";
const BROADEN_EMPTY: &str = "NA";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
enum GffStrand {
    Plus,
    Minus,
    Other,
}

impl GffStrand {
    #[inline]
    fn from_text(text: &str) -> Self {
        match text.trim() {
            "+" => Self::Plus,
            "-" => Self::Minus,
            _ => Self::Other,
        }
    }

    #[inline]
    fn as_text(self) -> &'static str {
        match self {
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Other => ".",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
enum GffFeatureKind {
    Gene,
    Cds,
    FivePrimeUtr,
    ThreePrimeUtr,
    Exon,
    Intron,
    Other,
}

impl GffFeatureKind {
    #[inline]
    fn from_text(text: &str) -> Self {
        match text.trim().to_ascii_lowercase().as_str() {
            "gene" => Self::Gene,
            "cds" => Self::Cds,
            "five_prime_utr" => Self::FivePrimeUtr,
            "three_prime_utr" => Self::ThreePrimeUtr,
            "exon" => Self::Exon,
            "intron" => Self::Intron,
            _ => Self::Other,
        }
    }

    #[inline]
    fn exact_rank(self) -> u8 {
        match self {
            Self::Cds => 0,
            Self::FivePrimeUtr => 1,
            Self::ThreePrimeUtr => 2,
            Self::Exon => 3,
            Self::Intron => 4,
            Self::Gene | Self::Other => 5,
        }
    }

    #[inline]
    fn exact_label_from_rank(rank: u8) -> &'static str {
        match rank {
            0 => "CDS",
            1 => "FivePrimeUTR",
            2 => "ThreePrimeUTR",
            3 => "Exon",
            4 => "Intron",
            _ => "Intron",
        }
    }

    #[inline]
    fn gene_panel_name(self) -> Option<&'static str> {
        match self {
            Self::Gene => Some("gene"),
            Self::Cds => Some("CDS"),
            Self::FivePrimeUtr => Some("five_prime_UTR"),
            Self::ThreePrimeUtr => Some("three_prime_UTR"),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct RawFeatureRecord {
    chrom: String,
    start: i64,
    end: i64,
    strand: GffStrand,
    kind: GffFeatureKind,
    id: Option<String>,
    parents: Vec<String>,
    gene_desc: Option<String>,
    gene_name: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GeneRecord {
    start: i64,
    end: i64,
    strand: GffStrand,
    id: String,
    desc: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FeatureRecord {
    start: i64,
    end: i64,
    kind: GffFeatureKind,
    strand: GffStrand,
    id: String,
    gene_indices: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ChromIndex {
    chrom: String,
    genes: Vec<GeneRecord>,
    features: Vec<FeatureRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct GffSourceMeta {
    path: String,
    size: u64,
    mtime_ns: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializedGffAnnotationIndex {
    version: u32,
    source: GffSourceMeta,
    chroms: Vec<ChromIndex>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GeneHandle {
    chrom_idx: u32,
    gene_idx: u32,
}

#[derive(Clone, Debug)]
struct GffAnnotationIndexInner {
    source: GffSourceMeta,
    chroms: Vec<ChromIndex>,
    chrom_lookup: HashMap<String, usize>,
    n_genes: usize,
    n_features: usize,
}

type GenePanelRangeColumns = (
    Vec<String>,
    Vec<String>,
    Vec<i64>,
    Vec<i64>,
    Vec<String>,
    Vec<String>,
);

impl GffAnnotationIndexInner {
    fn from_serialized(data: SerializedGffAnnotationIndex) -> Result<Self, String> {
        if data.version != GFF_CACHE_VERSION {
            return Err(format!(
                "Unsupported GFF cache version: {} (expected {})",
                data.version, GFF_CACHE_VERSION
            ));
        }
        let mut chrom_lookup: HashMap<String, usize> = HashMap::with_capacity(data.chroms.len());
        let mut n_genes = 0usize;
        let mut n_features = 0usize;
        for (idx, chrom) in data.chroms.iter().enumerate() {
            chrom_lookup.insert(chrom.chrom.clone(), idx);
            n_genes = n_genes.saturating_add(chrom.genes.len());
            n_features = n_features.saturating_add(chrom.features.len());
        }
        Ok(Self {
            source: data.source,
            chroms: data.chroms,
            chrom_lookup,
            n_genes,
            n_features,
        })
    }

    fn to_serialized(&self) -> SerializedGffAnnotationIndex {
        SerializedGffAnnotationIndex {
            version: GFF_CACHE_VERSION,
            source: self.source.clone(),
            chroms: self.chroms.clone(),
        }
    }

    fn annotate_site_desc(&self, chrom: &str, pos: i64, flank_bp: i64) -> String {
        let chrom_norm = normalize_chr(chrom);
        let Some(&chrom_idx) = self.chrom_lookup.get(&chrom_norm) else {
            return DESC_INTERGENIC.to_string();
        };
        annotate_site_desc_on_chrom(&self.chroms[chrom_idx], pos, flank_bp)
    }

    fn annotate_site_broaden(&self, chrom: &str, pos: i64, window_bp: i64) -> String {
        let chrom_norm = normalize_chr(chrom);
        let Some(&chrom_idx) = self.chrom_lookup.get(&chrom_norm) else {
            return BROADEN_EMPTY.to_string();
        };
        annotate_site_broaden_on_chrom(&self.chroms[chrom_idx], pos, window_bp)
    }

    fn annotate_many_desc(&self, chroms: &[String], poss: &[i64], flank_bp: i64) -> Result<Vec<String>, String> {
        if chroms.len() != poss.len() {
            return Err(format!(
                "chroms and poss length mismatch: {} vs {}",
                chroms.len(),
                poss.len()
            ));
        }
        let mut out = vec![DESC_INTERGENIC.to_string(); poss.len()];
        let mut grouped: HashMap<String, Vec<(usize, i64)>> = HashMap::new();
        for (idx, (chrom, pos)) in chroms.iter().zip(poss.iter()).enumerate() {
            grouped
                .entry(normalize_chr(chrom))
                .or_default()
                .push((idx, *pos));
        }
        for (chrom_norm, mut items) in grouped {
            let Some(&chrom_idx) = self.chrom_lookup.get(&chrom_norm) else {
                continue;
            };
            items.sort_by_key(|x| x.1);
            annotate_many_desc_on_chrom(&self.chroms[chrom_idx], &items, flank_bp, &mut out);
        }
        Ok(out)
    }

    fn annotate_many_broaden(
        &self,
        chroms: &[String],
        poss: &[i64],
        window_bp: i64,
    ) -> Result<Vec<String>, String> {
        if chroms.len() != poss.len() {
            return Err(format!(
                "chroms and poss length mismatch: {} vs {}",
                chroms.len(),
                poss.len()
            ));
        }
        let mut out = vec![BROADEN_EMPTY.to_string(); poss.len()];
        let mut grouped: HashMap<String, Vec<(usize, i64)>> = HashMap::new();
        for (idx, (chrom, pos)) in chroms.iter().zip(poss.iter()).enumerate() {
            grouped
                .entry(normalize_chr(chrom))
                .or_default()
                .push((idx, *pos));
        }
        for (chrom_norm, mut items) in grouped {
            let Some(&chrom_idx) = self.chrom_lookup.get(&chrom_norm) else {
                continue;
            };
            items.sort_by_key(|x| x.1);
            annotate_many_broaden_on_chrom(&self.chroms[chrom_idx], &items, window_bp, &mut out);
        }
        Ok(out)
    }

    fn fetch_gene_panel_ranges(
        &self,
        chroms: &[String],
        starts: &[i64],
        ends: &[i64],
    ) -> Result<GenePanelRangeColumns, String> {
        if chroms.len() != starts.len() || chroms.len() != ends.len() {
            return Err(format!(
                "chroms/starts/ends length mismatch: {} / {} / {}",
                chroms.len(),
                starts.len(),
                ends.len()
            ));
        }

        let mut grouped: HashMap<String, Vec<(usize, i64, i64)>> = HashMap::new();
        for idx in 0..chroms.len() {
            let mut start = starts[idx];
            let mut end = ends[idx];
            if start > end {
                std::mem::swap(&mut start, &mut end);
            }
            grouped
                .entry(normalize_chr(&chroms[idx]))
                .or_default()
                .push((idx, start, end));
        }

        let mut buckets: Vec<Vec<(String, String, i64, i64, String, String)>> =
            vec![Vec::new(); chroms.len()];
        for (chrom_norm, items) in grouped.into_iter() {
            let Some(&chrom_idx) = self.chrom_lookup.get(&chrom_norm) else {
                continue;
            };
            fetch_gene_panel_ranges_on_chrom(
                &self.chroms[chrom_idx],
                &items,
                &mut buckets,
            );
        }

        let total_rows: usize = buckets.iter().map(|x| x.len()).sum();
        let mut out_chroms: Vec<String> = Vec::with_capacity(total_rows);
        let mut out_features: Vec<String> = Vec::with_capacity(total_rows);
        let mut out_starts: Vec<i64> = Vec::with_capacity(total_rows);
        let mut out_ends: Vec<i64> = Vec::with_capacity(total_rows);
        let mut out_strands: Vec<String> = Vec::with_capacity(total_rows);
        let mut out_ids: Vec<String> = Vec::with_capacity(total_rows);
        for bucket in buckets.into_iter() {
            for (chrom, feature, start, end, strand, id) in bucket.into_iter() {
                out_chroms.push(chrom);
                out_features.push(feature);
                out_starts.push(start);
                out_ends.push(end);
                out_strands.push(strand);
                out_ids.push(id);
            }
        }
        Ok((
            out_chroms,
            out_features,
            out_starts,
            out_ends,
            out_strands,
            out_ids,
        ))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct GffAnnotationIndex {
    inner: Arc<GffAnnotationIndexInner>,
}

#[pymethods]
impl GffAnnotationIndex {
    #[staticmethod]
    #[pyo3(signature = (gff_path, cache_path=None))]
    fn from_gff(py: Python<'_>, gff_path: String, cache_path: Option<String>) -> PyResult<Self> {
        let gff_path_moved = gff_path.clone();
        let cache_path_moved = cache_path.clone();
        let inner = py
            .detach(move || build_or_load_gff_index(&gff_path_moved, cache_path_moved.as_deref()))
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    #[staticmethod]
    fn load(py: Python<'_>, cache_path: String) -> PyResult<Self> {
        let cache_path_moved = cache_path.clone();
        let inner = py
            .detach(move || load_gff_index(&cache_path_moved))
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn save(&self, cache_path: String) -> PyResult<()> {
        save_gff_index(&self.inner, &cache_path).map_err(PyRuntimeError::new_err)
    }

    #[getter]
    fn source_path(&self) -> String {
        self.inner.source.path.clone()
    }

    #[getter]
    fn n_chrom(&self) -> usize {
        self.inner.chroms.len()
    }

    #[getter]
    fn n_genes(&self) -> usize {
        self.inner.n_genes
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features
    }

    #[pyo3(signature = (chrom, pos, flank_bp=DESC_GENE_FLANK_BP_DEFAULT))]
    fn annotate_site_desc(&self, chrom: String, pos: i64, flank_bp: i64) -> String {
        self.inner.annotate_site_desc(&chrom, pos, flank_bp)
    }

    fn annotate_site_broaden(&self, chrom: String, pos: i64, window_bp: i64) -> String {
        self.inner.annotate_site_broaden(&chrom, pos, window_bp)
    }

    #[pyo3(signature = (chroms, poss, flank_bp=DESC_GENE_FLANK_BP_DEFAULT))]
    fn annotate_many_desc(&self, chroms: Vec<String>, poss: Vec<i64>, flank_bp: i64) -> PyResult<Vec<String>> {
        self.inner
            .annotate_many_desc(&chroms, &poss, flank_bp)
            .map_err(PyRuntimeError::new_err)
    }

    fn annotate_many_broaden(
        &self,
        chroms: Vec<String>,
        poss: Vec<i64>,
        window_bp: i64,
    ) -> PyResult<Vec<String>> {
        self.inner
            .annotate_many_broaden(&chroms, &poss, window_bp)
            .map_err(PyRuntimeError::new_err)
    }

    fn fetch_gene_panel_ranges(
        &self,
        chroms: Vec<String>,
        starts: Vec<i64>,
        ends: Vec<i64>,
    ) -> PyResult<GenePanelRangeColumns> {
        self.inner
            .fetch_gene_panel_ranges(&chroms, &starts, &ends)
            .map_err(PyRuntimeError::new_err)
    }
}

fn build_or_load_gff_index(gff_path: &str, cache_path: Option<&str>) -> Result<GffAnnotationIndexInner, String> {
    let source = gff_source_meta(gff_path)?;
    if let Some(cache) = cache_path {
        let cache_file = Path::new(cache);
        if cache_file.exists() {
            match load_gff_index(cache) {
                Ok(index) if index.source == source => return Ok(index),
                Ok(_) => {}
                Err(_) => {}
            }
        }
    }
    let index = parse_gff_annotation_index(gff_path, source)?;
    if let Some(cache) = cache_path {
        let _ = save_gff_index(&index, cache);
    }
    Ok(index)
}

fn load_gff_index(cache_path: &str) -> Result<GffAnnotationIndexInner, String> {
    let file = File::open(cache_path)
        .map_err(|e| format!("Failed to open GFF cache {}: {e}", cache_path))?;
    let reader = BufReader::new(file);
    let data: SerializedGffAnnotationIndex = bincode::deserialize_from(reader)
        .map_err(|e| format!("Failed to deserialize GFF cache {}: {e}", cache_path))?;
    GffAnnotationIndexInner::from_serialized(data)
}

fn save_gff_index(index: &GffAnnotationIndexInner, cache_path: &str) -> Result<(), String> {
    let path = Path::new(cache_path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                format!("Failed to create GFF cache directory {}: {e}", parent.display())
            })?;
        }
    }
    let file = File::create(path)
        .map_err(|e| format!("Failed to create GFF cache {}: {e}", cache_path))?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, &index.to_serialized())
        .map_err(|e| format!("Failed to serialize GFF cache {}: {e}", cache_path))
}

fn gff_source_meta(gff_path: &str) -> Result<GffSourceMeta, String> {
    let real = fs::canonicalize(gff_path)
        .map_err(|e| format!("Failed to resolve GFF path {}: {e}", gff_path))?;
    let meta = fs::metadata(&real)
        .map_err(|e| format!("Failed to stat GFF file {}: {e}", real.display()))?;
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|x| x.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|x| x.as_nanos() as u64)
        .unwrap_or(0);
    Ok(GffSourceMeta {
        path: real.to_string_lossy().to_string(),
        size: meta.len(),
        mtime_ns,
    })
}

fn parse_gff_annotation_index(
    gff_path: &str,
    source: GffSourceMeta,
) -> Result<GffAnnotationIndexInner, String> {
    let raw_features = parse_gff_rows(gff_path)?;

    let mut id_to_parents: HashMap<String, Vec<String>> = HashMap::new();
    for raw in raw_features.iter() {
        if let Some(id) = raw.id.as_ref() {
            if !raw.parents.is_empty() {
                id_to_parents.insert(id.clone(), raw.parents.clone());
            }
        }
    }

    let mut genes_by_chrom: HashMap<String, Vec<GeneRecord>> = HashMap::new();
    for raw in raw_features.iter() {
        if raw.kind != GffFeatureKind::Gene {
            continue;
        }
        let Some(id) = raw.id.as_ref() else {
            continue;
        };
        let desc = clean_or_na(raw.gene_desc.as_deref().unwrap_or(
            raw.gene_name.as_deref().unwrap_or("NA"),
        ));
        genes_by_chrom
            .entry(raw.chrom.clone())
            .or_default()
            .push(GeneRecord {
                start: raw.start,
                end: raw.end,
                strand: raw.strand,
                id: id.clone(),
                desc,
            });
    }

    let mut chrom_names: Vec<String> = genes_by_chrom.keys().cloned().collect();
    chrom_names.sort_by(chrom_sort_cmp);

    let mut chroms: Vec<ChromIndex> = Vec::with_capacity(chrom_names.len());
    let mut chrom_lookup: HashMap<String, usize> = HashMap::with_capacity(chrom_names.len());
    let mut gene_lookup: HashMap<String, GeneHandle> = HashMap::new();

    for chrom_name in chrom_names.iter() {
        let mut genes = genes_by_chrom.remove(chrom_name).unwrap_or_default();
        genes.sort_by(|a, b| {
            (a.start, a.end, a.id.as_str()).cmp(&(b.start, b.end, b.id.as_str()))
        });
        let chrom_idx = chroms.len();
        for (gene_idx, gene) in genes.iter().enumerate() {
            gene_lookup.insert(
                gene.id.clone(),
                GeneHandle {
                    chrom_idx: chrom_idx as u32,
                    gene_idx: gene_idx as u32,
                },
            );
        }
        chrom_lookup.insert(chrom_name.clone(), chrom_idx);
        chroms.push(ChromIndex {
            chrom: chrom_name.clone(),
            genes,
            features: Vec::new(),
        });
    }

    let mut resolve_cache: HashMap<String, Vec<GeneHandle>> = HashMap::new();
    for raw in raw_features.iter() {
        let Some(&chrom_idx) = chrom_lookup.get(&raw.chrom) else {
            continue;
        };

        let mut handles: Vec<GeneHandle> = Vec::new();
        let mut seen_handles: HashSet<GeneHandle> = HashSet::new();
        if raw.kind == GffFeatureKind::Gene {
            if let Some(id) = raw.id.as_ref() {
                if let Some(handle) = gene_lookup.get(id).copied() {
                    if handle.chrom_idx as usize == chrom_idx && seen_handles.insert(handle) {
                        handles.push(handle);
                    }
                }
            }
        } else {
            let mut candidates: Vec<String> = Vec::new();
            if let Some(id) = raw.id.as_ref() {
                candidates.push(id.clone());
            }
            candidates.extend(raw.parents.iter().cloned());
            for candidate in candidates.iter() {
                let resolved = resolve_gene_handles(
                    candidate,
                    &gene_lookup,
                    &id_to_parents,
                    &mut resolve_cache,
                );
                for handle in resolved.into_iter() {
                    if handle.chrom_idx as usize != chrom_idx || !seen_handles.insert(handle) {
                        continue;
                    }
                    handles.push(handle);
                }
            }
        }
        if handles.is_empty() {
            continue;
        }

        let local_gene_indices: Vec<u32> = handles.iter().map(|x| x.gene_idx).collect();
        chroms[chrom_idx].features.push(FeatureRecord {
            start: raw.start,
            end: raw.end,
            kind: raw.kind,
            strand: raw.strand,
            id: raw
                .id
                .as_ref()
                .map(|x| x.clone())
                .unwrap_or_else(|| "NA".to_string()),
            gene_indices: local_gene_indices,
        });
    }

    let mut n_genes = 0usize;
    let mut n_features = 0usize;
    for chrom in chroms.iter_mut() {
        chrom.features.sort_by(|a, b| (a.start, a.end).cmp(&(b.start, b.end)));
        n_genes = n_genes.saturating_add(chrom.genes.len());
        n_features = n_features.saturating_add(chrom.features.len());
    }

    Ok(GffAnnotationIndexInner {
        source,
        chroms,
        chrom_lookup,
        n_genes,
        n_features,
    })
}

fn parse_gff_rows(gff_path: &str) -> Result<Vec<RawFeatureRecord>, String> {
    let mut reader = open_text_maybe_gz(Path::new(gff_path))?;
    let mut line = String::new();
    let mut rows: Vec<RawFeatureRecord> = Vec::new();
    loop {
        line.clear();
        let n = std::io::BufRead::read_line(&mut reader, &mut line)
            .map_err(|e| format!("Failed to read GFF {}: {e}", gff_path))?;
        if n == 0 {
            break;
        }
        if line.starts_with('#') {
            continue;
        }
        let text = line.trim_end_matches(['\r', '\n']);
        if text.is_empty() {
            continue;
        }

        let mut parts: Vec<&str> = text.split('\t').collect();
        if parts.len() < 9 {
            parts = text.split_whitespace().collect();
            if parts.len() < 9 {
                continue;
            }
        }

        let start = match parts[3].trim().parse::<i64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let end = match parts[4].trim().parse::<i64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let feature_text = parts[2].trim();
        if feature_text.is_empty() {
            continue;
        }
        let kind = GffFeatureKind::from_text(feature_text);
        let merged_attr;
        let attr_text = if parts.len() > 9 {
            merged_attr = parts[8..].join(" ");
            merged_attr.as_str()
        } else {
            parts[8]
        };
        let attrs = parse_attr_subset(attr_text, kind == GffFeatureKind::Gene);
        rows.push(RawFeatureRecord {
            chrom: normalize_chr(parts[0]),
            start,
            end,
            strand: GffStrand::from_text(parts[6]),
            kind,
            id: attrs.id,
            parents: attrs.parents,
            gene_desc: attrs.desc,
            gene_name: attrs.name,
        });
    }
    Ok(rows)
}

#[derive(Default)]
struct AttrSubset {
    id: Option<String>,
    parents: Vec<String>,
    desc: Option<String>,
    name: Option<String>,
}

fn parse_attr_subset(text: &str, include_gene_meta: bool) -> AttrSubset {
    let mut out = AttrSubset::default();
    let mut saw_semicolon = false;
    for token in text.split(';') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        saw_semicolon = true;
        if let Some((key, value)) = token.split_once('=') {
            apply_attr_kv(&mut out, key.trim(), value.trim(), include_gene_meta);
        }
    }
    if !saw_semicolon {
        for token in text.split_whitespace() {
            if let Some((key, value)) = token.split_once('=') {
                apply_attr_kv(&mut out, key.trim(), value.trim(), include_gene_meta);
            }
        }
    }
    out
}

fn apply_attr_kv(out: &mut AttrSubset, key: &str, value: &str, include_gene_meta: bool) {
    match key {
        "ID" if out.id.is_none() => {
            out.id = normalize_entity_id_opt(value);
        }
        "Parent" if out.parents.is_empty() => {
            out.parents = split_entity_ids(value);
        }
        "description" if include_gene_meta && out.desc.is_none() => {
            out.desc = clean_opt(value);
        }
        "Name" if include_gene_meta && out.name.is_none() => {
            out.name = clean_opt(value);
        }
        _ => {}
    }
}

fn resolve_gene_handles(
    candidate: &str,
    gene_lookup: &HashMap<String, GeneHandle>,
    id_to_parents: &HashMap<String, Vec<String>>,
    memo: &mut HashMap<String, Vec<GeneHandle>>,
) -> Vec<GeneHandle> {
    let Some(candidate_norm) = normalize_entity_id_opt(candidate) else {
        return Vec::new();
    };
    if let Some(cached) = memo.get(&candidate_norm) {
        return cached.clone();
    }

    let mut out: Vec<GeneHandle> = Vec::new();
    let mut seen_genes: HashSet<GeneHandle> = HashSet::new();
    let mut seen_nodes: HashSet<String> = HashSet::new();
    let mut stack: Vec<String> = vec![candidate_norm.clone()];

    while let Some(node) = stack.pop() {
        let Some(node_norm) = normalize_entity_id_opt(&node) else {
            continue;
        };
        if let Some(handle) = gene_lookup.get(&node_norm).copied() {
            if seen_genes.insert(handle) {
                out.push(handle);
            }
            continue;
        }
        if !seen_nodes.insert(node_norm.clone()) {
            continue;
        }
        if let Some(parents) = id_to_parents.get(&node_norm) {
            for parent in parents.iter().rev() {
                if let Some(parent_norm) = normalize_entity_id_opt(parent) {
                    if !seen_nodes.contains(&parent_norm) {
                        stack.push(parent_norm);
                    }
                }
            }
        }
    }

    memo.insert(candidate_norm, out.clone());
    out
}

fn annotate_site_desc_on_chrom(chrom: &ChromIndex, pos: i64, flank_bp: i64) -> String {
    let mut out = vec![DESC_INTERGENIC.to_string()];
    annotate_many_desc_on_chrom(chrom, &[(0usize, pos)], flank_bp, &mut out);
    out.pop().unwrap_or_else(|| DESC_INTERGENIC.to_string())
}

fn annotate_many_desc_on_chrom(
    chrom: &ChromIndex,
    items: &[(usize, i64)],
    flank_bp: i64,
    out: &mut [String],
) {
    let flank = flank_bp.max(0);
    let mut active_features: BTreeSet<usize> = BTreeSet::new();
    let mut feature_heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    let mut feature_add_ptr = 0usize;

    let mut active_genes: BTreeSet<usize> = BTreeSet::new();
    let mut gene_heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    let mut gene_add_ptr = 0usize;

    for &(out_idx, pos) in items.iter() {
        while feature_add_ptr < chrom.features.len() && chrom.features[feature_add_ptr].start <= pos {
            active_features.insert(feature_add_ptr);
            feature_heap.push(Reverse((chrom.features[feature_add_ptr].end, feature_add_ptr)));
            feature_add_ptr += 1;
        }
        while let Some(Reverse((end, idx))) = feature_heap.peek().copied() {
            if end < pos {
                feature_heap.pop();
                active_features.remove(&idx);
            } else {
                break;
            }
        }

        let mut gene_order: Vec<usize> = Vec::new();
        let mut gene_best_rank: HashMap<usize, u8> = HashMap::new();
        for &feature_idx in active_features.iter() {
            let feature = &chrom.features[feature_idx];
            if feature.end < pos {
                continue;
            }
            let rank = feature.kind.exact_rank();
            for &gene_idx_u32 in feature.gene_indices.iter() {
                let gene_idx = gene_idx_u32 as usize;
                if gene_idx >= chrom.genes.len() {
                    continue;
                }
                match gene_best_rank.get_mut(&gene_idx) {
                    Some(current) => {
                        if rank < *current {
                            *current = rank;
                        }
                    }
                    None => {
                        gene_best_rank.insert(gene_idx, rank);
                        gene_order.push(gene_idx);
                    }
                }
            }
        }

        let add_limit = pos.saturating_add(flank);
        while gene_add_ptr < chrom.genes.len() && chrom.genes[gene_add_ptr].start <= add_limit {
            active_genes.insert(gene_add_ptr);
            gene_heap.push(Reverse((chrom.genes[gene_add_ptr].end, gene_add_ptr)));
            gene_add_ptr += 1;
        }
        let keep_min_end = pos.saturating_sub(flank);
        while let Some(Reverse((end, idx))) = gene_heap.peek().copied() {
            if end < keep_min_end {
                gene_heap.pop();
                active_genes.remove(&idx);
            } else {
                break;
            }
        }

        let mut entries: Vec<String> = Vec::new();
        for gene_idx in gene_order.iter().copied() {
            let gene = &chrom.genes[gene_idx];
            let label = GffFeatureKind::exact_label_from_rank(*gene_best_rank.get(&gene_idx).unwrap_or(&5));
            entries.push(format_annotation_triplet(label, &gene.id, &gene.desc));
        }

        let mut nearby: Vec<(i64, i64, usize, &'static str)> = Vec::new();
        for &gene_idx in active_genes.iter() {
            if gene_best_rank.contains_key(&gene_idx) {
                continue;
            }
            let gene = &chrom.genes[gene_idx];
            if pos < gene.start {
                let dist = gene.start.saturating_sub(pos);
                if dist > flank {
                    continue;
                }
                let label = match gene.strand {
                    GffStrand::Minus => "Downstream2kb",
                    _ => "Upstream2kb",
                };
                nearby.push((dist, gene.start, gene_idx, label));
            } else if pos > gene.end {
                let dist = pos.saturating_sub(gene.end);
                if dist > flank {
                    continue;
                }
                let label = match gene.strand {
                    GffStrand::Minus => "Upstream2kb",
                    _ => "Downstream2kb",
                };
                nearby.push((dist, gene.start, gene_idx, label));
            }
        }
        nearby.sort_by(|a, b| {
            (a.0, a.1, chrom.genes[a.2].id.as_str()).cmp(&(b.0, b.1, chrom.genes[b.2].id.as_str()))
        });
        for (_dist, _start, gene_idx, label) in nearby.into_iter() {
            let gene = &chrom.genes[gene_idx];
            entries.push(format_annotation_triplet(label, &gene.id, "NA"));
        }
        out[out_idx] = join_annotation_entries(entries);
    }
}

fn annotate_site_broaden_on_chrom(chrom: &ChromIndex, pos: i64, window_bp: i64) -> String {
    let mut out = vec![BROADEN_EMPTY.to_string()];
    annotate_many_broaden_on_chrom(chrom, &[(0usize, pos)], window_bp, &mut out);
    out.pop().unwrap_or_else(|| BROADEN_EMPTY.to_string())
}

fn annotate_many_broaden_on_chrom(
    chrom: &ChromIndex,
    items: &[(usize, i64)],
    window_bp: i64,
    out: &mut [String],
) {
    let window = window_bp.max(0);
    let mut active_genes: BTreeSet<usize> = BTreeSet::new();
    let mut gene_heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    let mut gene_add_ptr = 0usize;

    for &(out_idx, pos) in items.iter() {
        let add_limit = pos.saturating_add(window);
        while gene_add_ptr < chrom.genes.len() && chrom.genes[gene_add_ptr].start <= add_limit {
            active_genes.insert(gene_add_ptr);
            gene_heap.push(Reverse((chrom.genes[gene_add_ptr].end, gene_add_ptr)));
            gene_add_ptr += 1;
        }
        let keep_min_end = pos.saturating_sub(window);
        while let Some(Reverse((end, idx))) = gene_heap.peek().copied() {
            if end < keep_min_end {
                gene_heap.pop();
                active_genes.remove(&idx);
            } else {
                break;
            }
        }

        let mut parts: Vec<String> = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for &gene_idx in active_genes.iter() {
            let gene = &chrom.genes[gene_idx];
            if !seen.insert(gene.id.as_str()) {
                continue;
            }
            parts.push(format!("{}:{}/NA", gene.id, gene.desc));
        }
        out[out_idx] = if parts.is_empty() {
            BROADEN_EMPTY.to_string()
        } else {
            parts.join(";")
        };
    }
}

fn fetch_gene_panel_ranges_on_chrom(
    chrom: &ChromIndex,
    items: &[(usize, i64, i64)],
    buckets: &mut [Vec<(String, String, i64, i64, String, String)>],
) {
    if items.is_empty() || chrom.features.is_empty() {
        return;
    }

    let mut sorted_items = items.to_vec();
    sorted_items.sort_by_key(|x| x.1);

    let mut active_features: BTreeSet<usize> = BTreeSet::new();
    let mut feature_heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    let mut feature_add_ptr = 0usize;

    for (order_idx, start, end) in sorted_items.into_iter() {
        while feature_add_ptr < chrom.features.len() && chrom.features[feature_add_ptr].start <= end {
            active_features.insert(feature_add_ptr);
            feature_heap.push(Reverse((chrom.features[feature_add_ptr].end, feature_add_ptr)));
            feature_add_ptr += 1;
        }
        while let Some(Reverse((feature_end, feature_idx))) = feature_heap.peek().copied() {
            if feature_end < start {
                feature_heap.pop();
                active_features.remove(&feature_idx);
            } else {
                break;
            }
        }

        let Some(bucket) = buckets.get_mut(order_idx) else {
            continue;
        };
        for &feature_idx in active_features.iter() {
            let feature = &chrom.features[feature_idx];
            if feature.start > end || feature.end < start {
                continue;
            }
            let Some(feature_name) = feature.kind.gene_panel_name() else {
                continue;
            };
            bucket.push((
                chrom.chrom.clone(),
                feature_name.to_string(),
                feature.start,
                feature.end,
                feature.strand.as_text().to_string(),
                feature.id.clone(),
            ));
        }
    }
}

fn format_annotation_triplet(label: &str, gene_id: &str, desc: &str) -> String {
    format!(
        "{};{};{}",
        clean_or_na(label),
        normalize_entity_id_or_na(gene_id),
        clean_or_na(desc)
    )
}

fn join_annotation_entries(entries: Vec<String>) -> String {
    if entries.is_empty() {
        return DESC_INTERGENIC.to_string();
    }
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for entry in entries.into_iter() {
        let text = entry.trim().to_string();
        if text.is_empty() || !seen.insert(text.clone()) {
            continue;
        }
        out.push(text);
    }
    if out.is_empty() {
        DESC_INTERGENIC.to_string()
    } else {
        out.join(" | ")
    }
}

fn normalize_chr(chrom: &str) -> String {
    let mut text = chrom.trim().to_string();
    if text.len() >= 3 && text[..3].eq_ignore_ascii_case("chr") {
        text = text[3..].to_string();
    }
    text
}

fn normalize_entity_id_opt(value: &str) -> Option<String> {
    let cleaned = clean_opt(value)?;
    let norm = match cleaned.split_once(':') {
        Some((_, tail)) if !tail.trim().is_empty() => tail.trim().to_string(),
        _ => cleaned,
    };
    let trimmed = norm.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("na") {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn normalize_entity_id_or_na(value: &str) -> String {
    normalize_entity_id_opt(value).unwrap_or_else(|| "NA".to_string())
}

fn split_entity_ids(value: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for token in value.split(',') {
        let Some(norm) = normalize_entity_id_opt(token) else {
            continue;
        };
        if seen.insert(norm.clone()) {
            out.push(norm);
        }
    }
    out
}

fn clean_opt(value: &str) -> Option<String> {
    let decoded = percent_decode_lossy(value);
    let parts: Vec<&str> = decoded.split_whitespace().collect();
    if parts.is_empty() {
        return None;
    }
    let text = parts.join(" ");
    if text.is_empty() || text.eq_ignore_ascii_case("nan") || text.eq_ignore_ascii_case("na") {
        None
    } else {
        Some(text)
    }
}

fn clean_or_na(value: &str) -> String {
    clean_opt(value).unwrap_or_else(|| "NA".to_string())
}

fn percent_decode_lossy(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut idx = 0usize;
    while idx < bytes.len() {
        if bytes[idx] == b'%' && idx + 2 < bytes.len() {
            let hi = hex_value(bytes[idx + 1]);
            let lo = hex_value(bytes[idx + 2]);
            if let (Some(hi_v), Some(lo_v)) = (hi, lo) {
                out.push((hi_v << 4) | lo_v);
                idx += 3;
                continue;
            }
        }
        out.push(bytes[idx]);
        idx += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[inline]
fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn chrom_sort_cmp(a: &String, b: &String) -> Ordering {
    let a_num = a.parse::<u32>();
    let b_num = b.parse::<u32>();
    match (a_num, b_num) {
        (Ok(x), Ok(y)) => x.cmp(&y),
        (Ok(_), Err(_)) => Ordering::Less,
        (Err(_), Ok(_)) => Ordering::Greater,
        _ => a.cmp(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let uniq = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("janusx.{uniq}.{name}"))
    }

    fn write_test_gff() -> std::path::PathBuf {
        let path = temp_path("test.gff3");
        let gff = "\
chr1\tsrc\tgene\t100\t200\t.\t+\t.\tID=gene:GeneA;Name=GeneA;description=Alpha kinase\n\
chr1\tsrc\tmRNA\t100\t200\t.\t+\t.\tID=transcript:TxA;Parent=gene:GeneA\n\
chr1\tsrc\tCDS\t120\t150\t.\t+\t0\tID=cds:Cdsa;Parent=transcript:TxA\n\
chr1\tsrc\texon\t100\t180\t.\t+\t.\tID=exon:ExA;Parent=transcript:TxA\n\
chr1\tsrc\tgene\t300\t360\t.\t-\t.\tID=gene:GeneB;Name=GeneB;description=Beta protein\n\
chr1\tsrc\tmRNA\t300\t360\t.\t-\t.\tID=transcript:TxB;Parent=gene:GeneB\n\
chr1\tsrc\tCDS\t320\t340\t.\t-\t0\tID=cds:Cdsb;Parent=transcript:TxB\n";
        fs::write(&path, gff).unwrap();
        path
    }

    #[test]
    fn test_gff_annotation_index_desc_and_broaden() {
        let path = write_test_gff();
        let source = gff_source_meta(path.to_str().unwrap()).unwrap();
        let index = parse_gff_annotation_index(path.to_str().unwrap(), source).unwrap();
        assert_eq!(
            index.annotate_site_desc("1", 130, 2_000),
            "CDS;GeneA;Alpha kinase"
        );
        assert_eq!(
            index.annotate_site_desc("chr1", 250, 2_000),
            "Downstream2kb;GeneA;NA | Downstream2kb;GeneB;NA"
                .replace("Downstream2kb;GeneA;NA | Downstream2kb;GeneB;NA", "Downstream2kb;GeneA;NA | Downstream2kb;GeneB;NA")
        );
        assert_eq!(
            index.annotate_site_broaden("1", 250, 80),
            "GeneA:Alpha kinase/NA;GeneB:Beta protein/NA"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_gff_annotation_index_cache_roundtrip() {
        let gff_path = write_test_gff();
        let cache_path = temp_path("test.jxgff.bin");
        let built = build_or_load_gff_index(
            gff_path.to_str().unwrap(),
            Some(cache_path.to_str().unwrap()),
        )
        .unwrap();
        let loaded = load_gff_index(cache_path.to_str().unwrap()).unwrap();
        assert_eq!(built.source, loaded.source);
        assert_eq!(built.annotate_site_desc("1", 130, 2_000), loaded.annotate_site_desc("1", 130, 2_000));
        let _ = fs::remove_file(gff_path);
        let _ = fs::remove_file(cache_path);
    }

    #[test]
    fn test_fetch_gene_panel_ranges() {
        let path = write_test_gff();
        let source = gff_source_meta(path.to_str().unwrap()).unwrap();
        let index = parse_gff_annotation_index(path.to_str().unwrap(), source).unwrap();
        let (chroms, features, starts, ends, strands, ids) = index
            .fetch_gene_panel_ranges(&["1".to_string()], &[90], &[160])
            .unwrap();
        assert!(!chroms.is_empty());
        assert_eq!(chroms[0], "1");
        assert!(features.iter().any(|x| x == "gene"));
        assert!(features.iter().any(|x| x == "CDS"));
        assert!(starts.iter().zip(ends.iter()).all(|(s, e)| s <= e));
        assert!(strands.iter().all(|x| x == "+" || x == "-" || x == "."));
        assert!(ids.iter().any(|x| x == "GeneA"));
        let _ = fs::remove_file(path);
    }
}
