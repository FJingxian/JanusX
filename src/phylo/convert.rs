use std::ffi::OsString;
use std::path::{Path, PathBuf};

use rand::Rng;
use rayon::prelude::*;
use rayon::ThreadPool;

use crate::gfcore::{process_snp_row_with_stats, BedSnpIter, HmpSnpIter, SiteInfo, VcfSnpIter};

use super::writers::{write_fasta, write_phylip};
use super::{FastTreePrepOptions, FastTreePrepResult};

trait RawSnpReader {
    fn sample_ids(&self) -> &[String];
    fn next_site_raw(&mut self) -> Option<(Vec<f32>, SiteInfo)>;
}

impl RawSnpReader for VcfSnpIter {
    fn sample_ids(&self) -> &[String] {
        &self.samples
    }

    fn next_site_raw(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        self.next_snp_raw()
    }
}

impl RawSnpReader for HmpSnpIter {
    fn sample_ids(&self) -> &[String] {
        &self.samples
    }

    fn next_site_raw(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        self.next_snp_raw()
    }
}

impl RawSnpReader for BedSnpIter {
    fn sample_ids(&self) -> &[String] {
        &self.samples
    }

    fn next_site_raw(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        self.next_snp_raw()
    }
}

const PREP_TARGET_CELLS: usize = 1_000_000;
const PREP_MIN_BATCH_SITES: usize = 32;
const PREP_MAX_BATCH_SITES: usize = 512;
const PREP_PAR_MIN_CELLS: usize = 32 * 1024;

type RawSiteRecord = (Vec<f32>, SiteInfo);

fn append_suffix(path: &str, suffix: &str) -> PathBuf {
    let mut os = OsString::from(path);
    os.push(suffix);
    PathBuf::from(os)
}

fn base_byte(s: &str) -> Option<u8> {
    let bytes = s.as_bytes();
    if bytes.len() != 1 {
        return None;
    }
    let b = bytes[0].to_ascii_uppercase();
    if matches!(b, b'A' | b'C' | b'G' | b'T') {
        Some(b)
    } else {
        None
    }
}

fn iupac(refb: u8, altb: u8) -> u8 {
    match (refb, altb) {
        (b'A', b'G') | (b'G', b'A') => b'R',
        (b'C', b'T') | (b'T', b'C') => b'Y',
        (b'G', b'C') | (b'C', b'G') => b'S',
        (b'A', b'T') | (b'T', b'A') => b'W',
        (b'G', b'T') | (b'T', b'G') => b'K',
        (b'A', b'C') | (b'C', b'A') => b'M',
        _ => b'N',
    }
}

fn encode_base<R: Rng + ?Sized>(g: f32, refb: u8, altb: u8, resolve_het: bool, rng: &mut R) -> u8 {
    if g < 0.0 {
        return b'N';
    }
    if g < 0.5 {
        return refb;
    }
    if g > 1.5 {
        return altb;
    }
    if resolve_het {
        if rng.random_bool(0.5) {
            refb
        } else {
            altb
        }
    } else {
        iupac(refb, altb)
    }
}

fn uniquify_names(names: &[String]) -> Vec<String> {
    let mut used = std::collections::HashSet::<String>::new();
    let mut out = Vec::with_capacity(names.len());
    for raw in names.iter() {
        let base = raw.trim().to_string();
        let base = if base.is_empty() {
            "sample".to_string()
        } else {
            base
        };
        let mut name = base.clone();
        let mut idx = 2usize;
        while used.contains(&name) {
            name = format!("{base}_{idx}");
            idx += 1;
        }
        used.insert(name.clone());
        out.push(name);
    }
    out
}

fn sanitize_phylip_names(names: &[String]) -> Vec<String> {
    let mut used = std::collections::HashSet::<String>::new();
    let mut out = Vec::with_capacity(names.len());
    for raw in names.iter() {
        let mut base = raw.split_whitespace().collect::<Vec<_>>().join("_");
        base = base.replace('\'', "_");
        if base.is_empty() {
            base = "sample".to_string();
        }
        let mut name = base.clone();
        let mut idx = 2usize;
        while used.contains(&name) {
            name = format!("{base}_{idx}");
            idx += 1;
        }
        used.insert(name.clone());
        out.push(name);
    }
    out
}

fn reorder_outgroup<T: Clone>(
    samples: &[String],
    seqs: &[T],
    outgroup: &Option<String>,
) -> (Vec<String>, Vec<T>) {
    let Some(og) = outgroup.as_ref() else {
        return (samples.to_vec(), seqs.to_vec());
    };
    let Some(idx) = samples.iter().position(|x| x == og) else {
        return (samples.to_vec(), seqs.to_vec());
    };

    let mut ns = Vec::with_capacity(samples.len());
    let mut nq = Vec::with_capacity(seqs.len());
    ns.push(samples[idx].clone());
    nq.push(seqs[idx].clone());
    for i in 0..samples.len() {
        if i != idx {
            ns.push(samples[i].clone());
            nq.push(seqs[i].clone());
        }
    }
    (ns, nq)
}

fn prep_batch_sites(n_samples: usize) -> usize {
    let by_cells = PREP_TARGET_CELLS / n_samples.max(1);
    by_cells.clamp(PREP_MIN_BATCH_SITES, PREP_MAX_BATCH_SITES)
}

fn prep_threads(threads: usize) -> usize {
    if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

fn should_parallel_prep(n_samples: usize, batch_sites: usize, threads: usize) -> bool {
    prep_threads(threads) > 1 && n_samples.saturating_mul(batch_sites) >= PREP_PAR_MIN_CELLS
}

fn build_prep_pool(threads: usize) -> Result<Option<ThreadPool>, String> {
    if threads <= 1 {
        return Ok(None);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map(Some)
        .map_err(|e| format!("rayon pool: {e}"))
}

fn encode_site_row<R: Rng + ?Sized>(
    mut row: Vec<f32>,
    mut site: SiteInfo,
    opts: &FastTreePrepOptions,
    apply_het_filter: bool,
    n_samples: usize,
    rng: &mut R,
) -> Result<Option<Vec<u8>>, String> {
    if row.len() != n_samples {
        return Err(format!(
            "Sample size mismatch while preparing FastTree alignment: got {}, expected {}",
            row.len(),
            n_samples
        ));
    }

    let keep = process_snp_row_with_stats(
        &mut row,
        &mut site.ref_allele,
        &mut site.alt_allele,
        opts.maf as f32,
        opts.missing_rate as f32,
        false,
        apply_het_filter,
        opts.het as f32,
    );
    if keep.is_none() {
        return Ok(None);
    }

    let Some(refb) = base_byte(&site.ref_allele) else {
        return Ok(None);
    };
    let Some(altb) = base_byte(&site.alt_allele) else {
        return Ok(None);
    };

    let mut encoded = Vec::with_capacity(n_samples);
    for g in row.into_iter() {
        encoded.push(encode_base(g, refb, altb, opts.resolve_het, rng));
    }
    Ok(Some(encoded))
}

fn append_encoded_rows(
    seqs: &mut [Vec<u8>],
    encoded_rows: &[Vec<u8>],
    pool: Option<&ThreadPool>,
    parallel: bool,
) {
    if encoded_rows.is_empty() {
        return;
    }
    let n_sites = encoded_rows.len();
    let append_one = |sample_idx: usize, seq: &mut Vec<u8>| {
        seq.reserve(n_sites);
        for row in encoded_rows.iter() {
            seq.push(row[sample_idx]);
        }
    };

    if parallel {
        match pool {
            Some(p) => p.install(|| {
                seqs.par_iter_mut()
                    .enumerate()
                    .for_each(|(sample_idx, seq)| append_one(sample_idx, seq));
            }),
            None => {
                seqs.par_iter_mut()
                    .enumerate()
                    .for_each(|(sample_idx, seq)| append_one(sample_idx, seq));
            }
        }
    } else {
        for (sample_idx, seq) in seqs.iter_mut().enumerate() {
            append_one(sample_idx, seq);
        }
    }
}

fn flush_raw_batch(
    raw_batch: &mut Vec<RawSiteRecord>,
    seqs: &mut [Vec<u8>],
    opts: &FastTreePrepOptions,
    apply_het_filter: bool,
    pool: Option<&ThreadPool>,
) -> Result<(usize, usize), String> {
    if raw_batch.is_empty() {
        return Ok((0, 0));
    }

    let batch = std::mem::take(raw_batch);
    let n_samples = seqs.len();
    let parallel = should_parallel_prep(n_samples, batch.len(), opts.threads);
    let results: Vec<Result<Option<Vec<u8>>, String>> = if parallel {
        let work = || {
            batch
                .into_par_iter()
                .map_init(rand::rng, |rng, (row, site)| {
                    encode_site_row(row, site, opts, apply_het_filter, n_samples, rng)
                })
                .collect()
        };
        match pool {
            Some(p) => p.install(work),
            None => work(),
        }
    } else {
        let mut rng = rand::rng();
        batch
            .into_iter()
            .map(|(row, site)| {
                encode_site_row(row, site, opts, apply_het_filter, n_samples, &mut rng)
            })
            .collect()
    };

    let mut encoded_rows = Vec::with_capacity(results.len());
    let mut skipped = 0usize;
    for item in results.into_iter() {
        match item? {
            Some(row) => encoded_rows.push(row),
            None => skipped += 1,
        }
    }

    let accepted = encoded_rows.len();
    append_encoded_rows(seqs, &encoded_rows, pool, parallel);
    Ok((accepted, skipped))
}

fn build_alignment_from_reader<R: RawSnpReader>(
    reader: &mut R,
    opts: &FastTreePrepOptions,
) -> Result<FastTreePrepResult, String> {
    let sample_ids = uniquify_names(reader.sample_ids());
    if sample_ids.len() < 2 {
        return Err(format!(
            "Need at least 2 samples for FastTree, got {}",
            sample_ids.len()
        ));
    }

    let out_prefix = opts.out_prefix.trim();
    if out_prefix.is_empty() {
        return Err("FastTree output prefix cannot be empty".to_string());
    }
    let prefix_path = Path::new(out_prefix);
    if let Some(parent) = prefix_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }

    let fasta_path = append_suffix(out_prefix, ".fasta");
    let phylip_path = if opts.write_phylip {
        Some(append_suffix(out_prefix, ".phy"))
    } else {
        None
    };

    let mut seqs: Vec<Vec<u8>> = (0..sample_ids.len()).map(|_| Vec::new()).collect();
    let mut accepted = 0usize;
    let mut skipped = 0usize;
    let mut raw_batch: Vec<RawSiteRecord> = Vec::with_capacity(prep_batch_sites(sample_ids.len()));
    let pool = build_prep_pool(opts.threads)?;
    let batch_sites = prep_batch_sites(sample_ids.len());
    // Match the existing `load_genotype_chunks(..., model="add")` default:
    // additive tree workflows do not enable heterozygosity filtering.
    let apply_het_filter = false;

    while let Some((row, site)) = reader.next_site_raw() {
        raw_batch.push((row, site));
        if raw_batch.len() >= batch_sites {
            let (acc, skip) = flush_raw_batch(
                &mut raw_batch,
                &mut seqs,
                opts,
                apply_het_filter,
                pool.as_ref(),
            )?;
            accepted += acc;
            skipped += skip;
        }
    }
    let (acc, skip) = flush_raw_batch(
        &mut raw_batch,
        &mut seqs,
        opts,
        apply_het_filter,
        pool.as_ref(),
    )?;
    accepted += acc;
    skipped += skip;

    if accepted == 0 {
        return Err(
            "No variant sites passed filters; cannot prepare FastTree alignment.".to_string(),
        );
    }

    let (sample_ids, seqs) = reorder_outgroup(&sample_ids, &seqs, &opts.outgroup);
    write_fasta(&fasta_path, &sample_ids, &seqs)?;
    if let Some(path) = phylip_path.as_ref() {
        let phy_names = sanitize_phylip_names(&sample_ids);
        write_phylip(path, &phy_names, &seqs)?;
    }

    Ok(FastTreePrepResult {
        fasta_path: fasta_path.to_string_lossy().to_string(),
        phylip_path: phylip_path.map(|p| p.to_string_lossy().to_string()),
        n_samples: sample_ids.len(),
        n_sites: accepted,
        skipped_sites: skipped,
    })
}

pub fn prepare_alignment(
    input_kind: &str,
    input_path: &str,
    opts: &FastTreePrepOptions,
) -> Result<FastTreePrepResult, String> {
    let kind = input_kind.trim().to_ascii_lowercase();
    match kind.as_str() {
        "vcf" => {
            let mut reader = VcfSnpIter::new_with_fill(input_path, 0.0, 1.0, false, false, 1.0)?;
            build_alignment_from_reader(&mut reader, opts)
        }
        "hmp" => {
            let mut reader = HmpSnpIter::new_with_fill(input_path, 0.0, 1.0, false, false, 1.0)?;
            build_alignment_from_reader(&mut reader, opts)
        }
        "bfile" => {
            let mut reader = BedSnpIter::new_with_fill(input_path, 0.0, 1.0, false, false, 1.0)?;
            build_alignment_from_reader(&mut reader, opts)
        }
        _ => Err(format!(
            "FastTree input_kind must be one of: vcf, hmp, bfile; got {input_kind}"
        )),
    }
}
