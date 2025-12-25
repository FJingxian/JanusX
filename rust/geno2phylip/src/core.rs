use anyhow::{anyhow, Result};
use rand::Rng;
use std::{fs::File, io::{BufWriter, Write}, path::{Path, PathBuf}};

use crate::writers::*;
use crate::readers::{npz, vcf, plink};

#[derive(Clone)]
pub struct Site {
    pub chr: String,
    pub pos: u64,
    pub ref_base: u8, // A/C/G/T
    pub alt_base: u8,
}

/// IUPAC for simple diploid biallelic het
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

/// reorder outgroup first (if exists)
fn reorder_outgroup<T: Clone>(samples: &[String], seqs: &[T], outgroup: &Option<String>) -> (Vec<String>, Vec<T>) {
    let Some(og) = outgroup else { return (samples.to_vec(), seqs.to_vec()); };
    if let Some(idx) = samples.iter().position(|x| x == og) {
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
    } else {
        (samples.to_vec(), seqs.to_vec())
    }
}

fn prepare_out_paths(out_dir: &str, out_prefix: &str, min_samples: usize) -> Result<(PathBuf, Option<BufWriter<File>>)> {
    std::fs::create_dir_all(out_dir)?;
    let base = PathBuf::from(out_dir).join(format!("{}.min{}", out_prefix, min_samples));

    let used = None;
    Ok((base, used))
}

pub fn convert_npz(npz_path: &Path, snp_path: &Path, idv_path: &Path, opts_py: crate::ConvertOptions) -> Result<()> {
    let opts = opts_py;
    let samples = npz::load_idv(idv_path)?;
    let sites = npz::load_snp(snp_path)?;
    let mtx = npz::load_npz_i8_matrix(npz_path)?; // m x n, i8, missing=-9

    let n = samples.len();
    if n == 0 { return Err(anyhow!("No samples in idv")); }
    if mtx.ncols() != n { return Err(anyhow!("Matrix n != idv n")); }
    if mtx.nrows() != sites.len() { return Err(anyhow!("Matrix m != snp m")); }

    let min_samples = opts.min_samples_locus.min(n);
    let (outbase, mut used_writer) = {
        std::fs::create_dir_all(&opts.out_dir)?;
        let base = PathBuf::from(&opts.out_dir).join(format!("{}.min{}", opts.out_prefix, min_samples));
        let used = if opts.used_sites {
            let mut f = BufWriter::new(File::create(base.with_extension("used_sites.tsv"))?);
            writeln!(f, "#CHROM\tPOS\tNUM_SAMPLES")?;
            Some(f)
        } else { None };
        (base, used)
    };

    let mut seqs: Vec<Vec<u8>> = (0..n).map(|_| Vec::with_capacity(mtx.nrows())).collect();
    let mut seqs_bin: Option<Vec<Vec<u8>>> = if opts.write_nexus_binary {
        Some((0..n).map(|_| Vec::with_capacity(mtx.nrows())).collect())
    } else { None };

    let mut rng = rand::thread_rng();
    let mut accepted = 0usize;
    let mut shallow = 0usize;

    for (i, site) in sites.iter().enumerate() {
        let row = mtx.row(i);
        let present = row.iter().filter(|&&g| g != -9).count();
        if present < min_samples { shallow += 1; continue; }

        if let Some(w) = used_writer.as_mut() {
            writeln!(w, "{}\t{}\t{}", site.chr, site.pos, present)?;
        }

        for (s, &g) in row.iter().enumerate() {
            let dna = match g {
                -9 => b'N',
                0 => site.ref_base,
                2 => site.alt_base,
                1 => if opts.resolve_het {
                    if rng.gen_bool(0.5) { site.ref_base } else { site.alt_base }
                } else {
                    iupac(site.ref_base, site.alt_base)
                },
                _ => b'N',
            };
            seqs[s].push(dna);

            if let Some(sb) = seqs_bin.as_mut() {
                let b = match g {
                    -9 => b'?',
                    0 => b'0',
                    1 => b'1',
                    2 => b'2',
                    _ => b'?',
                };
                sb[s].push(b);
            }
        }
        accepted += 1;
    }

    let (samples2, seqs2) = reorder_outgroup(&samples, &seqs, &opts.outgroup);
    let seqs_bin2 = if let Some(sb) = seqs_bin {
        let (_, sb2) = reorder_outgroup(&samples, &sb, &opts.outgroup);
        Some(sb2)
    } else { None };

    if opts.write_phylip {
        write_phylip(&outbase.with_extension("phy"), &samples2, &seqs2)?;
    }
    if opts.write_fasta {
        write_fasta(&outbase.with_extension("fasta"), &samples2, &seqs2)?;
    }
    if opts.write_nexus {
        write_nexus_dna(&outbase.with_extension("nexus"), &samples2, &seqs2)?;
    }
    if opts.write_nexus_binary {
        write_nexus_binary(&outbase.with_extension("bin.nexus"), &samples2, seqs_bin2.as_ref().unwrap())?;
    }

    eprintln!("NPZ done. Accepted sites: {}, excluded by missing filter: {}", accepted, shallow);
    Ok(())
}

pub fn convert_vcf(vcf_path: &Path, opts: crate::ConvertOptions) -> Result<()> {
    // Reader returns samples, and an iterator-like stream over sites
    let mut reader = vcf::VcfStream::open(vcf_path)?;
    let samples = reader.samples().to_vec();
    let n = samples.len();
    if n == 0 { return Err(anyhow!("No samples in VCF header")); }
    let min_samples = opts.min_samples_locus.min(n);

    std::fs::create_dir_all(&opts.out_dir)?;
    let outbase = PathBuf::from(&opts.out_dir).join(format!("{}.min{}", opts.out_prefix, min_samples));
    let mut used_writer = if opts.used_sites {
        let mut f = BufWriter::new(File::create(outbase.with_extension("used_sites.tsv"))?);
        writeln!(f, "#CHROM\tPOS\tNUM_SAMPLES")?;
        Some(f)
    } else { None };

    let mut seqs: Vec<Vec<u8>> = (0..n).map(|_| Vec::new()).collect();
    let mut seqs_bin: Option<Vec<Vec<u8>>> = if opts.write_nexus_binary {
        Some((0..n).map(|_| Vec::new()).collect())
    } else { None };
    let mut rng = rand::thread_rng();

    let mut accepted = 0usize;
    let mut shallow = 0usize;
    let mut skipped = 0usize;

    while let Some(rec) = reader.next_site()? {
        // rec: (Site, genos_i8) where genos is len n with -9/0/1/2
        let (site, genos) = rec;

        let present = genos.iter().filter(|&&g| g != -9).count();
        if present < min_samples { shallow += 1; continue; }

        if let Some(w) = used_writer.as_mut() {
            writeln!(w, "{}\t{}\t{}", site.chr, site.pos, present)?;
        }

        for (i, g) in genos.iter().enumerate() {
            let dna = match *g {
                -9 => b'N',
                0 => site.ref_base,
                2 => site.alt_base,
                1 => if opts.resolve_het {
                    if rng.gen_bool(0.5) { site.ref_base } else { site.alt_base }
                } else {
                    iupac(site.ref_base, site.alt_base)
                },
                _ => { skipped += 1; b'N' }
            };
            seqs[i].push(dna);

            if let Some(sb) = seqs_bin.as_mut() {
                let b = match *g {
                    -9 => b'?',
                    0 => b'0',
                    1 => b'1',
                    2 => b'2',
                    _ => b'?',
                };
                sb[i].push(b);
            }
        }

        accepted += 1;
    }

    let (samples2, seqs2) = reorder_outgroup(&samples, &seqs, &opts.outgroup);
    let seqs_bin2 = if let Some(sb) = seqs_bin {
        let (_, sb2) = reorder_outgroup(&samples, &sb, &opts.outgroup);
        Some(sb2)
    } else { None };

    if opts.write_phylip { write_phylip(&outbase.with_extension("phy"), &samples2, &seqs2)?; }
    if opts.write_fasta { write_fasta(&outbase.with_extension("fasta"), &samples2, &seqs2)?; }
    if opts.write_nexus { write_nexus_dna(&outbase.with_extension("nexus"), &samples2, &seqs2)?; }
    if opts.write_nexus_binary {
        write_nexus_binary(&outbase.with_extension("bin.nexus"), &samples2, seqs_bin2.as_ref().unwrap())?;
    }

    eprintln!("VCF done. Accepted: {}, shallow: {}, skipped states: {}", accepted, shallow, skipped);
    Ok(())
}

pub fn convert_plink(bed: &Path, bim: &Path, fam: &Path, opts: crate::ConvertOptions) -> Result<()> {
    let mut reader = plink::PlinkStream::open(bed, bim, fam)?;
    let samples = reader.samples().to_vec();
    let n = samples.len();
    if n == 0 { return Err(anyhow!("No samples in FAM")); }
    let min_samples = opts.min_samples_locus.min(n);

    std::fs::create_dir_all(&opts.out_dir)?;
    let outbase = PathBuf::from(&opts.out_dir).join(format!("{}.min{}", opts.out_prefix, min_samples));
    let mut used_writer = if opts.used_sites {
        let mut f = BufWriter::new(File::create(outbase.with_extension("used_sites.tsv"))?);
        writeln!(f, "#CHROM\tPOS\tNUM_SAMPLES")?;
        Some(f)
    } else { None };

    let mut seqs: Vec<Vec<u8>> = (0..n).map(|_| Vec::new()).collect();
    let mut seqs_bin: Option<Vec<Vec<u8>>> = if opts.write_nexus_binary {
        Some((0..n).map(|_| Vec::new()).collect())
    } else { None };
    let mut rng = rand::thread_rng();

    let mut accepted = 0usize;
    let mut shallow = 0usize;

    while let Some((site, genos)) = reader.next_site()? {
        let present = genos.iter().filter(|&&g| g != -9).count();
        if present < min_samples { shallow += 1; continue; }

        if let Some(w) = used_writer.as_mut() {
            writeln!(w, "{}\t{}\t{}", site.chr, site.pos, present)?;
        }

        for (i, g) in genos.iter().enumerate() {
            let dna = match *g {
                -9 => b'N',
                0 => site.ref_base,
                2 => site.alt_base,
                1 => if opts.resolve_het {
                    if rng.gen_bool(0.5) { site.ref_base } else { site.alt_base }
                } else {
                    iupac(site.ref_base, site.alt_base)
                },
                _ => b'N',
            };
            seqs[i].push(dna);

            if let Some(sb) = seqs_bin.as_mut() {
                let b = match *g {
                    -9 => b'?',
                    0 => b'0',
                    1 => b'1',
                    2 => b'2',
                    _ => b'?',
                };
                sb[i].push(b);
            }
        }
        accepted += 1;
    }

    let (samples2, seqs2) = reorder_outgroup(&samples, &seqs, &opts.outgroup);
    let seqs_bin2 = if let Some(sb) = seqs_bin {
        let (_, sb2) = reorder_outgroup(&samples, &sb, &opts.outgroup);
        Some(sb2)
    } else { None };

    if opts.write_phylip { write_phylip(&outbase.with_extension("phy"), &samples2, &seqs2)?; }
    if opts.write_fasta { write_fasta(&outbase.with_extension("fasta"), &samples2, &seqs2)?; }
    if opts.write_nexus { write_nexus_dna(&outbase.with_extension("nexus"), &samples2, &seqs2)?; }
    if opts.write_nexus_binary {
        write_nexus_binary(&outbase.with_extension("bin.nexus"), &samples2, seqs_bin2.as_ref().unwrap())?;
    }

    eprintln!("PLINK done. Accepted: {}, shallow: {}", accepted, shallow);
    Ok(())
}