//! Bounded exact reference sequence codes for imputation clusters.

use super::composite::PackedBiallelicReference;
use rayon::prelude::*;
use rayon::ThreadPool;

pub(crate) struct ClusterSequenceIndex {
    clusters: Vec<Option<ClusterSequenceCodes>>,
}

pub(crate) enum ClusterSequenceCodes {
    U8(Box<[(u8, u8)]>),
    U16(Box<[(u16, u16)]>),
    U32(Box<[(u32, u32)]>),
    U64(Box<[(u64, u64)]>),
}

impl ClusterSequenceIndex {
    pub(crate) fn new(
        reference: &PackedBiallelicReference,
        members: &[Vec<usize>],
        budget_bytes: usize,
        pool: &ThreadPool,
    ) -> Self {
        // Admit clusters before allocating; parallel builders cannot overrun
        // the retained code budget. Oversized clusters retain scalar scanning.
        let started = std::time::Instant::now();
        let Some(metadata_bytes) = members
            .len()
            .checked_mul(std::mem::size_of::<Option<ClusterSequenceCodes>>())
        else {
            return Self {
                clusters: Vec::new(),
            };
        };
        let Some(mut remaining) = budget_bytes.checked_sub(metadata_bytes) else {
            return Self {
                clusters: Vec::new(),
            };
        };
        let mut width_counts = [0usize; 9];
        let admitted: Vec<_> = members
            .iter()
            .map(|markers| {
                let width: usize = match markers.len() {
                    1..=8 => 1,
                    9..=16 => 2,
                    17..=32 => 4,
                    33..=64 => 8,
                    _ => return 0,
                };
                width_counts[width] += 1;
                let Some(bytes) = reference.n_haplotypes().checked_mul(2 * width) else {
                    return 0;
                };
                if bytes > remaining {
                    return 0;
                }
                remaining -= bytes;
                width
            })
            .collect();
        let clusters: Vec<_> = pool.install(|| {
            members
                .par_iter()
                .zip(admitted.par_iter())
                .map(|(markers, &width)| {
                    let encode = |donor| {
                        let mut bits = 0u64;
                        let mut known = 0u64;
                        for (offset, &marker) in markers.iter().enumerate() {
                            if let Some(allele) = reference.allele_unchecked(marker, donor) {
                                bits |= u64::from(allele) << offset;
                                known |= 1u64 << offset;
                            }
                        }
                        (bits, known)
                    };
                    Some(match width {
                        1 => ClusterSequenceCodes::U8(
                            (0..reference.n_haplotypes())
                                .map(|d| {
                                    let (b, k) = encode(d);
                                    (b as u8, k as u8)
                                })
                                .collect(),
                        ),
                        2 => ClusterSequenceCodes::U16(
                            (0..reference.n_haplotypes())
                                .map(|d| {
                                    let (b, k) = encode(d);
                                    (b as u16, k as u16)
                                })
                                .collect(),
                        ),
                        4 => ClusterSequenceCodes::U32(
                            (0..reference.n_haplotypes())
                                .map(|d| {
                                    let (b, k) = encode(d);
                                    (b as u32, k as u32)
                                })
                                .collect(),
                        ),
                        8 => ClusterSequenceCodes::U64(
                            (0..reference.n_haplotypes()).map(encode).collect(),
                        ),
                        _ => return None,
                    })
                })
                .collect()
        });
        if std::env::var_os("JANUSX_IMP_HMM_PROFILE").is_some() {
            eprintln!("JANUSX_IMP_CLUSTER_CODE_PROFILE clusters={} indexed={} u8={} u16={} u32={} u64={} rejected={} retained_bytes={} budget_bytes={} build_s={:.6}",
                members.len(), clusters.iter().filter(|c| c.is_some()).count(),
                width_counts[1], width_counts[2], width_counts[4], width_counts[8],
                width_counts[0], budget_bytes - remaining, budget_bytes,
                started.elapsed().as_secs_f64());
        }
        Self { clusters }
    }

    pub(crate) fn cluster(&self, cluster: usize) -> Option<&ClusterSequenceCodes> {
        self.clusters.get(cluster).and_then(Option::as_ref)
    }
}

impl ClusterSequenceCodes {
    pub(crate) fn encode_query(
        &self,
        markers: &[usize],
        query: &[Option<u16>],
    ) -> Option<(u64, u64)> {
        debug_assert!(markers.len() <= 64);
        let mut bits = 0u64;
        let mut known = 0u64;
        for (offset, &marker) in markers.iter().enumerate() {
            if let Some(allele) = query[marker] {
                if allele > 1 {
                    return None;
                }
                bits |= u64::from(allele) << offset;
                known |= 1u64 << offset;
            }
        }
        Some((bits, known))
    }
}
