use std::sync::OnceLock;

#[inline]
pub(crate) fn decode_plink_bed_hardcall(code: u8) -> Option<f64> {
    match code {
        0b00 => Some(0.0),
        0b10 => Some(1.0),
        0b11 => Some(2.0),
        _ => None, // 0b01 => missing
    }
}

pub(crate) struct PackedByteLut {
    pub(crate) nonmiss: [u8; 256],
    pub(crate) alt_sum: [u8; 256],
    pub(crate) sq_sum: [u8; 256],
    pub(crate) code4: [[u8; 4]; 256],
}

pub(crate) fn packed_byte_lut() -> &'static PackedByteLut {
    static LUT: OnceLock<PackedByteLut> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut nonmiss = [0u8; 256];
        let mut alt_sum = [0u8; 256];
        let mut sq_sum = [0u8; 256];
        let mut code4 = [[0u8; 4]; 256];
        for b in 0u16..=255 {
            let byte = b as u8;
            let mut nm = 0u8;
            let mut alt = 0u8;
            let mut sq = 0u8;
            let mut codes = [0u8; 4];
            for lane in 0..4usize {
                let code = (byte >> (lane * 2)) & 0b11;
                codes[lane] = code;
                match code {
                    0b00 => {
                        nm += 1;
                    }
                    0b10 => {
                        nm += 1;
                        alt += 1;
                        sq += 1;
                    }
                    0b11 => {
                        nm += 1;
                        alt += 2;
                        sq += 4;
                    }
                    _ => {}
                }
            }
            let idx = byte as usize;
            nonmiss[idx] = nm;
            alt_sum[idx] = alt;
            sq_sum[idx] = sq;
            code4[idx] = codes;
        }
        PackedByteLut {
            nonmiss,
            alt_sum,
            sq_sum,
            code4,
        }
    })
}

pub(crate) struct PackedPairLut {
    pub(crate) dot_obs0: [u8; 65536],
    pub(crate) obs_ct: [u8; 65536],
    pub(crate) obs_sum_i: [u8; 65536],
    pub(crate) obs_sum_j: [u8; 65536],
    pub(crate) obs_sum_i2: [u8; 65536],
    pub(crate) obs_sum_j2: [u8; 65536],
    pub(crate) obs_sum_ij: [u8; 65536],
}

#[inline]
pub(crate) fn code_to_dosage_or_zero(code: u8) -> u8 {
    match code {
        0b00 => 0,
        0b10 => 1,
        0b11 => 2,
        _ => 0, // missing
    }
}

pub(crate) fn packed_pair_lut() -> &'static PackedPairLut {
    static LUT: OnceLock<PackedPairLut> = OnceLock::new();
    LUT.get_or_init(|| {
        let code4 = &packed_byte_lut().code4;
        let mut dot_obs0 = [0u8; 65536];
        let mut obs_ct = [0u8; 65536];
        let mut obs_sum_i = [0u8; 65536];
        let mut obs_sum_j = [0u8; 65536];
        let mut obs_sum_i2 = [0u8; 65536];
        let mut obs_sum_j2 = [0u8; 65536];
        let mut obs_sum_ij = [0u8; 65536];

        for bi in 0u16..=255 {
            for bj in 0u16..=255 {
                let idx = ((bi as usize) << 8) | (bj as usize);
                let ci = &code4[bi as usize];
                let cj = &code4[bj as usize];

                let mut dot_v: u8 = 0;
                let mut n_obs: u8 = 0;
                let mut sum_i_obs: u8 = 0;
                let mut sum_j_obs: u8 = 0;
                let mut sum_i2_obs: u8 = 0;
                let mut sum_j2_obs: u8 = 0;
                let mut sum_ij_obs: u8 = 0;
                for lane in 0..4usize {
                    let a = ci[lane];
                    let b = cj[lane];
                    let ga = code_to_dosage_or_zero(a);
                    let gb = code_to_dosage_or_zero(b);
                    dot_v = dot_v.saturating_add(ga.saturating_mul(gb));
                    if a != 0b01 && b != 0b01 {
                        n_obs = n_obs.saturating_add(1);
                        sum_i_obs = sum_i_obs.saturating_add(ga);
                        sum_j_obs = sum_j_obs.saturating_add(gb);
                        sum_i2_obs = sum_i2_obs.saturating_add(ga.saturating_mul(ga));
                        sum_j2_obs = sum_j2_obs.saturating_add(gb.saturating_mul(gb));
                        sum_ij_obs = sum_ij_obs.saturating_add(ga.saturating_mul(gb));
                    }
                }
                dot_obs0[idx] = dot_v;
                obs_ct[idx] = n_obs;
                obs_sum_i[idx] = sum_i_obs;
                obs_sum_j[idx] = sum_j_obs;
                obs_sum_i2[idx] = sum_i2_obs;
                obs_sum_j2[idx] = sum_j2_obs;
                obs_sum_ij[idx] = sum_ij_obs;
            }
        }

        PackedPairLut {
            dot_obs0,
            obs_ct,
            obs_sum_i,
            obs_sum_j,
            obs_sum_i2,
            obs_sum_j2,
            obs_sum_ij,
        }
    })
}

#[inline]
pub(crate) fn is_identity_indices(sample_idx: &[usize], n_samples: usize) -> bool {
    if sample_idx.len() != n_samples {
        return false;
    }
    sample_idx.iter().enumerate().all(|(i, &v)| i == v)
}

#[inline]
pub(crate) fn decode_row_centered_full_lut(
    row: &[u8],
    n_samples: usize,
    code4_lut: &[[u8; 4]; 256],
    value_lut: &[f32; 4],
    out_row: &mut [f32],
) {
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let mut col = 0usize;
    for &b in row.iter().take(full_bytes) {
        let codes = &code4_lut[b as usize];
        out_row[col] = value_lut[codes[0] as usize];
        out_row[col + 1] = value_lut[codes[1] as usize];
        out_row[col + 2] = value_lut[codes[2] as usize];
        out_row[col + 3] = value_lut[codes[3] as usize];
        col += 4;
    }
    if rem > 0 {
        let codes = &code4_lut[row[full_bytes] as usize];
        for lane in 0..rem {
            out_row[col + lane] = value_lut[codes[lane] as usize];
        }
    }
}

#[inline]
pub(crate) fn decode_row_centered_full_lut_f64(
    row: &[u8],
    n_samples: usize,
    code4_lut: &[[u8; 4]; 256],
    value_lut: &[f64; 4],
    out_row: &mut [f64],
) {
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let mut col = 0usize;
    for &b in row.iter().take(full_bytes) {
        let codes = &code4_lut[b as usize];
        out_row[col] = value_lut[codes[0] as usize];
        out_row[col + 1] = value_lut[codes[1] as usize];
        out_row[col + 2] = value_lut[codes[2] as usize];
        out_row[col + 3] = value_lut[codes[3] as usize];
        col += 4;
    }
    if rem > 0 {
        let codes = &code4_lut[row[full_bytes] as usize];
        for lane in 0..rem {
            out_row[col + lane] = value_lut[codes[lane] as usize];
        }
    }
}

#[inline]
pub(crate) fn adaptive_grm_block_rows(
    requested_block_rows: usize,
    m: usize,
    n_out: usize,
    subset_full_scratch_len: usize,
    threads: usize,
) -> usize {
    let base = requested_block_rows.max(1).min(m.max(1));
    let target_mb = std::env::var("JX_GRM_PACKED_BLOCK_TARGET_MB")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(64.0_f64);
    if !(target_mb.is_finite() && target_mb > 0.0) {
        return base;
    }
    let target_bytes = (target_mb * 1024.0_f64 * 1024.0_f64) as usize;
    if target_bytes == 0 {
        return base;
    }
    let workers = if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1)
    };
    let reserve_bytes = subset_full_scratch_len
        .saturating_mul(4)
        .saturating_mul(workers.max(1));
    let row_bytes = n_out.saturating_mul(4).max(1);
    if reserve_bytes >= target_bytes {
        return 1;
    }
    let cap_rows = ((target_bytes - reserve_bytes) / row_bytes).max(1);
    base.min(cap_rows)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn decode_subset_row_from_full_scratch(
    row: &[u8],
    n_samples: usize,
    sample_idx: &[usize],
    flip: bool,
    default_mean_g: f32,
    method: usize,
    eps: f32,
    code4_lut: &[[u8; 4]; 256],
    full_row: &mut [f32],
    out_row: &mut [f32],
) -> (f64, f64) {
    debug_assert!(full_row.len() >= n_samples);
    debug_assert_eq!(out_row.len(), sample_idx.len());
    let n = sample_idx.len();

    let value_lut: [f32; 4] = if flip {
        [2.0_f32, -9.0_f32, 1.0_f32, 0.0_f32]
    } else {
        [0.0_f32, -9.0_f32, 1.0_f32, 2.0_f32]
    };
    decode_row_centered_full_lut(
        row,
        n_samples,
        code4_lut,
        &value_lut,
        &mut full_row[..n_samples],
    );

    let mut obs_n: usize = 0;
    let mut obs_sum = 0.0_f64;
    let mut obs_sq = 0.0_f64;
    for (j, &sid) in sample_idx.iter().enumerate() {
        let gv = full_row[sid];
        out_row[j] = gv;
        if gv >= 0.0_f32 {
            let g = gv as f64;
            obs_n += 1;
            obs_sum += g;
            obs_sq += g * g;
        }
    }

    let (mean_g, var_centered, var_standardized) = if obs_n > 0 {
        let mean = obs_sum / obs_n as f64;
        let p_sub = (0.5_f64 * mean).clamp(0.0, 1.0);
        let v_std = (2.0_f64 * p_sub * (1.0_f64 - p_sub)).max(0.0);
        let ss = (obs_sq - (obs_sum * obs_sum) / obs_n as f64).max(0.0);
        let v_center = (ss / n as f64).max(0.0);
        (mean as f32, v_center as f32, v_std as f32)
    } else {
        (default_mean_g, 0.0_f32, 0.0_f32)
    };

    let std_scale = if method == 2 {
        if var_standardized > eps {
            1.0_f32 / var_standardized.sqrt()
        } else {
            0.0_f32
        }
    } else {
        1.0_f32
    };
    for v in out_row.iter_mut() {
        let gv = if *v >= 0.0_f32 { *v } else { mean_g };
        *v = (gv - mean_g) * std_scale;
    }

    let row_sum = (mean_g as f64) * (n as f64);
    (var_centered as f64, row_sum)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn decode_subset_row_from_full_scratch_f64(
    row: &[u8],
    n_samples: usize,
    sample_idx: &[usize],
    flip: bool,
    default_mean_g: f64,
    method: usize,
    eps: f64,
    code4_lut: &[[u8; 4]; 256],
    full_row: &mut [f64],
    out_row: &mut [f64],
) -> (f64, f64) {
    debug_assert!(full_row.len() >= n_samples);
    debug_assert_eq!(out_row.len(), sample_idx.len());
    let n = sample_idx.len();

    let value_lut: [f64; 4] = if flip {
        [2.0_f64, -9.0_f64, 1.0_f64, 0.0_f64]
    } else {
        [0.0_f64, -9.0_f64, 1.0_f64, 2.0_f64]
    };
    decode_row_centered_full_lut_f64(
        row,
        n_samples,
        code4_lut,
        &value_lut,
        &mut full_row[..n_samples],
    );

    let mut obs_n: usize = 0;
    let mut obs_sum = 0.0_f64;
    let mut obs_sq = 0.0_f64;
    for (j, &sid) in sample_idx.iter().enumerate() {
        let gv = full_row[sid];
        out_row[j] = gv;
        if gv >= 0.0_f64 {
            obs_n += 1;
            obs_sum += gv;
            obs_sq += gv * gv;
        }
    }

    let (mean_g, var_centered, var_standardized) = if obs_n > 0 {
        let mean = obs_sum / obs_n as f64;
        let p_sub = (0.5_f64 * mean).clamp(0.0, 1.0);
        let v_std = (2.0_f64 * p_sub * (1.0_f64 - p_sub)).max(0.0);
        let ss = (obs_sq - (obs_sum * obs_sum) / obs_n as f64).max(0.0);
        let v_center = (ss / n as f64).max(0.0);
        (mean, v_center, v_std)
    } else {
        (default_mean_g, 0.0_f64, 0.0_f64)
    };

    let std_scale = if method == 2 {
        if var_standardized > eps {
            1.0_f64 / var_standardized.sqrt()
        } else {
            0.0_f64
        }
    } else {
        1.0_f64
    };
    for v in out_row.iter_mut() {
        let gv = if *v >= 0.0_f64 { *v } else { mean_g };
        *v = (gv - mean_g) * std_scale;
    }

    let row_sum = mean_g * (n as f64);
    (var_centered, row_sum)
}
