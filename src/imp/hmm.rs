//! Li-Stephens copying HMM with checkpointed forward/backward smoothing.

use super::types::{
    read_bit_chunk, CompositePanel, HMMConfig, HMMParameterStats, HMMWorkspace, PackedHaplotypes,
    PosteriorWindow,
};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const CHECKPOINT_STRIDE: usize = 64;
const EM_BACKWARD_BLOCK: usize = 128;
// Bound the per-caller checkpoint vector to 16 MiB of f32 state values.
// Smaller marker windows use dense checkpoints to avoid backward
// recomputation; large high-density windows automatically coarsen them.
const CHECKPOINT_VALUE_BUDGET: usize = 4 * 1024 * 1024;

fn simd_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("JANUSX_IMP_DISABLE_SIMD").is_some())
}

#[inline]
fn detail_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("JANUSX_IMP_DETAIL_PROFILE").is_some())
}

#[inline]
fn checkpoint_stride(n_markers: usize, n_states: usize) -> usize {
    if n_markers == 0 {
        return CHECKPOINT_STRIDE;
    }
    if n_states == 0 {
        return n_markers;
    }
    let max_checkpoints = (CHECKPOINT_VALUE_BUDGET / n_states).max(1);
    n_markers
        .div_ceil(max_checkpoints)
        .max(CHECKPOINT_STRIDE)
        .min(n_markers)
        .max(1)
}

#[derive(Debug, Clone)]
pub(crate) struct HMMRecursionDiagnostic {
    pub(crate) marker: usize,
    pub(crate) transition_probability: f32,
    pub(crate) emission_probabilities: Vec<f32>,
    pub(crate) forward_probabilities: Vec<f32>,
    pub(crate) backward_probabilities: Vec<f32>,
    pub(crate) posterior_normalization: f32,
}

#[derive(Debug)]
pub(crate) struct EmTransitionData {
    genetic_distance_cm: Vec<f32>,
    recombination_probability: Vec<f32>,
}

#[derive(Default)]
pub(crate) struct EmPairWorkspace {
    forward_checkpoints: [Vec<f32>; 2],
    forward_checkpoint_sums: [Vec<f32>; 2],
    forward: [Vec<f32>; 2],
    joint: [Vec<f32>; 2],
    state_probability: [Vec<f32>; 2],
    discord: [Vec<f32>; 2],
    mismatch_blocks: [Vec<u8>; 2],
    backward_blocks: [Vec<f32>; 2],
    backward: [Vec<f32>; 2],
    stats_by_marker: Vec<[[f32; 2]; 2]>,
}

#[inline]
fn em_forward_values(
    forward: &mut [f32],
    backward: &[f32],
    mismatches: &[u8],
    joint: &mut [f32],
    state_probability: &mut [f32],
    discord: &mut [f32],
    match_probability: f32,
    error_probability: f32,
    shift: f32,
    scale: f32,
    no_switch_scale: f32,
) {
    for state in 0..forward.len() {
        let mismatch = mismatches[state] != 0;
        let emission = if mismatch {
            error_probability
        } else {
            match_probability
        };
        joint[state] = backward[state] * emission * no_switch_scale * forward[state];
        let updated = emission * (scale * forward[state] + shift);
        forward[state] = updated;
        let probability = updated * backward[state];
        state_probability[state] = probability;
        discord[state] = if mismatch { probability } else { 0.0_f32 };
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn em_forward_values_pair(
    forward0: &mut [f32],
    forward1: &mut [f32],
    backward0: &[f32],
    backward1: &[f32],
    mismatches0: &[u8],
    mismatches1: &[u8],
    joint0: &mut [f32],
    joint1: &mut [f32],
    state_probability0: &mut [f32],
    state_probability1: &mut [f32],
    discord0: &mut [f32],
    discord1: &mut [f32],
    match_probability: f32,
    error_probability: f32,
    shift: f32,
    scale0: f32,
    scale1: f32,
    no_switch_scale0: f32,
    no_switch_scale1: f32,
    avx2: bool,
) {
    let state_count = forward0.len();
    debug_assert!(
        forward1.len() == state_count
            && backward0.len() == state_count
            && backward1.len() == state_count
            && mismatches0.len() == state_count
            && mismatches1.len() == state_count
            && joint0.len() == state_count
            && joint1.len() == state_count
            && state_probability0.len() == state_count
            && state_probability1.len() == state_count
            && discord0.len() == state_count
            && discord1.len() == state_count
    );
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx2;
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // SAFETY: AVX2 availability was checked above and all slices have
        // matching lengths.
        unsafe {
            em_forward_values_pair_avx2(
                forward0,
                forward1,
                backward0,
                backward1,
                mismatches0,
                mismatches1,
                joint0,
                joint1,
                state_probability0,
                state_probability1,
                discord0,
                discord1,
                match_probability,
                error_probability,
                shift,
                scale0,
                scale1,
                no_switch_scale0,
                no_switch_scale1,
            );
        }
        return;
    }
    em_forward_values(
        forward0,
        backward0,
        mismatches0,
        joint0,
        state_probability0,
        discord0,
        match_probability,
        error_probability,
        shift,
        scale0,
        no_switch_scale0,
    );
    em_forward_values(
        forward1,
        backward1,
        mismatches1,
        joint1,
        state_probability1,
        discord1,
        match_probability,
        error_probability,
        shift,
        scale1,
        no_switch_scale1,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn em_forward_values_pair_avx2(
    forward0: &mut [f32],
    forward1: &mut [f32],
    backward0: &[f32],
    backward1: &[f32],
    mismatches0: &[u8],
    mismatches1: &[u8],
    joint0: &mut [f32],
    joint1: &mut [f32],
    state_probability0: &mut [f32],
    state_probability1: &mut [f32],
    discord0: &mut [f32],
    discord1: &mut [f32],
    match_probability: f32,
    error_probability: f32,
    shift: f32,
    scale0: f32,
    scale1: f32,
    no_switch_scale0: f32,
    no_switch_scale1: f32,
) {
    use std::arch::x86_64::*;

    let matches = _mm256_set1_ps(match_probability);
    let errors = _mm256_set1_ps(error_probability);
    let shift = _mm256_set1_ps(shift);
    let scale0 = _mm256_set1_ps(scale0);
    let scale1 = _mm256_set1_ps(scale1);
    let no_switch0 = _mm256_set1_ps(no_switch_scale0);
    let no_switch1 = _mm256_set1_ps(no_switch_scale1);
    let zero_values = _mm256_setzero_ps();
    let zero_lanes = _mm256_setzero_si256();
    let vector_end = forward0.len() / 8 * 8;
    let mut state = 0usize;
    while state < vector_end {
        let mismatch_bits0 =
            u64::from_le_bytes(mismatches0[state..state + 8].try_into().unwrap()) as i64;
        let mismatch_bits1 =
            u64::from_le_bytes(mismatches1[state..state + 8].try_into().unwrap()) as i64;
        let mask0 = _mm256_cmpgt_epi32(
            _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(mismatch_bits0)),
            zero_lanes,
        );
        let mask1 = _mm256_cmpgt_epi32(
            _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(mismatch_bits1)),
            zero_lanes,
        );
        let emission0 = _mm256_blendv_ps(matches, errors, _mm256_castsi256_ps(mask0));
        let emission1 = _mm256_blendv_ps(matches, errors, _mm256_castsi256_ps(mask1));
        let old_forward0 = unsafe { _mm256_loadu_ps(forward0.as_ptr().add(state)) };
        let old_forward1 = unsafe { _mm256_loadu_ps(forward1.as_ptr().add(state)) };
        let backward_values0 = unsafe { _mm256_loadu_ps(backward0.as_ptr().add(state)) };
        let backward_values1 = unsafe { _mm256_loadu_ps(backward1.as_ptr().add(state)) };
        let updated0 = _mm256_mul_ps(
            emission0,
            _mm256_add_ps(_mm256_mul_ps(scale0, old_forward0), shift),
        );
        let updated1 = _mm256_mul_ps(
            emission1,
            _mm256_add_ps(_mm256_mul_ps(scale1, old_forward1), shift),
        );
        let joint_values0 = _mm256_mul_ps(
            _mm256_mul_ps(_mm256_mul_ps(backward_values0, emission0), no_switch0),
            old_forward0,
        );
        let joint_values1 = _mm256_mul_ps(
            _mm256_mul_ps(_mm256_mul_ps(backward_values1, emission1), no_switch1),
            old_forward1,
        );
        let probabilities0 = _mm256_mul_ps(updated0, backward_values0);
        let probabilities1 = _mm256_mul_ps(updated1, backward_values1);
        unsafe {
            _mm256_storeu_ps(forward0.as_mut_ptr().add(state), updated0);
            _mm256_storeu_ps(forward1.as_mut_ptr().add(state), updated1);
            _mm256_storeu_ps(joint0.as_mut_ptr().add(state), joint_values0);
            _mm256_storeu_ps(joint1.as_mut_ptr().add(state), joint_values1);
            _mm256_storeu_ps(state_probability0.as_mut_ptr().add(state), probabilities0);
            _mm256_storeu_ps(state_probability1.as_mut_ptr().add(state), probabilities1);
            _mm256_storeu_ps(
                discord0.as_mut_ptr().add(state),
                _mm256_blendv_ps(zero_values, probabilities0, _mm256_castsi256_ps(mask0)),
            );
            _mm256_storeu_ps(
                discord1.as_mut_ptr().add(state),
                _mm256_blendv_ps(zero_values, probabilities1, _mm256_castsi256_ps(mask1)),
            );
        }
        state += 8;
    }
    if state < forward0.len() {
        em_forward_values(
            &mut forward0[state..],
            &backward0[state..],
            &mismatches0[state..],
            &mut joint0[state..],
            &mut state_probability0[state..],
            &mut discord0[state..],
            match_probability,
            error_probability,
            _mm256_cvtss_f32(shift),
            _mm256_cvtss_f32(scale0),
            _mm256_cvtss_f32(no_switch0),
        );
        em_forward_values(
            &mut forward1[state..],
            &backward1[state..],
            &mismatches1[state..],
            &mut joint1[state..],
            &mut state_probability1[state..],
            &mut discord1[state..],
            match_probability,
            error_probability,
            _mm256_cvtss_f32(shift),
            _mm256_cvtss_f32(scale1),
            _mm256_cvtss_f32(no_switch1),
        );
    }
}

#[inline]
fn em_forward_update_values(
    forward: &mut [f32],
    mismatches: &[u8],
    match_probability: f32,
    error_probability: f32,
    shift: f32,
    scale: f32,
) {
    for (value, &is_mismatch) in forward.iter_mut().zip(mismatches) {
        let emission = if is_mismatch != 0 {
            error_probability
        } else {
            match_probability
        };
        *value = emission * (scale * *value + shift);
    }
}

#[inline]
fn em_forward_update_pair(
    forward0: &mut [f32],
    forward1: &mut [f32],
    mismatches0: &[u8],
    mismatches1: &[u8],
    match_probability: f32,
    error_probability: f32,
    shift: f32,
    scale0: f32,
    scale1: f32,
    avx2: bool,
) {
    debug_assert_eq!(forward0.len(), forward1.len());
    debug_assert_eq!(forward0.len(), mismatches0.len());
    debug_assert_eq!(forward0.len(), mismatches1.len());
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx2;
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // SAFETY: AVX2 availability was checked above and all slices have
        // matching lengths.
        unsafe {
            em_forward_update_pair_avx2(
                forward0,
                forward1,
                mismatches0,
                mismatches1,
                match_probability,
                error_probability,
                shift,
                scale0,
                scale1,
            );
        }
        return;
    }
    em_forward_update_values(
        forward0,
        mismatches0,
        match_probability,
        error_probability,
        shift,
        scale0,
    );
    em_forward_update_values(
        forward1,
        mismatches1,
        match_probability,
        error_probability,
        shift,
        scale1,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn em_forward_update_pair_avx2(
    forward0: &mut [f32],
    forward1: &mut [f32],
    mismatches0: &[u8],
    mismatches1: &[u8],
    match_probability: f32,
    error_probability: f32,
    shift: f32,
    scale0: f32,
    scale1: f32,
) {
    use std::arch::x86_64::*;

    let matches = _mm256_set1_ps(match_probability);
    let errors = _mm256_set1_ps(error_probability);
    let shift = _mm256_set1_ps(shift);
    let scale0 = _mm256_set1_ps(scale0);
    let scale1 = _mm256_set1_ps(scale1);
    let zero = _mm256_setzero_si256();
    let vector_end = forward0.len() / 8 * 8;
    let mut state = 0usize;
    while state < vector_end {
        let mismatch_bits0 =
            u64::from_le_bytes(mismatches0[state..state + 8].try_into().unwrap()) as i64;
        let mismatch_bits1 =
            u64::from_le_bytes(mismatches1[state..state + 8].try_into().unwrap()) as i64;
        let lanes0 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(mismatch_bits0));
        let lanes1 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(mismatch_bits1));
        let mask0 = _mm256_cmpgt_epi32(lanes0, zero);
        let mask1 = _mm256_cmpgt_epi32(lanes1, zero);
        let emissions0 = _mm256_blendv_ps(matches, errors, _mm256_castsi256_ps(mask0));
        let emissions1 = _mm256_blendv_ps(matches, errors, _mm256_castsi256_ps(mask1));
        let values0 = unsafe { _mm256_loadu_ps(forward0.as_ptr().add(state)) };
        let values1 = unsafe { _mm256_loadu_ps(forward1.as_ptr().add(state)) };
        let updated0 = _mm256_mul_ps(
            emissions0,
            _mm256_add_ps(_mm256_mul_ps(scale0, values0), shift),
        );
        let updated1 = _mm256_mul_ps(
            emissions1,
            _mm256_add_ps(_mm256_mul_ps(scale1, values1), shift),
        );
        unsafe {
            _mm256_storeu_ps(forward0.as_mut_ptr().add(state), updated0);
            _mm256_storeu_ps(forward1.as_mut_ptr().add(state), updated1);
        }
        state += 8;
    }
    if state < forward0.len() {
        em_forward_update_values(
            &mut forward0[state..],
            &mismatches0[state..],
            match_probability,
            error_probability,
            _mm256_cvtss_f32(shift),
            _mm256_cvtss_f32(scale0),
        );
        em_forward_update_values(
            &mut forward1[state..],
            &mismatches1[state..],
            match_probability,
            error_probability,
            _mm256_cvtss_f32(shift),
            _mm256_cvtss_f32(scale1),
        );
    }
}

#[inline]
fn em_backward_values(
    backward: &mut [f32],
    mismatches: &[u8],
    match_probability: f32,
    error_probability: f32,
) {
    for (value, &is_mismatch) in backward.iter_mut().zip(mismatches) {
        *value *= if is_mismatch != 0 {
            error_probability
        } else {
            match_probability
        };
    }
}

#[inline]
fn em_backward_pair(
    backward0: &mut [f32],
    backward1: &mut [f32],
    mismatches0: &[u8],
    mismatches1: &[u8],
    match_probability: f32,
    error_probability: f32,
    avx2: bool,
) {
    debug_assert_eq!(backward0.len(), backward1.len());
    debug_assert_eq!(backward0.len(), mismatches0.len());
    debug_assert_eq!(backward0.len(), mismatches1.len());
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx2;
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // SAFETY: AVX2 availability was checked above and all slices have
        // matching lengths.
        unsafe {
            em_backward_pair_avx2(
                backward0,
                backward1,
                mismatches0,
                mismatches1,
                match_probability,
                error_probability,
            );
        }
        return;
    }
    em_backward_values(backward0, mismatches0, match_probability, error_probability);
    em_backward_values(backward1, mismatches1, match_probability, error_probability);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn em_backward_pair_avx2(
    backward0: &mut [f32],
    backward1: &mut [f32],
    mismatches0: &[u8],
    mismatches1: &[u8],
    match_probability: f32,
    error_probability: f32,
) {
    use std::arch::x86_64::*;

    let matches = _mm256_set1_ps(match_probability);
    let errors = _mm256_set1_ps(error_probability);
    let zero = _mm256_setzero_si256();
    let vector_end = backward0.len() / 8 * 8;
    let mut state = 0usize;
    while state < vector_end {
        let mismatch_bits0 =
            u64::from_le_bytes(mismatches0[state..state + 8].try_into().unwrap()) as i64;
        let mismatch_bits1 =
            u64::from_le_bytes(mismatches1[state..state + 8].try_into().unwrap()) as i64;
        let lanes0 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(mismatch_bits0));
        let lanes1 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(mismatch_bits1));
        let mask0 = _mm256_cmpgt_epi32(lanes0, zero);
        let mask1 = _mm256_cmpgt_epi32(lanes1, zero);
        let emissions0 = _mm256_blendv_ps(matches, errors, _mm256_castsi256_ps(mask0));
        let emissions1 = _mm256_blendv_ps(matches, errors, _mm256_castsi256_ps(mask1));
        let values0 = unsafe { _mm256_loadu_ps(backward0.as_ptr().add(state)) };
        let values1 = unsafe { _mm256_loadu_ps(backward1.as_ptr().add(state)) };
        unsafe {
            _mm256_storeu_ps(
                backward0.as_mut_ptr().add(state),
                _mm256_mul_ps(values0, emissions0),
            );
            _mm256_storeu_ps(
                backward1.as_mut_ptr().add(state),
                _mm256_mul_ps(values1, emissions1),
            );
        }
        state += 8;
    }
    if state < backward0.len() {
        em_backward_values(
            &mut backward0[state..],
            &mismatches0[state..],
            match_probability,
            error_probability,
        );
        em_backward_values(
            &mut backward1[state..],
            &mismatches1[state..],
            match_probability,
            error_probability,
        );
    }
}

#[inline]
fn normalize_em_backward(backward: &mut [f32], switch: f32, sum: f32) {
    let shift = switch / backward.len() as f32;
    let scale = (1.0_f32 - switch) / sum;
    for value in backward {
        *value = scale * *value + shift;
    }
}

#[inline]
fn normalize_em_backward_pair(
    backward0: &mut [f32],
    backward1: &mut [f32],
    switch: f32,
    sums: [f32; 2],
    avx2: bool,
) {
    debug_assert_eq!(backward0.len(), backward1.len());
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx2;
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // SAFETY: AVX2 availability was checked above and both slices have
        // matching lengths.
        unsafe {
            normalize_em_backward_pair_avx2(backward0, backward1, switch, sums);
        }
        return;
    }
    normalize_em_backward(backward0, switch, sums[0]);
    normalize_em_backward(backward1, switch, sums[1]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_em_backward_pair_avx2(
    backward0: &mut [f32],
    backward1: &mut [f32],
    switch: f32,
    sums: [f32; 2],
) {
    use std::arch::x86_64::*;

    let shift = _mm256_set1_ps(switch / backward0.len() as f32);
    let scale0 = _mm256_set1_ps((1.0_f32 - switch) / sums[0]);
    let scale1 = _mm256_set1_ps((1.0_f32 - switch) / sums[1]);
    let vector_end = backward0.len() / 8 * 8;
    let mut state = 0usize;
    while state < vector_end {
        let values0 = unsafe { _mm256_loadu_ps(backward0.as_ptr().add(state)) };
        let values1 = unsafe { _mm256_loadu_ps(backward1.as_ptr().add(state)) };
        unsafe {
            _mm256_storeu_ps(
                backward0.as_mut_ptr().add(state),
                _mm256_add_ps(_mm256_mul_ps(scale0, values0), shift),
            );
            _mm256_storeu_ps(
                backward1.as_mut_ptr().add(state),
                _mm256_add_ps(_mm256_mul_ps(scale1, values1), shift),
            );
        }
        state += 8;
    }
    if state < backward0.len() {
        let tail_shift = switch / backward0.len() as f32;
        let tail_scale0 = (1.0_f32 - switch) / sums[0];
        let tail_scale1 = (1.0_f32 - switch) / sums[1];
        for (value0, value1) in backward0[state..].iter_mut().zip(&mut backward1[state..]) {
            *value0 = tail_scale0 * *value0 + tail_shift;
            *value1 = tail_scale1 * *value1 + tail_shift;
        }
    }
}

#[inline]
fn sum_em_lanes_2(first: &[f32], second: &[f32]) -> [f32; 2] {
    debug_assert_eq!(first.len(), second.len());
    let mut sums = [0.0_f32; 2];
    for state in 0..first.len() {
        // SAFETY: both slices have the same length, checked above in debug
        // builds and guaranteed by all callers in release builds.
        unsafe {
            sums[0] += *first.get_unchecked(state);
            sums[1] += *second.get_unchecked(state);
        }
    }
    sums
}

struct EmMismatchCache {
    marker_count: usize,
    state_count: usize,
    track_count: usize,
    words_per_track: usize,
    state_major: bool,
    packed_mismatches: Vec<u64>,
}

impl EmMismatchCache {
    fn build(
        observations_by_track: &[&[Option<u16>]],
        states: &CompositePanel,
    ) -> Result<Self, String> {
        let marker_count = states.marker_indices.len();
        let state_count = states.haplotypes.n_haplotypes;
        if observations_by_track.is_empty()
            || observations_by_track
                .iter()
                .any(|observations| observations.len() != marker_count)
        {
            return Err("EM HMM mismatch cache dimensions differ from observations".to_owned());
        }
        let bits_per_track = marker_count
            .checked_mul(state_count)
            .ok_or_else(|| "EM HMM mismatch cache dimensions overflow".to_owned())?;
        let words_per_track = bits_per_track.div_ceil(64);
        let word_count = words_per_track
            .checked_mul(observations_by_track.len())
            .ok_or_else(|| "EM HMM mismatch cache dimensions overflow".to_owned())?;
        let mut cache = Self {
            marker_count,
            state_count,
            track_count: observations_by_track.len(),
            words_per_track,
            state_major: states.haplotypes.is_state_major(),
            packed_mismatches: vec![0_u64; word_count],
        };
        if states.haplotypes.bits_per_allele == 1 {
            let mut used_packed_path = true;
            for (track, observations) in observations_by_track.iter().enumerate() {
                let start = track * words_per_track;
                let filled = if cache.state_major {
                    fill_biallelic_mismatch_track_state_major(
                        &mut cache.packed_mismatches[start..start + words_per_track],
                        observations,
                        states,
                    )
                } else {
                    fill_biallelic_mismatch_track(
                        &mut cache.packed_mismatches[start..start + words_per_track],
                        observations,
                        states,
                    )
                };
                if !filled {
                    used_packed_path = false;
                    break;
                }
            }
            if used_packed_path {
                return Ok(cache);
            }
            cache.packed_mismatches.fill(0);
        }
        for marker in 0..marker_count {
            let panel_marker = states.marker_indices[marker];
            for state in 0..state_count {
                let state_allele = states.haplotypes.allele(panel_marker, state);
                let flat_index = marker * state_count + state;
                for (track, observations) in observations_by_track.iter().enumerate() {
                    let mismatch = match observations[marker] {
                        Some(allele) => state_allele != Some(allele),
                        None => true,
                    };
                    if mismatch {
                        let index = track * words_per_track + (flat_index >> 6);
                        cache.packed_mismatches[index] |= 1_u64 << (flat_index & 63);
                    }
                }
            }
        }
        Ok(cache)
    }

    fn is_mismatch(&self, track: usize, marker: usize, state: usize) -> bool {
        let flat_index = if self.state_major {
            state * self.marker_count + marker
        } else {
            marker * self.state_count + state
        };
        let word = self.packed_mismatches[track * self.words_per_track + (flat_index >> 6)];
        (word >> (flat_index & 63)) & 1 != 0
    }

    fn fill_block(&self, track: usize, start: usize, end: usize, mismatches: &mut [u8]) {
        debug_assert_eq!(mismatches.len(), (end - start) * self.state_count);
        if self.state_major {
            let block_len = end - start;
            let track_start = track * self.words_per_track;
            for state in 0..self.state_count {
                let bit_start = state * self.marker_count + start;
                let mut local = 0usize;
                while local < block_len {
                    let count = (block_len - local).min(64);
                    let bits = read_bit_chunk(
                        &self.packed_mismatches[track_start..track_start + self.words_per_track],
                        bit_start + local,
                        count,
                    );
                    for offset in 0..count {
                        mismatches[(local + offset) * self.state_count + state] =
                            ((bits >> offset) & 1) as u8;
                    }
                    local += count;
                }
            }
            return;
        }
        for marker in start..end {
            let row_start = (marker - start) * self.state_count;
            for state in 0..self.state_count {
                mismatches[row_start + state] = u8::from(self.is_mismatch(track, marker, state));
            }
        }
    }

    fn fill_block_pair(
        &self,
        start: usize,
        end: usize,
        mismatches0: &mut [u8],
        mismatches1: &mut [u8],
    ) {
        let values = (end - start) * self.state_count;
        debug_assert_eq!(mismatches0.len(), values);
        debug_assert_eq!(mismatches1.len(), values);
        debug_assert!(self.track_count >= 2);
        if self.state_major {
            let block_len = end - start;
            let track0_start = 0usize;
            let track1_start = self.words_per_track;
            let state_word_count = self.words_per_track;
            for state in 0..self.state_count {
                let bit_start = state * self.marker_count + start;
                let mut local = 0usize;
                while local < block_len {
                    let count = (block_len - local).min(64);
                    let bits0 = read_bit_chunk(
                        &self.packed_mismatches[track0_start..track0_start + state_word_count],
                        bit_start + local,
                        count,
                    );
                    let bits1 = read_bit_chunk(
                        &self.packed_mismatches[track1_start..track1_start + state_word_count],
                        bit_start + local,
                        count,
                    );
                    for offset in 0..count {
                        let index = (local + offset) * self.state_count + state;
                        mismatches0[index] = ((bits0 >> offset) & 1) as u8;
                        mismatches1[index] = ((bits1 >> offset) & 1) as u8;
                    }
                    local += count;
                }
            }
            return;
        }
        for marker in start..end {
            let row_start = (marker - start) * self.state_count;
            for state in 0..self.state_count {
                mismatches0[row_start + state] = u8::from(self.is_mismatch(0, marker, state));
                mismatches1[row_start + state] = u8::from(self.is_mismatch(1, marker, state));
            }
        }
    }
}

/// Pack mismatch bits directly from complete biallelic allele/missingness
/// planes. Output uses the cache's marker-major flat layout, whose marker
/// boundaries can shift when the state count is not a multiple of 64.
fn fill_biallelic_mismatch_track(
    output: &mut [u64],
    observations: &[Option<u16>],
    states: &CompositePanel,
) -> bool {
    let marker_count = states.marker_indices.len();
    let state_count = states.haplotypes.n_haplotypes;
    if states.haplotypes.bits_per_allele != 1
        || observations.len() != marker_count
        || marker_count
            .checked_mul(state_count)
            .map(|bits| bits.div_ceil(64))
            != Some(output.len())
    {
        return false;
    }
    for (marker, (&panel_marker, &observation)) in
        states.marker_indices.iter().zip(observations).enumerate()
    {
        let row_start = marker * state_count;
        for first_state in (0..state_count).step_by(64) {
            let Some((allele_word, missing_word)) = states
                .haplotypes
                .biallelic_marker_word(panel_marker, first_state)
            else {
                return false;
            };
            let valid_states = (state_count - first_state).min(64);
            let valid_mask = if valid_states == 64 {
                u64::MAX
            } else {
                (1u64 << valid_states) - 1
            };
            let mismatch_word = match observation {
                Some(0) => allele_word | missing_word,
                Some(1) => !allele_word | missing_word,
                Some(_) | None => valid_mask,
            } & valid_mask;
            write_packed_mismatch_bits(
                output,
                row_start + first_state,
                valid_states,
                mismatch_word,
            );
        }
    }
    true
}

fn fill_biallelic_mismatch_track_state_major(
    output: &mut [u64],
    observations: &[Option<u16>],
    states: &CompositePanel,
) -> bool {
    let marker_count = states.marker_indices.len();
    let state_count = states.haplotypes.n_haplotypes;
    if !states.haplotypes.is_state_major()
        || states.haplotypes.bits_per_allele != 1
        || observations.len() != marker_count
        || marker_count
            .checked_mul(state_count)
            .map(|bits| bits.div_ceil(64))
            != Some(output.len())
    {
        return false;
    }

    let word_count = marker_count.div_ceil(64);
    let mut observed = vec![0u64; word_count];
    let mut alternate = vec![0u64; word_count];
    for (marker, observation) in observations.iter().copied().enumerate() {
        let word = marker >> 6;
        let mask = 1u64 << (marker & 63);
        match observation {
            Some(0) => observed[word] |= mask,
            Some(1) => {
                observed[word] |= mask;
                alternate[word] |= mask;
            }
            Some(_) => return false,
            None => {}
        }
    }

    for state in 0..state_count {
        let Some((state_start, allele_bits, missing_bits)) =
            states.haplotypes.state_major_biallelic_words(state)
        else {
            return false;
        };
        let mut marker = 0usize;
        while marker < marker_count {
            let count = (marker_count - marker).min(64);
            let word = marker >> 6;
            let mask = if count == 64 {
                u64::MAX
            } else {
                (1u64 << count) - 1
            };
            let source_start = state_start + marker;
            let donor_alternate = read_bit_chunk(allele_bits, source_start, count);
            let donor_missing = read_bit_chunk(missing_bits, source_start, count);
            let compared = observed[word] & !donor_missing & mask;
            let mismatch = compared & (alternate[word] ^ donor_alternate);
            write_packed_mismatch_bits(output, state * marker_count + marker, count, mismatch);
            marker += count;
        }
    }
    true
}

#[inline]
fn write_packed_mismatch_bits(output: &mut [u64], bit_start: usize, bit_count: usize, bits: u64) {
    let word_index = bit_start >> 6;
    let shift = bit_start & 63;
    let low_count = (64 - shift).min(bit_count);
    let low_mask = if low_count == 64 {
        u64::MAX
    } else {
        (1u64 << low_count) - 1
    };
    output[word_index] |= (bits & low_mask) << shift;
    let remaining = bit_count - low_count;
    if remaining > 0 {
        let high_mask = if remaining == 64 {
            u64::MAX
        } else {
            (1u64 << remaining) - 1
        };
        output[word_index + 1] |= (bits >> low_count) & high_mask;
    }
}

pub(crate) fn prepare_em_transition_data(
    cm_positions: &[f64],
    config: &HMMConfig,
) -> Result<EmTransitionData, String> {
    if cm_positions.is_empty()
        || cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("EM HMM genetic positions must be finite and nondecreasing".to_owned());
    }
    let mut genetic_distance_cm = vec![0.0_f32; cm_positions.len()];
    let mut recombination_probability = vec![0.0_f32; cm_positions.len()];
    for marker in 1..cm_positions.len() {
        let distance = (cm_positions[marker] - cm_positions[marker - 1]) as f32;
        genetic_distance_cm[marker] = distance;
        // For every non-singleton EM panel, Beagle's transition probability
        // depends on the map and current intensity, not the panel state count.
        recombination_probability[marker] = transition_probability(
            f64::from(distance),
            2,
            config.haplotype_count,
            config.recombination_intensity,
        );
    }
    Ok(EmTransitionData {
        genetic_distance_cm,
        recombination_probability,
    })
}

pub(crate) fn forward_backward(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    scratch: &mut HMMWorkspace,
) -> Result<PosteriorWindow, String> {
    forward_backward_impl(
        observations,
        states,
        cm_positions,
        config,
        scratch,
        None,
        true,
        false,
        true,
        None,
        |_, _| Ok(()),
        None,
        |_| Ok(()),
    )
}

pub(crate) fn forward_backward_for_em(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    _scratch: &mut HMMWorkspace,
) -> Result<HMMParameterStats, String> {
    let transitions = prepare_em_transition_data(cm_positions, config)?;
    forward_backward_for_em_with_transition_data(
        observations,
        states,
        cm_positions,
        config,
        &transitions,
    )
}

pub(crate) fn forward_backward_for_em_with_transition_data(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    transitions: &EmTransitionData,
) -> Result<HMMParameterStats, String> {
    beagle_em_parameter_stats(observations, states, cm_positions, config, transitions)
}

pub(crate) fn forward_backward_for_em_pair_with_transition_data(
    observations0: &[Option<u16>],
    observations1: &[Option<u16>],
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    transitions: &EmTransitionData,
) -> Result<(HMMParameterStats, HMMParameterStats), String> {
    let mut workspace = EmPairWorkspace::default();
    forward_backward_for_em_pair_with_workspace(
        observations0,
        observations1,
        states,
        cm_positions,
        config,
        transitions,
        &mut workspace,
    )
}

pub(crate) fn forward_backward_for_em_single_observation_with_transition_data(
    observations: &[Option<u16>],
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    transitions: &EmTransitionData,
) -> Result<HMMParameterStats, String> {
    let state_count = validate_em_input(
        observations.len(),
        1,
        states,
        cm_positions,
        config,
        transitions,
    )?;
    if state_count <= 1 {
        return Ok(HMMParameterStats::default());
    }
    let mismatch_cache = EmMismatchCache::build(&[observations], states)?;
    beagle_em_parameter_stats_from_cache(
        observations.len(),
        states,
        config,
        transitions,
        &mismatch_cache,
        0,
    )
}

pub(crate) fn forward_backward_for_em_pair_with_workspace(
    observations0: &[Option<u16>],
    observations1: &[Option<u16>],
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    transitions: &EmTransitionData,
    workspace: &mut EmPairWorkspace,
) -> Result<(HMMParameterStats, HMMParameterStats), String> {
    if observations0.len() != observations1.len() {
        return Err("EM HMM paired observation lengths differ".to_owned());
    }
    let state_count = validate_em_input(
        observations0.len(),
        1,
        states,
        cm_positions,
        config,
        transitions,
    )?;
    if state_count <= 1 {
        return Ok((HMMParameterStats::default(), HMMParameterStats::default()));
    }
    let profile = detail_profiling_enabled();
    let cache_started = profile.then(Instant::now);
    let mismatch_cache = EmMismatchCache::build(&[observations0, observations1], states)?;
    let cache_elapsed = cache_started.map_or(Duration::ZERO, |started| started.elapsed());
    if !states.haplotypes.is_state_major() {
        let stats0 = beagle_em_parameter_stats_from_cache(
            observations0.len(),
            states,
            config,
            transitions,
            &mismatch_cache,
            0,
        )?;
        let stats1 = beagle_em_parameter_stats_from_cache(
            observations1.len(),
            states,
            config,
            transitions,
            &mismatch_cache,
            1,
        )?;
        return Ok((stats0, stats1));
    }
    let hmm_started = profile.then(Instant::now);
    let stats = beagle_em_parameter_stats_pair_from_cache(
        observations0.len(),
        states,
        config,
        transitions,
        &mismatch_cache,
        workspace,
    )?;
    if let Some(started) = hmm_started {
        let hmm_elapsed = started.elapsed();
        eprintln!(
            "JANUSX_IMP_EM_PAIR_PROFILE cache_s={:.6} hmm_s={:.6} total_s={:.6}",
            cache_elapsed.as_secs_f64(),
            hmm_elapsed.as_secs_f64(),
            cache_elapsed.as_secs_f64() + hmm_elapsed.as_secs_f64(),
        );
    }
    Ok(stats)
}

/// Computes Beagle's EM tallies with its single-haplotype `HmmParamData`
/// arithmetic. The state vectors are checkpointed so the estimator keeps
/// bounded memory while preserving Beagle's float update and accumulation
/// order at every marker.
fn beagle_em_parameter_stats(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    transitions: &EmTransitionData,
) -> Result<HMMParameterStats, String> {
    let marker_count = observations.n_markers;
    let state_count = validate_em_input(
        marker_count,
        observations.n_haplotypes,
        states,
        cm_positions,
        config,
        transitions,
    )?;
    if state_count <= 1 {
        // Beagle's HmmParamData skips parameter tallies for a single state.
        return Ok(HMMParameterStats::default());
    }
    let observation_alleles = (0..marker_count)
        .map(|marker| observations.allele(marker, 0))
        .collect::<Vec<_>>();
    let mismatch_cache = EmMismatchCache::build(&[&observation_alleles], states)?;
    beagle_em_parameter_stats_from_cache(
        marker_count,
        states,
        config,
        transitions,
        &mismatch_cache,
        0,
    )
}

fn validate_em_input(
    marker_count: usize,
    observation_haplotype_count: usize,
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    transitions: &EmTransitionData,
) -> Result<usize, String> {
    let state_count = states.haplotypes.n_haplotypes;
    if marker_count == 0 || state_count == 0 || observation_haplotype_count != 1 {
        return Err("EM HMM expects one observation haplotype and positive dimensions".to_owned());
    }
    if states.marker_indices.len() != marker_count
        || states
            .marker_indices
            .iter()
            .any(|&marker| marker >= states.haplotypes.n_markers)
        || cm_positions.len() != marker_count
    {
        return Err(
            "EM HMM marker mapping differs between observations, states, and map".to_owned(),
        );
    }
    if config.ne == 0
        || !config.recombination_intensity.is_finite()
        || config.recombination_intensity <= 0.0
        || config.haplotype_count == 0
        || !config.err.is_finite()
        || config.err < 0.0
        || config.err >= 0.5
    {
        return Err("EM HMM parameters are outside their valid range".to_owned());
    }
    if cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("EM HMM genetic positions must be finite and nondecreasing".to_owned());
    }
    if transitions.genetic_distance_cm.len() != marker_count
        || transitions.recombination_probability.len() != marker_count
    {
        return Err("EM HMM transition cache differs from the marker map".to_owned());
    }
    Ok(state_count)
}

/// Uses one shared, packed mismatch cache for each observation track. When
/// both sample haplotypes are analyzed together, each donor allele is decoded
/// only once for the pair instead of once per HMM pass and marker.
fn beagle_em_parameter_stats_from_cache(
    marker_count: usize,
    states: &CompositePanel,
    config: &HMMConfig,
    transitions: &EmTransitionData,
    mismatch_cache: &EmMismatchCache,
    track: usize,
) -> Result<HMMParameterStats, String> {
    let state_count = states.haplotypes.n_haplotypes;
    if mismatch_cache.marker_count != marker_count
        || mismatch_cache.state_count != state_count
        || track >= mismatch_cache.track_count
    {
        return Err("EM HMM mismatch cache dimensions differ from observations".to_owned());
    }

    // fast-beagle/HmmParamData uses 128-marker backward blocks. This changes
    // only checkpoint boundaries; each lane keeps the same state-order sums.
    let checkpoint_stride = EM_BACKWARD_BLOCK.min(marker_count);
    let checkpoint_count = marker_count.div_ceil(checkpoint_stride);
    let checkpoint_values = checkpoint_count
        .checked_mul(state_count)
        .ok_or_else(|| "EM HMM checkpoint dimensions overflow".to_owned())?;
    let error_probability = config.err as f32;
    let match_probability = 1.0_f32 - error_probability;
    let emission_probability = |marker: usize, state: usize| {
        if mismatch_cache.is_mismatch(track, marker, state) {
            error_probability
        } else {
            match_probability
        }
    };

    // Save the Java forward vector and its sum after the final marker of each
    // checkpoint block. Java keeps the vector unnormalized and divides by the
    // previous sum inside each `fwdUpdate`.
    let mut forward_checkpoints = vec![0.0_f32; checkpoint_values];
    let mut forward_checkpoint_sums = vec![0.0_f32; checkpoint_count];
    let mut forward = vec![1.0_f32 / state_count as f32; state_count];
    let mut previous_sum = 1.0_f32;
    for marker in 0..marker_count {
        let p_switch = transitions.recombination_probability[marker];
        let shift = p_switch / state_count as f32;
        let scale = (1.0_f32 - p_switch) / previous_sum;
        let mut forward_sum = 0.0_f32;
        for (state, value) in forward.iter_mut().enumerate() {
            *value = emission_probability(marker, state) * (scale * *value + shift);
            forward_sum += *value;
        }
        previous_sum = forward_sum;
        if (marker + 1) % checkpoint_stride == 0 || marker + 1 == marker_count {
            let checkpoint = marker / checkpoint_stride;
            let offset = checkpoint * state_count;
            forward_checkpoints[offset..offset + state_count].copy_from_slice(&forward);
            forward_checkpoint_sums[checkpoint] = forward_sum;
        }
    }

    let block_capacity = checkpoint_stride.min(marker_count);
    let mismatch_block_capacity = block_capacity
        .checked_mul(state_count)
        .ok_or_else(|| "EM HMM mismatch block dimensions overflow".to_owned())?;
    let mut mismatch_block = vec![0u8; mismatch_block_capacity];
    let mut backward_block = vec![0.0_f32; block_capacity * state_count];
    let mut backward = vec![1.0_f32; state_count];
    // Blocks are traversed backward to carry the smoothing vector forward in
    // the reverse recursion. Keep the per-marker f32 contributions so the
    // final f64 reductions still follow Beagle's ascending marker order.
    let mut stats_by_marker = vec![[0.0_f32; 2]; marker_count];
    let h_factor = state_count as f32 / (state_count as f32 - 1.0_f32);

    for block in (0..checkpoint_count).rev() {
        let start = block * checkpoint_stride;
        let end = ((block + 1) * checkpoint_stride).min(marker_count);
        let block_len = end - start;
        let mismatch_values = block_len * state_count;
        mismatch_cache.fill_block(track, start, end, &mut mismatch_block[..mismatch_values]);
        let block_mismatches = &mismatch_block[..mismatch_values];

        // Reconstruct this block's backward vectors from the carried value at
        // its right edge, then reuse the mismatch flags for the posterior
        // tally instead of decoding donor alleles a third time.
        let last_local = block_len - 1;
        backward_block[last_local * state_count..(last_local + 1) * state_count]
            .copy_from_slice(&backward);
        for marker in (start + 1..end).rev() {
            let p_switch = transitions.recombination_probability[marker];
            let mismatch_start = (marker - start) * state_count;
            let mut emission_sum = 0.0_f32;
            for (state, value) in backward.iter_mut().enumerate() {
                *value *= if block_mismatches[mismatch_start + state] != 0 {
                    error_probability
                } else {
                    match_probability
                };
                emission_sum += *value;
            }
            let shift = p_switch / state_count as f32;
            let scale = (1.0_f32 - p_switch) / emission_sum;
            for value in &mut backward {
                *value = scale * *value + shift;
            }
            let local = marker - start - 1;
            backward_block[local * state_count..(local + 1) * state_count]
                .copy_from_slice(&backward);
        }

        // Match HmmParamData's forward marker traversal and float reductions.
        if block == 0 {
            forward.fill(1.0_f32 / state_count as f32);
            previous_sum = 1.0_f32;
        } else {
            let offset = (block - 1) * state_count;
            forward.copy_from_slice(&forward_checkpoints[offset..offset + state_count]);
            previous_sum = forward_checkpoint_sums[block - 1];
        }
        for marker in start..end {
            let local = marker - start;
            let backward_start = local * state_count;
            let p_switch = transitions.recombination_probability[marker];
            let shift = p_switch / state_count as f32;
            let scale = (1.0_f32 - p_switch) / previous_sum;
            let no_switch_scale = ((1.0_f32 - p_switch) + shift) / previous_sum;
            let mut joint_state_sum = 0.0_f32;
            let mut state_sum = 0.0_f32;
            let mut mismatch_sum = 0.0_f32;
            let mut forward_sum = 0.0_f32;
            let mismatch_start = local * state_count;
            for state in 0..state_count {
                let is_mismatch = block_mismatches[mismatch_start + state] != 0;
                let emission = if is_mismatch {
                    error_probability
                } else {
                    match_probability
                };
                let backward_value = backward_block[backward_start + state];
                joint_state_sum += backward_value * emission * no_switch_scale * forward[state];
                forward[state] = emission * (scale * forward[state] + shift);
                forward_sum += forward[state];
                let state_probability = forward[state] * backward_value;
                state_sum += state_probability;
                if is_mismatch {
                    mismatch_sum += state_probability;
                }
            }
            let mismatch_probability = mismatch_sum / state_sum;
            let switch_probability = h_factor * (1.0_f32 - joint_state_sum / state_sum);
            stats_by_marker[marker] = [mismatch_probability, switch_probability];
            previous_sum = forward_sum;
        }

        // Continue the backward recurrence across the block boundary. At
        // this point `backward` is beta at `start`; projecting marker `start`
        // yields beta at `start - 1`, the next block's right edge.
        if start > 0 {
            let p_switch = transitions.recombination_probability[start];
            let mut emission_sum = 0.0_f32;
            for (state, value) in backward.iter_mut().enumerate() {
                *value *= if block_mismatches[state] != 0 {
                    error_probability
                } else {
                    match_probability
                };
                emission_sum += *value;
            }
            let shift = p_switch / state_count as f32;
            let scale = (1.0_f32 - p_switch) / emission_sum;
            for value in &mut backward {
                *value = scale * *value + shift;
            }
        }
    }

    let mut stats = HMMParameterStats::default();
    for (marker, [mismatch_probability, switch_probability]) in
        stats_by_marker.into_iter().enumerate()
    {
        stats.mismatch_marker_count += 1;
        stats.mismatch_probability_sum += f64::from(mismatch_probability);
        if switch_probability > 0.0 {
            stats.switch_distance_cm_sum += f64::from(transitions.genetic_distance_cm[marker]);
            stats.switch_probability_sum += f64::from(switch_probability);
        }
    }

    Ok(stats)
}

fn beagle_em_parameter_stats_pair_from_cache(
    marker_count: usize,
    states: &CompositePanel,
    config: &HMMConfig,
    transitions: &EmTransitionData,
    mismatch_cache: &EmMismatchCache,
    workspace: &mut EmPairWorkspace,
) -> Result<(HMMParameterStats, HMMParameterStats), String> {
    let state_count = states.haplotypes.n_haplotypes;
    if mismatch_cache.marker_count != marker_count
        || mismatch_cache.state_count != state_count
        || mismatch_cache.track_count < 2
    {
        return Err("paired EM HMM mismatch cache dimensions differ from observations".to_owned());
    }
    #[cfg(target_arch = "x86_64")]
    let avx2 = !simd_disabled() && std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let avx2 = false;

    // fast-beagle/HmmParamData uses 128-marker blocks for both haplotypes.
    // Keep the paired implementation on the same boundaries; changing a
    // block boundary only changes how many rows are recomputed and retains
    // each lane's state-order floating-point accumulation.
    let checkpoint_stride = EM_BACKWARD_BLOCK.min(marker_count);
    let checkpoint_count = marker_count.div_ceil(checkpoint_stride);
    let checkpoint_values = checkpoint_count
        .checked_mul(state_count)
        .ok_or_else(|| "paired EM HMM checkpoint dimensions overflow".to_owned())?;
    let error_probability = config.err as f32;
    let match_probability = 1.0_f32 - error_probability;
    let block_capacity = checkpoint_stride.min(marker_count);
    let mismatch_block_capacity = block_capacity
        .checked_mul(state_count)
        .ok_or_else(|| "paired EM HMM mismatch block dimensions overflow".to_owned())?;
    let mismatch_blocks = &mut workspace.mismatch_blocks;
    for track in 0..2 {
        workspace.forward_checkpoints[track].resize(checkpoint_values, 0.0);
        workspace.forward_checkpoint_sums[track].resize(checkpoint_count, 0.0);
        workspace.forward[track].resize(state_count, 0.0);
        workspace.forward[track].fill(1.0_f32 / state_count as f32);
        mismatch_blocks[track].resize(mismatch_block_capacity, 0);
    }
    let mut previous_sum = [1.0_f32; 2];
    let profile = detail_profiling_enabled();
    let forward_started = profile.then(Instant::now);

    for block in 0..checkpoint_count {
        let start = block * checkpoint_stride;
        let end = ((block + 1) * checkpoint_stride).min(marker_count);
        let mismatch_values = (end - start) * state_count;
        let (mismatch_blocks0, mismatch_blocks1) = mismatch_blocks.split_at_mut(1);
        mismatch_cache.fill_block_pair(
            start,
            end,
            &mut mismatch_blocks0[0][..mismatch_values],
            &mut mismatch_blocks1[0][..mismatch_values],
        );
        for marker in start..end {
            let local = marker - start;
            let p_switch = transitions.recombination_probability[marker];
            let mismatch_start = local * state_count;
            let mismatches0 = &mismatch_blocks[0][mismatch_start..mismatch_start + state_count];
            let mismatches1 = &mismatch_blocks[1][mismatch_start..mismatch_start + state_count];
            let shift = p_switch / state_count as f32;
            let scale0 = (1.0_f32 - p_switch) / previous_sum[0];
            let scale1 = (1.0_f32 - p_switch) / previous_sum[1];
            let (forward0, forward1) = workspace.forward.split_at_mut(1);
            em_forward_update_pair(
                &mut forward0[0],
                &mut forward1[0],
                mismatches0,
                mismatches1,
                match_probability,
                error_probability,
                shift,
                scale0,
                scale1,
                avx2,
            );
            previous_sum = sum_em_lanes_2(&workspace.forward[0], &workspace.forward[1]);
            if (marker + 1) % checkpoint_stride == 0 || marker + 1 == marker_count {
                let checkpoint = marker / checkpoint_stride;
                let offset = checkpoint * state_count;
                for track in 0..2 {
                    workspace.forward_checkpoints[track][offset..offset + state_count]
                        .copy_from_slice(&workspace.forward[track]);
                    workspace.forward_checkpoint_sums[track][checkpoint] = previous_sum[track];
                }
            }
        }
    }
    let forward_elapsed = forward_started.map_or(Duration::ZERO, |started| started.elapsed());

    let backward_block_capacity = block_capacity
        .checked_mul(state_count)
        .ok_or_else(|| "paired EM HMM backward block size overflow".to_owned())?;
    let forward_checkpoints = &mut workspace.forward_checkpoints;
    let forward_checkpoint_sums = &mut workspace.forward_checkpoint_sums;
    let forward = &mut workspace.forward;
    let joint = &mut workspace.joint;
    let state_probability = &mut workspace.state_probability;
    let discord = &mut workspace.discord;
    let backward_blocks = &mut workspace.backward_blocks;
    let backward = &mut workspace.backward;
    let stats_by_marker = &mut workspace.stats_by_marker;
    for track in 0..2 {
        backward_blocks[track].resize(backward_block_capacity, 0.0);
        backward[track].resize(state_count, 0.0);
        backward[track].fill(1.0_f32);
        joint[track].resize(state_count, 0.0);
        state_probability[track].resize(state_count, 0.0);
        discord[track].resize(state_count, 0.0);
    }
    stats_by_marker.resize(marker_count, [[0.0_f32; 2]; 2]);
    stats_by_marker.fill([[0.0_f32; 2]; 2]);
    let h_factor = state_count as f32 / (state_count as f32 - 1.0_f32);
    let moments_started = profile.then(Instant::now);
    let mut backward_elapsed = Duration::ZERO;
    let mut stats_elapsed = Duration::ZERO;

    for block in (0..checkpoint_count).rev() {
        let backward_started = profile.then(Instant::now);
        let start = block * checkpoint_stride;
        let end = ((block + 1) * checkpoint_stride).min(marker_count);
        let block_len = end - start;
        let mismatch_values = block_len * state_count;
        let (mismatch_blocks0, mismatch_blocks1) = mismatch_blocks.split_at_mut(1);
        mismatch_cache.fill_block_pair(
            start,
            end,
            &mut mismatch_blocks0[0][..mismatch_values],
            &mut mismatch_blocks1[0][..mismatch_values],
        );

        let last_local = block_len - 1;
        for track in 0..2 {
            backward_blocks[track][last_local * state_count..(last_local + 1) * state_count]
                .copy_from_slice(&backward[track]);
        }
        for marker in (start + 1..end).rev() {
            let p_switch = transitions.recombination_probability[marker];
            let mismatch_start = (marker - start) * state_count;
            let mismatches0 = &mismatch_blocks[0][mismatch_start..mismatch_start + state_count];
            let mismatches1 = &mismatch_blocks[1][mismatch_start..mismatch_start + state_count];
            let (backward0, backward1) = backward.split_at_mut(1);
            let backward0 = &mut backward0[0];
            let backward1 = &mut backward1[0];
            em_backward_pair(
                backward0,
                backward1,
                mismatches0,
                mismatches1,
                match_probability,
                error_probability,
                avx2,
            );
            let emission_sums = sum_em_lanes_2(backward0, backward1);
            normalize_em_backward_pair(backward0, backward1, p_switch, emission_sums, avx2);
            let local = marker - start - 1;
            backward_blocks[0][local * state_count..(local + 1) * state_count]
                .copy_from_slice(backward0);
            backward_blocks[1][local * state_count..(local + 1) * state_count]
                .copy_from_slice(backward1);
        }
        if let Some(started) = backward_started {
            backward_elapsed += started.elapsed();
        }

        let stats_started = profile.then(Instant::now);
        if block == 0 {
            for track in 0..2 {
                forward[track].fill(1.0_f32 / state_count as f32);
                previous_sum[track] = 1.0_f32;
            }
        } else {
            let offset = (block - 1) * state_count;
            for track in 0..2 {
                forward[track]
                    .copy_from_slice(&forward_checkpoints[track][offset..offset + state_count]);
                previous_sum[track] = forward_checkpoint_sums[track][block - 1];
            }
        }
        for marker in start..end {
            let local = marker - start;
            let backward_start = local * state_count;
            let mismatch_start = local * state_count;
            let p_switch = transitions.recombination_probability[marker];
            let shift = p_switch / state_count as f32;
            let no_switch_scales = [
                ((1.0_f32 - p_switch) + shift) / previous_sum[0],
                ((1.0_f32 - p_switch) + shift) / previous_sum[1],
            ];
            let scales = [
                (1.0_f32 - p_switch) / previous_sum[0],
                (1.0_f32 - p_switch) / previous_sum[1],
            ];
            let (forward0, forward1) = forward.split_at_mut(1);
            let (joint0, joint1) = joint.split_at_mut(1);
            let (state0, state1) = state_probability.split_at_mut(1);
            let (discord0, discord1) = discord.split_at_mut(1);
            let backward0 = &backward_blocks[0][backward_start..backward_start + state_count];
            let backward1 = &backward_blocks[1][backward_start..backward_start + state_count];
            let mismatch0 = &mismatch_blocks[0][mismatch_start..mismatch_start + state_count];
            let mismatch1 = &mismatch_blocks[1][mismatch_start..mismatch_start + state_count];
            em_forward_values_pair(
                &mut forward0[0][..state_count],
                &mut forward1[0][..state_count],
                backward0,
                backward1,
                mismatch0,
                mismatch1,
                &mut joint0[0][..state_count],
                &mut joint1[0][..state_count],
                &mut state0[0][..state_count],
                &mut state1[0][..state_count],
                &mut discord0[0][..state_count],
                &mut discord1[0][..state_count],
                match_probability,
                error_probability,
                shift,
                scales[0],
                scales[1],
                no_switch_scales[0],
                no_switch_scales[1],
                avx2,
            );
            let mut forward_sums = [0.0_f32; 2];
            let mut joint_state_sums = [0.0_f32; 2];
            let mut state_sums = [0.0_f32; 2];
            let mut mismatch_sums = [0.0_f32; 2];
            for state in 0..state_count {
                forward_sums[0] += forward0[0][state];
                joint_state_sums[0] += joint0[0][state];
                state_sums[0] += state0[0][state];
                mismatch_sums[0] += discord0[0][state];
                forward_sums[1] += forward1[0][state];
                joint_state_sums[1] += joint1[0][state];
                state_sums[1] += state1[0][state];
                mismatch_sums[1] += discord1[0][state];
            }
            for track in 0..2 {
                stats_by_marker[marker][track] = [
                    mismatch_sums[track] / state_sums[track],
                    h_factor * (1.0_f32 - joint_state_sums[track] / state_sums[track]),
                ];
                previous_sum[track] = forward_sums[track];
            }
        }
        if let Some(started) = stats_started {
            stats_elapsed += started.elapsed();
        }

        if start > 0 {
            let p_switch = transitions.recombination_probability[start];
            let (backward0, backward1) = backward.split_at_mut(1);
            em_backward_pair(
                &mut backward0[0],
                &mut backward1[0],
                &mismatch_blocks[0][..state_count],
                &mismatch_blocks[1][..state_count],
                match_probability,
                error_probability,
                avx2,
            );
            let sums = sum_em_lanes_2(&backward[0], &backward[1]);
            let (backward0, backward1) = backward.split_at_mut(1);
            normalize_em_backward_pair(&mut backward0[0], &mut backward1[0], p_switch, sums, avx2);
        }
    }
    let moments_elapsed = moments_started.map_or(Duration::ZERO, |started| started.elapsed());

    let reduce_started = profile.then(Instant::now);
    let mut stats = [HMMParameterStats::default(), HMMParameterStats::default()];
    for track in 0..2 {
        for (marker, marker_stats) in stats_by_marker.iter().enumerate() {
            let [mismatch_probability, switch_probability] = marker_stats[track];
            stats[track].mismatch_marker_count += 1;
            stats[track].mismatch_probability_sum += f64::from(mismatch_probability);
            if switch_probability > 0.0 {
                stats[track].switch_distance_cm_sum +=
                    f64::from(transitions.genetic_distance_cm[marker]);
                stats[track].switch_probability_sum += f64::from(switch_probability);
            }
        }
    }
    if let Some(started) = reduce_started {
        let reduce_elapsed = started.elapsed();
        eprintln!(
            "JANUSX_IMP_EM_HMM_PROFILE forward_s={:.6} backward_s={:.6} stats_s={:.6} moments_s={:.6} reduce_s={:.6} total_s={:.6}",
            forward_elapsed.as_secs_f64(),
            backward_elapsed.as_secs_f64(),
            stats_elapsed.as_secs_f64(),
            moments_elapsed.as_secs_f64(),
            reduce_elapsed.as_secs_f64(),
            forward_elapsed.as_secs_f64()
                + moments_elapsed.as_secs_f64()
                + reduce_elapsed.as_secs_f64(),
        );
    }
    Ok((stats[0], stats[1]))
}

pub(crate) fn forward_backward_with_state_callback<F>(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    scratch: &mut HMMWorkspace,
    state_callback: F,
) -> Result<PosteriorWindow, String>
where
    F: FnMut(usize, &[f32]) -> Result<(), String>,
{
    forward_backward_impl(
        observations,
        states,
        cm_positions,
        config,
        scratch,
        None,
        false,
        true,
        false,
        None,
        state_callback,
        None,
        |_| Ok(()),
    )
}

pub(crate) fn forward_backward_with_selected_state_callback<F>(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    scratch: &mut HMMWorkspace,
    selected_markers: &[usize],
    state_callback: F,
) -> Result<PosteriorWindow, String>
where
    F: FnMut(usize, &[f32]) -> Result<(), String>,
{
    forward_backward_selected_state_impl(
        observations,
        states,
        cm_positions,
        config,
        scratch,
        selected_markers,
        state_callback,
    )
}

pub(crate) fn forward_backward_with_selected_state_callbacks_pair<F>(
    observations0: &PackedHaplotypes,
    states0: &CompositePanel,
    observations1: &PackedHaplotypes,
    states1: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    scratch0: &mut HMMWorkspace,
    scratch1: &mut HMMWorkspace,
    selected_markers: &[usize],
    mut state_callback: F,
) -> Result<PosteriorWindow, String>
where
    F: FnMut(usize, &[f32], &[f32]) -> Result<(), String>,
{
    forward_backward_selected_states_pair_impl(
        observations0,
        states0,
        observations1,
        states1,
        cm_positions,
        config,
        scratch0,
        scratch1,
        selected_markers,
        &mut state_callback,
    )
}

pub(crate) fn forward_backward_with_selected_state_callbacks_pair_mismatch<F>(
    observations0: &PackedHaplotypes,
    states0: &CompositePanel,
    observations1: &PackedHaplotypes,
    states1: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    scratch0: &mut HMMWorkspace,
    scratch1: &mut HMMWorkspace,
    selected_markers: &[usize],
    mut state_callback: F,
) -> Result<usize, String>
where
    F: FnMut(usize, &[f32], &[f32]) -> Result<(), String>,
{
    forward_backward_selected_states_pair_mismatch_impl(
        observations0,
        states0,
        observations1,
        states1,
        cm_positions,
        config,
        scratch0,
        scratch1,
        selected_markers,
        &mut state_callback,
    )
}

const EMISSION_MATCH: u8 = 0;
const EMISSION_MISMATCH: u8 = 1;
const EMISSION_OBSERVATION_MISSING: u8 = 2;
const EMISSION_STATE_MISSING: u8 = 3;

struct MismatchEmissionMatrix {
    codes: Vec<u8>,
    state_missing_emissions: Vec<f32>,
    unscaled_forward: Vec<bool>,
    state_count: usize,
    match_emission: f32,
    mismatch_emission: f32,
}

impl MismatchEmissionMatrix {
    fn build(
        observations: &PackedHaplotypes,
        states: &CompositePanel,
        allele_counts: &[usize],
        config: &HMMConfig,
    ) -> Result<Self, String> {
        let marker_count = observations.n_markers;
        let state_count = states.haplotypes.n_haplotypes;
        if marker_count == 0
            || state_count == 0
            || observations.n_haplotypes != 1
            || states.marker_indices.len() != marker_count
            || allele_counts.len() != marker_count
        {
            return Err("mismatch HMM matrix dimensions are invalid".to_owned());
        }
        let code_count = marker_count
            .checked_mul(state_count)
            .ok_or_else(|| "mismatch HMM matrix size overflow".to_owned())?;
        let mut codes = vec![EMISSION_OBSERVATION_MISSING; code_count];
        let mut state_missing_emissions = vec![1.0f32; marker_count];
        let mut unscaled_forward = vec![false; marker_count];
        for marker in 0..marker_count {
            let panel_marker = states.marker_indices[marker];
            if panel_marker >= states.haplotypes.n_markers {
                return Err("mismatch HMM panel marker is out of range".to_owned());
            }
            let Some(observation) = observations.allele(marker, 0) else {
                continue;
            };
            unscaled_forward[marker] = states.haplotypes.bits_per_allele == 1
                && (0..states.haplotypes.n_haplotypes)
                    .step_by(64)
                    .all(|first_state| {
                        states
                            .haplotypes
                            .biallelic_marker_word(panel_marker, first_state)
                            .is_some_and(|(_, missing)| missing == 0)
                    });
            state_missing_emissions[marker] = 1.0 / allele_counts[marker].max(1) as f32;
            let start = marker * state_count;
            let row = &mut codes[start..start + state_count];
            for (state, code) in row.iter_mut().enumerate() {
                *code = match states.haplotypes.allele(panel_marker, state) {
                    None => EMISSION_STATE_MISSING,
                    Some(allele) if allele == observation => EMISSION_MATCH,
                    Some(_) => EMISSION_MISMATCH,
                };
            }
        }
        Ok(Self {
            codes,
            state_missing_emissions,
            unscaled_forward,
            state_count,
            match_emission: (1.0 - config.err) as f32,
            mismatch_emission: config.err.max(1.0e-12) as f32,
        })
    }

    #[inline(always)]
    fn emission(&self, marker: usize, state: usize) -> f32 {
        match self.codes[marker * self.state_count + state] {
            EMISSION_MATCH => self.match_emission,
            EMISSION_MISMATCH => self.mismatch_emission,
            EMISSION_OBSERVATION_MISSING => 1.0,
            EMISSION_STATE_MISSING => self.state_missing_emissions[marker],
            _ => unreachable!("mismatch HMM emission code is internally defined"),
        }
    }

    fn memory_bytes(&self) -> usize {
        self.codes.len()
            + self.state_missing_emissions.len() * std::mem::size_of::<f32>()
            + self.unscaled_forward.len() * std::mem::size_of::<bool>()
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_backward_selected_states_pair_mismatch_impl<F>(
    observations0: &PackedHaplotypes,
    states0: &CompositePanel,
    observations1: &PackedHaplotypes,
    states1: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    scratch0: &mut HMMWorkspace,
    scratch1: &mut HMMWorkspace,
    selected_markers: &[usize],
    state_callback: &mut F,
) -> Result<usize, String>
where
    F: FnMut(usize, &[f32], &[f32]) -> Result<(), String>,
{
    let n_markers = observations0.n_markers;
    let states_count0 = states0.haplotypes.n_haplotypes;
    let states_count1 = states1.haplotypes.n_haplotypes;
    if n_markers == 0
        || observations1.n_markers != n_markers
        || states_count0 == 0
        || states_count1 == 0
        || observations0.n_haplotypes != 1
        || observations1.n_haplotypes != 1
        || states0.marker_indices.len() != n_markers
        || states1.marker_indices.len() != n_markers
        || states0.allele_counts.len() != states0.haplotypes.n_markers
        || states1.allele_counts.len() != states1.haplotypes.n_markers
        || states0
            .marker_indices
            .iter()
            .any(|&marker| marker >= states0.haplotypes.n_markers)
        || states1
            .marker_indices
            .iter()
            .any(|&marker| marker >= states1.haplotypes.n_markers)
    {
        return Err("paired mismatch HMM dimensions are invalid".to_owned());
    }
    if cm_positions.len() != n_markers
        || selected_markers.is_empty()
        || selected_markers.iter().any(|&marker| marker >= n_markers)
        || selected_markers.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("paired mismatch HMM markers or map are invalid".to_owned());
    }
    if config.ne == 0
        || !config.recombination_intensity.is_finite()
        || config.recombination_intensity <= 0.0
        || config.haplotype_count == 0
        || !config.err.is_finite()
        || config.err < 0.0
        || config.err >= 0.5
        || cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("paired mismatch HMM parameters are invalid".to_owned());
    }
    let allele_counts0 = paired_hmm_allele_counts(observations0, states0, n_markers);
    let allele_counts1 = paired_hmm_allele_counts(observations1, states1, n_markers);
    let matrix0 = MismatchEmissionMatrix::build(observations0, states0, &allele_counts0, config)?;
    let matrix1 = MismatchEmissionMatrix::build(observations1, states1, &allele_counts1, config)?;
    scratch0.transition_probabilities.resize(n_markers, 0.0);
    scratch1.transition_probabilities.resize(n_markers, 0.0);
    for marker in 1..n_markers {
        let distance = cm_positions[marker] - cm_positions[marker - 1];
        scratch0.transition_probabilities[marker] = transition_probability(
            distance,
            states_count0,
            config.haplotype_count,
            config.recombination_intensity,
        );
        scratch1.transition_probabilities[marker] = transition_probability(
            distance,
            states_count1,
            config.haplotype_count,
            config.recombination_intensity,
        );
    }
    paired_mismatch_initialize_forward(&matrix0, scratch0);
    paired_mismatch_initialize_forward(&matrix1, scratch1);
    scratch0
        .selected_forward_values
        .resize(selected_markers.len() * states_count0, 0.0);
    scratch1
        .selected_forward_values
        .resize(selected_markers.len() * states_count1, 0.0);
    let mut sum0 = scratch0.forward.iter().sum::<f32>();
    let mut sum1 = scratch1.forward.iter().sum::<f32>();
    let mut selected_forward_index = 0usize;
    if selected_markers.first() == Some(&0) {
        paired_store_selected_forward(scratch0, selected_forward_index, states_count0)?;
        paired_store_selected_forward(scratch1, selected_forward_index, states_count1)?;
        selected_forward_index += 1;
    }
    for marker in 1..n_markers {
        paired_mismatch_transition_forward(
            &matrix0,
            scratch0,
            marker,
            scratch0.transition_probabilities[marker],
            sum0,
        );
        paired_mismatch_transition_forward(
            &matrix1,
            scratch1,
            marker,
            scratch1.transition_probabilities[marker],
            sum1,
        );
        sum0 = scratch0.forward.iter().sum::<f32>();
        sum1 = scratch1.forward.iter().sum::<f32>();
        if selected_markers.get(selected_forward_index) == Some(&marker) {
            paired_store_selected_forward(scratch0, selected_forward_index, states_count0)?;
            paired_store_selected_forward(scratch1, selected_forward_index, states_count1)?;
            selected_forward_index += 1;
        }
    }
    debug_assert_eq!(selected_forward_index, selected_markers.len());

    scratch0.backward.fill(1.0);
    scratch1.backward.fill(1.0);
    let mut selected_backward_index = selected_markers.len().checked_sub(1);
    for marker in (0..n_markers).rev() {
        paired_mismatch_fill_emissions(&matrix0, &mut scratch0.emission_block, marker);
        paired_mismatch_fill_emissions(&matrix1, &mut scratch1.emission_block, marker);
        let emit_this_state =
            selected_backward_index.is_some_and(|index| selected_markers[index] == marker);
        if emit_this_state {
            let index = selected_backward_index.unwrap();
            paired_selected_posterior(scratch0, states_count0, index, marker)?;
            paired_selected_posterior(scratch1, states_count1, index, marker)?;
            state_callback(marker, &scratch0.state_posterior, &scratch1.state_posterior)?;
            selected_backward_index =
                selected_backward_index.and_then(|index| index.checked_sub(1));
        }
        if marker > 0 {
            paired_backward_update(
                &mut scratch0.backward,
                &mut scratch1.backward,
                &scratch0.emission_block,
                &scratch1.emission_block,
                scratch0.transition_probabilities[marker],
                scratch1.transition_probabilities[marker],
            );
        }
    }
    debug_assert!(selected_backward_index.is_none());
    Ok(matrix0
        .memory_bytes()
        .saturating_add(matrix1.memory_bytes()))
}

fn paired_mismatch_initialize_forward(matrix: &MismatchEmissionMatrix, scratch: &mut HMMWorkspace) {
    scratch.forward.resize(matrix.state_count, 0.0);
    scratch.previous_forward.resize(matrix.state_count, 0.0);
    scratch.backward.resize(matrix.state_count, 0.0);
    scratch.emission_block.resize(matrix.state_count, 0.0);
    scratch.state_posterior.resize(matrix.state_count, 0.0);
    for state in 0..matrix.state_count {
        scratch.forward[state] = matrix.emission(0, state);
    }
}

fn paired_mismatch_transition_forward(
    matrix: &MismatchEmissionMatrix,
    scratch: &mut HMMWorkspace,
    marker: usize,
    switch: f32,
    previous_sum: f32,
) {
    let (scale, shift) = if matrix.unscaled_forward[marker] {
        (
            1.0 - switch,
            switch * previous_sum / matrix.state_count as f32,
        )
    } else {
        (
            (1.0 - switch) / previous_sum,
            switch / matrix.state_count as f32,
        )
    };
    for state in 0..matrix.state_count {
        scratch.previous_forward[state] =
            (scale * scratch.forward[state] + shift) * matrix.emission(marker, state);
    }
    std::mem::swap(&mut scratch.forward, &mut scratch.previous_forward);
}

fn paired_mismatch_fill_emissions(
    matrix: &MismatchEmissionMatrix,
    emissions: &mut Vec<f32>,
    marker: usize,
) {
    emissions.resize(matrix.state_count, 0.0);
    for state in 0..matrix.state_count {
        emissions[state] = matrix.emission(marker, state);
    }
}

#[allow(clippy::too_many_arguments)]
const STAGE2_FORWARD_CHECKPOINT_STRIDE: usize = 256;
const DEFAULT_STAGE2_FORWARD_BUDGET_BYTES: usize = 16 * 1024 * 1024;

static STAGE2_FORWARD_CALLS: AtomicU64 = AtomicU64::new(0);
static STAGE2_FORWARD_CHECKPOINTED_CALLS: AtomicU64 = AtomicU64::new(0);
static STAGE2_FORWARD_MAX_SELECTED_MARKERS: AtomicU64 = AtomicU64::new(0);
static STAGE2_FORWARD_MAX_STATES: AtomicU64 = AtomicU64::new(0);
static STAGE2_FORWARD_MAX_FULL_BYTES: AtomicU64 = AtomicU64::new(0);
static STAGE2_FORWARD_MAX_ACTIVE_BYTES: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Stage2ForwardProfile {
    pub(crate) calls: u64,
    pub(crate) checkpointed_calls: u64,
    pub(crate) max_selected_markers: u64,
    pub(crate) max_states: u64,
    pub(crate) max_full_forward_bytes: u64,
    pub(crate) max_active_forward_bytes: u64,
    pub(crate) budget_bytes: Option<usize>,
}

pub(crate) fn reset_stage2_forward_profile_peaks() {
    STAGE2_FORWARD_MAX_SELECTED_MARKERS.store(0, AtomicOrdering::Relaxed);
    STAGE2_FORWARD_MAX_STATES.store(0, AtomicOrdering::Relaxed);
    STAGE2_FORWARD_MAX_FULL_BYTES.store(0, AtomicOrdering::Relaxed);
    STAGE2_FORWARD_MAX_ACTIVE_BYTES.store(0, AtomicOrdering::Relaxed);
}

pub(crate) fn stage2_forward_profile_totals() -> Stage2ForwardProfile {
    Stage2ForwardProfile {
        calls: STAGE2_FORWARD_CALLS.load(AtomicOrdering::Relaxed),
        checkpointed_calls: STAGE2_FORWARD_CHECKPOINTED_CALLS.load(AtomicOrdering::Relaxed),
        max_selected_markers: STAGE2_FORWARD_MAX_SELECTED_MARKERS.load(AtomicOrdering::Relaxed),
        max_states: STAGE2_FORWARD_MAX_STATES.load(AtomicOrdering::Relaxed),
        max_full_forward_bytes: STAGE2_FORWARD_MAX_FULL_BYTES.load(AtomicOrdering::Relaxed),
        max_active_forward_bytes: STAGE2_FORWARD_MAX_ACTIVE_BYTES.load(AtomicOrdering::Relaxed),
        budget_bytes: stage2_forward_budget_bytes(),
    }
}

fn parse_stage2_forward_budget_mb(raw: &str) -> Result<Option<usize>, ()> {
    let mb = raw.trim().parse::<usize>().map_err(|_| ())?;
    if mb == 0 {
        Ok(None)
    } else {
        mb.checked_mul(1024 * 1024).map(Some).ok_or(())
    }
}

fn stage2_forward_budget_bytes() -> Option<usize> {
    match std::env::var("JX_IMP_STAGE2_FORWARD_BUDGET_MB") {
        Ok(value) => parse_stage2_forward_budget_mb(&value)
            .unwrap_or(Some(DEFAULT_STAGE2_FORWARD_BUDGET_BYTES)),
        Err(_) => Some(DEFAULT_STAGE2_FORWARD_BUDGET_BYTES),
    }
}

fn forward_backward_selected_states_pair_impl<F>(
    observations0: &PackedHaplotypes,
    states0: &CompositePanel,
    observations1: &PackedHaplotypes,
    states1: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    scratch0: &mut HMMWorkspace,
    scratch1: &mut HMMWorkspace,
    selected_markers: &[usize],
    state_callback: &mut F,
) -> Result<PosteriorWindow, String>
where
    F: FnMut(usize, &[f32], &[f32]) -> Result<(), String>,
{
    forward_backward_selected_states_pair_impl_with_budget(
        observations0,
        states0,
        observations1,
        states1,
        cm_positions,
        config,
        scratch0,
        scratch1,
        selected_markers,
        state_callback,
        stage2_forward_budget_bytes(),
    )
}

#[allow(clippy::too_many_arguments)]
fn forward_backward_selected_states_pair_impl_with_budget<F>(
    observations0: &PackedHaplotypes,
    states0: &CompositePanel,
    observations1: &PackedHaplotypes,
    states1: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    scratch0: &mut HMMWorkspace,
    scratch1: &mut HMMWorkspace,
    selected_markers: &[usize],
    state_callback: &mut F,
    forward_budget_bytes: Option<usize>,
) -> Result<PosteriorWindow, String>
where
    F: FnMut(usize, &[f32], &[f32]) -> Result<(), String>,
{
    let profile_hmm = detail_profiling_enabled();
    let profile_started = Instant::now();
    let n_markers = observations0.n_markers;
    let states_count0 = states0.haplotypes.n_haplotypes;
    let states_count1 = states1.haplotypes.n_haplotypes;
    if n_markers == 0
        || observations1.n_markers != n_markers
        || states_count0 == 0
        || states_count1 == 0
        || observations0.n_haplotypes != 1
        || observations1.n_haplotypes != 1
    {
        return Err("paired selected HMM dimensions are invalid".to_owned());
    }
    for (states, observations) in [(states0, observations0), (states1, observations1)] {
        if states.marker_indices.len() != n_markers
            || states.allele_counts.len() != states.haplotypes.n_markers
            || states
                .marker_indices
                .iter()
                .any(|&marker| marker >= states.haplotypes.n_markers)
            || observations.n_markers != n_markers
        {
            return Err(
                "paired selected HMM marker mapping differs between observations and states"
                    .to_owned(),
            );
        }
    }
    if cm_positions.len() != n_markers
        || selected_markers.iter().any(|&marker| marker >= n_markers)
        || selected_markers.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("paired selected HMM markers or map are invalid".to_owned());
    }
    if config.ne == 0
        || !config.recombination_intensity.is_finite()
        || config.recombination_intensity <= 0.0
        || config.haplotype_count == 0
        || !config.err.is_finite()
        || config.err < 0.0
        || config.err >= 0.5
        || cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("paired selected HMM parameters are invalid".to_owned());
    }
    let allele_counts0 = paired_hmm_allele_counts(observations0, states0, n_markers);
    let allele_counts1 = paired_hmm_allele_counts(observations1, states1, n_markers);
    scratch0.transition_probabilities.resize(n_markers, 0.0);
    scratch1.transition_probabilities.resize(n_markers, 0.0);
    for marker in 1..n_markers {
        let distance = cm_positions[marker] - cm_positions[marker - 1];
        scratch0.transition_probabilities[marker] = transition_probability(
            distance,
            states_count0,
            config.haplotype_count,
            config.recombination_intensity,
        );
        scratch1.transition_probabilities[marker] = transition_probability(
            distance,
            states_count1,
            config.haplotype_count,
            config.recombination_intensity,
        );
    }
    paired_initialize_forward(observations0, states0, &allele_counts0, config, scratch0);
    paired_initialize_forward(observations1, states1, &allele_counts1, config, scratch1);
    let full_forward_bytes0 = selected_markers
        .len()
        .saturating_mul(states_count0)
        .saturating_mul(std::mem::size_of::<f32>());
    let full_forward_bytes1 = selected_markers
        .len()
        .saturating_mul(states_count1)
        .saturating_mul(std::mem::size_of::<f32>());
    let checkpointed = forward_budget_bytes.is_some_and(|budget| {
        n_markers > STAGE2_FORWARD_CHECKPOINT_STRIDE
            && !selected_markers.is_empty()
            && full_forward_bytes0.max(full_forward_bytes1) > budget
    });
    let mut checkpoint_sums = Vec::new();
    if checkpointed {
        let block_count = n_markers.div_ceil(STAGE2_FORWARD_CHECKPOINT_STRIDE);
        let checkpoint_len0 = block_count
            .checked_mul(states_count0)
            .ok_or_else(|| "paired selected HMM checkpoint size overflow".to_owned())?;
        let checkpoint_len1 = block_count
            .checked_mul(states_count1)
            .ok_or_else(|| "paired selected HMM checkpoint size overflow".to_owned())?;
        scratch0.checkpoint_values.resize(checkpoint_len0, 0.0);
        scratch1.checkpoint_values.resize(checkpoint_len1, 0.0);
        checkpoint_sums.resize(block_count, (0.0f32, 0.0f32));
        let mut max_block_selected = 0usize;
        let mut sel_idx = 0usize;
        while sel_idx < selected_markers.len() {
            let block = selected_markers[sel_idx] / STAGE2_FORWARD_CHECKPOINT_STRIDE;
            let block_end = (block + 1).saturating_mul(STAGE2_FORWARD_CHECKPOINT_STRIDE);
            let mut count = 0usize;
            while sel_idx < selected_markers.len() && selected_markers[sel_idx] < block_end {
                count += 1;
                sel_idx += 1;
            }
            max_block_selected = max_block_selected.max(count);
        }
        let block_forward_len0 = max_block_selected
            .checked_mul(states_count0)
            .ok_or_else(|| "paired selected HMM block forward-cache size overflow".to_owned())?;
        let block_forward_len1 = max_block_selected
            .checked_mul(states_count1)
            .ok_or_else(|| "paired selected HMM block forward-cache size overflow".to_owned())?;
        scratch0
            .selected_forward_values
            .resize(block_forward_len0, 0.0);
        scratch1
            .selected_forward_values
            .resize(block_forward_len1, 0.0);
        if forward_budget_bytes.is_some_and(|budget| {
            scratch0
                .selected_forward_values
                .capacity()
                .saturating_mul(std::mem::size_of::<f32>())
                > budget
        }) {
            scratch0
                .selected_forward_values
                .shrink_to(block_forward_len0);
        }
        if forward_budget_bytes.is_some_and(|budget| {
            scratch1
                .selected_forward_values
                .capacity()
                .saturating_mul(std::mem::size_of::<f32>())
                > budget
        }) {
            scratch1
                .selected_forward_values
                .shrink_to(block_forward_len1);
        }
    } else {
        scratch0.checkpoint_values.clear();
        scratch0.checkpoint_values.shrink_to(0);
        scratch1.checkpoint_values.clear();
        scratch1.checkpoint_values.shrink_to(0);
        scratch0
            .selected_forward_values
            .resize(selected_markers.len() * states_count0, 0.0);
        scratch1
            .selected_forward_values
            .resize(selected_markers.len() * states_count1, 0.0);
    }
    STAGE2_FORWARD_CALLS.fetch_add(1, AtomicOrdering::Relaxed);
    if checkpointed {
        STAGE2_FORWARD_CHECKPOINTED_CALLS.fetch_add(1, AtomicOrdering::Relaxed);
    }
    STAGE2_FORWARD_MAX_SELECTED_MARKERS
        .fetch_max(selected_markers.len() as u64, AtomicOrdering::Relaxed);
    STAGE2_FORWARD_MAX_STATES.fetch_max(
        states_count0.max(states_count1) as u64,
        AtomicOrdering::Relaxed,
    );
    STAGE2_FORWARD_MAX_FULL_BYTES.fetch_max(
        full_forward_bytes0.max(full_forward_bytes1) as u64,
        AtomicOrdering::Relaxed,
    );
    let active_bytes0 = scratch0
        .selected_forward_values
        .capacity()
        .saturating_add(scratch0.checkpoint_values.capacity())
        .saturating_mul(std::mem::size_of::<f32>());
    let active_bytes1 = scratch1
        .selected_forward_values
        .capacity()
        .saturating_add(scratch1.checkpoint_values.capacity())
        .saturating_mul(std::mem::size_of::<f32>());
    STAGE2_FORWARD_MAX_ACTIVE_BYTES.fetch_max(
        active_bytes0.max(active_bytes1) as u64,
        AtomicOrdering::Relaxed,
    );
    let forward_started = Instant::now();
    let mut sums0 = (
        scratch0.forward.iter().sum::<f32>(),
        scratch1.forward.iter().sum::<f32>(),
    );
    let mut selected_forward_index = 0usize;
    if checkpointed {
        scratch0.checkpoint_values[0..states_count0].copy_from_slice(&scratch0.forward);
        scratch1.checkpoint_values[0..states_count1].copy_from_slice(&scratch1.forward);
        checkpoint_sums[0] = sums0;
    } else if selected_markers.first() == Some(&0) {
        paired_store_selected_forward(scratch0, selected_forward_index, states_count0)?;
        paired_store_selected_forward(scratch1, selected_forward_index, states_count1)?;
        selected_forward_index += 1;
    }
    let mut allele_row_words0 = Vec::new();
    let mut missing_row_words0 = Vec::new();
    let mut allele_row_words1 = Vec::new();
    let mut missing_row_words1 = Vec::new();
    let complete_state_major0 = has_complete_state_major_biallelic_rows(&states0.haplotypes);
    let complete_state_major1 = has_complete_state_major_biallelic_rows(&states1.haplotypes);
    for marker in 1..n_markers {
        paired_transition_forward(
            observations0,
            states0,
            &allele_counts0,
            config,
            scratch0,
            marker,
            scratch0.transition_probabilities[marker],
            sums0.0,
            complete_state_major0,
            &mut allele_row_words0,
            &mut missing_row_words0,
        )?;
        paired_transition_forward(
            observations1,
            states1,
            &allele_counts1,
            config,
            scratch1,
            marker,
            scratch1.transition_probabilities[marker],
            sums0.1,
            complete_state_major1,
            &mut allele_row_words1,
            &mut missing_row_words1,
        )?;
        sums0 = paired_sum_lanes(&scratch0.forward, &scratch1.forward);
        if checkpointed {
            if marker % STAGE2_FORWARD_CHECKPOINT_STRIDE == 0 {
                let block = marker / STAGE2_FORWARD_CHECKPOINT_STRIDE;
                let start0 = block * states_count0;
                let start1 = block * states_count1;
                scratch0.checkpoint_values[start0..start0 + states_count0]
                    .copy_from_slice(&scratch0.forward);
                scratch1.checkpoint_values[start1..start1 + states_count1]
                    .copy_from_slice(&scratch1.forward);
                checkpoint_sums[block] = sums0;
            }
        } else if selected_markers.get(selected_forward_index) == Some(&marker) {
            paired_store_selected_forward(scratch0, selected_forward_index, states_count0)?;
            paired_store_selected_forward(scratch1, selected_forward_index, states_count1)?;
            selected_forward_index += 1;
        }
    }
    if !checkpointed {
        debug_assert_eq!(selected_forward_index, selected_markers.len());
    }
    let forward_elapsed = forward_started.elapsed();

    let backward_started = Instant::now();
    scratch0.backward.fill(1.0);
    scratch1.backward.fill(1.0);
    let mut selected_backward_index = selected_markers.len().checked_sub(1);
    let mut loaded_block: Option<usize> = None;
    let mut loaded_block_selected_start = 0usize;
    let mut callback_elapsed = Duration::ZERO;
    for marker in (0..n_markers).rev() {
        let emit_this_state =
            selected_backward_index.is_some_and(|index| selected_markers[index] == marker);
        if emit_this_state {
            let index = selected_backward_index.unwrap();
            let forward_slot = if checkpointed {
                let block = marker / STAGE2_FORWARD_CHECKPOINT_STRIDE;
                if loaded_block != Some(block) {
                    let block_start = block * STAGE2_FORWARD_CHECKPOINT_STRIDE;
                    let block_end = (block_start + STAGE2_FORWARD_CHECKPOINT_STRIDE).min(n_markers);
                    let sel_start = selected_markers.partition_point(|&m| m < block_start);
                    let sel_end = selected_markers.partition_point(|&m| m < block_end);
                    debug_assert!(sel_start < sel_end);
                    let start0 = block * states_count0;
                    let start1 = block * states_count1;
                    scratch0.forward.copy_from_slice(
                        &scratch0.checkpoint_values[start0..start0 + states_count0],
                    );
                    scratch1.forward.copy_from_slice(
                        &scratch1.checkpoint_values[start1..start1 + states_count1],
                    );
                    let mut block_sums = checkpoint_sums[block];
                    let mut sel_cursor = sel_start;
                    if selected_markers[sel_cursor] == block_start {
                        paired_store_selected_forward(
                            scratch0,
                            sel_cursor - sel_start,
                            states_count0,
                        )?;
                        paired_store_selected_forward(
                            scratch1,
                            sel_cursor - sel_start,
                            states_count1,
                        )?;
                        sel_cursor += 1;
                    }
                    let replay_end = selected_markers[sel_end - 1] + 1;
                    for replay_marker in (block_start + 1)..replay_end {
                        paired_transition_forward(
                            observations0,
                            states0,
                            &allele_counts0,
                            config,
                            scratch0,
                            replay_marker,
                            scratch0.transition_probabilities[replay_marker],
                            block_sums.0,
                            complete_state_major0,
                            &mut allele_row_words0,
                            &mut missing_row_words0,
                        )?;
                        paired_transition_forward(
                            observations1,
                            states1,
                            &allele_counts1,
                            config,
                            scratch1,
                            replay_marker,
                            scratch1.transition_probabilities[replay_marker],
                            block_sums.1,
                            complete_state_major1,
                            &mut allele_row_words1,
                            &mut missing_row_words1,
                        )?;
                        block_sums = paired_sum_lanes(&scratch0.forward, &scratch1.forward);
                        if sel_cursor < sel_end && selected_markers[sel_cursor] == replay_marker {
                            paired_store_selected_forward(
                                scratch0,
                                sel_cursor - sel_start,
                                states_count0,
                            )?;
                            paired_store_selected_forward(
                                scratch1,
                                sel_cursor - sel_start,
                                states_count1,
                            )?;
                            sel_cursor += 1;
                        }
                    }
                    debug_assert_eq!(sel_cursor, sel_end);
                    loaded_block = Some(block);
                    loaded_block_selected_start = sel_start;
                }
                index - loaded_block_selected_start
            } else {
                index
            };
            paired_selected_posterior(scratch0, states_count0, forward_slot, marker)?;
            paired_selected_posterior(scratch1, states_count1, forward_slot, marker)?;
            if profile_hmm {
                let callback_started = Instant::now();
                state_callback(marker, &scratch0.state_posterior, &scratch1.state_posterior)?;
                callback_elapsed += callback_started.elapsed();
            } else {
                state_callback(marker, &scratch0.state_posterior, &scratch1.state_posterior)?;
            }
            selected_backward_index =
                selected_backward_index.and_then(|index| index.checked_sub(1));
        }
        if marker > 0 {
            paired_fill_emissions(
                observations0,
                states0,
                &allele_counts0,
                config,
                &mut scratch0.emission_block,
                marker,
                complete_state_major0,
                &mut allele_row_words0,
                &mut missing_row_words0,
            )?;
            paired_fill_emissions(
                observations1,
                states1,
                &allele_counts1,
                config,
                &mut scratch1.emission_block,
                marker,
                complete_state_major1,
                &mut allele_row_words1,
                &mut missing_row_words1,
            )?;
            let switch0 = scratch0.transition_probabilities[marker];
            let switch1 = scratch1.transition_probabilities[marker];
            paired_backward_update(
                &mut scratch0.backward,
                &mut scratch1.backward,
                &scratch0.emission_block,
                &scratch1.emission_block,
                switch0,
                switch1,
            );
        }
    }
    debug_assert!(selected_backward_index.is_none());
    let backward_elapsed = backward_started.elapsed();
    if profile_hmm {
        eprintln!(
            "JANUSX_IMP_HMM_PROFILE selected=2 markers={n_markers} states={} anchors={} forward_s={:.6} callback_s={:.6} backward_s={:.6} total_s={:.6}",
            states_count0 + states_count1,
            selected_markers.len(),
            forward_elapsed.as_secs_f64(),
            callback_elapsed.as_secs_f64(),
            backward_elapsed.as_secs_f64(),
            profile_started.elapsed().as_secs_f64(),
        );
    }
    Ok(PosteriorWindow {
        allele_probabilities: Vec::new(),
        leave_one_out_allele_probabilities: Vec::new(),
        allele_offsets: Vec::new(),
        state_probabilities: Vec::new(),
        parameter_stats: HMMParameterStats::default(),
    })
}

fn paired_hmm_allele_counts(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    n_markers: usize,
) -> Vec<usize> {
    (0..n_markers)
        .map(|marker| {
            let panel_marker = states.marker_indices[marker];
            let declared = states.allele_counts[panel_marker].max(1);
            let observed = observations
                .allele(marker, 0)
                .map_or(1, |allele| usize::from(allele) + 1);
            declared.max(observed)
        })
        .collect()
}

fn paired_initialize_forward(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    allele_counts: &[usize],
    config: &HMMConfig,
    scratch: &mut HMMWorkspace,
) {
    let n_states = states.haplotypes.n_haplotypes;
    scratch.forward.resize(n_states, 0.0);
    scratch.previous_forward.resize(n_states, 0.0);
    scratch.backward.resize(n_states, 0.0);
    scratch.emission_block.resize(n_states, 0.0);
    scratch.state_posterior.resize(n_states, 0.0);
    let observation = observations.allele(0, 0);
    let panel_marker = states.marker_indices[0];
    if observation.is_some() {
        for state in 0..n_states {
            scratch.forward[state] = emission(
                observation,
                states.haplotypes.allele(panel_marker, state),
                allele_counts[0],
                config.err,
            );
        }
    } else {
        scratch.forward.fill(1.0);
    }
}

fn paired_store_selected_forward(
    scratch: &mut HMMWorkspace,
    selected_index: usize,
    state_count: usize,
) -> Result<(), String> {
    let start = selected_index
        .checked_mul(state_count)
        .ok_or_else(|| "paired selected HMM forward-cache offset overflow".to_owned())?;
    let end = start
        .checked_add(state_count)
        .ok_or_else(|| "paired selected HMM forward-cache end overflow".to_owned())?;
    let cache_len = scratch.selected_forward_values.len();
    let destination = scratch
        .selected_forward_values
        .get_mut(start..end)
        .ok_or_else(|| {
            format!(
                "paired selected HMM forward-cache slice {start}..{end} is outside length {}",
                cache_len
            )
        })?;
    destination.copy_from_slice(&scratch.forward);
    Ok(())
}

fn has_complete_state_major_biallelic_rows(haplotypes: &PackedHaplotypes) -> bool {
    haplotypes.is_state_major()
        && haplotypes.bits_per_allele == 1
        && haplotypes.missing_bits.iter().all(|&word| word == 0)
}

#[allow(clippy::too_many_arguments)]
fn paired_transition_forward(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    allele_counts: &[usize],
    config: &HMMConfig,
    scratch: &mut HMMWorkspace,
    marker: usize,
    switch: f32,
    previous_sum: f32,
    complete_state_major: bool,
    allele_row_words: &mut Vec<u64>,
    missing_row_words: &mut Vec<u64>,
) -> Result<(), String> {
    let n_states = states.haplotypes.n_haplotypes;
    let match_emission = (1.0 - config.err) as f32;
    let mismatch_emission = config.err.max(1.0e-12) as f32;
    let panel_marker = states.marker_indices[marker];
    let observation = observations.allele(marker, 0);
    let fused = if let Some(observed) = observation {
        with_biallelic_row_words(
            &states.haplotypes,
            panel_marker,
            allele_row_words,
            missing_row_words,
            |allele_words, missing_words| {
                if missing_words.iter().all(|&word| word == 0) {
                    transition_emit_biallelic_row(
                        &scratch.forward,
                        &mut scratch.previous_forward,
                        allele_words,
                        observed,
                        switch,
                        previous_sum,
                        match_emission,
                        mismatch_emission,
                        None,
                    );
                    true
                } else {
                    false
                }
            },
        )?
        .unwrap_or(false)
    } else {
        false
    };
    if !fused && complete_state_major {
        paired_transition_forward_state_major_complete(
            observation,
            states,
            config,
            scratch,
            marker,
            switch,
            previous_sum,
        );
        std::mem::swap(&mut scratch.forward, &mut scratch.previous_forward);
        return Ok(());
    }
    if !fused {
        let scale = (1.0 - switch) / previous_sum;
        let shift = switch / n_states as f32;
        for state in 0..n_states {
            let state_emission = emission(
                observation,
                states.haplotypes.allele(panel_marker, state),
                allele_counts[marker],
                config.err,
            );
            scratch.previous_forward[state] =
                (scale * scratch.forward[state] + shift) * state_emission;
        }
    }
    std::mem::swap(&mut scratch.forward, &mut scratch.previous_forward);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn paired_transition_forward_state_major_complete(
    observation: Option<u16>,
    states: &CompositePanel,
    config: &HMMConfig,
    scratch: &mut HMMWorkspace,
    marker: usize,
    switch: f32,
    previous_sum: f32,
) {
    let n_states = states.haplotypes.n_haplotypes;
    let stride = states.haplotypes.n_markers;
    let panel_marker = states.marker_indices[marker];
    let match_emission = (1.0 - config.err) as f32;
    let mismatch_emission = config.err.max(1.0e-12) as f32;
    let scale = (1.0 - switch) / previous_sum;
    let shift = switch / n_states as f32;
    let mut bit_index = panel_marker;
    for state in 0..n_states {
        let state_emission = observation.map_or(1.0, |observed| {
            // SAFETY: the state-major panel dimensions were validated before
            // the HMM loop; the final index is below state_count * n_markers.
            let allele = unsafe {
                (*states.haplotypes.allele_bits.get_unchecked(bit_index >> 6) >> (bit_index & 63))
                    & 1
            } as u16;
            if allele == observed {
                match_emission
            } else {
                mismatch_emission
            }
        });
        scratch.previous_forward[state] = (scale * scratch.forward[state] + shift) * state_emission;
        bit_index += stride;
    }
}

#[inline]
fn paired_sum_lanes(left: &[f32], right: &[f32]) -> (f32, f32) {
    let common = left.len().min(right.len());
    let mut left_sum = 0.0f32;
    let mut right_sum = 0.0f32;
    for state in 0..common {
        left_sum += left[state];
        right_sum += right[state];
    }
    for &value in &left[common..] {
        left_sum += value;
    }
    for &value in &right[common..] {
        right_sum += value;
    }
    (left_sum, right_sum)
}

#[allow(clippy::too_many_arguments)]
fn paired_fill_emissions(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    allele_counts: &[usize],
    config: &HMMConfig,
    emissions: &mut [f32],
    marker: usize,
    complete_state_major: bool,
    allele_row_words: &mut Vec<u64>,
    missing_row_words: &mut Vec<u64>,
) -> Result<(), String> {
    let n_states = states.haplotypes.n_haplotypes;
    let match_emission = (1.0 - config.err) as f32;
    let mismatch_emission = config.err.max(1.0e-12) as f32;
    let panel_marker = states.marker_indices[marker];
    let observation = observations.allele(marker, 0);
    if complete_state_major {
        if let Some(observed) = observation {
            let stride = states.haplotypes.n_markers;
            let mut bit_index = panel_marker;
            for emission in emissions.iter_mut().take(n_states) {
                // SAFETY: dimensions were validated and bit_index advances by
                // exactly one panel row for each of the n_states entries.
                let allele = unsafe {
                    (*states.haplotypes.allele_bits.get_unchecked(bit_index >> 6)
                        >> (bit_index & 63))
                        & 1
                } as u16;
                *emission = if allele == observed {
                    match_emission
                } else {
                    mismatch_emission
                };
                bit_index += stride;
            }
        } else {
            emissions[..n_states].fill(1.0);
        }
        return Ok(());
    }
    let emitted_from_packed = if let Some(observed) = observation {
        with_biallelic_row_words(
            &states.haplotypes,
            panel_marker,
            allele_row_words,
            missing_row_words,
            |allele_words, missing_words| {
                if missing_words.iter().all(|&word| word == 0) {
                    for state in 0..n_states {
                        emissions[state] = biallelic_observed_emission(
                            allele_words,
                            state,
                            observed,
                            match_emission,
                            mismatch_emission,
                        );
                    }
                    true
                } else {
                    false
                }
            },
        )?
        .unwrap_or(false)
    } else {
        false
    };
    if !emitted_from_packed {
        if observation.is_some() {
            for state in 0..n_states {
                emissions[state] = emission(
                    observation,
                    states.haplotypes.allele(panel_marker, state),
                    allele_counts[marker],
                    config.err,
                );
            }
        } else {
            emissions.fill(1.0);
        }
    }
    Ok(())
}

fn paired_selected_posterior(
    scratch: &mut HMMWorkspace,
    state_count: usize,
    selected_index: usize,
    marker: usize,
) -> Result<(), String> {
    let start = selected_index
        .checked_mul(state_count)
        .ok_or_else(|| "paired selected HMM posterior offset overflow".to_owned())?;
    let end = start
        .checked_add(state_count)
        .ok_or_else(|| "paired selected HMM posterior end overflow".to_owned())?;
    let forward = scratch
        .selected_forward_values
        .get(start..end)
        .ok_or_else(|| {
            format!(
                "paired selected HMM forward-cache slice {start}..{end} is outside length {} at marker {marker}",
                scratch.selected_forward_values.len()
            )
        })?;
    scratch.state_posterior.resize(state_count, 0.0);
    let mut sum = 0.0f32;
    for state in 0..state_count {
        sum += forward[state] * scratch.backward[state];
    }
    let denominator = if sum > f32::MIN_POSITIVE { sum } else { 1.0 };
    for state in 0..state_count {
        scratch.state_posterior[state] = (forward[state] * scratch.backward[state]) / denominator;
    }
    Ok(())
}

#[inline]
fn paired_backward_update(
    backward0: &mut [f32],
    backward1: &mut [f32],
    emissions0: &[f32],
    emissions1: &[f32],
    switch0: f32,
    switch1: f32,
) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        unsafe {
            paired_backward_update_avx2(
                backward0, backward1, emissions0, emissions1, switch0, switch1,
            );
        }
        return;
    }
    let common = backward0
        .len()
        .min(backward1.len())
        .min(emissions0.len())
        .min(emissions1.len());
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    for state in 0..common {
        backward0[state] *= emissions0[state];
        backward1[state] *= emissions1[state];
        sum0 += backward0[state];
        sum1 += backward1[state];
    }
    for state in common..backward0.len() {
        backward0[state] *= emissions0[state];
        sum0 += backward0[state];
    }
    for state in common..backward1.len() {
        backward1[state] *= emissions1[state];
        sum1 += backward1[state];
    }
    paired_normalize_backward(backward0, sum0, switch0);
    paired_normalize_backward(backward1, sum1, switch1);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn paired_backward_update_avx2(
    backward0: &mut [f32],
    backward1: &mut [f32],
    emissions0: &[f32],
    emissions1: &[f32],
    switch0: f32,
    switch1: f32,
) {
    use std::arch::x86_64::*;
    let common = backward0
        .len()
        .min(backward1.len())
        .min(emissions0.len())
        .min(emissions1.len());
    let mut state = 0usize;
    while state + 8 <= common {
        let values0 = _mm256_loadu_ps(backward0.as_ptr().add(state));
        let values1 = _mm256_loadu_ps(backward1.as_ptr().add(state));
        let emissions0 = _mm256_loadu_ps(emissions0.as_ptr().add(state));
        let emissions1 = _mm256_loadu_ps(emissions1.as_ptr().add(state));
        _mm256_storeu_ps(
            backward0.as_mut_ptr().add(state),
            _mm256_mul_ps(values0, emissions0),
        );
        _mm256_storeu_ps(
            backward1.as_mut_ptr().add(state),
            _mm256_mul_ps(values1, emissions1),
        );
        state += 8;
    }
    for state in state..common {
        backward0[state] *= emissions0[state];
        backward1[state] *= emissions1[state];
    }
    for state in common..backward0.len() {
        backward0[state] *= emissions0[state];
    }
    for state in common..backward1.len() {
        backward1[state] *= emissions1[state];
    }
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    for state in 0..backward0.len() {
        sum0 += backward0[state];
    }
    for state in 0..backward1.len() {
        sum1 += backward1[state];
    }
    normalize_backward_avx2(backward0, sum0, switch0);
    normalize_backward_avx2(backward1, sum1, switch1);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_backward_avx2(backward: &mut [f32], sum: f32, switch: f32) {
    use std::arch::x86_64::*;
    if sum <= f32::MIN_POSITIVE {
        backward.fill(1.0 / backward.len() as f32);
        return;
    }
    let scale_value = (1.0 - switch) / sum;
    let shift_value = switch / backward.len() as f32;
    let scale = _mm256_set1_ps(scale_value);
    let shift = _mm256_set1_ps(shift_value);
    let mut state = 0usize;
    while state + 8 <= backward.len() {
        let values = _mm256_loadu_ps(backward.as_ptr().add(state));
        let normalized = _mm256_add_ps(_mm256_mul_ps(scale, values), shift);
        _mm256_storeu_ps(backward.as_mut_ptr().add(state), normalized);
        state += 8;
    }
    for value in &mut backward[state..] {
        *value = scale_value * *value + shift_value;
    }
}

#[inline]
fn paired_normalize_backward(backward: &mut [f32], sum: f32, switch: f32) {
    if sum > f32::MIN_POSITIVE {
        let scale = (1.0 - switch) / sum;
        let shift = switch / backward.len() as f32;
        for value in backward {
            *value = scale * *value + shift;
        }
    } else {
        backward.fill(1.0 / backward.len() as f32);
    }
}

fn forward_backward_selected_state_impl<F>(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    scratch: &mut HMMWorkspace,
    selected_markers: &[usize],
    mut state_callback: F,
) -> Result<PosteriorWindow, String>
where
    F: FnMut(usize, &[f32]) -> Result<(), String>,
{
    let profile_hmm = detail_profiling_enabled();
    let profile_started = Instant::now();
    let n_markers = observations.n_markers;
    let n_states = states.haplotypes.n_haplotypes;
    if n_markers == 0 || n_states == 0 || observations.n_haplotypes != 1 {
        return Err("HMM expects one observation haplotype and positive dimensions".to_owned());
    }
    if states.marker_indices.len() != n_markers
        || states.allele_counts.len() != states.haplotypes.n_markers
        || states
            .marker_indices
            .iter()
            .any(|&marker| marker >= states.haplotypes.n_markers)
        || cm_positions.len() != n_markers
    {
        return Err("HMM marker mapping differs between observations, states, and map".to_owned());
    }
    if selected_markers.iter().any(|&marker| marker >= n_markers)
        || selected_markers.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("HMM selected state markers must be ordered unique indices".to_owned());
    }
    if config.ne == 0
        || !config.recombination_intensity.is_finite()
        || config.recombination_intensity <= 0.0
        || config.haplotype_count == 0
        || !config.err.is_finite()
        || config.err < 0.0
        || config.err >= 0.5
    {
        return Err("HMM ne/err parameters are outside their valid range".to_owned());
    }
    if cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("HMM genetic positions must be finite and nondecreasing".to_owned());
    }

    let allele_counts = (0..n_markers)
        .map(|marker| {
            let panel_marker = states.marker_indices[marker];
            let declared = states.allele_counts[panel_marker].max(1);
            let observed = observations
                .allele(marker, 0)
                .map_or(1, |allele| usize::from(allele) + 1);
            declared.max(observed)
        })
        .collect::<Vec<_>>();

    scratch.transition_probabilities.resize(n_markers, 0.0);
    for marker in 1..n_markers {
        scratch.transition_probabilities[marker] = transition_probability(
            cm_positions[marker] - cm_positions[marker - 1],
            n_states,
            config.haplotype_count,
            config.recombination_intensity,
        );
    }

    scratch.forward.resize(n_states, 0.0);
    scratch.previous_forward.resize(n_states, 0.0);
    scratch.backward.resize(n_states, 0.0);
    scratch.emission_block.resize(n_states, 0.0);
    scratch.state_posterior.resize(n_states, 0.0);
    scratch.selected_forward_values.clear();
    let selected_value_count = selected_markers
        .len()
        .checked_mul(n_states)
        .ok_or_else(|| "selected HMM forward-cache size overflow".to_owned())?;
    scratch
        .selected_forward_values
        .resize(selected_value_count, 0.0);
    let mut allele_row_words = Vec::new();
    let mut missing_row_words = Vec::new();

    let forward_started = Instant::now();
    scratch.forward.fill(0.0);
    let first_observation = observations.allele(0, 0);
    let first_panel_marker = states.marker_indices[0];
    let first_allele_count = allele_counts[0];
    if first_observation.is_some() {
        for state in 0..n_states {
            scratch.forward[state] = emission(
                first_observation,
                states.haplotypes.allele(first_panel_marker, state),
                first_allele_count,
                config.err,
            );
        }
    } else {
        scratch.forward.fill(1.0);
    }
    let mut previous_sum = scratch.forward.iter().sum::<f32>();
    let mut selected_forward_index = 0usize;
    if selected_markers.first() == Some(&0) {
        let offset = selected_forward_index * n_states;
        let cache_len = scratch.selected_forward_values.len();
        scratch
            .selected_forward_values
            .get_mut(offset..offset + n_states)
            .ok_or_else(|| {
                format!(
                    "selected HMM forward-cache slice {offset}..{} is outside length {}",
                    offset + n_states,
                    cache_len
                )
            })?
            .copy_from_slice(&scratch.forward);
        selected_forward_index += 1;
    }
    for marker in 1..n_markers {
        let match_emission = (1.0 - config.err) as f32;
        let mismatch_emission = config.err.max(1.0e-12) as f32;
        let panel_marker = states.marker_indices[marker];
        let switch = scratch.transition_probabilities[marker];
        let observation = observations.allele(marker, 0);
        let fused = if let Some(observed) = observation {
            with_biallelic_row_words(
                &states.haplotypes,
                panel_marker,
                &mut allele_row_words,
                &mut missing_row_words,
                |allele_words, missing_words| {
                    if missing_words.iter().all(|&word| word == 0) {
                        transition_emit_biallelic_row(
                            &scratch.forward,
                            &mut scratch.previous_forward,
                            allele_words,
                            observed,
                            switch,
                            previous_sum,
                            match_emission,
                            mismatch_emission,
                            None,
                        );
                        true
                    } else {
                        false
                    }
                },
            )?
            .unwrap_or(false)
        } else {
            false
        };
        if !fused {
            let scale = (1.0 - switch) / previous_sum;
            let shift = switch / n_states as f32;
            let allele_count = allele_counts[marker];
            for state in 0..n_states {
                let state_allele = states.haplotypes.allele(panel_marker, state);
                let state_emission = emission(observation, state_allele, allele_count, config.err);
                scratch.previous_forward[state] =
                    (scale * scratch.forward[state] + shift) * state_emission;
            }
        }
        std::mem::swap(&mut scratch.forward, &mut scratch.previous_forward);
        previous_sum = scratch.forward.iter().sum::<f32>();
        if selected_markers.get(selected_forward_index) == Some(&marker) {
            let offset = selected_forward_index * n_states;
            let cache_len = scratch.selected_forward_values.len();
            scratch
                .selected_forward_values
                .get_mut(offset..offset + n_states)
                .ok_or_else(|| {
                    format!(
                        "selected HMM forward-cache slice {offset}..{} is outside length {}",
                        offset + n_states,
                        cache_len
                    )
                })?
                .copy_from_slice(&scratch.forward);
            selected_forward_index += 1;
        }
    }
    debug_assert_eq!(selected_forward_index, selected_markers.len());
    let forward_elapsed = forward_started.elapsed();

    let backward_started = Instant::now();
    scratch.backward.fill(1.0);
    let mut selected_backward_index = selected_markers.len().checked_sub(1);
    let mut callback_elapsed = Duration::ZERO;
    for marker in (0..n_markers).rev() {
        let match_emission = (1.0 - config.err) as f32;
        let mismatch_emission = config.err.max(1.0e-12) as f32;
        let panel_marker = states.marker_indices[marker];
        let observation = observations.allele(marker, 0);
        let emitted_from_packed = if let Some(observed) = observation {
            with_biallelic_row_words(
                &states.haplotypes,
                panel_marker,
                &mut allele_row_words,
                &mut missing_row_words,
                |allele_words, missing_words| {
                    if missing_words.iter().all(|&word| word == 0) {
                        for state in 0..n_states {
                            scratch.emission_block[state] = biallelic_observed_emission(
                                allele_words,
                                state,
                                observed,
                                match_emission,
                                mismatch_emission,
                            );
                        }
                        true
                    } else {
                        false
                    }
                },
            )?
            .unwrap_or(false)
        } else {
            false
        };
        if !emitted_from_packed {
            if observation.is_some() {
                let allele_count = allele_counts[marker];
                for state in 0..n_states {
                    scratch.emission_block[state] = emission(
                        observation,
                        states.haplotypes.allele(panel_marker, state),
                        allele_count,
                        config.err,
                    );
                }
            } else {
                scratch.emission_block.fill(1.0);
            }
        }
        let emit_this_state =
            selected_backward_index.is_some_and(|index| selected_markers[index] == marker);
        let forward_offset = selected_backward_index
            .map(|index| index * n_states)
            .unwrap_or_default();
        let stored_forward = if emit_this_state {
            Some(
                scratch
                    .selected_forward_values
                    .get(forward_offset..forward_offset + n_states)
                    .ok_or_else(|| {
                        format!(
                            "selected HMM forward-cache slice {forward_offset}..{} is outside length {}",
                            forward_offset + n_states,
                            scratch.selected_forward_values.len()
                        )
                    })?,
            )
        } else {
            None
        };
        let switch = if marker == 0 {
            0.0
        } else {
            scratch.transition_probabilities[marker]
        };
        if let Some(forward) = stored_forward {
            let mut posterior_sum = 0.0f32;
            for state in 0..n_states {
                posterior_sum += forward[state] * scratch.backward[state];
            }
            let denominator = if posterior_sum > f32::MIN_POSITIVE {
                posterior_sum
            } else {
                1.0
            };
            for state in 0..n_states {
                scratch.state_posterior[state] =
                    (forward[state] * scratch.backward[state]) / denominator;
            }
            if profile_hmm {
                let callback_started = Instant::now();
                state_callback(marker, &scratch.state_posterior)?;
                callback_elapsed += callback_started.elapsed();
            } else {
                state_callback(marker, &scratch.state_posterior)?;
            }
            selected_backward_index =
                selected_backward_index.and_then(|index| index.checked_sub(1));
        }
        if marker > 0 {
            let mut sum = 0.0f32;
            for state in 0..n_states {
                scratch.backward[state] *= scratch.emission_block[state];
                sum += scratch.backward[state];
            }
            if sum > f32::MIN_POSITIVE {
                let scale = (1.0 - switch) / sum;
                let shift = switch / n_states as f32;
                for value in &mut scratch.backward {
                    *value = scale * *value + shift;
                }
            } else {
                scratch.backward.fill(1.0 / n_states as f32);
            }
        }
    }
    debug_assert!(selected_backward_index.is_none());
    let backward_elapsed = backward_started.elapsed();
    if profile_hmm {
        eprintln!(
            "JANUSX_IMP_HMM_PROFILE selected=1 markers={n_markers} states={n_states} anchors={} forward_s={:.6} callback_s={:.6} backward_s={:.6} total_s={:.6}",
            selected_markers.len(),
            forward_elapsed.as_secs_f64(),
            callback_elapsed.as_secs_f64(),
            backward_elapsed.as_secs_f64(),
            profile_started.elapsed().as_secs_f64(),
        );
    }
    Ok(PosteriorWindow {
        allele_probabilities: Vec::new(),
        leave_one_out_allele_probabilities: Vec::new(),
        allele_offsets: Vec::new(),
        state_probabilities: Vec::new(),
        parameter_stats: HMMParameterStats::default(),
    })
}

pub(crate) fn forward_backward_with_state_callback_and_errors<F>(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    marker_errors: &[f64],
    scratch: &mut HMMWorkspace,
    state_callback: F,
) -> Result<PosteriorWindow, String>
where
    F: FnMut(usize, &[f32]) -> Result<(), String>,
{
    forward_backward_impl(
        observations,
        states,
        cm_positions,
        config,
        scratch,
        Some(marker_errors),
        false,
        true,
        false,
        None,
        state_callback,
        None,
        |_| Ok(()),
    )
}

pub(crate) fn forward_backward_dense_with_state_callback_and_errors<F>(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    marker_errors: &[f64],
    scratch: &mut HMMWorkspace,
    mut state_callback: F,
) -> Result<PosteriorWindow, String>
where
    F: FnMut(usize, &[f32]) -> Result<(), String>,
{
    const DENSE_FORWARD_BUDGET_BYTES: usize = 32 * 1024 * 1024;
    let profile_hmm = detail_profiling_enabled();
    let profile_started = Instant::now();
    let n_markers = observations.n_markers;
    let n_states = states.haplotypes.n_haplotypes;
    if n_markers == 0 || n_states == 0 || observations.n_haplotypes != 1 {
        return Err("HMM expects one observation haplotype and positive dimensions".to_owned());
    }
    if states.marker_indices.len() != n_markers
        || states.allele_counts.len() != states.haplotypes.n_markers
        || states
            .marker_indices
            .iter()
            .any(|&marker| marker >= states.haplotypes.n_markers)
        || cm_positions.len() != n_markers
        || marker_errors.len() != n_markers
    {
        return Err("dense HMM marker mapping is inconsistent".to_owned());
    }
    if config.ne == 0
        || !config.recombination_intensity.is_finite()
        || config.recombination_intensity <= 0.0
        || config.haplotype_count == 0
        || !config.err.is_finite()
        || config.err < 0.0
        || config.err >= 0.5
        || marker_errors
            .iter()
            .any(|error| !error.is_finite() || *error < 0.0 || *error > 0.5)
        || cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("dense HMM parameters or genetic positions are invalid".to_owned());
    }
    let dense_forward_bytes = n_markers
        .checked_mul(n_states)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "dense HMM forward storage size overflow".to_owned())?;
    if dense_forward_bytes > DENSE_FORWARD_BUDGET_BYTES {
        return forward_backward_with_state_callback_and_errors(
            observations,
            states,
            cm_positions,
            config,
            marker_errors,
            scratch,
            state_callback,
        );
    }
    let allele_counts = (0..n_markers)
        .map(|marker| {
            let panel_marker = states.marker_indices[marker];
            let declared = states.allele_counts[panel_marker].max(1);
            let observed = observations
                .allele(marker, 0)
                .map_or(1, |allele| usize::from(allele) + 1);
            declared.max(observed)
        })
        .collect::<Vec<_>>();
    let forward_value_count = n_markers
        .checked_mul(n_states)
        .ok_or_else(|| "dense HMM forward storage size overflow".to_owned())?;
    scratch.transition_probabilities.resize(n_markers, 0.0);
    for marker in 1..n_markers {
        scratch.transition_probabilities[marker] = transition_probability(
            cm_positions[marker] - cm_positions[marker - 1],
            n_states,
            config.haplotype_count,
            config.recombination_intensity,
        );
    }
    scratch.forward_values.clear();
    scratch.forward_values.resize(forward_value_count, 0.0);
    scratch.forward.resize(n_states, 0.0);
    scratch.previous_forward.resize(n_states, 0.0);
    scratch.backward.resize(n_states, 0.0);
    scratch.state_posterior.resize(n_states, 0.0);
    scratch.emission_block.resize(n_states, 0.0);
    let mut allele_row_words = Vec::new();
    let mut missing_row_words = Vec::new();

    let forward_started = Instant::now();
    scratch.forward.fill(1.0 / n_states as f32);
    for marker in 0..n_markers {
        let marker_error = marker_errors[marker];
        let match_emission = (1.0 - marker_error) as f32;
        let mismatch_emission = marker_error.max(1.0e-12) as f32;
        let panel_marker = states.marker_indices[marker];
        let switch = if marker == 0 {
            0.0
        } else {
            scratch.transition_probabilities[marker]
        };
        let previous_sum = scratch.forward.iter().sum::<f32>();
        let observation = observations.allele(marker, 0);
        let fused = if let Some(observed) = observation {
            with_biallelic_row_words(
                &states.haplotypes,
                panel_marker,
                &mut allele_row_words,
                &mut missing_row_words,
                |allele_words, missing_words| {
                    if missing_words.iter().all(|&word| word == 0) {
                        transition_emit_biallelic_row(
                            &scratch.forward,
                            &mut scratch.previous_forward,
                            allele_words,
                            observed,
                            switch,
                            previous_sum,
                            match_emission,
                            mismatch_emission,
                            None,
                        );
                        true
                    } else {
                        false
                    }
                },
            )?
            .unwrap_or(false)
        } else {
            false
        };
        if !fused {
            let scale = 1.0 - switch;
            let shift = switch * previous_sum / n_states as f32;
            for state in 0..n_states {
                scratch.previous_forward[state] = (scale * scratch.forward[state] + shift)
                    * emission(
                        observation,
                        states.haplotypes.allele(panel_marker, state),
                        allele_counts[marker],
                        marker_error,
                    );
            }
        }
        normalize(&mut scratch.previous_forward);
        std::mem::swap(&mut scratch.forward, &mut scratch.previous_forward);
        let offset = marker * n_states;
        scratch.forward_values[offset..offset + n_states].copy_from_slice(&scratch.forward);
    }
    let forward_elapsed = forward_started.elapsed();

    let backward_started = Instant::now();
    scratch.backward.fill(1.0);
    let mut callback_elapsed = Duration::ZERO;
    for marker in (0..n_markers).rev() {
        let marker_error = marker_errors[marker];
        let match_emission = (1.0 - marker_error) as f32;
        let mismatch_emission = marker_error.max(1.0e-12) as f32;
        let panel_marker = states.marker_indices[marker];
        let observation = observations.allele(marker, 0);
        let emitted_from_packed = if let Some(observed) = observation {
            with_biallelic_row_words(
                &states.haplotypes,
                panel_marker,
                &mut allele_row_words,
                &mut missing_row_words,
                |allele_words, missing_words| {
                    if missing_words.iter().all(|&word| word == 0) {
                        for state in 0..n_states {
                            scratch.emission_block[state] = biallelic_observed_emission(
                                allele_words,
                                state,
                                observed,
                                match_emission,
                                mismatch_emission,
                            );
                        }
                        true
                    } else {
                        false
                    }
                },
            )?
            .unwrap_or(false)
        } else {
            false
        };
        if !emitted_from_packed {
            if observation.is_some() {
                for state in 0..n_states {
                    scratch.emission_block[state] = emission(
                        observation,
                        states.haplotypes.allele(panel_marker, state),
                        allele_counts[marker],
                        marker_error,
                    );
                }
            } else {
                scratch.emission_block.fill(1.0);
            }
        }
        let switch = if marker == 0 {
            0.0
        } else {
            scratch.transition_probabilities[marker]
        };
        let forward_offset = marker * n_states;
        let mut posterior_sum = 0.0f32;
        let mut emission_backward_sum = 0.0f32;
        for state in 0..n_states {
            let backward = scratch.backward[state];
            posterior_sum += scratch.forward_values[forward_offset + state] * backward;
            if marker > 0 {
                emission_backward_sum += scratch.emission_block[state] * backward;
            }
        }
        if posterior_sum <= f32::MIN_POSITIVE {
            posterior_sum = 1.0;
        }
        let update_backward = marker > 0;
        if update_backward {
            let scale = 1.0 - switch;
            let uniform = switch * emission_backward_sum / n_states as f32;
            posterior_and_backward_update(
                &scratch.forward_values[forward_offset..forward_offset + n_states],
                &mut scratch.backward,
                &scratch.emission_block,
                posterior_sum,
                scale,
                uniform,
                &mut scratch.state_posterior,
                true,
            );
        } else {
            posterior_and_backward_update(
                &scratch.forward_values[forward_offset..forward_offset + n_states],
                &mut scratch.backward,
                &scratch.emission_block,
                posterior_sum,
                1.0,
                0.0,
                &mut scratch.state_posterior,
                false,
            );
        };
        if profile_hmm {
            let callback_started = Instant::now();
            state_callback(marker, &scratch.state_posterior)?;
            callback_elapsed += callback_started.elapsed();
        } else {
            state_callback(marker, &scratch.state_posterior)?;
        }
        if update_backward {
            normalize(&mut scratch.backward);
        }
    }
    let backward_elapsed = backward_started.elapsed();
    if profile_hmm {
        eprintln!(
            "JANUSX_IMP_HMM_PROFILE dense=1 markers={n_markers} states={n_states} forward_s={:.6} callback_s={:.6} backward_s={:.6} total_s={:.6}",
            forward_elapsed.as_secs_f64(),
            callback_elapsed.as_secs_f64(),
            backward_elapsed.as_secs_f64(),
            profile_started.elapsed().as_secs_f64(),
        );
    }
    Ok(PosteriorWindow {
        allele_probabilities: Vec::new(),
        leave_one_out_allele_probabilities: Vec::new(),
        allele_offsets: Vec::new(),
        state_probabilities: Vec::new(),
        parameter_stats: HMMParameterStats::default(),
    })
}

pub(crate) fn forward_backward_with_diagnostics<F>(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    scratch: &mut HMMWorkspace,
    selected_markers: &[usize],
    diagnostic_callback: F,
) -> Result<PosteriorWindow, String>
where
    F: FnMut(HMMRecursionDiagnostic) -> Result<(), String>,
{
    forward_backward_impl(
        observations,
        states,
        cm_positions,
        config,
        scratch,
        None,
        false,
        false,
        false,
        None,
        |_, _| Ok(()),
        Some(selected_markers),
        diagnostic_callback,
    )
}

fn forward_backward_impl<F, D>(
    observations: &PackedHaplotypes,
    states: &CompositePanel,
    cm_positions: &[f64],
    config: &HMMConfig,
    scratch: &mut HMMWorkspace,
    marker_errors: Option<&[f64]>,
    collect_allele_probabilities: bool,
    emit_state_probabilities: bool,
    collect_parameter_stats: bool,
    selected_state_markers: Option<&[usize]>,
    mut state_callback: F,
    diagnostic_markers: Option<&[usize]>,
    mut diagnostic_callback: D,
) -> Result<PosteriorWindow, String>
where
    F: FnMut(usize, &[f32]) -> Result<(), String>,
    D: FnMut(HMMRecursionDiagnostic) -> Result<(), String>,
{
    let profile_hmm = detail_profiling_enabled();
    let profile_started = Instant::now();
    let n_markers = observations.n_markers;
    let n_states = states.haplotypes.n_haplotypes;
    if n_markers == 0 || n_states == 0 || observations.n_haplotypes != 1 {
        return Err("HMM expects one observation haplotype and positive dimensions".to_owned());
    }
    if states.marker_indices.len() != n_markers
        || states.allele_counts.len() != states.haplotypes.n_markers
        || states
            .marker_indices
            .iter()
            .any(|&marker| marker >= states.haplotypes.n_markers)
        || cm_positions.len() != n_markers
    {
        return Err("HMM marker mapping differs between observations, states, and map".to_owned());
    }
    if diagnostic_markers.is_some_and(|markers| markers.iter().any(|&marker| marker >= n_markers)) {
        return Err("HMM diagnostic marker is outside the observation window".to_owned());
    }
    if selected_state_markers.is_some_and(|markers| {
        markers.iter().any(|&marker| marker >= n_markers)
            || markers.windows(2).any(|pair| pair[0] >= pair[1])
    }) {
        return Err("HMM selected state markers must be ordered unique indices".to_owned());
    }
    if config.ne == 0
        || !config.recombination_intensity.is_finite()
        || config.recombination_intensity <= 0.0
        || config.haplotype_count == 0
        || !config.err.is_finite()
        || config.err < 0.0
        || config.err >= 0.5
        || marker_errors.is_some_and(|errors| {
            errors.len() != n_markers
                || errors
                    .iter()
                    .any(|error| !error.is_finite() || *error < 0.0 || *error > 0.5)
        })
    {
        return Err("HMM ne/err parameters are outside their valid range".to_owned());
    }
    if cm_positions.iter().any(|cm| !cm.is_finite())
        || cm_positions.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("HMM genetic positions must be finite and nondecreasing".to_owned());
    }
    let allele_counts = (0..n_markers)
        .map(|marker| {
            let panel_marker = states.marker_indices[marker];
            let declared = states.allele_counts[panel_marker].max(1);
            let observed = observations
                .allele(marker, 0)
                .map_or(1, |allele| usize::from(allele) + 1);
            declared.max(observed)
        })
        .collect::<Vec<_>>();
    let checkpoint_stride = checkpoint_stride(n_markers, n_states);
    let mut allele_offsets = Vec::new();
    let mut allele_probabilities = Vec::new();
    let mut leave_one_out_allele_probabilities = Vec::new();
    if collect_allele_probabilities {
        allele_offsets.reserve(n_markers.saturating_add(1));
        allele_offsets.push(0usize);
        for &count in &allele_counts {
            let next = allele_offsets
                .last()
                .copied()
                .unwrap()
                .checked_add(count)
                .ok_or_else(|| "HMM allele-posterior size overflow".to_owned())?;
            allele_offsets.push(next);
        }
        allele_probabilities.resize(*allele_offsets.last().unwrap_or(&0), 0.0f32);
        leave_one_out_allele_probabilities.resize(*allele_offsets.last().unwrap_or(&0), 0.0f32);
    }

    scratch.transition_probabilities.resize(n_markers, 0.0);
    for marker in 1..n_markers {
        scratch.transition_probabilities[marker] = transition_probability(
            cm_positions[marker] - cm_positions[marker - 1],
            n_states,
            config.haplotype_count,
            config.recombination_intensity,
        );
    }

    scratch.checkpoint_markers.clear();
    scratch.checkpoint_values.clear();
    scratch.forward.resize(n_states, 0.0);
    scratch.backward.resize(n_states, 0.0);
    scratch.previous_forward.resize(n_states, 0.0);
    let mut allele_row_words = Vec::new();
    let mut missing_row_words = Vec::new();
    scratch.forward.fill(1.0 / n_states as f32);
    let forward_started = Instant::now();
    for marker in 0..n_markers {
        let marker_error = marker_errors.map_or(config.err, |errors| errors[marker]);
        let match_emission = (1.0 - marker_error) as f32;
        let mismatch_emission = marker_error.max(1.0e-12) as f32;
        let panel_marker = states.marker_indices[marker];
        let switch = if marker > 0 {
            scratch.transition_probabilities[marker]
        } else {
            0.0
        };
        let previous_sum = scratch.forward.iter().sum::<f32>();
        let observation = observations.allele(marker, 0);
        let fused = if let Some(observed) = observation {
            with_biallelic_row_words(
                &states.haplotypes,
                panel_marker,
                &mut allele_row_words,
                &mut missing_row_words,
                |allele_words, missing_words| {
                    if missing_words.iter().all(|&word| word == 0) {
                        transition_emit_biallelic_row(
                            &scratch.forward,
                            &mut scratch.previous_forward,
                            allele_words,
                            observed,
                            switch,
                            previous_sum,
                            match_emission,
                            mismatch_emission,
                            None,
                        );
                        true
                    } else {
                        false
                    }
                },
            )?
            .unwrap_or(false)
        } else {
            false
        };
        if !fused {
            let scale = 1.0 - switch;
            let shift = switch * previous_sum / n_states as f32;
            let allele_count = allele_counts[marker];
            for state in 0..n_states {
                let state_allele = states.haplotypes.allele(panel_marker, state);
                let state_emission =
                    emission(observation, state_allele, allele_count, marker_error);
                scratch.previous_forward[state] =
                    (scale * scratch.forward[state] + shift) * state_emission;
            }
        }
        normalize(&mut scratch.previous_forward);
        std::mem::swap(&mut scratch.forward, &mut scratch.previous_forward);
        if marker % checkpoint_stride == 0 {
            scratch.checkpoint_markers.push(marker);
            scratch
                .checkpoint_values
                .extend_from_slice(&scratch.forward);
        }
    }
    let forward_elapsed = forward_started.elapsed();

    scratch.backward.fill(1.0);
    let mut parameter_stats = HMMParameterStats::default();
    let checkpoint_count = scratch.checkpoint_markers.len();
    let mut callback_elapsed = Duration::ZERO;
    let mut selected_cursor = selected_state_markers.map_or(0, <[usize]>::len);
    let mut next_selected_marker =
        selected_state_markers.and_then(|markers| markers.last().copied());
    let backward_started = Instant::now();
    for checkpoint_idx in (0..checkpoint_count).rev() {
        let start = scratch.checkpoint_markers[checkpoint_idx];
        let end = if checkpoint_idx + 1 < checkpoint_count {
            scratch.checkpoint_markers[checkpoint_idx + 1] - 1
        } else {
            n_markers - 1
        };
        let block_len = end - start + 1;
        let block_values_len = block_len
            .checked_mul(n_states)
            .ok_or_else(|| "HMM checkpoint segment size overflow".to_owned())?;
        scratch.forward.resize(block_values_len, 0.0);
        scratch.emission_block.resize(block_values_len, 0.0);
        let checkpoint_start = checkpoint_idx
            .checked_mul(n_states)
            .ok_or_else(|| "HMM checkpoint offset overflow".to_owned())?;
        let start_error = marker_errors.map_or(config.err, |errors| errors[start]);
        let start_match_emission = (1.0 - start_error) as f32;
        let start_mismatch_emission = start_error.max(1.0e-12) as f32;
        scratch.forward[..n_states].copy_from_slice(
            &scratch.checkpoint_values[checkpoint_start..checkpoint_start + n_states],
        );
        let start_panel_marker = states.marker_indices[start];
        let start_observation = observations.allele(start, 0);
        let start_emitted_from_packed = if let Some(observed) = start_observation {
            with_biallelic_row_words(
                &states.haplotypes,
                start_panel_marker,
                &mut allele_row_words,
                &mut missing_row_words,
                |allele_words, missing_words| {
                    if missing_words.iter().all(|&word| word == 0) {
                        for state in 0..n_states {
                            scratch.emission_block[state] = biallelic_observed_emission(
                                allele_words,
                                state,
                                observed,
                                start_match_emission,
                                start_mismatch_emission,
                            );
                        }
                        true
                    } else {
                        false
                    }
                },
            )?
            .unwrap_or(false)
        } else {
            false
        };
        if !start_emitted_from_packed {
            if start_observation.is_some() {
                for state in 0..n_states {
                    scratch.emission_block[state] = emission(
                        start_observation,
                        states.haplotypes.allele(start_panel_marker, state),
                        allele_counts[start],
                        start_error,
                    );
                }
            } else {
                scratch.emission_block[..n_states].fill(1.0);
            }
        }
        for marker in start + 1..=end {
            let marker_error = marker_errors.map_or(config.err, |errors| errors[marker]);
            let match_emission = (1.0 - marker_error) as f32;
            let mismatch_emission = marker_error.max(1.0e-12) as f32;
            let panel_marker = states.marker_indices[marker];
            let previous = (marker - start - 1) * n_states;
            let current = (marker - start) * n_states;
            let switch = scratch.transition_probabilities[marker];
            let previous_sum = scratch.forward[previous..previous + n_states]
                .iter()
                .sum::<f32>();
            let observation = observations.allele(marker, 0);
            let fused = if let Some(observed) = observation {
                with_biallelic_row_words(
                    &states.haplotypes,
                    panel_marker,
                    &mut allele_row_words,
                    &mut missing_row_words,
                    |allele_words, missing_words| {
                        if missing_words.iter().all(|&word| word == 0) {
                            let (rows_before_current, rows_from_current) =
                                scratch.forward.split_at_mut(current);
                            let previous_values =
                                &rows_before_current[previous..previous + n_states];
                            let current_values = &mut rows_from_current[..n_states];
                            transition_emit_biallelic_row(
                                previous_values,
                                current_values,
                                allele_words,
                                observed,
                                switch,
                                previous_sum,
                                match_emission,
                                mismatch_emission,
                                Some(&mut scratch.emission_block[current..current + n_states]),
                            );
                            true
                        } else {
                            false
                        }
                    },
                )?
                .unwrap_or(false)
            } else {
                false
            };
            if !fused {
                let scale = 1.0 - switch;
                let shift = switch * previous_sum / n_states as f32;
                let allele_count = allele_counts[marker];
                for state in 0..n_states {
                    let state_emission = emission(
                        observation,
                        states.haplotypes.allele(panel_marker, state),
                        allele_count,
                        marker_error,
                    );
                    scratch.emission_block[current + state] = state_emission;
                    scratch.forward[current + state] =
                        (scale * scratch.forward[previous + state] + shift) * state_emission;
                }
            }
            normalize(&mut scratch.forward[current..current + n_states]);
        }

        if collect_parameter_stats && start > 0 {
            let previous_checkpoint = checkpoint_idx - 1;
            let previous_start = scratch.checkpoint_markers[previous_checkpoint];
            let checkpoint_offset = previous_checkpoint
                .checked_mul(n_states)
                .ok_or_else(|| "HMM previous checkpoint offset overflow".to_owned())?;
            scratch.previous_forward.resize(n_states, 0.0);
            scratch.previous_forward.copy_from_slice(
                &scratch.checkpoint_values[checkpoint_offset..checkpoint_offset + n_states],
            );
            for previous_marker in previous_start + 1..start {
                let panel_marker = states.marker_indices[previous_marker];
                let switch = scratch.transition_probabilities[previous_marker];
                apply_transition(&mut scratch.previous_forward, switch);
                let observation = observations.allele(previous_marker, 0);
                let allele_count = allele_counts[previous_marker];
                let previous_error =
                    marker_errors.map_or(config.err, |errors| errors[previous_marker]);
                for state in 0..n_states {
                    scratch.previous_forward[state] *= emission(
                        observation,
                        states.haplotypes.allele(panel_marker, state),
                        allele_count,
                        previous_error,
                    );
                }
                normalize(&mut scratch.previous_forward);
            }
        }

        for marker in (start..=end).rev() {
            while next_selected_marker.is_some_and(|selected| selected > marker) {
                selected_cursor = selected_cursor.saturating_sub(1);
                next_selected_marker = selected_state_markers.and_then(|markers| {
                    selected_cursor
                        .checked_sub(1)
                        .and_then(|index| markers.get(index).copied())
                });
            }
            let emit_this_state = emit_state_probabilities
                && selected_state_markers.map_or(true, |_| next_selected_marker == Some(marker));
            let local = (marker - start) * n_states;
            let mut posterior_sum = 0.0f32;
            let mut emission_backward_sum = 0.0f32;
            let need_diagnostic =
                diagnostic_markers.is_some_and(|markers| markers.contains(&marker));
            let need_state_probabilities = emit_this_state
                || collect_allele_probabilities
                || collect_parameter_stats
                || need_diagnostic;
            for state in 0..n_states {
                let backward = scratch.backward[state];
                if need_state_probabilities {
                    posterior_sum += scratch.forward[local + state] * backward;
                }
                if marker > 0 {
                    emission_backward_sum += scratch.emission_block[local + state] * backward;
                }
            }
            if need_state_probabilities && posterior_sum <= f32::MIN_POSITIVE {
                posterior_sum = 1.0;
            }
            let switch = if marker == 0 {
                0.0
            } else {
                scratch.transition_probabilities[marker]
            };
            let uniform_component = switch * emission_backward_sum / n_states as f32;
            let panel_marker = states.marker_indices[marker];
            let observation = observations.allele(marker, 0);
            let allele_count = allele_counts[marker];
            if need_diagnostic {
                let emission_probabilities =
                    scratch.emission_block[local..local + n_states].to_vec();
                diagnostic_callback(HMMRecursionDiagnostic {
                    marker,
                    transition_probability: switch,
                    emission_probabilities,
                    forward_probabilities: scratch.forward[local..local + n_states].to_vec(),
                    backward_probabilities: scratch.backward.clone(),
                    posterior_normalization: posterior_sum,
                })?;
            }
            if emit_this_state {
                scratch.state_posterior.resize(n_states, 0.0);
            }
            let probabilities: &mut [f32] = if collect_allele_probabilities {
                let allele_start = allele_offsets[marker];
                let allele_end = allele_offsets[marker + 1];
                &mut allele_probabilities[allele_start..allele_end]
            } else {
                &mut []
            };
            let leave_one_out_probabilities: &mut [f32] = if collect_allele_probabilities {
                let allele_start = allele_offsets[marker];
                let allele_end = allele_offsets[marker + 1];
                &mut leave_one_out_allele_probabilities[allele_start..allele_end]
            } else {
                &mut []
            };
            let leave_one_out_sum = if collect_allele_probabilities {
                (0..n_states)
                    .map(|state| {
                        let state_mass = f64::from(scratch.forward[local + state])
                            * f64::from(scratch.backward[state]);
                        let marker_emission = f64::from(scratch.emission_block[local + state]);
                        state_mass / marker_emission
                    })
                    .sum::<f64>()
            } else {
                0.0
            };
            let has_leave_one_out_posterior =
                leave_one_out_sum.is_finite() && leave_one_out_sum > f64::MIN_POSITIVE;
            if marker > 0 && collect_parameter_stats && switch > 0.0 && n_states > 1 {
                let alpha_previous = if marker > start {
                    &scratch.forward[local - n_states..local]
                } else {
                    &scratch.previous_forward
                };
                let mut transition_emission_sum = 0.0f64;
                let mut no_switch_posterior_mass = 0.0f64;
                let no_switch_probability =
                    1.0 - f64::from(switch) + f64::from(switch) / n_states as f64;
                for state in 0..n_states {
                    let state_emission = f64::from(scratch.emission_block[local + state]);
                    let previous_probability = f64::from(alpha_previous[state]);
                    let transitioned_probability = (1.0 - f64::from(switch)) * previous_probability
                        + f64::from(switch) / n_states as f64;
                    transition_emission_sum += transitioned_probability * state_emission;
                    no_switch_posterior_mass += previous_probability
                        * no_switch_probability
                        * state_emission
                        * f64::from(scratch.backward[state]);
                }
                let posterior_mass = transition_emission_sum * f64::from(posterior_sum);
                if posterior_mass.is_finite() && posterior_mass > f64::MIN_POSITIVE {
                    let same_state_probability =
                        (no_switch_posterior_mass / posterior_mass).clamp(0.0, 1.0);
                    let expected_switch =
                        (1.0 - same_state_probability) * n_states as f64 / (n_states - 1) as f64;
                    if expected_switch.is_finite() && expected_switch > 0.0 {
                        parameter_stats.switch_probability_sum += expected_switch;
                        parameter_stats.switch_distance_cm_sum +=
                            cm_positions[marker] - cm_positions[marker - 1];
                    }
                }
            }
            if collect_parameter_stats && observation.is_some() {
                parameter_stats.mismatch_marker_count += 1;
            }
            if emit_this_state && !collect_allele_probabilities && !collect_parameter_stats {
                posterior_and_backward_update(
                    &scratch.forward[local..local + n_states],
                    &mut scratch.backward,
                    &scratch.emission_block[local..local + n_states],
                    posterior_sum,
                    1.0 - switch,
                    uniform_component,
                    &mut scratch.state_posterior,
                    marker > 0,
                );
            } else {
                for state in 0..n_states {
                    let (state_mass, probability) = if need_state_probabilities {
                        let state_mass = f64::from(scratch.forward[local + state])
                            * f64::from(scratch.backward[state]);
                        (state_mass, (state_mass / f64::from(posterior_sum)) as f32)
                    } else {
                        (0.0, 0.0)
                    };
                    // The imputation callback only needs state probabilities.
                    // Avoid decoding the panel allele here unless one of the
                    // marginal/statistics outputs actually consumes it.
                    let state_allele = if collect_allele_probabilities || collect_parameter_stats {
                        states.haplotypes.allele(panel_marker, state)
                    } else {
                        None
                    };
                    if emit_this_state {
                        scratch.state_posterior[state] = probability;
                    }
                    if collect_allele_probabilities {
                        let marker_emission = f64::from(scratch.emission_block[local + state]);
                        let leave_one_out_probability = if has_leave_one_out_posterior {
                            (state_mass / marker_emission / leave_one_out_sum) as f32
                        } else {
                            probability
                        };
                        if let Some(allele) = state_allele {
                            let allele_index = usize::from(allele);
                            probabilities[allele_index] += probability;
                            leave_one_out_probabilities[allele_index] += leave_one_out_probability;
                        } else {
                            let share = probability / allele_count as f32;
                            let leave_one_out_share =
                                leave_one_out_probability / allele_count as f32;
                            for value in probabilities.iter_mut() {
                                *value += share;
                            }
                            for value in leave_one_out_probabilities.iter_mut() {
                                *value += leave_one_out_share;
                            }
                        }
                    }
                    if collect_parameter_stats {
                        if let Some(observed) = observation {
                            let mismatch_probability = match state_allele {
                                Some(state_allele) => f64::from(state_allele != observed),
                                None => 1.0 - 1.0 / allele_count.max(1) as f64,
                            };
                            parameter_stats.mismatch_probability_sum +=
                                f64::from(probability) * mismatch_probability;
                        }
                    }
                    if marker > 0 {
                        scratch.backward[state] = (1.0 - switch)
                            * scratch.emission_block[local + state]
                            * scratch.backward[state]
                            + uniform_component;
                    }
                }
            }
            if collect_allele_probabilities {
                normalize(probabilities);
                normalize(leave_one_out_probabilities);
            }
            if emit_this_state {
                if profile_hmm {
                    let callback_started = Instant::now();
                    state_callback(marker, &scratch.state_posterior)?;
                    callback_elapsed += callback_started.elapsed();
                } else {
                    state_callback(marker, &scratch.state_posterior)?;
                }
            }
            if emit_this_state && selected_state_markers.is_some() {
                selected_cursor = selected_cursor.saturating_sub(1);
                next_selected_marker = selected_state_markers.and_then(|markers| {
                    selected_cursor
                        .checked_sub(1)
                        .and_then(|index| markers.get(index).copied())
                });
            }

            if marker > 0 {
                normalize(&mut scratch.backward);
            }
        }
    }
    let backward_elapsed = backward_started.elapsed();
    if profile_hmm {
        eprintln!(
            "JANUSX_IMP_HMM_PROFILE markers={n_markers} states={n_states} stride={checkpoint_stride} checkpoints={checkpoint_count} forward_s={:.6} callback_s={:.6} backward_s={:.6} total_s={:.6}",
            forward_elapsed.as_secs_f64(),
            callback_elapsed.as_secs_f64(),
            backward_elapsed.as_secs_f64(),
            profile_started.elapsed().as_secs_f64(),
        );
    }
    Ok(PosteriorWindow {
        allele_probabilities,
        leave_one_out_allele_probabilities,
        allele_offsets,
        // Keeping per-state posterior vectors for every marker defeats the
        // checkpoint memory bound. Callers need only the allele marginals.
        state_probabilities: Vec::new(),
        parameter_stats,
    })
}

#[inline]
fn transition_emit_biallelic_row(
    previous: &[f32],
    current: &mut [f32],
    allele_words: &[u64],
    observed: u16,
    switch: f32,
    previous_sum: f32,
    match_emission: f32,
    mismatch_emission: f32,
    emission_output: Option<&mut [f32]>,
) {
    debug_assert_eq!(previous.len(), current.len());
    debug_assert_eq!(allele_words.len(), previous.len().div_ceil(64));
    debug_assert!(emission_output
        .as_ref()
        .map_or(true, |emissions| emissions.len() == current.len()));
    if previous.is_empty() {
        return;
    }
    let scale = 1.0 - switch;
    let shift = switch * previous_sum / previous.len() as f32;

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on AArch64, which is the native macOS build used
        // for the dense-reference benchmark.
        unsafe {
            transition_emit_biallelic_neon(
                previous,
                current,
                allele_words,
                observed,
                scale,
                shift,
                match_emission,
                mismatch_emission,
                emission_output,
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        unsafe {
            transition_emit_biallelic_avx2(
                previous,
                current,
                allele_words,
                observed,
                scale,
                shift,
                match_emission,
                mismatch_emission,
                emission_output,
            );
        }
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    transition_emit_biallelic_scalar(
        previous,
        current,
        allele_words,
        observed,
        0,
        scale,
        shift,
        match_emission,
        mismatch_emission,
        emission_output,
    );
}

#[inline]
fn transition_emit_biallelic_scalar(
    previous: &[f32],
    current: &mut [f32],
    allele_words: &[u64],
    observed: u16,
    state_offset: usize,
    scale: f32,
    shift: f32,
    match_emission: f32,
    mismatch_emission: f32,
    mut emission_output: Option<&mut [f32]>,
) {
    for (offset, (&previous_probability, current_probability)) in
        previous.iter().zip(current).enumerate()
    {
        let state = state_offset + offset;
        let state_emission = biallelic_observed_emission(
            allele_words,
            state,
            observed,
            match_emission,
            mismatch_emission,
        );
        *current_probability = (scale * previous_probability + shift) * state_emission;
        if let Some(emissions) = emission_output.as_deref_mut() {
            emissions[state] = state_emission;
        }
    }
}

const fn build_match_masks_4() -> [[u32; 4]; 16] {
    let mut masks = [[0u32; 4]; 16];
    let mut bits = 0usize;
    while bits < masks.len() {
        let mut lane = 0usize;
        while lane < 4 {
            if bits & (1 << lane) != 0 {
                masks[bits][lane] = u32::MAX;
            }
            lane += 1;
        }
        bits += 1;
    }
    masks
}

const MATCH_MASKS_4: [[u32; 4]; 16] = build_match_masks_4();

#[cfg(target_arch = "x86_64")]
const fn build_match_masks_8() -> [[i32; 8]; 256] {
    let mut masks = [[0i32; 8]; 256];
    let mut bits = 0usize;
    while bits < masks.len() {
        let mut lane = 0usize;
        while lane < 8 {
            if bits & (1 << lane) != 0 {
                masks[bits][lane] = -1;
            }
            lane += 1;
        }
        bits += 1;
    }
    masks
}

#[cfg(target_arch = "x86_64")]
const MATCH_MASKS_8: [[i32; 8]; 256] = build_match_masks_8();

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn transition_emit_biallelic_neon(
    previous: &[f32],
    current: &mut [f32],
    allele_words: &[u64],
    observed: u16,
    scale: f32,
    shift: f32,
    match_emission: f32,
    mismatch_emission: f32,
    mut emission_output: Option<&mut [f32]>,
) {
    use std::arch::aarch64::*;

    let scale = vdupq_n_f32(scale);
    let shift = vdupq_n_f32(shift);
    let matches = vdupq_n_f32(match_emission);
    let mismatches = vdupq_n_f32(mismatch_emission);
    let vector_end = previous.len() / 4 * 4;
    let mut state = 0usize;
    while state < vector_end {
        let allele_word = allele_words[state >> 6];
        let lane_bits = ((allele_word >> (state & 63)) & 0x0f) as usize;
        let matching_bits = match observed {
            0 => !lane_bits & 0x0f,
            1 => lane_bits,
            _ => 0,
        };
        let match_mask = unsafe { vld1q_u32(MATCH_MASKS_4[matching_bits].as_ptr()) };
        let previous_values = unsafe { vld1q_f32(previous.as_ptr().add(state)) };
        let transitioned = vaddq_f32(vmulq_f32(previous_values, scale), shift);
        let state_emissions = vbslq_f32(match_mask, matches, mismatches);
        let updated = vmulq_f32(transitioned, state_emissions);
        unsafe { vst1q_f32(current.as_mut_ptr().add(state), updated) };
        if let Some(emissions) = emission_output.as_deref_mut() {
            unsafe { vst1q_f32(emissions.as_mut_ptr().add(state), state_emissions) };
        }
        state += 4;
    }
    if state < previous.len() {
        transition_emit_biallelic_scalar(
            &previous[state..],
            &mut current[state..],
            allele_words,
            observed,
            state,
            vgetq_lane_f32(scale, 0),
            vgetq_lane_f32(shift, 0),
            match_emission,
            mismatch_emission,
            emission_output,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transition_emit_biallelic_avx2(
    previous: &[f32],
    current: &mut [f32],
    allele_words: &[u64],
    observed: u16,
    scale: f32,
    shift: f32,
    match_emission: f32,
    mismatch_emission: f32,
    mut emission_output: Option<&mut [f32]>,
) {
    use std::arch::x86_64::*;

    let scale = _mm256_set1_ps(scale);
    let shift = _mm256_set1_ps(shift);
    let matches = _mm256_set1_ps(match_emission);
    let mismatches = _mm256_set1_ps(mismatch_emission);
    let vector_end = previous.len() / 8 * 8;
    let mut state = 0usize;
    while state < vector_end {
        let allele_word = allele_words[state >> 6];
        let lane_bits = ((allele_word >> (state & 63)) & 0xff) as usize;
        let matching_bits = match observed {
            0 => !lane_bits & 0xff,
            1 => lane_bits,
            _ => 0,
        };
        let match_mask =
            unsafe { _mm256_loadu_si256(MATCH_MASKS_8[matching_bits].as_ptr().cast::<__m256i>()) };
        let previous_values = unsafe { _mm256_loadu_ps(previous.as_ptr().add(state)) };
        let transitioned = _mm256_add_ps(_mm256_mul_ps(previous_values, scale), shift);
        let state_emissions =
            _mm256_blendv_ps(mismatches, matches, _mm256_castsi256_ps(match_mask));
        let updated = _mm256_mul_ps(transitioned, state_emissions);
        unsafe { _mm256_storeu_ps(current.as_mut_ptr().add(state), updated) };
        if let Some(emissions) = emission_output.as_deref_mut() {
            unsafe { _mm256_storeu_ps(emissions.as_mut_ptr().add(state), state_emissions) };
        }
        state += 8;
    }
    if state < previous.len() {
        transition_emit_biallelic_scalar(
            &previous[state..],
            &mut current[state..],
            allele_words,
            observed,
            state,
            _mm256_cvtss_f32(scale),
            _mm256_cvtss_f32(shift),
            match_emission,
            mismatch_emission,
            emission_output,
        );
    }
}

fn transition_probability(
    distance_cm: f64,
    state_count: usize,
    haplotype_count: usize,
    recombination_intensity: f32,
) -> f32 {
    if distance_cm <= 0.0
        || state_count <= 1
        || haplotype_count == 0
        || recombination_intensity <= 0.0
    {
        return 0.0;
    }
    // Beagle stores recombination intensity and adjacent genetic distances as
    // floats, then evaluates -expm1(-intensity * distance) in double precision.
    let distance_cm = distance_cm as f32;
    let exponent = -f64::from(recombination_intensity) * f64::from(distance_cm);
    (-exponent.exp_m1()).clamp(0.0, 1.0) as f32
}

fn apply_transition(probabilities: &mut [f32], switch: f32) {
    let n_states = probabilities.len();
    let sum = probabilities.iter().sum::<f32>();
    for probability in probabilities {
        *probability = (1.0 - switch) * *probability + switch * sum / n_states as f32;
    }
}

#[inline(always)]
fn biallelic_observed_emission(
    allele_words: &[u64],
    state: usize,
    observed: u16,
    match_emission: f32,
    mismatch_emission: f32,
) -> f32 {
    let allele = ((allele_words[state >> 6] >> (state & 63)) & 1) as u16;
    if allele == observed {
        match_emission
    } else {
        mismatch_emission
    }
}

fn copy_biallelic_row_words(
    panel: &PackedHaplotypes,
    marker: usize,
    allele_words: &mut [u64],
    missing_words: &mut [u64],
) -> Result<(), String> {
    let expected_words = panel.n_haplotypes.div_ceil(64);
    if panel.bits_per_allele != 1
        || allele_words.len() != expected_words
        || missing_words.len() != expected_words
    {
        return Err("packed biallelic row dimensions are inconsistent".to_owned());
    }
    for (word, first_state) in (0..panel.n_haplotypes).step_by(64).enumerate() {
        let (alleles, missing) = panel
            .biallelic_marker_word(marker, first_state)
            .ok_or_else(|| "packed biallelic marker row is unavailable".to_owned())?;
        allele_words[word] = alleles;
        missing_words[word] = missing;
    }
    Ok(())
}

fn with_biallelic_row_words<R>(
    panel: &PackedHaplotypes,
    marker: usize,
    allele_scratch: &mut Vec<u64>,
    missing_scratch: &mut Vec<u64>,
    f: impl FnOnce(&[u64], &[u64]) -> R,
) -> Result<Option<R>, String> {
    if let Some((allele_words, missing_words)) = panel.biallelic_marker_planes(marker) {
        return Ok(Some(f(allele_words, missing_words)));
    }
    if panel.biallelic_marker_word(marker, 0).is_none() {
        return Ok(None);
    }
    let words = panel.n_haplotypes.div_ceil(64);
    allele_scratch.resize(words, 0);
    missing_scratch.resize(words, 0);
    copy_biallelic_row_words(panel, marker, allele_scratch, missing_scratch)?;
    Ok(Some(f(allele_scratch, missing_scratch)))
}

fn emission(observation: Option<u16>, state: Option<u16>, allele_count: usize, err: f64) -> f32 {
    let Some(observed) = observation else {
        return 1.0;
    };
    let Some(state) = state else {
        return 1.0 / allele_count.max(1) as f32;
    };
    if observed == state {
        (1.0 - err) as f32
    } else {
        // Beagle's phasing HMM emits on a binary match/mismatch indicator,
        // including at multiallelic markers: all nonmatching alleles share
        // the same pMismatch probability.
        err.max(1.0e-12) as f32
    }
}

/// Certify the same f32 reciprocal as an ordered f64 sum without its serial
/// dependency chain. Uncertain rounding boundaries retain the ordered path.
fn guarded_normalization_reciprocal(values: &[f32]) -> Option<f32> {
    match normalization_chain_count(values.len()) {
        4 => guarded_normalization_reciprocal_with_chains::<4>(values),
        8 => guarded_normalization_reciprocal_with_chains::<8>(values),
        16 => guarded_normalization_reciprocal_with_chains::<16>(values),
        _ => None,
    }
}

// Eight independent f64 accumulators fit the x86_64 vector register budget;
// the smaller 4-chain form wins below the measured 256-state crossover.
#[cfg(target_arch = "x86_64")]
#[inline]
const fn normalization_chain_count(state_count: usize) -> usize {
    if state_count < 256 {
        4
    } else {
        8
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
const fn normalization_chain_count(_state_count: usize) -> usize {
    16
}

fn guarded_normalization_reciprocal_with_chains<const CHAINS: usize>(
    values: &[f32],
) -> Option<f32> {
    if CHAINS == 0 || values.len() < 64 || values.len() > (1usize << 30) {
        return None;
    }
    let mut sums = [0.0f64; CHAINS];
    let mut largest_bits = [0u32; CHAINS];
    let mut chunks = values.chunks_exact(CHAINS);
    for row in &mut chunks {
        for lane in 0..CHAINS {
            sums[lane] += f64::from(row[lane]);
            largest_bits[lane] = largest_bits[lane].max(row[lane].to_bits());
        }
    }
    let mut sum = sums.iter().sum::<f64>();
    let mut largest_bits = *largest_bits.iter().max().unwrap();
    for &value in chunks.remainder() {
        sum += f64::from(value);
        largest_bits = largest_bits.max(value.to_bits());
    }
    // Unsigned bit ordering rejects every negative value (including -0),
    // infinity and NaN. Nonnegative finite f32 -> f64 conversion is exact.
    if largest_bits >= 0x7f80_0000 || sum <= 0.0 {
        return None;
    }
    // For n <= 2^30, both summation orders have relative error <= gamma_n
    // with u=2^-53. This wider radius (16*n*u) also covers rounding while
    // forming the bounds. All positive f32 terms/sums are normal in f64.
    let radius = 8.0 * f64::EPSILON * values.len() as f64;
    let lower = (1.0 / (sum * (1.0 + radius))) as f32;
    let upper = (1.0 / (sum * (1.0 - radius))) as f32;
    (lower.to_bits() == upper.to_bits()).then_some(lower)
}

fn normalize(values: &mut [f32]) {
    let reciprocal = guarded_normalization_reciprocal(values).or_else(|| {
        let sum = values.iter().map(|value| *value as f64).sum::<f64>();
        (sum.is_finite() && sum > f64::MIN_POSITIVE).then(|| (1.0 / sum) as f32)
    });
    if let Some(reciprocal) = reciprocal {
        for value in values {
            *value *= reciprocal;
        }
    } else if !values.is_empty() {
        values.fill(1.0 / values.len() as f32);
    }
}

fn posterior_and_backward_update(
    forward: &[f32],
    backward: &mut [f32],
    emissions: &[f32],
    posterior_sum: f32,
    scale: f32,
    uniform_component: f32,
    state_posterior: &mut [f32],
    update_backward: bool,
) {
    debug_assert_eq!(forward.len(), backward.len());
    debug_assert_eq!(forward.len(), emissions.len());
    debug_assert_eq!(forward.len(), state_posterior.len());

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        unsafe {
            posterior_and_backward_update_avx2(
                forward,
                backward,
                emissions,
                posterior_sum,
                scale,
                uniform_component,
                state_posterior,
                update_backward,
            );
        }
        return;
    }

    let denominator = f64::from(posterior_sum);
    for state in 0..forward.len() {
        let state_mass = f64::from(forward[state]) * f64::from(backward[state]);
        state_posterior[state] = (state_mass / denominator) as f32;
        if update_backward {
            backward[state] = scale * emissions[state] * backward[state] + uniform_component;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn posterior_and_backward_update_avx2(
    forward: &[f32],
    backward: &mut [f32],
    emissions: &[f32],
    posterior_sum: f32,
    scale: f32,
    uniform_component: f32,
    state_posterior: &mut [f32],
    update_backward: bool,
) {
    use std::arch::x86_64::*;

    unsafe {
        let denominator = _mm256_set1_pd(f64::from(posterior_sum));
        let scale = _mm256_set1_ps(scale);
        let uniform = _mm256_set1_ps(uniform_component);
        let vector_end = forward.len() / 8 * 8;
        let mut state = 0usize;
        while state < vector_end {
            let backward_values = _mm256_loadu_ps(backward.as_ptr().add(state));

            let forward_low = _mm256_cvtps_pd(_mm_loadu_ps(forward.as_ptr().add(state)));
            let backward_low = _mm256_cvtps_pd(_mm_loadu_ps(backward.as_ptr().add(state)));
            let probabilities_low = _mm256_cvtpd_ps(_mm256_div_pd(
                _mm256_mul_pd(forward_low, backward_low),
                denominator,
            ));

            let forward_high = _mm256_cvtps_pd(_mm_loadu_ps(forward.as_ptr().add(state + 4)));
            let backward_high = _mm256_cvtps_pd(_mm_loadu_ps(backward.as_ptr().add(state + 4)));
            let probabilities_high = _mm256_cvtpd_ps(_mm256_div_pd(
                _mm256_mul_pd(forward_high, backward_high),
                denominator,
            ));

            let probabilities = _mm256_insertf128_ps::<1>(
                _mm256_castps128_ps256(probabilities_low),
                probabilities_high,
            );
            _mm256_storeu_ps(state_posterior.as_mut_ptr().add(state), probabilities);

            if update_backward {
                let emission_values = _mm256_loadu_ps(emissions.as_ptr().add(state));
                let transitioned =
                    _mm256_mul_ps(_mm256_mul_ps(scale, emission_values), backward_values);
                _mm256_storeu_ps(
                    backward.as_mut_ptr().add(state),
                    _mm256_add_ps(transitioned, uniform),
                );
            }
            state += 8;
        }
        if state < forward.len() {
            let denominator = f64::from(posterior_sum);
            let scale = _mm256_cvtss_f32(scale);
            let uniform = _mm256_cvtss_f32(uniform);
            for state in state..forward.len() {
                let state_mass = f64::from(forward[state]) * f64::from(backward[state]);
                state_posterior[state] = (state_mass / denominator) as f32;
                if update_backward {
                    backward[state] = scale * emissions[state] * backward[state] + uniform;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        biallelic_observed_emission, checkpoint_stride, emission, forward_backward,
        forward_backward_dense_with_state_callback_and_errors, forward_backward_for_em,
        forward_backward_for_em_pair_with_transition_data,
        forward_backward_for_em_single_observation_with_transition_data,
        forward_backward_for_em_with_transition_data, forward_backward_with_diagnostics,
        forward_backward_with_selected_state_callback,
        forward_backward_with_selected_state_callbacks_pair,
        forward_backward_with_selected_state_callbacks_pair_mismatch,
        forward_backward_with_state_callback_and_errors, normalize, posterior_and_backward_update,
        prepare_em_transition_data, transition_emit_biallelic_row, transition_probability,
        EmMismatchCache, CHECKPOINT_STRIDE,
    };
    use crate::imp::types::{
        CompositePanel, HMMConfig, HMMParameterStats, HMMWorkspace, PackedHaplotypes,
    };

    #[test]
    fn adaptive_checkpoint_stride_balances_recomputation_and_memory() {
        assert_eq!(checkpoint_stride(1, 1_600), 1);
        assert_eq!(checkpoint_stride(202, 1_600), 64);

        let (n_markers, n_states) = (4_000_000usize, 1_600usize);
        let stride = checkpoint_stride(n_markers, n_states);
        assert!(stride > 64);
        assert!(
            n_markers.div_ceil(stride) * n_states <= 4 * 1024 * 1024,
            "stride={stride}"
        );
    }

    #[test]
    fn marker_error_override_controls_single_marker_state_posterior() {
        let mut observations = PackedHaplotypes::empty(1, 1, 1).unwrap();
        observations.set_allele(0, 0, Some(0)).unwrap();
        let mut haplotypes = PackedHaplotypes::empty(2, 1, 1).unwrap();
        haplotypes.set_allele(0, 0, Some(0)).unwrap();
        haplotypes.set_allele(0, 1, Some(1)).unwrap();
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![0, 1],
            marker_indices: vec![0],
            allele_counts: vec![2],
        };
        let config = HMMConfig {
            ne: 500,
            recombination_intensity: (0.04_f32 * 500.0) / 2.0,
            err: 0.01,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 100_000.0,
            phase_states: 2,
            haplotype_count: 2,
        };
        let mut posterior = Vec::new();

        forward_backward_with_state_callback_and_errors(
            &observations,
            &panel,
            &[0.0],
            &config,
            &[0.2],
            &mut HMMWorkspace::default(),
            |_, probabilities| {
                posterior = probabilities.to_vec();
                Ok(())
            },
        )
        .unwrap();

        assert!((posterior[0] - 0.8).abs() < 1.0e-6, "{posterior:?}");
        assert!((posterior[1] - 0.2).abs() < 1.0e-6, "{posterior:?}");
    }

    #[test]
    fn dense_cluster_hmm_matches_checkpointed_variable_error_recursion() {
        let mut observations = PackedHaplotypes::empty(1, 3, 1).unwrap();
        for (marker, allele) in [0, 1, 1].into_iter().enumerate() {
            observations.set_allele(marker, 0, Some(allele)).unwrap();
        }
        let mut haplotypes = PackedHaplotypes::empty(4, 3, 1).unwrap();
        let alleles = [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]];
        for (state, row) in alleles.into_iter().enumerate() {
            for (marker, allele) in row.into_iter().enumerate() {
                haplotypes.set_allele(marker, state, Some(allele)).unwrap();
            }
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![0, 1, 2, 3],
            marker_indices: vec![0, 1, 2],
            allele_counts: vec![2, 2, 2],
        };
        let config = HMMConfig {
            ne: 500,
            recombination_intensity: (0.04_f32 * 500.0) / 4.0,
            err: 0.01,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 100_000.0,
            phase_states: 4,
            haplotype_count: 4,
        };
        let errors = [0.05, 0.15, 0.02];
        let cm_positions = [0.0, 0.1, 0.3];
        let mut checkpointed = vec![Vec::new(); cm_positions.len()];
        let mut dense = vec![Vec::new(); cm_positions.len()];

        forward_backward_with_state_callback_and_errors(
            &observations,
            &panel,
            &cm_positions,
            &config,
            &errors,
            &mut HMMWorkspace::default(),
            |marker, posterior| {
                checkpointed[marker] = posterior.to_vec();
                Ok(())
            },
        )
        .unwrap();
        forward_backward_dense_with_state_callback_and_errors(
            &observations,
            &panel,
            &cm_positions,
            &config,
            &errors,
            &mut HMMWorkspace::default(),
            |marker, posterior| {
                dense[marker] = posterior.to_vec();
                Ok(())
            },
        )
        .unwrap();

        for (marker, (expected, observed)) in checkpointed.iter().zip(&dense).enumerate() {
            for (state, (expected, observed)) in expected.iter().zip(observed).enumerate() {
                assert!(
                    (expected - observed).abs() < 2.0e-6,
                    "marker={marker} state={state} expected={expected} observed={observed}"
                );
            }
        }
    }

    #[test]
    fn selected_state_callback_matches_dense_rows() {
        let mut observations = PackedHaplotypes::empty(1, 3, 1).unwrap();
        for (marker, allele) in [0, 1, 1].into_iter().enumerate() {
            observations.set_allele(marker, 0, Some(allele)).unwrap();
        }
        let mut haplotypes = PackedHaplotypes::empty(4, 3, 1).unwrap();
        let alleles = [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]];
        for (state, row) in alleles.into_iter().enumerate() {
            for (marker, allele) in row.into_iter().enumerate() {
                haplotypes.set_allele(marker, state, Some(allele)).unwrap();
            }
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![0, 1, 2, 3],
            marker_indices: vec![0, 1, 2],
            allele_counts: vec![2, 2, 2],
        };
        let config = HMMConfig {
            ne: 500,
            recombination_intensity: (0.04_f32 * 500.0) / 4.0,
            err: 0.01,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 1.0,
            phase_states: 4,
            haplotype_count: 4,
        };
        let cm_positions = [0.0, 0.1, 0.3];
        let mut dense = vec![Vec::new(); cm_positions.len()];
        forward_backward_with_state_callback_and_errors(
            &observations,
            &panel,
            &cm_positions,
            &config,
            &[0.01, 0.01, 0.01],
            &mut HMMWorkspace::default(),
            |marker, posterior| {
                dense[marker] = posterior.to_vec();
                Ok(())
            },
        )
        .unwrap();

        let mut selected = Vec::new();
        forward_backward_with_selected_state_callback(
            &observations,
            &panel,
            &cm_positions,
            &config,
            &mut HMMWorkspace::default(),
            &[1],
            |marker, posterior| {
                selected.push((marker, posterior.to_vec()));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].0, 1);
        for (&selected, &dense) in selected[0].1.iter().zip(&dense[1]) {
            assert!(
                (selected - dense).abs() <= 2.0e-6,
                "selected={selected} dense={dense}"
            );
        }
    }

    #[test]
    fn paired_selected_state_callback_matches_independent_haplotypes_bitwise() {
        let n_markers = 4;
        let left_alleles = [
            [0u16, 0, 1, 1, 1, 0],
            [1u16, 0, 1, 0, 1, 1],
            [0u16, 1, 1, 0, 0, 1],
            [1u16, 1, 0, 0, 1, 0],
        ];
        let right_alleles = [
            [1u16, 0, 0, 1, 0],
            [0u16, 1, 0, 1, 1],
            [1u16, 0, 1, 0, 1],
            [0u16, 1, 1, 0, 0],
        ];
        let mut observations0 = PackedHaplotypes::empty(1, n_markers, 1).unwrap();
        let mut observations1 = PackedHaplotypes::empty(1, n_markers, 1).unwrap();
        for (marker, allele) in [Some(0), Some(1), None, Some(1)].into_iter().enumerate() {
            observations0.set_allele(marker, 0, allele).unwrap();
        }
        for (marker, allele) in [Some(1), Some(0), Some(1), None].into_iter().enumerate() {
            observations1.set_allele(marker, 0, allele).unwrap();
        }
        let mut haplotypes0 = PackedHaplotypes::empty(left_alleles[0].len(), n_markers, 1).unwrap();
        let mut haplotypes1 =
            PackedHaplotypes::empty(right_alleles[0].len(), n_markers, 1).unwrap();
        for (marker, alleles) in left_alleles.into_iter().enumerate() {
            for (state, allele) in alleles.into_iter().enumerate() {
                haplotypes0.set_allele(marker, state, Some(allele)).unwrap();
            }
        }
        for (marker, alleles) in right_alleles.into_iter().enumerate() {
            for (state, allele) in alleles.into_iter().enumerate() {
                haplotypes1.set_allele(marker, state, Some(allele)).unwrap();
            }
        }
        haplotypes0.mark_complete();
        haplotypes1.mark_complete();
        let panel0 = CompositePanel {
            haplotypes: haplotypes0,
            source_samples: vec![0; left_alleles[0].len()],
            marker_indices: (0..n_markers).collect(),
            allele_counts: vec![2; n_markers],
        };
        let panel1 = CompositePanel {
            haplotypes: haplotypes1,
            source_samples: vec![0; right_alleles[0].len()],
            marker_indices: (0..n_markers).collect(),
            allele_counts: vec![2; n_markers],
        };
        let config = HMMConfig {
            ne: 500,
            recombination_intensity: (0.04_f32 * 500.0) / 11.0,
            err: 0.01,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 1.0,
            phase_states: 6,
            haplotype_count: 11,
        };
        let cm_positions = [0.0, 0.1, 0.25, 0.4];
        let selected_markers = [1, 3];
        let mut expected0 = Vec::new();
        let mut expected1 = Vec::new();
        forward_backward_with_selected_state_callback(
            &observations0,
            &panel0,
            &cm_positions,
            &config,
            &mut HMMWorkspace::default(),
            &selected_markers,
            |marker, posterior| {
                expected0.push((marker, posterior.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        forward_backward_with_selected_state_callback(
            &observations1,
            &panel1,
            &cm_positions,
            &config,
            &mut HMMWorkspace::default(),
            &selected_markers,
            |marker, posterior| {
                expected1.push((marker, posterior.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        let mut observed = Vec::new();
        forward_backward_with_selected_state_callbacks_pair(
            &observations0,
            &panel0,
            &observations1,
            &panel1,
            &cm_positions,
            &config,
            &mut HMMWorkspace::default(),
            &mut HMMWorkspace::default(),
            &selected_markers,
            |marker, posterior0, posterior1| {
                observed.push((marker, posterior0.to_vec(), posterior1.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(observed.len(), selected_markers.len());
        for (selected_index, ((marker, posterior0, posterior1), (expected_marker, expected0))) in
            observed.iter().zip(&expected0).enumerate()
        {
            assert_eq!(marker, expected_marker);
            assert_eq!(
                posterior0
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected0
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                posterior1
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected1[selected_index]
                    .1
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        let mut mismatch = Vec::new();
        forward_backward_with_selected_state_callbacks_pair_mismatch(
            &observations0,
            &panel0,
            &observations1,
            &panel1,
            &cm_positions,
            &config,
            &mut HMMWorkspace::default(),
            &mut HMMWorkspace::default(),
            &selected_markers,
            |marker, posterior0, posterior1| {
                mismatch.push((marker, posterior0.to_vec(), posterior1.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(observed.len(), mismatch.len());
        for ((marker, posterior0, posterior1), (mismatch_marker, mismatch0, mismatch1)) in
            observed.iter().zip(&mismatch)
        {
            assert_eq!(marker, mismatch_marker);
            assert_eq!(
                posterior0
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                mismatch0
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                posterior1
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                mismatch1
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn state_major_emission_fast_path_requires_a_zero_missing_plane() {
        let mut complete = PackedHaplotypes::empty_state_major(3, 4, 1).unwrap();
        for marker in 0..4 {
            for state in 0..3 {
                complete
                    .set_allele(marker, state, Some(((marker + state) & 1) as u16))
                    .unwrap();
            }
        }
        assert!(super::has_complete_state_major_biallelic_rows(&complete));

        complete.set_allele(1, 2, None).unwrap();
        assert!(!super::has_complete_state_major_biallelic_rows(&complete));

        let marker_major = PackedHaplotypes::empty(3, 4, 1).unwrap();
        assert!(!super::has_complete_state_major_biallelic_rows(
            &marker_major
        ));
        let multiallelic = PackedHaplotypes::empty_state_major(3, 4, 2).unwrap();
        assert!(!super::has_complete_state_major_biallelic_rows(
            &multiallelic
        ));
    }

    #[test]
    fn state_major_emission_fast_path_matches_scalar_oracle_across_word_boundaries() {
        let n_markers = 67usize;
        let panel_markers = 71usize;
        let marker_indices = (4..4 + n_markers).rev().collect::<Vec<_>>();
        let selected_markers = [0usize, 1, 33, 66];
        let cm_positions = (0..n_markers)
            .map(|marker| marker as f64 * 0.007)
            .collect::<Vec<_>>();

        for state_count in [63usize, 64, 65] {
            let mut observations0 = PackedHaplotypes::empty(1, n_markers, 1).unwrap();
            let mut observations1 = PackedHaplotypes::empty(1, n_markers, 1).unwrap();
            for marker in 0..n_markers {
                observations0
                    .set_allele(
                        marker,
                        0,
                        (marker != 1).then_some(((marker * 7 + 3) & 1) as u16),
                    )
                    .unwrap();
                observations1
                    .set_allele(
                        marker,
                        0,
                        (marker != 34).then_some(((marker * 11 + 5) & 1) as u16),
                    )
                    .unwrap();
            }
            let mut haplotypes0 =
                PackedHaplotypes::empty_state_major(state_count, panel_markers, 1).unwrap();
            let mut haplotypes1 =
                PackedHaplotypes::empty_state_major(state_count, panel_markers, 1).unwrap();
            for marker in 0..panel_markers {
                for state in 0..state_count {
                    haplotypes0
                        .set_allele(
                            marker,
                            state,
                            Some(((marker * 17 + state * 13 + 3) % 7 < 3) as u16),
                        )
                        .unwrap();
                    haplotypes1
                        .set_allele(
                            marker,
                            state,
                            Some(((marker * 19 + state * 5 + 1) % 11 < 5) as u16),
                        )
                        .unwrap();
                }
            }
            haplotypes0.mark_complete();
            haplotypes1.mark_complete();
            assert!(super::has_complete_state_major_biallelic_rows(&haplotypes0));
            assert!(super::has_complete_state_major_biallelic_rows(&haplotypes1));
            let panel0 = CompositePanel {
                haplotypes: haplotypes0,
                source_samples: vec![0; state_count],
                marker_indices: marker_indices.clone(),
                allele_counts: vec![2; panel_markers],
            };
            let panel1 = CompositePanel {
                haplotypes: haplotypes1,
                source_samples: vec![0; state_count],
                marker_indices: marker_indices.clone(),
                allele_counts: vec![2; panel_markers],
            };
            let config = HMMConfig {
                ne: 500,
                recombination_intensity: (0.04_f32 * 500.0) / (state_count * 2) as f32,
                err: 0.013,
                em: false,
                burnin: 1,
                iterations: 1,
                initial_lr: 1.0,
                phase_states: state_count,
                haplotype_count: state_count * 2,
            };

            let mut scalar0 = Vec::new();
            forward_backward_with_selected_state_callback(
                &observations0,
                &panel0,
                &cm_positions,
                &config,
                &mut HMMWorkspace::default(),
                &selected_markers,
                |marker, probabilities| {
                    scalar0.push((marker, probabilities.to_vec()));
                    Ok(())
                },
            )
            .unwrap();
            let mut scalar1 = Vec::new();
            forward_backward_with_selected_state_callback(
                &observations1,
                &panel1,
                &cm_positions,
                &config,
                &mut HMMWorkspace::default(),
                &selected_markers,
                |marker, probabilities| {
                    scalar1.push((marker, probabilities.to_vec()));
                    Ok(())
                },
            )
            .unwrap();
            let mut paired = Vec::new();
            forward_backward_with_selected_state_callbacks_pair(
                &observations0,
                &panel0,
                &observations1,
                &panel1,
                &cm_positions,
                &config,
                &mut HMMWorkspace::default(),
                &mut HMMWorkspace::default(),
                &selected_markers,
                |marker, probabilities0, probabilities1| {
                    paired.push((marker, probabilities0.to_vec(), probabilities1.to_vec()));
                    Ok(())
                },
            )
            .unwrap();

            assert_eq!(paired.len(), selected_markers.len());
            for (selected_index, (marker, probabilities0, probabilities1)) in
                paired.iter().enumerate()
            {
                assert_eq!(*marker, scalar0[selected_index].0);
                assert_eq!(*marker, scalar1[selected_index].0);
                assert_eq!(
                    probabilities0
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    scalar0[selected_index]
                        .1
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    probabilities1
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    scalar1[selected_index]
                        .1
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn selected_pair_mismatch_matrix_matches_production_posteriors_bitwise() {
        fn assert_bitwise(
            observations0: &PackedHaplotypes,
            panel0: &CompositePanel,
            observations1: &PackedHaplotypes,
            panel1: &CompositePanel,
            cm_positions: &[f64],
            selected_markers: &[usize],
            config: &HMMConfig,
        ) {
            let mut production = Vec::new();
            forward_backward_with_selected_state_callbacks_pair(
                observations0,
                panel0,
                observations1,
                panel1,
                cm_positions,
                config,
                &mut HMMWorkspace::default(),
                &mut HMMWorkspace::default(),
                selected_markers,
                |marker, probabilities0, probabilities1| {
                    production.push((marker, probabilities0.to_vec(), probabilities1.to_vec()));
                    Ok(())
                },
            )
            .unwrap();
            let mut mismatch = Vec::new();
            let matrix_bytes = forward_backward_with_selected_state_callbacks_pair_mismatch(
                observations0,
                panel0,
                observations1,
                panel1,
                cm_positions,
                config,
                &mut HMMWorkspace::default(),
                &mut HMMWorkspace::default(),
                selected_markers,
                |marker, probabilities0, probabilities1| {
                    mismatch.push((marker, probabilities0.to_vec(), probabilities1.to_vec()));
                    Ok(())
                },
            )
            .unwrap();
            assert!(matrix_bytes > 0);
            assert_eq!(production.len(), mismatch.len());
            for (production, mismatch) in production.iter().zip(&mismatch) {
                assert_eq!(production.0, mismatch.0);
                assert_eq!(
                    production
                        .1
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    mismatch
                        .1
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    production
                        .2
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    mismatch
                        .2
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }

        let config = HMMConfig {
            ne: 500,
            recombination_intensity: (0.04_f32 * 500.0) / 11.0,
            err: 0.013,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 1.0,
            phase_states: 6,
            haplotype_count: 11,
        };

        let n_markers = 5;
        let mut observations0 = PackedHaplotypes::empty(1, n_markers, 2).unwrap();
        let mut observations1 = PackedHaplotypes::empty(1, n_markers, 2).unwrap();
        for (marker, allele) in [Some(1), None, Some(2), Some(0), Some(1)]
            .into_iter()
            .enumerate()
        {
            observations0.set_allele(marker, 0, allele).unwrap();
        }
        for (marker, allele) in [Some(0), Some(2), None, Some(1), None]
            .into_iter()
            .enumerate()
        {
            observations1.set_allele(marker, 0, allele).unwrap();
        }
        let left_alleles = [
            [Some(0), Some(1), Some(2), None, Some(0)],
            [Some(1), None, Some(0), Some(2), Some(1)],
            [Some(2), Some(2), Some(1), Some(0), None],
            [None, Some(0), Some(2), Some(1), Some(2)],
        ];
        let right_alleles = [
            [Some(1), Some(2), Some(0), None, Some(2)],
            [Some(0), None, Some(1), Some(2), Some(0)],
            [Some(1), Some(0), None, Some(1), Some(2)],
        ];
        let mut haplotypes0 = PackedHaplotypes::empty(left_alleles[0].len(), n_markers, 2).unwrap();
        let mut haplotypes1 =
            PackedHaplotypes::empty(right_alleles[0].len(), n_markers, 2).unwrap();
        for (marker, alleles) in left_alleles.into_iter().enumerate() {
            for (state, allele) in alleles.into_iter().enumerate() {
                haplotypes0.set_allele(marker, state, allele).unwrap();
            }
        }
        for (marker, alleles) in right_alleles.into_iter().enumerate() {
            for (state, allele) in alleles.into_iter().enumerate() {
                haplotypes1.set_allele(marker, state, allele).unwrap();
            }
        }
        let panel0 = CompositePanel {
            haplotypes: haplotypes0,
            source_samples: vec![0; left_alleles.len()],
            marker_indices: (0..n_markers).collect(),
            allele_counts: vec![3; n_markers],
        };
        let panel1 = CompositePanel {
            haplotypes: haplotypes1,
            source_samples: vec![0; right_alleles.len()],
            marker_indices: (0..n_markers).collect(),
            allele_counts: vec![3; n_markers],
        };
        let cm_positions = [0.0, 0.05, 0.2, 0.21, 0.7];
        assert_bitwise(
            &observations0,
            &panel0,
            &observations1,
            &panel1,
            &cm_positions,
            &[0, 2, 4],
            &config,
        );
    }

    #[test]
    fn biallelic_plane_emissions_match_general_emissions() {
        let mut haplotypes = PackedHaplotypes::empty(128, 1, 1).unwrap();
        for state in 0..128 {
            haplotypes
                .set_allele(0, state, Some((state % 3 == 0) as u16))
                .unwrap();
        }
        let (allele_words, missing_words) = haplotypes.biallelic_marker_planes(0).unwrap();
        assert!(missing_words.iter().all(|&word| word == 0));

        for err in [0.0_f64, 0.025, 0.499] {
            let match_emission = (1.0 - err) as f32;
            let mismatch_emission = err.max(1.0e-12) as f32;
            for observed in [0, 1] {
                for state in 0..haplotypes.n_haplotypes {
                    assert_eq!(
                        biallelic_observed_emission(
                            allele_words,
                            state,
                            observed,
                            match_emission,
                            mismatch_emission,
                        ),
                        emission(Some(observed), haplotypes.allele(0, state), 2, err,)
                    );
                }
            }
        }
    }

    #[test]
    fn fused_biallelic_forward_step_matches_scalar_recurrence() {
        let state_count = 137usize;
        let mut allele_words = vec![0u64; state_count.div_ceil(64)];
        for state in 0..state_count {
            if (state * 17 + state / 3) % 11 < 4 {
                allele_words[state / 64] |= 1u64 << (state % 64);
            }
        }
        let initial = (0..state_count)
            .map(|state| ((state * 29 % 101) as f32 + 1.0) / 102.0)
            .collect::<Vec<_>>();

        for observed in [0, 1, 2] {
            for switch in [0.0_f32, 0.013, 0.25, 0.9] {
                let mut expected = initial.clone();
                let sum = expected.iter().sum::<f32>();
                for state in 0..state_count {
                    let transitioned =
                        (1.0 - switch) * expected[state] + switch * sum / state_count as f32;
                    expected[state] = transitioned
                        * biallelic_observed_emission(&allele_words, state, observed, 0.973, 0.027);
                }

                let mut observed_values = initial.clone();
                let previous_sum = initial.iter().sum::<f32>();
                let mut emissions = vec![0.0_f32; state_count];
                transition_emit_biallelic_row(
                    &initial,
                    &mut observed_values,
                    &allele_words,
                    observed,
                    switch,
                    previous_sum,
                    0.973,
                    0.027,
                    Some(&mut emissions),
                );
                assert_eq!(observed_values, expected);
                for state in 0..state_count {
                    assert_eq!(
                        emissions[state],
                        biallelic_observed_emission(&allele_words, state, observed, 0.973, 0.027,)
                    );
                }
            }
        }
    }

    fn beagle_em_reference(
        observations: &[u16],
        state_alleles: &[Vec<u16>],
        cm_positions: &[f64],
        config: &HMMConfig,
    ) -> HMMParameterStats {
        let n_markers = observations.len();
        let n_states = state_alleles.len();
        let err = config.err as f32;
        let emissions = [1.0_f32 - err, err];
        let mut p_recomb = vec![0.0_f32; n_markers];
        let mut gen_dist = vec![0.0_f32; n_markers];
        for marker in 1..n_markers {
            let distance = (cm_positions[marker] - cm_positions[marker - 1]) as f32;
            gen_dist[marker] = distance;
            p_recomb[marker] = (-(-f64::from(config.recombination_intensity) * f64::from(distance))
                .exp_m1()) as f32;
        }

        let mismatch = |marker: usize, state: usize| {
            u8::from(observations[marker] != state_alleles[state][marker]) as usize
        };
        let mut saved_backward = vec![vec![0.0_f32; n_states]; n_markers];
        let mut backward = vec![1.0_f32; n_states];
        saved_backward[n_markers - 1].copy_from_slice(&backward);
        for marker in (0..n_markers - 1).rev() {
            let next = marker + 1;
            let mut sum = 0.0_f32;
            for state in 0..n_states {
                backward[state] *= emissions[mismatch(next, state)];
                sum += backward[state];
            }
            let shift = p_recomb[next] / n_states as f32;
            let scale = (1.0_f32 - p_recomb[next]) / sum;
            for value in &mut backward {
                *value = scale * *value + shift;
            }
            saved_backward[marker].copy_from_slice(&backward);
        }

        let mut stats = HMMParameterStats::default();
        let mut forward = vec![1.0_f32 / n_states as f32; n_states];
        let mut previous_sum = 1.0_f32;
        let h_factor = n_states as f32 / (n_states as f32 - 1.0_f32);
        for marker in 0..n_markers {
            let p_switch = p_recomb[marker];
            let shift = p_switch / n_states as f32;
            let scale = (1.0_f32 - p_switch) / previous_sum;
            let no_switch_scale = ((1.0_f32 - p_switch) + shift) / previous_sum;
            let mut joint_state_sum = 0.0_f32;
            let mut state_sum = 0.0_f32;
            let mut mismatch_sum = 0.0_f32;
            let mut forward_sum = 0.0_f32;
            for state in 0..n_states {
                let differs = mismatch(marker, state);
                let emission = emissions[differs];
                joint_state_sum +=
                    saved_backward[marker][state] * emission * no_switch_scale * forward[state];
                forward[state] = emission * (scale * forward[state] + shift);
                forward_sum += forward[state];
                let state_probability = forward[state] * saved_backward[marker][state];
                state_sum += state_probability;
                if differs > 0 {
                    mismatch_sum += state_probability;
                }
            }
            stats.mismatch_marker_count += 1;
            stats.mismatch_probability_sum += f64::from(mismatch_sum / state_sum);
            let switch_probability = h_factor * (1.0_f32 - joint_state_sum / state_sum);
            if switch_probability > 0.0 {
                stats.switch_distance_cm_sum += f64::from(gen_dist[marker]);
                stats.switch_probability_sum += f64::from(switch_probability);
            }
            previous_sum = forward_sum;
        }
        stats
    }

    #[test]
    fn em_parameter_stats_follow_beagle_float_update_order() {
        let observations_by_marker = [0_u16, 1, 0, 1, 1, 0];
        let state_alleles = vec![
            vec![0_u16, 0, 0, 1, 1, 0],
            vec![1_u16, 1, 0, 1, 0, 0],
            vec![0_u16, 1, 1, 0, 1, 1],
        ];
        let cm_positions = [0.0, 0.0037, 0.0111, 0.0149, 0.0293, 0.0317];
        let mut observations = PackedHaplotypes::empty(1, observations_by_marker.len(), 1).unwrap();
        let mut haplotypes =
            PackedHaplotypes::empty(state_alleles.len(), observations_by_marker.len(), 1).unwrap();
        for marker in 0..observations_by_marker.len() {
            observations
                .set_allele(marker, 0, Some(observations_by_marker[marker]))
                .unwrap();
            for state in 0..state_alleles.len() {
                haplotypes
                    .set_allele(marker, state, Some(state_alleles[state][marker]))
                    .unwrap();
            }
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![0, 1, 2],
            marker_indices: (0..observations_by_marker.len()).collect(),
            allele_counts: vec![2; observations_by_marker.len()],
        };
        let config = HMMConfig {
            ne: 271,
            recombination_intensity: 0.083_123_45,
            err: 0.037_123_47,
            em: true,
            burnin: 3,
            iterations: 12,
            initial_lr: 100_000.0,
            phase_states: state_alleles.len(),
            haplotype_count: 128,
        };
        let expected = beagle_em_reference(
            &observations_by_marker,
            &state_alleles,
            &cm_positions,
            &config,
        );
        let actual = forward_backward_for_em(
            &observations,
            &panel,
            &cm_positions,
            &config,
            &mut HMMWorkspace::default(),
        )
        .unwrap();
        let transition_data = prepare_em_transition_data(&cm_positions, &config).unwrap();
        let precomputed = forward_backward_for_em_with_transition_data(
            &observations,
            &panel,
            &cm_positions,
            &config,
            &transition_data,
        )
        .unwrap();

        assert_eq!(actual.mismatch_marker_count, expected.mismatch_marker_count);
        assert_eq!(
            actual.mismatch_probability_sum.to_bits(),
            expected.mismatch_probability_sum.to_bits()
        );
        assert_eq!(
            actual.switch_probability_sum.to_bits(),
            expected.switch_probability_sum.to_bits()
        );
        assert_eq!(
            actual.switch_distance_cm_sum.to_bits(),
            expected.switch_distance_cm_sum.to_bits()
        );
        assert_eq!(
            precomputed.mismatch_marker_count,
            expected.mismatch_marker_count
        );
        assert_eq!(
            precomputed.mismatch_probability_sum.to_bits(),
            expected.mismatch_probability_sum.to_bits()
        );
        assert_eq!(
            precomputed.switch_probability_sum.to_bits(),
            expected.switch_probability_sum.to_bits()
        );
        assert_eq!(
            precomputed.switch_distance_cm_sum.to_bits(),
            expected.switch_distance_cm_sum.to_bits()
        );
    }

    #[test]
    fn em_mismatch_block_caches_missing_and_reordered_panel_alleles() {
        let observations_by_marker = [Some(0_u16), None, Some(2), Some(1)];
        let mut haplotypes = PackedHaplotypes::empty(3, 4, 2).unwrap();
        let panel_alleles = [
            [Some(1), Some(2), Some(0), Some(0)],
            [Some(0), Some(1), Some(1), Some(1)],
            [Some(2), None, None, Some(1)],
        ];
        for (state, markers) in panel_alleles.into_iter().enumerate() {
            for (marker, allele) in markers.into_iter().enumerate() {
                haplotypes.set_allele(marker, state, allele).unwrap();
            }
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![0, 1, 2],
            marker_indices: vec![2, 0, 1, 3],
            allele_counts: vec![3; 4],
        };
        let mismatch_cache = EmMismatchCache::build(&[&observations_by_marker], &panel).unwrap();
        let mut mismatches =
            vec![u8::MAX; observations_by_marker.len() * panel.haplotypes.n_haplotypes];

        mismatch_cache.fill_block(0, 0, observations_by_marker.len(), &mut mismatches);

        assert_eq!(mismatches, [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]);
    }

    #[test]
    fn packed_biallelic_em_mismatches_match_scalar_with_unaligned_marker_rows() {
        let state_count = 70usize;
        let marker_count = 6usize;
        let marker_indices = [4, 0, 5, 1, 3, 2];
        let observations = [Some(0_u16), Some(1), None, Some(1), Some(2), Some(0)];
        let mut haplotypes = PackedHaplotypes::empty(state_count, marker_count, 1).unwrap();
        for marker in 0..marker_count {
            for state in 0..state_count {
                let allele = if (marker + state * 5) % 29 == 0 {
                    None
                } else {
                    Some(((state * 7 + marker * 3) % 13 < 6) as u16)
                };
                haplotypes.set_allele(marker, state, allele).unwrap();
            }
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![0; state_count],
            marker_indices: marker_indices.to_vec(),
            allele_counts: vec![2; marker_count],
        };
        let words_per_track = (state_count * marker_count).div_ceil(64);
        let mut mismatches = vec![0_u64; words_per_track];

        assert!(super::fill_biallelic_mismatch_track(
            &mut mismatches,
            &observations,
            &panel,
        ));
        for marker in 0..marker_count {
            for state in 0..state_count {
                let flat_index = marker * state_count + state;
                let observed = (mismatches[flat_index >> 6] >> (flat_index & 63)) & 1 != 0;
                let expected = observations[marker].is_none()
                    || panel.haplotypes.allele(marker_indices[marker], state)
                        != observations[marker];
                assert_eq!(observed, expected, "marker={marker}, state={state}");
            }
        }
    }

    #[test]
    fn paired_em_tracks_match_independent_runs_with_missing_calls() {
        let observations0 = [Some(0_u16), None, Some(2), Some(1), Some(0)];
        let observations1 = [Some(1_u16), Some(2), None, Some(0), Some(1)];
        let cm_positions = [0.0, 0.005, 0.013, 0.024, 0.04];
        let panel_alleles = [
            [Some(0_u16), Some(1), None, Some(2), Some(1)],
            [Some(1_u16), Some(2), Some(0), Some(1), None],
            [Some(2_u16), None, Some(1), Some(0), Some(2)],
        ];
        let mut observations0_packed = PackedHaplotypes::empty(1, observations0.len(), 2).unwrap();
        let mut observations1_packed = PackedHaplotypes::empty(1, observations1.len(), 2).unwrap();
        for marker in 0..observations0.len() {
            observations0_packed
                .set_allele(marker, 0, observations0[marker])
                .unwrap();
            observations1_packed
                .set_allele(marker, 0, observations1[marker])
                .unwrap();
        }
        let mut haplotypes = PackedHaplotypes::empty(panel_alleles.len(), 5, 2).unwrap();
        for (state, alleles) in panel_alleles.iter().enumerate() {
            for (marker, allele) in alleles.iter().copied().enumerate() {
                haplotypes.set_allele(marker, state, allele).unwrap();
            }
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![0, 1, 2],
            marker_indices: vec![2, 0, 4, 1, 3],
            allele_counts: vec![3; observations0.len()],
        };
        let config = HMMConfig {
            ne: 271,
            recombination_intensity: 0.083_123_45,
            err: 0.037_123_47,
            em: true,
            burnin: 3,
            iterations: 12,
            initial_lr: 100_000.0,
            phase_states: panel_alleles.len(),
            haplotype_count: 128,
        };
        let transitions = prepare_em_transition_data(&cm_positions, &config).unwrap();
        let expected0 = forward_backward_for_em_with_transition_data(
            &observations0_packed,
            &panel,
            &cm_positions,
            &config,
            &transitions,
        )
        .unwrap();
        let expected1 = forward_backward_for_em_with_transition_data(
            &observations1_packed,
            &panel,
            &cm_positions,
            &config,
            &transitions,
        )
        .unwrap();
        let (actual0, actual1) = forward_backward_for_em_pair_with_transition_data(
            &observations0,
            &observations1,
            &panel,
            &cm_positions,
            &config,
            &transitions,
        )
        .unwrap();

        for (actual, expected) in [(actual0, expected0), (actual1, expected1)] {
            assert_eq!(actual.mismatch_marker_count, expected.mismatch_marker_count);
            assert_eq!(
                actual.mismatch_probability_sum.to_bits(),
                expected.mismatch_probability_sum.to_bits()
            );
            assert_eq!(
                actual.switch_probability_sum.to_bits(),
                expected.switch_probability_sum.to_bits()
            );
            assert_eq!(
                actual.switch_distance_cm_sum.to_bits(),
                expected.switch_distance_cm_sum.to_bits()
            );
        }
    }

    #[test]
    fn collapsed_identical_em_tracks_match_paired_hmm_bitwise() {
        let observations = [Some(0_u16), None, Some(2), Some(1), Some(0)];
        let cm_positions = [0.0, 0.005, 0.013, 0.024, 0.04];
        let panel_alleles = [
            [Some(0_u16), Some(1), None, Some(2), Some(1)],
            [Some(1_u16), Some(2), Some(0), Some(1), None],
            [Some(2_u16), None, Some(1), Some(0), Some(2)],
        ];
        let mut haplotypes = PackedHaplotypes::empty(panel_alleles.len(), 5, 2).unwrap();
        for (state, alleles) in panel_alleles.iter().enumerate() {
            for (marker, allele) in alleles.iter().copied().enumerate() {
                haplotypes.set_allele(marker, state, allele).unwrap();
            }
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![0, 1, 2],
            marker_indices: vec![2, 0, 4, 1, 3],
            allele_counts: vec![3; observations.len()],
        };
        let config = HMMConfig {
            ne: 271,
            recombination_intensity: 0.083_123_45,
            err: 0.037_123_47,
            em: true,
            burnin: 3,
            iterations: 12,
            initial_lr: 100_000.0,
            phase_states: panel_alleles.len(),
            haplotype_count: 128,
        };
        let transitions = prepare_em_transition_data(&cm_positions, &config).unwrap();
        let (paired, _) = forward_backward_for_em_pair_with_transition_data(
            &observations,
            &observations,
            &panel,
            &cm_positions,
            &config,
            &transitions,
        )
        .unwrap();
        let collapsed = forward_backward_for_em_single_observation_with_transition_data(
            &observations,
            &panel,
            &cm_positions,
            &config,
            &transitions,
        )
        .unwrap();

        assert_eq!(
            collapsed.mismatch_marker_count,
            paired.mismatch_marker_count
        );
        assert_eq!(
            collapsed.mismatch_probability_sum.to_bits(),
            paired.mismatch_probability_sum.to_bits()
        );
        assert_eq!(
            collapsed.switch_probability_sum.to_bits(),
            paired.switch_probability_sum.to_bits()
        );
        assert_eq!(
            collapsed.switch_distance_cm_sum.to_bits(),
            paired.switch_distance_cm_sum.to_bits()
        );
    }

    #[test]
    fn multiallelic_mismatch_emission_matches_beagle_binary_mismatch_model() {
        assert_eq!(emission(Some(0), Some(1), 3, 0.02), 0.02f32);
    }

    #[test]
    fn transition_uses_beagles_unrounded_recombination_intensity() {
        let haplotype_count = 128;
        let intensity = (0.04_f32 * 206.2) / haplotype_count as f32;
        let distance_cm = 0.05;
        let observed = transition_probability(distance_cm, 34, haplotype_count, intensity);
        let rounded_ne = f64::from(25.0_f32 * intensity * haplotype_count as f32).ceil() as u64;
        let reconstructed_intensity = (0.04_f32 * rounded_ne as f32) / haplotype_count as f32;
        let reconstructed =
            transition_probability(distance_cm, 34, haplotype_count, reconstructed_intensity);

        assert_ne!(observed, reconstructed);
        let expected = (-f64::from(intensity) * f64::from(distance_cm as f32)).exp_m1();
        assert_eq!(observed, (-expected) as f32);
    }

    #[test]
    fn checkpointed_hmm_matches_scalar_recursion_across_block_boundary() {
        let n_markers = CHECKPOINT_STRIDE + 7;
        let n_states = 3;
        let observations_by_marker = (0..n_markers)
            .map(|marker| ((marker * 7 + marker / 5) % 2) as u16)
            .collect::<Vec<_>>();
        let state_alleles = (0..n_states)
            .map(|state| {
                (0..n_markers)
                    .map(|marker| ((marker * 3 + state * 5 + marker / 11) % 2) as u16)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let cm_positions = (0..n_markers)
            .map(|marker| marker as f64 * 0.013_71)
            .collect::<Vec<_>>();

        let mut observations = PackedHaplotypes::empty(1, n_markers, 1).unwrap();
        let mut haplotypes = PackedHaplotypes::empty(n_states, n_markers, 1).unwrap();
        for marker in 0..n_markers {
            observations
                .set_allele(marker, 0, Some(observations_by_marker[marker]))
                .unwrap();
            for state in 0..n_states {
                haplotypes
                    .set_allele(marker, state, Some(state_alleles[state][marker]))
                    .unwrap();
            }
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: (0..n_states).collect(),
            marker_indices: (0..n_markers).collect(),
            allele_counts: vec![2; n_markers],
        };
        let config = HMMConfig {
            ne: 211,
            recombination_intensity: 0.071_234_56,
            err: 0.037_123_47,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 100_000.0,
            phase_states: n_states,
            haplotype_count: 2 * n_states,
        };

        let mut expected_forward = vec![vec![0.0f32; n_states]; n_markers];
        let mut alpha = vec![1.0f32 / n_states as f32; n_states];
        for marker in 0..n_markers {
            if marker > 0 {
                let switch = transition_probability(
                    cm_positions[marker] - cm_positions[marker - 1],
                    n_states,
                    config.haplotype_count,
                    config.recombination_intensity,
                );
                let sum = alpha.iter().sum::<f32>();
                for probability in &mut alpha {
                    *probability = (1.0 - switch) * *probability + switch * sum / n_states as f32;
                }
            }
            for state in 0..n_states {
                alpha[state] *= emission(
                    Some(observations_by_marker[marker]),
                    Some(state_alleles[state][marker]),
                    2,
                    config.err,
                );
            }
            normalize(&mut alpha);
            expected_forward[marker].copy_from_slice(&alpha);
        }

        let mut expected_backward = vec![vec![1.0f32; n_states]; n_markers];
        let mut beta = vec![1.0f32; n_states];
        for marker in (0..n_markers - 1).rev() {
            let next = marker + 1;
            let switch = transition_probability(
                cm_positions[next] - cm_positions[marker],
                n_states,
                config.haplotype_count,
                config.recombination_intensity,
            );
            let mut emission_backward_sum = 0.0f32;
            for state in 0..n_states {
                let state_emission = emission(
                    Some(observations_by_marker[next]),
                    Some(state_alleles[state][next]),
                    2,
                    config.err,
                );
                emission_backward_sum += state_emission * beta[state];
            }
            let uniform_component = switch * emission_backward_sum / n_states as f32;
            for state in 0..n_states {
                let state_emission = emission(
                    Some(observations_by_marker[next]),
                    Some(state_alleles[state][next]),
                    2,
                    config.err,
                );
                beta[state] = (1.0 - switch) * state_emission * beta[state] + uniform_component;
            }
            normalize(&mut beta);
            expected_backward[marker].copy_from_slice(&beta);
        }

        let mut diagnostics = Vec::new();
        let mut workspace = HMMWorkspace::default();
        forward_backward_with_diagnostics(
            &observations,
            &panel,
            &cm_positions,
            &config,
            &mut workspace,
            &(0..n_markers).collect::<Vec<_>>(),
            |diagnostic| {
                diagnostics.push(diagnostic);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(diagnostics.len(), n_markers);
        for diagnostic in diagnostics {
            let marker = diagnostic.marker;
            for state in 0..n_states {
                assert_eq!(
                    diagnostic.forward_probabilities[state].to_bits(),
                    expected_forward[marker][state].to_bits(),
                    "forward marker={marker}, state={state}"
                );
                assert_eq!(
                    diagnostic.backward_probabilities[state].to_bits(),
                    expected_backward[marker][state].to_bits(),
                    "backward marker={marker}, state={state}"
                );
            }
        }
    }

    fn assert_hmm_matches_exhaustive_paths(
        observations_by_marker: &[u16],
        state_alleles: &[Vec<u16>],
        cm_positions: &[f64],
    ) {
        let n_markers = observations_by_marker.len();
        let n_states = state_alleles.len();
        assert!(n_markers >= 2);
        assert!(state_alleles.iter().all(|track| track.len() == n_markers));
        assert_eq!(cm_positions.len(), n_markers);

        let mut observations = PackedHaplotypes::empty(1, n_markers, 1).unwrap();
        let mut haplotypes = PackedHaplotypes::empty(n_states, n_markers, 1).unwrap();
        for marker in 0..n_markers {
            observations
                .set_allele(marker, 0, Some(observations_by_marker[marker]))
                .unwrap();
            for state in 0..n_states {
                haplotypes
                    .set_allele(marker, state, Some(state_alleles[state][marker]))
                    .unwrap();
            }
        }
        let panel = CompositePanel {
            haplotypes,
            source_samples: (0..n_states).collect(),
            marker_indices: (0..n_markers).collect(),
            allele_counts: vec![2; n_markers],
        };
        let config = HMMConfig {
            ne: 190,
            recombination_intensity: (0.04_f32 * 190.0) / (n_states * 2) as f32,
            err: 0.07,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 100_000.0,
            phase_states: n_states,
            haplotype_count: n_states * 2,
        };

        let transition = |marker: usize| {
            if marker == 0 {
                0.0
            } else {
                transition_probability(
                    cm_positions[marker] - cm_positions[marker - 1],
                    n_states,
                    config.haplotype_count,
                    config.recombination_intensity,
                )
            }
        };
        let emission_probability = |marker: usize, state: usize| {
            if observations_by_marker[marker] == state_alleles[state][marker] {
                (1.0 - config.err) as f32
            } else {
                config.err as f32
            }
        };

        let mut expected_forward = vec![vec![0.0f64; n_states]; n_markers];
        for marker in 0..n_markers {
            let prefix_count = n_states.pow((marker + 1) as u32);
            for encoded in 0..prefix_count {
                let mut remaining = encoded;
                let mut path = vec![0usize; marker + 1];
                for state in path.iter_mut().rev() {
                    *state = remaining % n_states;
                    remaining /= n_states;
                }
                let mut mass = 1.0 / n_states as f64;
                for current_marker in 0..=marker {
                    if current_marker > 0 {
                        let switch = f64::from(transition(current_marker));
                        let same_state = path[current_marker - 1] == path[current_marker];
                        mass *= if same_state {
                            1.0 - switch + switch / n_states as f64
                        } else {
                            switch / n_states as f64
                        };
                    }
                    mass *= f64::from(emission_probability(current_marker, path[current_marker]));
                }
                expected_forward[marker][path[marker]] += mass;
            }
            let sum = expected_forward[marker].iter().sum::<f64>();
            for value in &mut expected_forward[marker] {
                *value /= sum;
            }
        }

        let mut expected_backward = vec![vec![0.0f64; n_states]; n_markers];
        for marker in 0..n_markers {
            for start_state in 0..n_states {
                let suffix_markers = n_markers - marker - 1;
                let suffix_count = n_states.pow(suffix_markers as u32);
                for encoded in 0..suffix_count {
                    let mut remaining = encoded;
                    let mut suffix = vec![0usize; suffix_markers];
                    for state in suffix.iter_mut().rev() {
                        *state = remaining % n_states;
                        remaining /= n_states;
                    }
                    let mut previous = start_state;
                    let mut mass = 1.0f64;
                    for (offset, &state) in suffix.iter().enumerate() {
                        let next_marker = marker + offset + 1;
                        let switch = f64::from(transition(next_marker));
                        mass *= if state == previous {
                            1.0 - switch + switch / n_states as f64
                        } else {
                            switch / n_states as f64
                        };
                        mass *= f64::from(emission_probability(next_marker, state));
                        previous = state;
                    }
                    expected_backward[marker][start_state] += mass;
                }
            }
            if marker + 1 < n_markers {
                let sum = expected_backward[marker].iter().sum::<f64>();
                for value in &mut expected_backward[marker] {
                    *value /= sum;
                }
            }
        }

        let mut diagnostics = Vec::new();
        forward_backward_with_diagnostics(
            &observations,
            &panel,
            cm_positions,
            &config,
            &mut HMMWorkspace::default(),
            &(0..n_markers).collect::<Vec<_>>(),
            |row| {
                diagnostics.push(row);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(diagnostics.len(), n_markers);

        for diagnostic in diagnostics {
            let marker = diagnostic.marker;
            assert!((diagnostic.transition_probability - transition(marker)).abs() < 1.0e-6);
            for state in 0..n_states {
                assert!(
                    (f64::from(diagnostic.emission_probabilities[state])
                        - f64::from(emission_probability(marker, state)))
                    .abs()
                        < 1.0e-6
                );
                assert!(
                    (f64::from(diagnostic.forward_probabilities[state])
                        - expected_forward[marker][state])
                        .abs()
                        < 1.0e-5
                );
                assert!(
                    (f64::from(diagnostic.backward_probabilities[state])
                        - expected_backward[marker][state])
                        .abs()
                        < 1.0e-5,
                    "backward marker={marker} state={state} observed={} expected={}",
                    diagnostic.backward_probabilities[state],
                    expected_backward[marker][state]
                );
            }
            let normalizer = expected_forward[marker]
                .iter()
                .zip(&expected_backward[marker])
                .map(|(&forward, &backward)| forward * backward)
                .sum::<f64>();
            assert!((f64::from(diagnostic.posterior_normalization) - normalizer).abs() < 1.0e-5);
        }
    }

    #[test]
    fn haploid_hmm_marginals_match_enumeration_with_one_transition() {
        assert_hmm_matches_exhaustive_paths(
            &[0, 1],
            &[vec![0, 1], vec![1, 1], vec![1, 0]],
            &[0.0, 0.037],
        );
    }

    #[test]
    fn haploid_hmm_marginals_match_enumeration_with_two_transitions() {
        assert_hmm_matches_exhaustive_paths(
            &[0, 1, 0],
            &[vec![0, 1, 0], vec![1, 1, 0], vec![1, 0, 1]],
            &[0.0, 0.037, 0.091],
        );
    }

    #[test]
    fn parameter_stats_match_exhaustive_two_marker_hmm() {
        let mut observations = PackedHaplotypes::empty(1, 2, 1).unwrap();
        observations.set_allele(0, 0, Some(0)).unwrap();
        observations.set_allele(1, 0, Some(0)).unwrap();
        let mut haplotypes = PackedHaplotypes::empty(2, 2, 1).unwrap();
        haplotypes.set_allele(0, 0, Some(0)).unwrap();
        haplotypes.set_allele(1, 0, Some(0)).unwrap();
        haplotypes.set_allele(0, 1, Some(1)).unwrap();
        haplotypes.set_allele(1, 1, Some(1)).unwrap();
        let panel = CompositePanel {
            haplotypes,
            source_samples: vec![1, 2],
            marker_indices: vec![0, 1],
            allele_counts: vec![2, 2],
        };
        let config = HMMConfig {
            ne: 500,
            recombination_intensity: (0.04_f32 * 500.0) / 4.0,
            err: 0.05,
            em: true,
            burnin: 1,
            iterations: 2,
            initial_lr: 100_000.0,
            phase_states: 2,
            haplotype_count: 4,
        };
        let distance_cm = 0.05;
        let switch = f64::from(transition_probability(
            distance_cm,
            2,
            config.haplotype_count,
            config.recombination_intensity,
        ));
        let mut total = 0.0f64;
        let mut mismatch_mass = 0.0f64;
        let mut changed_mass = 0.0f64;
        for first_state in 0..2 {
            for second_state in 0..2 {
                let first_allele = first_state;
                let second_allele = second_state;
                let first_emission = if first_allele == 0 { 0.95 } else { 0.05 };
                let second_emission = if second_allele == 0 { 0.95 } else { 0.05 };
                let transition = if first_state == second_state {
                    1.0 - switch + switch / 2.0
                } else {
                    switch / 2.0
                };
                let path_mass = 0.5 * first_emission * transition * second_emission;
                total += path_mass;
                mismatch_mass += path_mass * f64::from(first_allele != 0)
                    + path_mass * f64::from(second_allele != 0);
                if first_state != second_state {
                    changed_mass += path_mass;
                }
            }
        }

        let posterior = forward_backward(
            &observations,
            &panel,
            &[0.0, distance_cm],
            &config,
            &mut HMMWorkspace::default(),
        )
        .unwrap();
        let em_stats = forward_backward_for_em(
            &observations,
            &panel,
            &[0.0, distance_cm],
            &config,
            &mut HMMWorkspace::default(),
        )
        .unwrap();
        let mut diagnostics = Vec::new();
        forward_backward_with_diagnostics(
            &observations,
            &panel,
            &[0.0, distance_cm],
            &config,
            &mut HMMWorkspace::default(),
            &[0, 1],
            |trace| {
                diagnostics.push(trace);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(posterior.parameter_stats.mismatch_marker_count, 2);
        assert_eq!(em_stats.mismatch_marker_count, 2);
        assert!(
            (posterior.parameter_stats.mismatch_probability_sum - mismatch_mass / total).abs()
                < 1.0e-5
        );
        assert!(
            (posterior.parameter_stats.switch_probability_sum - 2.0 * changed_mass / total).abs()
                < 1.0e-5
        );
        assert!((posterior.parameter_stats.switch_distance_cm_sum - distance_cm).abs() < 1.0e-6);
        assert!(
            (em_stats.mismatch_probability_sum
                - posterior.parameter_stats.mismatch_probability_sum)
                .abs()
                < 1.0e-6
        );
        assert!(
            (em_stats.switch_probability_sum - posterior.parameter_stats.switch_probability_sum)
                .abs()
                < 1.0e-6
        );

        assert_eq!(diagnostics.len(), 2);
        let marker0 = diagnostics.iter().find(|trace| trace.marker == 0).unwrap();
        let marker1 = diagnostics.iter().find(|trace| trace.marker == 1).unwrap();
        let expected_alpha0 = [0.95, 0.05];
        let transitioned = [
            (1.0 - switch) * expected_alpha0[0] + switch / 2.0,
            (1.0 - switch) * expected_alpha0[1] + switch / 2.0,
        ];
        let alpha1_sum = transitioned[0] * 0.95 + transitioned[1] * 0.05;
        let expected_alpha1 = [
            transitioned[0] * 0.95 / alpha1_sum,
            transitioned[1] * 0.05 / alpha1_sum,
        ];
        let beta0_unnormalized = [
            (1.0 - switch) * 0.95 + switch / 2.0,
            (1.0 - switch) * 0.05 + switch / 2.0,
        ];
        let beta0_sum = beta0_unnormalized.iter().sum::<f64>();
        let expected_beta0 = [
            beta0_unnormalized[0] / beta0_sum,
            beta0_unnormalized[1] / beta0_sum,
        ];
        assert_eq!(marker0.transition_probability, 0.0);
        assert!((f64::from(marker1.transition_probability) - switch).abs() < 1.0e-6);
        for state in 0..2 {
            assert!(
                (f64::from(marker0.forward_probabilities[state]) - expected_alpha0[state]).abs()
                    < 1.0e-6
            );
            assert!(
                (f64::from(marker1.forward_probabilities[state]) - expected_alpha1[state]).abs()
                    < 1.0e-6
            );
            assert!(
                (f64::from(marker0.backward_probabilities[state]) - expected_beta0[state]).abs()
                    < 1.0e-6
            );
            assert_eq!(marker1.backward_probabilities[state], 1.0);
            assert_eq!(
                marker0.emission_probabilities[state],
                [0.95f32, 0.05f32][state]
            );
            assert_eq!(
                marker1.emission_probabilities[state],
                [0.95f32, 0.05f32][state]
            );
        }
        let expected_norm0 =
            expected_alpha0[0] * expected_beta0[0] + expected_alpha0[1] * expected_beta0[1];
        assert!((f64::from(marker0.posterior_normalization) - expected_norm0).abs() < 1.0e-6);
        assert!((marker1.posterior_normalization - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn posterior_and_backward_simd_matches_scalar_bit_for_bit() {
        for state_count in [1usize, 7, 8, 63, 64, 70, 1_600] {
            let forward = (0..state_count)
                .map(|state| ((state * 17) % 131 + 1) as f32 / 10_000.0)
                .collect::<Vec<_>>();
            let original_backward = (0..state_count)
                .map(|state| ((state * 29) % 137 + 1) as f32 / 10_000.0)
                .collect::<Vec<_>>();
            let emissions = (0..state_count)
                .map(|state| ((state * 43) % 149 + 1) as f32 / 10_000.0)
                .collect::<Vec<_>>();
            let mut posterior_sum = 0.0f32;
            for state in 0..state_count {
                posterior_sum += forward[state] * original_backward[state];
            }
            let scale = 0.83f32;
            let uniform = 0.0007f32;
            let mut expected_posterior = vec![0.0f32; state_count];
            let mut expected_backward = original_backward.clone();
            for state in 0..state_count {
                let state_mass = f64::from(forward[state]) * f64::from(original_backward[state]);
                expected_posterior[state] = (state_mass / f64::from(posterior_sum)) as f32;
                expected_backward[state] =
                    scale * emissions[state] * original_backward[state] + uniform;
            }

            let mut observed_posterior = vec![0.0f32; state_count];
            let mut observed_backward = original_backward;
            posterior_and_backward_update(
                &forward,
                &mut observed_backward,
                &emissions,
                posterior_sum,
                scale,
                uniform,
                &mut observed_posterior,
                true,
            );
            assert_eq!(
                observed_posterior
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_posterior
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "states={state_count}"
            );
            assert_eq!(
                observed_backward
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_backward
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "states={state_count}"
            );
        }
    }

    #[test]
    fn stage2_checkpointed_selected_forward_matches_full_bitwise_and_manages_capacity() {
        const MARKERS: usize = 600;
        const STATES: usize = 8;

        let make_panel = |seed: usize| {
            let mut haplotypes = PackedHaplotypes::empty_state_major(STATES, MARKERS, 1).unwrap();
            for marker in 0..MARKERS {
                for state in 0..STATES {
                    let allele = (((marker * 7 + state * 11 + seed * 5) / 3) % 2) as u16;
                    haplotypes.set_allele(marker, state, Some(allele)).unwrap();
                }
            }
            haplotypes.mark_complete();
            CompositePanel {
                haplotypes,
                source_samples: vec![0; STATES],
                marker_indices: (0..MARKERS).collect(),
                allele_counts: vec![2; MARKERS],
            }
        };
        let make_obs = |seed: usize| {
            let mut obs = PackedHaplotypes::empty_state_major(1, MARKERS, 1).unwrap();
            for marker in 0..MARKERS {
                let allele = if marker % 9 == 0 {
                    None
                } else {
                    Some((((marker * 5 + seed * 3) / 4) % 2) as u16)
                };
                obs.set_allele(marker, 0, allele).unwrap();
            }
            obs
        };

        let panel0 = make_panel(1);
        let panel1 = make_panel(2);
        let obs0 = make_obs(3);
        let obs1 = make_obs(4);
        let cm_positions = (0..MARKERS).map(|m| m as f64 * 0.01).collect::<Vec<_>>();
        let config = HMMConfig {
            ne: 1000,
            recombination_intensity: 0.04,
            err: 0.001,
            em: false,
            burnin: 1,
            iterations: 1,
            initial_lr: 100_000.0,
            phase_states: STATES,
            haplotype_count: 100,
        };
        // Cover block 0 start (0), interior, block boundaries (255/256, 511/512), and tail (599).
        let selected_markers = vec![0, 1, 17, 128, 255, 256, 257, 400, 511, 512, 599];

        let mut full_ws0 = HMMWorkspace::default();
        let mut full_ws1 = HMMWorkspace::default();
        let mut full_out = Vec::new();
        super::forward_backward_selected_states_pair_impl_with_budget(
            &obs0,
            &panel0,
            &obs1,
            &panel1,
            &cm_positions,
            &config,
            &mut full_ws0,
            &mut full_ws1,
            &selected_markers,
            &mut |marker, p0, p1| {
                full_out.push((marker, p0.to_vec(), p1.to_vec()));
                Ok(())
            },
            None,
        )
        .unwrap();
        let full_cap = full_ws0.selected_forward_values.capacity();
        assert!(full_cap >= selected_markers.len() * STATES);
        assert_eq!(full_ws0.checkpoint_values.capacity(), 0);

        let mut ckpt_ws0 = HMMWorkspace::default();
        let mut ckpt_ws1 = HMMWorkspace::default();
        let mut ckpt_out = Vec::new();
        super::forward_backward_selected_states_pair_impl_with_budget(
            &obs0,
            &panel0,
            &obs1,
            &panel1,
            &cm_positions,
            &config,
            &mut ckpt_ws0,
            &mut ckpt_ws1,
            &selected_markers,
            &mut |marker, p0, p1| {
                ckpt_out.push((marker, p0.to_vec(), p1.to_vec()));
                Ok(())
            },
            Some(1),
        )
        .unwrap();
        assert_eq!(ckpt_out.len(), full_out.len());
        for ((m_c, p0_c, p1_c), (m_f, p0_f, p1_f)) in ckpt_out.iter().zip(&full_out) {
            assert_eq!(m_c, m_f);
            assert_eq!(
                p0_c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                p0_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                p1_c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                p1_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        assert!(ckpt_ws0.selected_forward_values.capacity() < full_cap);
        assert!(ckpt_ws0.checkpoint_values.capacity() > 0);

        // Full -> checkpointed must shrink selected_forward_values below full capacity.
        super::forward_backward_selected_states_pair_impl_with_budget(
            &obs0,
            &panel0,
            &obs1,
            &panel1,
            &cm_positions,
            &config,
            &mut full_ws0,
            &mut full_ws1,
            &selected_markers,
            &mut |_, _, _| Ok(()),
            Some(1),
        )
        .unwrap();
        assert!(full_ws0.selected_forward_values.capacity() < full_cap);

        // Checkpointed -> full must release checkpoint_values capacity.
        super::forward_backward_selected_states_pair_impl_with_budget(
            &obs0,
            &panel0,
            &obs1,
            &panel1,
            &cm_positions,
            &config,
            &mut ckpt_ws0,
            &mut ckpt_ws1,
            &selected_markers,
            &mut |_, _, _| Ok(()),
            None,
        )
        .unwrap();
        assert_eq!(ckpt_ws0.checkpoint_values.capacity(), 0);
        assert_eq!(ckpt_ws1.checkpoint_values.capacity(), 0);
    }
}
