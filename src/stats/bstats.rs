#[inline]
pub fn words_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(64).max(1)
}

#[inline]
pub fn tail_mask(n_samples: usize) -> Option<u64> {
    let rem = n_samples & 63;
    if rem == 0 {
        None
    } else {
        Some((1u64 << rem) - 1u64)
    }
}

#[inline]
pub fn apply_tail_mask(bits: &mut [u64], mask: Option<u64>) {
    if let Some(mask_bits) = mask {
        if let Some(last) = bits.last_mut() {
            *last &= mask_bits;
        }
    }
}
