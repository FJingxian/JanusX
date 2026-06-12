use anyhow::{bail, Result};

pub const ENCODING_NAME: &str = "ACGT_2BIT_A00_C01_G10_T11";

#[inline]
pub fn encode_base_acgt(base: u8) -> Option<u8> {
    match base {
        b'A' | b'a' => Some(0),
        b'C' | b'c' => Some(1),
        b'G' | b'g' => Some(2),
        b'T' | b't' => Some(3),
        _ => None,
    }
}

pub fn encode_kmer_u64(seq: &str) -> Result<u64> {
    let raw = seq.as_bytes();
    if raw.is_empty() {
        bail!("empty k-mer is not allowed");
    }
    if raw.len() > 31 {
        bail!(
            "k-mer length {} exceeds current u64 limit (k <= 31)",
            raw.len()
        );
    }
    let mut code = 0u64;
    for &base in raw {
        let enc = encode_base_acgt(base)
            .ok_or_else(|| anyhow::anyhow!("unsupported base in k-mer: {}", base as char))?;
        code = (code << 2) | u64::from(enc);
    }
    Ok(code)
}

pub fn decode_kmer_u64(mut code: u64, k: u32) -> String {
    let mut out = vec![b'A'; k as usize];
    for idx in (0..k as usize).rev() {
        let base = (code & 0b11) as u8;
        out[idx] = match base {
            0 => b'A',
            1 => b'C',
            2 => b'G',
            _ => b'T',
        };
        code >>= 2;
    }
    String::from_utf8(out).expect("ASCII decode should never fail")
}

pub fn reverse_complement_code(mut code: u64, k: u32) -> u64 {
    let mut rev = 0u64;
    for _ in 0..k {
        let base = code & 0b11;
        let comp = 3u64 - base;
        rev = (rev << 2) | comp;
        code >>= 2;
    }
    rev
}

#[inline]
pub fn canonical_code(code: u64, k: u32) -> u64 {
    let rc = reverse_complement_code(code, k);
    code.min(rc)
}

pub fn bucket_id_from_code(code: u64, k: u32, bucket_bits: u8) -> usize {
    if bucket_bits == 0 {
        return 0;
    }
    let used_bits = 2 * k;
    let effective_bits = bucket_bits.min(used_bits as u8);
    let shift = used_bits.saturating_sub(u32::from(effective_bits));
    let mask = if effective_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << effective_bits) - 1
    };
    ((code >> shift) & mask) as usize
}

#[cfg(test)]
mod tests {
    use super::{bucket_id_from_code, canonical_code, decode_kmer_u64, encode_kmer_u64};

    #[test]
    fn encode_decode_roundtrip() {
        let code = encode_kmer_u64("ACGTAC").expect("encode");
        assert_eq!(decode_kmer_u64(code, 6), "ACGTAC");
    }

    #[test]
    fn canonical_matches_reverse_complement_min() {
        let fwd = encode_kmer_u64("ATGC").expect("encode");
        let rev = encode_kmer_u64("GCAT").expect("encode");
        assert_eq!(canonical_code(fwd, 4), canonical_code(rev, 4));
    }

    #[test]
    fn bucket_is_stable() {
        let code = encode_kmer_u64("ACGTACGT").expect("encode");
        let bucket = bucket_id_from_code(code, 8, 4);
        assert!(bucket < 16);
    }
}
