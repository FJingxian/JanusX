use crate::kmer::format::{BKMER_HEADER_SIZE, BSITE_HEADER_SIZE};

pub fn bkmer_body_offset() -> u64 {
    BKMER_HEADER_SIZE as u64
}

pub fn bsite_body_offset() -> u64 {
    BSITE_HEADER_SIZE as u64
}

pub fn bsite_column_offset(col_idx: u64, bytes_per_col: u64) -> u64 {
    bsite_body_offset() + col_idx * bytes_per_col
}

#[cfg(test)]
mod tests {
    use super::{bkmer_body_offset, bsite_body_offset, bsite_column_offset};

    #[test]
    fn offsets_are_stable() {
        assert_eq!(bkmer_body_offset(), 64);
        assert_eq!(bsite_body_offset(), 80);
        assert_eq!(bsite_column_offset(3, 5), 95);
    }
}
