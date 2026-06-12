use std::io::{self, Read, Write};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct KmerPresenceRec {
    pub kmer: u64,
    pub sample_id: u32,
    pub pad: u32,
}

pub fn write_rec<W: Write>(writer: &mut W, rec: KmerPresenceRec) -> io::Result<()> {
    writer.write_all(&rec.kmer.to_le_bytes())?;
    writer.write_all(&rec.sample_id.to_le_bytes())?;
    writer.write_all(&rec.pad.to_le_bytes())?;
    Ok(())
}

pub fn read_rec_opt<R: Read>(reader: &mut R) -> io::Result<Option<KmerPresenceRec>> {
    let mut k = [0u8; 8];
    match reader.read_exact(&mut k) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    }

    let mut sid = [0u8; 4];
    let mut pad = [0u8; 4];
    reader.read_exact(&mut sid)?;
    reader.read_exact(&mut pad)?;

    Ok(Some(KmerPresenceRec {
        kmer: u64::from_le_bytes(k),
        sample_id: u32::from_le_bytes(sid),
        pad: u32::from_le_bytes(pad),
    }))
}

#[cfg(test)]
mod tests {
    use super::{read_rec_opt, write_rec, KmerPresenceRec};

    #[test]
    fn record_roundtrip() {
        let rec = KmerPresenceRec {
            kmer: 42,
            sample_id: 7,
            pad: 0,
        };
        let mut buf = Vec::new();
        write_rec(&mut buf, rec).expect("write");
        let got = read_rec_opt(&mut buf.as_slice())
            .expect("read")
            .expect("record");
        assert_eq!(got, rec);
    }
}
