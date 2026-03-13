use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Internal writer wrapper: either plain text or gzip-compressed text.
pub enum VcfOut {
    Plain(BufWriter<File>),
    Gzip(BufWriter<GzEncoder<File>>),
}

impl VcfOut {
    #[inline]
    pub fn from_path(path: &str) -> std::io::Result<Self> {
        let file = File::create(path)?;
        if path.ends_with(".gz") {
            let enc = GzEncoder::new(file, Compression::default());
            Ok(VcfOut::Gzip(BufWriter::new(enc)))
        } else {
            Ok(VcfOut::Plain(BufWriter::new(file)))
        }
    }

    #[inline]
    pub fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        match self {
            VcfOut::Plain(w) => w.write_all(buf),
            VcfOut::Gzip(w) => w.write_all(buf),
        }
    }

    #[inline]
    pub fn flush(&mut self) -> std::io::Result<()> {
        match self {
            VcfOut::Plain(w) => w.flush(),
            VcfOut::Gzip(w) => w.flush(),
        }
    }

    /// Finish the writer. For gzip, this writes the gzip trailer (CRC/footer).
    #[inline]
    pub fn finish(mut self) -> std::io::Result<()> {
        self.flush()?;
        match self {
            VcfOut::Plain(_w) => Ok(()),
            VcfOut::Gzip(w) => {
                let enc = w.into_inner()?;
                let _file = enc.finish()?;
                Ok(())
            }
        }
    }
}
