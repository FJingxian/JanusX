use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub fn write_fasta(path: &Path, samples: &[String], seqs: &[Vec<u8>]) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    for (name, seq) in samples.iter().zip(seqs.iter()) {
        writeln!(w, ">{name}").map_err(|e| e.to_string())?;
        for chunk in seq.chunks(80) {
            w.write_all(chunk).map_err(|e| e.to_string())?;
            writeln!(w).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

pub fn write_phylip(path: &Path, samples: &[String], seqs: &[Vec<u8>]) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    let n_taxa = samples.len();
    let n_sites = seqs.first().map(|x| x.len()).unwrap_or(0);
    writeln!(w, "{n_taxa} {n_sites}").map_err(|e| e.to_string())?;

    let max_name = samples.iter().map(|s| s.len()).max().unwrap_or(0);
    for (name, seq) in samples.iter().zip(seqs.iter()) {
        let padding = " ".repeat(max_name + 2 - name.len());
        write!(w, "{name}{padding}").map_err(|e| e.to_string())?;
        w.write_all(seq).map_err(|e| e.to_string())?;
        writeln!(w).map_err(|e| e.to_string())?;
    }
    Ok(())
}
