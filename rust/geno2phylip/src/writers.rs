use anyhow::Result;
use std::{fs::File, io::{BufWriter, Write}, path::Path};

pub fn write_fasta(path: &Path, samples: &[String], seqs: &[Vec<u8>]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for (name, seq) in samples.iter().zip(seqs.iter()) {
        writeln!(w, ">{}", name)?;
        w.write_all(seq)?;
        writeln!(w)?;
    }
    Ok(())
}

pub fn write_phylip(path: &Path, samples: &[String], seqs: &[Vec<u8>]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    let n = samples.len();
    let len = seqs.first().map(|x| x.len()).unwrap_or(0);
    writeln!(w, "{} {}", n, len)?;

    let max_name = samples.iter().map(|s| s.len()).max().unwrap_or(0);
    for (name, seq) in samples.iter().zip(seqs.iter()) {
        let padding = " ".repeat(max_name + 3 - name.len());
        write!(w, "{}{}", name, padding)?;
        w.write_all(seq)?;
        writeln!(w)?;
    }
    Ok(())
}

pub fn write_nexus_dna(path: &Path, samples: &[String], seqs: &[Vec<u8>]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    let n = samples.len();
    let len = seqs.first().map(|x| x.len()).unwrap_or(0);

    writeln!(w, "#NEXUS\n")?;
    writeln!(w, "BEGIN DATA;")?;
    writeln!(w, "\tDIMENSIONS NTAX={} NCHAR={};", n, len)?;
    writeln!(w, "\tFORMAT DATATYPE=DNA MISSING=N GAP=- ;")?;
    writeln!(w, "MATRIX")?;

    let max_name = samples.iter().map(|s| s.len()).max().unwrap_or(0);
    for (name, seq) in samples.iter().zip(seqs.iter()) {
        let padding = " ".repeat(max_name + 3 - name.len());
        write!(w, "{}{}", name, padding)?;
        w.write_all(seq)?;
        writeln!(w)?;
    }

    writeln!(w, ";\nEND;")?;
    Ok(())
}

pub fn write_nexus_binary(path: &Path, samples: &[String], seqs_bin: &[Vec<u8>]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    let n = samples.len();
    let len = seqs_bin.first().map(|x| x.len()).unwrap_or(0);

    writeln!(w, "#NEXUS\n")?;
    writeln!(w, "BEGIN DATA;")?;
    writeln!(w, "\tDIMENSIONS NTAX={} NCHAR={};", n, len)?;
    writeln!(w, "\tFORMAT DATATYPE=SNP MISSING=? GAP=- ;")?;
    writeln!(w, "MATRIX")?;

    let max_name = samples.iter().map(|s| s.len()).max().unwrap_or(0);
    for (name, seq) in samples.iter().zip(seqs_bin.iter()) {
        let padding = " ".repeat(max_name + 3 - name.len());
        write!(w, "{}{}", name, padding)?;
        w.write_all(seq)?;
        writeln!(w)?;
    }

    writeln!(w, ";\nEND;")?;
    Ok(())
}