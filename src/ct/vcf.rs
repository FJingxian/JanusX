use crate::ct::input::{GenotypeCall, PhaseSet, Site, SiteSource};
use crate::gfcore::open_text_maybe_gz;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub(crate) struct VcfStats {
    pub(crate) records_seen: usize,
    pub(crate) sites_emitted: usize,
    pub(crate) skipped_non_snp: usize,
    pub(crate) skipped_unsupported_gt: usize,
}

pub(crate) struct VcfSource {
    reader: Box<dyn BufRead + Send + Sync + 'static>,
    sample_ids: Vec<String>,
    selected_columns: Vec<usize>,
    full_sample_count: usize,
    line_number: usize,
    next_record_index: usize,
    previous_chrom: Option<String>,
    previous_pos: i64,
    seen_chromosomes: HashSet<String>,
    seen_variant_ids: HashSet<String>,
    finished: bool,
    stats: VcfStats,
}

impl VcfSource {
    pub(crate) fn open(path: &str, selected_ids: &[String]) -> Result<Self, String> {
        let mut reader = open_text_maybe_gz(Path::new(path))?;
        let mut line = String::new();
        let mut line_number = 0usize;
        let header = loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| format!("read VCF header from {path}: {e}"))?;
            if n == 0 {
                return Err(format!("no #CHROM header found in VCF {path}"));
            }
            line_number += 1;
            if line.starts_with("#CHROM\t") || line.trim_end() == "#CHROM" {
                break line
                    .trim_end()
                    .split('\t')
                    .map(str::to_string)
                    .collect::<Vec<_>>();
            }
            if !line.starts_with('#') {
                return Err(format!(
                    "VCF data appeared before #CHROM header at line {line_number}"
                ));
            }
        };
        if header.len() < 10 || header.get(8).map(String::as_str) != Some("FORMAT") {
            return Err(format!("VCF #CHROM header has no sample columns: {path}"));
        }
        let full_sample_count = header.len() - 9;
        let all_ids = &header[9..];
        let mut id_to_column = HashMap::with_capacity(full_sample_count);
        for (sample_index, id) in all_ids.iter().enumerate() {
            if id.trim().is_empty() {
                return Err(format!(
                    "empty sample ID in VCF header at column {}",
                    sample_index + 10
                ));
            }
            if id_to_column.insert(id.clone(), sample_index + 9).is_some() {
                return Err(format!("duplicate VCF sample ID '{id}'"));
            }
        }

        let mut selected = Vec::new();
        let mut selected_columns = Vec::new();
        let requested_ids = if selected_ids.is_empty() {
            all_ids.to_vec()
        } else {
            selected_ids.to_vec()
        };
        let mut requested_seen = HashSet::with_capacity(requested_ids.len());
        for id in requested_ids {
            if !requested_seen.insert(id.clone()) {
                return Err(format!("duplicate selected sample ID '{id}'"));
            }
            let column = id_to_column
                .get(&id)
                .copied()
                .ok_or_else(|| format!("selected sample ID '{id}' is absent from VCF"))?;
            selected.push(id);
            selected_columns.push(column);
        }
        if selected.is_empty() {
            return Err("VCF contains no selected samples".to_string());
        }

        Ok(Self {
            reader,
            sample_ids: selected,
            selected_columns,
            full_sample_count,
            line_number,
            next_record_index: 0,
            previous_chrom: None,
            previous_pos: 0,
            seen_chromosomes: HashSet::new(),
            seen_variant_ids: HashSet::new(),
            finished: false,
            stats: VcfStats::default(),
        })
    }

    pub(crate) fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    pub(crate) fn stats(&self) -> &VcfStats {
        &self.stats
    }

    pub(crate) fn next_site(&mut self) -> Result<Option<Site>, String> {
        if self.finished {
            return Ok(None);
        }
        let mut line = String::new();
        loop {
            line.clear();
            let read = self
                .reader
                .read_line(&mut line)
                .map_err(|e| format!("read VCF record after line {}: {e}", self.line_number))?;
            if read == 0 {
                self.finished = true;
                return Ok(None);
            }
            self.line_number += 1;
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let record_index = self.next_record_index;
            self.next_record_index += 1;
            self.stats.records_seen += 1;
            let fields = line
                .trim_end_matches(['\r', '\n'])
                .split('\t')
                .collect::<Vec<_>>();
            if fields.len() != self.full_sample_count + 9 {
                return Err(format!(
                    "malformed VCF record at line {}: expected {} columns, found {}",
                    self.line_number,
                    self.full_sample_count + 9,
                    fields.len()
                ));
            }
            let chrom = fields[0].trim();
            if chrom.is_empty() || chrom.contains(char::is_whitespace) {
                return Err(format!(
                    "invalid chromosome at VCF line {}",
                    self.line_number
                ));
            }
            let pos = fields[1]
                .parse::<i64>()
                .map_err(|e| format!("invalid POS at VCF line {}: {e}", self.line_number))?;
            if pos <= 0 {
                return Err(format!(
                    "POS must be positive at VCF line {}",
                    self.line_number
                ));
            }
            self.validate_order(chrom, pos)?;

            let raw_id = fields[2];
            let id = if raw_id.is_empty() || raw_id == "." {
                format!("{chrom}_{pos}")
            } else {
                raw_id.to_string()
            };
            if !self.seen_variant_ids.insert(id.clone()) {
                return Err(format!(
                    "duplicate variant ID '{id}' at VCF line {}",
                    self.line_number
                ));
            }

            let allele0 = fields[3].to_ascii_uppercase();
            let allele1 = fields[4].to_ascii_uppercase();
            if !is_simple_biallelic_snp(&allele0, &allele1) {
                self.stats.skipped_non_snp += 1;
                continue;
            }

            let format_fields = fields[8].split(':').collect::<Vec<_>>();
            let gt_index = unique_format_index(&format_fields, "GT", self.line_number)?
                .ok_or_else(|| format!("VCF FORMAT lacks GT at line {}", self.line_number))?;
            let ps_index = unique_format_index(&format_fields, "PS", self.line_number)?;
            let mut calls = Vec::with_capacity(self.selected_columns.len());
            let mut unsupported_gt = false;
            for &column in &self.selected_columns {
                let sample_fields = fields[column].split(':').collect::<Vec<_>>();
                let gt = sample_fields.get(gt_index).copied().unwrap_or("");
                let ps = ps_index
                    .and_then(|index| sample_fields.get(index).copied())
                    .filter(|value| !value.is_empty() && *value != ".");
                match parse_gt(gt, ps) {
                    ParsedGt::Call(call) => calls.push(call),
                    ParsedGt::Unsupported => {
                        unsupported_gt = true;
                        break;
                    }
                }
            }
            if unsupported_gt {
                self.stats.skipped_unsupported_gt += 1;
                continue;
            }
            self.stats.sites_emitted += 1;
            return Ok(Some(Site {
                index: record_index,
                chrom: chrom.to_string(),
                pos,
                id,
                allele0,
                allele1,
                calls,
            }));
        }
    }

    fn validate_order(&mut self, chrom: &str, pos: i64) -> Result<(), String> {
        match self.previous_chrom.as_deref() {
            None => {
                self.seen_chromosomes.insert(chrom.to_string());
            }
            Some(previous) if previous == chrom => {
                if pos < self.previous_pos {
                    return Err(format!(
                        "VCF positions must be nondecreasing within chromosome {chrom}: {} after {} at line {}",
                        pos, self.previous_pos, self.line_number
                    ));
                }
            }
            Some(_) => {
                if !self.seen_chromosomes.insert(chrom.to_string()) {
                    return Err(format!(
                        "VCF chromosome '{chrom}' occurs in more than one block at line {}",
                        self.line_number
                    ));
                }
            }
        }
        self.previous_chrom = Some(chrom.to_string());
        self.previous_pos = pos;
        Ok(())
    }
}

impl SiteSource for VcfSource {
    fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    fn next_site(&mut self) -> Result<Option<Site>, String> {
        VcfSource::next_site(self)
    }
}

enum ParsedGt {
    Call(GenotypeCall),
    Unsupported,
}

fn parse_gt(gt: &str, ps: Option<&str>) -> ParsedGt {
    if gt == "." || gt.is_empty() {
        return ParsedGt::Call(GenotypeCall {
            dosage: None,
            alt_on_hap0: None,
            phase_set: PhaseSet::Unknown,
        });
    }
    let delimiter = if gt.contains('|') {
        '|'
    } else if gt.contains('/') {
        '/'
    } else {
        return ParsedGt::Unsupported;
    };
    if gt.chars().filter(|&c| c == delimiter).count() != 1
        || gt.contains(if delimiter == '|' { '/' } else { '|' })
    {
        return ParsedGt::Unsupported;
    }
    let mut alleles = gt.split(delimiter);
    let first = alleles.next().unwrap_or("");
    let second = alleles.next().unwrap_or("");
    if alleles.next().is_some() || first.is_empty() || second.is_empty() {
        return ParsedGt::Unsupported;
    }
    if first == "." || second == "." {
        return ParsedGt::Call(GenotypeCall {
            dosage: None,
            alt_on_hap0: None,
            phase_set: PhaseSet::Unknown,
        });
    }
    let Ok(a0) = first.parse::<u8>() else {
        return ParsedGt::Unsupported;
    };
    let Ok(a1) = second.parse::<u8>() else {
        return ParsedGt::Unsupported;
    };
    if a0 > 1 || a1 > 1 {
        return ParsedGt::Unsupported;
    }
    let dosage = a0 + a1;
    let is_phased_het = delimiter == '|' && dosage == 1;
    let phase_set = if is_phased_het {
        match ps {
            Some(value) => PhaseSet::Explicit(value.to_string()),
            None => PhaseSet::Implicit,
        }
    } else {
        PhaseSet::Unknown
    };
    let alt_on_hap0 = is_phased_het.then_some(a0 == 1);
    ParsedGt::Call(GenotypeCall {
        dosage: Some(dosage),
        alt_on_hap0,
        phase_set,
    })
}

fn is_simple_biallelic_snp(ref_allele: &str, alt_allele: &str) -> bool {
    fn base(s: &str) -> bool {
        matches!(s, "A" | "C" | "G" | "T")
    }
    base(ref_allele) && base(alt_allele) && ref_allele != alt_allele
}

fn unique_format_index(
    fields: &[&str],
    key: &str,
    line_number: usize,
) -> Result<Option<usize>, String> {
    let mut found = None;
    for (i, field) in fields.iter().enumerate() {
        if *field == key {
            if found.replace(i).is_some() {
                return Err(format!(
                    "duplicate {key} field in FORMAT at line {line_number}"
                ));
            }
        }
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::VcfSource;
    use crate::ct::input::PhaseSet;
    use crate::ct::packed::count_pair;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);

    fn path(suffix: &str) -> PathBuf {
        let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("janusx-ct-vcf-{}-{id}{suffix}", std::process::id()))
    }

    fn write_plain(contents: &str) -> PathBuf {
        let path = path(".vcf");
        fs::write(&path, contents).unwrap();
        path
    }

    fn write_gzip(contents: &[u8], truncate_trailer: bool) -> PathBuf {
        let path = path(".vcf.gz");
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(contents).unwrap();
        let mut bytes = encoder.finish().unwrap();
        if truncate_trailer {
            bytes.truncate(bytes.len() - 4);
        }
        fs::write(&path, bytes).unwrap();
        path
    }

    fn header(samples: &str) -> String {
        format!(
            "##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{samples}\n"
        )
    }

    #[test]
    fn reads_gt_at_its_declared_format_position_and_keeps_dosage_for_unphased_homozygotes() {
        let mut text = header("s1\ts2");
        text.push_str("1\t100\trs1\tA\tG\t.\tPASS\t.\tDP:GT:PS\t35:0|1:7\t40:1|0:7\n");
        text.push_str("1\t150\trs2\tC\tT\t.\tPASS\t.\tGT:DP:PS\t0|1:40:7\t1|0:40:8\n");
        text.push_str("1\t200\trs3\tG\tA\t.\tPASS\t.\tDP:GT\t20:0/0\t20:1/1\n");
        let path = write_plain(&text);
        let mut source = VcfSource::open(path.to_str().unwrap(), &[]).unwrap();
        assert_eq!(source.sample_ids(), &["s1", "s2"]);
        let first = source.next_site().unwrap().unwrap();
        let second = source.next_site().unwrap().unwrap();
        let third = source.next_site().unwrap().unwrap();
        assert_eq!(first.calls[0].dosage, Some(1));
        assert_eq!(first.calls[0].alt_on_hap0, Some(false));
        assert_eq!(
            first.calls[0].phase_set,
            PhaseSet::Explicit("7".to_string())
        );
        assert_eq!(first.calls[1].alt_on_hap0, Some(true));
        assert_eq!(second.calls[0].alt_on_hap0, Some(false));
        assert_eq!(
            second.calls[0].phase_set,
            PhaseSet::Explicit("7".to_string())
        );
        assert_eq!(second.calls[1].alt_on_hap0, Some(true));
        assert_eq!(
            second.calls[1].phase_set,
            PhaseSet::Explicit("8".to_string())
        );
        let pair = count_pair(&first.calls, &second.calls).unwrap();
        assert_eq!(pair.cis_observed(), 1);
        assert_eq!(pair.phase_unknown, 1);
        assert_eq!(third.calls[0].dosage, Some(0));
        assert_eq!(third.calls[1].dosage, Some(2));
        assert_eq!(pair.genotype_missing, 0);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn missing_ps_is_implicit_but_unphased_double_heterozygotes_are_unknown() {
        let mut text = header("s1\ts2");
        text.push_str("1\t10\ta\tA\tG\t.\tPASS\t.\tGT\t0|1\t0/1\n");
        text.push_str("1\t20\tb\tC\tT\t.\tPASS\t.\tGT\t0|1\t1|0\n");
        let path = write_plain(&text);
        let mut source = VcfSource::open(path.to_str().unwrap(), &[]).unwrap();
        let left = source.next_site().unwrap().unwrap();
        let right = source.next_site().unwrap().unwrap();
        let pair = count_pair(&left.calls, &right.calls).unwrap();
        assert_eq!(pair.cis_observed(), 1);
        assert_eq!(pair.phase_unknown, 1);
        assert_eq!(left.calls[0].phase_set, PhaseSet::Implicit);
        assert_eq!(left.calls[1].alt_on_hap0, None);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn partial_missing_calls_remain_genotype_missing_and_multiallelic_sites_are_skipped() {
        let mut text = header("s1");
        text.push_str("1\t10\tmulti\tA\tG,T\t.\tPASS\t.\tGT\t0|1\n");
        text.push_str("1\t20\tpartial\tC\tT\t.\tPASS\t.\tGT\t./1\n");
        let path = write_plain(&text);
        let mut source = VcfSource::open(path.to_str().unwrap(), &[]).unwrap();
        let site = source.next_site().unwrap().unwrap();
        assert_eq!(site.id, "partial");
        assert_eq!(site.index, 1);
        assert_eq!(site.calls[0].dosage, None);
        assert_eq!(source.next_site().unwrap(), None);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn selected_samples_keep_requested_order_and_unknown_ids_are_rejected() {
        let text = format!(
            "{}1\t1\trs1\tA\tG\t.\tPASS\t.\tGT\t0|0\t1|1\n",
            header("s1\ts2")
        );
        let path = write_plain(&text);
        let mut source =
            VcfSource::open(path.to_str().unwrap(), &["s2".into(), "s1".into()]).unwrap();
        assert_eq!(source.sample_ids(), &["s2", "s1"]);
        let site = source.next_site().unwrap().unwrap();
        assert_eq!(site.calls[0].dosage, Some(2));
        assert_eq!(site.calls[1].dosage, Some(0));
        assert!(VcfSource::open(path.to_str().unwrap(), &["absent".into()]).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn duplicate_samples_variant_ids_and_unsorted_chromosomes_are_errors() {
        let duplicate_samples = write_plain(&header("s1\ts1"));
        assert!(VcfSource::open(duplicate_samples.to_str().unwrap(), &[]).is_err());
        fs::remove_file(duplicate_samples).unwrap();

        let mut duplicate_ids = header("s1");
        duplicate_ids.push_str("1\t10\trs\tA\tG\t.\tPASS\t.\tGT\t0|0\n");
        duplicate_ids.push_str("1\t20\trs\tC\tT\t.\tPASS\t.\tGT\t0|0\n");
        let path = write_plain(&duplicate_ids);
        let mut source = VcfSource::open(path.to_str().unwrap(), &[]).unwrap();
        source.next_site().unwrap().unwrap();
        assert!(source
            .next_site()
            .unwrap_err()
            .contains("duplicate variant ID"));
        fs::remove_file(path).unwrap();

        let mut unsorted = header("s1");
        unsorted.push_str("1\t20\ta\tA\tG\t.\tPASS\t.\tGT\t0|0\n");
        unsorted.push_str("1\t10\tb\tC\tT\t.\tPASS\t.\tGT\t0|0\n");
        unsorted.push_str("2\t1\tc\tG\tA\t.\tPASS\t.\tGT\t0|0\n");
        unsorted.push_str("1\t30\td\tT\tC\t.\tPASS\t.\tGT\t0|0\n");
        let path = write_plain(&unsorted);
        let mut source = VcfSource::open(path.to_str().unwrap(), &[]).unwrap();
        source.next_site().unwrap().unwrap();
        assert!(source.next_site().unwrap_err().contains("nondecreasing"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn malformed_records_and_truncated_gzip_propagate_read_errors() {
        let mut malformed = header("s1");
        malformed.push_str("1\t10\tbad\tA\tG\t.\tPASS\t.\n");
        let path = write_plain(&malformed);
        let mut source = VcfSource::open(path.to_str().unwrap(), &[]).unwrap();
        assert!(source.next_site().unwrap_err().contains("record"));
        fs::remove_file(path).unwrap();

        let gzip = write_gzip(header("s1").as_bytes(), true);
        let mut source = VcfSource::open(gzip.to_str().unwrap(), &[]).unwrap();
        assert!(source.next_site().is_err());
        fs::remove_file(gzip).unwrap();
    }
}
