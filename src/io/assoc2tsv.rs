use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::fmt::Write as _;

use crate::gfcore::SiteInfo as CoreSiteInfo;
use crate::gfreader::SiteInfo as PySiteInfo;
use crate::linalg::{
    chisq_from_beta_se_and_optional_plrt, format_chisq_value, sanitize_assoc_pvalue,
};
use crate::stats_common::AsyncTsvWriter;

pub(crate) type AssocTsvMetadata = (Vec<String>, Vec<i64>, Vec<String>, Vec<String>, Vec<String>);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AssocResultCols {
    Basic3,
    Plrt4,
    Lmm2_6,
}

impl AssocResultCols {
    #[inline]
    pub(crate) fn from_usize(result_cols: usize) -> Result<Self, String> {
        match result_cols {
            3 => Ok(Self::Basic3),
            4 => Ok(Self::Plrt4),
            6 => Ok(Self::Lmm2_6),
            other => Err(format!(
                "unsupported GWAS result column count: {other} (expected 3, 4, or 6)"
            )),
        }
    }

    #[inline]
    pub(crate) fn as_usize(self) -> usize {
        match self {
            Self::Basic3 => 3,
            Self::Plrt4 => 4,
            Self::Lmm2_6 => 6,
        }
    }

    #[inline]
    pub(crate) fn header(self) -> &'static [u8] {
        match self {
            Self::Basic3 => {
                b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n"
            }
            Self::Plrt4 => {
                b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tplrt\n"
            }
            Self::Lmm2_6 => {
                b"chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\tlambda\tml\tplrt\n"
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum AssocMissBlock<'a> {
    CountI64(&'a [i64]),
    CountUsize(&'a [usize]),
    Rate(&'a [f32]),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AssocArraysBlock<'a> {
    pub(crate) chrom: &'a [String],
    pub(crate) pos: &'a [i64],
    pub(crate) snp: &'a [String],
    pub(crate) allele0: &'a [String],
    pub(crate) allele1: &'a [String],
    pub(crate) maf: &'a [f32],
    pub(crate) miss: AssocMissBlock<'a>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AssocCoreSitesBlock<'a> {
    pub(crate) sites: &'a [CoreSiteInfo],
    pub(crate) maf: &'a [f32],
    pub(crate) miss: AssocMissBlock<'a>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum AssocResultLayout {
    ResultCols {
        schema: AssocResultCols,
        row_stride: usize,
    },
    PrecomputedChisqBasic3 {
        row_stride: usize,
        beta_col: usize,
        se_col: usize,
        chisq_col: usize,
        raw_p_col: usize,
    },
}

impl AssocResultLayout {
    #[inline]
    fn row_stride(self) -> usize {
        match self {
            Self::ResultCols { row_stride, .. } => row_stride,
            Self::PrecomputedChisqBasic3 { row_stride, .. } => row_stride,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum AssocMissValue {
    Count(i64),
    Rate(f32),
}

#[inline]
pub(crate) fn transform_alleles_by_model(
    ref_allele: &str,
    alt_allele: &str,
    model_key: &str,
) -> (String, String) {
    match model_key {
        "dom" => (
            format!("{ref_allele}{ref_allele}"),
            format!("{ref_allele}{alt_allele}/{alt_allele}{alt_allele}"),
        ),
        "rec" => (
            format!("{ref_allele}{alt_allele}/{ref_allele}{ref_allele}"),
            format!("{alt_allele}{alt_allele}"),
        ),
        "het" => (
            format!("{ref_allele}{ref_allele}/{alt_allele}{alt_allele}"),
            format!("{ref_allele}{alt_allele}"),
        ),
        _ => (ref_allele.to_string(), alt_allele.to_string()),
    }
}

pub(crate) fn resolve_assoc_tsv_metadata(
    bed_prefix: Option<&str>,
    chrom: Vec<String>,
    pos: Vec<i64>,
    snp: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    row_indices: Option<&[usize]>,
    expected_len: usize,
) -> Result<AssocTsvMetadata, String> {
    let all_empty = chrom.is_empty()
        && pos.is_empty()
        && snp.is_empty()
        && allele0.is_empty()
        && allele1.is_empty();
    if all_empty {
        let prefix = bed_prefix
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "empty TSV metadata requires non-empty bed_prefix".to_string())?;
        let (chrom2, pos2, snp2, allele02, allele12) =
            crate::gfcore::read_bim_columns(prefix, row_indices)?;
        if chrom2.len() != expected_len
            || pos2.len() != expected_len
            || snp2.len() != expected_len
            || allele02.len() != expected_len
            || allele12.len() != expected_len
        {
            return Err(format!(
                "BIM metadata length mismatch: expected={expected_len}, chrom={}, pos={}, snp={}, allele0={}, allele1={}",
                chrom2.len(),
                pos2.len(),
                snp2.len(),
                allele02.len(),
                allele12.len(),
            ));
        }
        let pos64 = pos2.into_iter().map(|v| v as i64).collect::<Vec<i64>>();
        return Ok((chrom2, pos64, snp2, allele02, allele12));
    }
    if chrom.len() != expected_len
        || pos.len() != expected_len
        || snp.len() != expected_len
        || allele0.len() != expected_len
        || allele1.len() != expected_len
    {
        return Err(format!(
            "TSV metadata length mismatch: rows={expected_len}, chrom={}, pos={}, snp={}, allele0={}, allele1={}",
            chrom.len(),
            pos.len(),
            snp.len(),
            allele0.len(),
            allele1.len()
        ));
    }
    Ok((chrom, pos, snp, allele0, allele1))
}

pub(crate) struct AssocTsvSink {
    path: String,
    result_cols: Option<AssocResultCols>,
    rows_written: usize,
    buffer_bytes: usize,
    queue_depth: usize,
    text_buf: String,
    writer: Option<AsyncTsvWriter>,
}

impl AssocTsvSink {
    #[inline]
    pub(crate) fn with_config(
        path: impl Into<String>,
        buffer_bytes: usize,
        queue_depth: usize,
    ) -> Self {
        let path = path.into();
        let cap = buffer_bytes.max(8 * 1024);
        Self {
            path,
            result_cols: None,
            rows_written: 0,
            buffer_bytes: cap,
            queue_depth: queue_depth.max(1),
            text_buf: String::with_capacity(cap),
            writer: None,
        }
    }

    #[inline]
    pub(crate) fn rows_written(&self) -> usize {
        self.rows_written
    }

    #[inline]
    pub(crate) fn text_buf_mut(&mut self) -> &mut String {
        &mut self.text_buf
    }

    #[inline]
    pub(crate) fn ensure_result_cols(
        &mut self,
        result_cols: usize,
    ) -> Result<AssocResultCols, String> {
        let schema = AssocResultCols::from_usize(result_cols)?;
        match self.result_cols {
            None => {
                self.result_cols = Some(schema);
                self.writer = Some(AsyncTsvWriter::with_config(
                    &self.path,
                    schema.header(),
                    64 * 1024 * 1024,
                    self.queue_depth,
                )?);
            }
            Some(prev) if prev != schema => {
                return Err(format!(
                    "inconsistent results columns across chunks: expected {}, got {}",
                    prev.as_usize(),
                    schema.as_usize()
                ));
            }
            Some(_) => {}
        }
        Ok(schema)
    }

    #[inline]
    pub(crate) fn add_rows(&mut self, rows: usize) {
        self.rows_written = self.rows_written.saturating_add(rows);
    }

    pub(crate) fn maybe_flush_text(&mut self) -> Result<(), String> {
        if self.text_buf.len() >= self.buffer_bytes {
            self.flush_text()?;
        }
        Ok(())
    }

    pub(crate) fn flush_text(&mut self) -> Result<(), String> {
        if self.text_buf.is_empty() {
            return Ok(());
        }
        let writer = self
            .writer
            .as_ref()
            .ok_or_else(|| "writer is closed".to_string())?;
        let payload =
            std::mem::replace(&mut self.text_buf, String::with_capacity(self.buffer_bytes))
                .into_bytes();
        writer.send(payload)
    }

    pub(crate) fn append_text(
        &mut self,
        text: String,
        result_cols: usize,
        rows: usize,
    ) -> Result<(), String> {
        if rows == 0 || text.is_empty() {
            return Ok(());
        }
        self.ensure_result_cols(result_cols)?;
        self.rows_written = self.rows_written.saturating_add(rows);
        let writer = self
            .writer
            .as_ref()
            .ok_or_else(|| "writer is closed".to_string())?;
        writer.send(text.into_bytes())
    }

    pub(crate) fn send_block(&mut self, data: Vec<u8>) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        let writer = self
            .writer
            .as_ref()
            .ok_or_else(|| "writer is closed".to_string())?;
        writer.send(data)
    }

    pub(crate) fn finish(&mut self) -> Result<(), String> {
        self.flush_text()?;
        if let Some(w) = self.writer.take() {
            w.finish()?;
        }
        Ok(())
    }
}

impl Drop for AssocTsvSink {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

#[inline]
fn append_assoc_row_text(
    text_buf: &mut String,
    result_cols: AssocResultCols,
    model_key: &str,
    site: &PySiteInfo,
    snp_name: &str,
    maf: f32,
    miss_count: i64,
    row: &[f64],
) {
    let (a0, a1) = transform_alleles_by_model(&site.ref_allele, &site.alt_allele, model_key);
    let beta = row[0];
    let se = row[1];
    let pwald = sanitize_assoc_pvalue(beta, se, row[2]);
    let chisq = chisq_from_beta_se_and_optional_plrt(beta, se, None);
    let chisq_txt = format_chisq_value(chisq);
    match result_cols {
        AssocResultCols::Lmm2_6 => {
            let _ = writeln!(
                text_buf,
                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.6e}\t{:.6e}\t{:.4e}",
                site.chrom,
                site.pos,
                snp_name,
                a0,
                a1,
                maf,
                miss_count,
                beta,
                se,
                chisq_txt,
                pwald,
                row[3],
                row[4],
                row[5]
            );
        }
        AssocResultCols::Plrt4 => {
            let _ = writeln!(
                text_buf,
                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}",
                site.chrom,
                site.pos,
                snp_name,
                a0,
                a1,
                maf,
                miss_count,
                beta,
                se,
                chisq_txt,
                pwald,
                row[3]
            );
        }
        AssocResultCols::Basic3 => {
            let _ = writeln!(
                text_buf,
                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}",
                site.chrom, site.pos, snp_name, a0, a1, maf, miss_count, beta, se, chisq_txt, pwald,
            );
        }
    }
}

#[inline]
fn append_assoc_row_from_fields(
    text_buf: &mut String,
    layout: AssocResultLayout,
    model_key: &str,
    chrom: &str,
    pos: i64,
    snp_name: &str,
    allele0: &str,
    allele1: &str,
    maf: f32,
    miss: AssocMissValue,
    row: &[f64],
) {
    let (a0, a1) = transform_alleles_by_model(allele0, allele1, model_key);
    match layout {
        AssocResultLayout::ResultCols { schema, .. } => {
            let beta = row[0];
            let se = row[1];
            let pwald = sanitize_assoc_pvalue(beta, se, row[2]);
            let chisq_txt =
                format_chisq_value(chisq_from_beta_se_and_optional_plrt(beta, se, None));
            match (schema, miss) {
                (AssocResultCols::Basic3, AssocMissValue::Count(miss_count)) => {
                    let _ = writeln!(
                        text_buf,
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}",
                        chrom, pos, snp_name, a0, a1, maf, miss_count, beta, se, chisq_txt, pwald,
                    );
                }
                (AssocResultCols::Basic3, AssocMissValue::Rate(miss_rate)) => {
                    let _ = writeln!(
                        text_buf,
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.4e}",
                        chrom, pos, snp_name, a0, a1, maf, miss_rate, beta, se, chisq_txt, pwald,
                    );
                }
                (AssocResultCols::Plrt4, AssocMissValue::Count(miss_count)) => {
                    let _ = writeln!(
                        text_buf,
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}",
                        chrom,
                        pos,
                        snp_name,
                        a0,
                        a1,
                        maf,
                        miss_count,
                        beta,
                        se,
                        chisq_txt,
                        pwald,
                        row[3],
                    );
                }
                (AssocResultCols::Plrt4, AssocMissValue::Rate(miss_rate)) => {
                    let _ = writeln!(
                        text_buf,
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.4e}",
                        chrom,
                        pos,
                        snp_name,
                        a0,
                        a1,
                        maf,
                        miss_rate,
                        beta,
                        se,
                        chisq_txt,
                        pwald,
                        row[3],
                    );
                }
                (AssocResultCols::Lmm2_6, AssocMissValue::Count(miss_count)) => {
                    let _ = writeln!(
                        text_buf,
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.6e}\t{:.6e}\t{:.4e}",
                        chrom, pos, snp_name, a0, a1, maf, miss_count, beta, se, chisq_txt, pwald, row[3], row[4], row[5],
                    );
                }
                (AssocResultCols::Lmm2_6, AssocMissValue::Rate(miss_rate)) => {
                    let _ = writeln!(
                        text_buf,
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.4e}\t{:.6e}\t{:.6e}\t{:.4e}",
                        chrom, pos, snp_name, a0, a1, maf, miss_rate, beta, se, chisq_txt, pwald, row[3], row[4], row[5],
                    );
                }
            }
        }
        AssocResultLayout::PrecomputedChisqBasic3 {
            beta_col,
            se_col,
            chisq_col,
            raw_p_col,
            ..
        } => {
            let beta = row[beta_col];
            let se = row[se_col];
            let pwald = sanitize_assoc_pvalue(beta, se, row[raw_p_col]);
            let chisq_txt = format_chisq_value(row[chisq_col]);
            match miss {
                AssocMissValue::Count(miss_count) => {
                    let _ = writeln!(
                        text_buf,
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.4e}",
                        chrom, pos, snp_name, a0, a1, maf, miss_count, beta, se, chisq_txt, pwald,
                    );
                }
                AssocMissValue::Rate(miss_rate) => {
                    let _ = writeln!(
                        text_buf,
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.4e}",
                        chrom, pos, snp_name, a0, a1, maf, miss_rate, beta, se, chisq_txt, pwald,
                    );
                }
            }
        }
    }
}

fn validate_assoc_arrays_block(meta: AssocArraysBlock<'_>, rows: usize) -> Result<(), String> {
    if meta.chrom.len() != rows
        || meta.pos.len() != rows
        || meta.snp.len() != rows
        || meta.allele0.len() != rows
        || meta.allele1.len() != rows
        || meta.maf.len() != rows
    {
        return Err(format!(
            "assoc TSV arrays length mismatch: rows={rows}, chrom={}, pos={}, snp={}, allele0={}, allele1={}, maf={}",
            meta.chrom.len(),
            meta.pos.len(),
            meta.snp.len(),
            meta.allele0.len(),
            meta.allele1.len(),
            meta.maf.len(),
        ));
    }
    let miss_len = match meta.miss {
        AssocMissBlock::CountI64(v) => v.len(),
        AssocMissBlock::CountUsize(v) => v.len(),
        AssocMissBlock::Rate(v) => v.len(),
    };
    if miss_len != rows {
        return Err(format!(
            "assoc TSV miss length mismatch: rows={rows}, miss={miss_len}"
        ));
    }
    Ok(())
}

fn validate_assoc_core_sites_block(
    meta: AssocCoreSitesBlock<'_>,
    rows: usize,
) -> Result<(), String> {
    if meta.sites.len() != rows || meta.maf.len() != rows {
        return Err(format!(
            "assoc TSV sites length mismatch: rows={rows}, sites={}, maf={}",
            meta.sites.len(),
            meta.maf.len(),
        ));
    }
    let miss_len = match meta.miss {
        AssocMissBlock::CountI64(v) => v.len(),
        AssocMissBlock::CountUsize(v) => v.len(),
        AssocMissBlock::Rate(v) => v.len(),
    };
    if miss_len != rows {
        return Err(format!(
            "assoc TSV miss length mismatch: rows={rows}, miss={miss_len}"
        ));
    }
    Ok(())
}

pub(crate) fn append_assoc_block_from_arrays(
    text_buf: &mut String,
    layout: AssocResultLayout,
    model_key: &str,
    meta: AssocArraysBlock<'_>,
    results: &[f64],
) -> Result<(), String> {
    let row_stride = layout.row_stride();
    if row_stride == 0 || results.len() % row_stride != 0 {
        return Err(format!(
            "assoc TSV result stride mismatch: len={}, row_stride={row_stride}",
            results.len()
        ));
    }
    let rows = results.len() / row_stride;
    validate_assoc_arrays_block(meta, rows)?;
    match meta.miss {
        AssocMissBlock::CountI64(miss) => {
            for row_idx in 0..rows {
                let base = row_idx * row_stride;
                append_assoc_row_from_fields(
                    text_buf,
                    layout,
                    model_key,
                    meta.chrom[row_idx].as_str(),
                    meta.pos[row_idx],
                    meta.snp[row_idx].as_str(),
                    meta.allele0[row_idx].as_str(),
                    meta.allele1[row_idx].as_str(),
                    meta.maf[row_idx],
                    AssocMissValue::Count(miss[row_idx]),
                    &results[base..base + row_stride],
                );
            }
        }
        AssocMissBlock::CountUsize(miss) => {
            for row_idx in 0..rows {
                let base = row_idx * row_stride;
                append_assoc_row_from_fields(
                    text_buf,
                    layout,
                    model_key,
                    meta.chrom[row_idx].as_str(),
                    meta.pos[row_idx],
                    meta.snp[row_idx].as_str(),
                    meta.allele0[row_idx].as_str(),
                    meta.allele1[row_idx].as_str(),
                    meta.maf[row_idx],
                    AssocMissValue::Count(miss[row_idx] as i64),
                    &results[base..base + row_stride],
                );
            }
        }
        AssocMissBlock::Rate(miss) => {
            for row_idx in 0..rows {
                let base = row_idx * row_stride;
                append_assoc_row_from_fields(
                    text_buf,
                    layout,
                    model_key,
                    meta.chrom[row_idx].as_str(),
                    meta.pos[row_idx],
                    meta.snp[row_idx].as_str(),
                    meta.allele0[row_idx].as_str(),
                    meta.allele1[row_idx].as_str(),
                    meta.maf[row_idx],
                    AssocMissValue::Rate(miss[row_idx]),
                    &results[base..base + row_stride],
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn append_assoc_block_from_core_sites(
    text_buf: &mut String,
    layout: AssocResultLayout,
    model_key: &str,
    meta: AssocCoreSitesBlock<'_>,
    results: &[f64],
) -> Result<(), String> {
    let row_stride = layout.row_stride();
    if row_stride == 0 || results.len() % row_stride != 0 {
        return Err(format!(
            "assoc TSV result stride mismatch: len={}, row_stride={row_stride}",
            results.len()
        ));
    }
    let rows = results.len() / row_stride;
    validate_assoc_core_sites_block(meta, rows)?;
    match meta.miss {
        AssocMissBlock::CountI64(miss) => {
            for row_idx in 0..rows {
                let base = row_idx * row_stride;
                let site = &meta.sites[row_idx];
                append_assoc_row_from_fields(
                    text_buf,
                    layout,
                    model_key,
                    site.chrom.as_str(),
                    site.pos as i64,
                    site.snp.as_str(),
                    site.ref_allele.as_str(),
                    site.alt_allele.as_str(),
                    meta.maf[row_idx],
                    AssocMissValue::Count(miss[row_idx]),
                    &results[base..base + row_stride],
                );
            }
        }
        AssocMissBlock::CountUsize(miss) => {
            for row_idx in 0..rows {
                let base = row_idx * row_stride;
                let site = &meta.sites[row_idx];
                append_assoc_row_from_fields(
                    text_buf,
                    layout,
                    model_key,
                    site.chrom.as_str(),
                    site.pos as i64,
                    site.snp.as_str(),
                    site.ref_allele.as_str(),
                    site.alt_allele.as_str(),
                    meta.maf[row_idx],
                    AssocMissValue::Count(miss[row_idx] as i64),
                    &results[base..base + row_stride],
                );
            }
        }
        AssocMissBlock::Rate(miss) => {
            for row_idx in 0..rows {
                let base = row_idx * row_stride;
                let site = &meta.sites[row_idx];
                append_assoc_row_from_fields(
                    text_buf,
                    layout,
                    model_key,
                    site.chrom.as_str(),
                    site.pos as i64,
                    site.snp.as_str(),
                    site.ref_allele.as_str(),
                    site.alt_allele.as_str(),
                    meta.maf[row_idx],
                    AssocMissValue::Rate(miss[row_idx]),
                    &results[base..base + row_stride],
                );
            }
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn send_text_buf(writer: &AsyncTsvWriter, text_buf: &mut String) -> Result<(), String> {
    if text_buf.is_empty() {
        return Ok(());
    }
    writer.send(std::mem::take(text_buf).into_bytes())
}

#[pyclass]
pub struct GwasAssocTsvWriter {
    sink: AssocTsvSink,
    model_key: String,
}

#[pymethods]
impl GwasAssocTsvWriter {
    #[new]
    #[pyo3(signature = (path, genetic_model="add"))]
    fn new(path: String, genetic_model: &str) -> PyResult<Self> {
        let model_key = genetic_model.trim().to_ascii_lowercase();
        if !matches!(model_key.as_str(), "add" | "dom" | "rec" | "het") {
            return Err(PyValueError::new_err(
                "genetic_model must be one of: add, dom, rec, het",
            ));
        }
        Ok(Self {
            sink: AssocTsvSink::with_config(path, 8 * 1024 * 1024, 16),
            model_key,
        })
    }

    fn write_chunk(
        &mut self,
        sites: Vec<PySiteInfo>,
        snp: Vec<String>,
        maf: PyReadonlyArray1<'_, f32>,
        miss: PyReadonlyArray1<'_, f32>,
        results: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<usize> {
        if sites.is_empty() {
            return Ok(0);
        }
        if snp.len() != sites.len() {
            return Err(PyValueError::new_err(format!(
                "snp length mismatch: snp={}, sites={}",
                snp.len(),
                sites.len()
            )));
        }

        let maf_arr = maf.as_slice()?;
        if maf_arr.len() != sites.len() {
            return Err(PyValueError::new_err(format!(
                "maf length mismatch: maf={}, sites={}",
                maf_arr.len(),
                sites.len()
            )));
        }
        let miss_arr = miss.as_slice()?;
        if miss_arr.len() != sites.len() {
            return Err(PyValueError::new_err(format!(
                "miss length mismatch: miss={}, sites={}",
                miss_arr.len(),
                sites.len()
            )));
        }

        let res = results.as_array();
        let shape = res.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err("results must be 2D"));
        }
        if shape[0] != sites.len() {
            return Err(PyValueError::new_err(format!(
                "results row mismatch: results={}, sites={}",
                shape[0],
                sites.len()
            )));
        }
        if shape[1] < 3 {
            return Err(PyValueError::new_err(format!(
                "results must have at least 3 columns, got {}",
                shape[1]
            )));
        }

        let schema = self
            .sink
            .ensure_result_cols(shape[1])
            .map_err(PyValueError::new_err)?;
        let text_buf = self.sink.text_buf_mut();
        for i in 0..sites.len() {
            let row = res.row(i);
            let row_vec = row.as_slice().ok_or_else(|| {
                PyValueError::new_err("results must be row-major contiguous for write_chunk")
            })?;
            append_assoc_row_text(
                text_buf,
                schema,
                &self.model_key,
                &sites[i],
                &snp[i],
                maf_arr[i],
                miss_arr[i].round() as i64,
                row_vec,
            );
        }
        self.sink.add_rows(sites.len());
        self.sink.maybe_flush_text().map_err(PyIOError::new_err)?;
        Ok(sites.len())
    }

    fn append_text(&mut self, text: String, has_plrt: bool, rows: usize) -> PyResult<()> {
        let result_cols = if has_plrt { 4usize } else { 3usize };
        self.sink
            .append_text(text, result_cols, rows)
            .map_err(PyIOError::new_err)
    }

    fn send_block(&mut self, data: Vec<u8>) -> PyResult<()> {
        self.sink.send_block(data).map_err(PyIOError::new_err)
    }

    fn flush(&mut self) -> PyResult<()> {
        self.sink.flush_text().map_err(PyIOError::new_err)
    }

    fn close(&mut self) -> PyResult<()> {
        self.sink.finish().map_err(PyIOError::new_err)
    }

    #[getter]
    fn rows_written(&self) -> usize {
        self.sink.rows_written()
    }
}
