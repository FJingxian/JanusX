use crate::ct::packed::PackedGenotypes;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PhaseSet {
    /// VCF sites on the chromosome with no explicit PS share an implicit set.
    Implicit,
    Explicit(String),
    /// The heterozygote is unphased or cannot be linked to another site.
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GenotypeCall {
    pub(crate) dosage: Option<u8>,
    /// `true` means allele1 lies on homolog 0; meaningful only for dosage 1.
    pub(crate) alt_on_hap0: Option<bool>,
    pub(crate) phase_set: PhaseSet,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Site {
    pub(crate) index: usize,
    pub(crate) chrom: String,
    pub(crate) pos: i64,
    pub(crate) id: String,
    pub(crate) allele0: String,
    pub(crate) allele1: String,
    pub(crate) calls: Vec<GenotypeCall>,
}

pub(crate) trait SiteSource {
    fn sample_ids(&self) -> &[String];
    fn next_site(&mut self) -> Result<Option<Site>, String>;

    fn next_scanned_site(&mut self) -> Result<Option<ScannedSite>, String> {
        let Some(site) = self.next_site()? else {
            return Ok(None);
        };
        Ok(Some(ScannedSite {
            site,
            packed_calls: None,
        }))
    }
}

pub(crate) struct ScannedSite {
    pub(crate) site: Site,
    pub(crate) packed_calls: Option<PackedGenotypes>,
}
