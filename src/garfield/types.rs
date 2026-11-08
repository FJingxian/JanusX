use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct GarfieldUnitSpan {
    pub(super) chrom: String,
    pub(super) bp_start: i32,
    pub(super) bp_end: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GarfieldLogicSiteMode {
    Bin,
}

impl GarfieldLogicSiteMode {
    #[inline]
    fn label(self) -> &'static str {
        match self {
            Self::Bin => "BIN",
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum GarfieldLogicSiteBacking {
    Owned {
        chrom: Arc<str>,
        snp: Arc<str>,
        ref_allele: Arc<str>,
        alt_allele: Arc<str>,
    },
    SharedMax6 {
        source_sites: Arc<CompactMax6SiteMetadata>,
        source_index: usize,
        flipped: bool,
        snp_override: Option<Arc<str>>,
    },
}

#[derive(Clone, Debug)]
pub(super) struct GarfieldLogicSite {
    pub(super) backing: GarfieldLogicSiteBacking,
    pub(super) pos: i32,
    pub(super) mode: GarfieldLogicSiteMode,
}

impl GarfieldLogicSite {
    #[inline]
    pub(super) fn chrom(&self) -> &str {
        match &self.backing {
            GarfieldLogicSiteBacking::Owned { chrom, .. } => chrom.as_ref(),
            GarfieldLogicSiteBacking::SharedMax6 {
                source_sites,
                source_index,
                ..
            } => {
                source_sites
                    .fields(*source_index)
                    .expect("validated Max6 compact site index")
                    .chrom
            }
        }
    }

    #[inline]
    pub(super) fn snp(&self) -> &str {
        match &self.backing {
            GarfieldLogicSiteBacking::Owned { snp, .. } => snp.as_ref(),
            GarfieldLogicSiteBacking::SharedMax6 {
                source_sites,
                source_index,
                snp_override,
                ..
            } => snp_override.as_deref().unwrap_or_else(|| {
                source_sites
                    .fields(*source_index)
                    .expect("validated Max6 compact site index")
                    .snp
                    .trim()
            }),
        }
    }

    #[inline]
    pub(super) fn ref_allele(&self) -> &str {
        match &self.backing {
            GarfieldLogicSiteBacking::Owned { ref_allele, .. } => ref_allele.as_ref(),
            GarfieldLogicSiteBacking::SharedMax6 {
                source_sites,
                source_index,
                flipped,
                ..
            } => {
                let site = source_sites
                    .fields(*source_index)
                    .expect("validated Max6 compact site index");
                if *flipped {
                    site.alt_allele
                } else {
                    site.ref_allele
                }
            }
        }
    }

    #[inline]
    pub(super) fn alt_allele(&self) -> &str {
        match &self.backing {
            GarfieldLogicSiteBacking::Owned { alt_allele, .. } => alt_allele.as_ref(),
            GarfieldLogicSiteBacking::SharedMax6 {
                source_sites,
                source_index,
                flipped,
                ..
            } => {
                let site = source_sites
                    .fields(*source_index)
                    .expect("validated Max6 compact site index");
                if *flipped {
                    site.ref_allele
                } else {
                    site.alt_allele
                }
            }
        }
    }

    #[inline]
    pub(super) fn owned(
        chrom: impl Into<Arc<str>>,
        pos: i32,
        snp: impl Into<Arc<str>>,
        ref_allele: impl Into<Arc<str>>,
        alt_allele: impl Into<Arc<str>>,
        mode: GarfieldLogicSiteMode,
    ) -> Self {
        Self {
            backing: GarfieldLogicSiteBacking::Owned {
                chrom: chrom.into(),
                snp: snp.into(),
                ref_allele: ref_allele.into(),
                alt_allele: alt_allele.into(),
            },
            pos,
            mode,
        }
    }
}

impl GarfieldChromPosSite for SiteInfo {
    #[inline]
    fn garfield_chrom(&self) -> &str {
        self.chrom.as_str()
    }

    #[inline]
    fn garfield_pos(&self) -> i32 {
        self.pos
    }

    #[inline]
    fn garfield_snp_name(&self) -> &str {
        self.snp.as_str()
    }
}

impl GarfieldDisplaySite for SiteInfo {
    #[inline]
    fn garfield_model_label(&self) -> &'static str {
        "BIN"
    }

    #[inline]
    fn garfield_ref_allele(&self) -> &str {
        self.ref_allele
            .split('|')
            .next()
            .unwrap_or(self.ref_allele.as_str())
            .trim()
    }

    #[inline]
    fn garfield_alt_allele(&self) -> &str {
        self.alt_allele
            .split('|')
            .next()
            .unwrap_or(self.alt_allele.as_str())
            .trim()
    }
}

impl GarfieldChromPosSite for GarfieldLogicSite {
    #[inline]
    fn garfield_chrom(&self) -> &str {
        self.chrom()
    }

    #[inline]
    fn garfield_pos(&self) -> i32 {
        self.pos
    }

    #[inline]
    fn garfield_snp_name(&self) -> &str {
        self.snp()
    }
}

impl GarfieldDisplaySite for GarfieldLogicSite {
    #[inline]
    fn garfield_model_label(&self) -> &'static str {
        self.mode.label()
    }

    #[inline]
    fn garfield_ref_allele(&self) -> &str {
        self.ref_allele()
    }

    #[inline]
    fn garfield_alt_allele(&self) -> &str {
        self.alt_allele()
    }
}

#[derive(Clone, Debug)]
pub(super) struct GarfieldLogicUnit {
    pub(super) label: String,
    pub(super) indices: Vec<usize>,
    pub(super) spans: Vec<GarfieldUnitSpan>,
}

#[derive(Clone, Debug)]
pub(super) struct Max6WindowPositionSite {
    pub(super) chrom: Arc<str>,
    pub(super) pos: i32,
}

impl GarfieldChromPosSite for Max6WindowPositionSite {
    #[inline]
    fn garfield_chrom(&self) -> &str {
        self.chrom.as_ref()
    }

    #[inline]
    fn garfield_pos(&self) -> i32 {
        self.pos
    }

    #[inline]
    fn garfield_snp_name(&self) -> &str {
        ""
    }
}
