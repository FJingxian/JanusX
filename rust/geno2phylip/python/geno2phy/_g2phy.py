import os
import typing
import .geno2phy as gp

def geno2phy(genofile:os.PathLike,filetype:typing.Literal['vcf','bfile','npy'],out:os.PathLike=".",outprefix:str=None, min_samples_locus:int=1,
             outgroup:str=None,
             write_phylip:bool=True, write_fasta:bool=False,
             write_nexus:bool=False, write_nexus_binary:bool=False,
             resolve_het:bool=False, used_sites:bool=True):
    """
    Convert genotype data (NPZ, PLINK bfile, or VCF) into phylogenetic
    alignment formats using the Rust-based geno2phy backend.

    This function creates a ConvertOptions instance, configures output
    options, selects the appropriate input reader, and writes one or more
    alignment files (PHYLIP, FASTA, NEXUS, or binary NEXUS).

    Parameters
    ----------
    genofile : path-like
        Base path of the genotype dataset, without extension.
        For example, if the files are:
            sample.npz, sample.snp, sample.idv
        then genofile="sample".

    filetype : str
        Input genotype format. Must be one of:
            "npy"   - expects genofile.npz, genofile.snp, genofile.idv
            "bfile" - expects genofile.bed, genofile.bim, genofile.fam
            "vcf"   - expects genofile.vcf.gz (VCF or bgzipped VCF)

    out : path-like, optional
        Output directory for alignment files. Default is ".".

    outprefix : str, optional
        Prefix for output filenames. If None, the basename of genofile is used.

    min_samples_locus : int, optional
        Minimum number of non-missing samples required for a SNP to be
        retained in the alignment. Default is 1.

    outgroup : str, optional
        If provided, this sample name will be placed first in all output
        alignments.

    write_phylip : bool, optional
        Whether to write a PHYLIP alignment file. Default is True.

    write_fasta : bool, optional
        Whether to write a FASTA alignment file. Default is False.

    write_nexus : bool, optional
        Whether to write a standard DNA NEXUS file. Default is False.

    write_nexus_binary : bool, optional
        Whether to write a binary NEXUS file suitable for SNAPP.
        Only diploid biallelic SNPs will be included. Default is False.

    resolve_het : bool, optional
        If False (default), heterozygous genotypes (0/1) are represented
        as IUPAC ambiguity codes. If True, heterozygotes are randomly phased
        to one allele.

    used_sites : bool, optional
        Whether to write a table of all SNP sites that passed filtering
        (CHROM, POS, number of non-missing samples). Default is True.

    Notes
    -----
    Multi-allelic VCF sites are ignored.
    Non-diploid genotypes are treated as missing.
    The conversion is optimized for large genotype matrices.

    Returns
    -------
    None
        The function writes output files to disk.

    Examples
    --------
    Convert an NPZ genotype dataset:

    >>> geno2phy("data/mouse", filetype="npy", out="outdir")

    Convert a PLINK bfile with an outgroup:

    >>> geno2phy("data/maize", filetype="bfile", outgroup="B73")

    Convert a compressed VCF into FASTA:

    >>> geno2phy("panel", filetype="vcf", write_phylip=False, write_fasta=True)
    """
    # 1) Create option object
    outprefix = outprefix if outprefix is not None else os.path.basename(genofile)
    opts = gp.ConvertOptions(outprefix)
    opts.out_dir = out

    opts.min_samples_locus = min_samples_locus
    opts.outgroup = outgroup

    opts.write_phylip = write_phylip
    opts.write_fasta = write_fasta
    opts.write_nexus = write_nexus
    opts.write_nexus_binary = write_nexus_binary
    opts.resolve_het = resolve_het
    opts.used_sites = used_sites

    # 2) Call backend according to filetype
    if filetype == "npy":
        gp.convert_from_npz(f"{genofile}.npz",
                            f"{genofile}.snp",
                            f"{genofile}.idv",
                            opts)

    elif filetype == "bfile":
        gp.convert_from_plink(f"{genofile}.bed",
                              f"{genofile}.bim",
                              f"{genofile}.fam",
                              opts)

    elif filetype == "vcf":
        # ------ New smart VCF detection logic ------
        vcf_plain = f"{genofile}.vcf"
        vcf_gz = f"{genofile}.vcf.gz"

        if os.path.exists(vcf_plain):
            # priority: uncompressed VCF
            gp.convert_from_vcf(vcf_plain, opts)

        elif os.path.exists(vcf_gz):
            gp.convert_from_vcf(vcf_gz, opts)

        else:
            raise FileNotFoundError(
                f"Neither {vcf_plain} nor {vcf_gz} exist. "
                "Please supply a valid VCF file."
            )

    else:
        raise ValueError('filetype must be one of {"npy", "bfile", "vcf"}')