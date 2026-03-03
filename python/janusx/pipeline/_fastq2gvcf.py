from pathlib import Path
import shlex
from typing import List, Sequence, Union, Literal

Pathlike = Union[Path,str]


def _q(path: Pathlike) -> str:
    return shlex.quote(str(path))

def _singularity_prefix(singularity: str) -> str:
    singularity = str(singularity or "").strip()
    return f"{singularity} " if singularity else ""

# def node(ifiles:List[Pathlike],ofiles:List[Pathlike],cmd:str) -> tuple[str, Literal[-1,0,1]]:
#     '''
#     Docstring of node
    
#     :param ifiles: Input files
#     :type ifiles: List[PathLike]
#     :param ofiles: Output files
#     :type ofiles: List[PathLike]
#     :param cmd: One cmd of pipelines
#     :type cmd: str
#     :param nodename: Nodename
#     :type nodename: str
#     :return: cmd, status number {-1: back to pre-node, 0: run cmd, 1: go to next-node}
#     :rtype: tuple[str, Literal[-1, 0, 1]]
#     '''
#     outexists = all(Path(ofile).exists() for ofile in ofiles)
#     inexists = all(Path(ifile).exists() for ifile in ifiles)
#     if outexists:
#         return f"Go to next node, because output files is exists:\n"+'\n'.join(map(str,ofiles)), 1
#     else:
#         if inexists:
#             return f"Runing: {cmd}", 0
#         else:
#             return f"Back to previous node, because input files is not exists:\n"+'\n'.join(map(str,ifiles)), -1

def indexREF(reference:Pathlike, singularity: str = ""):
    """
    Build all basic indexes for a reference genome.

    This command string performs:
      1. `samtools faidx reference.fa`
      2. `bwa index reference.fa`
      3. `gatk CreateSequenceDictionary -R reference.fa -O reference.fa.dict`

    Parameters
    ----------
    reference : Pathlike
        Path to the reference FASTA file, e.g. "genome.fa".

    Returns
    -------
    str
        Shell command string that can be submitted to a scheduler or run via `os.system`.
    """
    reference = Path(reference)
    prefix = _singularity_prefix(singularity)
    samtoolsidx = f'{prefix}samtools faidx {reference}'
    bwaidx = f'{prefix}bwa index {reference}'
    outdict = f"{str(reference).replace('.fasta.gz','').replace('.fa.gz','').replace('.fasta','').replace('.fasta','')}.dict"
    gatkidx = f"{prefix}gatk CreateSequenceDictionary -R {reference} -O {outdict}"
    has_fai = Path(f"{reference}.fai").exists()
    has_ann = Path(f"{reference}.ann").exists()
    has_dict = Path(outdict).exists()
    if has_fai and has_ann and has_dict:
        return f"{prefix}echo 'all index exist'"
    else:
        return (f'{samtoolsidx} && '
                f'{bwaidx} && '
                f'{gatkidx}')

def fastp(sample:str, fastq1:Pathlike, fastq2:Union[Pathlike,None]=None,
          out:Pathlike='.',core=4, singularity: str = ""):
    """
    Generate fastp command for read trimming and QC.

    Outputs (for sample='S1', out='1.CleanFq'):
      - 1.CleanFq/S1.R1.clean.fastq.gz
      - 1.CleanFq/S1.R2.clean.fastq.gz  (if paired-end)
      - 1.CleanFq/S1.html
      - 1.CleanFq/S1.json

    Parameters
    ----------
    sample : str
        Sample name prefix for output files.
    fastq1 : Pathlike
        R1 input FASTQ(.gz) file.
    fastq2 : Pathlike or None
        R2 input FASTQ(.gz) file for paired-end. If None, single-end mode.
    out : Pathlike
        Output directory for clean FASTQ and reports.
    core : int
        Number of threads for fastp (`-w`).

    Returns
    -------
    str
        Shell command string for fastp.

    Notes
    -----
    Multi-threading:
      - `fastp` uses `-w {core}` and scales well with CPU cores.
    """
    out = Path(out)
    prefix = _singularity_prefix(singularity)
    out.mkdir(mode=0o755, exist_ok=True)
    outclean1 = out / f'{sample}.R1.clean.fastq.gz'
    outhtml = out / f'{sample}.html'
    outjson = out / f'{sample}.json'
    fastq1 = Path(fastq1)
    if fastq2 is None:
        if outhtml.exists() and outclean1.exists():
            return f"{prefix}echo '{outclean1} exists'"
        else:
            return (f'{prefix}fastp -i {fastq1}  '
                    f'-o {outclean1} '
                    f'--html {outhtml} --json {outjson} '
                    f'-w {core}')
    else:
        fastq2 = Path(fastq2)
        outclean2 = out / f'{sample}.R2.clean.fastq.gz'
        if outhtml.exists() and outclean1.exists() and outclean2.exists():
            return f"{prefix}echo '{outclean1} and {outclean2} exist'"
        else:
            return (f'{prefix}fastp -i {fastq1} -I {fastq2}  '
                    f'-o {outclean1} -O {outclean2} '
                    f'--html {outhtml} --json {outjson} '
                    f'-w {core}')
        

def bwamem(reference:Pathlike, sample:str, fastq1:Pathlike, fastq2:Union[Pathlike,None]=None,
            out:Pathlike='.',core:int=4, singularity: str = ""):
    """
    Run `bwa mem` followed by `samtools sort` to generate a sorted BAM.

    Output (for sample='S1', out='2.Mapping'):
      - 2.Mapping/S1.sorted.bam

    Parameters
    ----------
    reference : Pathlike
        Reference FASTA used by bwa, e.g. "genome.fa".
    sample : str
        Sample name, will be used in read group (RG) and output BAM name.
    fastq1 : Pathlike
        R1 clean FASTQ file.
    fastq2 : Pathlike or None
        R2 clean FASTQ file. If None, treat as single-end.
    out : Pathlike
        Output directory for sorted BAM.
    core : int
        Number of threads for `bwa mem` and `samtools sort`.

    Returns
    -------
    str
        Shell command string for mapping and sorting.

    Notes
    -----
    Multi-threading:
      - `bwa mem -t {core}` uses multiple threads for alignment.
      - `samtools sort -@ {core}` uses multiple threads for compression.
    """
    out = Path(out)
    prefix = _singularity_prefix(singularity)
    out.mkdir(mode=0o755, exist_ok=True)
    outbam = out / f'{sample}.sorted.bam'
    rg = rf"'@RG\tID:{sample}\tPL:illumina\tLB:{sample}\tSM:{sample}'"
    if outbam.exists() and (out / f'{sample}.sorted.bam.finished').exists():
        return f"{prefix}echo '{outbam} exists'"
    else:
        if fastq2 is None:
            bwa_cmd = f'{prefix}bwa mem -t {core} -R {rg} {reference} {fastq1}'
            sort_cmd = f'{prefix}samtools sort -@ {core} -o {outbam}'
            touch_cmd = f'{prefix}touch {outbam}.finished'
            return f'{bwa_cmd} | {sort_cmd} && {touch_cmd}'
        else:
            bwa_cmd = f'{prefix}bwa mem -t {core} -R {rg} {reference} {fastq1} {fastq2}'
            sort_cmd = f'{prefix}samtools sort -@ {core} -o {outbam}'
            touch_cmd = f'{prefix}touch {outbam}.finished'
            return f'{bwa_cmd} | {sort_cmd} && {touch_cmd}'

def markdup(sample:str, bam:Pathlike,
            out:Pathlike='.',core:int=4,mem:int=50, singularity: str = ""):
    """
    Mark / remove PCR duplicates and generate QC statistics.

    Input:
      - Sorted BAM, e.g. `2.Mapping/S1.sorted.bam`

    Outputs (for sample='S1', out='2.Mapping'):
      - 2.Mapping/S1.Markdup.bam
      - 2.Mapping/S1.Markdup.bam.bai  (index)
      - 2.Mapping/S1.Markdup.metrics.txt
      - 2.Mapping/S1.flagstat
      - 2.Mapping/S1.coverage

    Parameters
    ----------
    sample : str
        Sample name prefix.
    bam : Pathlike
        Input sorted BAM file (from `bwamem` step).
    out : Pathlike
        Output directory for MarkDuplicates results and stats.
    core : int
        Number of CPU cores to use in Java GC and samtools (`ParallelGCThreads`, `-@`).
    mem : int
        Java heap memory in GB (`-Xmx{mem}G`).

    Returns
    -------
    str
        Shell command string that runs MarkDuplicates, flagstat and coverage.

    Notes
    -----
    Multi-threading:
      - `gatk MarkDuplicates`: via `-XX:ParallelGCThreads={core}`.
      - `samtools flagstat -@ {core}`: multi-threaded stats.
      - `samtools coverage -@ {core}`: multi-threaded coverage computation.
    """
    out = Path(out)
    prefix = _singularity_prefix(singularity)
    out.mkdir(mode=0o755, exist_ok=True)
    outbam = out / f'{sample}.Markdup.bam'
    outflagstats = out / f'{sample}.flagstat'
    outcoverage = out / f'{sample}.coverage'
    outmetric = out / f'{sample}.Markdup.metrics.txt'
    if outmetric.exists() and outbam.exists():
        return f"{prefix}echo '{outbam} exists'"
    else:
        gatk_cmd = (
            f"{prefix}gatk --java-options '-Xmx{int(mem)}G -XX:ParallelGCThreads={core}' "
            f"MarkDuplicates -I {bam} -O {outbam} "
            "--CREATE_INDEX true --REMOVE_DUPLICATES false "
            f"--METRICS_FILE {outmetric}"
        )
        flagstat_cmd = f'{prefix}samtools flagstat {outbam} > {outflagstats}'
        coverage_cmd = f'{prefix}samtools coverage {outbam} > {outcoverage}'
        return f'{gatk_cmd} && {flagstat_cmd} && {coverage_cmd}'

def bam2gvcf(reference:Pathlike, sample:str, bam:Pathlike, chrom:Union[str,int],
            out:Pathlike='.',core:int=4, singularity: str = ""):
    """
    Call variants per-sample, per-chromosome (or interval) in GVCF mode.

    Input:
      - Deduplicated BAM, e.g. `2.Mapping/S1.Markdup.bam`

    Output:
      - GVCF file, e.g. `3.SNPcalling/S1.chr1.g.vcf.gz`

    Parameters
    ----------
    reference : Pathlike
        Reference FASTA.
    sample : str
        Sample name prefix.
    bam : Pathlike
        Deduplicated BAM file.
    chrom : str or int
        Chromosome name or interval passed to `-L`, e.g. "chr1" or "chr1:1-1000000".
    out : Pathlike
        Output directory for GVCFs.
    core : int
        Number of threads for HaplotypeCaller pair-HMM.

    Returns
    -------
    str
        Shell command string for GVCF calling.
    """
    out = Path(out)
    prefix = _singularity_prefix(singularity)
    out.mkdir(mode=0o755, exist_ok=True)
    outgvcf = out / f'{sample}.{chrom}.g.vcf.gz'
    if (out / f'{sample}.{chrom}.g.vcf.gz.tbi').exists() and outgvcf.exists():
        return f"{prefix}echo '{outgvcf} exists'"
    else:
        return (f'{prefix}gatk HaplotypeCaller -R {reference} --native-pair-hmm-threads {core} '
                f'-ERC GVCF -I {bam} -O {outgvcf} -L {chrom}')

def cgvcf(reference:Pathlike, chrom: str, gvcfs: List[Pathlike], out: Pathlike, singularity: str = ""):
    """
    Merge multiple per-sample GVCFs into a single GVCF for one chromosome.

    Inputs:
      - A list of GVCF paths, e.g.:
          ["3.SNPcalling/S1.chr1.g.vcf.gz",
           "3.SNPcalling/S2.chr1.g.vcf.gz",
           ...]

    Output:
      - Merged GVCF, e.g. `4.MergeVCF/Merge.chr1.g.vcf.gz`

    Parameters
    ----------
    reference : Pathlike
        Reference FASTA.
    chrom : str
        Chromosome name, e.g. "chr1".
    gvcfs : list of Pathlike
        List of per-sample GVCF files for this chromosome.
    out : Pathlike
        Output directory for merged GVCF.

    Returns
    -------
    str
        Shell command string for `gatk CombineGVCFs`.
    """
    reference = Path(reference)
    out = Path(out)
    prefix = _singularity_prefix(singularity)
    out.mkdir(mode=0o755, exist_ok=True)
    out_g = out / f"Merge.{chrom}.g.vcf.gz"
    variants_str = " ".join(f"--variant {p}" for p in map(str, gvcfs))
    if out_g.exists() and (out / f"Merge.{chrom}.g.vcf.gz.tbi").exists():
        return f"{prefix}echo '{out_g} exists'"
    else:
        return f"{prefix}gatk CombineGVCFs -R {reference} {variants_str} -O {out_g}"

def gvcf2vcf(reference: Pathlike, chrom: str, out: Pathlike,
             core: int = 4, mem: int = 50, singularity: str = "") -> str:
    """
    Genotype the merged GVCF to produce a multi-sample VCF.

    Input:
      - Merged GVCF, e.g. `4.MergeVCF/Merge.chr1.g.vcf.gz`

    Output:
      - Multi-sample VCF, e.g. `4.MergeVCF/Merge.chr1.vcf.gz`

    Parameters
    ----------
    reference : Pathlike
        Reference FASTA.
    chrom : str
        Chromosome name.
    out : Pathlike
        Output directory containing merged GVCF and the VCF to be generated.
    core : int
        Number of CPU cores to use in Java GC and samtools (`ParallelGCThreads`, `-@`).
    mem : int
        Java heap memory in GB (`-Xmx{mem}G`).


    Returns
    -------
    str
        Shell command string for `gatk GenotypeGVCFs`.
    """
    reference = Path(reference)
    out = Path(out)
    prefix = _singularity_prefix(singularity)
    in_g = out / f"Merge.{chrom}.g.vcf.gz"
    out_vcf = out / f"Merge.{chrom}.vcf.gz"
    if out_vcf.exists() and (out / f"Merge.{chrom}.vcf.gz.tbi").exists():
        return f"{prefix}echo '{out_vcf} exists'"
    else:
        return (
            f"{prefix}gatk --java-options '-Xmx{mem}G -XX:ParallelGCThreads={core}' "
            f"GenotypeGVCFs -R {reference} -V {in_g} -O {out_vcf}"
        )

def vcf2snpvcf(reference: Pathlike, chrom: str, out: Pathlike, singularity: str = "") -> str:
    """
    Extract bi-allelic SNPs from the multi-sample VCF.

    Input:
      - Multi-sample VCF, e.g. `4.MergeVCF/Merge.chr1.vcf.gz`

    Output:
      - Bi-allelic SNP VCF, e.g. `4.MergeVCF/Merge.chr1.SNP.vcf.gz`

    Parameters
    ----------
    reference : Pathlike
        Reference FASTA.
    chrom : str
        Chromosome name.
    out : Pathlike
        Output directory containing the multi-sample VCF and SNP VCF.

    Returns
    -------
    str
        Shell command string for `gatk SelectVariants`.
    """
    reference = Path(reference)
    out = Path(out)
    prefix = _singularity_prefix(singularity)
    in_vcf = out / f"Merge.{chrom}.vcf.gz"
    out_snp = out / f"Merge.{chrom}.SNP.vcf.gz"
    if out_snp.exists() and (out / f"Merge.{chrom}.SNP.vcf.gz.tbi").exists():
        return f"{prefix}echo '{out_snp} exists'"
    else:
        return (f"{prefix}gatk SelectVariants "
                f"-R {reference} "
                f"-V {in_vcf} "
                "--select-type SNP "
                "--restrict-alleles-to BIALLELIC "
                f"-O {out_snp}")

def filtersnp(reference: Pathlike, chrom: str, out: Pathlike, singularity: str = "") -> str:
    """
    Apply hard filters to SNP VCF.

    Input:
      - SNP VCF, e.g. `4.MergeVCF/Merge.chr1.SNP.vcf.gz`

    Output:
      - Filtered SNP VCF with FILTER tags, e.g. `4.MergeVCF/Merge.chr1.SNP.filter.vcf.gz`

    Parameters
    ----------
    reference : Pathlike
        Reference FASTA.
    chrom : str
        Chromosome name.
    out : Pathlike
        Output directory where filtered VCF will be written.

    Returns
    -------
    str
        Shell command string for `gatk VariantFiltration`.
    """
    reference = Path(reference)
    out = Path(out)
    prefix = _singularity_prefix(singularity)
    in_snp = out / f"Merge.{chrom}.SNP.vcf.gz"
    out_filtered = out / f"Merge.{chrom}.SNP.filter.vcf.gz"
    if out_filtered.exists() and (out / f"Merge.{chrom}.SNP.filter.vcf.gz.tbi").exists():
        return f"{prefix}echo '{out_filtered} exists'"
    else:
        return (f"{prefix}gatk VariantFiltration "
                f"-R {reference} "
                f"-V {in_snp} "
                f"-O {out_filtered} "
                "--filter-name 'QUAL30' --filter-expression 'QUAL < 30.0' "
                "--filter-name 'FS60' --filter-expression 'FS > 60.0' "
                "--filter-name 'QD2' --filter-expression 'QD < 2.0' "
                "--filter-name 'SOR3' --filter-expression 'SOR > 3.0' "
                "--filter-name 'MQ40' --filter-expression 'MQ < 40.0' "
                "--filter-name 'ReadPosRankSum-8' --filter-expression 'ReadPosRankSum < -8.0' "
                "--filter-name 'MQRankSum-12.5' --filter-expression 'MQRankSum < -12.5'")
        
def selectfiltersnp(reference: Pathlike, chrom: str, out: Pathlike, singularity: str = "") -> str:
    """
    Select only PASS SNPs from the filtered SNP VCF.

    Input:
      - Filtered SNP VCF, e.g. `4.MergeVCF/Merge.chr1.SNP.filter.vcf.gz`

    Output:
      - Final high-quality SNP VCF, e.g. `4.MergeVCF/Merge.chr1.SNP.filtered.vcf.gz`

    Parameters
    ----------
    reference : Pathlike
        Reference FASTA.
    chrom : str
        Chromosome name.
    out : Pathlike
        Output directory for PASS-only SNP VCF.

    Returns
    -------
    str
        Shell command string for SelectVariants to drop filtered variants.
    """
    reference = Path(reference)
    out = Path(out)
    prefix = _singularity_prefix(singularity)
    in_filtered = out / f"Merge.{chrom}.SNP.filter.vcf.gz"
    out_pass = out / f"Merge.{chrom}.SNP.filtered.vcf.gz"
    if out_pass.exists() and (out / f"Merge.{chrom}.SNP.filtered.vcf.gz.tbi").exists():
        return f"{prefix}echo '{out_pass} exists'"
    else:
        return (f"{prefix}gatk SelectVariants "
                f"-R {reference} -V {in_filtered} "
                "--exclude-filtered "
                f"-O {out_pass}")

def vcf2table(reference: Pathlike, chrom: str, out: Pathlike, singularity: str = "") -> str:
    """
    Convert final filtered SNP VCF into a tabular format.

    Input:
      - Final SNP VCF, e.g. `4.MergeVCF/Merge.chr1.SNP.filtered.vcf.gz`

    Output:
      - SNP table, e.g. `4.MergeVCF/Merge.chr1.SNP.tsv`, with columns:
        CHROM, POS, REF, ALT, DP, AD, GQ (per genotype)

    Parameters
    ----------
    reference : Pathlike
        Reference FASTA.
    chrom : str
        Chromosome name.
    out : Pathlike
        Output directory for the TSV file.

    Returns
    -------
    str
        Shell command string for `gatk VariantsToTable`.
    """
    reference = Path(reference)
    out = Path(out)
    prefix = _singularity_prefix(singularity)
    in_pass = out / f"Merge.{chrom}.SNP.filtered.vcf.gz"
    out_tsv = out / f"Merge.{chrom}.SNP.tsv"
    return (f"{prefix}gatk VariantsToTable "
            f"-R {reference} "
            f"-V {in_pass} "
            "-F CHROM -F POS -F REF -F ALT "
            "-GF DP -GF AD -GF GQ "
            f"-O {out_tsv}")


def snpvcf_to_gt_and_missing(
    chrom: str,
    merge_dir: Pathlike,
    impute_dir: Pathlike,
    *,
    min_dp: int = 5,
    min_gq: int = 20,
    min_ad: int = 2,
    core: int = 2,
    singularity: str = "",
) -> str:
    """
    Prepare per-chromosome GT-only VCF for imputation and compute missingness stats.

    Input:
      - 4.merge/Merge.{chrom}.SNP.filtered.vcf.gz

    Outputs:
      - 5.impute/Merge.{chrom}.SNP.GT.vcf.gz (+ .tbi)
      - 5.impute/Merge.{chrom}.SNP.GT.lmiss
      - 5.impute/Merge.{chrom}.SNP.GT.imiss
    """
    merge_dir = Path(merge_dir)
    impute_dir = Path(impute_dir)
    prefix = _singularity_prefix(singularity)
    impute_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    in_vcf = merge_dir / f"Merge.{chrom}.SNP.filtered.vcf.gz"
    out_gt = impute_dir / f"Merge.{chrom}.SNP.GT.vcf.gz"
    out_tbi = impute_dir / f"Merge.{chrom}.SNP.GT.vcf.gz.tbi"
    miss_prefix = impute_dir / f"Merge.{chrom}.SNP.GT"
    out_lmiss = Path(f"{miss_prefix}.lmiss")
    out_imiss = Path(f"{miss_prefix}.imiss")

    if out_gt.exists() and out_tbi.exists() and out_lmiss.exists() and out_imiss.exists():
        return f"{prefix}echo '{out_gt} exists'"

    # Use native bcftools filtering to set low-quality genotypes to missing.
    # AD index syntax varies across bcftools versions, so we try multiple
    # expressions and finally fall back to DP/GQ only.
    expr_with_ad_new = (
        f"FMT/DP<{int(min_dp)} || FMT/GQ<{int(min_gq)} || FMT/AD[:1]<{int(min_ad)}"
    )
    expr_with_ad_old = (
        f"FMT/DP<{int(min_dp)} || FMT/GQ<{int(min_gq)} || FMT/AD[1]<{int(min_ad)}"
    )
    expr_dp_gq_only = f"FMT/DP<{int(min_dp)} || FMT/GQ<{int(min_gq)}"
    setgt_cmd = (
        f"( {prefix}bcftools filter -Ou -S . -e '{expr_with_ad_new}' {_q(in_vcf)} "
        f"|| {prefix}bcftools filter -Ou -S . -e '{expr_with_ad_old}' {_q(in_vcf)} "
        f"|| {prefix}bcftools filter -Ou -S . -e '{expr_dp_gq_only}' {_q(in_vcf)} )"
    )
    keep_gt_cmd = (
        f"{prefix}bcftools annotate --threads {int(core)} "
        f"-x FORMAT,^FORMAT/GT -Oz -o {_q(out_gt)}"
    )
    index_cmd = f"{prefix}tabix -f -p vcf {_q(out_gt)}"
    plink_missing_cmd = (
        f"{prefix}plink --vcf {_q(out_gt)} --allow-extra-chr --double-id "
        f"--set-missing-var-ids @:# --threads {int(core)} --missing "
        f"--out {_q(miss_prefix)}"
    )
    write_empty_miss_cmd = (
        f"printf 'CHR\\tSNP\\tN_MISS\\tN_GENO\\tF_MISS\\n' > {_q(out_lmiss)} && "
        f"printf 'FID\\tIID\\tMISS_PHENO\\tN_MISS\\tN_GENO\\tF_MISS\\n' > {_q(out_imiss)}"
    )
    # Some chromosomes/contigs have zero SNPs after filtering; PLINK exits
    # non-zero on empty VCF. Detect empties and emit header-only .lmiss/.imiss.
    miss_cmd = (
        f"NVAR=$({prefix}bcftools index -n {_q(out_gt)}) && "
        f"if [ \"${{NVAR:-0}}\" -gt 0 ]; then "
        f"{plink_missing_cmd}; "
        f"else "
        f"{write_empty_miss_cmd}; "
        f"fi"
    )
    return f"{setgt_cmd} | {keep_gt_cmd} && {index_cmd} && {miss_cmd}"


def beagle_impute(
    chrom: str,
    impute_dir: Pathlike,
    *,
    core: int = 2,
    singularity: str = "",
) -> str:
    """
    Run BEAGLE imputation on per-chromosome GT VCF.

    Input:
      - 5.impute/Merge.{chrom}.SNP.GT.vcf.gz

    Output:
      - 5.impute/Merge.{chrom}.SNP.GT.imp.vcf.gz (+ .tbi)
    """
    impute_dir = Path(impute_dir)
    prefix = _singularity_prefix(singularity)
    impute_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    in_gt = impute_dir / f"Merge.{chrom}.SNP.GT.vcf.gz"
    out_prefix = impute_dir / f"Merge.{chrom}.SNP.GT.imp"
    out_imp = impute_dir / f"Merge.{chrom}.SNP.GT.imp.vcf.gz"
    out_tbi = impute_dir / f"Merge.{chrom}.SNP.GT.imp.vcf.gz.tbi"

    if out_imp.exists() and out_tbi.exists():
        return f"{prefix}echo '{out_imp} exists'"

    beagle_cmd = (
        f"{prefix}beagle gt={_q(in_gt)} out={_q(out_prefix)} nthreads={int(core)}"
    )
    index_cmd = f"{prefix}tabix -f -p vcf {_q(out_imp)}"
    copy_empty_cmd = (
        f"{prefix}bcftools view -Oz -o {_q(out_imp)} {_q(in_gt)} && {index_cmd}"
    )
    return (
        f"NVAR=$({prefix}bcftools index -n {_q(in_gt)}) && "
        f"if [ \"${{NVAR:-0}}\" -gt 0 ]; then "
        f"{beagle_cmd} && {index_cmd}; "
        f"else "
        f"{copy_empty_cmd}; "
        f"fi"
    )


def filter_imputed_by_maf_and_missing(
    chrom: str,
    chroms: Sequence[str],
    impute_dir: Pathlike,
    *,
    maf: float = 0.02,
    max_missing: float = 0.2,
    core: int = 2,
    singularity: str = "",
) -> str:
    """
    Filter imputed VCF by MAF and pre-imputation missingness constraints.

    Rules:
      1) Keep SNPs with pre-imputation site missingness <= max_missing (per chrom).
      2) Keep samples with pre-imputation global missingness <= max_missing
         (aggregated across all chromosomes).
      3) Keep variants with MAF >= maf.

    Input:
      - 5.impute/Merge.{chrom}.SNP.GT.imp.vcf.gz
      - 5.impute/Merge.{chrom}.SNP.GT.lmiss
      - 5.impute/Merge.{chrom}.SNP.GT.imiss (all chroms for global sample filtering)

    Output:
      - 5.impute/Merge.{chrom}.SNP.GT.imp.maf0.02.miss0.2.vcf.gz (+ .tbi)
    """
    impute_dir = Path(impute_dir)
    prefix = _singularity_prefix(singularity)
    impute_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    in_imp = impute_dir / f"Merge.{chrom}.SNP.GT.imp.vcf.gz"
    out_vcf = impute_dir / f"Merge.{chrom}.SNP.GT.imp.maf0.02.miss0.2.vcf.gz"
    out_tbi = impute_dir / f"Merge.{chrom}.SNP.GT.imp.maf0.02.miss0.2.vcf.gz.tbi"
    site_lmiss = impute_dir / f"Merge.{chrom}.SNP.GT.lmiss"
    site_keep = impute_dir / f"Merge.{chrom}.preimpute.site.keep.txt"
    sample_keep = impute_dir / "Merge.preimpute.sample.keep.txt"
    sample_tmp = Path(f"{sample_keep}.tmp")
    sample_lock = Path(f"{sample_keep}.lock")

    if out_vcf.exists() and out_tbi.exists():
        return f"{prefix}echo '{out_vcf} exists'"

    all_imiss = [impute_dir / f"Merge.{c}.SNP.GT.imiss" for c in chroms]
    all_imiss_txt = " ".join(_q(p) for p in all_imiss)
    build_sample_keep = (
        f"if [ ! -s {_q(sample_keep)} ]; then "
        f"if mkdir {_q(sample_lock)} 2>/dev/null; then "
        f"awk 'NR==1{{next}} {{id=$2; miss[id]+=$4; geno[id]+=$5}} "
        f"END{{for (id in miss) if (geno[id]>0 && miss[id]/geno[id]<={float(max_missing)}) print id}}' "
        f"{all_imiss_txt} | sort > {_q(sample_tmp)} && mv {_q(sample_tmp)} {_q(sample_keep)}; "
        f"rmdir {_q(sample_lock)}; "
        f"else "
        f"while [ -d {_q(sample_lock)} ]; do sleep 1; done; "
        f"fi; "
        f"fi"
    )
    build_site_keep = (
        f"awk 'NR>1 && $5<={float(max_missing)} {{n=split($2,a,\":\"); "
        f"if (n>=2) print a[1]\"\\t\"a[2]}}' "
        f"{_q(site_lmiss)} > {_q(site_keep)}"
    )
    filter_cmd = (
        f"{prefix}bcftools view --threads {int(core)} "
        f"-S {_q(sample_keep)} -T {_q(site_keep)} -i 'MAF>={float(maf)}' "
        f"-Oz -o {_q(out_vcf)} {_q(in_imp)}"
    )
    index_cmd = f"{prefix}tabix -f -p vcf {_q(out_vcf)}"
    copy_empty_cmd = (
        f"{prefix}bcftools view -Oz -o {_q(out_vcf)} {_q(in_imp)} && {index_cmd}"
    )
    return (
        f"NVAR=$({prefix}bcftools index -n {_q(in_imp)}) && "
        f"if [ \"${{NVAR:-0}}\" -gt 0 ]; then "
        f"{build_sample_keep} && {build_site_keep} && {filter_cmd} && {index_cmd}; "
        f"else "
        f"{copy_empty_cmd}; "
        f"fi"
    )


def concat_imputed_vcfs(
    chroms: Sequence[str],
    impute_dir: Pathlike,
    genotype_dir: Pathlike,
    *,
    core: int = 2,
    singularity: str = "",
) -> str:
    """
    Concatenate all per-chromosome filtered imputed VCFs into one merged VCF.

    Input:
      - 5.impute/Merge.{chrom}.SNP.GT.imp.maf0.02.miss0.2.vcf.gz

    Output:
      - 6.genotype/Merge.SNP.GT.imp.maf0.02.miss0.2.vcf.gz (+ .tbi)
    """
    impute_dir = Path(impute_dir)
    genotype_dir = Path(genotype_dir)
    prefix = _singularity_prefix(singularity)
    genotype_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    out_vcf = genotype_dir / "Merge.SNP.GT.imp.maf0.02.miss0.2.vcf.gz"
    out_tbi = genotype_dir / "Merge.SNP.GT.imp.maf0.02.miss0.2.vcf.gz.tbi"
    if out_vcf.exists() and out_tbi.exists():
        return f"{prefix}echo '{out_vcf} exists'"

    inputs = [
        impute_dir / f"Merge.{chrom}.SNP.GT.imp.maf0.02.miss0.2.vcf.gz"
        for chrom in chroms
    ]
    first_input = inputs[0]
    inputs_txt = " ".join(_q(p) for p in inputs)
    nonempty_list = genotype_dir / "Merge.concat.nonempty.list.txt"
    build_nonempty_cmd = (
        f": > {_q(nonempty_list)}; "
        f"for f in {inputs_txt}; do "
        f"n=$({prefix}bcftools index -n \"$f\" 2>/dev/null || echo 0); "
        f"if [ \"${{n:-0}}\" -gt 0 ]; then printf '%s\\n' \"$f\" >> {_q(nonempty_list)}; fi; "
        f"done"
    )
    copy_empty_cmd = f"{prefix}bcftools view -Oz -o {_q(out_vcf)} {_q(first_input)}"
    copy_single_cmd = (
        f"f=$(head -n 1 {_q(nonempty_list)}); "
        f"{prefix}bcftools view -Oz -o {_q(out_vcf)} \"$f\""
    )
    concat_cmd = (
        f"{prefix}bcftools concat --threads {int(core)} -a "
        f"-f {_q(nonempty_list)} -Oz -o {_q(out_vcf)}"
    )
    index_cmd = f"{prefix}tabix -f -p vcf {_q(out_vcf)}"
    return (
        f"{build_nonempty_cmd} && "
        f"nfile=$(wc -l < {_q(nonempty_list)}); "
        f"if [ \"${{nfile:-0}}\" -eq 0 ]; then "
        f"{copy_empty_cmd}; "
        f"elif [ \"${{nfile:-0}}\" -eq 1 ]; then "
        f"{copy_single_cmd}; "
        f"else "
        f"{concat_cmd}; "
        f"fi && "
        f"{index_cmd}"
    )
