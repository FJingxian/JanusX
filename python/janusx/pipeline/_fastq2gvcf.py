from pathlib import Path
from typing import List,Union,Literal

Pathlike = Union[Path,str]

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
    bwaidx = f'{prefix}samtools faidx {reference}'
    outdict = f"{str(reference).replace('.fasta.gz','').replace('.fa.gz','').replace('.fasta','').replace('.fasta','')}.dict"
    gatkidx = f"{prefix}gatk CreateSequenceDictionary -R {reference} -O {outdict}"
    if Path(gatkidx).exists():
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
