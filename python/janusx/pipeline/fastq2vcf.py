from pathlib import Path
from readline import backend
from typing import Any, Literal, Union, Dict
import itertools
from janusx.pipeline.pipeline import pipeline,wrap_cmd
from janusx.pipeline._fastq2gvcf import (
    filtersnp, gvcf2vcf, fastp, bwamem, markdup, bam2gvcf, cgvcf,
    selectfiltersnp, vcf2snpvcf, vcf2table
)

PathLike = Union[Path, str]


def fastq2vcf(metadata:dict=None,workdir:PathLike=".",backbend:Literal["nohup","csub"]="csub",nohup_max_jobs:int=2,singularity:str=''):
    Path("log").mkdir(exist_ok=True)
    scheduler = str(backbend).lower()
    nohup_max_jobs = int(nohup_max_jobs)

    cleanfolder = Path(workdir / '1.clean')
    mappingfolder = Path(workdir / '2.mapping')
    gvcffolder = Path(workdir / '3.gvcf')
    mergefolder = Path(workdir / '4.merge')

    cleanfolder.mkdir(0o755, exist_ok=True)
    mappingfolder.mkdir(0o755, exist_ok=True)
    gvcffolder.mkdir(0o755, exist_ok=True)
    mergefolder.mkdir(0o755, exist_ok=True)

    reference = Path(metadata["reference"])
    samples_fq: Dict[str, Dict[str, Any]] = metadata["samples"]
    samples = list(samples_fq.keys())
    CHROM = list(metadata["chrom"])

    # -------- step1: fastp -------- 15 mins
    step1_lines = []
    core = 16
    for sample in samples:
        fq1 = samples_fq[sample]["fq1"]
        fq2 = samples_fq[sample]["fq2"]
        cmd_fastp = fastp(sample, fq1, fq2, cleanfolder, core)
        step1_lines.append(
            wrap_cmd(cmd_fastp, f"fastp.{sample}", core, scheduler,singularity)
        )
    step1 = "\n".join(step1_lines)

    step1in = list(itertools.chain.from_iterable(
        [[samples_fq[s]["fq1"], samples_fq[s]["fq2"]] for s in samples]
    ))
    step1out = list(itertools.chain.from_iterable(
        [[cleanfolder / f"{s}.R1.clean.fastq.gz", cleanfolder / f"{s}.R2.clean.fastq.gz", cleanfolder / f"{s}.html", cleanfolder / f"{s}.json"] for s in samples]
    ))

    # -------- step2: bwamem -------- 255 mins
    step2_lines = []
    for sample in samples:
        r1 = cleanfolder / f"{sample}.R1.clean.fastq.gz"
        r2 = cleanfolder / f"{sample}.R2.clean.fastq.gz"
        cmd_bwa = bwamem(reference, sample, r1, r2, mappingfolder, 64)
        step2_lines.append(
            wrap_cmd(cmd_bwa, f"bwamem.{sample}", 64, scheduler,singularity)
        )
    step2 = "\n".join(step2_lines)

    step2out = list(itertools.chain.from_iterable(
        [[mappingfolder / f"{s}.sorted.bam", mappingfolder / f"{s}.sorted.bam.finished"] for s in samples]
    ))

    # -------- step3: markdup -------- 73 mins
    step3_lines = []
    for sample in samples:
        bam = mappingfolder / f"{sample}.sorted.bam"
        cmd_md = markdup(sample, bam, mappingfolder, 16, 200)
        step3_lines.append(
            wrap_cmd(cmd_md, f"markdup.{sample}", 16, scheduler,singularity)
        )
    step3 = "\n".join(step3_lines)

    step3out = list(itertools.chain.from_iterable(
        [[mappingfolder / f"{s}.Markdup.bam", mappingfolder / f"{s}.Markdup.metrics.txt"] for s in samples]
    ))

    # -------- step4: bam2gvcf per sample per chrom -------- 451 mins
    step4_lines = []
    for sample in samples:
        md_bam = mappingfolder / f"{sample}.Markdup.bam"
        for chrom in CHROM:
            cmd_g = bam2gvcf(reference, sample, md_bam, chrom, gvcffolder, 2)
            step4_lines.append(
                wrap_cmd(cmd_g, f"bam2gvcf.{sample}.{chrom}", 2, scheduler,singularity)
            )
    step4 = "\n".join(step4_lines)

    step4out = list(itertools.chain.from_iterable(
        [[gvcffolder / f"{s}.{chrom}.g.vcf.gz", gvcffolder / f"{s}.{chrom}.g.vcf.gz.tbi"]
         for s in samples for chrom in CHROM]
    ))

    # -------- step5: cgvcf per chrom -------- 36 mins
    step5_lines = []
    for chrom in CHROM:
        gvcfs = [gvcffolder / f"{s}.{chrom}.g.vcf.gz" for s in samples]
        cmd_c = cgvcf(reference, chrom, gvcfs, mergefolder)
        step5_lines.append(
            wrap_cmd(cmd_c, f"cgvcf.{chrom}", 1, scheduler,singularity)
        )
    step5 = "\n".join(step5_lines)

    step5out = list(itertools.chain.from_iterable(
        [[mergefolder / f"Merge.{chrom}.g.vcf.gz", mergefolder / f"Merge.{chrom}.g.vcf.gz.tbi"] for chrom in CHROM]
    ))

    # -------- step6: gvcf2vcf + snp + filter + table per chrom -------- 32 mins
    step6_lines = []
    for chrom in CHROM:
        cmd6 = (
            f'{gvcf2vcf(reference, chrom, mergefolder, 1, 50)} && '
            f'{vcf2snpvcf(reference, chrom, mergefolder)} && '
            f'{filtersnp(reference, chrom, mergefolder)} && '
            f'{selectfiltersnp(reference, chrom, mergefolder)} && '
            f'{vcf2table(reference, chrom, mergefolder)}'
        )
        step6_lines.append(
            wrap_cmd(cmd6, f"gvcf2vcf.{chrom}", 1, scheduler,singularity)
        )
    step6 = "\n".join(step6_lines)

    step6out = [mergefolder / f"Merge.{chrom}.SNP.tsv" for chrom in CHROM]
    
    pipeline(
        [step1, step2, step3, step4, step5, step6],
        [step1in, step1out, step2out, step3out, step4out, step5out],
        [step1out, step2out, step3out, step4out, step5out, step6out],
        scheduler=scheduler,
        nohup_max_jobs=nohup_max_jobs,
    )
