import time
import yaml
import subprocess
from pathlib import Path
from typing import Any, List, Union, Literal, Dict
import itertools

from _fastq2gvcf import (
    filtersnp, gvcf2vcf, fastp, bwamem, markdup, bam2gvcf, cgvcf,
    selectfiltersnp, vcf2snpvcf, vcf2table
)

PathLike = Union[Path, str]

def _wrap_cmd(cmd: str, job: str, threads: int, scheduler: str, queue: str = "c01",singularity:str="") -> str:
    
    if scheduler == "nohup":
        safe_cmd = cmd.replace('"', '\\"')
        return (
            f'nohup bash -c "{singularity} {safe_cmd}" '
            f'> ./log/{job}.o 2> ./log/{job}.e'
        )
    return f'csub -J {job} -o ./log/%J.o -e ./log/%J.e -q {queue} -n {threads} "{singularity} {cmd}"'


def node(ifiles: List[PathLike], ofiles: List[PathLike]) -> Literal[-1, 0, 1]:
    outexists = all(Path(p).exists() for p in ofiles)
    if outexists:
        return 1  # pass

    inexists = all(Path(p).exists() for p in ifiles)
    if inexists:
        return 0  # run
    return -1     # back


def pipeline(AllCMD: List[str],
             Alli: List[List[PathLike]],
             Allo: List[List[PathLike]],
             scheduler: str = "csub",
             nohup_max_jobs: int = 0) -> None:
    Path("tmp").mkdir(mode=0o755, parents=True, exist_ok=True)

    for num, cmd in enumerate(AllCMD):
        ifiles = Alli[num]
        ofiles = Allo[num]

        status = node(ifiles, ofiles)

        sh = Path(f"./tmp/{num}.sh")
        if scheduler == "nohup":
            lines = [line.strip() for line in cmd.splitlines() if line.strip()]
            if nohup_max_jobs > 0:
                parts = [f"MAX_JOBS={nohup_max_jobs}"]
                for line in lines:
                    parts.append('while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do sleep 2; done')
                    parts.append(f"{line} &")
                parts.append("wait")
                text = "\n".join(parts) + "\n"
            else:
                text = "\n".join([f"{line} &" for line in lines] + ["wait"]) + "\n"
        else:
            text = cmd if cmd.endswith("\n") else (cmd + "\n")
        sh.write_text(text, encoding="utf-8")
        sh.chmod(0o755)

        tstart = time.time()

        if status == 1:
            print(f"Step-{num} completed! (skip)")
            continue

        if status == -1:
            raise ValueError(f"Step-{num} inputs not exists: {ifiles}")

        # status == 0
        print(f"Run Step-{num}: {sh}")
        # 真实执行 + 检查退出码
        subprocess.run(["bash", str(sh)], check=True)

        # 等待输出出现（如果命令同步产生输出，这里一般很快结束）
        while True:
            if node(ifiles, ofiles) == 1:
                print(f"\nStep-{num} completed!")
                break
            print(f"\rWaiting outputs... time={time.time() - tstart:.1f}s", end="")
            time.sleep(60)


def main(yamlpath:PathLike):
    with open(yamlpath, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    Path("log").mkdir(exist_ok=True)
    scheduler = str(data.get("scheduler", "csub")).lower()
    nohup_max_jobs = int(data.get("nohup_max_jobs", 0))

    cleanfolder = Path(data["outdirs"]["clean"])
    mappingfolder = Path(data["outdirs"]["mapping"])
    gvcffolder = Path(data["outdirs"]["gvcf"])
    mergefolder = Path(data["outdirs"]["merge"])

    cleanfolder.mkdir(0o755, exist_ok=True)
    mappingfolder.mkdir(0o755, exist_ok=True)
    gvcffolder.mkdir(0o755, exist_ok=True)
    mergefolder.mkdir(0o755, exist_ok=True)

    reference = Path(data["reference"])
    samples_fq: Dict[str, Dict[str, Any]] = data["samples"]
    samples = list(samples_fq.keys())
    CHROM = list(data["chrom"])

    # -------- step1: fastp -------- 15 mins
    step1_lines = []
    core = 16
    for sample in samples:
        fq1 = samples_fq[sample]["fq1"]
        fq2 = samples_fq[sample]["fq2"]
        cmd_fastp = fastp(sample, fq1, fq2, cleanfolder, core)
        step1_lines.append(
            _wrap_cmd(cmd_fastp, f"fastp.{sample}", core, scheduler)
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
            _wrap_cmd(cmd_bwa, f"bwamem.{sample}", 64, scheduler)
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
            _wrap_cmd(cmd_md, f"markdup.{sample}", 16, scheduler)
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
                _wrap_cmd(cmd_g, f"bam2gvcf.{sample}.{chrom}", 2, scheduler)
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
            _wrap_cmd(cmd_c, f"cgvcf.{chrom}", 1, scheduler)
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
            _wrap_cmd(cmd6, f"gvcf2vcf.{chrom}", 1, scheduler)
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
