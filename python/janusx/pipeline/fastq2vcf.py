from pathlib import Path
from typing import Any, Literal, Union, Dict
import itertools
import hashlib
import json
import shlex
import sys
import time
from datetime import datetime, timezone
from janusx.pipeline.pipeline import pipeline,wrap_cmd
from janusx.pipeline._fastq2gvcf import (
    filtersnp, gvcf2vcf, fastp, bwamem, markdup, bam2gvcf, cgvcf,
    selectfiltersnp, vcf2snpvcf, vcf2table,indexREF
)

PathLike = Union[Path, str]

def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


def _params_signature(params: dict) -> str:
    body = json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _parse_utc_iso(ts: str) -> float:
    s = str(ts or "").strip()
    if len(s) == 0:
        raise ValueError("empty timestamp")
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    return dt.replace(tzinfo=timezone.utc).timestamp()


def _input_files_from_params(params: dict) -> list[Path]:
    files: list[Path] = []
    ref = params.get("reference")
    if ref:
        files.append(Path(str(ref)).expanduser().resolve())
    samples = params.get("samples", {})
    if isinstance(samples, dict):
        for v in samples.values():
            if isinstance(v, (list, tuple)):
                for p in v:
                    files.append(Path(str(p)).expanduser().resolve())
    out: list[Path] = []
    seen: set[str] = set()
    for p in files:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _validate_inputs_older_than_state(
    state_path: Path,
    state: dict,
    params: dict,
) -> tuple[bool, list[str], float]:
    try:
        ref_ts = _parse_utc_iso(str(state.get("created_at", "")))
    except Exception:
        ref_ts = float(state_path.stat().st_mtime)
    newer: list[str] = []
    for p in _input_files_from_params(params):
        if (not p.exists()) or (not p.is_file()):
            continue
        if float(p.stat().st_mtime) > ref_ts:
            newer.append(str(p))
    return (len(newer) == 0), newer, ref_ts


def _item_completed(outputs: list[Path]) -> bool:
    return all(Path(x).exists() for x in outputs)


def _load_json_or_empty(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _append_state_update_hook(
    cmd: str,
    state_path: Path,
    step_id: str,
    item_id: str,
    outputs: list[PathLike],
) -> str:
    pyexe = shlex.quote(sys.executable or "python")
    hook_script = shlex.quote(str((Path(__file__).resolve().parent / "workstate.py")))
    parts = [
        pyexe,
        hook_script,
        "--state",
        shlex.quote(str(state_path)),
        "--step",
        shlex.quote(str(step_id)),
        "--item",
        shlex.quote(str(item_id)),
    ]
    out_paths = [shlex.quote(str(Path(p))) for p in outputs]
    if len(out_paths) > 0:
        parts.append("--outputs")
        parts.extend(out_paths)
    check = " ".join(parts + ["--check"])
    hook = " ".join(parts)
    # Never let work-state hook failure fail the real compute task.
    return f"( {check} ) || ( ( {cmd} ) && ( {hook} || true ) )"


def _sync_state_from_fs(state: dict, steps_meta: list[dict]) -> None:
    steps_state = state.setdefault("steps", {})
    total_items = 0
    done_items = 0
    done_steps = 0

    for step in steps_meta:
        sid = str(step["id"])
        items = list(step["items"])
        entry = steps_state.get(sid, {})
        item_state = dict(entry.get("items", {}))
        item_done_at = dict(entry.get("items_completed_at", {}))
        done = 0
        for item in items:
            iid = str(item["id"])
            outputs = [Path(p) for p in item.get("outputs", [])]
            completed = _item_completed(outputs)
            prev = bool(item_state.get(iid, False))
            item_state[iid] = bool(completed)
            if completed and (not prev):
                item_done_at[iid] = _utc_now()
            elif (not completed) and prev:
                item_done_at.pop(iid, None)
            if completed:
                done += 1
        total = len(items)
        steps_state[sid] = {
            "name": str(step["name"]),
            "done": int(done),
            "total": int(total),
            "items": item_state,
            "items_completed_at": item_done_at,
        }
        total_items += total
        done_items += done
        if total > 0 and done == total:
            done_steps += 1

    state["summary"] = {
        "done_items": int(done_items),
        "total_items": int(total_items),
        "done_steps": int(done_steps),
        "total_steps": int(len(steps_meta)),
    }
    state["updated_at"] = _utc_now()


def _init_or_resume_work_state(
    state_path: Path,
    params: dict,
    steps_meta: list[dict],
) -> tuple[dict, bool]:
    signature = _params_signature(params)
    resumed = False
    state: dict

    if state_path.exists():
        old: dict = {}
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            old = {}
        try:
            ok_inputs, newer_inputs, ref_ts = _validate_inputs_older_than_state(
                state_path, old, params
            )
            if not ok_inputs:
                ts_txt = datetime.fromtimestamp(ref_ts, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                raise RuntimeError(
                    "Detected input files newer than existing .work.json "
                    f"(created_at={ts_txt}). Please remove {state_path} to restart.\n"
                    "Newer inputs:\n- " + "\n- ".join(newer_inputs)
                )
            if str(old.get("signature", "")) == signature:
                state = old
                resumed = True
            else:
                state = {}
        except RuntimeError:
            raise
        except Exception:
            state = {}
    else:
        state = {}

    if len(state) == 0:
        state = {
            "version": 1,
            "signature": signature,
            "params": params,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "status": "initialized",
            "steps": {},
            "summary": {},
        }

    _sync_state_from_fs(state, steps_meta)
    summary = state.get("summary", {})
    done_items = int(summary.get("done_items", 0))
    total_items = int(summary.get("total_items", 0))
    if total_items > 0 and done_items >= total_items:
        state["status"] = "completed"
    elif str(state.get("status", "")) != "failed":
        state["status"] = "running"
    _safe_write_json(state_path, state)
    return state, resumed


def fastq2vcf(metadata:dict=None,workdir:PathLike=".",backbend:Literal["nohup","csub"]="csub",nohup_max_jobs:int=2,singularity:str=''):
    """
    Run an end-to-end FASTQ→VCF pipeline (fastp → bwa-mem → markdup → gVCF → joint-calling → SNP filtering → TSV).

    This function generates per-sample QC/clean FASTQ files, maps reads to a reference genome,
    marks duplicates, calls gVCF per sample per chromosome, performs per-chromosome joint genotyping,
    extracts SNPs, filters them, and finally exports SNP tables (TSV) per chromosome.

    The pipeline is executed through `janusx.pipeline.pipeline.pipeline()` and each step command is
    wrapped by `wrap_cmd()` to support different execution backends (e.g. cluster submission or nohup).

    Parameters
    ----------
    metadata : dict, optional
        A configuration dictionary describing reference, chromosomes and FASTQ inputs.
        Required keys:
          - "reference": str | Path
              Path to the reference genome FASTA.
          - "chrom": list[str]
              Chromosome/contig names to process (e.g. ["Chr01", "Chr02", ...]).
          - "samples": dict[str, dict[str, Any]]
              Sample FASTQ mapping. Each sample must include:
                - "fq1": str | Path  (R1 fastq.gz)
                - "fq2": str | Path  (R2 fastq.gz)

        Example:
            metadata = {
              "reference": "/path/ref.fa",
              "chrom": ["Chr01", "Chr02"],
              "samples": {
                "S1": {"fq1": "S1_R1.fq.gz", "fq2": "S1_R2.fq.gz"},
                "S2": {"fq1": "S2_R1.fq.gz", "fq2": "S2_R2.fq.gz"},
              }
            }

    workdir : PathLike, default="."
        Working directory for all outputs. The function creates the following subdirectories
        under `workdir`:
          - 1.clean/   : cleaned FASTQs + fastp reports
          - 2.mapping/ : sorted BAMs, markdup BAMs, metrics
          - 3.gvcf/    : per-sample per-chrom gVCFs and indexes
          - 4.merge/   : joint-called per-chrom merged gVCFs/VCFs and SNP tables

    backbend : {"nohup", "csub"}, default="csub"
        Execution backend passed to `wrap_cmd()` and `pipeline()`.
        - "csub": submit jobs to a cluster scheduler (implementation depends on your wrapper).
        - "nohup": run commands locally with nohup, using `nohup_max_jobs` to limit concurrency.

    nohup_max_jobs : int, default=2
        Max number of concurrent jobs when `backbend="nohup"`.

    singularity : str, default=""
        Optional Singularity execution prefix. If provided, it will be injected before each
        tool command (including piped/chained commands) so they run inside the container.
        Typical value example:
            singularity="singularity exec /path/image.sif"
        or empty string to run on host.

    Outputs
    -------
    Directory structure (under `workdir`):
      - 1.clean/
          {sample}.R1.clean.fastq.gz
          {sample}.R2.clean.fastq.gz
          {sample}.html
          {sample}.json
      - 2.mapping/
          {sample}.sorted.bam
          {sample}.sorted.bam.finished
          {sample}.Markdup.bam
          {sample}.Markdup.metrics.txt
      - 3.gvcf/
          {sample}.{chrom}.g.vcf.gz
          {sample}.{chrom}.g.vcf.gz.tbi
      - 4.merge/
          Merge.{chrom}.g.vcf.gz
          Merge.{chrom}.g.vcf.gz.tbi
          Merge.{chrom}.vcf.gz / related outputs (depending on helper functions)
          Merge.{chrom}.SNP.tsv

    Pipeline Steps
    --------------
    1) fastp (per-sample)
       - Input: raw fq1/fq2
       - Output: cleaned FASTQs + HTML/JSON reports

    2) bwa-mem mapping (per-sample)
       - Input: cleaned FASTQs + reference
       - Output: sorted BAM + ".finished" marker

    3) mark duplicates (per-sample)
       - Input: sorted BAM
       - Output: Markdup BAM + metrics file

    4) gVCF calling (per-sample × per-chrom)
       - Input: Markdup BAM + reference
       - Output: {sample}.{chrom}.g.vcf.gz + .tbi

    5) joint genotyping / merge (per-chrom)
       - Input: all samples' gVCF of one chrom
       - Output: Merge.{chrom}.g.vcf.gz + .tbi

    6) VCF conversion + SNP extraction + filtering + table export (per-chrom)
       - Converts merged gVCF → VCF, extracts SNPs, applies filters, selects filtered SNPs,
         and exports per-chromosome SNP table.

    Notes
    -----
    - This function expects all helper command builders (`fastp`, `bwamem`, `markdup`, `bam2gvcf`,
      `cgvcf`, `gvcf2vcf`, `vcf2snpvcf`, `filtersnp`, `selectfiltersnp`, `vcf2table`) to be
      available and correctly configured for your environment (tools in PATH or inside container).
    - `workdir` should be a filesystem location with sufficient space for intermediate BAM/gVCF files.
    - Make sure `metadata["chrom"]` matches contig names in the reference FASTA.

    Returns
    -------
    None
        The function dispatches jobs via `pipeline()` and does not return a value.
    """
    workdir = Path(workdir).resolve()
    Path("log").mkdir(exist_ok=True)
    scheduler = str(backbend).lower()
    nohup_max_jobs = int(nohup_max_jobs)

    cleanfolder = workdir / '1.clean'
    mappingfolder = workdir / '2.mapping'
    gvcffolder = workdir / '3.gvcf'
    mergefolder = workdir / '4.merge'
    state_path = workdir / ".work.json"

    cleanfolder.mkdir(0o755, exist_ok=True)
    mappingfolder.mkdir(0o755, exist_ok=True)
    gvcffolder.mkdir(0o755, exist_ok=True)
    mergefolder.mkdir(0o755, exist_ok=True)

    reference = Path(metadata["reference"])
    samples_fq: Dict[str, Dict[str, Any]] = metadata["samples"]
    samples = list(samples_fq.keys())
    CHROM = list(metadata["chrom"])
    run_params = {
        "reference": str(Path(metadata["reference"]).resolve()),
        "samples": {str(k): [str(v[0]), str(v[1])] for k, v in samples_fq.items()},
        "chrom": [str(x) for x in CHROM],
        "scheduler": scheduler,
        "nohup_max_jobs": int(nohup_max_jobs),
        "singularity": str(singularity),
    }

    # -------- step1: fastp -------- 15 mins
    step1_lines = []
    step1_items: list[dict] = []
    core = 16
    for sample in samples:
        fq1 = samples_fq[sample][0]
        fq2 = samples_fq[sample][1]
        out_r1 = cleanfolder / f"{sample}.R1.clean.fastq.gz"
        out_r2 = cleanfolder / f"{sample}.R2.clean.fastq.gz"
        out_html = cleanfolder / f"{sample}.html"
        out_json = cleanfolder / f"{sample}.json"
        item_id = f"fastp.{sample}"
        item_outputs = [out_r1, out_r2, out_html, out_json]
        cmd_fastp = fastp(sample, fq1, fq2, cleanfolder, core, singularity=singularity)
        cmd_fastp = _append_state_update_hook(
            cmd_fastp,
            state_path,
            "step1_fastp",
            item_id,
            item_outputs,
        )
        step1_lines.append(
            wrap_cmd(cmd_fastp, f"fastp.{sample}", core, scheduler)
        )
        step1_items.append(
            {
                "id": item_id,
                "outputs": item_outputs,
            }
        )
    step1 = "\n".join(step1_lines)

    step1in = list(itertools.chain.from_iterable(
        [[samples_fq[s][0], samples_fq[s][1]] for s in samples]
    ))
    step1out = list(itertools.chain.from_iterable(
        [[cleanfolder / f"{s}.R1.clean.fastq.gz", cleanfolder / f"{s}.R2.clean.fastq.gz", cleanfolder / f"{s}.html", cleanfolder / f"{s}.json"] for s in samples]
    ))

    # -------- step2: bwamem -------- 255 mins
    step2_lines = []
    step2_items: list[dict] = []
    for sample in samples:
        r1 = cleanfolder / f"{sample}.R1.clean.fastq.gz"
        r2 = cleanfolder / f"{sample}.R2.clean.fastq.gz"
        out_bam = mappingfolder / f"{sample}.sorted.bam"
        out_done = mappingfolder / f"{sample}.sorted.bam.finished"
        item_id = f"bwamem.{sample}"
        item_outputs = [out_bam, out_done]
        cmd_bwa = bwamem(reference, sample, r1, r2, mappingfolder, 64, singularity=singularity)
        cmd_bwa = _append_state_update_hook(
            cmd_bwa,
            state_path,
            "step2_bwamem",
            item_id,
            item_outputs,
        )
        step2_lines.append(
            wrap_cmd(cmd_bwa, f"bwamem.{sample}", 64, scheduler)
        )
        step2_items.append(
            {
                "id": item_id,
                "outputs": item_outputs,
            }
        )
    step2 = "\n".join(step2_lines)

    step2out = list(itertools.chain.from_iterable(
        [[mappingfolder / f"{s}.sorted.bam", mappingfolder / f"{s}.sorted.bam.finished"] for s in samples]
    ))

    # -------- step3: markdup -------- 73 mins
    step3_lines = []
    step3_items: list[dict] = []
    for sample in samples:
        bam = mappingfolder / f"{sample}.sorted.bam"
        out_md_bam = mappingfolder / f"{sample}.Markdup.bam"
        out_md_metric = mappingfolder / f"{sample}.Markdup.metrics.txt"
        item_id = f"markdup.{sample}"
        item_outputs = [out_md_bam, out_md_metric]
        cmd_md = markdup(sample, bam, mappingfolder, 16, 200, singularity=singularity)
        cmd_md = _append_state_update_hook(
            cmd_md,
            state_path,
            "step3_markdup",
            item_id,
            item_outputs,
        )
        step3_lines.append(
            wrap_cmd(cmd_md, f"markdup.{sample}", 16, scheduler)
        )
        step3_items.append(
            {
                "id": item_id,
                "outputs": item_outputs,
            }
        )
    step3 = "\n".join(step3_lines)

    step3out = list(itertools.chain.from_iterable(
        [[mappingfolder / f"{s}.Markdup.bam", mappingfolder / f"{s}.Markdup.metrics.txt"] for s in samples]
    ))

    # -------- step4: bam2gvcf per sample per chrom -------- 451 mins
    step4_lines = []
    step4_items: list[dict] = []
    for sample in samples:
        md_bam = mappingfolder / f"{sample}.Markdup.bam"
        for chrom in CHROM:
            out_gvcf = gvcffolder / f"{sample}.{chrom}.g.vcf.gz"
            out_gvcf_tbi = gvcffolder / f"{sample}.{chrom}.g.vcf.gz.tbi"
            item_id = f"bam2gvcf.{sample}.{chrom}"
            item_outputs = [out_gvcf, out_gvcf_tbi]
            cmd_g = bam2gvcf(reference, sample, md_bam, chrom, gvcffolder, 2, singularity=singularity)
            cmd_g = _append_state_update_hook(
                cmd_g,
                state_path,
                "step4_bam2gvcf",
                item_id,
                item_outputs,
            )
            step4_lines.append(
                wrap_cmd(cmd_g, f"bam2gvcf.{sample}.{chrom}", 2, scheduler)
            )
            step4_items.append(
                {
                    "id": item_id,
                    "outputs": item_outputs,
                }
            )
    step4 = "\n".join(step4_lines)

    step4out = list(itertools.chain.from_iterable(
        [[gvcffolder / f"{s}.{chrom}.g.vcf.gz", gvcffolder / f"{s}.{chrom}.g.vcf.gz.tbi"]
         for s in samples for chrom in CHROM]
    ))

    # -------- step5: cgvcf per chrom -------- 36 mins
    step5_lines = []
    step5_items: list[dict] = []
    for chrom in CHROM:
        gvcfs = [gvcffolder / f"{s}.{chrom}.g.vcf.gz" for s in samples]
        out_merge_g = mergefolder / f"Merge.{chrom}.g.vcf.gz"
        out_merge_g_tbi = mergefolder / f"Merge.{chrom}.g.vcf.gz.tbi"
        item_id = f"cgvcf.{chrom}"
        item_outputs = [out_merge_g, out_merge_g_tbi]
        cmd_c = cgvcf(reference, chrom, gvcfs, mergefolder, singularity=singularity)
        cmd_c = _append_state_update_hook(
            cmd_c,
            state_path,
            "step5_cgvcf",
            item_id,
            item_outputs,
        )
        step5_lines.append(
            wrap_cmd(cmd_c, f"cgvcf.{chrom}", 1, scheduler)
        )
        step5_items.append(
            {
                "id": item_id,
                "outputs": item_outputs,
            }
        )
    step5 = "\n".join(step5_lines)

    step5out = list(itertools.chain.from_iterable(
        [[mergefolder / f"Merge.{chrom}.g.vcf.gz", mergefolder / f"Merge.{chrom}.g.vcf.gz.tbi"] for chrom in CHROM]
    ))

    # -------- step6: gvcf2vcf + snp + filter + table per chrom -------- 32 mins
    step6_lines = []
    step6_items: list[dict] = []
    for chrom in CHROM:
        item_id = f"gvcf2vcf.{chrom}"
        item_outputs = [mergefolder / f"Merge.{chrom}.SNP.tsv"]
        cmd6 = " && ".join([
            gvcf2vcf(reference, chrom, mergefolder, 1, 50, singularity=singularity),
            vcf2snpvcf(reference, chrom, mergefolder, singularity=singularity),
            filtersnp(reference, chrom, mergefolder, singularity=singularity),
            selectfiltersnp(reference, chrom, mergefolder, singularity=singularity),
            vcf2table(reference, chrom, mergefolder, singularity=singularity),
        ])
        cmd6 = _append_state_update_hook(
            cmd6,
            state_path,
            "step6_gvcf2vcf",
            item_id,
            item_outputs,
        )
        step6_lines.append(
            wrap_cmd(cmd6, f"gvcf2vcf.{chrom}", 1, scheduler)
        )
        step6_items.append(
            {
                "id": item_id,
                "outputs": item_outputs,
            }
        )
    step6 = "\n".join(step6_lines)

    step6out = [mergefolder / f"Merge.{chrom}.SNP.tsv" for chrom in CHROM]

    steps_meta = [
        {"id": "step1_fastp", "name": "fastp", "items": step1_items},
        {"id": "step2_bwamem", "name": "bwamem", "items": step2_items},
        {"id": "step3_markdup", "name": "markdup", "items": step3_items},
        {"id": "step4_bam2gvcf", "name": "bam2gvcf", "items": step4_items},
        {"id": "step5_cgvcf", "name": "cgvcf", "items": step5_items},
        {"id": "step6_gvcf2vcf", "name": "gvcf2vcf", "items": step6_items},
    ]
    state, resumed = _init_or_resume_work_state(state_path, run_params, steps_meta)
    if resumed:
        print(f"Resuming pipeline from {state_path}")
    summary = state.get("summary", {})
    if int(summary.get("total_items", 0)) > 0 and int(summary.get("done_items", 0)) >= int(summary.get("total_items", 0)):
        print("All FASTQ2VCF tasks already completed (same parameters).")
        return

    try:
        state["status"] = "running"
        state["started_at"] = state.get("started_at", _utc_now())
        _sync_state_from_fs(state, steps_meta)
        _safe_write_json(state_path, state)
        pipeline(
            [step1, step2, step3, step4, step5, step6],
            [step1in, step1out, step2out, step3out, step4out, step5out],
            [step1out, step2out, step3out, step4out, step5out, step6out],
            scheduler=scheduler,
            nohup_max_jobs=nohup_max_jobs,
            skip_if_outputs_exist=False,
            step_names=[
                "fastp",
                "bwamem",
                "markdup",
                "bam2gvcf",
                "cgvcf",
                "gvcf2vcf",
            ],
            use_rich=True,
        )
    except Exception as e:
        disk_state = _load_json_or_empty(state_path)
        if len(disk_state) > 0:
            state = disk_state
        _sync_state_from_fs(state, steps_meta)
        state["status"] = "failed"
        state["error"] = str(e)
        state["failed_at"] = _utc_now()
        _safe_write_json(state_path, state)
        raise
    else:
        disk_state = _load_json_or_empty(state_path)
        if len(disk_state) > 0:
            state = disk_state
        _sync_state_from_fs(state, steps_meta)
        state["status"] = "completed"
        state["completed_at"] = _utc_now()
        _safe_write_json(state_path, state)
