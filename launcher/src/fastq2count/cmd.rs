use crate::pipeline::safe_job_label;
use std::path::Path;

fn cmd_prefix(singularity: &str) -> String {
    let s = singularity.trim();
    if s.is_empty() {
        String::new()
    } else {
        format!("{s} ")
    }
}

fn qpath(path: &Path) -> String {
    sh_quote(&path.to_string_lossy())
}

pub(super) fn wrap_scheduler_cmd(cmd: &str, job: &str, threads: usize, backend: &str) -> String {
    let safe_job = safe_job_label(job);
    if backend == "csub" {
        let quoted_cmd = sh_quote(cmd);
        let marker = sh_quote(&format!("./log/{}.submitted", safe_job));
        return format!(
            "mkdir -p ./log && out=$(csub -J {} -o ./log/{}.%J.o -e ./log/{}.%J.e -q c01 -n {} {}); status=$?; printf '%s\\n' \"$out\"; if [ \"$status\" -ne 0 ]; then exit \"$status\"; fi; job_id=$(printf '%s\\n' \"$out\" | sed -n 's/.*<\\([0-9][0-9]*\\)>.*/\\1/p' | head -n 1); if [ -z \"$job_id\" ]; then job_id=$(printf '%s\\n' \"$out\" | awk '{{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/) {{print $i; exit}}}}'); fi; if [ -n \"$job_id\" ]; then printf '%s\\n' \"$job_id\" > {}; else : > {}; fi",
            safe_job,
            safe_job,
            safe_job,
            threads.max(1),
            quoted_cmd,
            marker,
            marker
        );
    }
    let quoted_cmd = sh_quote(cmd);
    format!(
        "nohup bash -lc {} > ./log/{}.o 2> ./log/{}.e",
        quoted_cmd, safe_job, safe_job
    )
}

pub(super) fn cmd_fastp(
    sample: &str,
    fq1: &Path,
    fq2: &Path,
    out_clean_dir: &Path,
    out_qc_dir: &Path,
    core: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let out_r1 = out_clean_dir.join(format!("{sample}.R1.clean.fastq.gz"));
    let out_r2 = out_clean_dir.join(format!("{sample}.R2.clean.fastq.gz"));
    let out_html = out_qc_dir.join(format!("{sample}.html"));
    let out_json = out_qc_dir.join(format!("{sample}.json"));
    format!(
        "mkdir -p {} {} && {prefix}fastp -i {} -I {} -o {} -O {} --html {} --json {} -w {}",
        qpath(out_clean_dir),
        qpath(out_qc_dir),
        qpath(fq1),
        qpath(fq2),
        qpath(&out_r1),
        qpath(&out_r2),
        qpath(&out_html),
        qpath(&out_json),
        core.max(1),
    )
}

pub(super) fn cmd_hisat2_index(
    reference: &Path,
    annotation: &Path,
    index_dir: &Path,
    threads: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let index_prefix = index_dir.join("reference");
    let ss_path = index_dir.join("reference.ss");
    let exon_path = index_dir.join("reference.exon");
    let done_path = index_dir.join("reference.index.ok");

    format!(
        "mkdir -p {idx} && rm -f {done} {ss} {exon} && IDX_PREF={pref} && \
    SP_SCRIPT=''; EX_SCRIPT=''; \
    if command -v hisat2_extract_splice_sites.py >/dev/null 2>&1; then SP_SCRIPT=$(command -v hisat2_extract_splice_sites.py); elif command -v extract_splice_sites.py >/dev/null 2>&1; then SP_SCRIPT=$(command -v extract_splice_sites.py); fi; \
    if command -v hisat2_extract_exons.py >/dev/null 2>&1; then EX_SCRIPT=$(command -v hisat2_extract_exons.py); elif command -v extract_exons.py >/dev/null 2>&1; then EX_SCRIPT=$(command -v extract_exons.py); fi; \
    if [ -n \"$SP_SCRIPT\" ] && [ -n \"$EX_SCRIPT\" ]; then \"$SP_SCRIPT\" {ann} > {ss} && \"$EX_SCRIPT\" {ann} > {exon}; else echo 'Warning: hisat2_extract_splice_sites.py/exons.py not found; build index without --ss/--exon' >&2; : > {ss}; : > {exon}; fi && \
    if [ -s {ss} ] && [ -s {exon} ]; then {prefix}hisat2-build -p {threads} --ss {ss} --exon {exon} {ref} \"$IDX_PREF\"; else {prefix}hisat2-build -p {threads} {ref} \"$IDX_PREF\"; fi && \
    ok_ht2=1; for i in 1 2 3 4 5 6 7 8; do [ -s \"$IDX_PREF.$i.ht2\" ] || ok_ht2=0; done; \
    ok_ht2l=1; for i in 1 2 3 4 5 6 7 8; do [ -s \"$IDX_PREF.$i.ht2l\" ] || ok_ht2l=0; done; \
    if [ \"$ok_ht2\" -eq 1 ] || [ \"$ok_ht2l\" -eq 1 ]; then touch {done}; else echo 'hisat2 index outputs are incomplete' >&2; exit 1; fi",
        idx = qpath(index_dir),
        done = qpath(&done_path),
        ss = qpath(&ss_path),
        exon = qpath(&exon_path),
        ann = qpath(annotation),
        prefix = prefix,
        threads = threads.max(1),
        ref = qpath(reference),
        pref = qpath(&index_prefix),
    )
}

pub(super) fn cmd_hisat2_align_sort_index(
    sample: &str,
    index_dir: &Path,
    in_clean_dir: &Path,
    out_mapping_dir: &Path,
    align_threads: usize,
    sort_threads: usize,
    strandness: Option<&str>,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let index_prefix = index_dir.join("reference");
    let fq1 = in_clean_dir.join(format!("{sample}.R1.clean.fastq.gz"));
    let fq2 = in_clean_dir.join(format!("{sample}.R2.clean.fastq.gz"));
    let out_bam = out_mapping_dir.join(format!("{sample}.bam"));
    let out_bai = out_mapping_dir.join(format!("{sample}.bam.bai"));
    let out_log = out_mapping_dir.join(format!("{sample}.hisat2.log"));
    let strand_opt = strandness
        .map(str::trim)
        .filter(|x| !x.is_empty() && !x.eq_ignore_ascii_case("none"))
        .map(sh_quote)
        .map(|x| format!("--rna-strandness {x}"))
        .unwrap_or_default();
    format!(
        "mkdir -p {mapdir} && rm -f {bam} {bai} {log} && {prefix}hisat2 -p {ath} --new-summary {strand_opt} -x {idx} -1 {r1} -2 {r2} 2> {log} | {prefix}samtools sort -@ {sth} -o {bam} - && {prefix}samtools index -@ {sth} {bam}",
        mapdir = qpath(out_mapping_dir),
        bam = qpath(&out_bam),
        bai = qpath(&out_bai),
        log = qpath(&out_log),
        prefix = prefix,
        ath = align_threads.max(1),
        strand_opt = strand_opt,
        idx = qpath(&index_prefix),
        r1 = qpath(&fq1),
        r2 = qpath(&fq2),
        sth = sort_threads.max(1),
    )
}

pub(super) fn cmd_featurecounts_and_metrics(
    annotation: &Path,
    feature_type: &str,
    gene_attr: &str,
    samples: &[String],
    mapping_dir: &Path,
    count_dir: &Path,
    metrics_script: &Path,
    threads: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let out_counts = count_dir.join("gene_counts.txt");
    let out_fpkm = count_dir.join("gene_counts.fpkm.tsv");
    let out_tpm = count_dir.join("gene_counts.tpm.tsv");
    let bam_inputs = samples
        .iter()
        .map(|sample| qpath(&mapping_dir.join(format!("{sample}.bam"))))
        .collect::<Vec<String>>()
        .join(" ");
    format!(
        "mkdir -p {count_dir} && rm -f {counts} {fpkm} {tpm} && {prefix}featureCounts -T {threads} -p -t {ftype} -g {gattr} -a {ann} -o {counts} {bams} && python3 {script} {counts} {fpkm} {tpm}",
        count_dir = qpath(count_dir),
        counts = qpath(&out_counts),
        fpkm = qpath(&out_fpkm),
        tpm = qpath(&out_tpm),
        prefix = prefix,
        threads = threads.max(1),
        ftype = sh_quote(feature_type),
        gattr = sh_quote(gene_attr),
        ann = qpath(annotation),
        bams = bam_inputs,
        script = qpath(metrics_script),
    )
}

pub(super) fn sh_quote(raw: &str) -> String {
    let escaped = raw.replace('\'', "'\"'\"'");
    format!("'{escaped}'")
}
