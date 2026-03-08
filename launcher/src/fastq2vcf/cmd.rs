use crate::pipeline::safe_job_label;
use std::path::{Path, PathBuf};

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
    if backend == "csub" {
        let safe_job = safe_job_label(job);
        return format!(
            "csub -J {} -o ./log/{}.%J.o -e ./log/{}.%J.e -q c01 -n {} \"{}\"",
            job, safe_job, safe_job, threads, cmd
        );
    }
    let safe_cmd = cmd.replace('"', "\\\"");
    format!(
        "nohup bash -c \"{}\" > ./log/{}.o 2> ./log/{}.e",
        safe_cmd, job, job
    )
}

pub(super) fn cmd_fastp(
    sample: &str,
    fq1: &Path,
    fq2: &Path,
    out: &Path,
    core: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let out_r1 = out.join(format!("{sample}.R1.clean.fastq.gz"));
    let out_r2 = out.join(format!("{sample}.R2.clean.fastq.gz"));
    let out_html = out.join(format!("{sample}.html"));
    let out_json = out.join(format!("{sample}.json"));
    format!(
        "{prefix}fastp -i {} -I {} -o {} -O {} --html {} --json {} -w {}",
        qpath(fq1),
        qpath(fq2),
        qpath(&out_r1),
        qpath(&out_r2),
        qpath(&out_html),
        qpath(&out_json),
        core
    )
}

pub(super) fn cmd_bwamem(
    reference: &Path,
    sample: &str,
    fq1: &Path,
    fq2: &Path,
    out: &Path,
    core: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let out_bam = out.join(format!("{sample}.sorted.bam"));
    let out_done = out.join(format!("{sample}.sorted.bam.finished"));
    let out_bai = out.join(format!("{sample}.sorted.bam.bai"));
    let sort_tmp_prefix = out.join(format!("{sample}.sorted.bam.tmp"));
    let sort_tmp_glob = format!("{}*", qpath(&sort_tmp_prefix));
    let rg = sh_quote(&format!(
        "@RG\\tID:{sample}\\tPL:illumina\\tLB:{sample}\\tSM:{sample}"
    ));
    let cleanup_cmd = format!(
        "rm -f {} {} {} {}",
        qpath(&out_bam),
        qpath(&out_bai),
        qpath(&out_done),
        sort_tmp_glob,
    );
    format!(
        "{cleanup_cmd} && {prefix}bwa mem -t {core} -R {rg} {} {} {} | {prefix}samtools sort -@ {core} -T {} -o {} && touch {}",
        qpath(reference),
        qpath(fq1),
        qpath(fq2),
        qpath(&sort_tmp_prefix),
        qpath(&out_bam),
        qpath(&out_done),
    )
}

pub(super) fn cmd_markdup(
    sample: &str,
    bam: &Path,
    out: &Path,
    core: usize,
    _mem: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let out_bam = out.join(format!("{sample}.Markdup.bam"));
    let out_bai = out.join(format!("{sample}.Markdup.bam.bai"));
    let out_flag = out.join(format!("{sample}.flagstat"));
    let out_cov = out.join(format!("{sample}.coverage"));
    let out_metric = out.join(format!("{sample}.Markdup.metrics.txt"));
    let cleanup_cmd = format!(
        "rm -f {} {} {} {} {}",
        qpath(&out_bam),
        qpath(&out_bai),
        qpath(&out_flag),
        qpath(&out_cov),
        qpath(&out_metric),
    );
    let tmpdir_setup_cmd = format!(
        "TMPDIR_BASE='/local/tmp'; if [ ! -d \"$TMPDIR_BASE\" ]; then TMPDIR_BASE={}; fi; mkdir -p \"$TMPDIR_BASE\"",
        qpath(out),
    );
    let markdup_cmd = format!(
        "{prefix}sambamba markdup -t {core} --tmpdir \"$TMPDIR_BASE\" {} {}",
        qpath(bam),
        qpath(&out_bam),
    );
    let index_cmd = format!("{prefix}samtools index -@ {core} {}", qpath(&out_bam));
    let metric_cmd = format!("printf 'tool\\tsambamba markdup\\n' > {}", qpath(&out_metric));
    let flagstat_cmd = format!("{prefix}samtools flagstat {} > {}", qpath(&out_bam), qpath(&out_flag));
    let coverage_cmd = format!("{prefix}samtools coverage {} > {}", qpath(&out_bam), qpath(&out_cov));
    format!(
        "{cleanup_cmd} && {tmpdir_setup_cmd} && {markdup_cmd} && {index_cmd} && {metric_cmd} && {flagstat_cmd} && {coverage_cmd}",
    )
}

pub(super) fn cmd_bam2gvcf(
    reference: &Path,
    sample: &str,
    bam: &Path,
    chrom: &str,
    out: &Path,
    core: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let out_g = out.join(format!("{sample}.{chrom}.g.vcf.gz"));
    format!(
        "{prefix}gatk HaplotypeCaller -R {} --native-pair-hmm-threads {} -ERC GVCF -I {} -O {} -L {}",
        qpath(reference),
        core,
        qpath(bam),
        qpath(&out_g),
        sh_quote(chrom),
    )
}

pub(super) fn cmd_cgvcf(
    reference: &Path,
    chrom: &str,
    gvcfs: &[PathBuf],
    out: &Path,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let out_g = out.join(format!("Merge.{chrom}.g.vcf.gz"));
    let variants = gvcfs
        .iter()
        .map(|p| format!("--variant {}", qpath(p)))
        .collect::<Vec<String>>()
        .join(" ");
    format!(
        "{prefix}gatk CombineGVCFs -R {} {} -O {}",
        qpath(reference),
        variants,
        qpath(&out_g),
    )
}

pub(super) fn cmd_gvcf2vcf(
    reference: &Path,
    chrom: &str,
    out: &Path,
    core: usize,
    mem: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let in_g = out.join(format!("Merge.{chrom}.g.vcf.gz"));
    let out_vcf = out.join(format!("Merge.{chrom}.vcf.gz"));
    format!(
        "{prefix}gatk --java-options '-Xmx{mem}G -XX:ParallelGCThreads={core}' GenotypeGVCFs -R {} -V {} -O {}",
        qpath(reference),
        qpath(&in_g),
        qpath(&out_vcf),
    )
}

pub(super) fn cmd_vcf2snpvcf(
    reference: &Path,
    chrom: &str,
    out: &Path,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let in_vcf = out.join(format!("Merge.{chrom}.vcf.gz"));
    let out_snp = out.join(format!("Merge.{chrom}.SNP.vcf.gz"));
    format!(
        "{prefix}gatk SelectVariants -R {} -V {} --select-type SNP --restrict-alleles-to BIALLELIC -O {}",
        qpath(reference),
        qpath(&in_vcf),
        qpath(&out_snp),
    )
}

pub(super) fn cmd_filtersnp(
    reference: &Path,
    chrom: &str,
    out: &Path,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let in_snp = out.join(format!("Merge.{chrom}.SNP.vcf.gz"));
    let out_filtered = out.join(format!("Merge.{chrom}.SNP.filter.vcf.gz"));
    format!(
        "{prefix}gatk VariantFiltration -R {} -V {} -O {} --filter-name 'QUAL30' --filter-expression 'QUAL < 30.0' --filter-name 'FS60' --filter-expression 'FS > 60.0' --filter-name 'QD2' --filter-expression 'QD < 2.0' --filter-name 'SOR3' --filter-expression 'SOR > 3.0' --filter-name 'MQ40' --filter-expression 'MQ < 40.0' --filter-name 'ReadPosRankSum-8' --filter-expression 'ReadPosRankSum < -8.0' --filter-name 'MQRankSum-12.5' --filter-expression 'MQRankSum < -12.5'",
        qpath(reference),
        qpath(&in_snp),
        qpath(&out_filtered),
    )
}

pub(super) fn cmd_selectfiltersnp(
    reference: &Path,
    chrom: &str,
    out: &Path,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let in_filtered = out.join(format!("Merge.{chrom}.SNP.filter.vcf.gz"));
    let out_pass = out.join(format!("Merge.{chrom}.SNP.filtered.vcf.gz"));
    format!(
        "{prefix}gatk SelectVariants -R {} -V {} --exclude-filtered -O {}",
        qpath(reference),
        qpath(&in_filtered),
        qpath(&out_pass),
    )
}

pub(super) fn cmd_vcf2table(
    reference: &Path,
    chrom: &str,
    out: &Path,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let in_pass = out.join(format!("Merge.{chrom}.SNP.filtered.vcf.gz"));
    let out_tsv = out.join(format!("Merge.{chrom}.SNP.tsv"));
    format!(
        "{prefix}gatk VariantsToTable -R {} -V {} -F CHROM -F POS -F REF -F ALT -GF DP -GF AD -GF GQ -O {}",
        qpath(reference),
        qpath(&in_pass),
        qpath(&out_tsv),
    )
}

pub(super) fn cmd_snpvcf_to_gt_and_missing(
    chrom: &str,
    merge_dir: &Path,
    impute_dir: &Path,
    min_dp: usize,
    min_gq: usize,
    min_ad: usize,
    core: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let in_vcf = merge_dir.join(format!("Merge.{chrom}.SNP.filtered.vcf.gz"));
    let out_gt = impute_dir.join(format!("Merge.{chrom}.SNP.GT.vcf.gz"));
    let miss_prefix = impute_dir.join(format!("Merge.{chrom}.SNP.GT"));
    let out_lmiss = impute_dir.join(format!("Merge.{chrom}.SNP.GT.lmiss"));
    let out_imiss = impute_dir.join(format!("Merge.{chrom}.SNP.GT.imiss"));
    let expr1 = format!("FMT/DP<{min_dp} || FMT/GQ<{min_gq} || FMT/AD[:1]<{min_ad}");
    let expr2 = format!("FMT/DP<{min_dp} || FMT/GQ<{min_gq} || FMT/AD[1]<{min_ad}");
    let expr3 = format!("FMT/DP<{min_dp} || FMT/GQ<{min_gq}");
    let setgt_cmd = format!(
        "( {prefix}bcftools filter -Ou -S . -e {} {} || {prefix}bcftools filter -Ou -S . -e {} {} || {prefix}bcftools filter -Ou -S . -e {} {} )",
        sh_quote(&expr1),
        qpath(&in_vcf),
        sh_quote(&expr2),
        qpath(&in_vcf),
        sh_quote(&expr3),
        qpath(&in_vcf),
    );
    let keep_gt_cmd = format!(
        "{prefix}bcftools annotate --threads {core} -x FORMAT,^FORMAT/GT -Oz -o {}",
        qpath(&out_gt),
    );
    let index_cmd = format!("{prefix}tabix -f -p vcf {}", qpath(&out_gt));
    let plink_missing_cmd = format!(
        "{prefix}plink --vcf {} --allow-extra-chr --double-id --set-missing-var-ids @:# --threads {core} --missing --out {}",
        qpath(&out_gt),
        qpath(&miss_prefix),
    );
    let write_empty_miss_cmd = format!(
        "printf 'CHR\\tSNP\\tN_MISS\\tN_GENO\\tF_MISS\\n' > {} && printf 'FID\\tIID\\tMISS_PHENO\\tN_MISS\\tN_GENO\\tF_MISS\\n' > {}",
        qpath(&out_lmiss),
        qpath(&out_imiss),
    );
    let miss_cmd = format!(
        "NVAR=$({prefix}bcftools index -n {}) && if [ \"${{NVAR:-0}}\" -gt 0 ]; then {plink_missing_cmd}; else {write_empty_miss_cmd}; fi",
        qpath(&out_gt),
    );
    format!("{setgt_cmd} | {keep_gt_cmd} && {index_cmd} && {miss_cmd}")
}

pub(super) fn cmd_beagle_impute(
    chrom: &str,
    impute_dir: &Path,
    core: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let in_gt = impute_dir.join(format!("Merge.{chrom}.SNP.GT.vcf.gz"));
    let out_prefix = impute_dir.join(format!("Merge.{chrom}.SNP.GT.imp"));
    let out_imp = impute_dir.join(format!("Merge.{chrom}.SNP.GT.imp.vcf.gz"));
    let beagle_cmd = format!(
        "{prefix}beagle gt={} out={} nthreads={core}",
        qpath(&in_gt),
        qpath(&out_prefix),
    );
    let index_cmd = format!("{prefix}tabix -f -p vcf {}", qpath(&out_imp));
    let copy_empty_cmd = format!(
        "{prefix}bcftools view -Oz -o {} {} && {index_cmd}",
        qpath(&out_imp),
        qpath(&in_gt),
    );
    format!(
        "NVAR=$({prefix}bcftools index -n {}) && if [ \"${{NVAR:-0}}\" -gt 0 ]; then {beagle_cmd} && {index_cmd}; else {copy_empty_cmd}; fi",
        qpath(&in_gt),
    )
}

pub(super) fn cmd_filter_imputed_by_maf_and_missing(
    chrom: &str,
    chroms: &[String],
    impute_dir: &Path,
    maf: f64,
    max_missing: f64,
    core: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let in_imp = impute_dir.join(format!("Merge.{chrom}.SNP.GT.imp.vcf.gz"));
    let out_vcf = impute_dir.join(format!("Merge.{chrom}.SNP.GT.imp.maf0.02.miss0.2.vcf.gz"));
    let site_lmiss = impute_dir.join(format!("Merge.{chrom}.SNP.GT.lmiss"));
    let site_keep = impute_dir.join(format!("Merge.{chrom}.preimpute.site.keep.txt"));
    let sample_keep = impute_dir.join("Merge.preimpute.sample.keep.txt");
    let sample_tmp = PathBuf::from(format!("{}.tmp", sample_keep.to_string_lossy()));
    let sample_lock = PathBuf::from(format!("{}.lock", sample_keep.to_string_lossy()));

    let all_imiss_txt = chroms
        .iter()
        .map(|c| impute_dir.join(format!("Merge.{c}.SNP.GT.imiss")))
        .map(|p| qpath(&p))
        .collect::<Vec<String>>()
        .join(" ");

    let build_sample_keep = format!(
        "if [ ! -s {} ]; then if mkdir {} 2>/dev/null; then awk 'NR==1{{next}} {{id=$2; miss[id]+=$4; geno[id]+=$5}} END{{for (id in miss) if (geno[id]>0 && miss[id]/geno[id]<={}) print id}}' {} | sort > {} && mv {} {}; rmdir {}; else while [ -d {} ]; do sleep 1; done; fi; fi",
        qpath(&sample_keep),
        qpath(&sample_lock),
        max_missing,
        all_imiss_txt,
        qpath(&sample_tmp),
        qpath(&sample_tmp),
        qpath(&sample_keep),
        qpath(&sample_lock),
        qpath(&sample_lock),
    );
    let build_site_keep = format!(
        "awk 'NR>1 && $5<={} {{n=split($2,a,\":\"); if (n>=2) print a[1]\"\\t\"a[2]}}' {} > {}",
        max_missing,
        qpath(&site_lmiss),
        qpath(&site_keep),
    );
    let filter_cmd = format!(
        "{prefix}bcftools view --threads {core} -S {} -T {} -i 'MAF>={maf}' -Oz -o {} {}",
        qpath(&sample_keep),
        qpath(&site_keep),
        qpath(&out_vcf),
        qpath(&in_imp),
    );
    let index_cmd = format!("{prefix}tabix -f -p vcf {}", qpath(&out_vcf));
    let copy_empty_cmd = format!(
        "{prefix}bcftools view -Oz -o {} {} && {index_cmd}",
        qpath(&out_vcf),
        qpath(&in_imp),
    );
    format!(
        "NVAR=$({prefix}bcftools index -n {}) && if [ \"${{NVAR:-0}}\" -gt 0 ]; then {build_sample_keep} && {build_site_keep} && {filter_cmd} && {index_cmd}; else {copy_empty_cmd}; fi",
        qpath(&in_imp),
    )
}

pub(super) fn cmd_concat_imputed_vcfs(
    chroms: &[String],
    impute_dir: &Path,
    genotype_dir: &Path,
    core: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let out_vcf = genotype_dir.join("Merge.SNP.GT.imp.maf0.02.miss0.2.vcf.gz");
    let inputs: Vec<PathBuf> = chroms
        .iter()
        .map(|c| impute_dir.join(format!("Merge.{c}.SNP.GT.imp.maf0.02.miss0.2.vcf.gz")))
        .collect();
    let first_input = inputs[0].clone();
    let inputs_txt = inputs
        .iter()
        .map(|p| qpath(p))
        .collect::<Vec<String>>()
        .join(" ");
    let nonempty_list = genotype_dir.join("Merge.concat.nonempty.list.txt");
    let build_nonempty_cmd = format!(
        ": > {}; for f in {}; do n=$({prefix}bcftools index -n \"$f\" 2>/dev/null || echo 0); if [ \"${{n:-0}}\" -gt 0 ]; then printf '%s\\n' \"$f\" >> {}; fi; done",
        qpath(&nonempty_list),
        inputs_txt,
        qpath(&nonempty_list),
    );
    let copy_empty_cmd = format!(
        "{prefix}bcftools view -Oz -o {} {}",
        qpath(&out_vcf),
        qpath(&first_input),
    );
    let copy_single_cmd = format!(
        "f=$(head -n 1 {}); {prefix}bcftools view -Oz -o {} \"$f\"",
        qpath(&nonempty_list),
        qpath(&out_vcf),
    );
    let concat_cmd = format!(
        "{prefix}bcftools concat --threads {core} -a -f {} -Oz -o {}",
        qpath(&nonempty_list),
        qpath(&out_vcf),
    );
    let index_cmd = format!("{prefix}tabix -f -p vcf {}", qpath(&out_vcf));
    format!(
        "{build_nonempty_cmd} && nfile=$(wc -l < {}) && if [ \"${{nfile:-0}}\" -eq 0 ]; then {copy_empty_cmd}; elif [ \"${{nfile:-0}}\" -eq 1 ]; then {copy_single_cmd}; else {concat_cmd}; fi && {index_cmd}",
        qpath(&nonempty_list),
    )
}

pub(super) fn sh_quote(raw: &str) -> String {
    let escaped = raw.replace('\'', "'\"'\"'");
    format!("'{escaped}'")
}
