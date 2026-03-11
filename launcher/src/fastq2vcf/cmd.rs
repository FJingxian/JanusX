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
    let safe_job = safe_job_label(job);
    if backend == "csub" {
        let quoted_cmd = sh_quote(cmd);
        return format!(
            "mkdir -p ./log && : > ./log/{}.submitted && csub -J {} -o ./log/{}.%J.o -e ./log/{}.%J.e -q c01 -n {} {}",
            safe_job, safe_job, safe_job, safe_job, threads, quoted_cmd
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

pub(super) fn cmd_bwamem_then_markdup(
    reference: &Path,
    sample: &str,
    fq1: &Path,
    fq2: &Path,
    out: &Path,
    bwa_core: usize,
    _markdup_core: usize,
    _markdup_mem: usize,
    aligner_cmd: &str,
    singularity: &str,
) -> String {
    // Best-path only: bwa-mem2 + samblaster, then GATK sort/index.
    // This avoids large intermediate sorted BAM artifacts from the old two-stage path.
    let prefix = cmd_prefix(singularity);
    let out_md_bam = out.join(format!("{sample}.Markdup.bam"));
    let out_md_bai = out.join(format!("{sample}.Markdup.bam.bai"));
    let out_tmp_sam = out.join(format!("{sample}.Markdup.sam.tmp"));
    let tmp_tag = safe_job_label(sample);
    let rg = sh_quote(&format!(
        "@RG\\tID:{sample}\\tPL:illumina\\tLB:{sample}\\tSM:{sample}"
    ));
    let setup_tmp_cmd = format!(
        "TMPDIR_BASE='/local/tmp'; if [ ! -d \"$TMPDIR_BASE\" ] || [ ! -w \"$TMPDIR_BASE\" ]; then TMPDIR_BASE={}; fi; mkdir -p \"$TMPDIR_BASE\"; PICARD_TMP=\"$TMPDIR_BASE/{}.sortsam.tmp\"; mkdir -p \"$PICARD_TMP\"",
        qpath(out),
        tmp_tag,
    );
    let cleanup_cmd = format!(
        "rm -f {} {} {} && rm -rf \"$PICARD_TMP\"",
        qpath(&out_md_bam),
        qpath(&out_md_bai),
        qpath(&out_tmp_sam),
    );
    let markdup_cmd = format!(
        "{aligner_cmd} mem -t {bwa_core} -R {rg} {} {} {} | {prefix}samblaster > {}",
        qpath(reference),
        qpath(fq1),
        qpath(fq2),
        qpath(&out_tmp_sam),
    );
    let sort_cmd = format!(
        "{prefix}gatk SortSam -I {} -O {} --SORT_ORDER coordinate --TMP_DIR \"$PICARD_TMP\"",
        qpath(&out_tmp_sam),
        qpath(&out_md_bam),
    );
    let index_cmd = format!(
        "{prefix}gatk BuildBamIndex -I {} -O {}",
        qpath(&out_md_bam),
        qpath(&out_md_bai),
    );
    format!("{setup_tmp_cmd} && {cleanup_cmd} && {markdup_cmd} && {sort_cmd} && {index_cmd} && rm -f {} && rm -rf \"$PICARD_TMP\"", qpath(&out_tmp_sam))
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
    core: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let db_dir = out.join(format!("Merge.{chrom}.gendb"));
    let done = out.join(format!("Merge.{chrom}.gendb.done"));
    let tmp_tag = safe_job_label(chrom);
    let variants = gvcfs
        .iter()
        .map(|p| format!("--variant {}", qpath(p)))
        .collect::<Vec<String>>()
        .join(" ");
    let setup_tmp_cmd = format!(
        "TMPDIR_BASE='/local/tmp'; if [ ! -d \"$TMPDIR_BASE\" ] || [ ! -w \"$TMPDIR_BASE\" ]; then TMPDIR_BASE={}; fi; mkdir -p \"$TMPDIR_BASE\"; GDB_TMP=\"$TMPDIR_BASE/{}.gendb.tmp\"; mkdir -p \"$GDB_TMP\"",
        qpath(out),
        tmp_tag,
    );
    let cleanup_cmd = format!(
        "rm -rf {} \"$GDB_TMP\" && mkdir -p \"$GDB_TMP\" && rm -f {}",
        qpath(&db_dir),
        qpath(&done)
    );
    format!(
        "{setup_tmp_cmd} && {cleanup_cmd} && {prefix}gatk GenomicsDBImport -R {} -L {} --genomicsdb-workspace-path {} --tmp-dir \"$GDB_TMP\" --reader-threads {core} {} && touch {} && rm -rf \"$GDB_TMP\"",
        qpath(reference),
        sh_quote(chrom),
        qpath(&db_dir),
        variants,
        qpath(&done),
    )
}

pub(super) fn cmd_gvcf2snp_table(
    reference: &Path,
    chrom: &str,
    out: &Path,
    core: usize,
    mem: usize,
    singularity: &str,
) -> String {
    let prefix = cmd_prefix(singularity);
    let db_dir = out.join(format!("Merge.{chrom}.gendb"));
    let raw_vcf = out.join(format!("Merge.{chrom}.raw.vcf.gz"));
    let raw_tbi = out.join(format!("Merge.{chrom}.raw.vcf.gz.tbi"));
    let out_snp_f = out.join(format!("Merge.{chrom}.SNP.filtered.vcf.gz"));
    let out_snp_f_tbi = out.join(format!("Merge.{chrom}.SNP.filtered.vcf.gz.tbi"));
    let out_snp_tsv = out.join(format!("Merge.{chrom}.SNP.tsv"));
    let filter_expr = sh_quote(
        "QUAL>=30 && INFO/FS<=60 && INFO/QD>=2 && INFO/SOR<=3 && INFO/MQ>=40 && INFO/ReadPosRankSum>=-8 && INFO/MQRankSum>=-12.5",
    );
    let setup_tmp_cmd = format!(
        "TMPDIR_BASE='/local/tmp'; if [ ! -d \"$TMPDIR_BASE\" ] || [ ! -w \"$TMPDIR_BASE\" ]; then TMPDIR_BASE={}; fi; mkdir -p \"$TMPDIR_BASE\"",
        qpath(out),
    );
    let cleanup_cmd = format!(
        "rm -f {} {} {} {} {}",
        qpath(&raw_vcf),
        qpath(&raw_tbi),
        qpath(&out_snp_f),
        qpath(&out_snp_f_tbi),
        qpath(&out_snp_tsv),
    );
    let genotype_cmd = format!(
        "{prefix}gatk --java-options '-Xmx{mem}G -XX:ParallelGCThreads={core}' GenotypeGVCFs -R {} -V {} -O {}",
        qpath(reference),
        sh_quote(&format!("gendb://{}", db_dir.to_string_lossy())),
        qpath(&raw_vcf),
    );
    let filter_cmd = format!(
        "{prefix}bcftools view --threads {core} -m2 -M2 -v snps {} -Ou | {prefix}bcftools filter --threads {core} -i {} -Oz -o {}",
        qpath(&raw_vcf),
        filter_expr,
        qpath(&out_snp_f),
    );
    let index_cmd = format!("{prefix}tabix -f -p vcf {}", qpath(&out_snp_f));
    let header_cmd = format!(
        "SAMPLE_COLS=$({prefix}bcftools query -l {} | awk '{{printf \"\\t%s.DP\\t%s.AD\\t%s.GQ\", $1,$1,$1}}'); printf 'CHROM\\tPOS\\tREF\\tALT%s\\n' \"$SAMPLE_COLS\" > {}",
        qpath(&out_snp_f),
        qpath(&out_snp_tsv),
    );
    let table_cmd = format!(
        "{prefix}bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%DP\\t%AD\\t%GQ]\\n' {} >> {}",
        qpath(&out_snp_f),
        qpath(&out_snp_tsv),
    );
    format!(
        "{setup_tmp_cmd} && {cleanup_cmd} && {genotype_cmd} && {filter_cmd} && {index_cmd} && {header_cmd} && {table_cmd} && test -s {} && rm -f {} {}",
        qpath(&out_snp_tsv),
        qpath(&raw_vcf),
        qpath(&raw_tbi),
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
    let out_imp = impute_dir.join(format!("Merge.{chrom}.SNP.GT.imp.vcf.gz"));
    let out_imp_tbi = impute_dir.join(format!("Merge.{chrom}.SNP.GT.imp.vcf.gz.tbi"));
    let out_ok = impute_dir.join(format!("Merge.{chrom}.SNP.GT.imp.ok"));
    let tmp_prefix = impute_dir.join(format!("Merge.{chrom}.SNP.GT.imp.tmp"));
    let tmp_imp = impute_dir.join(format!("Merge.{chrom}.SNP.GT.imp.tmp.vcf.gz"));
    let tmp_imp_tbi = impute_dir.join(format!("Merge.{chrom}.SNP.GT.imp.tmp.vcf.gz.tbi"));
    let heap_gb = core.saturating_mul(4).clamp(16, 256);
    let xmx_expr = format!("${{JANUSX_BEAGLE_XMX_GB:-{heap_gb}}}");
    let java_opts = format!(
        "_JAVA_OPTIONS=\"-Xmx{xmx_expr}g ${{_JAVA_OPTIONS:-}}\" JAVA_TOOL_OPTIONS=\"-Djava.io.tmpdir=/tmp ${{JAVA_TOOL_OPTIONS:-}}\""
    );
    let beagle_cmd = format!(
        "{java_opts} {prefix}beagle gt={} out={} nthreads={core}",
        qpath(&in_gt),
        qpath(&tmp_prefix),
    );
    let index_tmp_cmd = format!("{prefix}tabix -f -p vcf {}", qpath(&tmp_imp));
    let cleanup_cmd = format!(
        "rm -f {} {} {} {} {}",
        qpath(&tmp_imp),
        qpath(&tmp_imp_tbi),
        qpath(&out_imp),
        qpath(&out_imp_tbi),
        qpath(&out_ok),
    );
    let finalize_cmd = format!(
        "mv {} {} && mv {} {} && touch {}",
        qpath(&tmp_imp),
        qpath(&out_imp),
        qpath(&tmp_imp_tbi),
        qpath(&out_imp_tbi),
        qpath(&out_ok),
    );
    let copy_empty_cmd = format!(
        "{prefix}bcftools view -Oz -o {} {} && {index_tmp_cmd} && {finalize_cmd}",
        qpath(&tmp_imp),
        qpath(&in_gt),
    );
    let npos_cmd = format!(
        "NPOS=$({prefix}bcftools query -f '%POS\\n' {} | awk 'NR==1{{first=$1;next}} NR>1 && $1!=first{{found=1; print 2; exit}} END{{if(found==0){{if(NR==0) print 0; else print 1}}}}')",
        qpath(&in_gt),
    );
    format!(
        "NVAR=$({prefix}bcftools index -n {}) && {npos_cmd} && {cleanup_cmd} && if [ \"${{NVAR:-0}}\" -gt 1 ] && [ \"${{NPOS:-0}}\" -gt 1 ]; then {beagle_cmd} && {index_tmp_cmd} && {finalize_cmd}; else {copy_empty_cmd}; fi",
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
        "if [ ! -s {} ]; then if mkdir {} 2>/dev/null; then : > {} && (awk 'NR==1{{next}} {{id=$2; miss[id]+=$4; geno[id]+=$5}} END{{for (id in miss) if (geno[id]>0 && miss[id]/geno[id]<={}) print id}}' {} 2>/dev/null | sort > {} || true) && mv {} {} && rmdir {} || true; else while [ -d {} ]; do sleep 1; done; fi; fi; [ -f {} ] || : > {}",
        qpath(&sample_keep),
        qpath(&sample_lock),
        qpath(&sample_tmp),
        max_missing,
        all_imiss_txt,
        qpath(&sample_tmp),
        qpath(&sample_tmp),
        qpath(&sample_keep),
        qpath(&sample_lock),
        qpath(&sample_lock),
        qpath(&sample_keep),
        qpath(&sample_keep),
    );
    let build_site_keep = format!(
        ": > {} && (awk 'NR>1 && $5<={} {{n=split($2,a,\":\"); if (n>=2) print a[1]\"\\t\"a[2]}}' {} > {} || true)",
        qpath(&site_keep),
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
        "NVAR=$({prefix}bcftools index -n {}) && if [ \"${{NVAR:-0}}\" -gt 0 ]; then {build_sample_keep} && {build_site_keep} && if [ -s {} ] && [ -s {} ]; then {filter_cmd} && {index_cmd}; else {copy_empty_cmd}; fi; else {copy_empty_cmd}; fi",
        qpath(&in_imp),
        qpath(&sample_keep),
        qpath(&site_keep),
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
