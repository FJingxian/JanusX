# JanusX CLI Guide

Version baseline: `v1.0.13`  
Entry command: `jx`

## 1. Quick Start

```bash
jx -h
jx -v
jx <module> -h
```

Typical workflow:

```bash
# 1) Build DLC tools (docker/env/conda/singularity route chosen by launcher)
jx -update dlc

# 2) Run analysis
jx gwas -vcf geno.vcf.gz -p pheno.tsv -lmm -o out

# 3) Post-process
jx postgwas -gwasfile out/geno.trait.add.lmm.tsv -manh -qq -o out
```

## 2. Global Flags

```text
-h            Show launcher help
-v            Show version/build information
-update       Update JanusX runtime/toolchain
-upgrade      Upgrade launcher from source, then update core
-list         List available modules or DLC tools
-clean        Clear GWAS history DB, or remove one history row by ID
-uninstall    Remove JanusX runtime and launcher files
```

List modules and DLC routes:

```bash
jx -list module
jx -list dlc
```

## 3. Shared Input Rules

### 3.1 Genotype Inputs

Supported genotype switches in major modules:

```text
-vcf     .vcf / .vcf.gz
-hmp     .hmp / .hmp.gz
-bfile   PLINK prefix with .bed/.bim/.fam
-file    JanusX matrix input (text/npy path or prefix)
```

### 3.2 `-file` Matrix Format

Required files:

`prefix.id` (one sample ID per line)

```text
Sample_001
Sample_002
Sample_003
```

And one genotype matrix file:

`prefix.npy` or `prefix.txt` (also supports `.tsv/.csv` when passed as a direct path)

Example matrix (`variant x sample`):

```text
0 1 2
2 1 0
0 0 1
```

Optional metadata:

`prefix.site` (no header, 4 columns: `CHR POS REF ALT`)

```text
1 10583 A G
1 10611 C T
1 13302 G A
```

Notes:

- `prefix.id` is mandatory for `-file`.
- `prefix.site` is strongly recommended if you will export to `vcf/hmp/plink`.
- For text matrices, missing tokens such as `NA/NULL/./-` are treated as missing values.

### 3.3 Phenotype File

Most model modules expect: first column sample ID, remaining columns traits/covariates.

```text
sample_id trait1 trait2
S1 1.23 5.67
S2 2.34 6.78
S3 3.45 7.89
```

Delimiter is auto-detected (`tab/comma/whitespace`) in modules that support flexible parsing.

### 3.4 Trait Selector `-n`

In `gwas/gs/reml`, `-n` is zero-based over phenotype columns excluding sample ID.

```bash
-n 0
-n 0,2,5
-n 0:5
-n 0-5
-n 0 -n 3
```

### 3.5 GRM/Q/Covariates

GWAS GRM option:

```text
-k 1             Build centered GRM from genotype
-k 2             Build standardized GRM from genotype
-k <path>        Use external GRM file (.txt/.npy)
```

Q matrix in GWAS:

```text
-q 0             Disable PCA covariates
-q N             Use top N PCs from GRM
```

External Q-file mode is removed; use `-c` for external covariates.

Covariate file for `-c` must be ID-first-column format:

```text
S1 0.2 1.0
S2 -0.7 0.5
S3 1.3 -0.2
```

## 4. Module Reference (Detailed)

### 4.1 `grm`

Function:

- Build genomic relationship matrix from genotype.

Usage:

```bash
jx grm -vcf geno.vcf.gz -m 1 -o out -prefix panel
jx grm -hmp geno.hmp.gz -m 2 -npy -o out -prefix panel
jx grm -file geno_prefix -maf 0.01 -geno 0.1
```

Key options:

- `-vcf/-hmp/-bfile/-file`: genotype source.
- `-m`: GRM method (`1` centered, `2` standardized).
- `-maf/-geno`: site filtering.
- `-chunksize`: streaming chunk size.
- `-npy`: save GRM in `.npy` instead of text.

Outputs:

```text
<prefix>.grm.txt or <prefix>.grm.npy
<prefix>.grm.txt.id or <prefix>.grm.npy.id
<prefix>.grm.log
```

### 4.2 `pca`

Function:

- PCA for population structure from genotype or GRM.

Usage:

```bash
jx pca -vcf geno.vcf.gz -dim 3 -plot -o out
jx pca -k out/panel -dim 5
jx pca -q out/panel -plot -plot3D
```

Key options:

- Input mode: `-vcf/-hmp/-bfile/-file` or `-k` (GRM prefix) or `-q` (existing PCA prefix).
- `-dim`: number of PCs.
- `-plot/-plot3D`: 2D PDF and 3D GIF plotting.
- `-group`: optional sample group file for colored PCA.
- `-maf/-geno/-chunksize`: filters and streaming behavior.

Outputs:

```text
<prefix>.eigenvec
<prefix>.eigenval
<prefix>.eigenvec.2D.pdf   (if -plot)
<prefix>.eigenvec.3D.gif   (if -plot3D)
<prefix>.pca.log
```

### 4.3 `gwas`

Function:

- Genome-wide association analysis with streaming LM/LMM/FastLMM and FarmCPU.

Usage:

```bash
jx gwas -vcf geno.vcf.gz -p pheno.tsv -lmm -n 0 -q 3 -k 1 -o out
jx gwas -hmp geno.hmp.gz -p pheno.tsv -lm -model add -n 0,1,2
jx gwas -file geno_prefix -p pheno.tsv -farmcpu -c cov.tsv
```

Key options:

- Models: `-lmm`, `-fastlmm`, `-lm`, `-farmcpu`.
- Genetic coding: `-model {add,dom,rec,het}`.
- Traits: `-n` (zero-based, supports list/range syntax).
- GRM/Q/covariates: `-k`, `-q`, `-c`.
- Filters: `-snps-only`, `-maf`, `-geno`, `-het`.
- Performance: `-chunksize`, `-mmap-limit`, `-t`.

Model details (association engines):

- `-lmm`: full mixed-model scan with per-marker effect test; robust for population structure and relatedness.
- `-fastlmm`: mixed-model scan with null-model lambda fixed once; usually faster than full LMM with very similar ranking in many datasets.
- `-lm`: ordinary linear model scan; fastest baseline, but does not account for kinship unless controlled by covariates.
- `-farmcpu`: iterative fixed/random marker selection strategy; can improve power for complex architectures, but uses full genotype in memory.

Genetic coding details (`-model`) for LM/LMM/FastLMM:

- `add`: additive dosage coding (0/1/2), standard GWAS default.
- `dom`: dominance-oriented coding to emphasize heterozygote effects.
- `rec`: recessive coding to emphasize homozygous-alternative effects.
- `het`: heterozygosity-focused coding (with `-het` site filter).

Outputs:

```text
<prefix>.<trait>.<model>.tsv
<prefix>.gwas.log
```

### 4.4 `postgwas`

Function:

- Post-process GWAS outputs: Manhattan, QQ, LD block, annotation, merged visualization.

Usage:

```bash
jx postgwas -gwasfile out/panel.trait.add.lmm.tsv -manh -qq -o out
jx postgwas -gwasfile out/*.tsv -merge out/a.tsv out/b.tsv -manh 2 -qq
jx postgwas -gwasfile out/a.tsv -bimrange 1:214.4-214.6 -ldblock 2 -bfile geno/panel
```

Key options:

- Required: `-gwasfile` (one or many).
- Column mapping: `-chr`, `-pos`, `-pvalue`.
- Thresholding/range: `-threshold`, `-bimrange`, `-ylim`.
- Plot controls: `-manh`, `-qq`, `-fmt`, `-scatter-size`, `-pallete`, `-full`.
- LD block: `-ldblock` or `-ldblock-all` plus genotype input (`-bfile/-vcf/-file`).
- Annotation: `-a`, `-ab`, `-descItem`, `-hl`.
- Merge mode: `-merge`.

Outputs:

- Manhattan/QQ/LD figures and annotation tables in output directory.
- Run log: `<prefix>.postGWAS.log`.

### 4.5 `gs`

Function:

- Genomic selection with BLUP/Bayesian/ML models.

Usage:

```bash
jx gs -vcf geno.vcf.gz -p pheno.tsv -GBLUP -cv 5 -n 0 -o out
jx gs -file geno_prefix -p pheno.tsv -adBLUP -BayesA -BayesB -BayesCpi -cv 5
jx gs -hmp geno.hmp.gz -p pheno.tsv -RF -ET -GBDT -XGB -SVM -ENET -cv 5
```

Key options:

- Models: `-GBLUP`, `-adBLUP`, `-rrBLUP`, `-BayesA`, `-BayesB`, `-BayesCpi`, `-RF`, `-ET`, `-GBDT`, `-XGB`, `-SVM`, `-ENET`.
- Traits: `-n` (zero-based list/range).
- CV and tuning: `-cv`, `-strict-cv`.
- Filters and preprocessing: `-maf`, `-geno`, `-pcd`.
- Parallelism: `-t` (ML methods thread count).

Model details (prediction engines):

- `-GBLUP`: genomic BLUP with additive relationship matrix; strong default for many breeding datasets.
- `-adBLUP`: additive + dominance BLUP (pyBLUP/JAX backend) to capture non-additive signal.
- `-rrBLUP`: marker-effect ridge regression BLUP; typically similar behavior to GBLUP with marker-effect parameterization.
- `-BayesA`: Bayesian marker model with marker-specific shrinkage variance; flexible but slower.
- `-BayesB`: Bayesian sparse marker model (subset of markers with larger effects); useful when few major loci are expected.
- `-BayesCpi`: Bayesian mixture model with estimated zero-effect proportion (`pi`); balances sparsity and polygenic background.
- `-RF`: Random Forest regressor; non-linear and interaction-capable, usually robust baseline ML model.
- `-ET`: Extra Trees regressor; similar to RF with higher randomization, often faster and less variance in tuning.
- `-GBDT`: Gradient boosting trees (hist-based); strong non-linear learner but can be the slowest tree model to tune.
- `-XGB`: XGBoost regressor; efficient boosted trees with strong performance on medium/large marker sets.
- `-SVM`: RBF-SVR; can perform well on moderate sample sizes but tuning cost increases quickly with dataset size.
- `-ENET`: ElasticNet linear model; good high-dimensional baseline with joint L1/L2 shrinkage and good interpretability.

Outputs:

```text
<prefix>.<trait>.gs.tsv
<prefix>.gs.log
<prefix>.<trait>.gs.<method>.svg   (model plots, when enabled by workflow)
```

### 4.6 `reml`

Function:

- REML-BLUP mixed model for variance components, fixed/random effects, and BLUP per trait.

Usage:

```bash
jx reml -file test.reml.txt -n 3 -rh 0,1,2 -f covariate -o out -prefix reml
jx reml -file test.reml.txt -n trait1 -n trait2 -rh sample -rh env -fh sex
```

Key options:

- Required: `-file`, `-n`.
- Random categorical one-hot: `-rh`.
- Fixed categorical one-hot: `-fh`.
- Random effect columns: `-r`.
- Fixed effect columns: `-f`.
- Iteration control: `-maxiter`.

Outputs:

```text
<prefix>.blup.txt
<prefix>.reml.summary.tsv
<prefix>.reml.log
```

### 4.7 `garfield`

Function:

- Random-forest based marker-trait association and pseudo-marker generation.

Usage:

```bash
jx garfield -vcf geno.vcf.gz -p pheno.tsv -n 0 -o out
jx garfield -file geno_prefix -p pheno.tsv -g geneset.txt -gff gene.gff3 -forceset
```

Key options:

- Inputs: `-vcf/-hmp/-bfile/-file`, `-p`.
- Trait selector: `-n`.
- Gene-set/annotation: `-g`, `-gff`, `-forceset`.
- Model control: `-vartype`, `-step`, `-ext`, `-nsnp`, `-nestimators`.
- Performance: `-t`, `-mmap-limit`.

Outputs:

- GARFIELD result tables/pseudo-genotype artifacts in output directory.
- Log file under selected prefix.

### 4.8 `postgarfield`

Function:

- Run GWAS + postgwas pipeline on GARFIELD pseudo-genotype outputs.

Usage:

```bash
jx postgarfield -bfile demo.garfield -p pheno.tsv -k kinship.npy -n 0 -o out
jx postgarfield -vcf demo.pseudo.vcf.gz -p pheno.tsv -k kinship.npy -anno genes.gff3
```

Key options:

- Inputs: `-bfile` or `-vcf`, plus `-p`, `-k`.
- Optional covariates: `-q`, `-cov`.
- GWAS filters: `-maf`, `-geno`, `-chunksize`, `-t`.
- Postgwas controls: `-threshold`, `-bimrange`, `-fmt`, `-hl`, `-a`, `-ab`, `-pallete`, `-noplot`.

Outputs:

- Intermediate GWAS results and postgwas figures/tables.
- Log and summary files under the chosen prefix.

### 4.9 `postbsa`

Function:

- BSA post-processing with filtering, smoothing, thresholds, and chromosome-level/global plots.

Usage:

```bash
jx postbsa -file bsa.tsv -b1 Bulk1 -b2 Bulk2 -o out
jx postbsa -file '4.merge/Merge.*.SNP.tsv' -b1 Bulk1 -b2 Bulk2 -t 8 -fmt pdf
```

Key options:

- Required: `-file`, `-b1`, `-b2`.
- Filtering: `-minDP`, `-minGQ`, `-totalDP`, `-refAlleleFreq`, `-depthDifference`.
- Smoothing/window: `-window`, `-step`, `-ed`.
- Plot controls: `-ratio`, `-fmt`.
- Parallelism: `-t` (parallel chromosome jobs in glob mode).

Outputs:

- Raw/smoothed BSA tables, threshold region tables, and figures.
- Run log in output directory.

### 4.10 `fastq2vcf` (Rust pipeline)

Function:

- End-to-end variant pipeline from FASTQ to imputed merged VCF.

Usage:

```bash
jx fastq2vcf -r ref.fa -i raw_fastq_dir -w workdir
jx fastq2vcf -r ref.fa -i 3.gvcf -w workdir -from-step 4 -to-step 6
```

Core options:

- Required: `-r`, `-i`, `-w`.
- Resume range: `-from-step`, `-to-step`.

Pipeline steps:

```text
1. fastp
2. bwamem+markdup
3. bam2gvcf
4. gvcf2gtprep
5. impute_filter
6. mergevcf
```

Notes:

- `-from-step` supports 1..5.
- Step 4 enforces strict gVCF integrity checks (sample x chromosome completeness, readable non-empty `.g.vcf.gz` + `.tbi`).

### 4.11 `fastq2count` (Rust pipeline)

Function:

- RNA-seq pipeline from FASTQ to count/FPKM/TPM.

Usage:

```bash
jx fastq2count -r ref.fa -a anno.gtf -i fastq_dir -w workdir
jx fastq2count -r ref.fa -a anno.gff3 -i work/3.mapping -w work -from-step 3 -to-step 4
```

Core options:

- Required: `-r`, `-a`, `-i`, `-w`.
- Resume: `-from-step`, `-to-step`.
- RNA settings: `-strandness`, `-feature-type`, `-gene-attr`.
- Threads: `-t`.

Pipeline steps:

```text
1. fastp
2. hisat2-index
3. hisat2-align
4. featurecounts + FPKM/TPM
```

Execution note:

- Runtime UI may display a merged logical flow (`fastp+hisat2-index` pre-processing), but the recoverable internal step points are step-based.

### 4.12 `gformat`

Function:

- Convert genotype data across `plink/vcf/hmp/txt/npy` and perform sample/site extraction.

Usage:

```bash
jx gformat -vcf geno.vcf.gz -fmt npy -o out -prefix panel
jx gformat -file geno_prefix -fmt vcf --keep keep_samples.txt
jx gformat -bfile geno_prefix -fmt txt --chr 1-5,8 --from-bp 100000 --to-bp 500000
```

Key options:

- Inputs: `-vcf/-hmp/-bfile/-file`.
- Output format: `-fmt {plink,vcf,hmp,txt,npy}`.
- Subsetting: `--keep`, `--extract`, `--chr`, `--from-bp`, `--to-bp`.
- Output destination: `-o`, `-prefix`.

Outputs:

- Converted genotype files in chosen format.
- Log: `<prefix>.gformat.log`.

### 4.13 `gmerge`

Function:

- Merge multiple genotype datasets from mixed input types.

Usage:

```bash
jx gmerge -vcf a.vcf.gz b.vcf.gz -fmt vcf -o out -prefix merged
jx gmerge -bfile A B -fmt plink -sample-prefix
jx gmerge -vcf a.vcf.gz -file matrix_prefix -fmt npy
```

Key options:

- Inputs are repeatable across `-vcf`, `-bfile`, `-file`.
- Need at least 2 total inputs.
- Output format: `-fmt {plink,vcf,txt,npy}`.
- ID handling: `-sample-prefix` to add `D1_`, `D2_`...
- Post-merge filters: `-maf`, `-geno`.
- Output path: `-o`, `-prefix`.

Outputs:

- Merged genotype files in selected format.
- Log: `<prefix>.merge.log`.

### 4.14 `hybrid`

Function:

- Build pairwise hybrid genotype matrix from two parent lists.

Usage:

```bash
jx hybrid -vcf parents.vcf.gz -p1 p1.txt -p2 p2.txt -fmt npy -o out -prefix hybrid
jx hybrid -file geno_prefix -p1 male.txt -p2 female.txt -fmt vcf
```

Key options:

- Inputs: `-vcf/-bfile/-file`.
- Parent lists: `-p1`, `-p2` (one ID per line).
- Output format: `-fmt {plink,vcf,txt,npy}`.
- Output path: `-o`, `-prefix`.
- Streaming chunk size: `-chunksize`.

Outputs:

- Hybrid genotype files in selected format.
- Log: `<prefix>.hybrid.log`.

### 4.15 `webui`

Function:

- Launch interactive web interface for postgwas/postbsa workflows.

Usage:

```bash
jx webui
jx webui --host 0.0.0.0 --port 8765
jx webui --root ~/.janusx/webui --no-browser
```

Key options:

- `--host`, `--port`: bind endpoint.
- `--root`: runtime state directory.
- `--no-browser`: disable auto-open browser.

Behavior:

- Performs startup cleanup of invalid GWAS history records.
- Serves UI locally and keeps running until interrupted.

### 4.16 `sim`

Function:

- Quick synthetic genotype/phenotype generator (lightweight script).

Usage:

```bash
jx sim 10 500 test/demo
```

Semantics:

- First argument is SNP count in thousands (`10` means ~10,000 SNPs).
- Second argument is number of individuals.
- Third argument is output prefix.

Outputs:

- Simulated genotype matrix files and phenotype examples around output prefix.

### 4.17 `simulation`

Function:

- Extended simulation using existing genotype input; generates simulated phenotype under chosen architecture.

Usage:

```bash
jx simulation -vcf geno.vcf.gz -sm single -pve 0.5 -ve 1.0 -o out
jx simulation -file geno_prefix -sm garfield -windows 50000 -write-sites
```

Key options:

- Inputs: `-vcf/-hmp/-bfile/-file`.
- Filters: `-maf`, `-geno`, `-chunksize`.
- Effect architecture: `-sm {single,garfield}`.
- Variance setup: `-pve`, `-ve`.
- Reproducibility: `--seed`.
- Optional: `-write-sites` to output causal sites.

Outputs:

- Simulated phenotype files and optional causal site files.
- Log: `<prefix>.sim.log`.

## 5. Pipeline-Oriented Examples

### 5.1 FastQ to BSA

```bash
jx -update dlc
jx fastq2vcf -r ref.fa -i fastq/ -w run_fastq2vcf
jx postbsa -file 'run_fastq2vcf/4.merge/Merge.*.SNP.tsv' -b1 Bulk1 -b2 Bulk2 -t 8
```

### 5.2 GWAS to WebUI

```bash
jx gwas -vcf geno.vcf.gz -p pheno.tsv -lmm -q 3 -n 0 -o out
jx postgwas -gwasfile out/geno.trait.add.lmm.tsv -manh -qq -o out
jx webui
```

### 5.3 Genotype Utility Chain

```bash
jx gformat -vcf raw.vcf.gz -fmt npy -o work -prefix panel
jx gmerge -file work/panel -vcf extra.vcf.gz -fmt vcf -o work -prefix merged
jx hybrid -file work/merged -p1 p1.list -p2 p2.list -fmt npy -o work -prefix hybrid
```

## 6. Practical Notes

- Use `jx <module> -h` for authoritative module help when a flag changes.
- Keep `prefix.id` synchronized with genotype matrix columns for `-file` mode.
- For reproducibility, keep logs (`*.log`) and exact command lines in your run records.
