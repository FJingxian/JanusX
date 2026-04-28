# JanusX CLI Guide

Version baseline: `v1.0.21`

## 1. Scope and entrypoints

This document describes the current CLI behavior from source code in this repository.

Entrypoints:

- `jx` (installer/launcher build): Rust launcher binary
- `jx` / `jxpy` (pip/source install): Python dispatcher (`python -m janusx.script.JanusX`)

Important boundary:

- Launcher flags `-update/-upgrade/-list/-clean/-uninstall` are launcher-binary-only.
- In Python dispatcher mode (`jxpy`, and `jx` from pip/source install), `fastq2vcf/fastq2count` run preflight checks only and do not execute the full pipeline.
- Extra modules `kmerge/view/treeplot/gblupbench` are available in Python dispatcher mode but not in launcher module whitelist.
- `beam` script exists in repository but is not wired into launcher or Python dispatcher; run it directly via `python -m janusx.script.beam ...`.

Quick checks:

```bash
jx -h
jx -v
jx -list module
jx <module> -h
```

## 2. Launcher global flags (`jx`)

### `-h`, `--help`

Show launcher help and module overview.

### `-v`, `--version`

Show launcher/core version, build time, and mismatch hints.

### `-list [module|dlc]`

List modules or DLC tool routes.

Examples:

```bash
jx -list module
jx -list dlc
```

### `-update [dlc|latest|<local_path>] [-e|-editable] [-verbose]`

Update behavior:

- `jx -update` -> update Python core from PyPI
- `jx -update latest` -> update core from GitHub latest source/release route
- `jx -update <local_path>` -> update core from local source path
- `jx -update -e <local_path>` -> editable local core update
- `jx -update dlc` -> update external pipeline toolchain runtime

### `-upgrade [latest|<local_path>] [-verbose]`

Upgrade launcher binary first, then run core update flow:

- `jx -upgrade` behaves as launcher upgrade + core `latest` update
- `jx -upgrade latest`
- `jx -upgrade <local_path>`

### `-clean [history_id]`

- `jx -clean` clears GWAS history database in runtime
- `jx -clean <history_id>` removes a specific history record path

### `-uninstall [-yes]`

Remove launcher/runtime managed files.

- Interactive confirm by default
- Use `-yes` to skip prompt

## 3. Shared input conventions

### 3.1 Genotype input switches

Most model modules support one of:

- `-vcf` / `--vcf`: `.vcf` / `.vcf.gz`
- `-hmp` / `--hmp`: `.hmp` / `.hmp.gz`
- `-bfile` / `--bfile`: PLINK prefix (`.bed/.bim/.fam`)
- `-file` / `--file`: matrix path or prefix (`.txt/.tsv/.csv/.npy`)

### 3.2 `-file` sidecar files

Required sidecar:

- `prefix.txt` (numeric matrix)

```text
1	1	2
1	1	0
1	0	0
```

- `prefix.id` (sample IDs in matrix column order)

```text
idv1
idv2
idv3
```

Optional sidecar:

- `prefix.site` or `prefix.bim` (recommended for format conversion, LD, export)

```text
1	3197400	G	A
1	3407393	A	G
1	3492195	G	A
```

### 3.3 Matrix orientation

JanusX core genotype matrix APIs are SNP-major:

- rows: variants
- columns: samples

### 3.4 Phenotype file

Expected by major modules:

- first column: sample ID
- remaining columns: traits/covariates

Delimiter is auto-detected in many modules.

```text
	test0	test1	test2	test3
x	0.224	0.224	NA	1.0
1	-0.974	-0.974	NA	0.0
2	0.195	0.195	NA	1.0
```

### 3.5 Trait selector `-n`

In `gwas/gs/reml/postgarfield/garfield`, `-n` uses zero-based phenotype column index excluding the sample ID column.

Accepted forms include:

- single index: `-n 0`
- comma list: `-n 0,2,5`
- range: `-n 0:5` or `-n 0-5`
- repeat flag: `-n 0 -n 3`

## 4. Module reference

### 4.1 `grm`

Build genomic relationship matrix from genotype.

Examples:

```bash
jx grm -vcf geno.vcf.gz -m 1 -o out -prefix panel
jx grm -hmp geno.hmp.gz -m 2 -npy -o out
jx grm -file matrix_prefix -maf 0.01 -geno 0.1
```

Key options:

- `-m/--method`: `1` centered, `2` standardized
- `-maf`, `-geno`
- `-chunksize`
- `-mmap-limit` (BED mmap window mode)
- `-npy` (write `.npy` GRM)

Outputs:

- `<prefix>.grm.txt` or `<prefix>.grm.npy`
- matching `.id`
- `<prefix>.grm.log`

### 4.2 `pca`

PCA from genotype, GRM, or existing PCA results.

Examples:

```bash
jx pca -vcf geno.vcf.gz -dim 3 -plot
jx pca -k out/panel -dim 5
jx pca -c out/panel -plot -plot3D
jx pca -vcf geno.vcf.gz -rsvd 3 0.1
```

Key options:

- input mode: genotype or `-k/--grm` prefix or `-c/--cov` (existing eigen files)
- `-dim`
- `-plot`, `-plot3D`
- `-group`, `-color`
- `-rsvd [power] [tol]` (streaming Rust RSVD)
- `-maf`, `-geno`, `-chunksize`, `-mmap-limit`

Outputs:

- `<prefix>.eigenvec`
- `<prefix>.eigenval`
- optional plot files
- `<prefix>.pca.log`

### 4.3 `gwas`

Genome-wide association with streaming and mixed-model kernels.

Examples:

```bash
jx gwas -vcf geno.vcf.gz -p pheno.tsv -lmm -n 0 -q 3 -k 1 -o out
jx gwas -bfile geno -p pheno.tsv -lm -model add -n 0,1
jx gwas -file matrix_prefix -p pheno.tsv -farmcpu -c cov.tsv
jx gwas -bfile geno -p pheno.tsv -lrlmm 128
```

Models:

- `-lm`
- `-lmm`
- `-fastlmm`
- `-lrlmm [RANK]`
- `-farmcpu`

Coding mode:

- `-model {add,dom,rec,het}`

Other key options:

- `-k`: `1`, `2`, or external GRM path
- `-q`: PC count (integer)
- `-c`: repeatable covariate input

`-c` supports:

- covariate file (ID in first column)
- single-site token (`chr:pos` or `chr:start:end` where start=end)

```text
x	0.017407173	-0.0069945143
1	0.004649966	0.039312065
2	0.012239488	-0.02688213
3	0.010048621	-0.0106722545
4	0.017356979	-0.0052547506
5	0.012380351	-0.012446985
```

More options:

- `-snps-only`
- `-maf`, `-geno`, `-het`
- `-chunksize`, `-mmap-limit`
- `-t`

Outputs:

- `<prefix>.<trait>.<model>.tsv`
- `<prefix>.gwas.log`

### 4.4 `postgwas`

Post-process GWAS results: Manhattan, QQ, LD block, annotation, merge plots.

Examples:

```bash
jx postgwas -gwasfile out/a.tsv -manh -qq -o out
jx postgwas -gwasfile out/*.tsv -merge out/a.tsv out/b.tsv -manh
jx postgwas -gwasfile out/a.tsv -bimrange 1:214.4-214.6 -ldblock 2 -bfile geno
```

Key options:

- required: `-gwasfile`
- column mapping: `-chr`, `-pos`, `-pvalue`
- threshold: `-thr` (default `0.05/nSNP`)
- merge mode: `-merge` (repeatable)
- region filter: `-bimrange` (repeatable)
- Manhattan: `-manh [ratio]`
- QQ: `-qq`
- LD: `-ldblock` or `-ldblock-all`
- annotation: `-a`, `-ab`
- LD clump for annotation: `-LDclump <window> <r2>`
- style: `-pallete`, `-scatter-size`, `-full`, `-ylim`, `-fmt`
- genotype source for LD/clump: `-bfile/-vcf/-file`

Outputs:

- plots and annotation tables under output directory
- log: `<prefix>.postGWAS.log`

### 4.5 `gs`

Genomic selection and model comparison.

Examples:

```bash
jx gs -vcf geno.vcf.gz -p pheno.tsv -GBLUP -cv 5 -n 0 -o out
jx gs -file matrix_prefix -p pheno.tsv -adBLUP -BayesA -BayesB -BayesCpi -cv 5
jx gs -bfile geno -p pheno.tsv -RF -ET -GBDT -XGB -SVM -ENET -cv 5
jx gs -bfile geno -p pheno.tsv -rrBLUP -cv 5 -n 0
jx gs -bfile geno -p pheno.tsv -rrBLUP --rrblup-solver exact    # advanced hidden option
```

Model flags:

- `-GBLUP`, `-adBLUP`, `-rrBLUP`
- `-BayesA`, `-BayesB`, `-BayesCpi`
- `-RF`, `-ET`, `-GBDT`, `-XGB`, `-SVM`, `-ENET`

Other options:

- `-n`
- `-cv`
- `-strict-cv`
- `-pcd`
- `-maf`, `-geno`
- `-t`

Notes:

- `rrBLUP` defaults to mini-batch AdamW backend.
- rrBLUP backend/tuning switches are advanced flags and hidden from `jx gs -h`, but still accepted by CLI (for example `--rrblup-solver exact`).
- AdamW rrBLUP now runs a small automatic lambda grid (default 2 candidates) with fold-internal validation + early-stop in tuning trials; in CV mode the selected hyperparameters are reused across folds by default to reduce repeated tuning overhead (hidden advanced switch: `--rrblup-grid-reuse-cv off`).
- AdamW rrBLUP reports `h2/PVE` in lambda-consistent mode by default (`--rrblup-pve-mode lambda`), and can switch to train-variance mode via hidden advanced flag `--rrblup-pve-mode trainvar`.
- For `GBLUP/rrBLUP`-only runs with non-PLINK input (`-vcf/-hmp/-file`), `jx gs` will auto-probe data scale and switch to packed BED flow when `n_samples * n_snps` exceeds the hidden threshold `--packed-lmm-auto-min-cells` (default `200000000`).

Outputs:

- `<prefix>.<trait>.gs.tsv`
- `<prefix>.gs.log`

### 4.6 `reml`

REML-BLUP mixed model for variance components and BLUP outputs.

Examples:

```bash
# example/rice6048.reml.tsv
jx reml -file test.reml.txt -n 3 -rh 0 -rh 1 -rh 2 -o out -prefix reml
jx reml -file test.reml.txt -n trait1 -f sex -r family
```

Key options:

- required: `-file`, `-n`
- random categorical (one-hot): `-rh`
- fixed categorical (one-hot): `-fh`
- random terms: `-r`
- fixed terms: `-f`
- `-maxiter`

Outputs:

- `<prefix>.blup.txt`
- `<prefix>.reml.summary.tsv`
- `<prefix>.reml.log`

### 4.7 `garfield`

Random-forest based marker-trait association.

Examples:

```bash
jx garfield -vcf geno.vcf.gz -p pheno.tsv -n 0 -o out
jx garfield -file matrix_prefix -p pheno.tsv -g geneset.txt -gff gene.gff3 -forceset
```

Key options:

- input: `-vcf/-hmp/-bfile/-file`
- `-p`, `-n`
- optional gene mode: `-g`, `-gff`, `-forceset`
- model control: `-step`, `-ext`, `-nsnp`, `-nestimators`
- runtime: `-t`, `-mmap-limit`

Outputs:

- GARFIELD result tables/artifacts in output dir
- `<prefix>.garfield.log`

### 4.8 `postgarfield`

Run GWAS + postgwas flow on GARFIELD pseudo-genotype outputs.

Examples:

```bash
jx postgarfield -bfile demo.garfield -p pheno.tsv -k kinship.npy -n 0 -o out
jx postgarfield -vcf demo.pseudo.vcf.gz -p pheno.tsv -k kinship.npy -anno genes.gff3
```

Required:

- genotype: `-bfile` or `-vcf`
- `-p` phenotype
- `-k` GRM

Other options:

- `-q`, `-cov`
- `-n`
- `-maf`, `-geno`, `-chunksize`, `-t`
- `--pseudo`, `--pseudochrom`, `--only-set [BP]`
- postgwas controls: `-thr`, `-bimrange`, `-fmt`, `-noplot`, `-a`, `-ab`, `-pallete`

Outputs:

- GWAS/postgwas outputs and log (`<prefix>.postGARFIELD.log`)

### 4.9 `postbsa`

BSA post-processing with filtering, smoothing, and plotting.

Examples:

```bash
jx postbsa -file bsa.tsv -b1 Bulk1 -b2 Bulk2
jx postbsa -file '4.merge/Merge.*.SNP.tsv' -b1 Bulk1 -b2 Bulk2 -t 8
```

Key options:

- required: `-file`, `-b1`, `-b2`
- sliding window: `-window`, `-step`
- filtering: `-minDP`, `-minGQ`, `-totalDP`, `-refAlleleFreq`, `-depthDifference`
- ED control: `-ed`
- plots: `-ratio`, `-fmt`
- runtime: `-t`

Outputs:

- processed tables and figures
- `<prefix>.postbsa.log`

### 4.10 `adamixture`

ADAMIXTURE ancestry inference.

Examples:

```bash
jx adamixture -bfile geno -k 8 -o out -prefix cohort
jx adamixture -vcf geno.vcf.gz -k 2..10 -t 16
jx adamixture -vcf geno.vcf.gz -k 2..10..3 -t 16
jx adamixture -vcf geno.vcf.gz -k 2,5,8 -t 16
jx adamixture -vcf geno.vcf.gz -k 2..10 -cv
jx adamixture -vcf geno.vcf.gz -k 2..10 -cv 5
jx adamixture -vcf geno.vcf.gz -k 6 -tag sample1,sample2
jx adamixture -vcf geno.vcf.gz -k 6 --tag tag_samples.txt
```

Key options:

- required input: one of `-bfile/-vcf/-hmp/-file`
- required clusters: `-k` (single `8`, range `2..10` / `2:10`, stepped range `2..10..3` / `2:10:3` / `2..10:3`, list `2,5,8`)
- input filter: `-chunksize`, `-snps-only`, `-maf`, `-geno`
- structure axis labels: `-tag/--tag` (file with one sample per line, or `sample1,sample2`)
- runtime: `-t`, `-seed`
- optimization: `-solver {auto,adam,adam-em}`, `-max-iter`, `-tol`
- CVerror:
`-cv` (no value defaults to 5-fold),
`-cv N` (N-fold, N>=2).
Omit `-cv` to disable CVerror.

Outputs:

- `<prefix>.<k>.Q.txt`
- `<prefix>.<k>.P.npy`
- `<prefix>.<k>.P.site`
- `<prefix>.<k>.admix.svg`
- `<prefix>.<k>.adamix.log`
- `<prefix>.adamixture.cverror.summary.tsv` (when `-cv` enabled)
- `<prefix>.adamixture.summary.log`

### 4.11 `gformat`

Convert genotype formats and apply basic sample/site filters.

Examples:

```bash
jx gformat -vcf geno.vcf.gz -fmt npy -o out -prefix panel
jx gformat -file matrix_prefix -fmt vcf --keep keep_samples.txt
jx gformat -bfile geno -fmt txt --chr 1-5,8 --from-bp 100000 --to-bp 500000
jx gformat -bfile geno -fmt npy --prune 1 5 0.2
jx gformat -bfile geno -fmt npy --prune 0.1 5 0.2
jx gformat -bfile geno -fmt npy --prune 500kb 10 0.2
```

Key options:

- input: `-vcf/-hmp/-bfile/-file`
- output: `-fmt {plink,vcf,hmp,txt,npy}`
- `-keep`
- `-extract` (`--extract <file>` or `--extract range <file>`)
- chromosome/range filters: `--chr`, `--from-bp`, `--to-bp`
- site filters: `-maf`, `-geno`
- LD prune: `-prune/--prune <window size[kb|bp]> <step size (variant ct)> <r^2 threshold>`  
  Numeric window defaults to `kb` (`1` = `1kb`, `0.1` = `100bp`)

Outputs:

- converted genotype files
- `<prefix>.gformat.log`

### 4.12 `gmerge`

Merge multiple genotype datasets.

Examples:

```bash
jx gmerge -vcf a.vcf.gz b.vcf.gz -fmt vcf -o out -prefix merged
jx gmerge -bfile A B -fmt plink -sample-prefix
jx gmerge -vcf a.vcf.gz -file matrix_prefix -fmt npy
```

Key options:

- repeatable inputs: `-vcf`, `-bfile`, `-file`
- at least two total inputs required
- output format: `-fmt {plink,vcf,txt,npy}`
- sample renaming: `-sample-prefix`
- merged site filters: `-maf`, `-geno`

Outputs:

- merged genotype files
- `<prefix>.merge.log`

### 4.13 `hybrid`

Build pairwise hybrid genotype matrix from parent lists.

Examples:

```bash
jx hybrid -vcf parents.vcf.gz -p1 p1.txt -p2 p2.txt -fmt npy -o out -prefix hybrid
jx hybrid -bfile geno -p1 male.txt -p2 female.txt -fmt vcf
```

Key options:

- input: `-vcf/-bfile/-file`
- required parent lists: `-p1`, `-p2`
- output: `-fmt {plink,vcf,txt,npy}`
- `-chunksize`

Outputs:

- hybrid genotype outputs
- `<prefix>.hybrid.log`

### 4.14 `webui`

Start local JanusX Web UI service.

Examples:

```bash
jx webui
jx webui --host 0.0.0.0 --port 8765
jx webui --root ~/.janusx/webui --no-browser
```

Key options:

- `--host`
- `--port`
- `--root`
- `--no-browser`

Behavior notes:

- startup cleanup for invalid GWAS history entries
- current implementation is GWAS-history-first rendering flow

### 4.15 `sim`

Quick synthetic generator (supports unrelated/family/mixed sample structures).

Usage:

```bash
jx sim <nsnp_k> <n_individuals> <outprefix>
jx sim <nsnp_k> <n_individuals> <outprefix> -structure mixed -chunksize 10000
```

Example:

```bash
jx sim 10 500 demo/test
jx sim 250 2500 demo/sim_250k_2500 -structure mixed -chunksize 10000 -seed 1
```

Semantics:

- `nsnp_k`: SNP count in thousands (`10` -> about 10,000 SNPs)
- second arg: number of individuals
- third arg: output prefix
- optional structure controls:
  - `-structure {unrelated,family,mixed}` (default: `unrelated`)
  - family layout is auto-configured internally:
  - for `mixed`, target family fraction is fixed to `0.8`
  - family block size is fixed to `4` (2 parents + 2 offspring)
  - if family samples exist, `<prefix>.family.tsv` is written automatically
- optional streaming controls:
  - `-chunksize` (if unset, auto-tuned from sample size)
- MAF controls:
  - `-maf-high` (upper bound; lower bound is fixed to `0.02`)
- phenotype controls:
  - `-pve`, `-ve`, `-trait-name`, `-na-rate`

### 4.16 `simulation`

Extended simulation from existing genotype input.

Examples:

```bash
jx simulation -vcf geno.vcf.gz -sm single -pve 0.5 -ve 1.0 -o out
jx simulation -file matrix_prefix -sm garfield -windows 50000 -write-sites
```

Key options:

- input: `-vcf/-hmp/-bfile/-file`
- filter: `-maf`, `-geno`, `-chunksize`
- variance: `-pve`, `-ve`
- causal mode: `-sm {single,garfield}`
- `-windows`
- `-write-sites`
- `--seed`

Outputs:

- simulated phenotype files (`.pheno*`)
- optional causal site file (`.causal.sites.tsv`)
- `<prefix>.sim.log`

### 4.17 `fastq2vcf` (launcher-only)

End-to-end variant pipeline from FASTQ to merged imputed VCF.

Usage:

```bash
jx fastq2vcf -r ref.fa -i input_dir -w workdir
jx fastq2vcf -r ref.fa -i 3.gvcf -w workdir -from-step 4 -to-step 6
```

Step range:

- `-from-step` valid: `1..5`
- `-to-step` valid: `1..6`

Pipeline steps:

1. `fastp`
2. `bwamem+markdup`
3. `bam2gvcf`
4. `gvcf2gtprep`
5. `impute_filter`
6. `mergevcf`

Notes:

- step-4 resume performs strict gVCF integrity checks
- step-6 cannot be a start step
- typically requires DLC runtime setup (`jx -update dlc`)

### 4.18 `fastq2count` (launcher-only)

RNA-seq pipeline from FASTQ to count/FPKM/TPM.

Usage:

```bash
jx fastq2count -r ref.fa -a anno.gtf -i fastq_dir -w workdir
jx fastq2count -r ref.fa -a anno.gff3 -i work/3.mapping -w work -from-step 3 -to-step 4
```

Options:

- required: `-r`, `-a`, `-i`, `-w`
- resume: `-from-step`, `-to-step`
- RNA control: `-strandness`, `-feature-type`, `-gene-attr`
- runtime: `-t`, `--threads`

Pipeline steps:

1. `fastp`
2. `hisat2-index`
3. `hisat2-align`
4. `featurecounts + FPKM/TPM`

Execution note:

- UI may display merged preprocess (`fastp+hisat2-index`) while internal checkpoint steps remain 1..4.

### 4.19 `kmer`

K-mer workflow entry with two modes: KMC counting and WASTER tree inference.

Examples:

```bash
jx kmer -fa sample.fastq.gz --count -o out -prefix sample_k19 -k 19 -t 8
jx kmer -fa species1.fa species2.fa --tree -o out -prefix species_tree -t 8 --waster-mode 4
```

Key options:

- required input: `-fa` (one or more FASTQ/FASTA files)
- mode: `--count` or `--tree` (default falls back to count mode)
- shared: `-k`, `-t`, `-o`, `-prefix`, `-limit-mem`
- count mode: `-ci`, `-cx`, `--counter-max`, `--tmp-dir`
- tree mode: `--waster-mode`, `--waster-sampled`, `--waster-qcs`, `--waster-qcn`, `--waster-pattern`, `--waster-consensus`, `--waster-continue-file`

Outputs:

- count mode: `<prefix>.kmc_pre`, `<prefix>.kmc_suf`
- tree mode: `<prefix>.waster.nw`, plus optional `.waster.snps.fa` / `.waster.patterns`

### 4.20 `tree`

Tree workflow (current CLI-exposed path is NJ mode).

Examples:

```bash
jx tree -vcf cohort.vcf.gz -o out --prefix cohort_tree -nj
jx tree -fa aln.fasta -o out --prefix aln_tree -nj bionj
jx tree -bfile panel -o out --prefix panel_tree --write-phylip -nj
```

Key options:

- input: `-vcf/-hmp/-file/-bfile/-fa`
- model: `-nj [exact|bionj|bionj-dist|bionj-jc|bionj-binom|bionj-auto|approx]`
- filters/runtime: `-maf`, `-geno`, `-het`, `-snps-only`, `-chunksize`, `-t`
- output controls: `--write-phylip`, `--profile`, `-o`, `-prefix`

Outputs:

- `<prefix>.nwk`
- `<prefix>.fasta`
- optional `<prefix>.phy`
- optional `<prefix>.profile.tsv`
- log `<prefix>.tree.log`

### 4.21 `benchmark`

FarmCPU benchmark workflow across JanusX / GAPIT / rMVP.

Examples:

```bash
jx benchmark -bfile example_prefix -p pheno.tsv -n 0
jx benchmark -vcf example.vcf.gz -p pheno.tsv -n 0 --kernels janusx,gapit,rmvp
```

Key options:

- input: `-vcf/-hmp/-file/-bfile`, phenotype `-p`, trait selector `-n`
- kernels: `--kernels janusx,gapit,rmvp`
- filters/runtime: `-maf`, `-geno`, `-chunksize`, `-t`, `-q`, `-c`
- analysis controls: `--topk`, `--check`
- advanced (dev help): FarmCPU grid/threshold tuning flags with `-h -dev`

Outputs:

- benchmark workspace: `<prefix>.farmcpu_bench/`
- summary tables: `summary/<prefix>.benchmark.tsv`, `.md`

### 4.22 `kmerge` (Python dispatcher mode)

Merge multiple KMC databases to binary k-mer matrix outputs.

Examples:

```bash
jxpy kmerge -db s1 s2 s3 -o out -prefix merged_bin
jxpy kmerge -db sampleA.kmc_pre sampleB -o out -prefix panel
```

Key options:

- required: `-db` (KMC prefixes or `.kmc_pre/.kmc_suf` paths)
- optional: `-sid`, `-t`, `--max-kmers`, `-kmerf`
- safety controls: `--max-output-sites`, `--allow-large-output`

Outputs:

- `<prefix>.bin`
- `<prefix>.bin.id`
- `<prefix>.bin.site`

### 4.23 `view` (Python dispatcher mode)

Decode JanusX binary files for shell pipelines.

Examples:

```bash
jxpy view -bin test/kmerge_new.bin | head
jxpy view -bin test/kmerge_new.bin.site | head
```

Key options:

- `-bin` (one or more `.bin` / `.bin.site` files)

Behavior:

- `.bin` -> rows of `0/1` strings
- `.bin.site` -> decoded DNA k-mer rows

### 4.24 `treeplot` (Python dispatcher mode)

Visualize Newick/GRM trees with toytree.

Examples:

```bash
jxpy treeplot -nwk result.tree.nwk -o out -prefix fig1 --layout c --showlabels -fmt svg
jxpy treeplot -k panel.grm.npy -kid panel.grm.npy.id -o out -prefix panel_tree --method nj -fmt pdf
```

Key options:

- input: `-nwk` or `-k` (GRM)
- GRM metadata: `-kid`
- layout/render: `-layout`, `-root`, `-showlabels`, `-regexlabels`, `--node-size`, `--edge-width`, `--scale-bar`, `--hover`
- output: `-fmt`, `-o`, `-prefix`, `--height`, `-ratio`

Outputs:

- `<prefix>.tree.<fmt>` (`svg/png/pdf/html`)
- `<prefix>.tree.nwk` when GRM is converted to NJ tree

### 4.25 `gblupbench` (Python dispatcher mode)

GBLUP engine benchmark across JanusX / sommer / rrBLUP / BLUPF90 family / HIBLUP.

Examples:

```bash
jxpy gblupbench -bfile example_prefix -p pheno.tsv -n 0
jxpy gblupbench -vcf example.vcf.gz -p pheno.tsv -n 0 --engines janusx,sommer,rrblup
```

Key options:

- input: `-vcf/-hmp/-file/-bfile`, phenotype `-p`, trait selector `-n`
- engine set: `--engines janusx,sommer,rrblup,blupf90,blupf90apy,hiblup`
- CV/runtime: `--cv`, `--seed`, `-t`, `-chunksize`, `-limit-mem`
- utility: `--check` (env + smoke test)

Outputs:

- benchmark workspace: `<prefix>.gblup_bench/`
- summary: `summary/<prefix>.gblupbench.tsv`, `.md`
- prediction compare artifacts: `summary/<prefix>.gblupbench.pred*`

### 4.26 `beam` (direct script entry)

AND-combination beam search over `JXBIN001` (`.bin`) genotype rows.

Entry:

```bash
python -m janusx.script.beam -bin panel.bin -p pheno.tsv -n 0 -m 3 -nsnp 5 -ext 50000
```

Key options:

- required: `-bin`, `-p`
- trait selection: `-n`
- search control: `-m` (max-pick), `-nsnp`, `-ext`, `-step`
- optional output prefix: `-o`

Outputs:

- `<prefix>.beam.tsv`
- `<prefix>.beam.windows.tsv`

## 5. Practical notes

- Use `jx <module> -h` as the final authority when flags change.
- For `-file` input, keep `prefix.id` consistent with genotype column order.
- For large cohorts, prefer streaming/chunked workflows (`-chunksize`, low-memory modes).
- Keep generated log files for reproducibility (`*.log`).
- For a full structure/entrypoint matrix, see `doc/PROJECT_MAP.md`.
