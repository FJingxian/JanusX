# JanusX CLI Guide

Version baseline: `v1.0.13`  
Entry command: `jx`

## 1. Overview

```bash
jx -h
jx -v
jx <module> -h
```

Rust launcher (`jx`): environment setup, update/upgrade, DLC runtime tool routing, and Rust pipelines (`fastq2vcf`, `fastq2count`).
Python core modules: statistical modeling, plotting, and downstream analysis (`gwas`, `gs`, `postgwas`, `postbsa`, etc.).

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

## 3. Module List

**Genome-wide Association Studies (GWAS)**:

```text
grm
pca
gwas
postgwas
```

**Genomic Selection (GS)**:

```text
gs
reml
```

**GARFIELD**:

```text
garfield
postgarfield
```

**Bulk Segregation Analysis (BSA)**:

```text
postbsa
```

**Pipelines and utilities**:

```text
fastq2count
fastq2vcf
hybrid
gformat
gmerge
webui
loadanno
```

**Benchmark**:

```text
sim
simulation
```

## 4. Input File Conventions

### 4.1 Genotype Inputs (`-vcf/-hmp/-bfile/-file`)

```text
-vcf     .vcf / .vcf.gz
-hmp     .hmp / .hmp.gz
-bfile   PLINK prefix with .bed/.bim/.fam
-file    JanusX matrix input (text/npy path or prefix)
```

**VCF (`-vcf`)**

```vcf
##fileformat=VCFv4.2
#CHROM POS ID REF ALT QUAL FILTER INFO FORMAT S1 S2 S3
1 10583 rs1 A G . PASS . GT 0/1 0/0 1/1
1 10611 rs2 C T . PASS . GT 0/0 0/1 0/0
1 13302 rs3 G A . PASS . GT 1/1 0/1 0/0
```

**HMP (`-hmp`)**

```hmp
rs# alleles chrom pos strand assembly# center protLSID assayLSID panelLSID QCcode S1 S2 S3
SNP_1 A/G 1 1001 + NA NA NA NA NA NA AA AG GG
SNP_2 C/T 1 2050 + NA NA NA NA NA NA CT CC TT
SNP_3 G/T 2 3300 + NA NA NA NA NA NA GG NN GT
```

**PLINK (`-bfile`)**

```text
Input argument: -bfile /path/to/panel
Required files:
  /path/to/panel.bed
  /path/to/panel.bim
  /path/to/panel.fam
```

**JanusX FILE (`-file`)**

`prefix.id` (required):

```text
Sample_001
Sample_002
Sample_003
```

`prefix.txt` (required unless using `prefix.npy`):

```text
1 0 1
1 2 1
1 0 0
```

`prefix.site` (optional, but required for VCF/HMP/PLINK export):

```text
1 10583 A G
1 10611 C T
1 13302 G A
```

### 4.2 Phenotype Input (`-p/--pheno`)

```text
sample_id trait1 trait2
S1 1.23 5.67
S2 2.34 6.78
S3 3.45 7.89
```

### 4.3 Trait Column Selection (`-n`)

It is **zero-based** over phenotype columns **excluding** the first sample-ID column.

```bash
-n 0
-n 0,2,5
-n 0:5
-n 0-5
-n 0 -n 3
```

### 4.4 GRM Input (`gwas -k/--grm`)

```text
-k 1             Build centered GRM from genotype
-k 2             Build standardized GRM from genotype
-k <path>        Use external GRM file (.txt/.npy)
```

`grm.txt` or `grm.npy`:

```text
2.2 0.0 1.1
0.0 2.1 1.5
1.1 1.5 2.1
```

`grm.txt.id` or `grm.npy.id`:

```text
Sample_001
Sample_002
Sample_003
```

### 4.5 Q/PC and Covariates (`gwas -q/-c`)

```text
-q 0             Disable PCA covariates
-q N             Use top N PCs from GRM
```

External Q matrix via `-q <file>` is deprecated/removed.
Use `-c` for external covariates.

`-c` is repeatable and supports:

1. Covariate file path.
2. Single-site token: `chr:pos` or `chr:start:end` (single-site form).

**Covariate file examples**:

```text
S1 0.2 1.0
S2 -0.7 0.5
S3 1.3 -0.2
```

## 5. Core Commands

### 5.1 GWAS / GS / REML

```bash
# GWAS
jx gwas -vcf geno.vcf.gz -p pheno.tsv -lmm -o outdir

# GS
jx gs -hmp geno.hmp.gz -p pheno.tsv -GBLUP -cv 5 -o outdir

# REML
jx reml -file test.reml.txt -rh 0,1,2 -n 3 -o outdir -prefix reml
```

### 5.2 Post-analysis

```bash
# postgwas
jx postgwas -gwasfile result.tsv -manh -qq -o outdir

# postbsa (glob supported)
jx postbsa -file '4.merge/Merge.*.SNP.tsv' -b1 Bulk1 -b2 Bulk2 -t 8
```

### 5.3 Genotype Utilities

```bash
# format conversion
jx gformat -vcf panel.vcf.gz -fmt npy -o out -prefix panel

# merge inputs
jx gmerge -vcf a.vcf.gz b.vcf.gz -fmt vcf -o out -prefix merged

# build hybrid matrix
jx hybrid -file geno_prefix -p1 p1.list -p2 p2.list -fmt npy -o out -prefix hybrid
```

## 6. Pipeline Modules

### 6.1 `fastq2vcf`

```bash
jx fastq2vcf -r ref.fa -i raw_fastq_dir -w workdir
jx fastq2vcf -r ref.fa -i 3.gvcf -w workdir -from-step 4 -to-step 6
```

Current step names:

```text
1. fastp
2. bwamem+markdup
3. bam2gvcf
4. gvcf2gtprep
5. impute_filter
6. mergevcf
```

### 6.2 `fastq2count`

```bash
jx fastq2count -r ref.fa -a anno.gtf -i fastq_dir -w workdir
jx fastq2count -r ref.fa -a anno.gff3 -i work/3.mapping -w work -from-step 3 -to-step 4
```

## 7. Update and Upgrade Policy

1. `jx -update latest`: update/install Python core from latest source.
2. `jx -update dlc`: validate/install external runtime tools.
3. `jx -upgrade`: rebuild/replace launcher from source, then update core.
