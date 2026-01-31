# JanusX

[简体中文](./doc/README_zh.md) | [English](./README.md) | [Zea Eureka](https://mp.weixin.qq.com/s/jl3h2DPRC21l8QJ0WxzXDA)

**JanusX** is a high-performance, all-in-one suite for quantitative genetics that unifies genome-wide association studies (GWAS) and genomic selection (GS). It includes established GWAS methods (GLM, LMM, fastLMM, FarmCPU) and GS models (GBLUP, rrBLUP, BayesA/B/Cpi), plus routine analyses from data processing to publication-ready visualization.

It delivers strong performance gains compared with [GEMMA](https://github.com/genetics-statistics/GEMMA), [GCTA](https://yanglab.westlake.edu.cn/software/gcta), and [rMVP](https://github.com/xiaolei-lab/rMVP), especially for multi-threaded computation.

---

## Quick Start

**GWAS Analysis**:

```bash
jx gwas -bfile data -p pheno.txt -lmm
jx gwas -vcf data.vcf.gz -p pheno.txt -lmm
```

**Genomic Selection**:

```bash
jx gs -vcf data.vcf.gz -p pheno.txt -GBLUP
```

**postGWAS**:

```bash
# Generate Manhattan and QQ plots
jx postGWAS -f result.lmm.tsv
# With SNP annotation
jx postGWAS -f results/test.lmm.tsv -a annotation.gff -ab 30
```

![Manhattan and QQ plots](./fig/test0.png "Simple visualization")

Datasets used in the screenshots are in `example/` (sourced from [genetics-statistics/GEMMA](https://github.com/genetics-statistics/GEMMA)).

**Population Structure**:

```bash
# Compute GRM
jx grm -bfile data.vcf.gz
# PCA
jx pca -bfile data.vcf.gz
```

---

## Installation

Build from source required Python 3.9+ and Rust toolchain.

### From Source (Latest)

Build via Python

```bash
pip install git+https://github.com/FJingxian/JanusX.git
jx -h
```

or docker

```bash
git clone https://github.com/FJingxian/JanusX.git
cd JanusX
docker build -t janusx:latest .
docker run --rm janusx:latest -h
```

### PyPI (Stable)

```bash
pip install janusx
jx -h
```

### Pre-compiled Releases (Stable)

We provide pre-compiled binaries on the [GitHub Releases](https://github.com/FJingxian/JanusX/releases) page for Windows (build in Windows11) and Linux (build in Ubuntu20.04).
Download and extract the archive, then run the executable directly.

---

## Modules in JanusX

| Module | Description |
| --- | --- |
| `gwas` | Unified GWAS wrapper (GLM/LMM/fastLMM/FarmCPU) |
| `gs` | Genomic Selection (GBLUP, rrBLUP, BayesA/B/Cpi) |
| `postGWAS` | Visualization and annotation |
| `grm` | Genetic relationship matrix (GRM) |
| `pca` | Principal component analysis |
| `sim` | Genotype and phenotype simulation |

---

## Input File Formats

### Phenotype File

Tab-delimited. First column is sample ID; remaining columns are phenotypes.

```text
samples	trait1	trait2	trait3
indv1	10.5	0.85	0.05
indv2	12.3	1.00	0.08
indv3	8.6	0.92	0.04
```

### Genotype Files

- **VCF**: `.vcf` or `.vcf.gz`

```text
##fileformat=VCFv4.2
##fileDate=20251201
##source=JanusXv1.0.10
##contig=<ID=1,length=2>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	A	B	C	D	E	F
1	1	10000235	A	C	.	.	PR	GT	0/1	0/0	0/0	0/0	0/0	0/1
1	1	10000345	A	G	.	.	PR	GT	0/0	0/0	0/0	0/0	1/1	1/1
1	1	10004575	G	.	.	.	PR	GT	0/0	0/0	0/0	0/0	0/0	0/0
1	1	10006974	C	T	.	.	PR	GT	0/0	0/0	0/1	1/1	0/1	1/1
1	1	10006986	A	G	.	.	PR	GT	0/0	0/0	0/1	./.	1/1	1/1
```

- **PLINK**: `.bed`/`.bim`/`.fam` (use file prefix, details see http://zzz.bwh.harvard.edu/plink/data.shtml#bed)

### GWAS Optional Matrices (GRM / Q / Covariate)

- **GRM**: A or G matrix (Kinship between individuals).

```text
1.00	0.12	0.05
0.12	1.00	0.08
0.05	0.08	1.00
```

- **Q**: Q matrix generated from admixture or principal components.

```text
0.12	-0.03
-0.05	0.08
0.02	-0.01
```

- **Covariate**: Sex, environments or date.

```text
0	1
1	1
0	2
```

Files passed to `--grm`|`-k`, `--qcov`|`-q`, and `--cov`|`-c` must be numeric only and aligned to the genotype sample order. Numbers are splited by space- or tab-delimited without row names or column headers.

---

## CLI Reference

### GWAS (`gwas`)

Unified GWAS runner supporting GLM, LMM, fastLMM (fixed lambda), and FarmCPU.

```bash
# LM / LMM (estimated variance parameter per snp) / fastLMM (fixed variance parameter) are streaming-friendly
jx gwas -bfile data -p pheno.txt -lm -o out
jx gwas -bfile data -p pheno.txt -lmm -o out
jx gwas -bfile data -p pheno.txt -fastlmm -o out

# FarmCPU uses full-memory to load genotype matrix
jx gwas -bfile data -p pheno.txt -farmcpu -o out

# Run multiple models at once
jx gwas -bfile data -p pheno.txt -lm -lmm -fastlmm -farmcpu -o out
```

Note: Select at least one GWAS model flag when running `gwas`.

**Input**:

- `-vcf, --vcf` / `-bfile, --bfile` — Genotype source (VCF path or PLINK prefix).  
  Default: required
- `-p, --pheno` — Phenotype file (tab-delimited).  
  Default: required
- `-n, --ncol` — Phenotype column indices (0-based, repeatable).  
  Default: all columns

**Models**:

- `-lm, --lm` — Run linear model (LM) GWAS.  
  Default: `False`
- `-lmm, --lmm` — Run linear mixed model (LMM) GWAS.  
  Default: `False`
- `-fastlmm, --fastlmm` — Run FaST-LMM style approximation.  
  Default: `False`
- `-farmcpu, --farmcpu` — Run FarmCPU.  
  Default: `False`

**Relatedness &amp; Covariates**:

- `-k, --grm` — GRM setting: `1` = centered, `2` = standardized, or path to precomputed GRM.  
  Default: `1`
- `-q, --qcov` — Population covariates: PC count or Q-matrix path.  
  Default: `0`
- `-c, --cov` — Covariate file (without intercept column).  
  Default: `None`

**Variant QC**:

- `-maf, --maf` — Exclude variants with MAF &lt; threshold.  
  Default: `0.02`
- `-geno, --geno` — Exclude variants with missing rate &gt; threshold.  
  Default: `0.05`

**Output &amp; Performance**:

- `-plot, --plot` — Generate Manhattan + QQ plots.  
  Default: `False`
- `-chunksize, --chunksize` — SNP block size for streaming.  
  Default: `100000`
- `-mmap-limit, --mmap-limit` — Enable windowed mmap for BED inputs (benchmark only).  
  Default: `False`
- `-t, --thread` — CPU threads (`-1` = all cores).  
  Default: `-1`
- `-o, --out` — Output directory.  
  Default: `"."`
- `-prefix, --prefix` — Output prefix (auto-generated if omitted).  
  Default: auto

**Output files**:

| File | Description |
| ---- | ----------- |
| `{prefix}.{trait}.lm.tsv` | GLM results |
| `{prefix}.{trait}.lmm.tsv` | LMM results |
| `{prefix}.{trait}.fastlmm.tsv` | fastLMM results |
| `{prefix}.{trait}.farmcpu.tsv` | FarmCPU results |
| `{prefix}.{trait}.{model}.svg` | Histogram, Manhattan, and QQ plots (if `--plot`) |

### Genomic Selection (`gs`)

Run genomic prediction with linear and Bayesian models (GBLUP, rrBLUP, BayesA/B/Cpi).

```bash
# Run selected models
jx gs -vcf data.vcf.gz -p pheno.txt -GBLUP -rrBLUP -o out
jx gs -vcf data.vcf.gz -p pheno.txt -BayesA -BayesB -BayesCpi -o out

# Choose phenotype column and enable PCA-based dimensionality reduction and plotting with 5-fold cross-validazation.
jx gs -vcf data.vcf.gz -pheno pheno.txt -n 0 -GBLUP -pcd -plot -o out -cv 5
```

**Input**:

- `-vcf, --vcf` / `-bfile, --bfile` — Genotype source (VCF path or PLINK prefix).  
  Default: required
- `-p, --pheno` — Phenotype file (tab-delimited).  
  Default: required
- `-n, --ncol` — Phenotype column indices (0-based, repeatable).  
  Default: all columns

**Models**:

- `-GBLUP/--GBLUP`, `-rrBLUP/--rrBLUP`, `-BayesA/--BayesA`, `-BayesB/--BayesB`, `-BayesCpi/--BayesCpi` — Models to run.  
  Default: all `False`

**Variant QC**:

- `-maf, --maf` — Exclude variants with MAF &lt; threshold.  
  Default: `0.02`
- `-geno, --geno` — Exclude variants with missing rate &gt; threshold.  
  Default: `0.05`

**Output &amp; Performance**:

- `-cv/--cv` — Enable K fold of cross-validazation for models.  
  Default: `None`
- `-pcd/--pcd` — Enable PCA-based dimensionality reduction.  
  Default: `False`
- `-plot/--plot` — Scatter plots for predicted vs. observed.  
  Default: `False`
- `-o/--out`, `-prefix/--prefix` — Output directory/prefix.  
  Default: `"."` / auto

**Outputs**:

| File | Description |
| ---- | ----------- |
| `{prefix}.{trait}.gs.tsv` | GEBV predictions per model |
| `{prefix}.{trait}.gs.{model}.svg` | Prediction scatter plots (if `--plot`) |

### postGWAS (`postGWAS`)

Visualization and annotation of GWAS results (Manhattan, QQ, optional SNP annotation and highlighting).

```bash
# Basic plotting
jx postGWAS -f result.lmm.tsv

# Custom column names and threshold
jx postGWAS -f result.tsv -chr "chr" -pos "ps" -pvalue "p_wald" -threshold 1e-6

# Highlight regions and annotate significant SNPs
jx postGWAS -f result.tsv -hl snps.bed -a annotation.gff -ab 100
```

**Input**:

- `-f/--file` — GWAS result files (supports glob).  
  Default: required

**Name of columns**:

- `-chr/--chr`, `-pos/--pos`, `-pvalue/--pvalue` — Column names.  
  Default: `"chrom"`, `"pos"`, `"pwald"`

**Other parameters**:

- `-threshold/--threshold` — Significance cutoff.  
  Default: `0.05 / SNPs`
- `-noplot/--noplot` — Disable plots (annotation only).  
  Default: `False`
- `-color/--color` — Color scheme (0-6).  
  Default: `0`
- `-hl/--highlight` — BED file for highlighted regions.  
  Default: `None`
- `-format/--format` — Image format `pdf/png/svg/tif`.  
  Default: `"png"`
- `-a/--anno` — Annotation file (GFF/GTF/BED).  
  Default: `None`
- `-ab/--annobroaden` — Annotation window (kb).  
  Default: `None`
- `-descItem/--descItem` — GFF description key.  
  Default: `"description"`
- `-o/--out`, `-prefix/--prefix`, `-t/--thread` — Output dir/prefix; threads (`-1` all cores).  
  Default: `"."` / `"JanusX"` / `-1`

Highlight file format:

```text
chr1    start   end     name    description
chr1    1000000 1000000 GeneA   Description of GeneA
chr1    2000000 2000000 GeneB   Description of GeneB
```

Annotation file format (GFF example):

```text
chr1    .       gene    1000000 2000000 .   +   .   ID=gene1;description=Example gene
chr1    .       mRNA    1000000 2000000 .   +   .   ID=mRNA1;Parent=gene1
```

**Outputs**:

| File | Description |
| ---- | ----------- |
| `{prefix}.manh.{format}` | Manhattan plot |
| `{prefix}.qq.{format}` | QQ plot |
| `{prefix}.{threshold}.anno.tsv` | Annotated significant SNPs (if enabled) |

### Simulation (`sim`)

Generate synthetic genotype/phenotype data for workflow testing.

```bash
jx sim [nsnp(k)] [nidv] [outprefix]
```

---

## Citation

```bibtex
@article {FuJanusX,
  title = {JanusX: an integrated and high-performance platform for scalable genome-wide association studies and genomic selection},
  author = {Fu, Jingxian and Jia, Anqiang and Wang, Haiyang and Liu, Hai-Jun},
  year = {2026},
  doi = {10.64898/2026.01.20.700366},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2026/01/23/2026.01.20.700366},
  journal = {bioRxiv}
}
```
