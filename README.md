# JanusX

[简体中文(推荐)](./doc/README_zh.md) | [English](./README.md)

## Overview

JanusX is a high-performance, all-in-one suite for quantitative genetics that unifies genome-wide association studies (GWAS) and genomic selection (GS). It includes established GWAS methods (GLM, LMM, fastLMM, FarmCPU) and GS models (GBLUP, rrBLUP, BayesA/B/Cpi), plus routine analyses from data processing to publication-ready visualization.

It delivers strong performance gains compared with [GEMMA](https://github.com/genetics-statistics/GEMMA), [GCTA](https://yanglab.westlake.edu.cn/software/gcta), and [rMVP](https://github.com/xiaolei-lab/rMVP), especially for multi-threaded computation.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the CLI](#running-the-cli)
- [Modules](#modules)
- [Quick Start](#quick-start)
- [Input File Formats](#input-file-formats)
- [CLI Reference](#cli-reference)
- [FAQ](#faq)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Key Algorithms](#key-algorithms)
- [Example Data](#example-data)
- [Citation](#citation)

## Requirements

- Python 3.10+
- Rust toolchain for source builds (maturin/PyO3)

## Installation

### From Source (Latest)

```bash
git clone https://github.com/FJingxian/JanusX.git
cd JanusX
pip install .
```

### PyPI (Stable)

```bash
pip install janusx
```

### Pre-compiled Releases (Stable)

We provide pre-compiled binaries on the [GitHub Releases](https://github.com/FJingxian/JanusX/releases) page for Windows, Linux, and macOS.
Download and extract the archive, then run the executable directly.

## Running the CLI

```bash
jx -h
jx <module> -h
jx <module> [options]
```

The first run may be slower while Python builds bytecode in `__pycache__`. Subsequent runs load the cached bytecode and start faster.

## Modules

| Module | Description |
| --- | --- |
| `gwas` | Unified GWAS wrapper (GLM/LMM/fastLMM/FarmCPU) |
| `gs` | Genomic Selection (GBLUP, rrBLUP, BayesA/B/Cpi) |
| `postGWAS` | Visualization and annotation |
| `grm` | Genetic relationship matrix (GRM) |
| `pca` | Principal component analysis |
| `sim` | Genotype and phenotype simulation |

## Quick Start

### GWAS Analysis

```bash
# Select one or more GWAS models
jx gwas --vcf data.vcf.gz --pheno pheno.txt --lmm -o results

# Run multiple models at once
jx gwas --vcf data.vcf.gz --pheno pheno.txt --lm --lmm --fastlmm --farmcpu -o results

# With PLINK format
jx gwas --bfile genotypes --pheno phenotypes.txt --grm 1 --qcov 3 --thread 8 -o results

# With diagnostic plots (SVG)
jx gwas --vcf data.vcf.gz --pheno pheno.txt --lmm --plot -o results
```

### Genomic Selection

```bash
# Run both GS models
jx gs --vcf data.vcf.gz --pheno pheno.txt --GBLUP --rrBLUP -o results

# Specific models
jx gs --vcf data.vcf.gz --pheno pheno.txt --GBLUP -o results

# Bayesian GS models
jx gs --vcf data.vcf.gz --pheno pheno.txt --BayesA --BayesB --BayesCpi -o results

# With PCA-based dimensionality reduction
jx gs --vcf data.vcf.gz --pheno pheno.txt --GBLUP --pcd -o results
```

### Visualization

```bash
# Generate Manhattan and QQ plots
jx postGWAS -f results/*.lmm.tsv --threshold 1e-6

# With SNP annotation
jx postGWAS -f results/*.lmm.tsv --threshold 1e-6 -a annotation.gff --annobroaden 50
```

![Manhattan and QQ plots](./fig/test0.png "Simple visualization")

Datasets used in the screenshots are in `example/` (sourced from [genetics-statistics/GEMMA](https://github.com/genetics-statistics/GEMMA)).

### Population Structure

```bash
# Compute GRM
jx grm --vcf data.vcf.gz -o results

# PCA analysis
jx pca --vcf data.vcf.gz --dim 5 --plot --plot3D -o results
```

## Input File Formats

### Phenotype File

Tab-delimited. First column is sample ID; remaining columns are phenotypes.

| samples | trait1 | trait2 |
|---------|--------|--------|
| indv1   | 10.5   | 0.85   |
| indv2   | 12.3   | 0.92   |

### Genotype Files

- **VCF**: `.vcf` or `.vcf.gz`
- **PLINK**: `.bed`/`.bim`/`.fam` (use file prefix)

## CLI Reference

### GWAS (`gwas`)

Unified GWAS runner supporting GLM, LMM, fastLMM (fixed lambda), and FarmCPU.

```bash
# LM / LMM (streaming-friendly)
jx gwas --vcf data.vcf.gz --pheno pheno.txt --lm -o out/
jx gwas --vcf data.vcf.gz --pheno pheno.txt --lmm -o out/

# fastLMM (fixed lambda) and FarmCPU (full-memory)
jx gwas --vcf data.vcf.gz --pheno pheno.txt --fastlmm -o out/
jx gwas --vcf data.vcf.gz --pheno pheno.txt --farmcpu -o out/

# Run multiple models at once
jx gwas --vcf data.vcf.gz --pheno pheno.txt --lm --lmm --fastlmm --farmcpu -o out/
```

| Option | What it does | Default |
| ------ | ------------ | ------- |
| `-vcf/--vcf`, `-bfile/--bfile` | Genotype source (VCF or PLINK prefix) | required |
| `-p/--pheno` | Phenotype file (tab-delimited) | required |
| `-n/--ncol` | Zero-based phenotype column indices (repeatable) | all columns |
| `-lm/--lm`, `-lmm/--lmm`, `-fastlmm/--fastlmm`, `-farmcpu/--farmcpu` | Choose GWAS model(s) | all `False` |
| `-k/--grm` | GRM method: `1` (centered), `2` (standardized), or path to precomputed | `1` |
| `-q/--qcov` | PC count or Q-matrix path for covariates | `0` |
| `-c/--cov` | Covariate file (without intercept column) | `None` |
| `-plot/--plot` | Generate Manhattan + QQ plots | `False` |
| `-chunksize/--chunksize` | SNP block size for streaming | `100000` |
| `-t/--thread` | CPU threads (`-1` for all cores) | `-1` |
| `-o/--out`, `-prefix/--prefix` | Output directory/prefix | `"."` / auto |

Select at least one GWAS model flag when running `gwas`.

**Outputs**

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
jx gs --vcf data.vcf.gz --pheno pheno.txt --GBLUP --rrBLUP -o out/
jx gs --vcf data.vcf.gz --pheno pheno.txt --BayesA --BayesB --BayesCpi -o out/

# Choose phenotype column and enable PCA-based dimensionality reduction and plotting
jx gs --vcf data.vcf.gz --pheno pheno.txt --n 0 --GBLUP --pcd --plot -o out/
```

| Option | What it does | Default |
| ------ | ------------ | ------- |
| `-vcf/--vcf`, `-bfile/--bfile` | Genotype source (VCF or PLINK prefix) | required (exclusive group) |
| `-p/--pheno` | Phenotype file | required |
| `-GBLUP/--GBLUP`, `-rrBLUP/--rrBLUP`, `-BayesA/--BayesA`, `-BayesB/--BayesB`, `-BayesCpi/--BayesCpi` | Models to run | all `False` |
| `-pcd/--pcd` | Enable PCA-based dimensionality reduction | `False` |
| `-n/--ncol` | Zero-based phenotype column index | all columns |
| `-plot/--plot` | Scatter plots for predicted vs. observed | `False` |
| `-o/--out`, `-prefix/--prefix` | Output directory/prefix | `"."` / auto |

**Outputs**

| File | Description |
| ---- | ----------- |
| `{prefix}.{trait}.gs.tsv` | GEBV predictions per model |
| `{prefix}.{trait}.gs.{model}.svg` | Prediction scatter plots (if `--plot`) |

### Genetic Relationship Matrix (`grm`)

Compute kinship matrices for LMM/GBLUP using centered (VanRaden) or standardized (Yang) formulas.

```bash
jx grm --vcf data.vcf.gz -o out/          # centered GRM
jx grm --vcf data.vcf.gz --method 2 -o out/  # standardized GRM
jx grm --vcf data.vcf.gz --npz -o out/       # compressed NPZ output
```

| Option | What it does | Default |
| ------ | ------------ | ------- |
| `-vcf/--vcf`, `-bfile/--bfile` | Genotype source | required (exclusive group) |
| `-m/--method` | `1` centered, `2` standardized | `1` |
| `-npz/--npz` | Save compressed NPZ matrix | `False` |
| `-o/--out`, `-prefix/--prefix` | Output directory/prefix | `"."` / auto |

**Outputs**

| File | Description |
| ---- | ----------- |
| `{prefix}.grm.id` | Sample IDs |
| `{prefix}.grm.txt` | GRM matrix (text) |
| `{prefix}.grm.npz` | GRM matrix (compressed, if enabled) |

### PCA (`pca`)

Principal component analysis for population structure, with 2D/3D visualization and grouping.

```bash
# From genotype or precomputed GRM/PCA
jx pca --vcf data.vcf.gz -o out/
jx pca --grm prefix -o out/
jx pca --pcfile prefix --plot --plot3D -o out/

# Control dimensions and grouping
jx pca --vcf data.vcf.gz --dim 5 --plot --plot3D --group groups.txt -o out/
```

| Option | What it does | Default |
| ------ | ------------ | ------- |
| `-vcf/--vcf`, `-bfile/--bfile`, `-grm/--grm`, `-pcfile/--pcfile` | Input source (exclusive) | required |
| `-dim/--dim` | Number of PCs to output | `3` |
| `-plot/--plot`, `-plot3D/--plot3D` | 2D scatter; 3D rotating GIF | `False` |
| `-group/--group` | Group file (`ID\tGroup\tLabel?`) | `None` |
| `-color/--color` | Color palette index (0-6) | `1` |
| `-o/--out`, `-prefix/--prefix` | Output directory/prefix | `"."` / auto |

Group file format:

```text
ID      Group   Label (optional)
indv1   PopA    Sample1
indv2   PopA    Sample2
indv3   PopB    Sample3
```

**Outputs**

| File | Description |
| ---- | ----------- |
| `{prefix}.eigenvec` | PC coordinates (samples × dims) |
| `{prefix}.eigenvec.id` | Sample IDs |
| `{prefix}.eigenval` | Eigenvalues |
| `{prefix}.eigenvec.2D.pdf` | 2D scatter (if `--plot`) |
| `{prefix}.eigenvec.3D.gif` | 3D rotating GIF (if `--plot3D`) |

### postGWAS (`postGWAS`)

Visualization and annotation of GWAS results (Manhattan, QQ, optional SNP annotation and highlighting).

```bash
# Basic plotting
jx postGWAS -f result.assoc.tsv

# Custom column names and threshold
jx postGWAS -f result.tsv -chr "chr" -pos "pos" -pvalue "P_wald" --threshold 1e-6

# Highlight regions and annotate significant SNPs
jx postGWAS -f result.tsv --highlight snps.bed --anno annotation.gff --annobroaden 100
```

| Option | What it does | Default |
| ------ | ------------ | ------- |
| `-f/--file` | GWAS result files (supports glob) | required |
| `-chr/--chr`, `-pos/--pos`, `-pvalue/--pvalue` | Column names | `"#CHROM"`, `"POS"`, `"p"` |
| `-threshold/--threshold` | Significance cutoff | `0.05 / SNPs` |
| `-noplot/--noplot` | Disable plots (annotation only) | `False` |
| `-color/--color` | Color scheme (0-6) | `0` |
| `-hl/--highlight` | BED file for highlighted regions | `None` |
| `-format/--format` | Image format `pdf/png/svg/tif` | `"png"` |
| `-a/--anno` | Annotation file (GFF/GTF/BED) | `None` |
| `-ab/--annobroaden` | Annotation window (kb) | `None` |
| `-descItem/--descItem` | GFF description key | `"description"` |
| `-o/--out`, `-prefix/--prefix`, `-t/--thread` | Output dir/prefix; threads (`-1` all cores) | `"."` / `"JanusX"` / `-1` |

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

**Outputs**

| File | Description |
| ---- | ----------- |
| `{prefix}.manh.{format}` | Manhattan plot |
| `{prefix}.qq.{format}` | QQ plot |
| `{prefix}.{threshold}.anno.tsv` | Annotated significant SNPs (if enabled) |

### Simulation (`sim`)

Generate synthetic genotype/phenotype data for workflow testing.

```bash
# 100k SNPs, 500 individuals
jx sim 100 500 output_prefix

# 500k SNPs, 1000 individuals
jx sim 500 1000 mydata
```

| Argument | Description |
| -------- | ----------- |
| `nsnp(k)` | SNP count in thousands |
| `nind` | Number of individuals |
| `outprefix` | Output prefix |

**Outputs**

| File | Description |
| ---- | ----------- |
| `{prefix}.vcf.gz` | Simulated genotypes |
| `{prefix}.pheno` | Full phenotype file |
| `{prefix}.pheno.txt` | Simplified phenotype (GS-friendly) |

## FAQ

- **Which GWAS model to pick?** Large, structure-free cohorts -> GLM; population structure/kinship -> LMM; need speed with acceptable approximation -> fastLMM; balance power/false positives with full genotype in memory -> FarmCPU.
- **Memory pressure?** Prefer GLM/LMM (streaming), lower `--chunksize`, avoid FarmCPU unless full genotypes fit in memory.
- **Intermediate caching** (auto-reused across runs): `{prefix}.k.{method}.npy` (GRM), `{prefix}.q.{dim}.txt` (PCA), `{prefix}.grm.bin` (binary GRM).
- **Empty outputs?** Verify genotype/phenotype sample IDs align, check phenotype columns for missing values, try `--thread 1` for debugging, inspect log files.
- **Covariates format** (no header, one row per sample, no intercept column):
  ```text
  0.1    -0.2     25
  0.3     0.1     30
  ```
- **Reading QQ plots**: points along diagonal -> OK; upper-right inflation -> significant hits; overall deviation -> likely population structure or technical confounding.

## Architecture

### Core Libraries

- `python/janusx/pyBLUP` - Core statistical engine
  - GWAS implementations (GLM, LMM, FarmCPU)
  - K/Q matrix calculation with memory-optimized chunking
  - PCA computation with randomized SVD
  - Cross-validation utilities

- `python/janusx/gfreader` - Genotype file I/O
  - VCF reader
  - PLINK binary reader (.bed/.bim/.fam)
  - NumPy format support

- `python/janusx/bioplotkit` - Visualization
  - Manhattan and QQ plots
  - PCA visualization (2D and 3D GIF)
  - LD block visualization

### Native Core (`src/`)

Rust kernels for fast linear algebra and association testing.

### CLI Entry Points (`python/janusx/script/`)

Each module corresponds to a CLI command. The launcher script (`jx`) dispatches to `script/<name>.py`.

## Key Features

- **Two Core Functions**: Unified GWAS and GS workflows in one tool
- **Easy to Use**: Simple CLI interface, minimal configuration required
- **High Performance**: Optimized LMM computation with multi-threading

## Key Algorithms

### GWAS Methods

| Method | Description | Best For |
| --- | --- | --- |
| **Linear Model (GLM)** | Standard linear model for association testing | Large datasets without population structure |
| **Linear Mixed Model (LMM)** | Incorporates kinship matrix to control population structure | Most GWAS scenarios |
| **fastLMM** | Fixed-lambda mixed model for speed | Fast approximate screening |
| **FarmCPU** | Iterative fixed/random effect alternation | High power with strict false positive control |

### GS Methods

| Method | Description | Best For |
| --- | --- | --- |
| **GBLUP** | Genomic Best Linear Unbiased Prediction | Baseline prediction |
| **rrBLUP** | Ridge Regression BLUP | Additive genetic value estimation |
| **BayesA** | Marker effects with scaled-t prior | Polygenic traits with heavier tails |
| **BayesB** | Variable selection with marker-specific variance | Sparse genetic architecture |
| **BayesCpi** | Variable selection with shared variance | Sparse architecture with shared variance |

### Kinship Methods

- **Method 1 (VanRaden)**: Centered GRM (default)
- **Method 2 (Yang)**: Standardized/weighted GRM

## Example Data

Sample datasets live in `example/` (sourced from [GEMMA](https://github.com/genetics-statistics/GEMMA.git)). Additional fixtures and example outputs are in `test/`.

## Citation

```bibtex
@software{JanusX,
  title = {JanusX: High-performance GWAS and Genomic Selection Suite},
  author = {Jingxian FU},
  url = {https://github.com/FJingxian/JanusX}
}
```
