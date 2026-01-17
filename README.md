# JanusX

[简体中文(推荐)](./doc/README_zh.md) | [English](./README.md)

## Project Overview

JanusX is a high-performance, ALL-in-ONE suite for quantitative genetics that unifies genome-wide association studies (GWAS) and genomic selection (GS). It incorporates well-established GWAS methods (LM, LMM, and FarmCPU) and a flexible GS toolkit including GBLUP and various machine learning models. It also combines routine genomic analyses, from data processing to publication-ready visualisation.

It provides significant performance improvements over tools like GEMMA, GCTA, and rMVP, especially in multi-threaded computation.

## Installation

### PyPI

```bash
pip install janusx
```

### From Source

```bash
git clone https://github.com/FJingxian/JanusX.git
cd JanusX
pip install .
```

Building from source requires a Rust toolchain (maturin will compile the native core).

### Pre-compiled Releases

We provide pre-compiled binaries on the [GitHub Releases](https://github.com/FJingxian/JanusX/releases) page for Windows, Linux, and macOS.
Download and extract the archive, then run the executable directly.

## Running the CLI

```bash
jx -h
jx <module> [options]
```

Note that running `jx -h` might take a while at first! This is because the Python interpreter is compiling source code into the `pycache` directory. Subsequent runs will use the pre-compiled code and load much faster!

## Available Modules

| Module | Description |
|:-------|:------------|
| `gwas` | Unified GWAS wrapper (LM/LMM/fastLMM/FarmCPU) |
| `gs` | Genomic Selection (GBLUP, rrBLUP, BayesA/B/Cpi) |
| `postGWAS` | Visualization and annotation |
| `grm` | Genetic relationship matrix calculation |
| `pca` | Principal component analysis |
| `sim` | Genotype and phenotype simulation |

## Quick Start Examples

### GWAS Analysis

```bash
# Using unified gwas module (select one or more models)
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

![manhanden&qq](./fig/test0.png "Simple visualization")

Test data in example is from [genetics-statistics/GEMMA](https://github.com/genetics-statistics/GEMMA), published in [Parker et al, Nature Genetics, 2016](https://doi.org/10.1038/ng.3609)

### Population Structure

```bash
# Compute GRM
jx grm --vcf data.vcf.gz -o results

# PCA analysis
jx pca --vcf data.vcf.gz --dim 5 --plot --plot3D -o results
```

## Input File Formats

### Phenotype File

Tab-delimited, first column is sample ID, subsequent columns are phenotypes:

| samples | trait1 | trait2 |
|---------|--------|--------|
| indv1   | 10.5   | 0.85   |
| indv2   | 12.3   | 0.92   |

### Genotype Files

- **VCF**: `.vcf` or `.vcf.gz`
- **PLINK**: `.bed`/`.bim`/`.fam` (use prefix)

## Architecture

### Core Libraries

- **python/janusx/pyBLUP** - Core statistical engine
  - GWAS implementations (LM, LMM, FarmCPU)
  - QK matrix calculation with memory-optimized chunking
  - PCA computation with randomized SVD
  - Cross-validation utilities

- **python/janusx/gfreader** - Genotype file I/O
  - VCF reader
  - PLINK binary reader (.bed/.bim/.fam)
  - NumPy format support

- **python/janusx/bioplotkit** - Visualization
  - Manhattan and QQ plots
  - PCA visualization (2D and 3D GIF)
  - LD block visualization

### Native Core (src/)

Rust kernels for fast linear algebra and association testing.

### CLI Entry Points (python/janusx/script/)

Each module corresponds to a CLI command. The launcher script (`jx`) dispatches to `script/<name>.py`.

## Key Features

- **Two Core Functions**: Unified GWAS and GS workflows in one tool
- **Easy to Use**: Simple CLI interface, minimal configuration required
- **High Performance**: Optimized LMM computation with multi-threading

## Key Algorithms

### GWAS Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Linear Model (LM)** | Standard GLM for association testing | Large datasets without population structure |
| **Linear Mixed Model (LMM)** | Incorporates kinship matrix to control population structure | Most GWAS scenarios |
| **fastLMM** | Fixed-lambda mixed model for speed | Fast approximate screening |
| **FarmCPU** | Iterative fixed/random effect alternation | High power with strict false positive control |

### GS Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **GBLUP** | Genomic Best Linear Unbiased Prediction | Baseline prediction |
| **rrBLUP** | Ridge Regression BLUP | Additive genetic value estimation |
| **BayesA** | Marker effects with scaled-t prior | Polygenic traits with heavier tails |
| **BayesB** | Variable selection with marker-specific variance | Sparse genetic architecture |
| **BayesCpi** | Variable selection with shared variance | Sparse architecture with shared variance |

### Kinship Methods

- **Method 1 (VanRaden)**: Centered GRM (default)
- **Method 2 (Yang)**: Standardized/weighted GRM

## Python Version

Requires Python 3.10+

## Test Data

Example data in `example/` directory from Parker et al, Nature Genetics, 2016 (via GEMMA project)

## Citation

```bibtex
@software{JanusX,
  title = {JanusX: High-performance GWAS and Genomic Selection Suite},
  author = {Jingxian FU},
  url = {https://github.com/FJingxian/JanusX}
}
```
