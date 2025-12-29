# JanusX

[简体中文(推荐)](./doc/README_zh.md) | [English](./README.md) | [算法分享](./doc/algorithm_zh.md)

## Project Overview

JanusX is a high-performance, ALL-in-ONE suite for quantitative genetics that unifies genome-wide association studies (GWAS) and genomic selection (GS). It incorporates well-established GWAS methods (LM, LMM, and FarmCPU) and a flexible GS toolkit including GBLUP and various machine learning models. It also combines routine genomic analyses, from data processing to publication-ready visualisation.

It provides significant performance improvements over tools like GEMMA, GCTA, and rMVP, especially in multi-threaded computation.

## Installation

```bash
git clone https://github.com/MaizeMan-JxFU/JanusX.git
cd JanusX
sh ./install.sh
```

The install script uses `uv` for dependency management and creates a virtual environment in `.venv/`.

### Pre-compiled Releases

For convenience, we also provide pre-compiled binaries that don't require building from source. The releases are available at [Releases v1.0.0](https://github.com/MaizeMan-JxFU/JanusX/releases/tag/v1.0.0) for the following platforms:

- **Linux AMD64**: [JanusX-v1.0.0-linux-AMD64.tgz](https://github.com/MaizeMan-JxFU/JanusX/releases/download/v1.0.0/JanusX-v1.0.0-linux-AMD64.tgz)
- **Windows AMD64**: [JanusX-v1.0.0-windows-AMD64.tgz](https://github.com/MaizeMan-JxFU/JanusX/releases/download/v1.0.0/JanusX-v1.0.0-windows-AMD64.tgz)

Simply download and extract the archive, then run the executable directly.

**Note**: Windows installation is no longer supported. Please use Linux/macOS or Windows Subsystem for Linux (WSL). But there is a pre-build version for Windows.

## Running the CLI

```bash
./jx -h
./jx <module> [options]
```

Note that running `./jx -h` might take a while at first! This is because the Python interpreter is compiling source code into the `pycache` directory. Subsequent runs will use the pre-compiled code and load much faster!

## Available Modules

| Module | Description |
|:-------|:------------|
| `gwas` | Unified GWAS wrapper (LM/LMM/FarmCPU) |
| `lm` | Linear Model GWAS (streaming, low-memory) |
| `lmm` | Linear Mixed Model GWAS (streaming, low-memory) |
| `farmcpu` | FarmCPU GWAS (high-memory) |
| `gs` | Genomic Selection (GBLUP, rrBLUP, RF, SVM, AdaBoost) |
| `postGWAS` | Visualization and annotation |
| `grm` | Genetic relationship matrix calculation |
| `pca` | Principal component analysis |
| `sim` | Genotype and phenotype simulation |

## Quick Start Examples

### GWAS Analysis

```bash
# Using unified gwas module (select one or more models)
./jx gwas --vcf data.vcf.gz --pheno pheno.txt --lmm --out results

# Run all three models at once
./jx gwas --vcf data.vcf.gz --pheno pheno.txt --lm --lmm --farmcpu --out results

# Or use individual modules directly
./jx lm --vcf data.vcf.gz --pheno pheno.txt --out results
./jx lmm --vcf data.vcf.gz --pheno pheno.txt --out results
./jx farmcpu --vcf data.vcf.gz --pheno pheno.txt --out results

# With PLINK format
./jx gwas --bfile genotypes --pheno phenotypes.txt --out results --grm 1 --qcov 3 --thread 8

# With diagnostic plots
./jx gwas --vcf data.vcf.gz --pheno pheno.txt --lmm --plot --out results
```

### Genomic Selection

```bash
# Run all GS models
./jx gs --vcf data.vcf.gz --pheno pheno.txt --out results

# Specific models
./jx gs --vcf data.vcf.gz --pheno pheno.txt --GBLUP --rrBLUP --RF --out results

# With PCA-based dimensionality reduction
./jx gs --vcf data.vcf.gz --pheno pheno.txt --GBLUP --pcd --out results
```

### Visualization

```bash
# Generate Manhattan and QQ plots
./jx postGWAS -f results/*.lmm.tsv --threshold 1e-6

# With SNP annotation
./jx postGWAS -f results/*.lmm.tsv --threshold 1e-6 -a annotation.gff --annobroaden 100
```

![manhanden&qq](./fig/test0.png "Simple visualization")

Test data in example is from [genetics-statistics/GEMMA](https://github.com/genetics-statistics/GEMMA), published in [Parker et al, Nature Genetics, 2016](https://doi.org/10.1038/ng.3609)

### Population Structure

```bash
# Compute GRM
./jx grm --vcf data.vcf.gz --out results

# PCA analysis
./jx pca --vcf data.vcf.gz --dim 5 --plot --plot3D --out results
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
- **NPY**: `.npz`/`.snp`/`.idv` (use prefix)

## Architecture

### Core Libraries (src/)

- **pyBLUP** - Core statistical engine
  - GWAS implementations (LM, LMM, FarmCPU)
  - QK matrix calculation with memory-optimized chunking
  - PCA computation with randomized SVD
  - Cross-validation utilities

- **gfreader** - Genotype file I/O
  - VCF reader
  - PLINK binary reader (.bed/.bim/.fam)
  - NumPy format support

- **bioplotkit** - Visualization
  - Manhattan and QQ plots
  - PCA visualization (2D and 3D GIF)
  - LD block visualization

### CLI Entry Points (src/script/)

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
| **FarmCPU** | Iterative fixed/random effect alternation | High power with strict false positive control |

### GS Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **GBLUP** | Genomic Best Linear Unbiased Prediction | Baseline prediction |
| **rrBLUP** | Ridge Regression BLUP | Additive genetic value estimation |
| **Random Forest** | Ensemble tree-based prediction | Non-linear effects |
| **SVM** | Support Vector Machine | Complex pattern capture |
| **AdaBoost** | Adaptive Boosting ensemble | High-dimensional data |

### Kinship Methods

- **Method 1 (VanRaden)**: Centered GRM (default)
- **Method 2 (Yang)**: Standardized/weighted GRM

## Conda Packaging

If you want to upload JanusX to conda/bioconda, prepare a recipe with:

- `meta.yaml` (package metadata, dependencies, entry points)
- `build.sh` / `bld.bat` (build steps, call `pip install .`)
- `conda_build_config.yaml` (optional pinning)
- Tests in `meta.yaml` (e.g. `jx -h`, `python -c "import JanusX"`)

Minimal steps:

```bash
# build local recipe in this repo
conda build conda/recipe
```

If you publish to conda-forge, you will also need:
- `conda-smithy` and a feedstock repo
- CI setup for linux/macOS builds

For Bioconda submission, fork `bioconda-recipes`, add `recipes/janusx`,
and run `bioconda-utils build --packages janusx`.

## Python Version

Requires Python 3.9+

## Citation

```bibtex
@software{JanusX,
  title = {JanusX: High-performance GWAS and Genomic Selection Suite},
  author = {MaizeMan-JxFU},
  url = {https://github.com/MaizeMan-JxFU/JanusX}
}
```

## Test Data

Example data in `example/` directory from Parker et al, Nature Genetics, 2016 (via GEMMA project)

## Documentation

For detailed CLI documentation, see [CLI_README.md](./doc/CLI_README.md)
