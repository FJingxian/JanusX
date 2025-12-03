# GTOOLS

[English](./README.md) | [简体中文(推荐)](./doc/README_zh.md)

## Project Overview

gtools is a high-performance toolkit for Genome-Wide Association Studies (GWAS) and genomic selection, built on mixed linear models (MLM). It provides significant performance improvements over tools like GEMMA, GCTA, and rMVP, especially in multi-threaded computation.

## Development Setup

### Installation

```bash
git clone https://github.com/MaizeMan-JxFU/gtools.git
# Linux/macOS
cd gtools; sh ./install.sh
# # Windows
# .\install.bat
```

The install scripts use `uv` for dependency management and create a virtual environment in `.venv/`.

### Running the CLI

```bash
# Linux/macOS
./gtools <module> [options]

# Windows
.\gtools.bat <module> [options]
```

### Available Modules

- `gwas` - Mixed linear model GWAS analysis
- `postGWAS` - Visualization and annotation
- `transanno` - Annotation genome version migration
- `gformat` - Genotype format conversion
- `grm` - Genetic relationship matrix calculation
- `pca` - Principal component analysis

### Example Commands

```bash
# GWAS with VCF input
gtools gwas --vcf example/mouse_hs1940.vcf.gz --pheno example/mouse_hs1940.pheno --out test

# GWAS with PLINK format
gtools gwas --bfile genotypes --pheno phenotypes.txt --out results --grm 1 --qcov 3 --thread 8

# Visualize GWAS results
gtools postGWAS --file test/test0.assoc.tsv --threshold 1e-6
```

![manhanden&qq](./fig/test0.png "Simple visualization")

Test data in example is from [genetics-statistics/GEMMA](https://github.com/genetics-statistics/GEMMA), published in [Parker et al, Nature Genetics, 2016](https://doi.org/10.1038/ng.3609)

## Architecture

### Core Libraries (src/)

- **pyBLUP** - Core statistical engine
  - `gwas.py` - GWAS class implementing mixed linear model with REML optimization
  - `QK.py` - Q matrix (population structure) and K matrix (kinship) calculation with memory-optimized chunking
  - `QK2.py` - Alternative QK implementation
  - `QC.py` - Quality control functions (MAF, missing rate filters)
  - `mlm.py` - Mixed linear model utilities
  - `pca.py` - PCA computation (includes randomized SVD for large datasets)
  - `kfold.py` - Cross-validation utilities

- **gfreader** - Genotype file I/O
  - `base.py` - Readers for VCF, PLINK binary (.bed/.bim/.fam), HapMap, and numpy formats
  - Supports genotype conversion between formats

- **bioplotkit** - Visualization
  - `manhanden.py` - Manhattan and QQ plots
  - `LDBlock.py` - LD block visualization
  - `gffplot.py` - GFF/annotation plotting
  - `pcshow.py` - PCA visualization (uses Plotly)

### CLI Entry Points (module/)

Each module corresponds to a CLI command. The launcher script (`gtools.bat`/`gtools`) dispatches to `module/<name>.py`.

### Key Algorithms

**Mixed Linear Model**: Uses SVD decomposition of the kinship matrix to simplify variance computation, with Brent's method for REML parameter optimization. Lambda (variance ratio) is the single parameter being optimized.

**Kinship Methods**: VanRanden (default), GEMMA-style methods (gemma1, gemma2), Pearson correlation

**PCA**: Supports both exact SVD and randomized SVD for large datasets

## File Formats

**Phenotype file**: Tab-delimited, first column = sample ID, subsequent columns = phenotypes

| samples | pheno_name |
| :-----: | :--------: |
| indv1   | value1     |
| indv2   | value2     |

**Supported genotype formats**: VCF (.vcf, .vcf.gz), PLINK binary (.bed/.bim/.fam), numpy archives (.npz/.snp/.idv)

## Python Version

Requires Python3

## Test Data

Example data in `example/` directory from Parker et al, Nature Genetics, 2016 (via GEMMA project)
