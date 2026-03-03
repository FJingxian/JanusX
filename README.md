# JanusX

[简体中文](./doc/README_zh.md) | [English](./README.md) | [Zea Eureka](https://mp.weixin.qq.com/s/jl3h2DPRC21l8QJ0WxzXDA)

JanusX is a high-performance toolkit for quantitative genetics. It combines Rust-accelerated kernels (PyO3) with Python CLI workflows for GWAS, genomic selection (GS), post-analysis visualization, and variant-processing pipelines.

## Overview

- Unified CLI entry: `jx`
- Core GWAS models: `LM`, `LMM`, `FarmCPU`
- Core GS models: `GBLUP`, `rrBLUP`, `BayesA`, `BayesB`, `BayesCpi`
- Streaming genotype reader for VCF/PLINK/TXT with low-memory workflows
- Integrated post-analysis tools: `postgwas`, `postbsa`, `postgarfield`
- Additional utilities: `grm`, `pca`, `gmerge`, `fastq2vcf`, `sim`, `simulation`

## Installation

Requirements:

- Python `>=3.9`
- Rust toolchain (only needed when local wheel is unavailable and build from source is required)

### PyPI (recommended)

```bash
pip install -U janusx
jx -h
```

### Latest GitHub version

```bash
pip install --upgrade --force-reinstall --no-cache-dir git+https://github.com/FJingxian/JanusX.git
jx -h
```

### Docker (source image build)

```bash
git clone https://github.com/FJingxian/JanusX.git
cd JanusX
docker build -t janusx:latest .
docker run --rm janusx:latest -h
```

### In-place update for an installed JanusX

```bash
jx --update
```

## CLI Modules

### Genome-wide Association Studies (GWAS)

- `grm`: build genomic relationship matrix from genotype (`-vcf` or `-bfile`)
- `pca`: PCA for population structure from genotype, GRM prefix, or existing PCA prefix
- `gwas`: run genome-wide association analysis (`LM`, `LMM`, `fastLMM`, `FarmCPU`)
- `postgwas`: post-process GWAS results (Manhattan/QQ/annotation/merge/LD views)

### Genomic Selection (GS)

- `gs`: genomic prediction and model-based selection

### GARFIELD

- `garfield`: random-forest based marker-trait association
- `postgarfield`: summarize and visualize GARFIELD outputs

### Bulk Segregation Analysis (BSA)

- `postbsa`: post-process and visualize BSA results

### Pipeline and Utility

- `fastq2vcf`: variant-calling pipeline from FASTQ to VCF
- `gmerge`: merge genotype/variant datasets

### Benchmark

- `sim`: quick simulation workflow
- `simulation`: extended simulation and benchmarking workflow

For full options, run:

```bash
jx <module> -h
```

## Quick Start

### 1) GWAS

```bash
jx gwas -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -lmm -plot -o test
```

### 2) Post-GWAS (single result)

```bash
jx postgwas -gwasfile test/mouse_hs1940.test0.lmm.tsv -manh -qq -threshold 1e-6 -o testpost
```

### 3) Post-GWAS merge plot (multiple GWAS files in one figure)

```bash
jx postgwas \
  -gwasfile testb/mouse_hs1940.test0.add.lmm.tsv \
  -merge testb/mouse_hs1940.test0.dom.lmm.tsv \
  -merge testb/mouse_hs1940.test0.het.lmm.tsv \
  -merge testb/mouse_hs1940.test0.rec.lmm.tsv \
  -manh -qq -o testpost
```

### 4) Genomic Selection

```bash
jx gs -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -GBLUP -cv 5 -plot -o testgs
```

### 5) Post-BSA

```bash
jx postbsa -file QBproject/5.table/all.tsv -b1 JS -b2 JR -window 1 -format pdf -o QBproject/5.table
```

### 6) FASTQ to VCF pipeline

```bash
jx fastq2vcf -r /path/ref.fa -i /path/fastq_dir -w /path/workdir -b nohup -m 2
```

## Input File Formats

### Phenotype file

Tab-delimited. First column is sample ID, remaining columns are traits.

```text
sample	trait1	trait2
id1	10.5	0.85
id2	12.3	1.00
id3	8.6	0.92
```

### Genotype files

- VCF: `.vcf` or `.vcf.gz` (`-vcf`)
- PLINK: `.bed/.bim/.fam` prefix (`-bfile`)
- Text genotype matrix (`-file`, selected modules):
  - header row is sample IDs
  - subsequent rows are SNP-major numeric genotype values

Example (text matrix):

```text
id1	id2	id3	id4
0	1	2	0
2	2	1	0
0	0	1	1
```

### GWAS optional matrices

- GRM (`-k/--grm` in `gwas`): numeric matrix aligned to genotype sample order
- Q/PC covariate (`-q/--qcov` in `gwas`): PC count or covariate matrix file
- Covariate (`-c/--cov` in `gwas`): numeric matrix aligned to genotype sample order

All matrix files should be numeric-only, space/tab-delimited, without row/column headers.

### GRM/PCA file conventions (updated)

- `jx grm` accepts genotype input only: `-vcf` or `-bfile`
- `jx grm` outputs:
  - default text mode: `{prefix}.grm.txt` and `{prefix}.grm.txt.id`
  - with `--npy`: `{prefix}.grm.npy` and `{prefix}.grm.npy.id`
- `jx pca` inputs (mutually exclusive):
  - genotype: `-vcf` or `-bfile` (compute PCA from genotype)
  - GRM prefix: `-k/--grm <prefix>` (expects `{prefix}.grm.id` and `{prefix}.grm.txt` or `{prefix}.grm.npy`)
  - existing PCA prefix: `-q/--qcov <prefix>` (expects `{prefix}.eigenval` and `{prefix}.eigenvec`, visualization-only mode)

## Common Output Naming

- `gwas`: `{prefix}.{trait}.{model}.tsv` and optional `{prefix}.{trait}.{genetic_model}.{model}.svg`
- `postgwas`: `{prefix}.manh.{format}`, `{prefix}.qq.{format}`, `{prefix}.{thr}.anno.tsv`
- `postgwas` merge mode: `{prefix}.merge.manh.{format}`, `{prefix}.merge.qq.{format}`
- `postbsa`:
  - `{prefix}.{bulk1}vs{bulk2}.snpidx.{format}`
  - `{prefix}.{bulk1}vs{bulk2}.bsa.{format}`
  - `{prefix}.{bulk1}vs{bulk2}.smooth.tsv`
  - `{prefix}.{bulk1}vs{bulk2}.thr.tsv`
- `gs`: `{prefix}.{trait}.gs.tsv`

## Python API (selected)

```python
from janusx.gfreader import load_genotype_chunks

for geno, sites in load_genotype_chunks("example/mouse_hs1940.vcf.gz", chunk_size=50000):
    # geno: (m, n) float32 genotype chunk
    # sites: variant metadata list
    pass
```

```python
from janusx.gtools.wgcna import cor, adj, tom, cluster
```

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

## License

MIT License. See [LICENSE](./LICENSE).
