# JanusX

[CLI Guide](./doc/JanusXcli.md) | [Core API Guide](./doc/JanusXcore.md) | [Zea Eureka](https://mp.weixin.qq.com/s/jl3h2DPRC21l8QJ0WxzXDA)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) ![License](https://img.shields.io/badge/License-AGPLv3-blue.svg)

## Overview

JanusX (Joint Association and Novel Utility for Selection) is a GWAS and genomic selection toolkit that combines:

- Rust-accelerated kernels (PyO3 extension)
- Python analysis modules
- A Rust launcher (`jx`) for runtime/toolchain management and pipeline orchestration

```text
       _                      __   __
      | |                     \ \ / /
      | | __ _ _ __  _   _ ___ \ V /
  _   | |/ _` | '_ \| | | / __| > <
 | |__| | (_| | | | | |_| \__ \/ . \
  \____/ \__,_|_| |_|\__,_|___/_/ \_\ Tools for GWAS and GS
  ---------------------------------------------------------
```

**Main capabilities**:

- GWAS: `LM`, `LMM`, `FastLMM`, `FarmCPU`
- Genomic selection: `GBLUP`, `adBLUP`, `rrBLUP`, `BayesA/B/Cpi`, and ML models (`RF/ET/GBDT/XGB/SVM/ENET`)
- Streaming genotype IO for VCF/HMP/PLINK/TXT/NPY
- Post-analysis workflows: `postgwas`, `postgarfield`, `postbsa`
- Utility workflows: `grm`, `pca`, `gformat`, `gmerge`, `hybrid`, `adamixture`, `webui`, `sim`, `simulation`
- Launcher pipelines: `fastq2vcf`, `fastq2count`

---

## Installation

### Quick installation: Python with uv (Recommend)

* Linux | MacOS
```bash
curl -fsSL https://raw.githubusercontent.com/FJingxian/JanusX/main/scripts/install.sh | sh
```

* Windows
```powershell
Set-ExecutionPolicy RemoteSigned -scope CurrentUser
irm https://raw.githubusercontent.com/FJingxian/JanusX/main/scripts/install.ps1 | iex
```

### Option A: Python package install

```bash
pip install janusx==1.0.24
```

### Option B: Conda / Bioconda

Recommended Bioconda channel order:

```bash
conda create -n janusx \
  --channel conda-forge \
  --channel bioconda \
  janusx
```

---

## Quick start

### 1) GWAS

```bash
# Estimate variance once in NULL model, similar with EMMAX. (Fast, recommand)
jx gwas -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -fvlmm -o test
# Estimate variance for every snp, similar with GEMMA. (Exact)
jx gwas -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -lmm -o test
# Linear mixed model with sparse GRM, similar with fastGWA. (Fast, grammar gamma)
jx gwas -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -splmm-approx -o test
# FarmCPU (Fast, and more sites)
jx gwas -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -farmcpu -o test
```

<p align="center">
  <img src="./doc/mouse_hs1940.test0.add.lmm.svg" alt="overview" />
</p>

### 2) Post-GWAS

```bash
jx postgwas -gwasfile test/mouse_hs1940.test0.add.lmm.tsv -manh -qq -thr 1e-6 -o testpost
```

<p align="center">
  <img src="./doc/ldblock.png" alt="ldblock" />
</p>

### 3) Genomic selection

```bash
# BLUP method
# n≤15,000 GBLUP
# n>15,000 & m≤15,000 rrBLUP
# n>15,000 & m>15,000 rrBLUP with PCG (Jacobi)
jx gs -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -BLUP -BayesA -BayesB -BayesCpi -o test -cv 5
```

```text
* Genomic Selection for trait: test0
Train size: 1410, Test size: 530, EffSNPs: 8960
** BLUP
✔︎ Cross-validation ...Finished [0.8s]
✔︎ Fitting ...Finished [0.3s]
✔︎ Predicting ...Finished [0.0s]
** BayesA
✔︎ Cross-validation ...Finished [17.9s]
✔︎ Fitting ...Finished [4.1s]
✔︎ Predicting ...Finished [0.0s]
...
------------------------------------------------------------
Fold Method     Pearsonr Spearmanr R2     time(s)  Best
1    BLUP       0.704    0.675     0.493  0.198    
1    BayesA     0.709    0.680     0.493  3.507    
...
------------------------------------------------------------
```

<p align="center">
  <img src="./doc/mouse_hs1940.test0.gs.XGB.svg" alt="gsoverview" />
</p>

### 4) Get module help

```bash
jx -h
jx <module> -h
```

**See full usages in [CLI Guide](./doc/JanusXcli.md).**

---

## Module map

**Genome-wide Association Studies (GWAS)**:

- `grm`
- `pca`
- `gwas`
- `postgwas` (Visualization, `manh` `qq` `ldblock`)
- `adamixture` (Based on https://github.com/AI-sandbox/ADAMIXTURE.git)

**Genomic Selection (GS)**:

- `gs`
- `postgs` (Visualization)
- `reml` (Estimation of broaden heritability and blup values)

**GARFIELD**:

- `garfield` (Based on https://github.com/heroalone/Garfield)
- `postgarfield`

**Utility**:

- `gformat` (Conversion between genotype data formats, support fast splicing/filtering/prune)
- `gmerge` (Merge genotype between samples)
- `gstats` (State freq/het/missing/ldscore of genotype)

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

---

## License

This project is licensed under **GNU Affero General Public License v3.0** (AGPL-3.0-or-later). See [LICENSE](./LICENSE).
