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

- GWAS: `LM`, `LMM`, `FastLMM`, `LRLMM`, `FarmCPU`
- Genomic selection: `GBLUP`, `adBLUP`, `rrBLUP`, `BayesA/B/Cpi`, and ML models (`RF/ET/GBDT/XGB/SVM/ENET`)
- Streaming genotype IO for VCF/HMP/PLINK/TXT/NPY
- Post-analysis workflows: `postgwas`, `postgarfield`, `postbsa`
- Utility workflows: `grm`, `pca`, `gformat`, `gmerge`, `hybrid`, `adamixture`, `webui`, `sim`, `simulation`
- Launcher pipelines: `fastq2vcf`, `fastq2count`

---

## `jx` & `jxpy`

**JanusX provides two command entry styles**:

- `jx` (Rust launcher): manages runtime/update/toolchain and supports all launcher modules
- `jxpy` (Python package entry): runs Python-side modules directly

**Practical difference**:

- `fastq2vcf` and `fastq2count` are launcher-only (`jx`)
- launcher-only flags (`-update/-upgrade/-list/-clean/-uninstall`) are not available in `jxpy`

---

## Installation

### Option A: launcher install (recommended for end users)

Download installer assets from [Releases](https://github.com/FJingxian/JanusX/releases):

Then run installer:

```bash
# Linux
./JanusX-vX.Y.Z-linux-x86_64.run

# macOS
./JanusX-vX.Y.Z-darwin-universal.command
```

On Windows, double click or run the `.exe` installer.

After install:

```bash
jx -v
jx -list module
```

### Option B: Python package install (API-first / development)

```bash
pip install janusx
# command entry: jxpy (version>=1.0.14) or jx (version<1.0.14)
```

- If you only need a turnkey CLI environment with pipeline tooling, prefer launcher install.
- Enhanced Rich-styled CLI help is optional: `pip install "janusx[cli]"`.

### Option C: Conda / Bioconda

Recommended Bioconda channel order:

```bash
conda create -n janusx \
  --channel conda-forge \
  --channel bioconda \
  --strict-channel-priority \
  janusx
```

- JanusX keeps `rich-argparse` as an optional CLI enhancement so the core runtime does not require that conda-forge-only package at install time.
- Even so, Bioconda officially recommends keeping `conda-forge` above `bioconda`, because many Bioconda packages rely on the broader conda-forge ecosystem.

---

## BLAS backend and threads

- OpenBLAS is available on macOS/Linux/Windows.
- JanusX now applies `-t/--thread` to BLAS thread env vars (`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, etc.) in major analysis CLIs.
- For strongest cross-platform numerical/performance consistency, prefer a conda-forge OpenBLAS environment.

Example (OpenBLAS environment):

```bash
conda create -n jx_openblas -c conda-forge python=3.14 numpy scipy "libblas=*=*openblas"
conda activate jx_openblas
```

Default behavior is: prefer OpenBLAS, warn when unavailable, and automatically
fallback to platform-priority backends.

---

## Quick start

### 1) GWAS

```bash
jx gwas -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -lmm -o test
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
jx gs -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -GBLUP -cv 5 -o testgs
```

```text
* Genomic Selection for trait: test0
Train size: 1410, Test size: 530, EffSNPs: 8960
✔︎ GBLUP ...Finished [3.5s]
✔︎ adBLUP ...Finished [10.1s]
✔︎ BayesA ...Finished [32.6s]
✔︎ BayesB ...Finished [34.2s]
✔︎ BayesCpi ...Finished [32.0s]
✔︎ RF ...Finished [1m46s]
✔︎ XGB ...Finished [9m27s]
✔︎ SVM ...Finished [2m05s]
✔︎ ENET ...Finished [9.7s]
Fold Method     Pearsonr Spearmanr     R² h²/PVE time(secs)
   1 GBLUP         0.704     0.671  0.493  0.610      0.531
   1 adBLUP        0.717     0.679  0.512  0.694      2.341
   1 BayesA        0.721     0.695  0.514  0.714      5.307
   1 BayesB        0.722     0.693  0.517  0.667      5.522
   1 BayesCpi      0.699     0.671  0.476  0.672      4.971
   1 RF            0.709     0.702  0.471  0.468      3.055
   1 XGB           0.754     0.733  0.564  0.563      5.231
   1 SVM           0.703     0.664  0.490  0.485      9.027
   1 ENET          0.720     0.704  0.514  0.517      0.274
```

<p align="center">
  <img src="./doc/mouse_hs1940.test0.gs.XGB.svg" alt="gsoverview" />
</p>

**See full usages in [CLI Guide](./doc/JanusXcli.md).**

### 4) Python thin wrappers

```python
from janusx.assoc import LinearModel
from janusx.gs import GenomicSelection

# GWAS (file mode; internally reuses janusx.script.gwas)
lm = LinearModel(
    genotype="test/bench4k5k",
    phenotype="test/bench4k5k.pheno.txt",
    traits=[0],
    threads=4,
    out="tmp_assoc",
    prefix="bench",
)
assoc_res = lm.lm(log=False)

# GS (file mode; internally reuses janusx.script.gs)
gs = GenomicSelection(
    genotype="test/bench4k5k",
    phenotype="test/bench4k5k.pheno.txt",
    traits=[0],
    cv=2,
    threads=4,
    out="tmp_gs",
    prefix="bench",
)
gs_res = gs.gblup(log=False)
```

### 5) Get module help

```bash
jx <module> -h
```

---

## Module map

**Genome-wide Association Studies (GWAS)**:

- `grm`
- `pca`
- `gwas`
- `postgwas` (Visualization, `manh` `qq` `ldblock`)

**Genomic Selection (GS)**:

- `gs`
- `reml` (Estimation of broaden heritability and blup values)

**[GARFIELD](https://github.com/heroalone/Garfield)**:

- `garfield`
- `postgarfield`

**Bulk Segregation Analysis (BSA)**:

- `postbsa` (Visualization, after BSA pipeline)

**Pipeline and utility**:

- `fastq2count` (RNAseq pipeline, launcher-only)
- `fastq2vcf` (BSA/Reseq pipeline, launcher-only)
- `adamixture` (Based on [ADAMIXTURE](https://github.com/AI-sandbox/ADAMIXTURE.git))
- `hybrid` (Generate F1 genotype base on Parents)
- `gformat` (Conversion between genotype data formats)
- `gmerge` (Merge genotype between samples)
- `webui` (Visualization, beta version)

**Benchmark**:

- `sim`
- `simulation`

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
