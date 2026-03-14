# JanusX Core (Python API Usage)

Version baseline: `v1.0.13`  
Python package: `janusx`

This document focuses on **Python package calling patterns** only (import + code usage).  
CLI details are intentionally omitted.

## 1. Install and Import

```bash
pip install janusx
# or in repo root:
pip install -e .
```

Recommended Python imports:

```python
from janusx.gfreader import (
    inspect_genotype_file,
    load_genotype_chunks,
    save_genotype_streaming,
)

from janusx.pyBLUP import (
    build_streaming_grm_from_chunks,
    LM,
    LMM,
    FastLMM,
    BLUP,      # marker-matrix BLUP (mlm.BLUP)
    MLGS,
)

from janusx.pyBLUP.blup import BLUP as REMLBLUP  # multi-random-effect REML-BLUP

from janusx.bioplotkit import GWASPLOT, LDblock, PCSHOW, plot_haplotype
from janusx.gtools import gffreader, readanno, GFFQuery
```

## 2. Core Data Conventions

### 2.1 Genotype input types

Supported by `inspect_genotype_file()` / `load_genotype_chunks()`:

```text
VCF:   *.vcf / *.vcf.gz
HMP:   *.hmp / *.hmp.gz
PLINK: prefix + .bed/.bim/.fam
FILE:  prefix/path + .npy/.txt/.tsv/.csv, with sibling .id
```

For FILE/prefix mode:

```text
required: prefix.id + one matrix file (prefix.npy or prefix.txt/tsv/csv)
optional: prefix.site or prefix.bim
```

### 2.2 Matrix shape

Most JanusX core genotype APIs use **SNP-major** shape:

```text
M.shape = (m_snps, n_samples)
```

Phenotype/covariate side uses sample-major:

```text
y.shape = (n_samples,) or (n_samples, 1)
X.shape = (n_samples, p)
```

## 3. Genotype Streaming (`janusx.gfreader`)

### 3.1 Inspect input quickly

```python
geno_path = "example/mouse_hs1940.vcf.gz"
sample_ids, n_sites = inspect_genotype_file(geno_path, snps_only=True)
print(len(sample_ids), n_sites)
```

### 3.2 Iterate SNP chunks

```python
for geno_chunk, sites in load_genotype_chunks(
    geno_path,
    chunk_size=50_000,
    maf=0.02,
    missing_rate=0.05,
    impute=True,
    model="add",           # add/dom/rec/het
    snps_only=True,
):
    # geno_chunk: (m_chunk, n_samples), float32
    # sites: list[SiteInfo], fields: chrom, pos, ref_allele, alt_allele
    print(geno_chunk.shape, sites[0].chrom, sites[0].pos)
    break
```

### 3.3 Stream write / format conversion

```python
sample_ids, n_sites = inspect_genotype_file(geno_path)
chunks = load_genotype_chunks(geno_path, chunk_size=50_000, maf=0.0, missing_rate=1.0)

# fmt in {"plink", "vcf", "hmp"}; auto-infer supported
save_genotype_streaming(
    out="out/panel",          # plink prefix
    sample_ids=sample_ids,
    chunks=chunks,
    fmt="plink",
    total_snps=n_sites,
)
```

## 4. GWAS Kernels from Python (`janusx.pyBLUP`)

### 4.1 Build GRM from stream

```python
import numpy as np

sample_ids, _ = inspect_genotype_file(geno_path)
chunks = load_genotype_chunks(geno_path, chunk_size=100_000, maf=0.02, missing_rate=0.05)

K, stats = build_streaming_grm_from_chunks(
    chunks,
    n_samples=len(sample_ids),
    method=1,  # 1=centered, 2=standardized
)
print(K.shape, stats.eff_m)
```

### 4.2 LM/LMM/FastLMM chunk scan

```python
import numpy as np
import pandas as pd

# y must align to sample order in genotype
# y: (n,1), X: (n,p)
y = np.random.randn(len(sample_ids), 1)
X = None

# LM
lm = LM(y, X)

# LMM/FastLMM require kinship
lmm = LMM(y, X, kinship=K.copy())
flmm = FastLMM(y, X, kinship=K.copy())

rows = []
for geno_chunk, sites in load_genotype_chunks(geno_path, chunk_size=50_000, maf=0.02, missing_rate=0.05):
    out = lmm.gwas(geno_chunk, threads=8)  # columns: beta, se, pwald, plrt
    rows.append(pd.DataFrame({
        "CHR": [s.chrom for s in sites],
        "BP": [s.pos for s in sites],
        "beta": out[:, 0],
        "se": out[:, 1],
        "P": out[:, 2],
    }))

res = pd.concat(rows, ignore_index=True)
```

## 5. Genomic Selection APIs

### 5.1 Marker-matrix BLUP (`janusx.pyBLUP.BLUP`)

```python
import numpy as np
from janusx.pyBLUP import BLUP

# M: (m_snps, n_samples)
M_train = np.random.randint(0, 3, size=(5000, 300)).astype(float)
y_train = np.random.randn(300, 1)

model = BLUP(y=y_train, M=M_train, cov=None, kinship=1, log=False)

M_test = np.random.randint(0, 3, size=(5000, 80)).astype(float)
y_pred = model.predict(M_test)
print(y_pred.shape)
```

### 5.2 MLGS (`RF/ET/GBDT/XGB/SVM/ENET`)

```python
from janusx.pyBLUP import MLGS

ml = MLGS(
    y=y_train,
    M=M_train,
    method="rf",      # rf/et/gbdt/xgb/svm/enet
    cv=3,
    n_jobs=8,
    fit_on_init=True,
)

pred = ml.predict(M_test)
print(pred.shape)
```

### 5.3 Multi-random-effect REML-BLUP (`janusx.pyBLUP.blup.BLUP`)

```python
import numpy as np
from janusx.pyBLUP.blup import BLUP as REMLBLUP

n = 200
y = np.random.randn(n, 1)
X = np.random.randn(n, 2)  # fixed effects (intercept auto-added)

# Example random terms
Z_id = np.eye(n)
K_add = np.random.randn(n, n)
K_add = (K_add + K_add.T) / 2
K_add += np.eye(n) * 1e-3

m = REMLBLUP(
    y=y,
    X=X,
    Z=[Z_id],
    G=[K_add],
    maxiter=100,
    progress=False,
)

print("beta:", m.beta.ravel())
print("theta:", m.theta)  # variance components + residual
print("var:", m.var)      # scaled variance components
```

## 6. Plotting APIs (`janusx.bioplotkit`)

### 6.1 Manhattan + QQ

```python
import matplotlib.pyplot as plt
from janusx.bioplotkit import GWASPLOT

# df must contain chromosome/position/p-value columns
plotter = GWASPLOT(df=res, chr="CHR", pos="BP", pvalue="P", compression=True)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
plotter.manhattan(ax=axes[0], threshold=5e-8)
plotter.qq(ax=axes[1], ci=95)
plt.tight_layout()
```

### 6.2 LD block

```python
import numpy as np
import matplotlib.pyplot as plt
from janusx.bioplotkit import LDblock

C = np.corrcoef(np.random.randn(200, 80))
fig, ax = plt.subplots(figsize=(6, 4))
LDblock(C, ax=ax, cmap="Greys")
plt.show()
```

### 6.3 PCA scatter

```python
from janusx.bioplotkit import PCSHOW

# pc_df example columns: PC1, PC2, group
pc = PCSHOW(pc_df)
fig, ax = plt.subplots(figsize=(6, 5))
pc.pcplot(x="PC1", y="PC2", group="group", ax=ax)
```

### 6.4 Haplotype plot

```python
from janusx.bioplotkit import plot_haplotype

# phenotype: DataFrame (sample index + one phenotype column)
# genotype: DataFrame or chunk iterator from load_genotype_chunks
ax, summary = plot_haplotype(
    phenotype=pheno_df,
    genotype=geno_df_or_chunks,
    draw_bar=True,
    draw_matrix=True,
    draw_letters=True,
    min_haplotype_n=30,
    mode="continuous",  # or "binomial"
)
print(summary.head())
```

## 7. Annotation Readers (`janusx.gtools`)

```python
from janusx.gtools import gffreader, readanno, GFFQuery

# Parse GFF/GFF3
gff = gffreader("Zea_mays.gff3.gz", attr=["ID", "Name"])

# Indexed range query
q = GFFQuery(gff)
genes = q.query_bimrange("1:2000000-2500000", features=["gene"], attr=["ID", "Name"])

# Convert to JanusX unified annotation schema
anno = readanno("Zea_mays.gff3.gz")
```

## 8. Practical Tips

1. Keep genotype as **SNP-major** `(m, n)` for pyBLUP kernels.
2. Always align phenotype/covariates to `inspect_genotype_file()` sample order.
3. For shared/HPC paths, set cache dir explicitly:

```python
import os
os.environ["JANUSX_CACHE_DIR"] = "/path/to/writable_cache"
```

4. For `-file` style numeric genotype input, ensure sidecar `.id` exists.
5. Prefer chunked processing (`load_genotype_chunks`) for large cohorts.

## 9. Related Docs

1. CLI documentation: `doc/JanusXcli.md`
2. Project overview: `README.md`
