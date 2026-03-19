# JanusX Core API Guide

Version baseline: `v1.0.14`

This document focuses on Python package usage (`janusx`) and core APIs.
CLI behavior is documented separately in `doc/JanusXcli.md`.

## 1. Install and runtime requirements

## 1.1 Install

```bash
pip install janusx
# or from source repository
pip install -e .
```

## 1.2 Runtime note about extension module

Core compute and IO acceleration depend on the compiled extension module:

- `janusx.janusx` (PyO3 Rust extension)

If you run from source tree without building/installing the extension, imports that depend on Rust kernels will fail.

## 2. Public module map

Main Python-facing modules:

- `janusx.gfreader`
- `janusx.pyBLUP`
- `janusx.adamixture`
- `janusx.bioplotkit`
- `janusx.gtools`

Typical imports:

```python
from janusx.gfreader import (
    inspect_genotype_file,
    load_genotype_chunks,
    save_genotype_streaming,
    set_genotype_cache_dir,
    load_bed_2bit_packed,
)

from janusx.pyBLUP import (
    GRM,
    QK,
    LM,
    LMM,
    FastLMM,
    farmcpu,
    BLUP,
    MLGS,
    build_streaming_grm_from_chunks,
)

from janusx.adamixture import ADAMixtureConfig, train_adamixture, rsvd_streaming
from janusx.bioplotkit import GWASPLOT, LDblock, PCSHOW, plot_haplotype
from janusx.gtools import gffreader, readanno, GFFQuery
```

## 3. Data conventions

Genotype layout in core kernels is SNP-major:

- `M.shape == (m_snps, n_samples)`

Phenotype/covariate layout is sample-major:

- `y.shape == (n_samples,)` or `(n_samples, 1)`
- `X.shape == (n_samples, p)`

Always align phenotype/covariates to genotype sample order returned by `inspect_genotype_file()`.

## 4. `janusx.gfreader` API

## 4.1 Inspect genotype metadata

```python
sample_ids, n_sites = inspect_genotype_file("example/mouse_hs1940.vcf.gz")
print(len(sample_ids), n_sites)
```

Supported inputs:

- VCF (`.vcf/.vcf.gz`)
- HMP (`.hmp/.hmp.gz`)
- PLINK prefix
- FILE mode (`prefix.npy/.txt/.tsv/.csv` with `prefix.id`)

## 4.2 Stream genotype chunks

```python
for geno_chunk, sites in load_genotype_chunks(
    "example/mouse_hs1940.vcf.gz",
    chunk_size=50_000,
    maf=0.02,
    missing_rate=0.05,
    impute=True,
    model="add",        # add/dom/rec/het
    snps_only=True,
):
    print(geno_chunk.shape, sites[0].chrom, sites[0].pos)
    break
```

`sites` elements are `SiteInfo` records from Rust extension.

## 4.3 Stream output writing / conversion

```python
sample_ids, n_sites = inspect_genotype_file("example/mouse_hs1940.vcf.gz")
chunks = load_genotype_chunks("example/mouse_hs1940.vcf.gz", chunk_size=50_000)

save_genotype_streaming(
    out="out/panel",
    sample_ids=sample_ids,
    chunks=chunks,
    fmt="plink",          # plink/vcf/hmp/auto
    total_snps=n_sites,
)
```

## 4.4 Cache controls

Set cache directory explicitly:

```python
set_genotype_cache_dir("/path/to/cache")
```

Environment variable equivalent:

- `JANUSX_CACHE_DIR`

## 4.5 BED packed access

For PLINK-focused low-memory workflows:

```python
packed, miss_rate, maf, std_denom, n_samples = load_bed_2bit_packed("geno_prefix")
```

## 5. `janusx.pyBLUP` API

## 5.1 Build GRM from streamed chunks

```python
from janusx.pyBLUP import build_streaming_grm_from_chunks

sample_ids, _ = inspect_genotype_file("example/mouse_hs1940.vcf.gz")
chunks = load_genotype_chunks("example/mouse_hs1940.vcf.gz", chunk_size=100_000)
K, stats = build_streaming_grm_from_chunks(chunks, n_samples=len(sample_ids), method=1)
print(K.shape, stats.eff_m)
```

## 5.2 Association models (`LM`, `LMM`, `FastLMM`)

```python
import numpy as np
from janusx.pyBLUP import LM, LMM, FastLMM

y = np.random.randn(len(sample_ids), 1)
X = None

lm = LM(y, X)
lmm = LMM(y, X, kinship=K.copy())
flmm = FastLMM(y, X, kinship=K.copy())

for geno_chunk, sites in load_genotype_chunks("example/mouse_hs1940.vcf.gz", chunk_size=20_000):
    out = lmm.gwas(geno_chunk, threads=8)
    # out columns include beta/se/pwald/plrt depending on method
    break
```

## 5.3 FarmCPU

```python
from janusx.pyBLUP import farmcpu

# M: SNP-major genotype, y: phenotype, chrlist/poslist aligned to SNP rows
res = farmcpu(y=y, M=M, X=None, chrlist=chr_list, poslist=pos_list, threads=8)
```

## 5.4 BLUP and MLGS

Marker-matrix BLUP (`janusx.pyBLUP.BLUP`):

```python
from janusx.pyBLUP import BLUP

model = BLUP(y=y_train, M=M_train, cov=None, kinship=1, log=False)
y_pred = model.predict(M_test)
```

Machine-learning GS (`janusx.pyBLUP.MLGS`):

```python
from janusx.pyBLUP import MLGS

ml = MLGS(y=y_train, M=M_train, method="rf", cv=3, n_jobs=8, fit_on_init=True)
y_pred = ml.predict(M_test)
```

## 5.5 Multi-random-effect REML BLUP

Use `janusx.pyBLUP.blup.BLUP` for multi-term REML model fitting:

```python
from janusx.pyBLUP.blup import BLUP as REMLBLUP

m = REMLBLUP(y=y, X=X, Z=[Z1], G=[K1], maxiter=100, progress=False)
print(m.beta.ravel())
print(m.theta)
print(m.var)
```

## 6. `janusx.adamixture` API

## 6.1 Streaming RSVD

```python
from janusx.adamixture import rsvd_streaming

eigvals, eigvecs = rsvd_streaming(
    "geno.vcf.gz",
    k=8,
    seed=42,
    power=5,
    tol=1e-1,
    snps_only=True,
    maf=0.02,
    missing_rate=0.05,
)
```

## 6.2 Full ADAMixture training

```python
import logging
from janusx.adamixture import ADAMixtureConfig, train_adamixture

logger = logging.getLogger("admx")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

cfg = ADAMixtureConfig(
    genotype_path="geno.vcf.gz",
    k=8,
    outdir="out",
    prefix="cohort",
    threads=8,
    solver="adam-em",   # auto/adam/adam-em
    max_iter=500,
)

P, Q, m, n = train_adamixture(cfg, logger)
```

## 7. `janusx.bioplotkit` API

## 7.1 Manhattan + QQ

```python
import matplotlib.pyplot as plt
from janusx.bioplotkit import GWASPLOT

plotter = GWASPLOT(df=res_df, chr="CHR", pos="BP", pvalue="P", compression=True)
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
plotter.manhattan(ax=axes[0], threshold=5e-8)
plotter.qq(ax=axes[1], ci=95)
plt.tight_layout()
```

## 7.2 LD block

```python
import numpy as np
import matplotlib.pyplot as plt
from janusx.bioplotkit import LDblock

C = np.corrcoef(np.random.randn(200, 80))
fig, ax = plt.subplots(figsize=(6, 4))
LDblock(C, ax=ax, cmap="Greys")
```

## 7.3 PCA scatter

```python
from janusx.bioplotkit import PCSHOW

pc = PCSHOW(pc_df)
pc.pcplot(x="PC1", y="PC2", group="group")
```

## 7.4 Haplotype plot

```python
from janusx.bioplotkit import plot_haplotype

ax, summary = plot_haplotype(
    phenotype=pheno_df,
    genotype=geno_df_or_chunks,
    mode="continuous",
    min_haplotype_n=30,
)
```

## 8. `janusx.gtools` API

## 8.1 Read and query annotation

```python
from janusx.gtools import gffreader, readanno, GFFQuery

gff = gffreader("Zea_mays.gff3.gz", attr=["ID", "Name"])
q = GFFQuery(gff)
subset = q.query_bimrange("1:2.0-2.5", features=["gene"], attr=["ID", "Name"])
anno = readanno("Zea_mays.gff3.gz")
```

`query_bimrange` uses Mb range strings (`chr:start-end` or `chr:start:end`).

## 9. Practical guidance

- Keep genotype matrix SNP-major for pyBLUP/association kernels.
- Align all phenotype/covariate rows to genotype sample order before modeling.
- Use chunked loading for large inputs (`load_genotype_chunks`).
- Set `JANUSX_CACHE_DIR` to a writable fast path on shared/HPC systems.

## 10. Related docs

- CLI guide: `doc/JanusXcli.md`
- Project overview: `README.md`