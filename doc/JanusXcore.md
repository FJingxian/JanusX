# JanusX Core API Guide

Version baseline: `v1.0.24`

This document focuses on Python usage through the `janusx` package. It complements `doc/JanusXcli.md`: the CLI guide explains command-line workflows, while this file explains recommended imports, wrappers, and lower-level kernels.

## 1. Public package map

Use these modules as the main public surface:

| Package | Use it for | Notes |
| --- | --- | --- |
| `janusx.gfreader` | genotype IO, chunk streaming, format conversion, packed BED preparation | best first stop for file-backed data |
| `janusx.assoc` | GWAS-style wrappers and file-mode orchestration | mirrors `jx gwas` |
| `janusx.gs` | GS-style wrappers and result objects | mirrors `jx gs` |
| `janusx.pyBLUP` | direct in-memory association, GRM, BLUP, and ML kernels | lower-level than `assoc` / `gs` |
| `janusx.pyBLUP.bayes` | `BayesA`, `BayesB`, `BayesCpi`, `BAYES` | Bayesian GS imports live here, not at `janusx.pyBLUP` top level |
| `janusx.adamixture` | RSVD, ancestry decomposition, CVerror | mirrors `jx adamixture` |
| `janusx.garfield.logreg` | logical conjunction search on binary features | lightweight Python-facing GARFIELD entry |
| `janusx.gtools` | GFF/BED readers and region queries | annotation-centric helpers |
| `janusx.bioplotkit` | GWAS, PCA, GS, and structure plots | some helpers live in submodules rather than the top-level export list |
| `janusx.pipeline` | `fastq2vcf` / `fastq2count` compatibility checks | preflight only |
| `janusx.ui` | parser/server entrypoints for the web UI | integration-oriented |
| `janusx.janusx` | native Rust extension | useful for backend diagnostics and advanced users |

Avoid importing `janusx.assoc.workflow` and `janusx.gs.workflow` directly for ordinary use. The wrapper packages above are the safer API layer.

## 2. Installation and backend probe

```bash
pip install janusx
# or from source
pip install -e .
```

Backend sanity check:

```python
import janusx
import janusx.janusx as jxrs

print("SGEMM backend:", jxrs.rust_sgemm_backend())
print("EIGH backend:", jxrs.rust_eigh_lapack_backend())
print("BLAS threads:", jxrs.rust_blas_get_num_threads())
```

On macOS, this is the quickest way to confirm whether you are using bundled OpenBLAS or an Accelerate fallback path.

## 3. Genotype IO patterns

### 3.1 Inspect a file and stream chunks

```python
from janusx.gfreader import inspect_genotype_file, load_genotype_chunks

sample_ids, n_sites = inspect_genotype_file(
    "example/mouse_hs1940.vcf.gz",
    snps_only=True,
    maf=0.02,
    missing_rate=0.05,
)
print(len(sample_ids), n_sites)

for geno_chunk, sites in load_genotype_chunks(
    "example/mouse_hs1940.vcf.gz",
    chunk_size=50_000,
    maf=0.02,
    missing_rate=0.05,
    impute=True,
    model="add",
    snps_only=True,
):
    print(geno_chunk.shape)  # (m_chunk, n_samples), SNP-major
    print(sites[0].chrom, sites[0].pos)
    break
```

`load_genotype_chunks()` is the main Python entry for streaming file-backed genotype data. It returns SNP-major blocks and site metadata together.

### 3.2 Save or convert streamed data

```python
from janusx.gfreader import inspect_genotype_file, load_genotype_chunks, save_genotype_streaming

src = "example/mouse_hs1940.vcf.gz"
sample_ids, n_sites = inspect_genotype_file(src)
chunks = load_genotype_chunks(src, chunk_size=50_000)

save_genotype_streaming(
    out="demo/mouse_hs1940",
    sample_ids=sample_ids,
    chunks=chunks,
    fmt="plink",
    total_snps=n_sites,
)
```

This is the package-level equivalent of using `jx gformat` for simple conversion tasks.

### 3.3 Prepare packed BED once for repeated large jobs

```python
from janusx.gfreader import prepare_bed_2bit_packed

packed, missing_rate, maf, std_denom, row_flip, site_keep, n_samples, n_total_sites = prepare_bed_2bit_packed(
    "example/~mouse_hs1940",
    maf_threshold=0.02,
    max_missing_rate=0.05,
    het_threshold=0.02,
    snps_only=True,
)

print(packed.shape, n_samples, n_total_sites)
print(site_keep.sum(), "sites retained")
```

Use this path when you want to reuse filtered packed BED payloads across multiple analyses. `site_keep` maps filtered rows back to original site order.

If you only need raw packed bytes plus per-SNP statistics, `load_bed_2bit_packed()` is the simpler entry.

## 4. Workflow-style wrappers

### 4.1 File-backed GWAS via `janusx.assoc`

```python
from janusx.assoc import AssociationConfig, run_gwas_config

cfg = AssociationConfig(
    genotype="example/~mouse_hs1940",
    phenotype="example/mouse_hs1940.pheno",
    traits=[0],
    maf=0.02,
    geno=0.05,
    het=0.02,
    threads=8,
    qcov=3,
    grm="1",
    out="demo",
    prefix="mouse_hs1940",
)

payload = run_gwas_config(cfg, model_key="lmm")
print(payload["status"])
print(payload["result_files"])
```

Recommended when:

- your data already lives on disk
- you want behavior close to `jx gwas`
- you want JanusX to decide the appropriate workflow branch for file-backed input

`run_gwas_config()` returns a payload `dict`. If you want a typed result object, use `LinearModel(...).lm()`, `.lmm()`, `.fastlmm()`, or `.farmcpu()`.

### 4.2 In-memory GWAS via `LinearModel`

```python
import numpy as np
from janusx.assoc import LinearModel

rng = np.random.default_rng(42)
X_sample_major = rng.integers(0, 3, size=(240, 500), dtype=np.int8)   # samples x SNPs
y = rng.normal(size=240)
cov = rng.normal(size=(240, 2))

model = LinearModel(
    genotype=X_sample_major,
    phenotype=y,
    covariates=cov,
    traits=[0],
    threads=4,
)
res = model.lm(write_files=False)
print(res.ok, res.traits, len(res.summary_rows))
```

Important distinction:

- `LinearModel` matrix-mode is easiest when genotype is sample-major (`n_samples x n_snps`)
- lower-level `janusx.pyBLUP` kernels usually expect SNP-major (`m_snps x n_samples`)

Passing a square genotype matrix is ambiguous and should be avoided.

### 4.3 GS via `janusx.gs`

```python
from janusx.gs import GsConfig, run_gs_config, GenomicSelection

cfg = GsConfig(
    genotype="example/~mouse_hs1940",
    phenotype="example/mouse_hs1940.pheno",
    traits=[0],
    cv=5,
    threads=8,
    rrblup=True,
    gblup_kernels=("a",),
    out="demo",
    prefix="mouse_hs1940",
)

payload = run_gs_config(cfg)
print(payload["status"], payload["methods"])

runner = GenomicSelection(
    "example/~mouse_hs1940",
    "example/mouse_hs1940.pheno",
    traits=[0],
    cv=5,
    threads=8,
)
res = runner.gblup(kernels=("a",))
print(res.ok, res.methods)
```

Use `janusx.gs` when you want the Python equivalent of `jx gs`. This layer is file-oriented and is the recommended entry for GS orchestration.

## 5. Direct model kernels

### 5.1 Low-level association kernels

```python
import numpy as np
from janusx.pyBLUP import LM, LMM, FastLMM

rng = np.random.default_rng(42)
M_snp_major = rng.normal(size=(500, 240)).astype(np.float32)  # SNPs x samples
y = rng.normal(size=240).astype(np.float64)
cov = rng.normal(size=(240, 2)).astype(np.float64)
K = np.eye(240, dtype=np.float64)

lm = LM(y=y, X=cov)
lm_stats = lm.gwas(M_snp_major, threads=4)

lmm = LMM(y=y, X=cov, kinship=K)
lmm_stats = lmm.gwas(M_snp_major, threads=4)

fast = FastLMM(y=y, X=cov, kinship=K)
fast_stats = fast.gwas(M_snp_major, threads=4)
```

This layer is best when you already have in-memory matrices and want to bypass file-mode workflow orchestration.

### 5.2 BLUP, ML, and Bayes kernels

```python
from janusx.pyBLUP import BLUP, MLGS
from janusx.pyBLUP.bayes import BayesA, BayesB, BayesCpi, BAYES
```

Use:

- `BLUP` for direct mixed-model prediction on in-memory matrices
- `MLGS` for ML-based GS in Python
- `BayesA`, `BayesB`, `BayesCpi`, or `BAYES` for Bayesian GS

The Bayesian imports live in `janusx.pyBLUP.bayes`, not in `janusx.pyBLUP` top-level exports.

### 5.3 In-memory GRM helpers

```python
from janusx.pyBLUP import GRM, QK
```

Use these only when your genotype matrix is already in memory and SNP-major. For large file-backed GRM construction, prefer `jx grm`.

Important caveat:

- `janusx.pyBLUP.build_streaming_grm_from_chunks()` is currently not the recommended public path in this Rust-only build mode; it raises `RuntimeError` instead of acting as a production streaming GRM builder
- for large on-disk GRM work, use the CLI `jx grm`
- advanced users can inspect native functions such as `janusx.janusx.grm_stream_bed_f32` and `janusx.janusx.grm_packed_bed_f32`, but those are lower-level interfaces

## 6. ADAMixture, annotation, and plotting

### 6.1 ADAMixture

```python
import logging
from janusx.adamixture import ADAMixtureConfig, rsvd_streaming, train_adamixture

eigvals, eigvecs = rsvd_streaming(
    "example/~mouse_hs1940",
    k=6,
    seed=42,
    power=5,
    tol=0.1,
    snps_only=True,
    maf=0.02,
    missing_rate=0.05,
)
print(eigvals.shape, eigvecs.shape)

cfg = ADAMixtureConfig(
    genotype_path="example/~mouse_hs1940",
    k=4,
    outdir="demo/admixture",
    prefix="mouse_hs1940",
    threads=8,
)
logger = logging.getLogger("admixture")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

P, Q, m, n = train_adamixture(cfg, logger)
print(P.shape, Q.shape, m, n)
```

### 6.2 Annotation and range queries

```python
from janusx.gtools import gffreader, readanno, GFFQuery

gff = gffreader("ref/example.gff3.gz", attr=["ID", "Name"])
query = GFFQuery(gff)
hits = query.query_bimrange("1:2.0-2.5", features=["gene"], attr=["ID", "Name"])
print(hits.head())

anno = readanno("ref/example.gff3.gz")
print(len(anno))
```

### 6.3 GARFIELD logic search

```python
import numpy as np
from janusx.garfield.logreg import backend_available, logreg

if backend_available():
    x = np.random.randint(0, 2, size=(100, 12), dtype=np.uint8)
    y = np.random.randint(0, 2, size=100, dtype=np.uint8)
    res = logreg(x, y, response="binary", max_literals=3, allow_empty=True)
    print(res["expression"], res["score"])
```

### 6.4 Plotting

```python
from janusx.bioplotkit import GWASPLOT, PCSHOW, LDblock
from janusx.bioplotkit.popstructure import plot_admixture_structure
from janusx.bioplotkit.gsplot import plot_accuracy_runtime_scatter
```

Top-level `janusx.bioplotkit` exports the most common entrypoints, but some plotting helpers still live in submodules.

## 7. Pipeline checks and web entrypoints

```python
from janusx.pipeline import run_fastq2vcf_checks, run_fastq2count_checks, format_report
from janusx.ui import build_parser, main

print(format_report(run_fastq2vcf_checks()))
print(format_report(run_fastq2count_checks()))
```

Use `janusx.pipeline` for environment compatibility checks. Use `janusx.ui` when embedding or extending the web interface entrypoint.

## 8. CLI-to-Python mapping

| CLI module | Recommended Python entry |
| --- | --- |
| `jx gwas` | `janusx.assoc.AssociationConfig`, `run_gwas_config`, `LinearModel` |
| `jx gs` | `janusx.gs.GsConfig`, `run_gs_config`, `GenomicSelection` |
| `jx adamixture` | `janusx.adamixture.ADAMixtureConfig`, `rsvd_streaming`, `train_adamixture` |
| `jx gformat` | `janusx.gfreader.load_genotype_chunks`, `save_genotype_streaming` |
| `jx gmerge` | `janusx.gfreader.gmerge.merge` |
| `jx grm` | CLI recommended for file-backed runs; `janusx.pyBLUP.GRM` or `QK` for in-memory matrices |
| `jx pca` | `janusx.adamixture.rsvd_streaming`, `janusx.pyBLUP.QK.PCA()`, or plotting helpers, depending on the workflow |
| `jx fastq2vcf` / `jx fastq2count` | `janusx.pipeline.run_fastq2vcf_checks`, `run_fastq2count_checks` |

## 9. Practical notes

- Keep sample alignment explicit. The safest order is the sample order returned by `inspect_genotype_file()`.
- Remember the orientation split: file-backed streams are SNP-major; `LinearModel` matrix-mode is easiest with sample-major input.
- Use `janusx.gfreader.set_genotype_cache_dir()` or `JANUSX_CACHE_DIR` on shared systems with slow home directories.
- Prefer wrapper packages first. Drop to `janusx.pyBLUP` or `janusx.janusx` only when you already know you need lower-level control.
- When results differ across systems, inspect `rust_sgemm_backend()` and `rust_eigh_lapack_backend()` before debugging the model code itself.
