# JanusX Core API Guide

Version baseline: `v1.0.20`

This guide focuses on Python usage (`janusx`) and how it maps to Rust kernels (`janusx.janusx`).
CLI commands are documented separately in `doc/JanusXcli.md`.

## 1. Start Here

If you are new to JanusX, read in this order:

1. `Section 2` to understand the core data flow.
2. `Section 3` to run the minimum working example.
3. `Section 4` to copy an end-to-end workflow for your task.
4. `Section 5-8` as reference when you need details.

If you already know the project:

1. Use `Section 4` (workflow recipes).
2. Use `Section 8` (quick reference table).
3. Use `Section 6` when you need Rust-to-Python tracing.

## 2. Core Mental Model

JanusX usually follows this execution path:

1. Read genotype metadata and stream SNP chunks from `janusx.gfreader`.
2. Build matrices or statistics (`GRM`, fixed-effect scans, LMM/FaST-LMM) in `janusx.pyBLUP`.
3. Run task-specific inference (`adamixture`, `garfield`, tree/BSA workflows).
4. Render or summarize outputs (`bioplotkit`, `gtools`, `ui`).

### 2.1 Data shape conventions

- Genotype blocks are SNP-major: `M.shape == (m_snps, n_samples)`.
- Phenotype is sample-major: `y.shape == (n_samples,)` or `(n_samples, 1)`.
- Covariates are sample-major: `X.shape == (n_samples, p)`.
- Always align phenotype/covariates to sample order from `inspect_genotype_file()`.

### 2.2 Main Python packages

- `janusx.gfreader`: genotype IO, chunk streaming, format conversion, packed BED access.
- `janusx.pyBLUP`: GRM, LM/LMM/FaST-LMM, FarmCPU, BLUP, MLGS, Bayes prediction.
- `janusx.adamixture`: RSVD + ADAMixture ancestry inference.
- `janusx.gtools`: GFF/BED readers and range query helpers.
- `janusx.bioplotkit`: Manhattan/QQ/LD/PCA/haplotype/admixture visualization.
- `janusx.garfield`: logic-gate style AND/NOT search.
- `janusx.pipeline`: runtime compatibility checks for external pipelines.
- `janusx.ui`: parser/server entry points.

## 3. Minimum Working Example

### 3.1 Install and backend check

```bash
pip install janusx
# or from source
pip install -e .
```

```python
import janusx
import janusx.janusx as jxrs

print("janusx imported")
print("rust extension loaded:", jxrs is not None)
```

If `import janusx.janusx` fails, build/install the Rust extension first.

### 3.2 Inspect + stream one chunk

```python
from janusx.gfreader import inspect_genotype_file, load_genotype_chunks

geno_path = "example/mouse_hs1940.vcf.gz"
sample_ids, n_sites = inspect_genotype_file(geno_path)
print("samples:", len(sample_ids), "sites:", n_sites)

for geno_chunk, sites in load_genotype_chunks(
    geno_path,
    chunk_size=50_000,
    maf=0.02,
    missing_rate=0.05,
    impute=True,
    model="add",
    snps_only=True,
):
    print("chunk shape:", geno_chunk.shape)
    print("first site:", sites[0].chrom, sites[0].pos)
    break
```

## 4. Workflow Recipes (With Examples)

### 4.1 End-to-end GWAS (stream -> GRM -> LMM -> Manhattan/QQ)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from janusx.gfreader import inspect_genotype_file, load_genotype_chunks
from janusx.pyBLUP import build_streaming_grm_from_chunks, LMM
from janusx.bioplotkit import GWASPLOT

geno_path = "example/mouse_hs1940.vcf.gz"
sample_ids, _ = inspect_genotype_file(geno_path)
n = len(sample_ids)

# 1) Build GRM from streamed chunks
chunks_for_grm = load_genotype_chunks(
    geno_path, chunk_size=80_000, maf=0.02, missing_rate=0.05
)
K, grm_stats = build_streaming_grm_from_chunks(
    chunks_for_grm,
    n_samples=n,
    method=1,
)
print("GRM:", K.shape, "effective SNPs:", grm_stats.eff_m)

# 2) Prepare phenotype/covariates in sample order
rng = np.random.default_rng(42)
y = rng.normal(size=(n, 1)).astype(np.float64)
X = None

# 3) Run LMM scan chunk by chunk
lmm = LMM(y=y, X=X, kinship=K.astype(np.float64))
rows = []
for geno_chunk, sites in load_genotype_chunks(geno_path, chunk_size=20_000):
    beta_se_p = lmm.gwas(geno_chunk, threads=8)  # shape (m_chunk, 4)
    for i, s in enumerate(sites):
        rows.append(
            {
                "CHR": s.chrom,
                "BP": int(s.pos),
                "P": float(beta_se_p[i, 2]),  # pwald
            }
        )

res_df = pd.DataFrame(rows)

# 4) Plot Manhattan + QQ
plotter = GWASPLOT(df=res_df, chr="CHR", pos="BP", pvalue="P", compression=True)
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
plotter.manhattan(ax=axes[0], threshold=5e-8)
plotter.qq(ax=axes[1], ci=95)
plt.tight_layout()
plt.show()
```

### 4.2 Streaming write and format conversion

```python
from janusx.gfreader import (
    inspect_genotype_file,
    load_genotype_chunks,
    save_genotype_streaming,
)

src = "example/mouse_hs1940.vcf.gz"
sample_ids, n_sites = inspect_genotype_file(src)
chunks = load_genotype_chunks(src, chunk_size=50_000)

# auto infers output format from path suffix/prefix
save_genotype_streaming(
    out="out/panel",  # PLINK prefix
    sample_ids=sample_ids,
    chunks=chunks,
    fmt="auto",
    total_snps=n_sites,
)
```

### 4.3 Merge multiple genotype datasets

```python
from janusx.gfreader.gmerge import merge

stats, stats_dict = merge(
    inputs=["data/cohort_a", "data/cohort_b.vcf.gz"],  # PLINK prefix + VCF
    out="out/merged",
    out_fmt="plink",
    maf=0.01,
    geno=0.10,
)
print(stats_dict["n_sites_written"], stats_dict["n_samples_total"])
```

### 4.4 Ancestry inference with ADAMixture

```python
import logging
from janusx.adamixture import ADAMixtureConfig, rsvd_streaming, train_adamixture

geno_path = "example/mouse_hs1940.vcf.gz"

# Optional: inspect principal components first
eigvals, eigvecs = rsvd_streaming(
    geno_path,
    k=8,
    seed=42,
    power=5,
    tol=1e-1,
    snps_only=True,
    maf=0.02,
    missing_rate=0.05,
)
print("top eigvals:", eigvals[:5])

logger = logging.getLogger("admx")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

cfg = ADAMixtureConfig(
    genotype_path=geno_path,
    k=8,
    outdir="out/admx",
    prefix="cohort",
    threads=8,
    solver="adam-em",  # auto | adam | adam-em
    max_iter=500,
)

P, Q, m, n = train_adamixture(cfg, logger)
print("P:", P.shape, "Q:", Q.shape, "m:", m, "n:", n)
```

### 4.5 Annotation range query with gtools

```python
from janusx.gtools import gffreader, readanno, GFFQuery

gff_df = gffreader("ref/Zea_mays.gff3.gz", attr=["ID", "Name"])
query = GFFQuery(gff_df)

# Mb-range string: chr:start-end
genes = query.query_bimrange("1:2.0-2.5", features=["gene"], attr=["ID", "Name"])
print(genes.head())

anno_df = readanno("ref/Zea_mays.gff3.gz")
print("normalized rows:", len(anno_df))
```

### 4.6 Logic-gate search (GARFIELD backend)

```python
import numpy as np
from janusx.garfield.logreg import backend_available, logreg

if not backend_available():
    raise RuntimeError("janusx.janusx backend is not available")

rng = np.random.default_rng(0)
x = rng.integers(0, 2, size=(100, 12), dtype=np.uint8)  # n_samples x n_features
y = rng.integers(0, 2, size=(100,), dtype=np.uint8)

res = logreg(
    x,
    y,
    response="binary",
    max_literals=3,
    allow_empty=True,
)
print(res["expression"], res["score"])
```

### 4.7 Pipeline compatibility check

```python
from janusx.pipeline import run_fastq2vcf_checks, run_fastq2count_checks, format_report

r1 = run_fastq2vcf_checks()
r2 = run_fastq2count_checks()

print(format_report(r1))
print(format_report(r2))
```

## 5. API Reference By Package

### 5.1 `janusx.gfreader`

Use this package when genotype IO is your primary bottleneck.

- `inspect_genotype_file(path_or_prefix) -> (sample_ids, n_sites)`
- `load_genotype_chunks(...) -> iterator[(geno_chunk, sites)]`
- `save_genotype_streaming(out, sample_ids, chunks, fmt=...)`
- `set_genotype_cache_dir(cache_dir)`
- `load_bed_2bit_packed(prefix) -> (packed, missing_rate, maf, std_denom, n_samples)`
- `janusx.gfreader.gmerge.merge(...)` for dataset merging.

### 5.2 `janusx.pyBLUP`

Use this package for GRM construction, association scans, and prediction.

- GRM APIs: `build_streaming_grm_from_chunks`, `QK`, `GRM`.
- Association APIs: `LM`, `LMM`, `FastLMM`, `FEM`, `lmm_reml`, `lmm_reml_null`, `fastlmm_reml`, `fastlmm_reml_null`, `fastlmm_assoc_chunk`, `farmcpu`.
- Prediction APIs: `BLUP` (from `pyBLUP.mlm`), `MLGS`.
- Bayes APIs: `janusx.pyBLUP.bayes.BayesA`, `BayesB`, `BayesCpi`, `BAYES`.

### 5.3 `janusx.adamixture`

Use this package for population structure and ancestry decomposition.

- `ADAMixtureConfig`
- `rsvd_streaming`
- `train_adamixture`
- `evaluate_adamixture_cverror`

### 5.4 `janusx.gtools`

Use this package for genome annotation loading and range filtering.

- `gffreader`, `bedreader`, `readanno`
- `GFFQuery.query_range`, `GFFQuery.query_bimrange`
- `chrom_sort_key`, `natural_tokens`

### 5.5 `janusx.bioplotkit`

Use this package for publication-facing charts.

- `GWASPLOT` (`manhattan`, `qq`)
- `LDblock`
- `PCSHOW` (`pcplot`, `pcplot3D_gif`)
- `plot_haplotype`
- `plot_admixture_structure`

### 5.6 `janusx.garfield`

Use this package for logical conjunction search on binary features.

- `backend_available`
- `logreg` (returns `indices`, `expression`, `xcombine`, `score`)

### 5.7 `janusx.pipeline` and `janusx.ui`

- `janusx.pipeline`: external toolchain checks (`run_fastq2vcf_checks`, `run_fastq2count_checks`, `format_report`)
- `janusx.ui`: parser/server entry (`build_parser`, `main`)

## 6. Rust-to-Python Mapping (Complete)

This appendix helps with code tracing and backend debugging.

### 6.1 IO and parsing kernels

- `src/io/gfcore.rs`
- Purpose: low-level parsing for `FAM/BIM/VCF/HMP/TXT/NPY/BIN`, SNP row preprocessing, cache detection, iterators.
- Consumed by: `janusx.gfreader` high-level readers/writers.

- `src/io/gfreader.rs`
- Rust exports: `SiteInfo`, `BedChunkReader`, `VcfChunkReader`, `HmpChunkReader`, `TxtChunkReader`, `PlinkStreamWriter`, `VcfStreamWriter`, `HmpStreamWriter`, `count_vcf_snps`, `count_hmp_snps`, `load_bed_2bit_packed`, `load_bed_u8_matrix`, `load_site_info`, `gfd_packbits_from_dosage_block`.
- Python entries: `load_genotype_chunks`, `inspect_genotype_file`, `save_genotype_streaming`, `load_bed_2bit_packed`.

- `src/io/gmerge.rs`
- Rust exports: `merge_genotypes`, `convert_genotypes`, `PyMergeStats`, `PyConvertStats`.
- Python entries: `janusx.gfreader.gmerge.merge`, script workflows (`gmerge`, `gformat`).

- `src/io/gwasio.rs`
- Rust export: `load_gwas_triplet_fast`.
- Python consumers: `bioplotkit.manhanden.GWASPLOT`, `script.postgwas`, `ui.render`.

### 6.2 Math and scoring kernels

- `src/math/bitwise.rs`
- Rust exports: `popcount_py`, `and_popcount_py`, `bitand_assign_py`, `bitor_into_py`, `bitnot_masked_py`.
- Used by packed-genotype and bitset-heavy workflows.

- `src/stats/score.rs`
- Rust exports: `score_binary_ba_py`, `score_binary_mcc_py`, `score_binary_ba_mcc_batch_py`, `score_cont_mean_diff_py`, `score_cont_corr_py`, `score_cont_mean_diff_corr_batch_py`.

- `src/stats/beam.rs`
- Rust exports: `beam_search_and_binary_mcc_bin_py`, `beam_search_and_binary_mcc_bin_indices_py`, `beam_scan_windows_binary_mcc_bin_py`.

- `src/stats/logreg.rs`
- Rust export: `fit_best_and_not_py`.
- Python entry: `janusx.garfield.logreg.logreg`.

### 6.3 Association and prediction kernels

- `src/stats/assoc.rs`
- Fixed-effect kernels: `glmf32`, `glmf32_full`, `glmf32_packed`.
- LMM kernels: `lmm_reml_null_f32`, `lmm_assoc_chunk_f32`, `lmm_reml_chunk_f32`, `lmm_reml_chunk_from_snp_f32`, `lmm_assoc_chunk_from_snp_f32`.
- FaST-LMM kernels: `fastlmm_reml_null_f32`, `fastlmm_reml_chunk_f32`, `fastlmm_assoc_chunk_f32`, `fastlmm_assoc_packed_f32`.
- GRM/FarmCPU/REML helpers: `grm_packed_bed_f32`, `grm_packed_f32`, `grm_packed_f32_with_stats`, `bed_packed_row_flip_mask`, `bed_packed_decode_rows_f32`, `bed_packed_decode_stats_f64`, `cross_grm_times_alpha_packed_f64`, `packed_malpha_f64`, `farmcpu_rem_dense`, `farmcpu_rem_packed`, `farmcpu_super_dense`, `farmcpu_super_packed`, `ai_reml_null_f64`, `ai_reml_multi_f64`, `ml_loglike_null_f32`, `rust_sgemm_backend`.
- Python entries: `janusx.pyBLUP.assoc` APIs (`LM/LMM/FastLMM`, `farmcpu`, etc.).

- `src/stats/lmm.rs`
- Rust exports: `fastlmm_reml_null_f32`, `fastlmm_reml_chunk_f32`.
- Python wrappers: FaST-LMM paths in `pyBLUP.assoc`.

- `src/stats/bayes.rs`
- Rust exports: `bayesa`, `bayesb`, `bayescpi`.
- Python wrappers: `janusx.pyBLUP.bayes`.

### 6.4 ADAMixture, BSA, tree kernels

- `src/stats/rsvd.rs`
- Rust export: `py_rsvd_packed_subset`.
- Python wrappers: `rsvd_streaming`, ADAMixture training initialization.

- `src/stats/adamixture.rs`
- RSVD/multiply exports: `admx_rsvd_stream`, `admx_rsvd_stream_sample`, `admx_rsvd_power_step_inplace`, `admx_multiply_at_omega`, `admx_multiply_a_omega`, `admx_multiply_at_omega_inplace`, `admx_multiply_a_omega_inplace`.
- EM/Adam exports: `admx_allele_frequency`, `admx_loglikelihood`, `admx_loglikelihood_f32`, `admx_rmse_f32`, `admx_rmse_f64`, `admx_kl_divergence`, `admx_map_q_f32`, `admx_map_p_f32`, `admx_em_step`, `admx_em_step_inplace`, `admx_em_step_inplace_f32`, `admx_adam_update_p`, `admx_adam_update_q`, `admx_adam_update_p_inplace`, `admx_adam_update_q_inplace`, `admx_adam_update_p_inplace_f32`, `admx_adam_update_q_inplace_f32`, `admx_adam_optimize_f32`, `admx_set_threads`.

- `src/stats/bsa.rs`
- Rust export: `preprocess_bsa`.
- Python consumer: `janusx.script.postbsa`.

- `src/stats/tree.rs`
- Rust exports: `geno_chunk_to_alignment_u8`, `geno_chunk_to_alignment_u8_siteinfo`, `geno_chunk_to_alignment_u8_sites`, `nj_newick_from_alignment_u8`, `nj_newick_from_distance_matrix`, `ml_newick_from_alignment_u8`.
- Python consumers: tree/haplotype script pipelines.

## 7. Main Return Objects

- `SiteInfo`: `chrom`, `pos`, `ref_allele`, `alt_allele`.
- `PyMergeStats`: merge statistics with counts and per-input site status.
- `PyConvertStats`: conversion statistics.
- `StreamingGrmStats`: `eff_m`, `varsum`, `used_syrk`.
- `PipelineCompatReport`: pipeline check summary (`checks`, `passed_count`, etc.).
- `ADAMixtureConfig`: ADAMixture training configuration dataclass.

## 8. Quick Reference Table

| Requirement | Python entry point | Rust core |
| --- | --- | --- |
| Stream VCF/HMP/PLINK/TXT | `janusx.gfreader.load_genotype_chunks` | `BedChunkReader` / `VcfChunkReader` / `HmpChunkReader` / `TxtChunkReader` |
| Read packed BED | `janusx.gfreader.load_bed_2bit_packed` | `load_bed_2bit_packed` |
| Write VCF/HMP/PLINK | `janusx.gfreader.save_genotype_streaming` | `PlinkStreamWriter` / `VcfStreamWriter` / `HmpStreamWriter` |
| Linear-model GWAS | `janusx.pyBLUP.assoc.FEM` / `LM.gwas()` | `glmf32` / `glmf32_full` / `glmf32_packed` |
| LMM GWAS | `janusx.pyBLUP.assoc.lmm_reml` / `LMM.gwas()` | `lmm_reml_chunk_f32` / `lmm_reml_null_f32` |
| FaST-LMM | `janusx.pyBLUP.assoc.fastlmm_reml` / `FastLMM.gwas()` | `fastlmm_reml_chunk_f32` / `fastlmm_reml_null_f32` |
| FarmCPU | `janusx.pyBLUP.assoc.farmcpu` | `farmcpu_rem_*` / `farmcpu_super_*` |
| Bayes prediction | `janusx.pyBLUP.bayes.BayesA/BayesB/BayesCpi` | `bayesa` / `bayesb` / `bayescpi` |
| Ancestry inference | `janusx.adamixture.train_adamixture` | `admx_*` family |
| GFF/gene queries | `janusx.gtools.GFFQuery` | pure Python |
| GWAS plotting | `janusx.bioplotkit.GWASPLOT` | pure Python (`load_gwas_triplet_fast` for fast summary loading) |
| Logic-gate search | `janusx.garfield.logreg.logreg` | `fit_best_and_not_py` |

## 9. Practical Tips and Common Pitfalls

- Keep genotype matrices SNP-major before calling `pyBLUP` kernels.
- Keep phenotype/covariate arrays in exact sample order from genotype metadata.
- Prefer chunked loading for large cohorts (`load_genotype_chunks`).
- Set a writable fast cache path with `JANUSX_CACHE_DIR` on shared systems.
- If performance is unstable, check both Python thread settings and Rust-side thread counts.
- If results look inconsistent after merge/convert, verify sample IDs and site alignment first.

## 10. Related Docs

- CLI guide: `doc/JanusXcli.md`
- Project overview: `README.md`
