# JanusX Project Map

Version snapshot: `1.0.21` (from `pyproject.toml` / `Cargo.toml`)

This file is the fast entrypoint for understanding the whole repository: project layers, directory ownership, command entry differences, and capability map.

## 1. Read Order

For first-time onboarding:

1. Read this file (`doc/PROJECT_MAP.md`) for global map.
2. Read [`doc/JanusXcli.md`](./JanusXcli.md) for command usage.
3. Read [`doc/JanusXcore.md`](./JanusXcore.md) for Python API and Rust mapping.

## 2. Architecture in One View

JanusX is a mixed Rust + Python toolkit with three layers:

1. Rust kernel layer (`src/`): high-performance genotype IO, association, Bayes, admixture, tree, bitwise/score kernels via PyO3.
2. Python analysis layer (`python/janusx/`): API modules (`gfreader`, `pyBLUP`, `adamixture`, plotting, pipeline checks) and CLI scripts.
3. Launcher/runtime layer (`launcher/`): runtime bootstrap, self-update, DLC tooling, and full `fastq2vcf/fastq2count` pipeline orchestration.

## 3. Repository Structure

| Path | Role | Notes |
| --- | --- | --- |
| `src/lib.rs` | PyO3 export hub | Registers all Rust symbols exposed as `janusx.janusx` |
| `src/io/` | Genotype IO kernels | `gfcore`, `gfreader`, `gmerge`, `gwasio`, `vcfout` |
| `src/stats/` | Core statistics kernels | `assoc`, `lmm`, `bayes`, `adamixture`, `beam`, `tree`, `bsa`, `score`, `logreg` |
| `src/math/` | Numeric helpers | `linalg`, `brent`, `bitwise` |
| `python/janusx/gfreader/` | Streaming genotype API | VCF/HMP/PLINK/TXT/NPY read/write and conversion |
| `python/janusx/pyBLUP/` | GWAS/GS API | GRM, LM/LMM/FaST-LMM, FarmCPU, BLUP, MLGS, Bayes wrappers |
| `python/janusx/adamixture/` | ADAMixture training API | RSVD + EM/Adam workflows |
| `python/janusx/bioplotkit/` | Visualization API | Manhattan/QQ/LD/PCA/structure/GS plots |
| `python/janusx/gtools/` | Annotation tools | GFF/BED parsing and range query |
| `python/janusx/pipeline/` | Environment checks | Fast preflight for fastq pipelines |
| `python/janusx/ui/` | Web UI service | Parser + server/render |
| `python/janusx/script/` | CLI module implementations | Dispatcher plus per-module command logic |
| `launcher/src/main.rs` | Rust launcher entry | Module whitelist, update/upgrade, runtime boot |
| `launcher/src/fastq2vcf/` | FASTQ->VCF pipeline | Step orchestration and command construction |
| `launcher/src/fastq2count/` | FASTQ->count pipeline | RNA-seq pipeline orchestration |
| `launcher/src/dlc.rs` | DLC/toolchain manager | Conda/docker runtime and tool route management |
| `scripts/` | Build/release tooling | custom build backend, version sync |
| `example/` | Example inputs + derived demo outputs | Quick reproducible sample data |
| `test/` | Generated test artifacts | Mostly output snapshots, not a full unit-test suite |
| `bioconda-recipes/` | Packaging manifests | bioconda metadata and build scripts |
| `todo/` | Experimental/archived work | Not part of main runtime path |
| `doc/` | Documentation | CLI/API guide + this map |

## 4. Entrypoints and Module Availability

### 4.1 Entrypoint modes

| Mode | Typical command | Source | Notes |
| --- | --- | --- | --- |
| Rust launcher install | `jx ...` | `launcher/src/main.rs` | Supports runtime update/DLC/full pipelines |
| Python package entry | `jxpy ...` | `python/janusx/script/JanusX.py` | Script dispatcher |
| pip/source `jx` | `jx ...` | `pyproject.toml` script entry | Maps to Python dispatcher, not launcher binary |
| Direct module script | `python -m janusx.script.<module> ...` | individual script files | Needed for unregistered scripts like `beam` |

### 4.2 Module support matrix

Legend: `Y` supported, `N` unsupported, `Check` means preflight only.

| Module | Launcher `jx` | Python dispatcher `jxpy` | Notes |
| --- | --- | --- | --- |
| `grm` | Y | Y | GRM construction |
| `pca` | Y | Y | PCA + optional RSVD |
| `gwas` | Y | Y | LM/LMM/FaST-LMM/FarmCPU |
| `postgwas` | Y | Y | plotting + LD/annotation |
| `gs` | Y | Y | genomic selection models |
| `reml` | Y | Y | REML-BLUP |
| `garfield` | Y | Y | RF-based feature selection |
| `postgarfield` | Y | Y | GARFIELD->GWAS postflow |
| `postbsa` | Y | Y | BSA post-processing |
| `adamixture` | Y | Y | ancestry inference |
| `gformat` | Y | Y | genotype conversion/filter |
| `gmerge` | Y | Y | genotype merging |
| `hybrid` | Y | Y | hybrid genotype construction |
| `webui` | Y | Y | UI server entry |
| `sim` | Y | Y | quick simulator |
| `simulation` | Y | Y | extended simulation |
| `kmer` | Y | Y | KMC count / WASTER tree workflow |
| `tree` | Y | Y | currently NJ-focused exposed path |
| `benchmark` | Y | Y | FarmCPU benchmark |
| `fastq2vcf` | Y | Check | Python entry does compatibility check only |
| `fastq2count` | Y | Check | Python entry does compatibility check only |
| `kmerge` | N | Y | KMC DB merge to `.bin` |
| `view` | N | Y | decode `.bin` / `.bin.site` |
| `treeplot` | N | Y | toytree visualizer |
| `gblupbench` | N | Y | multi-engine GBLUP benchmark |
| `beam` | N | N | script exists; run direct `python -m janusx.script.beam` |

## 5. Capability Map

### 5.1 Genotype IO and conversion

- Streaming read/write across `vcf/hmp/plink/txt/npy`.
- Merge and conversion pipelines (`gmerge`, `gformat`).
- Packed BED optimized kernels and bit-level transforms.
- K-mer binary matrix generation (`kmerge`) and text decode (`view`).

Main code: `src/io/*`, `python/janusx/gfreader/*`, `python/janusx/script/{gformat,gmerge,kmerge,view}.py`

### 5.2 Association and selection

- GWAS models: `LM`, `LMM`, `FaST-LMM`, `LRLMM`, `FarmCPU`.
- GS models: `GBLUP`, `adBLUP`, `rrBLUP` (exact + mini-batch AdamW backend), `BayesA/B/Cpi`, ML models.
- REML variance estimation and BLUP output.

Main code: `src/stats/{assoc,lmm,bayes}.rs`, `python/janusx/pyBLUP/*`, `python/janusx/script/{gwas,gs,reml}.py`

### 5.3 Population structure and tree

- ADAMixture (`adamixture` module and API).
- PCA/RSVD and structure plots.
- NJ tree inference from genotype/FASTA and GRM-tree plotting.
- WASTER reference-free tree path in `kmer --tree`.

Main code: `src/stats/{adamixture,rsvd,tree}.rs`, `python/janusx/adamixture/*`, `python/janusx/script/{tree,treeplot,kmer}.py`

### 5.4 Post-analysis visualization

- Manhattan/QQ/LD block/annotation overlays.
- PCA/admixture/haplotype/popstructure plotting.
- BSA and GARFIELD post-processing visuals.

Main code: `python/janusx/bioplotkit/*`, `python/janusx/script/{postgwas,postgarfield,postbsa}.py`

### 5.5 Pipelines and runtime orchestration

- Full launcher pipelines:
  - `fastq2vcf`: `fastp -> bwa-mem2/samblaster -> gatk -> beagle -> merge`
  - `fastq2count`: `fastp -> hisat2 -> featureCounts -> FPKM/TPM`
- DLC runtime/toolchain management for external bioinformatics commands.
- Python-side compatibility probes for fastq pipelines.

Main code: `launcher/src/{main,pipeline,dlc,fastq2vcf,fastq2count}.rs`, `python/janusx/pipeline/*`

## 6. Rust <-> Python Binding Surface

Single binding entry: `src/lib.rs` -> module name `janusx.janusx`.

Representative exported groups:

- IO: chunk readers/writers, site metadata, packed BED loaders, merge/convert kernels.
- Stats: GWAS kernels, GRM, LMM/FaST-LMM, FarmCPU, Bayes, ADAMixture, tree kernels.
- Utility: bitwise ops, score ops, beam/logreg helpers.

For function-level mapping, see `doc/JanusXcore.md` section `6`.

## 7. Build, Packaging, Release

- Rust/PyO3 build config: `Cargo.toml`, `build.rs`, `pyproject.toml`, `scripts/build_backend.py`
- OpenBLAS and fallback logic handled in `build.rs`.
- KMC native binding can be prebuilt during wheel build (`JANUSX_PREBUILD_KMC_BIND` path).
- Version sync script: `scripts/sync_version_from_tag.py` updates core package + launcher + bioconda recipes.

## 8. Current Documentation Gaps That Were Closed

This map and updated CLI docs explicitly cover previously under-documented areas:

- entrypoint differences between launcher `jx` and pip/source `jx`
- module availability mismatch (`kmerge/view/treeplot/gblupbench/beam`)
- k-mer and tree-adjacent workflows beyond standard GWAS/GS
- benchmark module family scope

## 9. Maintenance Checklist (when adding/changing modules)

When a new CLI/API capability is added, update in this order:

1. `python/janusx/script/JanusX.py` module registry (if dispatcher exposure is required).
2. `launcher/src/main.rs` `KNOWN_MODULES` / `MODULE_LIST_ENTRIES` (if launcher exposure is required).
3. `doc/JanusXcli.md` module section.
4. `doc/JanusXcore.md` API mapping (if Python/Rust API surface changed).
5. `doc/PROJECT_MAP.md` support matrix and structure notes.
