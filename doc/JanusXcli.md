# JanusX CLI Guide

Version baseline: `v2.0.0`

This guide tracks the current command-line surface in this repository. For exact flags of any single module, always prefer:

```bash
jx <module> -h
```

## 1. Entrypoints and install modes

JanusX is used in two common ways:

- Launcher install: `jx`
- Source or pip install: `python -m janusx.script.JanusX`
- Some Python installs also expose the dispatcher as `jxpy`

The research modules are shared across both modes. The main boundary is lifecycle management:

- Launcher-only flags: `-update`, `-upgrade`, `-list`, `-clean`, `-uninstall`
- Python-dispatcher note: `fastq2vcf` and `fastq2count` run compatibility checks only; use launcher `jx fastq2vcf ...` or `jx fastq2count ...` for the full external pipeline

Quick checks:

```bash
jx -h
jx -v
jx gwas -h
python -m janusx.script.JanusX gs -h
```

## 2. Shared input conventions

### 2.1 Genotype inputs

Most scientific modules accept one of these input families:

- `-vcf`: `.vcf` or `.vcf.gz`
- `-hmp`: `.hmp` or `.hmp.gz`
- `-bfile`: PLINK prefix for `.bed/.bim/.fam`
- `-file`: numeric genotype matrix `.txt/.tsv/.csv/.npy`, or a shared prefix

For large repeated analyses, `-bfile` is usually the best default because packed BED paths can be reused across `grm`, `pca`, `gwas`, `gs`, `fastpop`, `gformat`, and related workflows.

### 2.2 `-file` sidecars

When using `-file prefix`, JanusX expects:

- `prefix.id`: sample IDs in matrix column order
- `prefix.site` or `prefix.bim`: optional site metadata for conversion, LD, plotting, or variant-aware exports

Minimal examples:

```text
# prefix.id
sample_1
sample_2
sample_3
```

```text
# prefix.site
1	3197400	G	A
1	3407393	A	G
1	3492195	G	A
```

### 2.3 Matrix orientation

The CLI and most Rust-backed workflows treat genotype matrices as SNP-major:

- rows: variants
- columns: samples

This matters when you prepare matrices outside the CLI and then feed them back through `-file`.

### 2.4 Phenotype files

For `gwas`, `gs`, and related modules:

- first column: sample ID
- remaining columns: traits or covariates
- `-n/--ncol` is zero-based and excludes the sample ID column

Examples:

```bash
jx gwas -bfile panel -p trait.tsv -lmm -n 0
jx gs   -bfile panel -p trait.tsv -BLUP -cv 5 -n 0,2
jx reml -p effects.tsv -n trait1 -c environment
```

### 2.5 Output and threads

Common flags you will see repeatedly:

- `-o`: output prefix. To write into a directory, pass an existing directory or include a trailing slash (for example, `-o results/`).
- `-prefix`: supported by many modules to set the output basename under the directory given by `-o`.
- `-t`: thread count

Inspect a module-specific log first when debugging unexpected behavior. Most modules emit a dedicated log file under the chosen output prefix or directory.

## 3. Common workflows

### 3.1 Build GRM and PCA

```bash
jx grm -vcf example/mouse_hs1940.vcf.gz -m 1 -o demo/mouse_hs1940
jx pca -k demo/mouse_hs1940 -dim 10 -plot -o demo/mouse_hs1940.pca
```

Notes:

- `jx grm` builds centered (`-m 1`) or standardized (`-m 2`) GRM
- `jx pca -k` consumes the GRM prefix, not the raw `.npy` filename
- `jx pca -rsvd` is the direct genotype-to-PCA route when you do not want to materialize GRM first

### 3.2 Run GWAS

```bash
jx gwas -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -lmm -n 0 -q 3 -o demo -prefix mouse_hs1940
jx gwas -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -lm -n 0 -o demo -prefix mouse_hs1940
jx gwas -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -fvlmm -n 0 -o demo -prefix mouse_hs1940
```

Current model flags:

- `-lm`
- `-lmm`
- `-lmm2` for per-marker Wald and likelihood-ratio results
- `-fvlmm`
- `-splmm` for the approximate GRAMMAR-gamma SparseLMM scan
- `-splmm-exact` for the exact sparse-Cholesky SparseLMM scan
- `-farmcpu`

Useful notes:

- `-k 1` and `-k 2` ask JanusX to build GRM internally; `-k <path>` reuses an external GRM
- `-q` adds leading PCs as Q covariates
- `-c` accepts either a covariate file or a single-site token such as `1:1000`

### 3.3 Run GS and REML

```bash
jx gs -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -BLUP -cv 5 -n 0 -o demo -prefix mouse_hs1940
jx gs -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -RF -ET -GBDT -SVM -ENET -cv 5 -n 0 -o demo -prefix mouse_hs1940

jx reml -p example/rice6048.reml.tsv -n Plant_height -c year,loc -o demo/rice6048
```

`-BLUP` selects GBLUP or rrBLUP automatically based on the sample and marker
counts. Set `GS_BLUP=0` to force GBLUP, `1` to force exact rrBLUP, or `2` to
force PCG rrBLUP.

Current GS model groups:

- automatic BLUP dispatch: `-BLUP`
- Bayesian models: `-BayesA`, `-BayesB`, `-BayesC`, `-BayesR`

`-BayesB` and `-BayesC` optionally accept a marker inclusion probability, for
example `-BayesB 0.05`. BayesB uses a fixed `pi=0.05` when no numeric
argument is given; a numeric argument overrides that fixed value. BayesC
estimates the inclusion probability during sampling when no numeric argument
is given.

`-BayesR` uses the four-component mixture prior
`pi=(0.95,0.03,0.01,0.01)` and `gamma=(0,0.01,0.1,1)`. These two priors are
available through the Python `BayesR` function, but are intentionally not
separate CLI options. With `-save-model`, the effect table includes `pip` and
`component_prob_0` through `component_prob_3`.

Bayesian GS uses a 10000-iteration R-hat upper limit (stability threshold 1.10).
After R-hat converges, the next 1000 iterations are retained as posterior
samples. If it does not converge by the limit, the following 1000 iterations
form a fallback posterior window (maximum 11000 iterations). Production
sampling does not expose a thinning option.
- ML models: `-RF`, `-ET`, `-GBDT`, `-XGB`, `-SVM`, `-ENET`

Useful notes:

- `jx gs -model saved.jxmodel ...` reuses a saved model artifact for prediction or evaluation
- `-ldprune` and `-hash` are available for preprocessing before GS model fitting
- `-debug` prints thread and backend diagnostics for GS

REML uses the first phenotype-table column as the sample/line ID. Fixed effects
come from `-c/--cov`, random nuisance effects from `-rc/--rcov`, discrete
Line×environment terms from `-gxe`, and continuous reaction-norm slopes from
`-gxc`. `-k/--grm` and `-spk/--grm-sparse` are optional and mutually exclusive;
without either, REML reports BLUE/BLUP and logs raw variance components without
narrow-sense h².

### 3.4 Run FastPop

```bash
jx fastpop -vcf example/mouse_hs1940.vcf.gz -k 1..6 -cv -o demo -prefix mouse_hs1940
jx fastpop -vcf example/mouse_hs1940.vcf.gz -k 4 -tag x,1 -o demo -prefix mouse_hs1940
```

Compatibility note: `jx adamixture` is still accepted as an alias.

Accepted `-k` forms:

- single value: `4`
- range: `1..6` or `1:6`
- stepped range: `1..10..2`
- list: `1,3,5`

### 3.5 Convert and merge genotype data

```bash
jx gformat -vcf example/mouse_hs1940.vcf.gz -o demo -prefix mouse_hs1940
jx gformat -vcf example/mouse_hs1940.vcf.gz -fmt plink -o demo -prefix mouse_hs1940
jx gformat -vcf example/mouse_hs1940.vcf.gz --prune 500 10 0.2 -o demo -prefix mouse_hs1940_pruned

jx gmerge -bfile cohort_a cohort_b -fmt plink -o demo -prefix merged
```

Use these modules when you need:

- cross-format conversion: `gformat`
- sample filtering or site extraction: `gformat`
- LD pruning for downstream GWAS or GS: `gformat --prune`
- multi-panel merging: `gmerge`

`gformat` defaults to PLINK BED output when `-fmt` is omitted. Set
`JX_GFORMAT_MAX_OUTPUT_SITES=0` to benchmark filtering/extraction/prune stages
without writing variant rows, or set it to a positive integer to cap BED output.
`gformat --prune` follows PLINK-style window syntax: a bare number is a variant
count window (e.g. `500 50 0.2`), while `kb`/`bp` suffixes request physical
position windows.
For PLINK input/output LD pruning, the default Rust path uses the lower-memory
streaming writer. Set `JX_GFORMAT_PRUNE_STREAMING_IO=0` to force the packed
speed path when memory is not a concern. SIMD prune kernel stats are written to
the `.gformat.log` file by default; set `JX_GFORMAT_PRINT_PRUNE_KERNEL_STATS=1`
to also print them to the terminal.

### 3.6 Post-processing and visualization

```bash
jx postgwas -gwasfile demo/mouse_hs1940.test0.lmm.tsv -manh -qq -o demo/postgwas/
jx postgwas -i demo/mouse_hs1940.test0.lmm.tsv demo/mouse_hs1940.test0.lm.tsv -manh-merge -qq-merge -fontsize 11 -fontstyle arial -o demo/postgwas/
jx postgwas -i demo/mouse_hs1940.test0.lmm.tsv demo/mouse_hs1940.test0.lm.tsv demo/mouse_hs1940.test0.fvlmm.tsv -manh-merge -qq-merge -scatter-size 4 10 -alpha 0.2 0.5 -ylim 2 10 -o demo/postgwas/
jx postgwas -bfile demo/mouse_hs1940 -bimrange 1:3-5 -ldblock-all -o demo/postgwas/

jx postgs -json demo/mouse_hs1940.gs.model/summary.json -o demo/postgs/
```

The main reporting modules are:

- `postgwas`: Manhattan, QQ, LD block, annotation, merged GWAS views
- `postgs`: GS summary plotting from `summary.json`
- `postgarfield`: pseudo-genotype GWAS/post-GWAS workflow
- `postbsa`: BSA filtering, smoothing, and plotting
- `webui`: browser UI for exploring generated results

### 3.7 External pipeline preflight

In Python-dispatcher mode, these commands validate the environment and stop:

```bash
python -m janusx.script.JanusX fastq2vcf --check-only
python -m janusx.script.JanusX fastq2count --check-only
```

Use launcher `jx` when you want the full external workflow to run.

## 4. Module catalog

### 4.1 GWAS

- `grm`: build centered or standardized genomic relationship matrix
- `pca`: principal component analysis from genotype, GRM, or existing PCA output
- `gwas`: LM, LMM, FaST-LMM, or FarmCPU association workflow
- `postgwas`: visualization, annotation, LD block, merged GWAS reporting

### 4.2 GS

- `gs`: genomic selection and model comparison
- `reml`: mixed-model REML/BLUP for phenotype and effect tables
- `postgs`: summarize GS outputs and effect plots

### 4.3 GARFIELD and BSA

- `garfield`: logic-gate or interval-based marker-trait association workflow
- `postgarfield`: pseudo-genotype GWAS and post-processing
- `postbsa`: BSA table processing and figures

### 4.4 Utility and format tools

- `gformat`: convert genotype formats, filter samples, extract sites, prune LD
- `gmerge`: merge multiple genotype panels
- `view`: print `.bin` and `.bin.site` style files as plain text
- `hybrid`: generate pairwise hybrid genotype matrices from parent lists
- `fastpop`: ancestry inference and CVerror scan (`jx adamixture` remains a compatibility alias)
- `tree`: tree workflow entry
- `treeplot`: visualize Newick or GRM-derived trees
- `webui`: start the JanusX web interface

### 4.5 K-mer and external pipeline helpers

- `kmer`: KMC counting or WASTER tree workflow entry
- `kmerge`: merge KMC outputs into a genotype-like matrix
- `fastq2vcf`: FASTQ-to-VCF compatibility check in Python mode; full pipeline in launcher mode
- `fastq2count`: FASTQ-to-count compatibility check in Python mode; full pipeline in launcher mode

### 4.6 Benchmark and simulation

- `sim`: quick simulation workflow
- `simulation`: extended simulation and benchmarking workflow
- `benchmark`: FarmCPU benchmark workflow
- `gblupbench`: GBLUP benchmark workflow
- `garfieldbench`: GARFIELD local-interval benchmark workflow

## 5. Practical notes

- Use `jx <module> -h` as the source of truth for flags. The dispatcher help is intentionally broad; module help is precise.
- Prefer PLINK BED input when you expect to rerun the same cohort many times. Packed BED paths are reused more efficiently than repeated VCF parsing.
- Use `-prefix` deliberately. It makes downstream files easier to chain into `postgwas`, `postgs`, and `webui`.
- For large GS runs, use `-debug` when you need thread policy, BLAS backend, or packed-path diagnostics.
