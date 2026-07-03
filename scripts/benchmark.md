# Benchmark Reproducibility Note

## Main-text paragraph

Tool-specific parameters, command lines, thread settings, input preprocessing, runtime definitions, and peak RSS measurement procedures are provided in Methods S3. For transparency and direct reproducibility, we consolidated all benchmark workflows into a single shell script that records software versions and executes the exact comparison commands used for each analysis, including BLUP-engine benchmarking across JanusX, rrBLUP, sommer, BLUPF90, BLUPF90-APY, and HIBLUP via `jx gblupbench`; Bayesian GS comparisons of JanusX against BGLR and HIBayes via `jx bayesbench compare`; FarmCPU benchmarking of JanusX against rMVP via `jx benchmark`; and the GWAS/LMM benchmark matrix for JanusX, GCTA, GEMMA, REGENIE, and rMVP extracted from the benchmark driver script. The same script archives command logs, module help snapshots, Git commit information, and raw system `time` outputs used to summarize wall-clock runtime and peak resident set size (RSS).

## Shorter main-text alternative

Tool-specific benchmark settings are detailed in Methods S3. To ensure exact reproducibility, we additionally provide a unified shell script that records software versions and runs the complete BLUP, Bayesian GS, FarmCPU, and GWAS/LMM benchmark commands used in this study.

## Slightly more formal alternative

Detailed benchmark settings are provided in Methods S3, including tool-specific parameters, full command lines, thread allocation, genotype preprocessing, runtime definitions, and peak RSS measurement. For exact reproducibility, we prepared a unified shell script that executes the GS, Bayesian GS, FarmCPU, and GWAS/LMM benchmark workflows and simultaneously archives software-version records, module help text, command logs, and raw timing outputs for each run.

## Data/Code availability note

Most plant-science journals will accept a shell script either as a supplementary methods file or as a supplementary source-data/code file. If the journal does not allow executable shell scripts as supplementary material, the exact script should be deposited in the project GitHub repository and cited with a stable URL plus commit hash, ideally together with a Zenodo-archived release DOI. A suitable repository citation format is:
