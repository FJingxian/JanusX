#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="${1:-help}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/.janusx_test/reproduce_methods_s3}"
HELP_DIR="${OUT_DIR}/help"
LOG_DIR="${OUT_DIR}/logs"
TIME_DIR="${OUT_DIR}/time"
DATA_DIR="${OUT_DIR}/data"
VERS_DIR="${OUT_DIR}/versions"

JX_BIN="${JX_BIN:-jx}"
RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"
GCTA_BIN="${GCTA_BIN:-gcta}"
GEMMA_BIN="${GEMMA_BIN:-gemma}"
REGENIE_BIN="${REGENIE_BIN:-regenie}"
HIBLUP_BIN="${HIBLUP_BIN:-hiblup}"
BLUPF90_BIN="${BLUPF90_BIN:-blupf90}"
PREGSF90_BIN="${PREGSF90_BIN:-preGSf90}"

THREADS="${THREADS:-4}"
RMVP_THREADS="${RMVP_THREADS:-1}"
MAF="${MAF:-0.02}"
GENO="${GENO:-0.05}"
SPARSE_CUTOFF="${SPARSE_CUTOFF:-0.05}"
JX_MEM_GB="${JX_MEM_GB:-0.512}"

SIM_NSNP_K="${SIM_NSNP_K:-500}"
SIM_N="${SIM_N:-5000}"
SIM_PVE="${SIM_PVE:-0.5}"
SIM_STRUCTURE="${SIM_STRUCTURE:-family}"
SIM_SEED="${SIM_SEED:-20260609}"
SIM_TRAIT_NAME="${SIM_TRAIT_NAME:-trait1}"

CUBIC_BFILE="${CUBIC_BFILE:-}"
CUBIC_PHENO="${CUBIC_PHENO:-}"
CUBIC_FARMCPU_TRAIT="${CUBIC_FARMCPU_TRAIT:-0}"

BAYES_METHODS="${BAYES_METHODS:-BayesA,BayesB,BayesCpi}"
BAYES_N_ITER="${BAYES_N_ITER:-2000}"
BAYES_BURNIN="${BAYES_BURNIN:-500}"
BAYES_THIN="${BAYES_THIN:-5}"
BAYES_REFERENCE="${BAYES_REFERENCE:-all}"

WHEAT_TRAIT="${WHEAT_TRAIT:-trait1}"
WHEAT_CV_FOLDS="${WHEAT_CV_FOLDS:-5}"
WHEAT_RUN_FOLDS="${WHEAT_RUN_FOLDS:-5}"
WHEAT_ENGINES="${WHEAT_ENGINES:-janusx,janusxrrblup,sommer,rrblup,blupf90,blupf90apy,hiblup}"

SIM_TAG="sim_${SIM_NSNP_K}k_${SIM_N}n_pve${SIM_PVE//./p}_${SIM_STRUCTURE}"
SIM_PREFIX="${DATA_DIR}/${SIM_TAG}"
SIM_GWAS_DIR="${OUT_DIR}/gwas_lmm/${SIM_TAG}"

WHEAT_ROOT="${OUT_DIR}/wheat_builtin"
WHEAT_RAW_DIR="${WHEAT_ROOT}/raw"
WHEAT_FILTERED_DIR="${WHEAT_ROOT}/filtered"
WHEAT_RAW_PREFIX="${WHEAT_RAW_DIR}/wheat"
WHEAT_PLINK_PREFIX="${WHEAT_FILTERED_DIR}/wheat_filt"
WHEAT_PHENO="${WHEAT_RAW_DIR}/wheat.pheno.tsv"

export MPLCONFIGDIR="${OUT_DIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}" "${HELP_DIR}" "${LOG_DIR}" "${TIME_DIR}" "${DATA_DIR}" "${VERS_DIR}"

TIME_TOOL=()

show_help() {
  cat <<'EOF'
reproduce_benchmarks_methods_s3.sh

Modes
  help         Show this help.
  versions     Record software versions and benchmark help snapshots.
  gs-blup      Reproduce the BLUP-engine benchmark on the built-in BGLR wheat dataset.
  gs-bayes     Reproduce the Bayesian GS comparison using `jx bayesbench compare --builtin wheat`.
  farmcpu      Reproduce the JanusX vs rMVP FarmCPU benchmark on a real CUBIC dataset.
  gwas-lmm     Reproduce the simulated GWAS/LMM benchmark matrix derived from test.final.mtask0.sh.
  all          Run versions + gs-blup + gs-bayes + farmcpu + gwas-lmm.

Key environment variables
  OUT_DIR               Output root (default: repo/.janusx_test/reproduce_methods_s3)
  JX_BIN                JanusX executable (default: jx)
  RSCRIPT_BIN           Rscript executable (default: Rscript)
  GCTA_BIN              GCTA executable (default: gcta)
  GEMMA_BIN             GEMMA executable (default: gemma)
  REGENIE_BIN           REGENIE executable (default: regenie)
  HIBLUP_BIN            HIBLUP executable (default: hiblup)
  BLUPF90_BIN           BLUPF90 executable (default: blupf90)
  PREGSF90_BIN          preGSf90 executable (default: preGSf90)
  THREADS               Thread count for benchmark wrappers (default: 4)
  RMVP_THREADS          Thread count for the standalone rMVP MLM run (default: 1)
  MAF                   Variant MAF threshold (default: 0.02)
  GENO                  Variant missing-rate threshold (default: 0.05)
  CUBIC_BFILE           PLINK prefix for the real CUBIC benchmark dataset
  CUBIC_PHENO           Phenotype table for the real CUBIC benchmark dataset
  CUBIC_FARMCPU_TRAIT   Trait selector passed to `jx benchmark` (default: 0)

Examples
  bash reproduce_benchmarks_methods_s3.sh versions
  CUBIC_BFILE=/path/to/cubic CUBIC_PHENO=/path/to/cubic.pheno.tsv bash reproduce_benchmarks_methods_s3.sh farmcpu
  bash reproduce_benchmarks_methods_s3.sh gwas-lmm
  bash reproduce_benchmarks_methods_s3.sh all
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: required command not found in PATH: ${cmd}" >&2
    exit 127
  fi
}

detect_time_tool() {
  TIME_TOOL=()
  if command -v gtime >/dev/null 2>&1; then
    TIME_TOOL=(gtime -v)
    return
  fi
  if [[ -x /usr/bin/time ]]; then
    if /usr/bin/time -v true >/dev/null 2>&1; then
      TIME_TOOL=(/usr/bin/time -v)
      return
    fi
    if /usr/bin/time -l true >/dev/null 2>&1; then
      TIME_TOOL=(/usr/bin/time -l)
      return
    fi
  fi
}

run_timed() {
  local label="$1"
  local log_file="$2"
  local time_file="$3"
  shift 3

  mkdir -p "$(dirname "${log_file}")" "$(dirname "${time_file}")"
  log "RUN ${label}: $*"
  if [[ ${#TIME_TOOL[@]} -gt 0 ]]; then
    "${TIME_TOOL[@]}" -o "${time_file}" "$@" >"${log_file}" 2>&1
  else
    "$@" >"${log_file}" 2>&1
  fi
}

set_blas_threads() {
  local n="$1"
  export OMP_NUM_THREADS="${n}"
  export OMP_THREAD_LIMIT="${n}"
  export OMP_DYNAMIC=FALSE
  export OPENBLAS_NUM_THREADS="${n}"
  export MKL_NUM_THREADS="${n}"
  export MKL_DYNAMIC=FALSE
  export GOTO_NUM_THREADS="${n}"
  export BLIS_NUM_THREADS="${n}"
  export NUMEXPR_NUM_THREADS="${n}"
  export VECLIB_MAXIMUM_THREADS="${n}"
  export R_PARALLEL_NUM_THREADS="${n}"
}

snapshot_help() {
  mkdir -p "${HELP_DIR}"
  "${JX_BIN}" gblupbench -h >"${HELP_DIR}/jx_gblupbench.help.txt" 2>&1 || true
  "${JX_BIN}" bayesbench -h >"${HELP_DIR}/jx_bayesbench.help.txt" 2>&1 || true
  "${JX_BIN}" bayesbench compare -h >"${HELP_DIR}/jx_bayesbench_compare.help.txt" 2>&1 || true
  "${JX_BIN}" benchmark -h >"${HELP_DIR}/jx_benchmark.help.txt" 2>&1 || true
  cp "${REPO_ROOT}/test.final.mtask0.sh" "${HELP_DIR}/test.final.mtask0.sh"
  cp "${SCRIPT_DIR}/reproduce_benchmarks_methods_s3.sh" "${HELP_DIR}/reproduce_benchmarks_methods_s3.sh"
}

capture_external_version() {
  local label="$1"
  local bin="$2"
  {
    echo "## ${label}"
    if ! command -v "${bin}" >/dev/null 2>&1; then
      echo "not found: ${bin}"
      return 0
    fi
    echo "path: $(command -v "${bin}")"
    ("${bin}" --version || "${bin}" -v || "${bin}" -V || "${bin}" --help || "${bin}" -h || true) 2>&1 | sed -n '1,5p'
  } >>"${VERS_DIR}/software_versions.txt"
}

record_versions() {
  require_cmd "${JX_BIN}"
  require_cmd "${RSCRIPT_BIN}"
  snapshot_help

  {
    echo "# Benchmark software versions"
    echo "date_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "repo_root: ${REPO_ROOT}"
    if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "git_commit: $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo NA)"
      if [[ -n "$(git -C "${REPO_ROOT}" status --short 2>/dev/null || true)" ]]; then
        echo "git_worktree: dirty"
      else
        echo "git_worktree: clean"
      fi
    fi
    echo
    echo "## JanusX"
    "${JX_BIN}" -v || true
    echo
    echo "## R packages"
    "${RSCRIPT_BIN}" -e "pkgs <- c('BGLR','rrBLUP','sommer','rMVP','HIBayes','bigmemory'); for (p in pkgs) { if (requireNamespace(p, quietly=TRUE)) { cat(sprintf('%s\t%s\n', p, as.character(packageVersion(p)))) } else { cat(sprintf('%s\tNA\n', p)) } }" || true
    echo
  } >"${VERS_DIR}/software_versions.txt" 2>&1

  capture_external_version "GCTA" "${GCTA_BIN}"
  capture_external_version "GEMMA" "${GEMMA_BIN}"
  capture_external_version "REGENIE" "${REGENIE_BIN}"
  capture_external_version "HIBLUP" "${HIBLUP_BIN}"
  capture_external_version "BLUPF90" "${BLUPF90_BIN}"
  capture_external_version "preGSf90" "${PREGSF90_BIN}"
}

prepare_wheat_builtin() {
  require_cmd "${RSCRIPT_BIN}"
  require_cmd "${JX_BIN}"
  mkdir -p "${WHEAT_RAW_DIR}" "${WHEAT_FILTERED_DIR}"

  if [[ ! -f "${WHEAT_RAW_PREFIX}.tsv" || ! -f "${WHEAT_RAW_PREFIX}.id" || ! -f "${WHEAT_PHENO}" ]]; then
    run_timed \
      "export_builtin_wheat" \
      "${LOG_DIR}/export_builtin_wheat.log" \
      "${TIME_DIR}/export_builtin_wheat.time.txt" \
      "${RSCRIPT_BIN}" -e \
      "library(BGLR); data('wheat', package='BGLR'); ids <- sprintf('wheat_%04d', seq_len(nrow(wheat.X))); write.table(t(wheat.X), file='${WHEAT_RAW_PREFIX}.tsv', sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE); writeLines(ids, '${WHEAT_RAW_PREFIX}.id'); write.table(data.frame(chr=1, pos=seq_len(ncol(wheat.X)), ref='A', alt='G'), file='${WHEAT_RAW_PREFIX}.site', sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE); write.table(data.frame(id=ids, trait1=wheat.Y[,1], trait2=wheat.Y[,2], trait4=wheat.Y[,3], trait5=wheat.Y[,4]), file='${WHEAT_PHENO}', sep='\t', quote=FALSE, row.names=FALSE, col.names=TRUE)"
  fi

  if [[ ! -f "${WHEAT_PLINK_PREFIX}.bed" ]]; then
    run_timed \
      "wheat_gformat_plink" \
      "${LOG_DIR}/wheat_gformat_plink.log" \
      "${TIME_DIR}/wheat_gformat_plink.time.txt" \
      "${JX_BIN}" gformat \
      -file "${WHEAT_RAW_PREFIX}" \
      -fmt plink \
      -maf "${MAF}" \
      -geno "${GENO}" \
      -o "${WHEAT_FILTERED_DIR}" \
      -prefix "$(basename "${WHEAT_PLINK_PREFIX}")" \
      -t "${THREADS}"
  fi
}

run_gs_blup_benchmark() {
  prepare_wheat_builtin
  run_timed \
    "jx_gblupbench_wheat" \
    "${LOG_DIR}/jx_gblupbench_wheat.log" \
    "${TIME_DIR}/jx_gblupbench_wheat.time.txt" \
    "${JX_BIN}" gblupbench \
    -bfile "${WHEAT_PLINK_PREFIX}" \
    -p "${WHEAT_PHENO}" \
    -n "${WHEAT_TRAIT}" \
    -o "${OUT_DIR}/gs_blup_benchmark" \
    -prefix wheat_trait1 \
    -maf "${MAF}" \
    -geno "${GENO}" \
    --cv "${WHEAT_CV_FOLDS}" \
    --run-folds "${WHEAT_RUN_FOLDS}" \
    --engines "${WHEAT_ENGINES}" \
    -t "${THREADS}"
}

run_gs_bayes_benchmark() {
  require_cmd "${JX_BIN}"
  run_timed \
    "jx_bayesbench_compare_wheat" \
    "${LOG_DIR}/jx_bayesbench_compare_wheat.log" \
    "${TIME_DIR}/jx_bayesbench_compare_wheat.time.txt" \
    "${JX_BIN}" bayesbench compare \
    --builtin wheat \
    -n "${WHEAT_TRAIT}" \
    --methods "${BAYES_METHODS}" \
    --n-iter "${BAYES_N_ITER}" \
    --burnin "${BAYES_BURNIN}" \
    --thin "${BAYES_THIN}" \
    --reference "${BAYES_REFERENCE}" \
    --rscript "${RSCRIPT_BIN}" \
    -t "${THREADS}" \
    -o "${OUT_DIR}/gs_bayes_benchmark"
}

run_farmcpu_benchmark() {
  require_cmd "${JX_BIN}"
  if [[ -z "${CUBIC_BFILE}" || -z "${CUBIC_PHENO}" ]]; then
    echo "ERROR: CUBIC_BFILE and CUBIC_PHENO must be set for farmcpu mode." >&2
    exit 2
  fi
  run_timed \
    "jx_benchmark_cubic_farmcpu" \
    "${LOG_DIR}/jx_benchmark_cubic_farmcpu.log" \
    "${TIME_DIR}/jx_benchmark_cubic_farmcpu.time.txt" \
    "${JX_BIN}" benchmark \
    -bfile "${CUBIC_BFILE}" \
    -p "${CUBIC_PHENO}" \
    -n "${CUBIC_FARMCPU_TRAIT}" \
    -o "${OUT_DIR}/farmcpu_cubic" \
    -prefix cubic_farmcpu \
    -maf "${MAF}" \
    -geno "${GENO}" \
    --kernels janusx,rmvp \
    -t "${THREADS}"
}

prepare_simulated_lmm_dataset() {
  require_cmd "${JX_BIN}"
  mkdir -p "${DATA_DIR}"
  if [[ ! -f "${SIM_PREFIX}.bed" || ! -f "${SIM_PREFIX}.pheno" || ! -f "${SIM_PREFIX}.pheno.txt" ]]; then
    run_timed \
      "jx_sim_gwas_lmm_dataset" \
      "${LOG_DIR}/jx_sim_gwas_lmm_dataset.log" \
      "${TIME_DIR}/jx_sim_gwas_lmm_dataset.time.txt" \
      "${JX_BIN}" sim \
      "${SIM_NSNP_K}" \
      "${SIM_N}" \
      "${SIM_PREFIX}" \
      -structure "${SIM_STRUCTURE}" \
      -pve "${SIM_PVE}" \
      -seed "${SIM_SEED}" \
      -trait-name "${SIM_TRAIT_NAME}"
  fi
}

run_jx_gwas_method() {
  local method="$1"
  local extra_flag="$2"
  local outdir="${SIM_GWAS_DIR}/janusx_${method}"
  local grm_dir="${SIM_GWAS_DIR}/grm_dense"
  local grm_path="${grm_dir}/${SIM_TAG}.cGRM.npy"

  mkdir -p "${outdir}" "${grm_dir}"
  if [[ ! -f "${grm_path}" || ! -f "${grm_path}.id" ]]; then
    run_timed \
      "jx_grm_dense" \
      "${LOG_DIR}/gwas_lmm.jx_grm_dense.log" \
      "${TIME_DIR}/gwas_lmm.jx_grm_dense.time.txt" \
      "${JX_BIN}" grm \
      -bfile "${SIM_PREFIX}" \
      -m 1 \
      -t "${THREADS}" \
      -o "${grm_dir}" \
      -prefix "${SIM_TAG}" \
      -mem "${JX_MEM_GB}"
  fi

  run_timed \
    "jx_gwas_${method}" \
    "${LOG_DIR}/gwas_lmm.jx_${method}.log" \
    "${TIME_DIR}/gwas_lmm.jx_${method}.time.txt" \
    env JX_GWAS_PLOT=0 "${JX_BIN}" gwas \
    -bfile "${SIM_PREFIX}" \
    -p "${SIM_PREFIX}.pheno.txt" \
    -k "${grm_path}" \
    -t "${THREADS}" \
    -o "${outdir}" \
    -prefix "${SIM_TAG}" \
    -mem "${JX_MEM_GB}" \
    "${extra_flag}"
}

run_jx_sparse_method() {
  local method="$1"
  local extra_flag="$2"
  local outdir="${SIM_GWAS_DIR}/janusx_${method}"
  local grm_dir="${SIM_GWAS_DIR}/grm_sparse"
  local spgrm_path="${grm_dir}/${SIM_TAG}.cGRM.spgrm"

  mkdir -p "${outdir}" "${grm_dir}"
  if [[ ! -f "${spgrm_path}" || ! -f "${spgrm_path}.id" ]]; then
    run_timed \
      "jx_grm_sparse" \
      "${LOG_DIR}/gwas_lmm.jx_grm_sparse.log" \
      "${TIME_DIR}/gwas_lmm.jx_grm_sparse.time.txt" \
      "${JX_BIN}" grm \
      -bfile "${SIM_PREFIX}" \
      -m 1 \
      -sparse "${SPARSE_CUTOFF}" \
      -t "${THREADS}" \
      -o "${grm_dir}" \
      -prefix "${SIM_TAG}" \
      -mem "${JX_MEM_GB}"
  fi

  run_timed \
    "jx_gwas_${method}" \
    "${LOG_DIR}/gwas_lmm.jx_${method}.log" \
    "${TIME_DIR}/gwas_lmm.jx_${method}.time.txt" \
    env JX_GWAS_PLOT=0 "${JX_BIN}" gwas \
    -bfile "${SIM_PREFIX}" \
    -p "${SIM_PREFIX}.pheno.txt" \
    -spk "${spgrm_path}" \
    -t "${THREADS}" \
    -o "${outdir}" \
    -prefix "${SIM_TAG}" \
    -mem "${JX_MEM_GB}" \
    ${extra_flag} "${SPARSE_CUTOFF}"
}

run_gcta_mlma() {
  local outdir="${SIM_GWAS_DIR}/gcta_mlma"
  local grm_prefix="${outdir}/${SIM_TAG}.grm"
  mkdir -p "${outdir}"
  run_timed \
    "gcta_mlma_make_grm" \
    "${LOG_DIR}/gwas_lmm.gcta_mlma.make_grm.log" \
    "${TIME_DIR}/gwas_lmm.gcta_mlma.make_grm.time.txt" \
    "${GCTA_BIN}" --bfile "${SIM_PREFIX}" --make-grm --out "${grm_prefix}" --thread-num "${THREADS}"
  run_timed \
    "gcta_mlma_assoc" \
    "${LOG_DIR}/gwas_lmm.gcta_mlma.assoc.log" \
    "${TIME_DIR}/gwas_lmm.gcta_mlma.assoc.time.txt" \
    "${GCTA_BIN}" --bfile "${SIM_PREFIX}" --grm "${grm_prefix}" --mlma --mpheno 1 --pheno "${SIM_PREFIX}.pheno" --out "${outdir}/${SIM_TAG}" --thread-num "${THREADS}"
}

run_gcta_fastgwa() {
  local mode="$1"
  local flag="$2"
  local outdir="${SIM_GWAS_DIR}/gcta_${mode}"
  local grm_prefix="${outdir}/${SIM_TAG}.grm"
  local sparse_prefix="${outdir}/${SIM_TAG}.sparse"
  mkdir -p "${outdir}"
  run_timed \
    "gcta_${mode}_make_grm" \
    "${LOG_DIR}/gwas_lmm.gcta_${mode}.make_grm.log" \
    "${TIME_DIR}/gwas_lmm.gcta_${mode}.make_grm.time.txt" \
    "${GCTA_BIN}" --bfile "${SIM_PREFIX}" --make-grm --out "${grm_prefix}" --thread-num "${THREADS}"
  run_timed \
    "gcta_${mode}_make_sparse" \
    "${LOG_DIR}/gwas_lmm.gcta_${mode}.make_sparse.log" \
    "${TIME_DIR}/gwas_lmm.gcta_${mode}.make_sparse.time.txt" \
    "${GCTA_BIN}" --grm "${grm_prefix}" --make-bK-sparse "${SPARSE_CUTOFF}" --out "${sparse_prefix}" --thread-num "${THREADS}"
  run_timed \
    "gcta_${mode}_assoc" \
    "${LOG_DIR}/gwas_lmm.gcta_${mode}.assoc.log" \
    "${TIME_DIR}/gwas_lmm.gcta_${mode}.assoc.time.txt" \
    "${GCTA_BIN}" --bfile "${SIM_PREFIX}" --grm-sparse "${sparse_prefix}" "${flag}" --mpheno 1 --pheno "${SIM_PREFIX}.pheno" --out "${outdir}/${SIM_TAG}" --thread-num "${THREADS}"
}

run_gemma_lmm() {
  local outdir="${SIM_GWAS_DIR}/gemma_lmm"
  local gemma_pheno="${outdir}/${SIM_TAG}.gemma.pheno.txt"
  mkdir -p "${outdir}/output"
  awk 'NR > 1 { print $2 }' "${SIM_PREFIX}.pheno.txt" >"${gemma_pheno}"
  run_timed \
    "gemma_kinship" \
    "${LOG_DIR}/gwas_lmm.gemma_kinship.log" \
    "${TIME_DIR}/gwas_lmm.gemma_kinship.time.txt" \
    bash -lc "cd '${outdir}' && '${GEMMA_BIN}' -bfile '${SIM_PREFIX}' -p '${gemma_pheno}' -gk 1 -o '${SIM_TAG}.kinship'"
  run_timed \
    "gemma_lmm_assoc" \
    "${LOG_DIR}/gwas_lmm.gemma_lmm_assoc.log" \
    "${TIME_DIR}/gwas_lmm.gemma_lmm_assoc.time.txt" \
    bash -lc "cd '${outdir}' && '${GEMMA_BIN}' -bfile '${SIM_PREFIX}' -p '${gemma_pheno}' -k 'output/${SIM_TAG}.kinship.cXX.txt' -lmm 4 -n 1 -o '${SIM_TAG}.assoc'"
}

run_regenie() {
  local outdir="${SIM_GWAS_DIR}/regenie"
  local regenie_pheno="${outdir}/${SIM_TAG}.regenie.pheno.tsv"
  local step1_prefix="${outdir}/${SIM_TAG}.step1"
  local step2_prefix="${outdir}/${SIM_TAG}"
  mkdir -p "${outdir}"
  awk 'BEGIN { OFS="\t"; print "FID","IID","trait1" } { print $1,$2,$3 }' "${SIM_PREFIX}.pheno" >"${regenie_pheno}"
  run_timed \
    "regenie_step1" \
    "${LOG_DIR}/gwas_lmm.regenie_step1.log" \
    "${TIME_DIR}/gwas_lmm.regenie_step1.time.txt" \
    "${REGENIE_BIN}" --step 1 --bed "${SIM_PREFIX}" --phenoFile "${regenie_pheno}" --phenoCol trait1 --qt --threads "${THREADS}" --bsize 1000 --lowmem --lowmem-prefix "${outdir}/regenie_tmp" --out "${step1_prefix}"
  run_timed \
    "regenie_step2" \
    "${LOG_DIR}/gwas_lmm.regenie_step2.log" \
    "${TIME_DIR}/gwas_lmm.regenie_step2.time.txt" \
    "${REGENIE_BIN}" --step 2 --bed "${SIM_PREFIX}" --phenoFile "${regenie_pheno}" --phenoCol trait1 --qt --pred "${step1_prefix}_pred.list" --threads "${THREADS}" --bsize 400 --out "${step2_prefix}"
}

write_rmvp_mlm_runner() {
  local runner="$1"
  cat >"${runner}" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
bfile <- args[[1]]
phe_path <- args[[2]]
outdir <- args[[3]]
prefix <- args[[4]]
cpus <- as.integer(args[[5]])
if (!is.finite(cpus) || cpus < 1L) cpus <- 1L

Sys.setenv(
  OMP_NUM_THREADS = cpus,
  OMP_THREAD_LIMIT = cpus,
  OMP_DYNAMIC = "FALSE",
  OPENBLAS_NUM_THREADS = cpus,
  MKL_NUM_THREADS = cpus,
  MKL_DYNAMIC = "FALSE",
  GOTO_NUM_THREADS = cpus,
  BLIS_NUM_THREADS = cpus,
  VECLIB_MAXIMUM_THREADS = cpus,
  R_PARALLEL_NUM_THREADS = cpus
)
options(mc.cores = cpus)

suppressPackageStartupMessages({
  library(rMVP)
  library(bigmemory)
})

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
cache_prefix <- file.path(outdir, "mvp_cache", prefix)
dir.create(dirname(cache_prefix), recursive = TRUE, showWarnings = FALSE)

call_supported <- function(fun, args) {
  fm <- tryCatch(names(formals(fun)), error = function(e) NULL)
  if (!is.null(fm)) {
    keep <- intersect(names(args), setdiff(fm, "..."))
    args <- args[keep]
  }
  do.call(fun, args)
}

pick_first <- function(paths) {
  hits <- unique(paths[file.exists(paths)])
  if (length(hits) == 0) stop("missing expected file")
  hits[[1]]
}

compute_rmvp_kinship <- function(geno, cpus) {
  kin_fun <- MVP.K.VanRaden
  kin_args <- list(M = geno, cpu = cpus, verbose = TRUE)
  out <- try(call_supported(kin_fun, kin_args), silent = TRUE)
  if (!inherits(out, "try-error")) return(out)
  kin_args$maxLine <- 512L
  call_supported(kin_fun, kin_args)
}

call_supported(
  MVP.Data,
  list(
    fileBed = bfile,
    filePhe = phe_path,
    sep.phe = "\t",
    out = cache_prefix,
    ncpus = cpus,
    cpu = cpus,
    verbose = TRUE
  )
)

geno_desc <- pick_first(c(paste0(cache_prefix, ".geno.desc"), paste0(cache_prefix, ".genotype.desc")))
map_path <- pick_first(c(paste0(cache_prefix, ".geno.map"), paste0(cache_prefix, ".map.map"), paste0(cache_prefix, ".map")))
phe_aligned_path <- pick_first(c(paste0(cache_prefix, ".phe"), paste0(cache_prefix, ".phenotype.phe")))

geno <- attach.big.matrix(geno_desc)
map <- read.table(map_path, header = TRUE, stringsAsFactors = FALSE)
phe <- read.table(phe_aligned_path, header = TRUE, stringsAsFactors = FALSE)
kin <- compute_rmvp_kinship(geno, cpus)

call_supported(
  MVP,
  list(
    phe = phe,
    geno = geno,
    map = map,
    K = kin,
    method = c("MLM"),
    ncpus = cpus,
    outpath = outdir,
    memo = paste0(prefix, ".rmvp_mlm"),
    verbose = TRUE
  )
)
RS
}

run_rmvp_mlm() {
  local outdir="${SIM_GWAS_DIR}/rmvp_mlm"
  local rmvp_pheno="${outdir}/${SIM_TAG}.rmvp.pheno.tsv"
  local rmvp_runner="${outdir}/run_rmvp_mlm.R"
  mkdir -p "${outdir}"
  awk 'BEGIN { OFS="\t"; print "Taxa","trait1" } NR > 1 { print $1,$2 }' "${SIM_PREFIX}.pheno.txt" >"${rmvp_pheno}"
  write_rmvp_mlm_runner "${rmvp_runner}"
  run_timed \
    "rmvp_mlm" \
    "${LOG_DIR}/gwas_lmm.rmvp_mlm.log" \
    "${TIME_DIR}/gwas_lmm.rmvp_mlm.time.txt" \
    "${RSCRIPT_BIN}" "${rmvp_runner}" "${SIM_PREFIX}" "${rmvp_pheno}" "${outdir}" "${SIM_TAG}" "${RMVP_THREADS}"
}

run_gwas_lmm_matrix() {
  require_cmd "${JX_BIN}"
  require_cmd "${GCTA_BIN}"
  require_cmd "${GEMMA_BIN}"
  require_cmd "${REGENIE_BIN}"
  require_cmd "${RSCRIPT_BIN}"
  prepare_simulated_lmm_dataset
  mkdir -p "${SIM_GWAS_DIR}"
  set_blas_threads "${THREADS}"

  run_jx_gwas_method "lmm" "-lmm"
  run_jx_gwas_method "fvlmm" "-fvlmm"
  run_jx_sparse_method "splmm_exact" "-splmm"
  run_jx_sparse_method "splmm_approx" "-splmm-approx"
  run_gcta_mlma
  run_gcta_fastgwa "fastgwa" "--fastGWA-mlm"
  run_gcta_fastgwa "fastgwa_exact" "--fastGWA-mlm-exact"
  run_gemma_lmm
  run_regenie
  run_rmvp_mlm
}

case "${MODE}" in
  help|-h|--help)
    show_help
    ;;
  versions)
    detect_time_tool
    record_versions
    ;;
  gs-blup)
    detect_time_tool
    record_versions
    run_gs_blup_benchmark
    ;;
  gs-bayes)
    detect_time_tool
    record_versions
    run_gs_bayes_benchmark
    ;;
  farmcpu)
    detect_time_tool
    record_versions
    run_farmcpu_benchmark
    ;;
  gwas-lmm)
    detect_time_tool
    record_versions
    run_gwas_lmm_matrix
    ;;
  all)
    detect_time_tool
    record_versions
    run_gs_blup_benchmark
    run_gs_bayes_benchmark
    run_farmcpu_benchmark
    run_gwas_lmm_matrix
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    show_help >&2
    exit 1
    ;;
esac
