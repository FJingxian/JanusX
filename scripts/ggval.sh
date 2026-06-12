#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTDIR="${JX_GGVAL_OUTDIR:-test}"
LOGDIR="${JX_GGVAL_LOGDIR:-${OUTDIR}/logs}"
KEEP_LOGS="${JX_GGVAL_KEEP_LOGS:-1}"
POSTGS_ENABLED="${JX_GGVAL_POSTGS:-1}"

mkdir -p "${OUTDIR}" "${LOGDIR}"

step() {
  echo
  echo "==> $*"
}

sep() {
  echo "============================================"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  local msg="${2:-required file not found}"
  [[ -f "${path}" ]] || fail "${msg}: ${path}"
}

require_any_file() {
  local msg="$1"
  shift
  local path
  for path in "$@"; do
    [[ -f "${path}" ]] && return 0
  done
  echo "Missing candidates:" >&2
  for path in "$@"; do
    echo "  ${path}" >&2
  done
  fail "${msg}"
}

run() {
  echo "+ $*"
  "$@"
}

cleanup_tmp_logs() {
  if [[ "${KEEP_LOGS}" != "1" ]]; then
    rm -f "${LOGDIR}/jx_gwas_vcf_snp0.log" "${LOGDIR}/jx_gwas_vcf_snp1.log"
  fi
}
trap cleanup_tmp_logs EXIT

MODE="${JX_GGVAL_MODE:-full}"
THREADS="${JX_GGVAL_THREADS:-2}"
CV_FOLDS="${JX_GGVAL_CV:-2}"

case "${MODE}" in
  smoke|full) ;;
  *) fail "JX_GGVAL_MODE must be 'smoke' or 'full', got '${MODE}'" ;;
esac

[[ "${THREADS}" =~ ^[0-9]+$ && "${THREADS}" -ge 1 ]] || fail "JX_GGVAL_THREADS must be a positive integer, got '${THREADS}'"
[[ "${CV_FOLDS}" =~ ^[0-9]+$ && "${CV_FOLDS}" -ge 2 ]] || fail "JX_GGVAL_CV must be an integer >= 2, got '${CV_FOLDS}'"

command -v jx >/dev/null 2>&1 || fail "jx command not found in PATH"
run jx --version

if [[ "${MODE}" == "smoke" ]]; then
  step "SMOKE 1. Build PLINK cache from example VCF"
  run jx gformat -vcf example/mouse_hs1940.vcf.gz -fmt plink -o "${OUTDIR}"
  require_file "${OUTDIR}/mouse_hs1940.bed" "PLINK BED output missing"
  require_file "${OUTDIR}/mouse_hs1940.bim" "PLINK BIM output missing"
  require_file "${OUTDIR}/mouse_hs1940.fam" "PLINK FAM output missing"
  sep

  step "SMOKE 1b. VCF cache split regression (-snps-only vs default)"
  VCF_SRC="example/mouse_hs1940.vcf.gz"
  VCF_PHENO="example/mouse_hs1940.pheno"
  VCF_CACHE_BASE="example/~mouse_hs1940"
  LOG_SNP0="${LOGDIR}/jx_gwas_vcf_snp0.log"
  LOG_SNP1="${LOGDIR}/jx_gwas_vcf_snp1.log"
  rm -f "${VCF_CACHE_BASE}.snp0.bed" "${VCF_CACHE_BASE}.snp0.bim" "${VCF_CACHE_BASE}.snp0.fam" \
        "${VCF_CACHE_BASE}.snp1.bed" "${VCF_CACHE_BASE}.snp1.bim" "${VCF_CACHE_BASE}.snp1.fam"
  run jx gwas -vcf "${VCF_SRC}" -p "${VCF_PHENO}" -lm -n 0 -t "${THREADS}" -o "${OUTDIR}" >"${LOG_SNP0}" 2>&1
  run jx gwas -vcf "${VCF_SRC}" -p "${VCF_PHENO}" -lm -n 0 -snps-only -t "${THREADS}" -o "${OUTDIR}" >"${LOG_SNP1}" 2>&1
  grep -Eq "(Cache prefix:|Genotype Cache[[:space:]]*:).*\\.snp0" "${LOG_SNP0}" || { cat "${LOG_SNP0}"; fail "default VCF GWAS did not use snp0 cache prefix."; }
  grep -Eq "(Cache prefix:|Genotype Cache[[:space:]]*:).*\\.snp1" "${LOG_SNP1}" || { cat "${LOG_SNP1}"; fail "-snps-only VCF GWAS did not use snp1 cache prefix."; }
  [[ -f "${VCF_CACHE_BASE}.snp0.bed" && -f "${VCF_CACHE_BASE}.snp0.bim" && -f "${VCF_CACHE_BASE}.snp0.fam" ]] \
    || { ls -l "${VCF_CACHE_BASE}".snp* 2>/dev/null || true; fail "missing VCF cache files for snp0 mode."; }
  [[ -f "${VCF_CACHE_BASE}.snp1.bed" && -f "${VCF_CACHE_BASE}.snp1.bim" && -f "${VCF_CACHE_BASE}.snp1.fam" ]] \
    || { ls -l "${VCF_CACHE_BASE}".snp* 2>/dev/null || true; fail "missing VCF cache files for snp1 mode."; }
  rm -f "${VCF_CACHE_BASE}.snp0.bed" "${VCF_CACHE_BASE}.snp0.bim" "${VCF_CACHE_BASE}.snp0.fam" \
        "${VCF_CACHE_BASE}.snp1.bed" "${VCF_CACHE_BASE}.snp1.bim" "${VCF_CACHE_BASE}.snp1.fam"
  sep

  step "SMOKE 2. GWAS runtime check (LM/LMM/FastLMM/FarmCPU)"
  run jx grm -bfile "${OUTDIR}/mouse_hs1940" -o "${OUTDIR}"
  run jx pca -bfile "${OUTDIR}/mouse_hs1940" -o "${OUTDIR}"
  GRM_K="${OUTDIR}/mouse_hs1940.cGRM.txt"
  if [[ ! -f "${GRM_K}" && -f "${OUTDIR}/mouse_hs1940.grm.txt" ]]; then
    GRM_K="${OUTDIR}/mouse_hs1940.grm.txt"
  fi
  [[ -f "${GRM_K}" ]] || {
    ls -l "${OUTDIR}"/mouse_hs1940*.cGRM.* "${OUTDIR}"/mouse_hs1940*.sGRM.* "${OUTDIR}"/mouse_hs1940*.grm.* 2>/dev/null || true
    fail "GRM file not found for smoke GWAS (-k)."
  }
  require_file "${OUTDIR}/mouse_hs1940.eigenvec" "PCA eigenvec output missing"
  run jx gwas -bfile "${OUTDIR}/mouse_hs1940" -p example/mouse_hs1940.pheno \
      -farmcpu -lmm -lm -fastlmm -k "${GRM_K}" -c "${OUTDIR}/mouse_hs1940.eigenvec" \
      -t "${THREADS}" -o "${OUTDIR}"
  sep

  step "SMOKE 3. rrBLUP runtime check"
  run jx gs -bfile "${OUTDIR}/mouse_hs1940" -p example/mouse_hs1940.pheno -n 0 \
      -rrBLUP -cv "${CV_FOLDS}" -t "${THREADS}" -o "${OUTDIR}"
  require_any_file "GS summary output missing" \
      "${OUTDIR}/mouse_hs1940.gs.model/summary.json" \
      "${OUTDIR}/mouse_hs1940.gs/summary.json"
  sep
  exit 0
fi

step "STEP 1. Simulation for validation flow"
run jx sim 10 1000 "${OUTDIR}/mouse_hs1940"
run jx gformat -vcf example/mouse_hs1940.vcf.gz -fmt plink -o "${OUTDIR}"
require_file "${OUTDIR}/mouse_hs1940.bed" "PLINK BED output missing"
require_file "${OUTDIR}/mouse_hs1940.bim" "PLINK BIM output missing"
require_file "${OUTDIR}/mouse_hs1940.fam" "PLINK FAM output missing"
run jx gformat -vcf example/mouse_hs1940.vcf.gz -fmt hmp -o "${OUTDIR}"
run jx gformat -vcf example/mouse_hs1940.vcf.gz -fmt txt -o "${OUTDIR}"
sep

step "STEP 2. Validation of GWAS-related modules"
run jx grm -bfile "${OUTDIR}/mouse_hs1940" -o "${OUTDIR}" -m 1
run jx grm -bfile "${OUTDIR}/mouse_hs1940" -o "${OUTDIR}" -m 2
run jx pca -bfile "${OUTDIR}/mouse_hs1940" -o "${OUTDIR}"
run jx pca -bfile "${OUTDIR}/mouse_hs1940" -rsvd -o "${OUTDIR}"
GRM_K="${OUTDIR}/mouse_hs1940.cGRM.txt"
if [[ ! -f "${GRM_K}" && -f "${OUTDIR}/mouse_hs1940.grm.txt" ]]; then
  GRM_K="${OUTDIR}/mouse_hs1940.grm.txt"
fi
[[ -f "${GRM_K}" ]] || {
  ls -l "${OUTDIR}"/mouse_hs1940*.cGRM.* "${OUTDIR}"/mouse_hs1940*.sGRM.* "${OUTDIR}"/mouse_hs1940*.grm.* 2>/dev/null || true
  fail "GRM file not found for full GWAS (-k)."
}
require_file "${OUTDIR}/mouse_hs1940.eigenvec" "PCA eigenvec output missing"
GWAS_FILES=()
GWAS_FIND_LIST="${LOGDIR}/gwas_files.list"
find "${OUTDIR}" -maxdepth 1 -type f -name 'mouse_hs1940.test*.tsv' ! -name '*.gs.tsv' | sort >"${GWAS_FIND_LIST}"
while IFS= read -r gwas_file; do
  [[ -n "${gwas_file}" ]] && GWAS_FILES+=("${gwas_file}")
done <"${GWAS_FIND_LIST}"
[[ "${#GWAS_FILES[@]}" -gt 0 ]] || fail "No GWAS result TSV files found for postgwas."
run jx postgwas -gwasfile "${GWAS_FILES[@]}" -bfile "${OUTDIR}/mouse_hs1940" \
    -manh 4 -qq -scatter-size 16 -fmt pdf -full -palette tab10 \
    -o "${OUTDIR}"
sep

step "STEP 3. Validation of GS-related modules"
run jx gs -bfile "${OUTDIR}/mouse_hs1940" -p example/mouse_hs1940.pheno -n 0 \
    -GBLUP -GBLUP ad -rrBLUP -BayesA -BayesB -BayesCpi \
    -cv 5 -o "${OUTDIR}"
require_any_file "GS summary output missing" \
    "${OUTDIR}/mouse_hs1940.gs.model/summary.json" \
    "${OUTDIR}/mouse_hs1940.gs/summary.json"

if [[ "${POSTGS_ENABLED}" == "1" ]]; then
  GS_SUMMARY="${OUTDIR}/mouse_hs1940.gs.model/summary.json"
  if [[ ! -f "${GS_SUMMARY}" ]]; then
    GS_SUMMARY="${OUTDIR}/mouse_hs1940.gs/summary.json"
  fi
  step "STEP 3b. Validation of postGS-related modules"
  run jx postgs -json "${GS_SUMMARY}" -o "${OUTDIR}"
  sep
fi

run jx reml -file example/rice6048.reml.tsv -n Plant_height \
    -rh year -rh loc -o "${OUTDIR}"
sep

step "Validation completed"
echo "Mode      : ${MODE}"
echo "Output dir: ${OUTDIR}"
echo "Log dir   : ${LOGDIR}"
