#!/usr/bin/env bash

set -euo pipefail

MODE="${JX_GGVAL_MODE:-full}"
THREADS="${JX_GGVAL_THREADS:-2}"
CV_FOLDS="${JX_GGVAL_CV:-2}"

jx --version

if [[ "${MODE}" == "smoke" ]]; then
  echo "SMOKE 1. Build PLINK cache from example VCF"
  jx gformat -vcf example/mouse_hs1940.vcf.gz -fmt plink -o test
  echo "============================================"

  echo "SMOKE 1b. VCF cache split regression (-snps-only vs default)"
  VCF_SRC="example/mouse_hs1940.vcf.gz"
  VCF_PHENO="example/mouse_hs1940.pheno"
  VCF_CACHE_BASE="example/~mouse_hs1940"
  LOG_SNP0="/tmp/jx_gwas_vcf_snp0.log"
  LOG_SNP1="/tmp/jx_gwas_vcf_snp1.log"
  rm -f "${VCF_CACHE_BASE}.snp0.bed" "${VCF_CACHE_BASE}.snp0.bim" "${VCF_CACHE_BASE}.snp0.fam" \
        "${VCF_CACHE_BASE}.snp1.bed" "${VCF_CACHE_BASE}.snp1.bim" "${VCF_CACHE_BASE}.snp1.fam"
  jx gwas -vcf "${VCF_SRC}" -p "${VCF_PHENO}" -lm -n 0 -t "${THREADS}" -o test >"${LOG_SNP0}" 2>&1
  jx gwas -vcf "${VCF_SRC}" -p "${VCF_PHENO}" -lm -n 0 -snps-only -t "${THREADS}" -o test >"${LOG_SNP1}" 2>&1
  grep -Eq "Cache prefix: .*\\.snp0" "${LOG_SNP0}" || { echo "ERROR: default VCF GWAS did not use snp0 cache prefix."; cat "${LOG_SNP0}"; exit 1; }
  grep -Eq "Cache prefix: .*\\.snp1" "${LOG_SNP1}" || { echo "ERROR: -snps-only VCF GWAS did not use snp1 cache prefix."; cat "${LOG_SNP1}"; exit 1; }
  [[ -f "${VCF_CACHE_BASE}.snp0.bed" && -f "${VCF_CACHE_BASE}.snp0.bim" && -f "${VCF_CACHE_BASE}.snp0.fam" ]] \
    || { echo "ERROR: missing VCF cache files for snp0 mode."; ls -l "${VCF_CACHE_BASE}".snp* 2>/dev/null || true; exit 1; }
  [[ -f "${VCF_CACHE_BASE}.snp1.bed" && -f "${VCF_CACHE_BASE}.snp1.bim" && -f "${VCF_CACHE_BASE}.snp1.fam" ]] \
    || { echo "ERROR: missing VCF cache files for snp1 mode."; ls -l "${VCF_CACHE_BASE}".snp* 2>/dev/null || true; exit 1; }
  rm -f "${VCF_CACHE_BASE}.snp0.bed" "${VCF_CACHE_BASE}.snp0.bim" "${VCF_CACHE_BASE}.snp0.fam" \
        "${VCF_CACHE_BASE}.snp1.bed" "${VCF_CACHE_BASE}.snp1.bim" "${VCF_CACHE_BASE}.snp1.fam"
  echo "============================================"

  echo "SMOKE 2. GWAS runtime check (LM/LMM/FastLMM/FarmCPU)"
  jx grm -bfile test/mouse_hs1940 -o test
  jx pca -bfile test/mouse_hs1940 -o test
  GRM_K="test/mouse_hs1940.cGRM.txt"
  if [[ ! -f "${GRM_K}" && -f test/mouse_hs1940.grm.txt ]]; then
    GRM_K="test/mouse_hs1940.grm.txt"
  fi
  [[ -f "${GRM_K}" ]] || {
    echo "ERROR: GRM file not found for smoke GWAS (-k)."
    ls -l test/mouse_hs1940*.cGRM.* test/mouse_hs1940*.sGRM.* test/mouse_hs1940*.grm.* 2>/dev/null || true
    exit 1
  }
  jx gwas -bfile test/mouse_hs1940 -p example/mouse_hs1940.pheno \
      -farmcpu -lmm -lm -fastlmm -k "${GRM_K}" -c test/mouse_hs1940.eigenvec \
      -t "${THREADS}" -o test
  echo "============================================"

  echo "SMOKE 3. rrBLUP runtime check"
  jx gs -bfile test/mouse_hs1940 -p example/mouse_hs1940.pheno -n 0 \
      -rrBLUP -cv "${CV_FOLDS}" -t "${THREADS}" -o test
  echo "============================================"
  exit 0
fi

echo "STEP 1. Simulation for validation flow"
jx sim 10 1000 test/mouse_hs1940
jx gformat -vcf example/mouse_hs1940.vcf.gz -fmt plink -o test
jx gformat -vcf example/mouse_hs1940.vcf.gz -fmt hmp -o test
jx gformat -vcf example/mouse_hs1940.vcf.gz -fmt txt -o test
echo "============================================"

echo "STEP 2. Validation of GWAS-related modules"
jx grm -bfile test/mouse_hs1940 -o test -m 1
jx grm -bfile test/mouse_hs1940 -o test -m 2
jx pca -bfile test/mouse_hs1940 -o test
jx pca -bfile test/mouse_hs1940 -rsvd -o test
jx gwas -bfile test/mouse_hs1940 -p example/mouse_hs1940.pheno \
    -farmcpu -lmm -lm -fastlmm -k test/mouse_hs1940.cGRM.txt -c test/mouse_hs1940.eigenvec \
    -o test
jx postgwas -gwasfile $(find test/mouse_hs1940.test*.tsv | grep -v gs) -bfile test/mouse_hs1940 \
    -manh 4 -qq -scatter-size 16 -fmt pdf -full -palette tab10 \
    -o test
echo "============================================"

echo "STEP 3. Validation of GS-related modules"
jx gs -bfile test/mouse_hs1940 -p example/mouse_hs1940.pheno -n 0 \
    -GBLUP -rrBLUP -BayesA -BayesB -BayesCpi \
    -cv 5 -o test
jx reml -file example/rice6048.reml.tsv -n Plant_height \
    -rh year -rh loc -o test
echo "============================================"
