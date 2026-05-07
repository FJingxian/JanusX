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

  echo "SMOKE 2. GWAS runtime check (LM/LMM/FastLMM/FarmCPU)"
  jx grm -bfile test/mouse_hs1940 -o test
  jx pca -bfile test/mouse_hs1940 -o test
  jx gwas -bfile test/mouse_hs1940 -p example/mouse_hs1940.pheno \
      -farmcpu -lmm -lm -fastlmm -k test/mouse_hs1940.grm.txt -c test/mouse_hs1940.eigenvec \
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
jx grm -bfile test/mouse_hs1940 -o test
jx pca -bfile test/mouse_hs1940 -o test
jx pca -bfile test/mouse_hs1940 -rsvd -o test
jx gwas -bfile test/mouse_hs1940 -p example/mouse_hs1940.pheno \
    -farmcpu -lmm -lm -fastlmm -k test/mouse_hs1940.grm.txt -c test/mouse_hs1940.eigenvec \
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
