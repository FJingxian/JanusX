# GARFIELD: Max6 → GRM-INT

The production window and gene scans use a two-stage interaction workflow:

1. Apply genotype quality filters on the phenotype-aligned samples and pack the
   retained pure-line genotypes for the scan.
2. Exhaustively screen eligible marker pairs with the Max6 statistic, using
   genotype-identical LD tags by default.
3. Keep up to `-topk` pairs per scan unit and score their one-degree-of-freedom
   interaction with the GRM-INT FWL test. The test adjusts for both main
   effects, supplied covariates, and the aligned GRM.

Example:

```bash
jx garfield -bfile cohort -p traits.tsv -k cohort.grm.npy \
  -c cohort.pcs.tsv -w -topk 100 -o results/cohort
```

GRM-INT is enabled by default for the Max6 scan. Each result row reports the
Max6 screen rank and score, the four-cell counts, the minimum cell count, the
GRM-INT chi-square statistic, and its one-df p-value. The p-value is an
unadjusted per-pair value; the printed Bonferroni threshold is a reference for
the unique-pair family and does not filter rows.

`--min-cell` sets the minimum sample count required in every binary genotype
cell (default 10); `--min-cell-frac` can impose an additional fraction of the
effective sample size. `--ld-tag-r2 1` keeps only identical encoded genotypes.
Lower values enable approximate LD compression and can change which pairs are
screened.
