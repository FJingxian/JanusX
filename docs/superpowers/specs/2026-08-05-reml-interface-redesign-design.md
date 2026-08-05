# REML Interface and Variance-Model Redesign

**Date:** 2026-08-05

**Status:** Approved for implementation planning

**Scope:** `jx reml`, shared phenotype-column selector spelling, dense/sparse narrow-h2 handoff

## 1. Goals

This redesign gives `jx reml` the same phenotype-input vocabulary as GWAS and GS, removes ambiguous legacy flags, and makes every requested model term reach every fitting stage. It also changes dense and sparse narrow-sense heritability estimation so that line BLUEs are not treated as error-free observations.

The implementation must:

1. replace the REML input contract with `-p/--pheno FILE` and `-n/--ncol COL`;
2. provide one term grammar for fixed, random, GxE, and GxC effects;
3. fix the current bug in which a within-line `-f/--fixed` term can be silently omitted from both fitting stages;
4. report raw random-effect variances in the log instead of forcing users to interpret variance proportions;
5. use first-stage estimation uncertainty when dense or sparse GRM REML estimates narrow h2; and
6. make `-n/--ncol` the public trait-selector spelling in other JanusX commands that use the shared selector helper.

## 2. Non-goals

- This change does not add a general R-style formula language.
- REML covariate flags do not load additional files. Users align and merge all observation-level variables before invoking JanusX.
- This change does not alter the existing meaning of GWAS/GS `-c/--cov`; only their shared trait-selector spelling changes from `--n` to `--ncol`.
- Random intercept/slope covariance matrices are not introduced. Each declared random term has one independent variance component.
- LOCO GRM construction is outside this change.

## 3. Command-line Contract

### 3.1 Required phenotype input

```text
-p, --pheno FILE
```

`FILE` is the only observation-level data source. Its first column is always the line/sample identifier. All phenotype, fixed-effect, random-effect, environmental, and continuous-gradient columns come from this table.

The existing flexible delimiter and optional-header reader is retained. A column can be selected by name when a header is present or by zero-based index excluding the first sample-ID column.

### 3.2 Trait selection

```text
-n, --ncol COL [COL ...]
```

Selectors accept names, zero-based indices excluding the sample-ID column, comma lists, repeated flags, and the existing trait range syntax. If omitted, REML selects usable numeric columns that are not referenced by any model term.

The shared JanusX trait-selector helper will expose only `-n/--ncol`. The old public `--n` alias is removed from all consumers of that helper.

### 3.3 Model terms

```text
-c,   --cov  TERM      fixed effect
-rc,  --rcov TERM      random nuisance effect
-gxe, --gxe  TERM      random Line-by-discrete-environment effect
-gxc, --gxc  COL       random Line-by-continuous-gradient slope
```

Each option is repeatable. A comma list expands into independent terms:

```bash
-gxe loc,year
```

is equivalent to:

```bash
-gxe loc -gxe year
```

In contrast, a colon creates one interaction term:

```bash
-gxe loc:year
```

means `Line x combined(loc, year)` and has one variance component.

These flags accept only phenotype-table column selectors. A token that resolves to a filesystem path is not loaded as a covariate file and produces an unknown-column error.

### 3.4 GRM inputs

```text
-k,   --grm        FILE
-spk, --grm-sparse FILE
```

Both flags are optional and mutually exclusive:

- neither: fit the observation-level mixed model and produce BLUE/BLUP, but do not estimate narrow h2 or GBLUP;
- `-k`: run noise-corrected dense-GRM narrow REML and GBLUP;
- `-spk`: run noise-corrected sparse-GRM narrow REML and GBLUP;
- both: fail before loading or fitting data.

The obsolete single-dash `-grm` alias is removed. `--spk-mode` remains accepted but is shown only in developer help (`jx reml -h -dev`).

### 3.5 Removed REML flags

The following flags are removed rather than silently translated:

```text
-file/--file
-l/--line
-e/--env
-f/--fixed
-r/--random
-grm
--n
```

Argparse errors should point users to the replacement when practical.

## 4. Term Grammar and Type Inference

### 4.1 Atomic column types

Type inference runs once per selected source column after missing-value normalization:

1. a column that cannot be fully converted to finite numeric values is categorical;
2. a numeric integer-valued column is categorical when its unique count is at most 10 and at most 5% of its valid observation count;
3. every other numeric column is continuous.

The inferred type, valid count, unique count, and reason are written to the REML log. Constant columns fail validation when they cannot contribute an estimable term.

### 4.2 Colon interactions

For `A:B`:

| A type | B type | Compiled effect |
|---|---|---|
| categorical | categorical | one combined categorical factor from observed `(A, B)` levels |
| continuous | continuous | one numeric column containing the elementwise product |
| categorical | continuous | category-specific slope columns |
| continuous | categorical | category-specific slope columns |

An interaction does not implicitly add either main effect. Users request hierarchy explicitly, for example:

```bash
-c treatment -c temperature -c treatment:temperature
```

For a fixed categorical-by-continuous term, the resulting columns are fixed slopes. For a random categorical-by-continuous term, they form one random-slope term with one shared variance component; a random categorical intercept must be requested separately.

### 4.3 GxE and GxC validation

`-gxe` accepts an atomic categorical column or a colon expression whose compiled result is categorical. It is compiled as `Line x ENV` and receives its own variance component.

`-gxc` accepts continuous atomic columns. Each column is mean-centered, but not rescaled, before constructing line-specific slope columns. Centering separates the baseline Line intercept from the reaction-norm slope while preserving the original measurement unit. A categorical or constant `-gxc` input is rejected with an actionable error.

## 5. Internal Representation

CLI strings are parsed into typed term specifications before any model is fitted. Each specification records:

```text
kind: fixed | random | gxe | gxc
source columns
inferred source types
display label
compiled matrix/factor strategy
variance-component identity (for random terms)
```

All overlap checks and missing-data requirements operate on source columns, not generated dummy-column names. A source column may not simultaneously be a selected trait. Reusing a column across compatible model terms is allowed only when the terms are genuinely distinct (for example `-c temperature` with `-c treatment:temperature`); exact duplicate terms are rejected.

This representation is the single source of truth for configuration logging, design-matrix construction, the BLUP fit, and the BLUE refit. No fitting stage may independently re-parse the original CLI tokens.

## 6. Observation-level Model

For each trait, the primary mixed model is conceptually:

```text
y = X(cov) beta
  + Z_line u_line
  + sum Z_rc u_rc
  + sum Z_gxe u_gxe
  + sum Z_gxc u_gxc
  + e
```

Each declared random term receives a separate non-negative variance component. `u_line` is the baseline line effect and `e` is residual noise.

The BLUP fit estimates all variance components and line BLUPs. The BLUE refit uses the same fixed and nuisance-random terms but treats Line as fixed. In particular, a fixed covariate that varies within line remains in the observation-level BLUE model. This invariant fixes the present `-f loc` propagation bug.

The log reports, per trait:

- baseline Line variance;
- every `-rc` variance;
- every `-gxe` variance;
- every `-gxc` variance;
- residual variance;
- convergence and boundary diagnostics.

No broad-sense PVE or per-component variance proportion is required. Users can compute derived ratios from logged raw variances.

## 7. Narrow-h2 Handoff

### 7.1 First-stage uncertainty

The BLUE refit produces line estimates and their estimation uncertainty under the fitted observation-level covariance model. The scalable handoff uses the diagonal of the line-BLUE covariance/PEV, denoted `D_stage1`. Because it is computed from the fitted mixed-model covariance, it incorporates residual, replication, `-rc`, `-gxe`, and `-gxc` uncertainty projected onto the line scale.

Missing or non-finite uncertainty values are an error; they must not be silently replaced by zero. The log records the minimum, median, mean, and maximum diagonal uncertainty.

### 7.2 Dense and sparse models

Both GRM routes fit the same conceptual line-level covariance:

```text
Var(line BLUE) = Va K + Vline I + D_stage1
```

where `Va` is additive genomic variance and `Vline` is remaining independent line-level variance. `D_stage1` is known during this fit and is never optimized as another free component.

The dense route uses the existing exact joint REML machinery after wiring `D_stage1` into the active path. The sparse route must implement the corresponding heteroskedastic diagonal correction rather than falling back to the former error-free-BLUE objective.

The reported phenotype-scale narrow h2 is:

```text
h2_narrow = Va / (Va + Vline + mean(diag(D_stage1)))
```

The latent ratio `Va / (Va + Vline)` may be emitted only as a developer diagnostic and must be clearly labeled so it cannot be confused with reported phenotype-scale h2.

The log reports narrow h2, the fitted line-level components, optimizer status, boundary warnings, GRM alignment details, and uncertainty diagnostics. GBLUP uses the same corrected fitted model.

## 8. Outputs

Normal result artifacts retain the existing prefix conventions.

Without a GRM:

- line BLUE output;
- line BLUP output;
- REML log;
- no narrow-h2 or GBLUP output.

With a dense or sparse GRM:

- line BLUE output;
- line BLUP output;
- GBLUP output;
- trait-level narrow-h2 summary where required by the existing result contract;
- REML log containing all detailed variances and diagnostics.

Raw random-effect variances and residual variance belong in the log, not in a forced variance-proportion result table.

## 9. Validation and Failure Behavior

Validation occurs before expensive fitting and must cover:

- phenotype file exists and has at least ID plus one data column;
- first-column sample IDs are non-missing;
- selected columns exist and numeric indices are in range;
- traits do not overlap model source columns;
- duplicate exact terms are rejected;
- interaction arity and source types are valid;
- `-gxe` resolves to a categorical environment;
- `-gxc` resolves to a non-constant continuous column;
- the analysis retains observations and line replication sufficient for fitting;
- dense and sparse GRMs are mutually exclusive and align with phenotype IDs;
- every declared fixed/random term appears in the compiled model metadata;
- optimizer failures, non-finite variances, and invalid PEVs are explicit errors.

Boundary variance estimates are permitted but produce warnings naming the affected term.

## 10. Test Checklist

### 10.1 CLI contract

- [ ] `jx reml -p pheno.tsv -n PH` parses without a GRM.
- [ ] `--pheno`, `--ncol`, `--cov`, `--rcov`, `--gxe`, and `--gxc` long forms parse.
- [ ] `-file`, `--file`, `-l`, `-e`, `-f`, `-r`, `-grm`, and `--n` are rejected.
- [ ] shared GWAS/GS and other trait-selector consumers advertise `-n/--ncol`, not `--n`.
- [ ] neither GRM is valid; each GRM alone is valid; both together fail early.
- [ ] normal help hides `--spk-mode`; `-h -dev` shows it.
- [ ] covariate flags reject a file path as an unknown column.

### 10.2 Selector and grammar parsing

- [ ] names, zero-based indices excluding ID, comma lists, and repeated flags resolve consistently.
- [ ] `-gxe loc,year` produces two terms and two variance-component identities.
- [ ] `-gxe loc:year` produces one combined term and one identity.
- [ ] duplicate exact terms fail with a clear message.
- [ ] unknown names and out-of-range indices fail before fitting.
- [ ] phenotype/model-column overlap fails before fitting.

### 10.3 Type inference and interaction matrices

- [ ] string and numeric-low-cardinality columns infer categorical.
- [ ] ordinary numeric columns infer continuous.
- [ ] inference boundary cases at 10 levels and 5% are deterministic.
- [ ] categorical-by-categorical produces observed combination levels.
- [ ] continuous-by-continuous equals elementwise multiplication.
- [ ] categorical-by-continuous produces category-specific slopes for fixed and random terms.
- [ ] interactions do not implicitly add main effects.
- [ ] categorical/constant `-gxc` and continuous `-gxe` inputs fail clearly.
- [ ] GxC centering is correct and does not rescale units.

### 10.4 Propagation regression tests

- [ ] a within-line fixed covariate appears in the observation-level BLUP design.
- [ ] the same fixed covariate appears in the line-BLUE refit.
- [ ] removing that covariate changes BLUEs in a synthetic confounded dataset.
- [ ] every requested `-rc`, `-gxe`, and `-gxc` term reaches the mixed-model fit.
- [ ] compiled-model metadata and configuration logging contain every requested term.
- [ ] no term can be classified and then silently dropped.

### 10.5 Variance and output behavior

- [ ] no-GRM analysis writes BLUE and BLUP but no GBLUP/narrow-h2 artifact.
- [ ] log contains Line, each random term, and residual raw variance.
- [ ] normal output does not require random-effect variance proportions.
- [ ] dense and sparse GRM analyses write GBLUP and narrow-h2 diagnostics.
- [ ] multiple GxE/GxC terms remain separately labeled in the log.
- [ ] boundary estimates are labeled as warnings, not hidden.

### 10.6 Narrow-h2 correction

- [ ] first-stage PEV diagonal is finite, non-negative, and nonzero in a noisy replicated simulation.
- [ ] dense narrow REML actually receives `D_stage1` (regression test against the former ignored argument).
- [ ] sparse narrow REML receives the same diagonal correction semantics.
- [ ] zero `D_stage1` reproduces the corresponding legacy/error-free objective within tolerance.
- [ ] increasing first-stage uncertainty lowers reported phenotype-scale narrow h2 when other fitted components are held fixed.
- [ ] reported h2 exactly matches `Va / (Va + Vline + mean(D_stage1))`.
- [ ] latent variance ratio, if logged in dev mode, is distinctly labeled.
- [ ] corrected dense/sparse fits remain finite on synthetic boundary cases and warn on boundary solutions.
- [ ] a regression dataset reproducing narrow PVE near 1 no longer reaches 1 solely because BLUE uncertainty was discarded.

### 10.7 Integration and numerical checks

- [ ] existing single-environment no-covariate REML workflow remains usable under the new flags.
- [ ] multi-environment fixed, random, GxE, and GxC example completes end to end.
- [ ] phenotype and GRM IDs are reordered/aligned correctly and mismatches fail explicitly.
- [ ] multi-trait runs isolate missingness and model state per trait.
- [ ] deterministic test inputs produce stable results across supported thread counts.
- [ ] focused REML tests, shared CLI tests, and relevant GWAS/GS parser tests pass.

## 11. Documentation Examples

No GRM, fixed environments and random blocks:

```bash
jx reml -p pheno.tsv -n PH -c year,loc -rc block -o ph
```

Discrete GxE with a combined environment and dense GRM:

```bash
jx reml -p pheno.tsv -n PH -c year:loc -gxe year:loc \
  -k maize.cGRM.npy -o ph.gxe
```

Continuous reaction norm and sparse GRM:

```bash
jx reml -p pheno.tsv -n PH -c temperature -gxc temperature \
  -spk maize.cGRM.spgrm -o ph.gxc
```

Separate GxE components:

```bash
jx reml -p pheno.tsv -n PH -c loc,year -gxe loc,year -o ph.multi
```
