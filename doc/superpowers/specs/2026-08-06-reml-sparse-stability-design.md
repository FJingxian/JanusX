# REML Sparse Stability Design

**Goal:** Make one-hot sparse REML variance estimation numerically reliable on the balanced rice data and prevent multi-random-effect CSC/AMD panics.

## Scope

1. The Python sparse one-hot REML path will optimize positive random-to-residual variance ratios in log space. A single random term will use a bounded scalar (bracketed) minimization; multiple terms will use bounded log-space L-BFGS-B with a post-fit objective/gradient/convergence check.
2. The Rust sparse symbolic analysis will validate and sort the lower CSC pattern before ordering. The one-hot incidence path will disable faer's unstable dense-node AMD heuristic while retaining AMD ordering; ordinary sparse-GRM analysis keeps the default heuristic.
3. The existing general sparse-GRM path will keep its current AMD behavior unless the same validated fallback is needed.

## Success criteria

- The rice base model estimates the lines/residual ratio near the lme4 value (~8.225) rather than the current raw-scale result (~7.985).
- A two-factor one-hot cache with hundreds of line levels constructs without a `faer::amd` panic and can evaluate a fit/objective.
- Existing C-order, fixed-only, G×E/G×C, and integration tests remain green.
- A regression test fails on the old implementation and passes after the fix.

## Non-goals

- Do not redesign the REML CLI or change variance-component definitions.
- Do not make dense factorization the default for large random-effect designs.
- Do not hide genuine non-positive-definite factorization errors; only AMD symbolic-order failures get the deterministic fallback.
