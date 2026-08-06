# REML Sparse Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize sparse REML variance optimization and multi-random one-hot CSC symbolic analysis.

**Architecture:** Keep the existing sparse cache objective and fitted-state API. Add a Python optimizer helper that works in log variance-ratio coordinates and selects a bracketed scalar solver for one random component, with a joint log-space solve plus bounded refinements for multiple components. In Rust, preserve lower CSC validation and use a narrowly scoped no-dense AMD control for one-hot incidence patterns while leaving general sparse-GRM ordering unchanged.

**Tech Stack:** Python 3.13, NumPy/SciPy, PyO3, Rust, faer 0.18.2, pytest/unittest, cargo.

## Global Constraints

- Preserve existing variance-ratio convention: residual variance is fixed to 1 during optimization and recovered by the fitted state.
- Keep all NumPy arrays passed to Rust C-contiguous and `float64`.
- Never let a Rust panic cross the PyO3 boundary for a user-supplied valid one-hot model.
- Use `apply_patch` for source edits and validate through the `jxfu` environment.

### Task 1: Regression tests

**Files:**
- Modify: `test/test_reml_fix_regression.py`
- Modify: `src/stats/heritability.rs` or `src/math/cholesky.rs` test module

- [ ] Add a Python test that constructs the rice-like one-random sparse cache/objective and asserts the log-scale optimizer returns a ratio in the lme4 neighborhood.
- [ ] Add a Rust test for a two-factor one-hot lower CSC pattern with at least 500 line levels and six environment levels; the symbolic analysis must return `Ok` and preserve lower sorted CSC invariants.
- [ ] Run each new test before implementation and record the expected failure (old ratio/panic).

### Task 2: Log-scale sparse REML

**Files:**
- Modify: `python/janusx/pyBLUP/blup.py`

- [ ] Add a bounded log-variance objective wrapper around `SparseOneHotBlupCache.objective`.
- [ ] Use `scipy.optimize.minimize_scalar(method="bounded")` for one random component; use bounded L-BFGS-B in log space for multiple components.
- [ ] Reject non-finite, non-positive, non-converged, or materially non-stationary solutions and return `None` so the existing fallback remains available.
- [ ] Preserve the returned `theta`, fit arrays, and `OptimizeResult` shape expected by `BLUP._fit`.

### Task 3: Multi-random CSC/AMD safety

**Files:**
- Modify: `src/math/cholesky.rs`
- Modify if needed: `src/stats/heritability.rs`

- [ ] Keep lower-triangle validation, explicitly verify each column is strictly increasing and contains its diagonal before symbolic analysis.
- [ ] Route one-hot CSC analysis through a dedicated AMD control with dense-node detection disabled; keep the default AMD control for ordinary sparse GRM CSC.
- [ ] Return ordinary `Result` errors for malformed CSC or failed numeric factorization; do not catch those as successful ordering.

### Task 4: Verification

- [ ] Run `cargo fmt --all -- --check`.
- [ ] Rebuild with `maturin develop --release --locked --features python-extension` in `jxfu`.
- [ ] Run focused Rust tests and Python regression/integration tests.
- [ ] Run rice base REML, rice 300-line `-gxe loc`, and `-rc year:loc` smoke tests.
- [ ] Compare full rice variance components and BLUE/BLUP with the saved lme4 references.
