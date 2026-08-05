# FvLMM Scan Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Measure and optimize the JanusX GWAS FvLMM scan on macOS without changing statistical results or default output behavior.

**Architecture:** Keep the existing BED unified pipeline and single-SGEMM default. Add stage telemetry first, then make projection/association thread budgets explicit, then add an opt-in tiled-CBLAS projection for A/B benchmarking. Implement fusion or multi-trait reuse only if measurements show a stable wall-time benefit.

**Tech Stack:** Rust/PyO3, Rayon, CBLAS/Accelerate, Python GWAS workflow, pytest-compatible local tests, Cargo tests, `maturin develop --release`.

## Global Constraints

- Preserve SNP order, filtering, output schema, and default numerical results.
- Keep tiled projection opt-in until a benchmark proves it is faster.
- Do not modify or commit unrelated GARFIELD/docs/test changes.
- Use the `jxfu` environment for Python and CLI validation.
- Do not export `DYLD_LIBRARY_PATH` when running Python or `jx` on macOS.

### Task 1: Add failing tests for BED timing and thread-plan semantics

**Files:**
- Modify: `src/stats/fvlmm.rs` test module
- Modify: `python/janusx/assoc/workflow.py` testable thread-plan helpers
- Test: `test/test_fvlmm_scan_optimization.py` (local-only; do not commit)

**Interfaces:**
- Rust helper returns a deterministic stage-timing summary formatter and tile/thread decisions.
- Python helper returns explicit `{blas_threads, rayon_threads, proj_threads, assoc_threads}` without relying on a shared hidden override.

- [ ] Write tests asserting that `full` maps both Rust stages to the requested thread budget, `blas_t_rayon_1` maps projection and association exactly as documented, and invalid/zero values fall back to one.
- [ ] Write a Rust test for timing accounting where all stage durations are zero and the reported `other` duration is non-negative.
- [ ] Run the focused tests and confirm they fail because the new helpers/fields do not exist.

### Task 2: Implement BED unified stage telemetry

**Files:**
- Modify: `src/stats/fvlmm.rs:750-870, 2580-2970`
- Modify: `python/janusx/assoc/workflow_model_stream.py` to document the environment switch
- Test: `test/test_fvlmm_scan_optimization.py`

**Interfaces:**
- Environment: `JX_FVLMM_BED_STAGE_TIMING=1`.
- Output: one stderr line beginning `FvLMM BED timing:` with named stage durations and execution metadata.

- [ ] Add the failing telemetry assertions from Task 1 and run them.
- [ ] Add a `FvlmmBedStageTiming` accumulator for decode, projection, association, TSV, and wall-clock nanoseconds.
- [ ] Instrument the BED producer/consumer without changing data ownership or pipeline ordering.
- [ ] Emit timing only when the environment variable is truthy; retain existing profiling behavior.
- [ ] Run the focused Rust/Python tests and confirm they pass.

### Task 3: Decouple projection and association thread budgets

**Files:**
- Modify: `src/stats/fvlmm.rs:1273-1290, 1870-2260, 2580-2600, 5080-5100`
- Modify: `python/janusx/assoc/workflow.py:5342-5385`
- Modify: `python/janusx/assoc/workflow_model_stream.py` stage comments/logging
- Test: `test/test_fvlmm_scan_optimization.py`

**Interfaces:**
- Explicit Rust stage controls: `JX_FVLMM_PROJ_THREADS` and `JX_FVLMM_ASSOC_THREADS`.
- `JX_FVLMM_SCAN_STAGE` remains accepted, but its displayed plan and Rust effective plan must agree.

- [ ] Add failing tests showing that `blas_t_rayon_1` currently leaks a Rayon=1 setting into projection.
- [ ] Implement a single resolver that computes projection and association budgets once per scan, with explicit variables taking precedence and CLI threads as fallback.
- [ ] Pass the resolved budgets to both BLAS guards and Rayon pools; avoid reading `JX_MLM_RUST_THREADS` inside the hot path.
- [ ] Emit resolved budgets in the BED timing line and verbose CLI diagnostics.
- [ ] Run focused tests, then run a small CLI scan with `full`, `generic`, and `blas_t_rayon_1` to verify behavior and result hashes.

### Task 4: Add opt-in tiled-CBLAS projection

**Files:**
- Modify: `src/stats/fvlmm.rs:390-445`
- Modify: `python/janusx/assoc/workflow_model_stream.py` only if a user-facing opt-in is needed
- Test: `test/test_fvlmm_scan_optimization.py`

**Interfaces:**
- Environment: `JX_FVLMM_ROTATE_KERNEL=blas_tiled` selects the experimental path; all other values use the current single-GEMM path.
- Tiled path accepts existing `proj_threads` and preserves row-major f32 output.

- [ ] Add failing tests for tile partitioning at rows 0, rows <= tile, and rows not divisible by tile count.
- [ ] Implement row-tiled CBLAS calls with a single-thread BLAS guard per tile and projection-pool scheduling; use an adaptive minimum tile size.
- [ ] Ensure no nested BLAS oversubscription and no overlapping writes.
- [ ] Run kernel-level numerical comparison against the default path.
- [ ] Benchmark default versus tiled at `-t 1,2,4,8` on the example data and keep tiled opt-in if it does not win consistently.

### Task 5: Evaluate fusion and multi-trait reuse

**Files:**
- Inspect/modify only if justified: `src/stats/fvlmm.rs`, `python/janusx/assoc/workflow_model_stream.py`
- Test: `test/test_fvlmm_scan_optimization.py` and local benchmark scripts (not committed)

**Interfaces:**
- No new default behavior unless the benchmark and numerical tests pass.

- [ ] Use BED timing output to determine whether projection or association dominates.
- [ ] Prototype block-local fused reduction in a test-only branch/path and compare wall time, RSS, and numerical output.
- [ ] Check whether multiple traits share the same sample mask/eigenspace before considering rotation reuse.
- [ ] Implement only a measured win; otherwise record the result and leave the stable path unchanged.

### Task 6: Full verification and handoff

**Files:**
- Modify: none unless verification exposes a regression

- [ ] Run `cargo fmt --all -- --check`.
- [ ] Run focused FvLMM Rust tests with `DYLD_LIBRARY_PATH` unset for Python-facing runs.
- [ ] Rebuild the extension using the `jxfu` environment and run `python -m py_compile` on edited Python modules.
- [ ] Run the example FvLMM scan with timing enabled and compare output hashes across thread counts and kernels.
- [ ] Run the relevant broader tests, record unrelated failures separately, and summarize whether Accelerate appears saturated.
