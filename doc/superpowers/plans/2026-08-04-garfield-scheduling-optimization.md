# GARFIELD Scheduling Optimization Implementation Plan

> **For agentic workers:** This plan is executed inline in the current workspace. It is stored under the repository's canonical `doc/superpowers/` documentation tree.

**Goal:** Increase effective CPU utilization and reduce GARFIELD scan/permutation wall time without changing scoring, pruning, random seeds, calibration semantics, or output ordering.

**Architecture:** Add optional task-level profiling around the existing outer schedulers. Replace scan's coarse serial chunk loop with bounded, cost-aware work items while retaining one-level parallelism. Add a cache-aware flattened permutation task path for low-slot and heterogeneous workloads, preserving the per-repeat adaptive-calibration barrier.

**Tech Stack:** Rust, Rayon, existing GARFIELD JSON diagnostics, cargo fmt/test, benchmark TSV hash comparison.

---

### Task 1: Add scheduler profiling data structures

**Files:**
- Modify: `src/garfield/mod.rs`
- Test: `src/garfield/mod.rs` unit tests near existing helper tests

- [ ] Add optional environment-gated counters for scan chunk and permutation task timing.
- [ ] Track count, total elapsed, min, max, and a bounded sample list sufficient for p50/p95 summaries.
- [ ] Keep profiling disabled by default and ensure disabled mode performs no per-task allocation.
- [ ] Add unit tests for percentile calculation and empty/single-sample behavior.
- [ ] Run the focused Rust test/build command with the configured Python interpreter.

### Task 2: Instrument current schedulers and establish A/B baseline

**Files:**
- Modify: `src/garfield/mod.rs`

- [ ] Wrap each current scan chunk and permutation slot task with the optional profiler.
- [ ] Emit scheduler summaries into the existing trait JSON only when profiling is enabled.
- [ ] Run the existing Chr1 benchmark with profiling enabled.
- [ ] Record elapsed time, effective concurrency, scheduler p95/max, RSS, and rules TSV hash before changing dispatch.

### Task 3: Improve scan scheduling

**Files:**
- Modify: `src/garfield/mod.rs`

- [ ] Introduce bounded scan work items using unit-index ranges with a workload estimate based on unit row counts.
- [ ] Use enough work items to keep at least 8-16 schedulable items per worker, without creating one Python callback or allocation per SNP.
- [ ] Preserve original unit order in the collected result vector.
- [ ] Keep nested beam parallelism disabled for large scans.
- [ ] Add a deterministic scheduling test showing all unit indices are covered exactly once.
- [ ] Run Chr1 A/B and compare output hashes.

### Task 4: Improve permutation scheduling

**Files:**
- Modify: `src/garfield/mod.rs`

- [ ] Extend the flattened permutation worker to accept and reuse the prepared bit cache.
- [ ] Dispatch `(slot, repeat)` tasks when slots are fewer than effective threads or when task-size imbalance is detected.
- [ ] Preserve repeat-by-repeat collection and adaptive calibration barriers.
- [ ] Keep deterministic seed derivation and restore results by repeat before calibration.
- [ ] Add tests for task coverage and repeat ordering.
- [ ] Run low-slot and normal-slot A/B benchmarks and compare output hashes.

### Task 5: Verify and report

**Files:**
- No production files beyond Tasks 1-4

- [ ] Run cargo fmt check.
- [ ] Run focused Rust tests/build; report the existing local Python dylib runtime limitation if still present.
- [ ] Run Chr1 and full benchmark comparisons.
- [ ] Report CPU utilization proxy, wall time, RSS, result hash, and any remaining bottleneck.
- [ ] Commit only production changes; keep generated benchmark artifacts out of the repository.
