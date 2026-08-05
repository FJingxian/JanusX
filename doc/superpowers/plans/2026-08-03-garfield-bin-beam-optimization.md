# GARFIELD BIN Beam Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the cost of the ordinary `BIN` beam expansion used by `jx garfield -w` without changing results.

**Architecture:** Add a lightweight ordinary beam state for the parallel expansion. Workers score and prune candidates without cloning bitsets; after global deduplication and truncation, materialize only retained states. Keep the existing serial implementation as the semantic reference and compare both paths in a Rust test.

**Tech Stack:** Rust, Rayon, Cargo release tests, JanusX `jx garfield` CLI.

---

### Task 1: Add a failing ordinary-path equivalence test

**Files:**
- Modify: `src/garfield/bs.rs` test module near `test_standard_fuzzy_parallel_expansion_matches_serial`

- [ ] **Step 1: Construct a small packed 0/1 dataset and ordinary beam parents.**

- [ ] **Step 2: Run the current serial expansion and a new optimized entry point.**

- [ ] **Step 3: Compare rule keys, train support, scores, and ordering.**

- [ ] **Step 4: Run the targeted test and confirm it fails because the optimized ordinary entry point does not yet exist.**

### Task 2: Implement deferred ordinary-state materialization

**Files:**
- Modify: `src/garfield/bs.rs` in `expand_beam_once`

- [ ] **Step 1: Introduce an internal lightweight ordinary beam state containing all fields needed by `cmp_state` and pruning.**

- [ ] **Step 2: Move worker-local and global deduplication to the lightweight state.**

- [ ] **Step 3: Materialize `combined_train` only after global top-k truncation.**

- [ ] **Step 4: Preserve the current serial fallback and final `filter_beam_candidates` call.**

- [ ] **Step 5: Run the targeted equivalence test and confirm it passes.**

### Task 3: Verify the CLI path

**Files:**
- No production files beyond Task 2.

- [ ] **Step 1: Build the release extension with Python 3.13.**

- [ ] **Step 2: Run a Chr1 or explicit `--bimrange` CUBIC GARFIELD command with the same genotype, phenotype, GRM, trait, width, and thread count before and after the change.**

- [ ] **Step 3: Compare rules and FvLMM outputs byte-for-byte.**

- [ ] **Step 4: Record wall-clock and resource changes, and report if the optimization is not beneficial.**
