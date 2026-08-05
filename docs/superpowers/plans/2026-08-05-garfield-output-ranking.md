# GARFIELD Output Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GARFIELD choose final reportable rules by `raw_score - output_penalty`, while retaining `gain - search_penalty > eps` for interaction expansion pruning.

**Architecture:** `evaluate_logic_unit_prepared_continuous` already computes `ranked_hits` using `final_output_score_for_candidate` and sorts it by output score. The fix keeps that ordering through `select_reportable_ranked_hits`; only the search path in `src/garfield/bs.rs` remains responsible for gain/search-penalty pruning.

**Tech Stack:** Rust, PyO3 GARFIELD implementation, Cargo unit tests.

## Global Constraints

- Preserve unrelated worktree changes in `python/janusx/pyBLUP/blup.py`, `python/janusx/script/reml.py`, and `src/garfield/bs.rs`.
- Do not change beam ranking, interaction-gain calculation, null calibration, or layer-one singleton retention.
- Use `apply_patch` for manual source edits.
- Verify with `cargo fmt --all -- --check` and focused GARFIELD Rust tests before claiming completion.

---

### Task 1: Lock the output-score contract with regression tests

**Files:**
- Modify: `src/garfield/mod.rs:16841-16882`

**Interfaces:**
- Consumes: `select_reportable_ranked_hits`, `BeamRuleCandidate`, and the existing `ranked_hits` `(candidate_index, output_score)` representation.
- Produces: tests that fail under the current raw-score re-sort and pass when output-score ordering is preserved.

- [ ] **Step 1: Change the ranking regression to encode the desired contract**

Use the existing `test_select_reportable_ranked_hits_uses_highest_unpenalized_raw_score` fixture, but make the combination's stored output score higher than the singleton's output score while its raw score is lower. Assert that the combination is selected. The expected assertion must be:

```rust
assert_eq!(selected, vec![(0usize, 12.0_f64)]);
```

Keep `combo.test.raw_score = 5.0` and `singleton.test.raw_score = 20.0`; this makes the test fail specifically if the function re-sorts by raw score.

- [ ] **Step 2: Make top-k cutoff verify output-score ordering**

Update `test_select_reportable_ranked_hits_applies_topk_to_singleton_only_unit` so the candidate output scores remain ordered as `-0.10, -0.20, -0.30, -0.40`, but set the first candidate's raw score below the second candidate's raw score. Keep the expected top-two indices `(0, 1)`. This catches raw-score reordering before the cutoff.

- [ ] **Step 3: Run the focused tests and confirm the expected RED failure**

Run:

```bash
cargo test garfield::tests::test_select_reportable_ranked_hits --lib
```

Expected result: the changed output-score regression fails because the current implementation re-sorts non-raw candidates by `beam_hits[*idx].test.raw_score`.

### Task 2: Preserve the already-computed output ranking

**Files:**
- Modify: `src/garfield/mod.rs:4470-4505`

**Interfaces:**
- Consumes: the already output-penalty-adjusted and sorted `ranked_hits` from `evaluate_logic_unit_prepared_continuous`.
- Produces: final report hits whose order and top-k selection use `score - output_penalty`.

- [ ] **Step 1: Remove the non-raw raw-score re-sort**

Change `select_reportable_ranked_hits` so `base` is the existing `ranked_hits.to_vec()` for both `raw_design` and normal output. Retain the `top_rules_per_unit` logic, but change its tie-score closure to use the stored pair score:

```rust
let base = ranked_hits.to_vec();
...
|(_, score)| *score
```

Do not alter `final_output_score_for_candidate`, `rank_rule_score_components_with_bucket`, or any beam-search functions.

- [ ] **Step 2: Run the focused tests and confirm GREEN**

Run:

```bash
cargo test garfield::tests::test_select_reportable_ranked_hits --lib
```

Expected result: all selection tests pass, including the output-score-over-raw-score regression.

### Task 3: Verify search/output separation and formatting

**Files:**
- No additional source changes expected.

- [ ] **Step 1: Run existing search-pruning regressions**

Run:

```bash
cargo test garfield::bs::tests::test_min_gain_pruning_uses_gain_minus_null_penalty_scale --lib
cargo test garfield::bs::tests::test_interaction_gain_scoring_uses_ancestor_baseline_with_null_penalty --lib
```

Expected result: both pass, showing that interaction search still uses gain minus search penalty.

- [ ] **Step 2: Run the full GARFIELD Rust test module**

Run:

```bash
cargo test garfield --lib
```

Record any unrelated pre-existing failures separately; do not modify unrelated modules.

- [ ] **Step 3: Run formatting and inspect the final diff**

Run:

```bash
cargo fmt --all -- --check
git diff --check
git diff -- src/garfield/mod.rs
```

Confirm that only the intended selection function and its tests changed, and that existing user modifications remain untouched.

- [ ] **Step 4: Rebuild only if extension-level validation is required**

If the focused Rust tests pass and no Python-facing behavior is exercised, do not rebuild the extension. If a CLI smoke test is run, rebuild with the `jxfu` environment procedure from `janusx-project` and verify the CLI output separately.
