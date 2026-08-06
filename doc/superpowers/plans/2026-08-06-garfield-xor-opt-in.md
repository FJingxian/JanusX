# GARFIELD XOR Opt-in Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make XOR rule expansion an explicit `--xor-search` opt-in while keeping the default GARFIELD search AND-only and preserving the separate XOR-substate filter.

**Architecture:** Resolve the CLI policy in Python, pass it as a dedicated boolean through the PyO3 boundary, and store it in `BeamSearchParams`. Rust beam expansion helpers will consume that field directly at every expansion and branch-count site, eliminating the global environment lookup for search policy. The legacy environment variable remains a Python-side force-off override.

**Tech Stack:** Python 3.13, argparse, pytest/unittest, PyO3, Rust, cargo, maturin.

## Global Constraints

- `--xor-search` defaults to `False`; only that flag enables XOR generation.
- `JX_GARFIELD_DISABLE_XOR_SEARCH=1` remains a compatibility force-off override.
- `-nf-xor/--nf-xor` continues to control only XOR substate filtering.
- Preserve the existing score, gain, penalty, output-ranking, and rule-canonicalization behavior.
- Preserve all unrelated uncommitted work in `src/garfield/bs.rs`, `src/garfield/mod.rs`, and existing REML documents.
- Use `apply_patch` for source edits and validate in the `jxfu` environment.

---

### Task 1: Lock the policy contract with failing tests

**Files:**
- Modify: `src/garfield/bs.rs:162-205, 9788-9855`
- Create: `test/test_garfield_xor_search.py`

**Interfaces:**
- Rust tests will require `beam_binary_ops_for_rule(&rule, xor_search_enabled)` and a default `BeamSearchParams` with `xor_search_enabled == false`.
- Python tests will exercise `_resolve_garfield_xor_search(requested)` and verify requested-off, requested-on, and force-off environment behavior.

- [ ] **Step 1: Write the failing Rust tests**

  Change the existing operation-policy tests so the helper takes an explicit boolean. Assert that a positive singleton returns `[And]` with `false` and `[And, Xor]` with `true`, and assert that `BeamSearchParams::default().xor_search_enabled` is false. Keep the existing invariants that a negated singleton and an already-XOR rule only expand with AND.

- [ ] **Step 2: Write the failing Python policy tests**

  Add tests equivalent to:

  ```python
  def test_xor_search_is_off_by_default(monkeypatch):
      monkeypatch.delenv("JX_GARFIELD_DISABLE_XOR_SEARCH", raising=False)
      assert _resolve_garfield_xor_search(False) is False

  def test_xor_search_requires_explicit_opt_in(monkeypatch):
      monkeypatch.delenv("JX_GARFIELD_DISABLE_XOR_SEARCH", raising=False)
      assert _resolve_garfield_xor_search(True) is True

  def test_legacy_disable_env_is_a_force_off(monkeypatch):
      monkeypatch.setenv("JX_GARFIELD_DISABLE_XOR_SEARCH", "1")
      assert _resolve_garfield_xor_search(True) is False
  ```

- [ ] **Step 3: Run the focused tests and verify they fail**

  Run:

  ```bash
  cargo test garfield::bs::tests::test_beam_binary_ops_for_rule --lib
  mamba run -n jxfu python -m pytest -q test/test_garfield_xor_search.py
  ```

  Expected result: compilation/test collection fails because the explicit Rust argument, default field, and Python resolver do not yet exist.

### Task 2: Implement explicit Python CLI policy and propagation

**Files:**
- Modify: `python/janusx/script/garfield.py:200-220, 2585-2635, 2670-2840, 3220-3245, 3435-3455`
- Modify: `src/garfield/mod.rs:12420-12490, 14708-14955`

**Interfaces:**
- Add `_resolve_garfield_xor_search(requested: bool) -> bool` in the Python script. It returns `requested and not _env_truthy("JX_GARFIELD_DISABLE_XOR_SEARCH")`.
- Add `--xor-search` with `action="store_true"`, default false, and a user-visible description.
- Add `xor_search_enabled: bool` to the owned Rust pipeline and `xor_search=false` to the PyO3 signature. Pass the value into the owned function and then into `BeamSearchParams`.

- [ ] **Step 1: Add the minimal Python resolver and CLI option**

  Add a small truthy-environment parser local to `garfield.py`, add `--xor-search` beside the existing GARFIELD search options, and after `parse_known_args()` preserve the requested value while replacing the runtime value with the effective force-off-resolved value. Do not alter `disable_xor_substate_filter`.

- [ ] **Step 2: Thread the boolean through the PyO3 function**

  Add the new defaulted keyword after `filter_xor_substates` in the PyO3 signature, add the corresponding Rust function parameter, and pass it to `garfield_logic_search_bed_owned`. Add the keyword `xor_search=bool(args.xor_search)` at the Python call site.

- [ ] **Step 3: Record the effective policy**

  Add `xor_search_enabled` and `xor_search_requested` to the per-trait manifest metadata and the aggregate manifest metadata, while retaining `xor_substate_lmaf_filter` unchanged.

### Task 3: Make Rust beam expansion consume the explicit policy

**Files:**
- Modify: `src/garfield/bs.rs:35-205, 315-365, 3750-8135`
- Modify: `src/garfield/mod.rs:13012-13040`

**Interfaces:**
- `BeamSearchParams` gains `pub xor_search_enabled: bool`, defaulting to `false`.
- `beam_binary_ops_for_rule(rule, xor_search_enabled)` is the only operation-policy helper used by active search.
- `beam_child_branch_count_for_rule(rule, xor_search_enabled)` uses the same policy for capacity estimates.

- [ ] **Step 1: Replace the global search-policy lookup**

  Remove the `OnceLock`/`JX_GARFIELD_DISABLE_XOR_SEARCH` lookup from the beam operation selector, keep the existing rule-specific AND-only guards, and make the selector accept the explicit boolean. Keep unrelated `OnceLock` environment controls intact.

- [ ] **Step 2: Add the default-disabled parameter**

  Add `xor_search_enabled: false` to `BeamSearchParams::default()` and set the field from `xor_search_enabled` in the main GARFIELD beam-parameter construction. All cloned preparation/permutation/posterior/scan parameter sets will inherit the same value.

- [ ] **Step 3: Update every expansion and capacity call site**

  Pass `params.xor_search_enabled` to all binary-op loops and branch-count calculations in both binary and fuzzy beam paths. The only allowed operation differences are `[And]` when disabled and `[And, Xor]` for eligible rules when enabled.

- [ ] **Step 4: Run the Rust focused tests and verify they pass**

  Run:

  ```bash
  cargo fmt --all -- --check
  cargo test garfield::bs::tests::test_beam_binary_ops_for_rule --lib
  ```

  Expected result: all focused policy tests pass.

### Task 4: Rebuild, run Python tests, and perform CLI smoke verification

**Files:**
- Test: `test/test_garfield_xor_search.py`
- Verify: generated GARFIELD manifest/help output; no source changes expected.

- [ ] **Step 1: Run Python syntax and policy tests**

  ```bash
  mamba run -n jxfu python -m py_compile python/janusx/script/garfield.py test/test_garfield_xor_search.py
  mamba run -n jxfu python -m pytest -q test/test_garfield_xor_search.py
  ```

- [ ] **Step 2: Rebuild the extension**

  ```bash
  PREFIX="$(mamba run -n jxfu python -c 'import sys; print(sys.prefix)')"
  CONDA_PREFIX="$PREFIX" PIP_BREAK_SYSTEM_PACKAGES=1 \
    "$PREFIX/bin/python" -m maturin develop --release --locked --features python-extension
  ```

- [ ] **Step 3: Verify the installed CLI exposes the option**

  ```bash
  mamba run -n jxfu jx garfield -h | rg -- '--xor-search'
  ```

  Verify the option is absent from the default search policy in the manifest and appears as enabled when a small existing GARFIELD input is run with `--xor-search`; run the same input without the flag and confirm the manifest records `xor_search_enabled: false`.

- [ ] **Step 4: Inspect the final diff and report evidence**

  Run `git diff --check`, `git status --short`, and the focused test/build commands again as needed. Report the commit containing only the design document separately from the still-uncommitted implementation changes; do not commit the implementation unless explicitly requested.
