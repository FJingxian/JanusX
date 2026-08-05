# REML Interface Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Replace the legacy REML CLI with the approved phenotype-column/effect-term interface, preserve every requested effect through BLUE/BLUP fitting, and use first-stage line-estimation uncertainty in dense and sparse narrow-h2 estimation.

**Architecture:** Keep REML orchestration in `python/janusx/script/reml.py`, but introduce a typed effect compiler that is the single source for fixed, random, GxE, and GxC matrices. Keep the existing BLUP and GRM backends, repair the BLUE stage to use the compiled fixed model, and route a validated line-level uncertainty diagonal into the existing joint dense objective plus a matching sparse objective adapter. Update the shared trait-selector helper so all GWAS/GS consumers expose `-n/--ncol`.

**Tech Stack:** Python 3.10+, argparse, pandas, NumPy, SciPy sparse/linalg, existing JanusX `BLUP`/`LMM`/sparse REML backends, stdlib `unittest`.

## Global Constraints

- `-p/--pheno FILE` is the only observation-level file input for REML; the first phenotype-table column is the sample/line ID.
- `-n/--ncol` is the public trait-selector spelling; remove the old public `--n` alias from shared consumers.
- `-c/-rc/-gxe/-gxc` resolve only phenotype-table column names or zero-based indices excluding the sample-ID column; no covariate-file loading.
- `-k/--grm` and `-spk/--grm-sparse` are optional and mutually exclusive; neither means BLUE/BLUP only, one means corrected narrow-h2/GBLUP.
- Remove REML `-file/--file`, `-l/--line`, `-e/--env`, `-f/--fixed`, `-r/--random`, `-grm`, and `--n` aliases instead of silently translating them.
- `--spk-mode` is accepted but shown only in developer help.
- Categorical × categorical compiles to a combined factor; continuous × continuous to an elementwise product; categorical × continuous to category-specific slopes.
- Automatic type inference is logged; integer low-cardinality columns (≤10 levels and ≤5% of valid rows) are categorical, other numeric columns are continuous.
- Raw random-effect and residual variances are written to the REML log; variance proportions are not required in normal output.
- Existing unrelated changes, especially `src/garfield/bs.rs`, must remain unstaged and untouched.

## Files and Responsibilities

- Modify `python/janusx/script/reml.py`: new parser, effect-spec dataclasses/compiler, design construction, BLUE propagation, stage-one PEV handoff, corrected dense/sparse narrow route, log/output labels.
- Modify `python/janusx/script/_common/cli_args.py`: expose only visible `-n/--ncol` in the shared trait selector helper.
- Modify any direct trait-selector consumers found by tests/search (GWAS, GS, ggval, and benchmark wrappers) only where they define or forward the old public `--n` spelling.
- Create `test/test_reml_interface.py`: parser, grammar, type inference, design propagation, uncertainty handoff, and output-contract tests using stdlib `unittest`.
- Create `test/test_reml_integration.py` only if the focused suite becomes too large for one module; use synthetic data and temporary files, never repository benchmark outputs.
- Update `doc/superpowers/specs/2026-08-05-reml-interface-redesign-design.md` only if implementation discovers a contract discrepancy; otherwise keep the approved spec unchanged.

### Task 1: Establish failing CLI and selector tests

**Files:**
- Create: `test/test_reml_interface.py`
- Modify: `python/janusx/script/_common/cli_args.py` only after the red run
- Modify: `python/janusx/script/reml.py` only after the red run

**Interfaces:**
- Tests import `build_parser` from `janusx.script.reml` and the shared selector helper through a small argparse parser.
- Expected parser namespace fields are `pheno`, `ncol`, `cov`, `rcov`, `gxe`, `gxc`, `grm`, and `grm_sparse`.

- [ ] **Step 1: Write the failing tests**

```python
class RemlCliContractTests(unittest.TestCase):
    def test_new_pheno_and_effect_flags_parse(self):
        args = reml.build_parser().parse_args([
            "-p", "pheno.tsv", "-n", "PH", "-c", "loc,year",
            "-rc", "block", "-gxe", "loc:year", "-gxc", "temperature",
            "-k", "kinship.npy",
        ])
        self.assertEqual(args.pheno, "pheno.tsv")
        self.assertEqual(args.ncol, ["PH"])
        self.assertEqual(args.cov, ["loc,year"])
        self.assertEqual(args.rcov, ["block"])
        self.assertEqual(args.gxe, ["loc:year"])
        self.assertEqual(args.gxc, ["temperature"])

    def test_legacy_reml_file_and_effect_flags_are_rejected(self):
        for flag in ("-file", "--file", "-f", "-r", "-e", "-l", "-grm", "--n"):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    reml.build_parser().parse_args([flag, "x"])

    def test_grm_inputs_are_optional_but_mutually_exclusive(self):
        args = reml.build_parser().parse_args(["-p", "p.tsv", "-n", "PH"])
        self.assertIsNone(args.grm)
        self.assertIsNone(args.grm_sparse)
        with self.assertRaises(SystemExit):
            reml.build_parser().parse_args([
                "-p", "p.tsv", "-n", "PH", "-k", "a.npy", "-spk", "a.spgrm"
            ])

    def test_shared_trait_selector_exposes_ncol_only(self):
        parser = argparse.ArgumentParser()
        add_common_trait_selector_args(parser)
        self.assertEqual(parser.parse_args(["-n", "0"]).ncol, ["0"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--n", "0"])
```

- [ ] **Step 2: Run the focused tests and verify the expected red failure**

Run:

```bash
PYTHONPATH=python python -m unittest -v test.test_reml_interface.RemlCliContractTests
```

Expected: FAIL because the current parser requires `-file`, still uses `-p` for trait columns, accepts legacy aliases, and the shared helper still exposes `--n`.

### Task 2: Implement the new parser contract and shared `--ncol`

**Files:**
- Modify: `python/janusx/script/reml.py:build_parser`, `main` argument normalization
- Modify: `python/janusx/script/_common/cli_args.py:add_common_trait_selector_args`
- Test: `test/test_reml_interface.py::RemlCliContractTests`

**Interfaces:**
- `build_parser()` returns `args.pheno` as one required path, `args.ncol` as an extendable selector list, and repeatable lists for `cov`, `rcov`, `gxe`, and `gxc`.
- `main()` treats the first phenotype-table column as `line_col` and never reads `args.file`, `args.line`, `args.env`, `args.fixed`, or `args.random`.

- [ ] **Step 1: Implement only the parser and argument normalization needed by the failing tests.**
- [ ] **Step 2: Run the focused CLI tests.**

Run:

```bash
PYTHONPATH=python python -m unittest -v test.test_reml_interface.RemlCliContractTests
```

Expected: PASS, with old flags rejected and `-k`/`-spk` mutually exclusive.

- [ ] **Step 3: Run all parser consumers and compile edited modules.**

Run:

```bash
PYTHONPATH=python python -m py_compile python/janusx/script/reml.py python/janusx/script/_common/cli_args.py python/janusx/assoc/workflow.py python/janusx/gs/workflow.py
```

Expected: exit 0.

### Task 3: Add failing effect-compiler/type-inference tests

**Files:**
- Modify: `test/test_reml_interface.py`
- Modify: `python/janusx/script/reml.py` only after the red run

**Interfaces:**
- Add `_EffectSpec` (or an equivalent typed dataclass) with `kind`, `sources`, `label`, source types, and compiled strategy.
- Add `_parse_effect_specs(values, kind, columns, df)` returning ordered effect specs; comma splits independent terms, colon produces one interaction spec.
- Add `_compile_effect_matrix(df, spec, for_random)` returning a numeric dense/sparse matrix and stable display column names.

- [ ] **Step 1: Write these failing tests:**

```python
def test_categorical_pair_compiles_to_combined_factor(self):
    df = pd.DataFrame({"loc": ["A", "A", "B"], "year": ["Y1", "Y2", "Y1"]})
    spec = reml._parse_effect_specs(["loc:year"], "fixed", list(df.columns), df)[0]
    matrix, names = reml._compile_effect_matrix(df, spec, for_random=False)
    self.assertEqual(spec.source_types, ("categorical", "categorical"))
    self.assertEqual(matrix.shape[0], 3)
    self.assertEqual(len(names), 2)  # treatment-coded combined factor

def test_numeric_product_and_categorical_slopes(self):
    df = pd.DataFrame({"dose": [1.0, 2.0, 3.0], "temp": [2.0, 4.0, 5.0],
                       "treatment": ["A", "B", "A"]})
    product = reml._parse_effect_specs(["dose:temp"], "fixed", list(df.columns), df)[0]
    product_matrix, _ = reml._compile_effect_matrix(df, product, for_random=False)
    np.testing.assert_allclose(product_matrix.ravel(), [2.0, 8.0, 15.0])
    slope = reml._parse_effect_specs(["treatment:temp"], "random", list(df.columns), df)[0]
    slope_matrix, slope_names = reml._compile_effect_matrix(df, slope, for_random=True)
    self.assertEqual(slope_matrix.shape, (3, 2))
    self.assertEqual(len(slope_names), 2)

def test_gxe_requires_categorical_and_gxc_requires_continuous(self):
    with self.assertRaisesRegex(ValueError, "categorical"):
        reml._parse_effect_specs(["temp"], "gxe", ["temp"], pd.DataFrame({"temp": [1.0, 2.0]}))
    with self.assertRaisesRegex(ValueError, "continuous"):
        reml._parse_effect_specs(["loc"], "gxc", ["loc"], pd.DataFrame({"loc": ["A", "B"]}))
```

- [ ] **Step 2: Run these tests and verify they fail because the typed compiler does not exist.**
- [ ] **Step 3: Implement deterministic type inference, comma/colon parsing, categorical-combination, product, slope, and GxE/GxC validation.**
- [ ] **Step 4: Run the effect tests and then the complete focused interface module.**

Run:

```bash
PYTHONPATH=python python -m unittest -v test.test_reml_interface
```

Expected: parser and compiler tests PASS; model integration tests added below remain red until their tasks are implemented.

### Task 4: Wire compiled effects through BLUP and BLUE and fix `-f` regression

**Files:**
- Modify: `python/janusx/script/reml.py:_encode_fixed_design`, `_encode_random_design`, `_fit_stage1_blue`, main trait loop
- Modify: `test/test_reml_interface.py`

**Interfaces:**
- `_fit_stage1_blue` accepts the compiled fixed design in addition to environment/line terms, or receives the same `fixed_terms` list and compiles it internally through the shared compiler.
- `_build_stage1_blue_terms` must no longer drop fixed terms that vary within line; it returns the full fixed list plus random list or is removed in favor of the compiled model object.
- Main model metadata contains every requested effect and its matrix names.

- [ ] **Step 1: Write a failing propagation test using a synthetic two-line, two-location table.**

```python
def test_varying_fixed_covariate_reaches_blue_stage(self):
    sub = pd.DataFrame({
        "line": ["L1", "L1", "L2", "L2"],
        "loc": ["A", "B", "A", "B"],
        "yield": [1.0, 4.0, 2.0, 5.0],
    })
    fixed = reml._parse_effect_specs(["loc"], "fixed", list(sub.columns), sub)
    compiled = reml._compile_model_terms(sub, line_col="line", fixed_specs=fixed,
                                         random_specs=[], gxe_specs=[], gxc_specs=[])
    self.assertIn("loc", compiled.fixed_labels)
    blue = reml._fit_stage1_blue(
        y_obs=sub["yield"].to_numpy(), sub=sub, line_col="line", trait="yield",
        compiled=compiled, maxiter=20, logger=logging.getLogger("test"),
    )
    self.assertNotAlmostEqual(float(blue.values[0]), float(blue.values[1]), places=3)
```

- [ ] **Step 2: Run the test and verify it fails with the current stage-two signature/drop warning.**
- [ ] **Step 3: Replace the old env/fixed/random split with one compiled model object and pass the fixed design into both BLUP and BLUE paths.**
- [ ] **Step 4: Add regression assertions that every `-rc`, `-gxe`, and `-gxc` label appears in the compiled metadata and log configuration.**
- [ ] **Step 5: Run focused propagation tests and a `py_compile` check.**

### Task 5: Add failing variance-log and no-GRM output tests

**Files:**
- Modify: `test/test_reml_interface.py`
- Modify: `python/janusx/script/reml.py` only after red tests

**Interfaces:**
- Add a `main(argv: list[str] | None = None)` entry point (or a private equivalent used by `main`) so tests invoke REML with explicit arguments without mutating process-global `sys.argv`.
- Per-trait log emits raw `Line`, each random effect, and `Residual` variances.
- No-GRM output omits GBLUP and narrow-h2 computation while retaining BLUE/BLUP.

- [ ] **Step 1: Write a temporary synthetic phenotype test that invokes the no-GRM route with `-p`, `-n`, `-c`, and `-rc`.**
- [ ] **Step 2: Assert the red test currently fails because raw per-term variance labels and the conditional no-GRM output behavior are not implemented.**
- [ ] **Step 3: Implement raw variance extraction and logging from the compiled random-term order, preserving existing BLUE/BLUP filenames.**
- [ ] **Step 4: Implement no-GRM conditional output and assert no `.gblup.txt` is created.**
- [ ] **Step 5: Run the test with temporary directories and clean only those test-owned directories.**

### Task 6: Add failing dense narrow-h2 uncertainty-handoff tests

**Files:**
- Modify: `test/test_reml_interface.py`
- Modify: `python/janusx/script/reml.py:_line_level_noise_diag`, joint narrow call site

**Interfaces:**
- `_line_level_noise_diag` returns a finite non-negative vector aligned to the kept BLUE line IDs.
- Dense path calls `_fit_joint_line_kernel_exact` (or the approved equivalent) with `noise_diag`, fixed-line design, and returns `va`, `vline`, `noise_mean`, additive BLUP, and corrected h2.

- [ ] **Step 1: Write a failing unit test that monkeypatches the dense backend call boundary and asserts the nonzero `noise_diag` is received.**
- [ ] **Step 2: Write a failing numerical test that calls `_fit_joint_line_kernel_exact` with identical data and two different uncertainty diagonals, then asserts the reported phenotype-scale h2 decreases as uncertainty increases.**
- [ ] **Step 3: Replace the current direct `_splmm_exact_null_fit_from_grm` call with the joint corrected fit.**
- [ ] **Step 4: Use the joint additive BLUP for GBLUP output and log the corrected h2 plus separately labeled raw latent ratio.**
- [ ] **Step 5: Run dense correction tests and compare the zero-noise result with the current error-free objective within tolerance.**

### Task 7: Add sparse correction adapter and parser/dev-help tests

**Files:**
- Modify: `python/janusx/script/reml.py` sparse narrow call path
- Modify: `test/test_reml_interface.py`
- Modify: `python/janusx/assoc/workflow.py`, `python/janusx/gs/workflow.py`, or other direct consumers only if search identifies old `--n` declarations/forwarding

**Interfaces:**
- Sparse narrow fitting receives the same line-level `noise_diag` semantics as dense fitting, either through an existing backend argument or a small wrapper that adds the diagonal to the sparse objective.
- `--spk-mode` remains hidden from normal help and visible under `-dev`.

- [ ] **Step 1: Write a failing sparse-call boundary test asserting `noise_diag` is passed or incorporated.**
- [ ] **Step 2: Write a failing help test for normal/dev visibility of `--spk-mode`.**
- [ ] **Step 3: Implement the sparse diagonal correction using the smallest compatible backend change; if the Rust/PyO3 function needs a new optional argument, add a Python compatibility wrapper and focused Rust/Python test.**
- [ ] **Step 4: Update only the shared/consumer `-n/--ncol` declarations and forwarding found by search; do not alter unrelated genotype `-file` options.**
- [ ] **Step 5: Run parser, dense, sparse-boundary, and shared-consumer tests.**

### Task 8: End-to-end integration and regression verification

**Files:**
- Create or modify: `test/test_reml_integration.py`
- Modify: `python/janusx/script/reml.py` only for test-discovered defects

**Interfaces:**
- Synthetic multi-trait, multi-environment data exercises fixed effects, random effects, GxE, GxC, no-GRM, dense-GRM, and sparse-GRM routes.
- The test inspects generated files and log text, not private optimizer internals alone.

- [ ] **Step 1: Add a deterministic synthetic dataset with repeated lines across locations, a categorical block, a categorical environment, and a continuous temperature.**
- [ ] **Step 2: Run no-GRM end to end and assert BLUE/BLUP plus variance labels in log.**
- [ ] **Step 3: Run dense-GRM end to end and assert corrected narrow-h2/GBLUP plus finite line uncertainty diagnostics.**
- [ ] **Step 4: Run sparse-GRM end to end when a repository-compatible sparse fixture is available; otherwise run the boundary adapter test and record the unavailable fixture.**
- [ ] **Step 5: Run all focused tests, `python -m py_compile`, and the JanusX smoke check.**

Commands:

```bash
PYTHONPATH=python python -m unittest -v test.test_reml_interface test.test_reml_integration
PYTHONPATH=python python -m py_compile python/janusx/script/reml.py python/janusx/script/_common/cli_args.py python/janusx/assoc/workflow.py python/janusx/gs/workflow.py
```

### Task 9: Final review and handoff

- [ ] Run `git diff --check` and inspect the complete diff for accidental edits to `src/garfield/bs.rs` or generated benchmark outputs.
- [ ] Run the focused tests again from a clean temporary output directory.
- [ ] Run `mamba run -n jxfu python -c 'import janusx.janusx as m; print(m.__file__)'` and the relevant `jx reml -h` smoke checks.
- [ ] Update the approved design doc only for verified implementation deviations.
- [ ] Summarize exact test commands, passing/failing status, and any pre-existing unrelated failures.
- [ ] Commit implementation changes separately from the design-doc commit.
