# FvLMM Metal Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and benchmark an opt-in Apple Metal implementation of the FvLMM `G × Uᵀ` projection while preserving the default Accelerate route.

**Architecture:** A new `fvlmm_metal.rs` owns a reusable Metal projector and tiled f32 compute shader. `fvlmm.rs` selects it only when `JX_FVLMM_ROTATE_KERNEL=metal`; the existing association, pipeline, and TSV stages remain unchanged. Non-Metal builds expose an error-only stub.

**Tech Stack:** Rust/PyO3, `metal` 0.32, `objc`, Metal Shading Language, Rayon/Accelerate, macOS `jxfu` environment.

## Global Constraints

- Default/empty/`blas` kernel selection must remain the current CPU/BLAS path.
- Metal selection is explicit and must fail clearly when the feature/device is unavailable.
- Do not modify GARFIELD Metal code or unrelated working-tree changes.
- Keep output order and association calculations unchanged; only the rotation implementation is replaced.
- Do not export `DYLD_LIBRARY_PATH` when running Python or `jx` on macOS.
- Local test scripts remain untracked and are not committed.

---

### Task 1: Add failing unit tests and backend-selection contract

**Files:**
- Modify: `src/stats/fvlmm.rs` test module
- Create: `src/stats/fvlmm_metal.rs`
- Test: local-only `test/test_fvlmm_metal_projection.py` if a Python-visible contract is needed

**Interfaces:**
- `fvlmm_rotate_kernel_prefers_metal() -> bool` recognizes `metal` only.
- `FvlmmMetalProjector::new(n, max_rows, requested_tg) -> Result<Self, String>` exists under the Metal feature and the stub returns a descriptive error otherwise.

- [ ] Add Rust tests for empty/default/`blas`/`metal` kernel parsing without touching process-global backend state.
- [ ] Add Rust tests for tile-grid dimensions and rejected zero/overflow shapes.
- [ ] Run `cargo test fvlmm::tests --lib -q` with the Rust Python-library environment and confirm the new tests fail because the helpers are absent.

### Task 2: Implement the reusable Metal projector

**Files:**
- Create: `src/stats/fvlmm_metal.rs`
- Modify: `src/lib.rs` module declarations only if the module needs crate-level visibility
- Modify: `Cargo.toml` only if an existing optional feature cannot compile the module

**Interfaces:**
- `pub(crate) struct FvlmmMetalProjector` (Metal build) owns device, queue, pipeline, and buffers.
- `pub(crate) fn project(&mut self, snp_block: &[f32], rows: usize, n: usize, u_t: &[f32], out: &mut [f32]) -> Result<(), String>` performs synchronous f32 `G × Uᵀ`.
- Non-Metal stub has the same constructor/project signatures and returns `metal-gpu feature unavailable`.

- [ ] Add the Metal shader source with 16×16 tiled multiplication, bounds checks, and f32 output.
- [ ] Compile the shader at runtime using the existing `metal` crate patterns and validate the pipeline/device limits.
- [ ] Allocate/reuse shared buffers sized for the scan's maximum block; upload `Uᵀ` once and copy each input/output block explicitly.
- [ ] Submit and wait on a command buffer before returning so the current CPU association stage sees complete output.
- [ ] Return contextual errors for device creation, shader compilation, shape mismatch, buffer allocation, and command completion.
- [ ] Run `cargo fmt --all -- --check` and the focused Rust tests on the default feature set.

### Task 3: Wire Metal into the FvLMM rotation path

**Files:**
- Modify: `src/stats/fvlmm.rs`

**Interfaces:**
- `rotate_snp_block_with_ut_blas` remains the default CPU/BLAS helper.
- A Metal-aware scan closure owns one projector and calls it for every block.

- [ ] Add `fvlmm_rotate_kernel_prefers_metal()` next to the existing tiled selector.
- [ ] Construct the projector once after `n` and `block_rows` are known in the BED unified path; do not construct a device/pipeline per SNP block.
- [ ] Route only the projection stage through Metal and leave `assoc_fixed_lambda_rot_block_blas_f32` untouched.
- [ ] Disable the CPU double-buffer rotation pipeline for Metal until asynchronous command-buffer overlap is explicitly implemented; preserve decode/association correctness.
- [ ] Include backend and actual threadgroup width in the stage timing line.
- [ ] Ensure explicit `metal` errors before creating output files if the feature/device is unavailable.

### Task 4: Build and run projection-level numerical tests

**Files:**
- Test: local-only `test/test_fvlmm_metal_projection.py`
- Modify: no production files unless a test exposes a defect

- [ ] Rebuild the extension with `maturin develop --release --locked --features "python-extension metal-gpu"` in `jxfu`.
- [ ] Run a synthetic projection comparison for `m=512,n=1410` and `m=8192,n=1410`; assert max absolute and relative error ≤ `1e-4`.
- [ ] Run `JX_FVLMM_ROTATE_KERNEL=metal` on a small real BED block and verify rows, metadata, and finite output against `blas`.
- [ ] Confirm the normal extension, built without `metal-gpu`, reports a clear explicit-backend error rather than silently using Accelerate.

### Task 5: Benchmark Metal versus Accelerate

**Files:**
- Test/output only: `/tmp/jx_fvlmm_metal_*`

- [ ] Run the same FvLMM example with `JX_FVLMM_BED_STAGE_TIMING=1` and `JX_FVLMM_ROTATE_KERNEL=blas` at `-t 1,2,4,8`.
- [ ] Run the same example with `JX_FVLMM_ROTATE_KERNEL=metal` and record device, projection, total wall time, RSS, and threadgroup width.
- [ ] If the local ~200k-SNP BED/phenotype/GRM alignment is confirmed, repeat the comparison there; otherwise report why it was not run.
- [ ] Keep Metal opt-in unless it is consistently faster and numerically within tolerance.

### Task 6: Final verification and handoff

- [ ] Run `cargo fmt --all -- --check`.
- [ ] Run `cargo test fvlmm::tests --lib -q` with `DYLD_LIBRARY_PATH` only for Rust tests.
- [ ] Run `mamba run -n jxfu python -m py_compile` on edited Python files (if any).
- [ ] Inspect `git diff --check` and confirm unrelated GARFIELD/docs/local-test changes are not staged by this work.
- [ ] Summarize whether Metal improves projection and total FvLMM time; do not claim completion without fresh benchmark output.
