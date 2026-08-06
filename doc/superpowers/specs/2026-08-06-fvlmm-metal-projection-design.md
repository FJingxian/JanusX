# FvLMM Metal Projection Design

## Goal

Add an opt-in Apple Metal implementation of the dominant FvLMM projection
kernel, `G (m × n) × Uᵀ (n × n)`, while leaving the CPU association stage,
BED filtering, TSV ordering, and the default Accelerate path unchanged.

## Context

The current macOS FvLMM BED path decodes a SNP block, performs one large
Accelerate SGEMM for `G × Uᵀ`, runs the existing fixed-lambda association, and
writes TSV output. On the target machine, projection is about half of the scan
stage and Accelerate exposes only a process-wide single/multi-thread switch.
The repository already has an optional `metal-gpu` dependency and Metal
infrastructure for GARFIELD, but that backend is not connected to FvLMM.

## Scope and interface

- `JX_FVLMM_ROTATE_KERNEL=metal` selects Metal projection explicitly.
- The default, empty value, and `blas` continue to use the current Accelerate
  or Rust/BLAS dispatch.
- Selecting `metal` in a build without `metal-gpu`, on a non-macOS platform, or
  when no Metal device is available returns a descriptive runtime error. It
  must not silently change the requested backend to CPU.
- The Metal path is compiled only under `all(target_os = "macos", feature =
  "metal-gpu")`; non-Metal builds retain a small error-only stub.
- No new public Python API is required. The existing FvLMM CLI selects the
  backend through the environment variable and continues to call the same
  `fvlmm_assoc_bed_to_tsv_f32` entry point.

## Architecture

Create a focused `src/stats/fvlmm_metal.rs` module with a reusable
`FvlmmMetalProjector`:

1. Construct one Metal device, command queue, compute pipeline, and reusable
   `Uᵀ`, input, and output buffers per FvLMM scan.
2. Upload the fixed `Uᵀ` once. For each SNP block, copy the contiguous f32
   genotype block into a shared input buffer, dispatch a 2-D tiled f32 matrix
   multiplication kernel, wait for completion, and copy the output into the
   existing rotation buffer.
3. Keep association and TSV code unchanged. The caller selects the projector
   once at scan setup and invokes it synchronously for each block, so output
   order and buffer ownership remain deterministic.
4. Use a conservative 16×16 threadgroup tile initially. The kernel guards
   partial tiles and uses `threadgroup_barrier`; requested thread count maps to
   Metal threadgroup size only after clamping to the pipeline/device limit.

The projector must validate `n`, block dimensions, byte lengths, and finite
buffer sizes before dispatch. `Drop`/autorelease handling follows the existing
GARFIELD Metal code. Shared buffers may use explicit copies first; no unsafe
zero-copy host pointer lifetime is introduced in this first implementation.

## Diagnostics and fallback

When `JX_FVLMM_BED_STAGE_TIMING=1` is enabled, the existing timing line reports
`backend=metal` and the effective projection threadgroup size. The verbose GWAS
notes report the selected rotate kernel. The explicit Metal mode fails early
with the device/build error so benchmark results cannot be mistaken for a CPU
fallback.

## Correctness and performance criteria

- Metal and Accelerate projection outputs are compared on the same f32 inputs.
  The acceptance threshold is max absolute error ≤ `1e-4` and max relative
  error ≤ `1e-4` for finite values.
- A full CLI scan must preserve SNP count, metadata, output order, and all
  displayed association values within the existing f32 formatting tolerance.
- Benchmarks record projection time, total scan time, peak RSS, device name,
  threadgroup width, and backend. Metal is not made the default unless it is
  consistently faster on the real workload and passes the tolerance checks.
- The first benchmark uses the existing macOS example and then the local
  ~200k-SNP maize input if its phenotype/GRM alignment is available.

## Risks and non-goals

- A per-block device copy or command-buffer synchronization may erase GPU
  gains; this is measured rather than hidden behind a default switch.
- This phase does not move fixed-lambda association, variance estimation, or
  TSV formatting to Metal.
- It does not alter OpenBLAS/Linux behavior or the existing GARFIELD Metal
  implementation.
