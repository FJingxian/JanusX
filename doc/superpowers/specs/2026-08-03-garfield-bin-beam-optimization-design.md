# GARFIELD BIN Beam Optimization Design

## Goal

Optimize the ordinary PLINK BED (`BIN`) GARFIELD beam-search path used by `jx garfield -w`, while preserving rule generation, score calculation, pruning, ordering, and downstream FvLMM inputs exactly.

## Scope

- Modify only the ordinary `BeamState` expansion path in `src/garfield/bs.rs`.
- Leave the fuzzy/dual path unchanged.
- Preserve the existing serial fallback and all scoring/pruning predicates.
- Defer `combined_train` bitset materialization until after worker-local and global candidate selection.
- Add a serial-versus-parallel ordinary expansion regression test.

## Data Flow

1. Compute the parent-candidate intersection once, as the current path does.
2. Evaluate the child and all existing score/pruning predicates.
3. Store a lightweight state containing the rule, scores, and singleton metadata, but no materialized combined bitset.
4. Deduplicate and truncate lightweight states using the existing comparator.
5. Materialize `combined_train` only for the retained states.
6. Run the existing final beam filtering unchanged.

## Correctness

The comparator and rule-key deduplication remain unchanged. Materialization is a representation step only; it must not participate in candidate acceptance or ordering. The test compares the optimized parallel expansion against the existing serial expansion using the complete rule and score state.

## Verification

- Run the targeted Rust test in release mode with the repository Python 3.13 environment.
- Run a real `jx garfield -w` CUBIC Chr1/`--bimrange` benchmark before and after the change.
- Compare rules TSV and FvLMM TSV outputs byte-for-byte where the same output prefix is used.
- Report wall-clock, CPU time, and peak RSS if available.
