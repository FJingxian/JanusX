# GARFIELD Exact LD-Clumping Optimization

## Goal

Reduce the cost of the inner LD comparisons in GARFIELD candidate-pool
selection without changing score ordering, LD threshold semantics, selected
rows, or downstream output.

The current operation is score-prioritized greedy LD clumping: candidates are
visited in descending single-site score order and a candidate is removed when
its binary LD with an already retained candidate satisfies `r2 >= 0.8`.

## Non-goals

- Do not change single-site scoring or prescreen size.
- Do not replace exact binary `r2` with an approximate similarity test.
- Do not change the default LD threshold or candidate ordering.
- Do not add cross-window caches in this iteration.

## Design

### Exact bounded pair test

Replace the unconditional full-vector `and_popcount(row_i, row_j)` call in the
LD inner loop with an exact word-wise bounded test. The test receives the
support counts and the precomputed integer bounds for the current support pair.

For each word, it maintains:

- the current intersection count `c`;
- the number of set bits seen in each row;
- an upper bound on remaining intersection, `min(remaining_i, remaining_j)`.

The test may return early only when the result is mathematically determined:

- `c >= high_min`: the high-intersection LD tail is already guaranteed;
- `c + remaining_max <= low_max`: the low-intersection LD tail is guaranteed;
- `c > low_max` and `c + remaining_max < high_min`: neither LD tail can be
  reached, so the pair cannot conflict.

If no bound decides the result, it returns the existing exact
`bounds.matches(c)` result after all words are processed. No floating-point
calculation is added to the hot loop.

### Integration boundary

Keep `prune_candidate_rows_by_ld_priority` and its support buckets unchanged.
Only replace the pair predicate inside the existing greedy loop. The retained
row order and `kept_by_support` layout remain unchanged.

## Correctness invariants

For every finite binary row pair and every support-pair bound:

1. The bounded predicate equals `bounds.matches(and_popcount(row_i, row_j))`.
2. A candidate is retained or removed exactly as before.
3. The output of the existing CUBIC profile is byte-identical, including
   rules, pseudo-GWAS BED/BIM, and FvLMM TSV.

## Testing

### Unit tests first

Add a Rust test that exhaustively checks all short binary vectors and several
sample sizes and LD thresholds against the existing full `and_popcount`
predicate. Add randomized longer-vector coverage including low-tail,
high-tail, and non-conflicting pairs.

The test must fail before the bounded predicate is implemented, then pass
after implementation.

### Regression and performance checks

- `cargo fmt -- --check`
- `cargo test --lib garfield`
- `cargo check --release --features python-extension`
- Run the CUBIC ATI whole-genome profile with 8 threads and stage profiling.
- Compare rules, FvLMM, BED, and BIM outputs with `cmp`.
- Report `timing_corr_stage1_pool_ld_s`, total wall time, and beam timing.

## Rollback

If any byte-level output differs or the exactness tests fail, retain the
profiling-only changes and revert only the bounded predicate integration.
