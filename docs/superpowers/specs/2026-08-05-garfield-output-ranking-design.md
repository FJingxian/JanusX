# GARFIELD output ranking correction

## Objective

Restore the intended distinction between GARFIELD search pruning and final
reporting:

- Search expansion prunes interaction children using
  `gain - search_penalty > eps`.
- Final reporting ranks every surviving singleton or combination using
  `raw_score - output_penalty`.

## Current defect

`evaluate_logic_unit_prepared_continuous` already computes and sorts
`ranked_hits` by the final output score. `select_reportable_ranked_hits` then
re-sorts non-raw candidates by `test.raw_score`, discarding the output penalty
and potentially changing the selected rule.

## Design

`select_reportable_ranked_hits` will preserve the existing `ranked_hits`
ordering. The `top_rules_per_unit` cutoff and score-tie extension will use the
stored output score in each `(candidate_index, output_score)` pair. No beam
ranking, interaction-gain calculation, null calibration, or search pruning
behavior will change.

## Verification

Add focused Rust regressions proving that:

1. A singleton with a lower raw score but a higher output score is selected
   ahead of a combination.
2. Top-k tie handling uses output scores.
3. Existing search tests continue to prove that interaction pruning uses the
   gain-minus-search-penalty scale and does not prune layer-one singleton
   seeds.

Run formatting checks and focused GARFIELD tests. Existing unrelated worktree
changes are outside this correction and must remain untouched.
