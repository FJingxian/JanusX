# GARFIELD XOR Opt-in Search Design

## Goal

Make XOR an explicit, opt-in GARFIELD search space. The default search must
expand only additive/AND rules; XOR expansion occurs only when the user passes
`--xor-search`.

## Scope

1. Add a visible `--xor-search` CLI flag with a default of `False`.
2. Thread the effective policy explicitly from the Python CLI through the
   PyO3 function into Rust beam-search parameters.
3. Keep `-nf-xor/--nf-xor` independent: it continues to control XOR substate
   filtering, not whether XOR rules are generated.
4. Preserve `JX_GARFIELD_DISABLE_XOR_SEARCH=1` as a compatibility force-off
   override. The effective policy is `args.xor_search and not env_disable`.
5. Preserve all existing score, gain, output-ranking, and penalty semantics.

## Design

The Python wrapper owns the user-facing policy. It computes a boolean
`xor_search` once and passes it as a separate argument to
`garfield_logic_search_bed`. The run manifest records the effective value so a
result can be reproduced from its metadata.

The Rust `BeamSearchParams` receives `xor_search_enabled: bool`. Every beam
expansion site uses that parameter when selecting binary operations. The
operation policy remains:

- a rule already containing XOR expands only with AND, preventing repeated
  XOR branching;
- a negated singleton expands only with AND;
- other rules expand with AND only when XOR is disabled, or with AND and XOR
  when it is enabled.

The current Rust `OnceLock` environment lookup is removed from the operation
selector so behavior is explicit, deterministic, and testable. The legacy
environment variable is interpreted at the Python boundary as a force-off
override.

## Compatibility

Existing commands that do not mention `--xor-search` become AND-only searches
by default. Commands that need XOR add `--xor-search`. Existing `-nf-xor`
commands retain their current substate-filter behavior and do not implicitly
enable XOR search.

## Verification

- Rust unit tests cover disabled and enabled operation lists and the default
  `BeamSearchParams` value.
- Python tests cover CLI parsing, compatibility override, and manifest/call
  propagation.
- Run formatting, focused Rust tests, Python syntax/tests, rebuild the jxfu
  extension, and execute a small GARFIELD smoke invocation with and without
  `--xor-search`.
