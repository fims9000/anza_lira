# Provenance gaps

Snapshot: 2026-09-20.

This file records what is **not yet reproducible from the branch**, so exploratory chat results cannot silently become paper claims.

## Preserved exactly

The branch contains exact machine artifacts supplied from the prior research session for:

- sequence CNN / Transformer summary;
- per-patient sequence runs;
- perturbation risk-coverage table;
- per-scene perturbation uncertainty table (gzip stored as base64).

The branch also contains raw-data hashes / filenames for the local CCTA and ImageCAS-X files used in the current continuation.

## Missing source code from the previous chat session

The original scripts that produced the following prior-session artifacts were not attached when the old chat ended:

- the 15-perturbation consistency runner;
- the exact sequence-token CNN / Transformer training runner;
- the earlier radial 2.5-D / candidate-aligned 3-D tube pilot;
- the exploratory joint / variable-degree Graph-LIRA stress runners.

Therefore:

1. the preserved CSV numbers are evidence of those runs;
2. the branch does **not** yet claim one-command reproduction of those historical exploratory runs;
3. before any of those numbers are used as final paper results, the corresponding experiment must be reconstructed as a committed canonical script and re-run from a frozen config;
4. if the re-run differs, the new canonical run takes precedence and the discrepancy must be documented.

## Current canonical work

From this checkpoint onward, new experiments must be committed with:

- script;
- config / frozen constants;
- input hashes;
- split / case identifiers;
- output metrics with denominators;
- code commit SHA.

The first new canonical task is the scan-953 coordinate/alignment audit.
