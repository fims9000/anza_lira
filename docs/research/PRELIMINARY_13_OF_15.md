# ANZA-LIRA v2: preliminary 13-of-15 writing bundle

Generated while the frozen Setting A matrix is still running. This bundle is
for drafting the methods, protocol, and experimental-status sections. It is
not a final results package.

## Current status

- Setting A protocol hash: `04e35c267f0031ce`.
- Completed training runs: 13 of 15.
- Main comparison: U-Net, deformable U-Net, ANZA v1, and ANZA-LIRA v2 are
  complete for seeds 41, 42, and 43.
- Completed ablation: ANZA-LIRA v2 without structural replay, seed 42.
- Still running/pending at package time: `v2_no_fuzzy_s42` and
  `v2_no_directional_s42`.
- Expert annotations were not used by any included training run.
- Full-section crowd threshold selection is not yet frozen, so expert
  evaluation remains locked.

The `heldout_crop_dice_at_0_5` values in the status files are training
diagnostics on the permitted crowd validation stream. They are not the frozen
full-section paper metrics and must not be quoted as final model performance.

## Frozen synthetic result

- Development protocol hash: `56fdaab7e2591c3a`.
- Evaluation protocol hash: `b4dc4cda3e245458`.
- Freeze SHA-256:
  `3ea5fe55828282f1a66746429ca157456c190f6bf65cf226473e90ea8dab5d4b`.
- Candidate C3 was frozen before the single test opening.
- The structural quality gate was `NOT_MET`.

This is an informative negative result. The current evidence does not support
a claim that ANZA-LIRA v2 improves branch identity, continuation, or false
merge behavior on CrossingTraceBench. It also does not support an Anosov-like
dynamics claim.

## Claims safe to draft now

- The mathematical definition and implementation of mode-resolved axial and
  half-mode directional transport.
- The visible-versus-latent structural-completion contract.
- Overlapping fault-instance masks at crossings.
- Generator-defined X/T/Y topology and nontrivial continuation identity.
- Positive-gap recovery paired with negative-gap false-bridge control.
- The frozen CRACKS split, annotation policy, expert lock, threshold-freeze
  rule, and cluster-bootstrap plan.
- The negative synthetic gate result, with its exact frozen evidence.

## Claims that must wait

- Final CRACKS Dice, IoU, clDice, skeleton, trace, orientation, fragmentation,
  human-comparison, or uncertainty associations.
- Any superiority statement over U-Net, deformable U-Net, ANZA v1, or humans.
- Any ablation conclusion from the two unfinished runs.
- Any final manuscript table, confidence interval, or efficiency trade-off.

Those claims require all 15 Setting A runs, frozen full-section thresholds,
expert evaluation, Settings B and C, source-section cluster bootstrap, final
figures/evidence generation, and the final validator PASS.

## Bundle layout

- `protocol/`: frozen protocol and data-semantics receipts.
- `synthetic/`: frozen validation/test evidence and diagnostics, without model
  checkpoints.
- `setting_a_completed/`: the 13 complete `status.json` histories only.
- `docs/`: method, semantics, and audit notes useful for manuscript drafting.
- `method_code/`: model, benchmark, training, evaluation, statistics, and
  evidence source files needed to inspect the implementation.
- `tests/`: the most relevant mathematical and provenance contract tests.
- `SHA256SUMS.txt`: hashes of every packaged file except the hash list itself.
