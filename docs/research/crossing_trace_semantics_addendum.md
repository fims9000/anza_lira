# CrossingTraceBench target semantics addendum

This addendum freezes the clarification made before structural evaluation. It
separates observed segmentation from latent structural completion.

## Canonical targets

- `visible_fault_mask`: only fault pixels rendered in the input image.
- `latent_fault_mask`: complete generated fault geometry, including a hidden
  positive gap.
- `gap_mask = latent_fault_mask & ~visible_fault_mask`.
- `instance_masks[N,H,W]`: overlapping Boolean latent instances; a crossing
  pixel may belong to more than one instance.

Visible Dice, IoU, precision, recall, and clDice use
`visible_fault_mask`. Latent clDice, positive-gap recovery, false-bridge
control, branch continuation, false merge/split, and identity switch are
reported separately as structural-completion metrics.

Positive gaps join fragments of one generator-defined latent instance.
Matched negative gaps contain nearby fragments with no common latent instance.
High gap recovery without low false-bridge rate is not accepted as structural
completion.

Ground-truth branch continuation comes only from generator lineage. It is never
reconstructed from minimum turning angle. X, T, and Y junctions retain distinct
topology contracts, and the `nontrivial_pairing` stratum includes cases where
the smallest local turning angle is not the true continuation.

The primary real/synthetic segmentation objective remains the visible target.
Route and gap supervision are separately named ablations; latent gap pixels are
not silently relabeled as ordinary visible foreground.
