# S70 — frozen retrospective audit of S64 pixel-only synthetic results

Status: **evaluation repair only; code and test seed set frozen before this replay**. Source S64 predictor and fitted S56 geometry-only model remain unchanged. No threshold, feature, extraction or split decisions are modified. Inference is blind to artificial identities until the post-hoc audit.

## Motivation and source locks

S69 proved that array order of independently reversed artificial fragment coordinates can cause the old S64 endpoint evaluator to count a geometrically correct synthetic join as a wrong-end join. This protocol checks **the original S64 benchmark seeds** (005→006: 20266403, 20266404; 006→005: 20266413, 20266414; 24 synthetic scenes each). Do not use results for method selection: these scenes have already been seen in the old S64 publication.

Frozen original S64 benchmark SHA-256 `ad31fef03bc40fc70606830527300873ae013967e6ac19836a115e1b95ed55c7` and original S64 evaluator SHA-256 `c148e55dc9a94f2841e34d2f5e1f3f6b8c41a1ebd345152d2ca21ee7c3ff824c`; new S69 evaluator SHA-256 `3e4ad80c0fc544b7b231b6a7b7dcdc3501484830a98d6d19c133dc0eb1c5b495`. Original frozen public S64 aggregate SHA-256 `0cf514960cc47651a0b0b5579d0cd728d108f8d639eaf0ee6dc89565f33c3301`.

## Re-evaluation protocol

Use original S64 `run` unchanged except replace the **evaluation-only** imported callable `audit_synthetic_scene` temporarily with a wrapper that performs both old and corrected scoring *on the same already-finished pixel predictions*. Restore the original callable after run. Compare every old aggregate across all seeds and both original S64 arms, not just headline metrics; require numeric equality for the original source fields, CI bootstrap and model-selection independent metrics. Verify the corrected evaluator has identical counts for candidates, source associations, predicted pairs, chosen edges and solver failures. Produce paired scene bootstrap differences (1200 resamples), grouped by full artificial scene rather than pixels. Freeze all other functions, parameters and source hashes.

Evaluation only: 145 genuine film-070 visualMetro measurement frames with frozen 79/30/36 frame split; H50Z source photos only as 39 private background tiles from one physical run, **not** matching visualMetro geometry or confirmed real physical GT. Never upload photo pixels, raw 7z, private point coordinates, individual scene predictions or frame identities. Report only aggregated correction, phase IDs, code/protocol digests and negative limitations. Keep physical automatic joins disabled.

## Acceptance

Replay must reproduce **ALL** unmodified original S64 aggregate metrics exactly or abort; any discrepancy is a scientific blocker. The new counts are a retrospective correction of a biased synthetic-only evaluator, not new heldout evidence and not a training/calibration criterion. Any change in prior conclusions must be reported as erratum, not hidden.
