# Research decisions

## 2026-08-18 — Original ANZA interaction forensic confirm

- Previous Structural Reachability Phase A remains frozen as
  `STOP_ARCHITECTURAL_ANZA_NO_CAUSAL_GEOMETRY_GAIN`.
- New question: whether the literal original ANZA interaction carries an
  independent structural signal on segmentation-unseen CRACKS sections.
- Pre-specified Phase-0 gate: the frozen checkpoint must implement the stated
  independent-fuzzy directed interaction literally before read-only
  instrumentation or confirm scoring is allowed.
- Result: **FAIL**, `STOP_OPERATOR_DEFINITION_MISMATCH`.
- Root cause: legacy code uses categorical softmax memberships and symmetric
  center/neighbor pair geometry; the packet equation assumes independent fuzzy
  degrees and directed center geometry. The missing Gaussian factor 1/2 is also
  not literal, although scale-equivalent.
- Independent data blocker: the only segmentation-unseen images (49, 73, 385)
  contain no crowd annotations, giving
  `STOP_NO_INDEPENDENT_CONFIRM_SPLIT`.
- Allowed next: owner may formulate a materially new study with newly reserved
  annotated sections or an independently trained split.
- Blocked: instrumentation under this packet, S0-S4 confirm, learned affinity,
  new training, reachability, and expert access.

## 2026-08-18 — Open ANZA-2 as a separate operator family

- Historical LegacyANZA and every prior negative result remain frozen.
- The owner authorized a new from-scratch operator rather than changing the
  legacy checkpoint or retroactively reinterpreting its results.
- Phase-0 question: whether the verified CRACKS release supports a new grouped
  section-disjoint OOF protocol without expert access.
- Phase-0 result: **PASS with limitation**. The official-checksum-matched archive
  has 396 images; nominal IDs 9, 185, 249, and 336 are absent from that archive,
  while 393 image sections have non-expert annotations. Physical coordinates
  are not present, so the frozen five-fold protocol is grouped by numeric IDs
  and is not described as proven spatial OOF.
- Phase-1 gate: all Golden mathematical tests and deterministic
  Straight/Parallel/Crossing/Curved fixtures must pass before dataset training.
- Allowed next after Phase-1 PASS: controlled synthetic mechanism only.
- Blocked until later gates: CRACKS training, continuation selector, expert
  evaluation, and any positive scientific claim.

## 2026-08-18 — Phase-2A saturation and independent Phase-2B replacement

- Phase-2A froze path TPR at FPR≤0.05 as its sole primary effect. Both
  Legacy-normalized and ANZA-2 oracle relations reached TPR 1.0 with zero false
  bridges, so the required positive TPR delta was impossible and Phase-2A is
  preserved as `STOP_ANZA2_GEOMETRY_NOT_STRUCTURALLY_SELECTIVE`.
- The same unopened result also showed a pre-specified branch-preservation
  endpoint: ANZA-2 retained all junction branches while the legacy control lost
  one X-crossing branch. The Phase-2A primary was not changed post hoc.
- Before a new seed stream was opened, Phase-2B froze branch-recall delta as the
  primary metric with path TPR and false-bridge non-inferiority constraints.
- Phase-2B result: `PHASE2_GEOMETRY_SELECTIVITY_PASS`. ANZA-2 branch recall was
  1.00 versus 0.90 for the legacy-normalized control; paired delta +0.10 with
  95% bootstrap CI [0.10, 0.10]. Path TPR was 1.00 versus 0.9922 and both false
  bridge rates were zero.
- Claim boundary: controlled oracle-field mechanism evidence only. No learned
  image field, CRACKS result, or expert result exists yet.
- Allowed next: Phase-3 learned synthetic affinity/reachability under a new
  frozen protocol.

## 2026-08-18 — Phase-3 learned field and one bounded causal repair

- The frozen Phase-3 development run compared a generic edge model with a
  separately trained generic-plus-ANZA model. It did not meet the `+0.08` TPR
  gate and exposed two confounds: the backbones were no longer identical, and
  plain orientation-set coverage did not require the aligned mode to be active.
- One bounded repair was frozen before rerun. It added
  membership-weighted axial coverage and compared ANZA affinity OFF/ON inside
  the same checkpoint, with the encoder and generic edge head held fixed.
- An evaluator audit corrected inclusive threshold ties so every reported FPR
  is actually `<=0.05`; checkpoints were not retrained for that audit.
- Re-audited result: three-seed TPR delta `+0.00027406`, 95% sample-bootstrap
  CI `[+0.00011431, +0.00043699]`, at FPR `0.049973` for both conditions.
  Although the sign is positive, it is far below the frozen practical gate
  `+0.08`; status is `STOP_PHASE3B_LEARNED_AFFINITY_NO_GAIN`.
- Phase-2B oracle evidence remains valid, but learned-image, CRACKS, and expert
  claims are not supported. Confirm and Phase 4 remain closed.

## 2026-08-18 — Phase-3C-A component forensics selects RC1

- No training was performed. Frozen Phase-2B/3B artifacts and previously opened
  CrossingTraceBench-v4 development data were used for the predeclared F0-F9
  replacement, field-fidelity, and fusion audits.
- Phase 2B reproduced exactly. The full oracle relation retained branch/path
  recall 1.0 and false-bridge FPR 0.0.
- Full learned affinity had TPR 0.0653 at FPR <= 0.05 and parallel false bridge
  1.0. Learned geometry with oracle membership improved TPR to 0.5430 and
  restored parallel false bridge 0.0; learned membership with oracle geometry
  yielded TPR 0.0111 and false bridge 1.0.
- Field fidelity localized the failure: active-mode recall 0.0030 and all-zero
  membership on 0.9948 of target pixels, while orientation q90 error remained
  0.1068 radians and mean along/perpendicular geometry ratio remained 6.156.
- Exactly one predeclared cause is selected:
  `RC1 ROOT_CAUSE_MEMBERSHIP_LEARNING`.
- Allowed next: one bounded membership-learning repair on development data,
  frozen before confirm. Confirm, CRACKS, and expert data remain closed.
- Claim boundary: this is forensic localization, not a repaired learned-model
  result or a real-data improvement claim.

## 2026-08-18 — Bounded RC1 membership repair stops after M-A/M-B

- The repair used the frozen Phase-3B seed-41 checkpoint and changed only
  `field.membership_head`. M-A and M-B ran exactly five epochs with
  `lambda_bg=0.25/0.50`; every other parameter remained bitwise unchanged.
- Train-monitor activation improved, but development membership recall was only
  0.7562/0.7433. Both variants activated fewer than two modes on essentially
  every crossing pixel, produced parallel false bridge 1.0, and achieved raw
  TPR only 0.0853/0.0872 at FPR <= 0.05 versus the frozen gate 0.45.
- The predeclared selection therefore chose no configuration and froze
  `STOP_RC1_MEMBERSHIP_REPAIR_FAILED`.
- The exact frozen train[0:256] case inventory contains 128 positive gaps and
  128 negative gaps, with no junction/context strata. This explains an
  important observability limitation but was not used to change the protocol.
- Three-seed repair, beta refit, confirm, CRACKS, and expert access were not
  performed. No third weight or additional epoch is authorized under RC1.

## 2026-08-18 — Phase-3D mode-state oracle closes the final branch

- The complete frozen CrossingTraceBench-v4 manifest was audited: 512 samples
  per split, zero seed overlap, and all eight mandatory strata present in
  train, validation, and confirm. Confirm access was metadata-only.
- Local membership targets were separated from privileged latent continuation:
  positive-gap latent directions are not allowed as local learned supervision.
- Deterministic tests verified the exact `(pixel, mode)` edge formula, no free
  intra-pixel mode switch, axial permutation invariance, curved reachability,
  and agreement with an exhaustive tiny-graph reference.
- Thresholds were frozen on the complete train stream and the oracle gate was
  evaluated on validation. Mode-state positive recall was 0.6806 versus 0.6910
  for scalar; X wrong-turn FPR was 0.1125 versus 0.0875, a relative reduction
  of -0.2857 versus the required +0.50.
- The gate therefore froze `FINAL_STOP_MODE_STATE_ORACLE_NO_VALUE`. No training,
  confirm evaluation, CRACKS access, or expert access occurred. Phase 3D-C is
  not authorized and no further ANZA version is opened by this result.

## 2026-08-18 — Separate ANZA-S cocycle-shadowing oracle

- ANZA-S was evaluated as a new hypothesis without altering the frozen ANZA-2
  result. Twenty-seven targeted math/mechanism/validator tests passed.
- The frozen O0-O4 oracle used train-only operating-point calibration and one
  validation gate. Confirm, test, CRACKS, expert, and all training stayed closed.
- O4 passed the packet's formal comparison against O0 and O2: macro positive
  recall 0.50, X wrong-turn FPR 0.0, parallel and negative-gap false bridge
  0.0, and derived curved-gap recall 1.0.
- This pass is not causal evidence for the hyperbolic cocycle. O2 and O3 are
  exactly identical, while O4 X-correct and StraightGap recall are both 0.0 at
  the safe threshold. The gain belongs to the combined shadowing readout under
  this protocol, and a generic tangent-plus-shadowing control was not included.
- Phase B was not run. Any field learning requires a new frozen protocol and
  must preserve these negative/ambiguous mechanism findings.
