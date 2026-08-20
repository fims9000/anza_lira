# ANZA-LIRA Agent Contract

## Mission

Complete the current research task end-to-end. The active study is CRACKS;
GeoCrack infrastructure remains reusable but is not the current submission.
Stop only after the final validator reports:

```text
CRACKS STUDY STATUS: COMPLETE
```

or after a documented external blocker.

## Existing Project First

Before implementing anything:

1. inspect existing code and `.agents/AGENTS.md`;
2. discover skills with `python scripts/list_agent_skills.py`;
3. load no more than two phase-relevant skills;
4. inspect relevant tests, configs, and scripts;
5. reuse existing mechanisms and avoid duplicate implementations.

## Progressive Context

Never load the entire repository or a full training log into context. Prefer:

```text
metadata -> compact summary -> targeted excerpt -> full file only if required
```

Use `TASK_STATE.json` and `EVIDENCE.json` instead of rereading proven facts.

## Small Verified Changes

Before each feature, update `.agent-state/TASK_STATE.json` with:

- goal;
- expected files;
- existing implementation to reuse;
- acceptance test;
- rollback strategy.

Implement one responsibility at a time. After every non-trivial change:

1. run the targeted test;
2. inspect `git diff --stat` and the targeted diff;
3. review scientific and data-split validity;
4. update evidence only with facts produced by commands or artifacts.

## No Patch Storms

If the same failure survives two repair attempts, stop editing. Reproduce the
smallest failure, localize the layer, record the root-cause hypothesis, test it,
and only then make another repair.

## Scientific Integrity

Never:

- tune on test data or change a split after observing test results;
- change comparison conditions between architectures;
- hide negative results or invent metrics;
- fabricate missing experiments or completion artifacts;
- call semantic-mask graph branches uniquely identified geological faults;
- claim a dataset, run, or gate is complete without stored evidence.

Frozen CRACKS expert folds and synthetic test generator settings must not change
after model results are observed. Thresholds, trace merging, and all
hyperparameters are selected without expert-test feedback. Aggregate uncertainty
by section, never by pixels, skeleton points, or crops.

## Experiment Workflow

Required order:

```text
dataset -> split/leakage -> loader -> unit tests -> vertical smoke
-> full training -> evaluation -> traces -> statistics -> figures/report
-> final validation
```

Full CRACKS training must be resumable and automated by
`scripts/run_cracks_full_study.sh` on Linux. `evaluate`, `traces`,
`statistics`, `figures`, and `report` are independent stages. Never retrain
because a downstream figure, table, GeoJSON, or report failed.

CRACKS uses three explicitly separated settings: A is crowd-only training with
expert evaluation, B is crowd pretraining plus five-fold limited-expert
fine-tuning, and C is image-disjoint robustness. Never describe Setting A as
unseen-image generalization. Certain/uncertain fault colors and annotator
expertise weights follow the frozen semantics policy; undocumented colors never
silently become truth. The normal development path must not read or print expert
test metrics before the relevant architecture and parameters are frozen.

## Runtime Discipline

- Reuse the existing environment under `/home/lebedeffson/Code/venv` on Linux.
- The documented Windows CUDA interpreter is
  `C:\ProgramData\anaconda3\envs\mcda-xai\python.exe`.
- Do not create a new environment or reinstall PyTorch without a demonstrated
  need.
- Store full logs on disk and expose only compact status/heartbeat output.
- Keep datasets, checkpoints, caches, and large generated files out of Git.

## Project Memory

Short durable rules stay in this file. Narrow debugging lessons belong under
`.codex/notes/` and should only be loaded when linked from this file. Never store
secrets, credentials, personal data, private keys, or sensitive environment
values.

The active specification is `docs/research/anza_v2_master_spec.md`. The earlier
`docs/research/cracks_master_spec.md` is historical and must not override v2.
Phase-specific discoveries belong in `.codex/notes/cracks-data-audit.md`.
The frozen structural-affinity negative result and its no-C4 boundary are in
`.codex/notes/affinity-repair-negative.md`.
The GT-connectivity diffusion feasibility failure and its no-D0-D4 boundary are
in `.codex/notes/connectivity-diffusion-negative.md`.
The frozen Structural Reachability Phase-A causal-geometry failure and its
no-Phase-B boundary are in `.codex/notes/structural-reachability-negative.md`.
The frozen Original-ANZA Phase-0 operator-definition mismatch and unavailable
independent confirm split are in `.codex/notes/original-anza-forensics-stop.md`.
The frozen ANZA-2 Phase-3C-A component forensics and RC1 membership-learning
root cause are in `.codex/notes/anza2-phase3c-membership-root-cause.md`.
The bounded Phase-3C-B RC1 M-A/M-B negative result and its no-third-variant
boundary are in `.codex/notes/anza2-rc1-membership-repair-stop.md`.
The frozen Phase-3D-A/B complete-manifest audit and negative oracle mode-state
gate are in `.codex/notes/anza2-phase3d-mode-state-stop.md`.
The separate ANZA-S O0-O4 formal oracle PASS, exact O2/O3 equality, zero
X-correct/StraightGap recall boundary, and no-training lock are in
`.codex/notes/anza-s-oracle-gate-a.md`.
The Phase-A2 Cauchy--Green causal audit, reset-local anisotropy ceiling, zero
incremental A3 gain, and no-Phase-B boundary are in
`.codex/notes/anza-s-a2-cocycle-redundant.md`.
The frozen ANZA-HS H0/H1 StressBench protocol, small favorable B3 point
estimates, failed practical structural gate, and no-H2 boundary are in
`.codex/notes/anza-hs-h1-not-incremental.md`.
The frozen ANZA-FS H3 five-lobe foliation experiment, worse false-bridge and
Dice results, and final no-more-local-kernels boundary are in
`.codex/notes/anza-fs-h3-negative.md`.
The frozen ANZA-EK exact Cat-map E0 PASS, zero-training E1 saturation against
static anisotropy, and no-E2/no-conjugacy boundary are in
`.codex/notes/anza-ek-e0-e1-negative.md`.
The frozen ANZA-KS K0 static-matched PASS, evaluator-only OR-gate repair,
feature-level Kolmogorov/Anosov K1 PASS, and separately-frozen K2 boundary are
in `.codex/notes/anza-ks-k0-k1-causal-pass.md`.
The K1.5 CatKS-vs-ShearKS tie and K2 seed-41 dense-transfer STOP, including the
no-seeds/no-confirm/no-CRACKS boundary, are in
`.codex/notes/anza-ks-k2-transfer-stop.md`.
The immutable ANZA-2 Phase-2A saturation failure, independent Phase-2B oracle
confirmation, and learned/CRACKS claim boundary are in
`.codex/notes/anza2-phase2-selectivity.md`.
The failed learned Phase-3/3B gate, its single causal repair, and the no-confirm
boundary are in `.codex/notes/anza2-phase3-negative.md`.
The evidence-anchored ANZA-KIR IR2 seed-41 result, modest hard-pair signal,
failed practical/Kolmogorov gates, and final no-more-local-symbolic-architectures
boundary are in `.codex/notes/anza-kir-ir2-result.md`.
The frozen ANZA-TraceGraph predicted-tracelet TG1 candidate bottleneck, separate
generator-visible relation diagnostic, and no-TG2/no-confirm/no-CRACKS boundary
are in `.codex/notes/anza-tracegraph-tg2-result.md`.
The zero-training Candidate Audit V2 exact miss reproduction, A/B/C/D/E
taxonomy, longitudinal port-localization finding, K-saturation result, and
generator/protocol mismatches are in
`.codex/notes/anza-tracegraph-candidate-audit-v2.md`.
The corrected TRACEGRAPH_RELATION_V2 semantic audit and SBPP V3-A calibration
coverage stop, including the unopened-development boundary, are in
`.codex/notes/tracegraph-sbpp-v3-a-stop.md`.
The independent SBPP V3-B soft-support calibration/development PASS, selected
`tau_s=0.20`, bounded candidate burden, and weak-branch limitation are in
`.codex/notes/tracegraph-sbpp-v3-b-pass.md`.
The fresh-split exact-P0 Endgame V1 E1--E3 result, strong ranking but failed
safe recovery gate, NONE-score-separation attribution, and no-E4 boundary are
in `.codex/notes/tracegraph-p0-endgame-v1-stop.md`.
The section-disjoint ANZA-LIRA LEADS A0/A1 low-label experiment, large favorable
L3-vs-L2 Dice/clDice diagnostics, failed unknown-white safety gate, L0 boundary,
and no-A2/no-ANZA-MS lock are in `.codex/notes/anza-leads-a1-safety-stop.md`.
The fresh cross-fit LEADS RC1 high-tail calibration result, reversed L3-vs-L2
Dice/clDice deltas, unsupported-white decomposition, operating-point-specific
STOP, and no-downstream boundary are in
`.codex/notes/anza-leads-rc1-operating-point-stop.md`.
The SurfTrack S0 observable 3D-lineage benchmark, train-fitted `lambda=0`, small
non-practical G4-vs-reset effect, shear equivalence, final Anosov-specific STOP,
and no-S1/no-real-data boundary are in
`.codex/notes/anza-surftrack-s0-final-stop.md`.
The final LIRA-Seismic F0/F1 ledger freeze, raster trace-identity boundary,
frozen T1 ensemble calibration, insufficient real natural-gap cohort, and
no-F2/no-F3/no-confirm/no-expert terminal boundary are in
`.codex/notes/anza-lira-final-f1-data-stop.md`.
The separate CRACKS Intervention I1 benchmark, frozen SBPP I2 candidate STOP,
same-component/no-source diagnostic, and no-I3/no-path/no-confirm boundary are
in `.codex/notes/anza-lira-intervention-endgame-stop.md`.
The final Graph-Cut V2 treatment-validity audit, zero-retention context/collateral
failure, locked SBPP/P0/path boundary, and no-V3 rule are in
`.codex/notes/anza-lira-graph-cut-v2-stop.md`.
The terminal H1 flat-cap ribbon correction, passed geometric implementation
tests, 34/1753 mechanical-audit retention failure, unopened sections 347--400,
and final no-more-rescue boundary are in
`.codex/notes/anza-lira-h1-final.md`.
The separate CRACKS Structural Stability V1 SS0/SS1 implementation-validity
PASS, rank-frozen split, historically exposed expert downgrade, frozen-H0
perturbation matrix, and no-new-training boundary are in
`.codex/notes/anza-lira-ss-v1-s0-s1.md`.
The V1.1 SS1.5 pre-training freeze, train-only normalization, crowd geometry
targets, SPD metric transport, shared three-seed initialization/manifests,
full-suite PASS, and unopened 12-job boundary are in
`.codex/notes/anza-lira-ss-v1-1-pretrain-freeze.md`.

## Completion

Only `scripts/validate_anza_v2_study.py` may declare completion. If an external
blocker remains after three reasonable workarounds, write
`results/cracks_study/BLOCKER_REPORT.md` with the exact commands and logs. If
work stops for a platform limit, persist the next exact command and current
failure in `.agent-state/TASK_STATE.json` and use status
`INCOMPLETE_PLATFORM_LIMIT`.
