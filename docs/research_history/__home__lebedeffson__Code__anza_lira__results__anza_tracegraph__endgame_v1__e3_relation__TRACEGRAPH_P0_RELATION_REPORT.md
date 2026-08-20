# TRACEGRAPH P0 ENDGAME V1 — E1 to E3

Status: `STOP_P0_RELATION_SELECTOR`

Exact historical five-convolution corridor P0 was retrained on fresh source-disjoint relation streams. SBPP V3-B remained frozen at `tau_s=0.20`, `K=12`; path, confirm, CRACKS, expert, Transformer, and ANZA changes remained locked.

## Calibration

- selected threshold: `0.999192417`
- calibration RR / FB / WB: `0.469866 / 0.019965 / 0.001153`
- selection rule: maximize all-positive RelationRecovery subject to `FB<=0.02` and `WB<=0.03`.

## Fresh relation development

- CCR: `0.461420` (gate >=0.87)
- RelationRecovery: `0.447173` (gate >=0.84)
- FalseBridge: `0.016493` (gate <=0.02)
- WrongBranch: `0.002687` (gate <=0.03)
- NONE recall: `0.983507` (gate >=0.90)
- candidate availability: `2605/2688`

## Secondary diagnostics

- AUROC: `0.978948`
- Brier: `0.057815`
- ECE: `0.047784`
- TPR_at_FPR_0_05: `0.862956`
- balanced_AUPRC: `0.975338`
- low_FPR_pAUC: `0.832852`
- pair_ranking: `0.978503`

## Weak branch boundary

- candidate availability: `0.614583`
- candidate-conditional CCR: `0.220339`
- all-source RR: `0.135417`

No weak-branch system-success claim is made unless its all-source RR reaches 0.70.

## Frozen failure attribution

- bottleneck: `NONE_SCORE_SEPARATION`
- best development RR under `FB<=0.02, WB<=0.03`: `0.466518`
- minimum development FB at `CCR>=0.87`: `0.155382`
- accepted NONE sources at the frozen threshold: `19`; by stratum: `{"independent_collinear_fault": 18, "parallel_wrong_only": 1}`

This is a post-gate diagnostic only. It does not reopen calibration, change the selected threshold, or authorize a repair in this cycle.

## Decision

E4 widest path is authorized only when status is `P0_RELATION_SELECTOR_PASS`. This run stops before E4 in every case.
