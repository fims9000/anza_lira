# ANZA-2 Phase 3D-A/B mode-state oracle STOP

Date: 2026-08-18.

Protocol `ANZA2_PHASE3D_CONTEXT_MODE_STATE_V1` was executed without training.
The complete frozen CrossingTraceBench-v4 composition contains 512 samples in
each of train, validation, and confirm, with zero seed overlap. Confirm was
inspected only for manifest composition; no confirm scores or thresholds were
computed. All eight mandatory curriculum strata occur in every split.

The visible/latent adapter removes positive-gap latent axes from local
membership supervision. Latent axes were used only in the mathematical oracle
feasibility calculation. The frozen 11 px encoder receptive field covers the
7.347 px q90 primary local scale, so no context-block change was authorized in
this no-training phase.

Thresholds were selected from all eligible train samples at FPR <= 0.05 and the
gate was evaluated once on validation. G0 scalar versus G1 mode-state gave:

- positive continuation recall: 0.6910 versus 0.6806;
- X correct recall: 0.1500 versus 0.2000;
- X wrong-turn FPR: 0.0875 versus 0.1125;
- X wrong-turn relative reduction: -0.2857 versus required >= 0.50;
- parallel false bridge: 0.0 versus 0.0;
- negative-gap false bridge: 0.0 versus 0.0.

The frozen gate failed both positive non-inferiority and X wrong-turn reduction.
Research status is `FINAL_STOP_MODE_STATE_ORACLE_NO_VALUE`. Phase 3D-C,
synthetic confirm evaluation, CRACKS, and expert evaluation are not authorized.
Do not add a new mode-switch penalty, tune domains, change thresholds, or launch
training under this master protocol. The result is a valid negative oracle
feasibility result, not evidence about trained or real-data performance.

Canonical evidence is under `results/anza2/phase3d_ab/` and is validated by
`scripts/validate_anza2_phase3d_ab.py`.
