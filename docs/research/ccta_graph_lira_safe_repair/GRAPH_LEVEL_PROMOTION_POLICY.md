# Frozen graph-level promotion gate for CT-conditioned Graph-LIRA

Date: 2026-09-21

This gate is frozen before the 28-patient real CCTA experiment and before any new CT-conditioned graph result is available.

## Input contract

Two scene-level prediction files on exactly the same held-out scenes:

Canonical:

`scene_id, scan_id, exact, false_scene, incomplete`

CT-conditioned:

`scene_id, scan_id, exact, false_scene, incomplete`

Optional shared columns may include:

- `kind`;
- `junction_degree`;
- `junction_segment`;
- `accepted`;
- `needs_repair`.

The evaluator refuses mismatched scene sets or duplicate keys.

## No policy tuning

This script does not select a relation threshold, CT threshold, `tau`, or consistency setting.

Those decisions must already be frozen from training/validation.

For the canonical research line:

- relation confidence `tau=0.85`;
- perturbation consistency `0.60`.

## Main graph-level rule

A CT-conditioned Graph-LIRA model is a held-out Pareto improvement only if:

1. structural exact rate is strictly higher than canonical;
2. false structural-repair rate does not increase.

An exact gain accompanied by more false structural repairs is explicitly labeled:

`EXACT_GAIN_WITH_FALSE_RISK_INCREASE`

and is not promoted as an improved model.

If false repair decreases but exact does not improve, the result is labeled as a safety/veto tradeoff rather than a general improvement.

## Patient-cluster uncertainty

The patient is the resampling unit.

The evaluator performs paired patient-cluster bootstrap on:

- exact delta;
- false structural-repair delta;
- incomplete delta.

A stronger descriptive support flag requires:

- the Pareto rule;
- P(delta exact > 0) >= 0.95;
- P(delta false > 0) <= 0.20.

This is not used to retune the held-out result.

## Anatomy / topology strata

If available in the input, the exact same paired comparison is emitted for:

- scene kind;
- junction degree;
- junction segment.

This is intended to expose LAD/LCX and high-degree junction behavior without changing the global acceptance rule.

## Executable source

`scripts/research/ccta_graph_lira_safe_repair/evaluate_graph_promotion_gate.py.gz.b64`

Source SHA256:

`54e7f9cb2de826c01bce93baf8159106d6000f6fbbf0e87df28b35c70a72bd6c`

Outputs:

- `graph_promotion_gate.json`;
- `GRAPH_PROMOTION_GATE.md`;
- `graph_patient_metrics.csv`;
- `graph_paired_patient_cluster_bootstrap.csv`;
- optional `graph_strata.csv`.

A deterministic synthetic paired contract test was run and produced the expected `STRONG_GRAPH_LEVEL_SUPPORT` status for an intentionally Pareto-improved fixture. Those synthetic values are test fixtures only, not research results.
