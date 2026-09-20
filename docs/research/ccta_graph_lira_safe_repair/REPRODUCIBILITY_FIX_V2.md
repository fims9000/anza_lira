# Cross-patient Graph-LIRA V2 reproducibility correction

Date: 2026-09-20.

## Problem found

The first cross-patient perturbation-consistency implementation derived the random perturbation seed partly from a global integer scene index.

That index depended on the full list of stress configurations. As a consequence, adding a new configuration such as `60 deg + 1.5 mm` changed the perturbation draws for pre-existing `30 deg + 1 mm` scenes.

The base structural predictions were deterministic and unchanged, so the exact/false summary table remained stable. The **risk/coverage table**, however, could shift slightly because its Monte Carlo perturbations were not keyed to a stable scene identity.

This is a reproducibility defect, not a scientific effect.

## Fix

V2 assigns each controlled scene a stable identity:

```text
<case>:b<branch_index>:r<rep>:a<angle*10>:j<jitter*100>
```

The perturbation seed is derived only from:

- case ID;
- branch index;
- repetition;
- angle;
- jitter;
- perturbation repetition `k`.

It no longer depends on the global scene position or on what other configurations are included in the same run.

## Verification

Two V2 runs were executed:

1. full stress grid including `60 deg + 1.5 mm`;
2. otherwise identical grid with the 60-degree configuration removed.

Checks:

- the complete `30 deg + 1 mm` risk/coverage table was identical;
- all summary rows for 15, 30 and 45 degrees were identical;
- the script SHA256 is `6beb28bb28a6687260763eb30e40e71c8bccf8e6fb1422156692b12b1dcf2aad`.

## Consequence

Canonical files use the `cross_patient_graph_v2_*` prefix.

The older files without `v2` are retained for provenance but must not be used for final perturbation-consistency numbers.

This correction changes previously quoted **coverage / accepted-count values** from the exploratory risk table. It does not change the primary base-scene exact/false rates for joint Graph-LIRA, sequential repair, or local independent repair.

## Canonical primary risk numbers

At `30 deg + 1 mm`, consistency `>= 0.90`:

- train 921 -> test 953: coverage `63.43%`, 137 accepted, 0 false, exact among accepted `99.27%`;
- train 953 -> test 921: coverage `44.97%`, 67 accepted, 0 false, exact among accepted `98.51%`.

These replace the earlier exploratory values `130` and `80` accepted scenes.

## Archive

- source: `scripts/research/ccta_graph_lira_safe_repair/ccta_graph_lira_cross_patient_v2.py.gz.b64`
- protocol: `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_protocol.json`
- summary: `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_summary.csv`
- risk/coverage: `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_risk_coverage.csv`
