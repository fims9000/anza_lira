# Evidence ledger — Tver manuscript (2026-10-29)

Status: arithmetic and source-text evidence, **not** a new trained-model evaluation. Historical results retain their frozen labels, including STOP statuses.

| Evidence family | Provenance and count | Allowed statement | Unallowed inference |
|---|---|---|---|
| Historical pair test | 128 positive + 128 negative; true 86/128; false 1/128; AUROC 0.9923 | Contextual classifier can discriminate controlled hidden-gap cases | Generalization to geological natural gaps |
| Historical maximin path | 128/128 when correct endpoints are provided | The path routine succeeds conditional on an oracle relation | End-to-end system 100% success |
| SBPP V3-B candidate generator | 2611/2688; weak-branch 117/192; 75 of 77 misses weak | Candidate generation is strong on the controlled development generator, but weak branches dominate residual error | Cross-pipeline accuracy or a real seismic external score |
| TG1 Candidate Audit V2 | Endpoint@6 818/1024 (misses: A=148, B=1, C=21, D=1, E=35) | The original port-localization bottleneck is mostly endpoint shift; increasing K has limited effect | Same experiment or paired model improvement relative to SBPP V3-B |
| TraceGraph P0 Endgame | Available candidates 2605/2688, accepted correct 1202/2688, false bridges 19/1152, 7 wrong branches /2605, ranking AUROC 0.978948 | The fixed low-false-bridge decision achieves only 44.72% relation recovery; frozen gate failed | Path reconstruction ran in E4; 97% candidate recall guarantees acceptance |

Source paths:

- `docs/RESEARCH_STATUS_FINAL.md`
- `docs/research_history/__home__lebedeffson__Code__anza_lira__results__anza_tracegraph__sbpp_v3_b__TRACEGRAPH_SBPP_V3_B_REPORT.md`
- `docs/research_history/__home__lebedeffson__Code__anza_lira__results__anza_tracegraph__candidate_audit_v2__CANDIDATE_AUDIT_V2_REPORT.md`
- `docs/research_history/__home__lebedeffson__Code__anza_lira__results__anza_tracegraph__endgame_v1__e3_relation__TRACEGRAPH_P0_RELATION_REPORT.md`
- `.codex/notes/anza-lira-final-f1-data-stop.md`

## Analytic results and cautions

The P0 fixed-decision upper bound after restoring every missing candidate is (1202 + (2688-2605))/2688 = **1285/2688 = 47.8051%**. It is a **counterfactual bound conditional on unchanged decisions for available candidates**, not a retraining forecast.

In the SBPP V3-B development study, 75/77 candidate misses are in the weak-branch stratum. This fraction is not a guarantee about natural faults.

In the TG1 audit, 148/206 original misses are due to endpoint shifting; this diagnosis uses the frozen operational taxonomy, not independently annotated error-cause truth.

The source generator `synthetic/seismic_background.py` draws a random transition width per *latent instance*, then draws background frequencies, phases and noise from the same RNG. Two examples with the same seed but a different instance count therefore have **different nuisance backgrounds by construction**. This is a potential confound to test, not proven label leakage or an estimate of bias magnitude.

The CRACKS natural-gap audit found only 3 calibration and 1 development positive gaps; expert identity across natural disconnected fault structures has not been established. STOP decisions stay immutable.

Future Q1 2.5-D and calibrated LINK/REJECT/REVIEW are **prospective**, not measured results for Tver.

## Editorial checks

Confirm all authors, organizations and contribution statements; authorization for specific state-assignment acknowledgment; peer-reviewed prior publication overlap; equation editability; DOI and URL correctness; journal licensing and expert conclusion. Do not insert results from the Varvara medical or bubble-chamber branches.
