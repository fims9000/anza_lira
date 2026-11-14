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


## Additional REAL-PIXEL but artificial-gap E2E evidence, V27–V30 (14 November 2026)

These numbers are **not** historical P0 replay and not a replacement for the 2026-10-29 synthetic/historical claims above. Original ANZA/P0 model checkpoint bytes are not in Git. Real CRACKS F3 image intensity and the original repository AZConv/SBPP P0 code were used with *new* weights, *expert-selected artificial 12-point gap rectangles*, and candidate-absent fake NO_LINK controls, with source labels available in only 12 of 40 original expert sections. Label-eligible split: TRAIN 10,13,17,22,24,35,41; CAL 43,67; BUFFER 106; TEST 108/124. TEST has been repeatedly inspected, from V27 onward; all V28 and V30 iterations are **posthoc**, not confirmatory.

| Evidence family | Verified stage counts | Allowed statement | Forbidden inference |
|---|---|---|---|
| V27 real AZConv ANZA at pixel-F1 CAL threshold .60 | 0/2 SBPP source ports, 0/2 true candidates, 0 trained positive pairs, 0/2 accepted paths | Newly trained ANZA predicted a dense segmentation unsuitable for the SBPP extractor at this threshold | The historical ANZA segmentation was inferior or a native LIRA was trained in V27 |
| V27 real U-Net at pixel-F1 CAL .50 | 2/2 source ports, 1/2 true candidate, 0/2 accepted paths | More useful predicted candidate source, but CAL 12-feature logistic refused all | Any model has recovered real natural fault-instance gaps |
| V28 changed only segmentation probability threshold to ANZA .90 / U-Net .60; weights identical SHA as V27 | Both 2/2 true SBPP predicted candidates; both actual 12-feature logistic pair heads TRAIN fitted; 0/2 accepted paths | Topology-aware CAL thresholding can make correct source/candidate available in both controlled F3 TEST scenes | Model architecture was changed or a new independent TEST has confirmed superiority |
| V30 newly trained original 5-conv P0 **architecture** on true predicted SBPP inputs and real observed RGB, not old weights | ANZA P0 seed11/23 ungated top1 1/2 each; U-Net P0 0/2 each. Same-positive-budget matched logistic top1 2/2 for both; all ten method×frontend CAL refusals, 0/2 accepted/recovered | Native contextual P0 architecture is now actually exercised with 4 checkpoint-hashed new CNNs, but on 2 already seen scenes it does NOT beat 12-feature logistic | The original historical contextual LIRA weights have been recovered; CAL full refusal is a meaningful geological low-false-link guarantee; a large modern CNN is globally inferior |

Audit sources:
- [V27 CI 43596275581](https://github.com/fims9000/anza_lira/actions/runs/43596275581) and [V27 detailed report](./V27_REAL_PREDICTED_E2E_EXECUTED_NEGATIVE_RESULTS_20261114.md)
- [V28 CI 43596665171](https://github.com/fims9000/anza_lira/actions/runs/43596665171) and [V27–V28 independent review](./V27_V28_REAL_CRACKS_PREDICTED_E2E_COMPARATIVE_AUDIT_20261114.md)
- [V30 CI 43598608411](https://github.com/fims9000/anza_lira/actions/runs/43598608411) and [V30 40-case independent model-state audit](./V30_REAL_F3_PREDICTED_P0_EXECUTED_NEGATIVE_RESULTS_20261114.md). V30 native original P0 two seeds per frontend, strict CAL-only risk gate, all decisions refused.

**Editorial consequence:** If a Tver revision includes these new results, it must present V27–V30 as a separate exploratory real-image case study with only two source-section TEST scenes, clearly mark all 0/2 complete recovery and avoid mixing with historical synthetic 2611/2688 or 0.9923 AUROC. No SOTA or statistical architecture superiority has been established.
