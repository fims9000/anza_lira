# Tver V23: executed counterfactual 2.5D source-conditioned LINK/NO_LINK experiment
**Executed and independently case-audited: 14 November 2026. Controlled synthetic data; no historical ANZA/LIRA E2E.**

- Branch PR: https://github.com/fims9000/anza_lira/pull/28 (Draft).
- Frozen pre-outcome protocol: \`V23_25D_COUNTERFACTUAL_SYNTHETIC_FROZEN_PROTOCOL_20261114.md\`.
- Confirmed CI: https://github.com/fims9000/anza_lira/actions/runs/43591977935 (**success**).
- Aggregate + full synthetic case evidence artifact: \`tver-v23-25d-counterfactual-synthetic-case-audit\`, artifact ID \`14112198781\`, SHA-256 of ZIP \`c9455bdde2ec6e124d52fd545046760576ef233477aa129690630bddff56d010\`.
- Independent verifier \`scripts/verify_tver_v23_case_evidence.py\` returned **V23_INDEPENDENT_CASEWISE_SYNTHETIC_AUDIT_PASS**: recomputed all **960** IID/OOD individual decisions and metrics without running the model, 160 real *independent synthetic nuisance* bases per test, all CAL thresholds, paired base-group 3000-bootstrap CIs, frozen receipts and OOD protocol verdict. No per-world pseudo-replication.

## The actual falsifiable question
A single base generator creates **three counterfactual physical-world hypotheses** with the same exact central 64×64 grayscale fault-like image, same provided source + candidate endpoints and candidate order: source→candidate0, source→candidate1 and NO_LINK. Labels and 2D inputs are therefore mathematically unidentifiable from central pixels alone: a 2D-only deterministic policy must achieve **exactly 1/3 three-way accuracy on complete triplets**. The three worlds differ *only* in z=±1 neighbors (continuation-compatible versus discontinuous near-slice shifts). Nuisance backgrounds are shared across counterfactual worlds within each matched group; all groups are fully segregated across TRAIN/CAL/IID TEST/OOD TEST.

Compared a geometry-only tie, central straight-path image evidence, adjacent straight-path intensity, far-slice sham, a **new source-conditioned 2.5D ridge-continuity matcher** and two label-matched HistGradientBoosting classifiers: 2D-only and **same observed 2.5D feature vector as the matcher**. This prototype estimates a shift of the observed left source ridge in each neighbor and demands right-candidate ridge support at the **same** shift. It does not read a hidden lineage or target annotation. TRAIN = 200 bases/600 worlds. CAL = 100 bases/300 worlds, where thresholds for each scorer were locked at empirical false acceptance ≤2% (CAL NO_LINK) and wrong-branch acceptance ≤1% (CAL positives). IID and OOD TEST each contain 160 fresh bases, 320 positives and 160 NO_LINK counterfactuals.

The first CI stopped before generating any V23 benchmark outcome because the proposed seed spans overlapped. The correction to 10k-separated TRAIN 202623010 / CAL 202633010 / IID 202643010 / OOD 202653010 and a float-assert tolerance is recorded explicitly in the frozen protocol as a **pre-outcome** technical amendment. Models, 2%/1% risk rule, test sizes and OOD family definitions were not altered afterward.

## Results: SAME cases, source-identical matched-world constraints

| Policy | IID correct positive /320 | IID false NO_LINK /160 | OOD correct positive /320 | OOD wrong branch /320 | OOD false NO_LINK /160 | OOD three-way accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Geometry only | 0 | 0 | 0 | 0 | 0 | 33.33% |
| Central straight image gradient | 0 | 0 | 16 | 16 | 16 | 33.33% |
| Adjacent straight-path intensity | 0 | 0 | 72 | 74 | 50 | 37.92% |
| Distant-section sham | 16 | 16 | 10 | 10 | 10 | 33.33% |
| Source-conditioned 2.5D continuity **prototype** | **320** | **0** | **312** | **0** | **0** | **98.33%** |
| Learned 2D HGB | 0 | 0 | 0 | 0 | 0 | 33.33% |
| Learned 2.5D HGB, same features | 318 | 0 | 288 | 0 | 0 | 93.33% |

The geometry/central-only 33.33% is **not a published CNN failing to detect seismic faults**; it follows exactly from the three-world 2D identifiability construction. Even the geometry rule implemented here is intentionally tied and not an optimized real fault tracker. The 2.5D HGB is a genuine trained equal-information baseline on this **specific synthetic generator**, not a trained FaultSeg3D, U-Net or original LIRA. The far-slice sham collapses to one-third accuracy, so independent unrelated textures do not solve the task.

### OOD subfamilies (160 bases total; not independent within triplets)
- Triangle-like longitudinal z deformation: proposed matcher **158/160** positive links, 0 wrong, 0 false NO_LINK; 2.5D HGB **143/160**.
- Two-lobed deformation: matcher **154/160**, 0 wrong, 0 false NO_LINK; HGB **145/160**.
- OOD additionally changes ridge amplitude, background noise, distractor lines and ±2.4px per-slice registration; this is controlled perturbation, **not geological distribution shift**.
- Matcher minus 2.5D HGB whole-three-way accuracy = **+5.00 percentage points**, paired base-bootstrap 95% percentile interval **[+2.9167,+7.0885] pp** across **160** independent nuisance bases, not 480 worlds. IID advantage = +0.42 pp, interval [0,+1.04] pp; it does not establish a genuine IID advantage.

Both IID (320/320 true links, 0/160 false) and OOD (312/320 true links, 0/160 false, 0/320 wrong) exceeded the **predeclared conditional selective acceptance criteria** on this synthetic distribution, including beating the same-information HGB on OOD. Zero observed false accepts does **not** bound geological risk to 0%; under unrealistically independent 160 negatives a one-sided 95% upper bound would still be about 1.85%, and actual counterfactual worlds share nuisance groups. Need group-level uncertainty for risk and independent generators.

## Scientific interpretation and limitations
**What is supported:** 2.5D neighboring images can resolve a constructed 2D non-identifiability problem; matching the source strand's shift to a destination provides much better *synthetic* link/no-link decisions than geometry, naive adjacent gradients and the current trained same-feature HGB on two held-out synthetic deformation families. All classification thresholds were frozen on separate CAL groups. The independent auditor verified every case and exact group provenance.

**What is NOT supported:** a claim that the **historical** ANZA→SBPP→LIRA pipeline outperforms published models; any learned ANZA segmentation, predicted endpoint detection, naturally negative geological fault-instance labels, accurate recovery of missing seismic pixels, inter-survey generalization, or comparison with 3D FaultSeg3D/FaultVitNet/STA-Fault3D segmentation quality. The required original model checkpoints are absent from PR #25 and #28. The new method is a **designed bridge-matching prototype** with oracle-provided endpoints and a synthetic generator that rewards an explicit z-shift continuity assumption. Generating NO_LINK via mismatched right-arm displacement may create an artificial detector shortcut; this must be challenged by a separate renderer whose marginal nonlink geometry is matched. The central image construction forces a hard 2D problem by design; comparing with a 2D CNN would test information availability, not architectural superiority.

## Publication-safe Tver conclusion
Use V23 to state a *principle*: a source-conditioned, abstaining 2.5D path-verification module can overcome deliberately constructed 2D projection ambiguity while controlling false-accept decisions on selected held-out synthetic deformation regimes. Pair it with V22 (actual F3 image gradients: 38/40 vs synthetic geometric tie 17/40) only as separate evidence that observed seismic pixels contain conditional path information. Do not join V23+V22 numbers into a fabricated real ANZA-LIRA score. V16's older two-lobed OOD risk failure must still be mentioned: V23 is a different model, scene generator and task, not a corrected V16 pretrained model.

**Next necessary and not yet completed:** a pre-registered independent *second* synthetic 3D renderer, materially harder matched-marginal NO_LINKs, hallucinated/absent-slice controls, a 2.5D CNN or graph-network baseline trained on the identical labeled worlds, and real predicted ANZA ports after retrieving/retraining missing original checkpoint. Only after that can a stronger model-level comparative claim be considered.
