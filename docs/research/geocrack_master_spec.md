# ANZA-LIRA GeoCrack Master Specification

This document is the durable scientific contract for the GeoCrack fracture
trace study. `AGENTS.md` owns workflow; phase skills route to the relevant
sections here. Missing evidence remains incomplete and negative results remain
part of the study.

## 1. Scope and terminology

Research question: does anisotropic fuzzy local aggregation preserve geological
fractures as thin connected structures better than a conventional U-Net, and
can ANZA-LIRA's native direction field support extraction of separate trace
segments?

The study covers photogrammetric geological outcrop imagery, not satellite
lineaments. Extracted objects are candidate `fracture trace segments` between
network endpoints/junctions. They are not labeled geological instances,
tectonic faults, an Anosov system, or proof of ergodicity.

## 2. Completion contract

Completion requires all of the following:

- official data metadata/download verification and checksums;
- a reproducible source-grouped train/val/test split with zero leakage;
- train-only normalization and a validated dataset loader;
- two-model 32/16/16 one-epoch vertical smoke;
- baseline and `az_thesis` at seeds 41, 42, 43;
- `az_no_fuzzy`, `az_no_aniso`, and `attention_unet` at seed 42;
- pixel, topology, trace, orientation, and runtime metrics;
- source-level bootstrap statistics;
- valid GeoJSON traces, tables, article figures, thesis numbers, and report;
- one-command resumable orchestration and a passing final validator.

No fake values, manual metric transcription, test-set tuning, or research-path
TODOs are allowed. A scientifically negative AZ result is complete evidence.

## 3. Repository and environment

Work in the existing `anza_lira` checkout and environments. Record commit,
branch, remote, OS, Python, torch, torchvision, CUDA, GPU, memory, and package
freeze under `results/geocrack_study/`. Never commit datasets, large
checkpoints, environments, or caches. RTK status is evidence when available.

## 4. Official dataset

Dataset: GeoCrack, DOI `10.7910/DVN/E4OXHQ`; article DOI
`10.1038/s41597-024-04107-0`. It contains 12,158 binary 224x224 patch pairs from
49 source images at 11 sites. Download through the Harvard Dataverse API and
save metadata. Select only patched image/mask data, avoid raw data, support
range-safe/resumable download, and verify file size, SHA-256, and one-to-one
pairing. Store under `data/geocrack/`.

## 5. Source-grouped small split

Create `geocrack_small_v1` with approximate sizes 1245 train, 300 validation,
and 450 test. Extract a source ID such as `DJI_0194` from
`DJI_0194_original_patch155.png`. Group by source image with seed 2026; never
split one source across partitions. Store three CSVs plus a manifest containing
source and patch counts, foreground pixels/fraction, source lists, and SHA-256
for each CSV. Freeze and record the test hash.

The split checker must exit nonzero for any pairwise source overlap and otherwise
print all three overlap counts followed by `STATUS: PASS`.

## 6. Dataset loader, normalization, augmentation

`GeoCrackDataset` returns image `FloatTensor[3,224,224]`, mask
`FloatTensor[1,224,224]`, and metadata with patch/source IDs and paths. Values
must be finite and masks strictly binary. Compute channel mean/std only on train
and reuse it for val/test. Train augmentation may synchronously apply horizontal
or vertical flips, right-angle rotations, and moderate brightness/contrast.
Masks use label-safe discrete transforms. Val/test are deterministic.

## 7. Segmentation configuration and fairness

`configs/geocrack_small.yaml` uses 224 images, 30 epochs, batch 8, seed 42, and
a validation threshold sweep from 0.30 through 0.80 by 0.05 optimizing Dice.
Only architecture may differ across compared models. Split, augmentation,
optimizer, learning rate, epochs, batch, loss, threshold grid, checkpoint rule,
and evaluation code stay identical. If OOM forces batch 8 -> 4 -> 2 -> 1, apply
the selected batch to every model and record it.

## 8. Native axial orientation and anisotropy

For memberships `mu_r`, axial directions `theta_r`, and scales `sigma_u,r`,
`sigma_s,r`:

```text
C = sum_r mu_r cos(2 theta_r)
S = sum_r mu_r sin(2 theta_r)
orientation = 0.5 atan2(S, C)
coherence = sqrt(C^2 + S^2) / (sum_r mu_r + eps)
a_r = tanh(abs(log(sigma_u,r / sigma_s,r)))
anisotropy = coherence * sum_r(mu_r a_r) / (sum_r mu_r + eps)
```

Angles are axes, so `theta` and `theta + pi` are equivalent. Export orientation,
coherence, and anisotropy maps as NPY plus visualizations.

## 9. Mask to skeleton graph

Apply only the validation-selected probability threshold. Skeletonize to one
pixel and build an 8-connected pixel graph. Degree 1 is endpoint, degree 2 is a
normal point, and degree >=3 is a junction. Each maximal chain between endpoint
and/or junction nodes is an initial trace segment.

For edge direction `phi_pq`, axial distance is
`0.5*acos(cos(2*(theta1-theta2)))`. Geometry compatibility is
`(1+cos(2*(orientation-phi_pq)))/2`. Edge confidence is the clipped product:

```text
sqrt(Pp Pq) * sqrt(rho_p rho_q) * sqrt(A_p A_q) * G_pq * G_qp
```

At junctions, estimate branch tangents over five pixels and pair continuations
by axial angle, edge confidence, and probability. Select endpoint merge distance
in {1,2,3,4}, max axial angle in {10,20,30} degrees, and minimum length in
{5,8,12} using validation criterion `0.7*trace_f1 + 0.3*endpoint_f1`.

## 10. Trace objects and GeoJSON

Each object stores trace/source/patch IDs, polyline, pixel and chord lengths,
orientation, coherence, probability, anisotropy, sinuosity, endpoint types, and
confidence. Per test image, export deterministic GeoJSON `LineString` features
with model and seed provenance. Serialization must be standards-valid and
round-trip parseable.

## 11. Metrics

Pixel/topology metrics: Dice, IoU, precision, recall, specificity, balanced
accuracy, clDice, skeleton precision, and skeleton recall.

Trace metrics: skeleton-distance precision/recall/F1 at 2 px, endpoint F1 at 3
px, junction F1 at 3 px, symmetric skeleton distance, matched orientation error
(mean/median/p90 degrees; GT orientation from local PCA radius 5), and relative
total trace-length error. Empty/degenerate cases must be explicit and finite
where mathematically defined, never silently replaced with attractive values.

## 12. Test-first synthetic acceptance

Required tests cover:

- horizontal line: one trace, two endpoints, no junction, <5 degree error;
- two disconnected lines: two trace objects;
- T junction: one junction cluster and three branches;
- X crossing: junction detected, four branches, correct axial pairing;
- `theta` versus `theta + pi` periodicity;
- deliberate train/test source leakage rejection;
- valid GeoJSON round trip and trace metric edge cases.

Observe the expected failure before implementing each risky component.

## 13. Smoke and full run matrix

Before full runs, complete the entire pipeline on 32 train, 16 validation, and
16 test patches for `baseline` and `az_thesis`, one epoch. Verify forward,
backward, finite loss/metrics, checkpoint save/load, validation threshold, test
inference, native geometry, skeleton/graph/traces, metrics, GeoJSON, one figure,
and report.

Full matrix (nine runs):

```text
baseline: seeds 41, 42, 43
az_thesis: seeds 41, 42, 43
az_no_fuzzy: seed 42
az_no_aniso: seed 42
attention_unet: seed 42
```

## 14. Run provenance, resume, and watchdog

Every run stores config hash, commit hash, split hash, seed, start/end time, and
status. Identical COMPLETE runs skip, compatible interrupted runs resume, and a
changed config creates a distinct run ID. A small heartbeat contains run, epoch,
best epoch/validation Dice, last update, and status. Full logs stay on disk.
Training, evaluation, trace extraction, statistics, figures, and reporting are
independent stages.

## 15. Statistics and tables

Do not treat patches as independent. Aggregate patch metrics within each source
image, then across seeds, and perform 2000 source-cluster bootstrap replicates of
AZ-minus-baseline deltas. Report means and 95% intervals. If zero is inside the
interval, say the advantage is not established.

Generate `summary_by_seed.csv`, `summary_mean_std.csv`,
`bootstrap_comparison.csv`, and `trace_metrics.csv` from stored artifacts.
Per-seed columns include model, seed, all core pixel/topology/trace metrics,
orientation/length errors, parameters, and inference milliseconds.

## 16. Figures and example selection

Generate neutral scientific figures as SVG, PDF, and 300-dpi PNG. Required
panels cover segmentation comparison, error changes, native orientation plus
anisotropy plus traces, and model metrics with interval points (not decorative
bars). Keep legends outside content. Automatically select median, best, and
worst per-patch AZ-minus-baseline Dice; use median in the main figure and keep
best/worst as supplemental evidence.

## 17. Thesis numbers and report

Generate `THESIS_NUMBERS.json` with dataset, split, training, baseline, ANZA,
deltas, ablations, bootstrap intervals, trace extraction, runtime, and
limitations. Every value is derived automatically and finite where required.

`FINAL_REPORT.md` explains the question, dataset, trace definition, split,
models, native orientation, trace pipeline, protocol, pixel/topology/trace
results, ablations, uncertainty, median/best/worst cases, gains, regressions,
limitations, claim-safe thesis language, prohibited claims, and exact commands.

## 18. One-command orchestration and gates

`scripts/run_geocrack_full_study.ps1` checks environment, downloads/validates
data, builds/freezes the split, checks leakage, runs tests and smoke, executes or
resumes nine runs, evaluates, extracts traces, bootstraps, renders figures,
generates evidence/report, and invokes final validation. Each stage fails
nonzero with a clear name. A broken downstream artifact never retrains a model.

`scripts/check_current_phase.py` provides compact dataset, smoke, training,
traces, and final gates. `scripts/validate_geocrack_study.py` additionally checks
agent/rule/skill infrastructure, duplicate implementations, RTK evidence when
available, run/config/split hashes, frozen test hash, table consistency, finite
thesis values, nine run/checkpoint/metric sets, GeoJSON, figures, report, and
complete task state. Only then it prints:

```text
------------------------------------------------
GE0CRACK STUDY STATUS: COMPLETE
------------------------------------------------
```

## 19. External blockers and Git

Operational errors such as paths, names, shapes, OOM, checkpoints, or figures
must be diagnosed and repaired. A real external blocker requires three reasonable
workarounds plus `BLOCKER_REPORT.md` containing exact commands and logs. Before
handoff, inspect status/diff, keep large artifacts out of Git, and make a local
commit on `feature/geocrack-trace-study`; never force-push.

## 20. Claim-safe outcome interpretation

Possible valid outcomes include: better pixels and connectivity; similar pixels
but better clDice/trace F1; useful native orientation despite similar pixels; or
a complete negative result explained through ablations. Never change the study
after seeing an unfavorable test result.
