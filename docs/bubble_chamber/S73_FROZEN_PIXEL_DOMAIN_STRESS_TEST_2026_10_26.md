# S73 preregistration — locked pixel-domain stress testing after S72

**Frozen 2026-10-26, before S73 scene generation or viewing any S73 outcomes.** Branch `research/bubble-chamber-graph-lira`. Experiment ID `ANZA-LIRA-BC-S73-20261026`.

## Why

S69–S71 corrected a source-array orientation bug in historical *synthetic-only* scoring. S72 CAL-selected `s68_branch_endpoint_refine` for 005→006 and `s39_original` for 006→005, and found a conditional advantage on 96 untouched artificially injected scenes. This does NOT establish robustness across cameras, physical events, different photographic runs, film emulsions, or conditions outside one original H50Z run.

S73 measures how these **frozen** image-only methods behave if the same artificially inserted scenes are observed under several explicitly defined pixel perturbations of the *real-photo-backed simulated image*. This tests sensitivity to plausible acquisition artifacts, not authentic independent runs. The source photograph backgrounds remain two H50Z frames from one physical run; measured source curves come from distinct visualMetro film 070; none are native H50Z ground truth.

## Locked algorithms and test split

- Unchanged predictors and geometry scorer, same fingerprints and S37 frame-disjoint 79/30/36 visualMetro source split as S72. S69 orientation-invariant synthetic evaluator only, never historic orientation-dependent evaluator.
- **Do not refit or retune any S39/S67/S68 detector or connection threshold** on S73 TEST. Refit frozen S56 source-photo geometry scorer from its original TRAIN/CAL seeds exactly as S72 did. Load **immutable S72 CAL JSON** and verify its SHA, S72 protocol SHA, original code hashes, source split hash, and source-photo scorer cutoff. The S72 CAL-selected method per source transfer is immutable.
- Test four original S72 arms, including S39, S67, S68 branch split, S68 branch split + endpoint refinement. For each injected scene, run all four on the **same image pixels** under each stress condition; only after predictions finish does the S69 evaluator receive the generator-only source polylines.
- Fresh **S73** scene seeds: 005→006 `20267303`, `20267304`; 006→005 `20267313`, `20267314`; exactly 24 scenes per seed (96 scenes). Geometry templates from S37's 36 held-out film-070 source frames. H50Z background from the opposite-photo bank as in S72; no S64–S72 scene seed reused.

## Predeclared image perturbations (no source truth used)

Every identical artificial scene is evaluated at five conditions, with per-scene perturbation randomness derived from the frozen S73 scene seed and index, independent of true source label:

1. `clean` – unchanged gray image.
2. `contrast_072` – `round((gray-128)*0.72 + 128)`; compress intensity contrast about mid-gray.
3. `blur_sigma_11` – `cv2.GaussianBlur(gray, sigma=1.1)`.
4. `grain_sigma_9` – add i.i.d. zero-mean Gaussian grayscale noise, sigma=9/255, clip and round; fixed independent RNG for each synthetic scene.
5. `two_film_streaks` – draw two 1-pixel straight synthetic dark/light lines at randomly selected image coordinates. They are **negative-control perturbations**, not expert-confirmed historical film defects. Use no label information to position them.

No other perturbation families, strength selection or post-test threshold adjustments permitted. The test does NOT evaluate different camera exposure hardware, newly collected data, or independent photographic experiments.

## Predeclared metrics and uncertainties

For each direction, condition, and arm report total synthetic-correct selected bonds, selected synthetic-noncorrect bonds, scenes with >=1 noncorrect bond, fully recovered 3-piece artificial paths, candidate polylines, source fragment associations, solver failures. Report CAL-selected-vs-S39 **paired by artificial scene** bootstrap differences and each method's degradation-vs-clean with 1200 paired scene-bootstrap replicates per seed; resample whole scenes, not pixels, points, line fragments, crops or original photo IDs. An S72 choice can be called **conditionally tolerant** only if, for all four stressors and both photo directions, synthetic-correct and full-path totals are no lower than S39 and synthetic-noncorrect totals are no greater than S39. Strong generalization to real physics is *never* supported by this experiment, regardless of outcomes.

Optional descriptive pass/fail of this stringent gate is recorded **after** evaluation; it cannot affect selecting the algorithm. No output is allowed to authorize any native physical join.

## Provenance and privacy

Save SHA-256 source code locks for S56, S68 predictor+benchmark, S69 evaluator, S72 protocol, S72 CAL JSON, S73 frozen protocol, S37 source split; public source-only JSON aggregate by condition and direction. Preserve private inputs, per-scene images, operator measured geometry, photo-specific 2D polylines, weights, and native H50Z coordinates outside public Git. Never serialize frame names or reconstructable private identifiers to GitHub.
