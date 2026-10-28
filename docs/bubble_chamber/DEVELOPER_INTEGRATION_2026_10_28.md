# ANZA–LIRA S35–S85: договор интеграции 2D-треков и решение о применимости

**28.10.2026 · ветка `research/bubble-chamber-graph-lira` · `ANZA-LIRA-2D-INTEROP-V1`**

**Короткое решение:** модуль пригоден как **исследовательский, read-only рекомендатель 2D-соединений** после специально обученного/откалиброванного для нового детектора оценщика. **НЕ пригоден для автоматической физической идентификации частиц, 3D-импульса или достоверной оценки треков H50Z**. В текущем H50Z исследовании default — исходный S72; S85 обучаемые дополнительные линии ещё не проходили приватный TRAIN/CAL/TEST. Не интерпретировать одно физическое событие с 39 кропами как 39 независимых примеров.

## Что система делает и что не делает

```
                                  ВЫБИРАЕТСЯ РАЗРАБОТЧИКОМ
  uint8 192×192 image ───► detector.detect() ───► 2D (x,y) polylines (N×2)
                              │                       │
                              │        geometric pair features / S59 strict gate
                              │                       ▼
                              │           scorer(rows) — обучается отдельно
                              │                       │
                              │           cutoff + ambiguity margin from CAL
                              │                       ▼
                              │           S56 forest / port-greedy / mutual
                              │                       │
                              └──────────────────────►2D proposal graph
                                                        UNKNOWN joins
                                                    no real physics GT
```

Скорер использует только геометрию пары (зазор концов, касательные, взаимную ориентацию, относительную длину, кривизну), если явно не добавлены собственные признаки внешнего детектора. S56 ограничивает один стык на **конец** фрагмента и строит **ациклический лес**. Это осознанная гипотеза для нескольких открытых кривых, **не** универсальный закон физических событий. Петли, разветвления, вторичные взаимодействия, перекрёстные проекции и 3D без дополнительного аппарата могут нарушать модель.

Базовый S72 в текущей ветке — **машинные Hessian/ridge-2D-полилинии, S59, обученный на синтетике S56 geometric scorer, графовый решатель**. Название «ANZA–LIRA» относится ко всему исследовательскому проекту; не следует выдавать S72 за эквивалент любой предобученной ANZA-сегментационной нейросети. S80/S85 предлагают дополнительные пиксельные полилинии, а не реконструируют физические частицы.

## Minimal API (verified without private data)

Install into isolated environment from branch root:

```bash
python -m pip install -r requirements-interop.txt
python scripts/research/bubble_chamber/example_external_detector.py
python -m pytest -q tests/test_interop_*.py
```

Use your own detector and geometry scorer only after **source-only TRAIN and CAL** for its specific candidate distribution:

```python
import numpy as np
from bubble_chamber.interop_api import process_image
from bubble_chamber.interop_calibration import CalibrationReceipt

class MyPolylineDetector:
    detector_id = 'my-curvilinear-detector-v1'

    def detect(self, gray: np.ndarray) -> list[np.ndarray]:
        # Return x,y image coordinates, >=8 points per connected open curve,
        # 0..191 range, up to 32 curves. Implement own segmentation here.
        return my_detector(gray)  # implemented by embedding application

receipt = CalibrationReceipt.from_file('/PRIVATE/my_detector_calibration.json')
scorer_model = load_trusted_model('/PRIVATE/model')  # your *trusted* loader
scorer = lambda rows: scorer_model.predict_pair_probabilities(rows)
binding = receipt.binding(detector_id=MyPolylineDetector.detector_id,
                          model_file='/PRIVATE/model',
                          calibration_file='/PRIVATE/source_cal_artifact.json',
                          scorer=scorer)
result = process_image(gray_tile_uint8_192, MyPolylineDetector(), binding)
for proposed in result.joins:
    assert proposed.status == 'UNKNOWN'
# The return value is not a 3D/reconstructed particle: it is a 2D hypothesis.
```

`MyPolylineDetector`, `my_detector`, `load_trusted_model` and `gray_tile_uint8_192` are **user-provided**, not hidden framework APIs. The included runnable `example_external_detector.py` uses generated pixels and an explicitly **uncalibrated** geometry heuristic, only to demonstrate the interface.

### Contract v1

| Component | Required input | Output and safety | Source |
|---|---|---|---|
| Detector | single `uint8` 192×192, no GT | ≤32 open polylines, arrays `(N≥8, 2)` in **(x,y)** with finite coords 0..191; stable `detector_id` | `interop_detectors.py`, `interop_api.py` |
| Scorer | S59-approved `list[dict]` of 6 geometric features | finite shape `(K,)` values [0,1], not necessarily probabilities without CAL | `ScoreBinding` |
| Frozen calibration receipt | detector ID, private model SHA-256, CAL SHA-256, source domain, cutoff, strategy | fails closed if detector/model changed; actual binary **outside Git** | `interop_calibration.py` |
| Solver | `none`, `node_mutual`, `port_greedy`, `global_milp` | endpoint-oriented reversible proposals with status `UNKNOWN`, does not authorize physics links | `interop_api.py` |
| Optional S80 extra candidates | original lines + `S80ConsensusAugmenter` | original identity and order preserved; **separate composite-detector model/CAL is mandatory** | `process_with_extras` |
| S85 learned extra gate | private TRAIN-only model, CAL selections | **not portable** across detector/domain without retraining; current S85 private model was not trained here | S85 research workflow |

**On changing detector**: never copy a cutoff or learned model from S72 as-is. Change `detector_id`, retrain the pair classifier on the new detector's outputs using TRAIN only, recalibrate the false-positive budget on CAL, freeze `CalibrationReceipt` with the *new* model checksum, then open untouched TEST. If adding S80 extras, the **composite detector** `my-detector+s80-consensus-add6` requires independent score/CAL binding, not just changing a Python function name.

### Geometry / performance / failure boundaries

- Input image and candidate coordinates must use the **same scale and crop**. 512×512 H50Z tiles need a documented 192×192 extraction transformation. The caller owns offset and inverse transform; do not silently report crop-local pixels as original image or 3D coordinates.
- `pair_rows` enumerates up to `n(n-1)/2` pairs; 32 candidates mean up to **496 raw pairs**, then S59 filters. `global_milp` can solve repeatedly for ambiguity via counterfactual bans; **`solver_timeout` is per MILP call, not a hard end-to-end request deadline**. For interactive services use a caller-level timeout/circuit breaker, max-candidate budget and fallback to no joins.
- Default API explicitly **fails closed** on invalid image shape/dtype, corrupt geometry, missing/changed calibration binding, invalid scores and solver failure. It never returns native physical truth. It does not silently fall back to the legacy ANZA score.
- Model confidence is detector-specific: photometric sign, image domain, degree of blur, noise, camera scan resolution, particle line widths, line order/orientation and incomplete fragments all change distributions. Imported scores must be evaluated on **CAL and TEST**; the bare numeric score is not a verified physical probability.
- The previous native H50Z crop audit is **machine-only UNKNOWN**. The source visualMetro film070 curves are **unrelated to those same H50Z physical events**. Synthetic-injected correctness does not establish physical track identity.

## Results already verified *on the original branch*, not interchangeable

Source: frozen `docs/bubble_chamber/S84_PUBLIC_FRESH_TEST_2026_10_26.json`, 96 **artificially inserted** scenes/condition on H50Z image backgrounds of **one physical run**, S69 corrected scorer. Correct/noncorrect/complete *synthetic* graph results:

| Condition | S72 baseline | S84 pixel-only extra gate | S84 one-contact gate | S84 two-contact gate |
|---|---|---|---|---|
| Clean | **43 / 35 / 10** | 47 / 42 / 12 | 45 / 40 / 11 | 45 / 35 / 11 |
| Blur σ=1.1 | **14 / 19 / 3** | 15 / 24 / 3 | 15 / 22 / 3 | 14 / 19 / 3 |
| Blur σ=1.6 | **10 / 13 / 2** | 10 / 12 / 2 | 10 / 13 / 2 | 10 / 13 / 2 |

Format: `correct selected links / other or unverified selected links / complete three-fragment chains`. CAL chose original S72 for `005→006` and photo gate for `006→005`, but that second choice failed its no-increase-of-noncorrect requirement on fresh TEST. **S84 NOT promoted**. S85 learned gate remained **SOURCE_AND_PIPELINE_VERIFIED / PRIVATE_TRAIN_CAL_TEST_NOT_RUN** on 28 Oct 2026. Do **not** report toy QA numbers as S85/H50Z improvement.

## Comparison when swapping the detection baseline (GENERATED-ONLY controlled QA)

Source: `docs/bubble_chamber/S85_INTEROP_PUBLIC_TOY_AGGREGATES_2026_10_28.json` and frozen protocol; **20 generated synthetic scenes per condition**, two generated backgrounds, artificial sine-shaped curves, shared uncalibrated geometry scorer and fixed cutoff 0.70. **No H50Z photos / no visualMetro source lines**.

| Input to same global graph | Clean complete paths | Blur σ1.1 | Scratch/grain | Clean correct / other joins |
|---|---:|---:|---:|---:|
| Hessian/ridge, pixel-only | **10** | **5** | **10** | 27 / 20 |
| OpenCV LSD straight segments | 4 | 1 | 2 | 12 / 47 |
| OpenCV probabilistic Hough | 0 | 0 | 1 | 4 / 36 |
| Artificial oracle segments (evaluator-only, NEVER deployable) | 13 | 13 | 13 | 46 / 10 |

Maximum = 40 full artificial paths/condition (2 per scene). Neither oracle nor synthetic line identity is allowed into inference. These outcomes demonstrate that **input fragment quality strongly limits graph performance**. In this controlled QA, changing to straight-line detectors is often worse for curved tracks, although LSD/Hough might be useful for cleaner/straighter data. An independently trained neural segmenter might improve candidate recall with a suitable labeled domain, but is **not evaluated here**.

**Ablation of the graph itself:** with Hessian lines, `port_greedy` and `global_milp` both assemble 10 clean full paths (27 correct / 20 other), and 5 under blur. On LSD clean, global MILP assembles 4 paths (12 correct / **47 other**); port-greedy assembles 3 paths (7 correct / **21 other**). Thus the more complex graph optimization can **harm precision**. It must be treated as a tunable alternative, not an unconditional upgrade. Sample size is tiny and the geometry heuristic is uncalibrated: no production choice should be made from this table.

## Alternative methods, fair applicability, and what they buy

| Family | Possible benefit | Main trade-off / caveat | Tested here? |
|---|---|---|---|
| **Steger-style curvilinear extractor** | Subpixel centerlines/width; appropriate when thin curved dark tracks matter | Scale width, contrasts, crossings and neighboring lines change localization | Conceptual alternative only [Steger, TPAMI 1998](https://doi.org/10.1109/34.659930) |
| **Frangi/Hessian ridge** | Captures curved structures at multiple scales and can work without labeled target tracks | Scratches and other ridges also activate; needs line tracing, junction handling | Existing S39-style detector tested; **not an exact Frangi implementation** [Frangi, MICCAI 1998](https://doi.org/10.1007/BFb0056195) |
| **LSD and probabilistic Hough** | Fast, conventional, reproducible baselines for relatively straight dark segments | Splits curved centerlines; edge/line localization may differ | **Tested in toy QA**, not on native H50Z with expert labels [LSD, TPAMI 2010](https://doi.org/10.1109/TPAMI.2008.300), [Hough documentation](https://scikit-image.org/docs/stable/auto_examples/edges/plot_line_hough_transform.html) |
| **Trainable U-Net/nnU-Net segmentation** | Can learn contrast and bubble morphology if labels/domain coverage are strong | Requires annotated TRAIN, domain shift and considerable training; mask-to-instance/graph logic still required | **Not trained here** [nnU-Net, Nature Methods 2021](https://doi.org/10.1038/s41592-020-01008-z) |
| **Graph neural track learning** | Learns context and junction/association features beyond hand-crafted pair features | Requires particle-level GT and physics/domain-specific architecture; other detectors' 3D tracking gains don't transfer directly to H50Z photographs | **Not trained here** [Belle II, 2025](https://doi.org/10.1007/s41781-025-00135-6), [CERN GNN work, 2025](https://cds.cern.ch/record/2946080) |

## Release recommendation by use case

| Use case | Recommendation |
|---|---|
| Integrate as developer-facing **reversible 2D proposal layer** | **YES, research/assisted review**, if geometry is valid and model+CAL bound to exact detector |
| Compare Hessian/ANZA with an external neural detector | **YES, A/B in shadow mode**, retrain/recalibrate the S56 scorer and S85 gate per detector; independent untouched TEST required |
| Enable S80 / S85 additional-line gate on a new detector | **NO by default**; only after paired source-CAL and fresh TEST demonstrate more correct/full chains without more wrong links |
| Auto-merge actual H50Z physical tracks | **NO**, one known physical run and zero native expert GT |
| Use for arbitrary 3D tracks, loops, forked trajectories, multiple-camera geometry | **NO**, 2D forest assumptions and no physical model |

## Acceptance for another team (do not skip)

1. Freeze detector version/hash, pixel preprocessing, crop coordinate transform, max-candidate budget, pair-feature definition, source model hash and numeric CAL threshold.
2. Build **paired** comparisons on the *same independent scenes*: detector without graph, detector+port-greedy, detector+global and detector+extra-gate; retain existing S72 as historical control.
3. Refit scorer on new detector's TRAIN split, freeze threshold/strategy using CAL only. Never choose a winner by TEST or cite synthetic labels as H50Z truth.
4. Measure fragment coverage, endpoint error, correct/noncorrect links, complete chain recall, solver failures and compute paired **scene/run** confidence intervals. Add ablations for noise, scratches, straight vs curved tracks, crossings, source film / camera shifts.
5. Use **at least one additional independent physical run with expert-confirmed joins** (suggest candidates for expert confirmation; no demand for massive manual annotation) to authorize any physics-level claim. Count by independent run, not correlated crops.
6. Rollout in shadow mode: no writes to upstream labels, no forced join. If noncorrect predictions rise or domain drift is detected, fall back to original detector output / no joins.

**Bottom line:** extract the graph as a reusable *uncertainty-aware 2D hypothesis engine*, not an alleged universal track reconstructor. Current new code is an integration/benchmark capability; original S72 remains the only source-registered baseline and native H50Z performance remains unknown.
