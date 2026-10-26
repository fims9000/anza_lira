# S80 — frozen multiscale pixel-instance proposals (2026-10-26)

**Experiment ID:** `ANZA-LIRA-BC-S80-20261026`  
**Status:** CAL rejected all new arms. Original S72 remains default.  
**Scope:** H50Z photographic backgrounds + artificially injected curves from visualMetro film 070, corrected S69 orientation-independent audit. **Not a native physical particle validation.**

## Research objective and isolation

Previous S76–S79 tests established that a major limitation is failure to extract isolated line instances from pixels; naive stitching can reduce correct graph joins. S80, preregistered in `S80_FROZEN_MULTISCALE_PIXEL_INSTANCES_2026_10_26.md` before CAL or TEST, adds **optional, distinct ridge curves** from sigma 2.2 Hessian response with a four-line cap or from an overlapping sigma 2.2/sigma 3.5 support consensus with a six-line cap, without changing or discarding existing image lines. These are alternative *detector* proposals, **not new physics tracks**. The S72 pixel/graph method, S56 geometry-trained pair classifier and S69 evaluator are kept byte-identical.

Whole-frame source geometry split 79 TRAIN / 30 CAL / 36 TEST from film 070. H50Z source photos 005/006 represent **one** physical run and have zero native expert particle-track labels. New S80 CAL source-background seeds 20268001/20268011 (16 scenes each). Fresh opposite-photo TEST seeds 20268003/20268004/20268013/20268014 (24 scenes each), 96 new synthetic scenes per condition. No S76–S79 TEST used for parameter selection. CAL criterion required increased blurred correct joins and shape recall without any additional incorrect joins or loss of clean performance. Both transfer directions selected baseline **before** TEST.

## Fresh TEST, 96 artificial scenes per condition

Numbers below count **synthetic** correct links, other proposed links (unmatched or incorrect relative to artificial ground truth), full artificial three-fragment chains and shape-matched artificial fragments. Do NOT quote as precision on native H50Z physics.

| Input condition | Image-only method | Correct synthetic joins | Other proposed joins | Full artificial 3-piece chains | Matched true artificial fragments by shape | Added image-derived line hypotheses |
|---|---|---:|---:|---:|---:|---:|
| No blur | Frozen S72 | 51 | 41 | 16 | 261 | 0 |
| No blur | S80 medium scale +<=4 | 53 | 55 | 17 | 295 | 125 |
| No blur | S80 two-scale consensus +<=6 | 53 | 46 | 17 | 268 | 58 |
| Blur sigma=1.1 | Frozen S72 | 20 | 21 | 5 | 140 | 0 |
| Blur sigma=1.1 | S80 medium scale +<=4 | 24 | 28 | 6 | 180 | 101 |
| Blur sigma=1.1 | S80 two-scale consensus +<=6 | 24 | 20 | 6 | 150 | 39 |
| Blur sigma=1.6 | Frozen S72 | 15 | 16 | 4 | 131 | 0 |
| Blur sigma=1.6 | S80 medium scale +<=4 | 18 | 20 | 5 | 169 | 100 |
| Blur sigma=1.6 | S80 two-scale consensus +<=6 | 17 | 16 | 5 | 138 | 38 |

### CAL verdict and interpretation

CAL source 005 chose `baseline`: medium scale added **2** synthetic-correct joins on blur but **4** additional noncorrect proposals; two-scale consensus added **1** synthetic-correct join (less than the preregistered +2) and added **1** extra noncorrect clean proposal. CAL source 006 also chose `baseline` because neither new arm improved blurred correct joins and both added at least one noncorrect clean proposal. Accordingly the TEST cannot overrule the frozen decision, even though the two-scale consensus did better under blur.

At blur sigma 1.1, the consensus arm yielded **24 vs 20** correct artificial joins, **20 vs 21** noncorrect and **6 vs 5** complete artificial chains, using 39 extra hypotheses over all 96 scenes. At stronger blur sigma 1.6, it yielded **17 vs 15** correct, 16 vs 16 noncorrect and 5 vs 4 chains. This is an *exploratory held-out observation*, **not** a passing preregistered CAL safety gate, a physical accuracy estimate, or approval to deploy. On clean images the same consensus arm adds **5 noncorrect joins**, which is a practical safety problem. The medium-only arm produces more matches but consistently more false/unmatched graph choices.

Replaying all 96 fresh TEST scenes with fixed seeds produced byte-identical JSON: SHA-256 `37e6f0711bab4d14924a3b2e79779bc08ee0613913a53bdfb49000ae34a9abdc`. Scene-paired 1200-bootstrap conditional intervals for each seed and transfer direction are embedded in the public aggregate JSON. They do not account for variation across independent physical runs (only one is available).

## Read-only audit on 39 genuine H50Z photo tiles (no injected lines)

Exactly one pre-fixed central 192x192 region per each 512x512 private tile. The machine does not receive a true native track/endpoint label.

| Method | Image-only line instances | Graph advisory links, all UNKNOWN | Newly proposed scale-specific instances |
|---|---:|---:|---:|
| Frozen S72 | 83 | 3 | 0 |
| S80 medium scale | 114 | 6 | 31 |
| S80 two-scale consensus | 95 | 5 | 12 |


On the real photographs, adding proposals cannot be interpreted as recovering **physical particles**: every candidate remains a reversible UNKNOWN line hypothesis. The private overlay gallery is NOT published to GitHub.

## Verification and reproduction

- Corrected S69 scorer and S76 separation of shape/endpoint remain unchanged; no manual labels used and no native reference identities fabricated.
- Python research test subset `pytest -q tests/test_s[0-9]*.py` passed **578 tests; 9 skipped**. A broader `pytest tests` cannot collect eight unrelated modules because this *research-only archive* lacks their non-bubble-chamber package dependencies; therefore **full repository test coverage is not claimed**.
- New frozen S80 CAL SHA is `a85134ddc709da48b7851ea6998f75c789a35010a0faff50ee00ee8d727ded28`.
- No PDF, photographs, source point coordinates, AI model weights, per-scene pixels or raw imaging archives enter public source.

```bash
python -m pytest -q tests/test_s80_*.py
python scripts/research/bubble_chamber/s80_calibrate_then_test_multiscale.py cal \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/tiles \
  --out /PRIVATE/S80_FROZEN_CAL_PRIVATE.json
python scripts/research/bubble_chamber/s80_calibrate_then_test_multiscale.py test \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/tiles \
  --calibration /PRIVATE/S80_FROZEN_CAL_PRIVATE.json \
  --out /PRIVATE/S80_FRESH_TEST_PRIVATE.json
python scripts/research/bubble_chamber/s80_native_unlabelled_audit.py \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/tiles \
  --out /PRIVATE/S80_NATIVE_UNKNOWN_PRIVATE.json
```

## Outcome

**NO-PROMOTION / new image instance alternatives remain for research only.** A future experiment may test image-quality-conditioned use of scale-specific candidates on NEW CAL+TEST seeds; it must not tune on the held-out S80 test. Stronger proof of real physics capability ultimately requires independent camera runs and physician/physicist-reviewed reference tracks. Without these, native particle identities and automatic joins remain UNKNOWN.
