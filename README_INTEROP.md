# ANZA–LIRA 2D Interop / developer handoff

**Status: RESEARCH-ONLY | SOURCE-ONLY QA VERIFIED | NO REAL-PHYSICS AUTO JOIN**  
**Branch:** `research/bubble-chamber-graph-lira`.  
**Maintained source modules:** `bubble_chamber/interop_*.py`, `scripts/research/bubble_chamber/interop_*.py`.

## Use this module for

**Inspecting** possible joins between open 2D curved fragments found by your own detector, with explicit uncertainty and read-only outputs. It does not segment 3D particle tracks, assign particle identity, infer momentum, or substitute for real physicist labels. The old source baseline remains S72, not the learned S85 unvalidated gate.

**Not recommended by default:** automatically welding original tracks, transferring ANZA scores to another detector, applying S80/S85 added-line gates without detector-specific TRAIN/CAL, or claiming physical improvement from synthetic proxies.

## Quick start

```bash
python -m pip install -r requirements-interop.txt
python scripts/research/bubble_chamber/example_external_detector.py
python scripts/research/bubble_chamber/example_verified_external_detector.py
python -m pytest -q tests/test_interop_*.py
```

One example script runs against generated gray pixels only and uses an *untrained* toy score, not a real model. To integrate your own scoring model, read [DEVELOPER_INTEGRATION_2026_10_28.md](docs/bubble_chamber/DEVELOPER_INTEGRATION_2026_10_28.md). Use `process_verified` + a SHA-checked `CalibrationReceipt` to reject mismatched detectors/weights/domain. The external model loader and private input data remain the host application's responsibility.

Public API (all coordinates `(x,y)` in 192×192 source crop):

```python
from bubble_chamber.interop_api import process_image, ScoreBinding
from bubble_chamber.interop_coordinates import CropTransform, private_review_payload
from bubble_chamber.interop_verified_runtime import process_verified

crop = CropTransform(x0=160, y0=160, full_width=512, full_height=512)
# patch = crop.extract(your_512x512_gray_uint8)
# result = process_verified(patch, your_detector, receipt=receipt,
#                           model_file=private_model_path,
#                           source_cal_file=private_cal_path,
#                           score_pairs=your_loaded_scorer,
#                           expected_training_domain='your-frozen-TRAIN-domain')
# private = private_review_payload(result, crop, source_image_id='internal-image-key')
# PUBLIC: only result.as_public_dict(), NEVER private_review_payload(...)
```

`private_review_payload` contains pixel coordinates and therefore **MUST remain private**. The default `as_public_dict()` is aggregate-only and labels every proposed join `UNKNOWN`. The image crop is **not resized**; source coordinate offsets are explicit.

## What our comparisons actually show

- **Historic H50Z-background synthetic test (S84):** attempted extra fragments increased some synthetic-correct joins but also increased incorrect/unverified joins. **Not promoted.** Two photo backgrounds are from **one physical run**, no native physical GT.
- **Generated-only detector swap benchmark:** Hessian/ridge recovered 10/40 complete artificial chains in clean conditions, LSD 4/40, Hough 0/40 with the same *uncalibrated* geometry rule. For LSD global MILP got 4 paths vs greedy 3 but produced 47 vs 21 noncorrect joins (different absolute scores, **only toy**).
- **Generated-only detector-specific TRAIN/CAL/TEST benchmark:** every detector learned a distinct logistic pair scorer, but none passed the predeclared safe wrong-join constraint on CAL; all abstained on TEST. This rejects any claim of a *confirmed safe improvement*, not a claim that all graph methods are inherently incapable.

Read the protocol and exact results in `docs/bubble_chamber/S85_INTEROP_*.md` and corresponding JSON. The source of truth for limitations and integration is `DEVELOPER_INTEGRATION_2026_10_28.md`.

## Native integration acceptance

1. External detector `.detect(gray:uint8[192,192]) -> list[np.ndarray[N,2]]` returns 3–32 finite 2D open curves (`N>=8`); 0–2 curves cause **no joins** in legacy S56 graph, by design.
2. Run detector-specific **TRAIN** and **CAL** on independent source scenes, record hashes/model_version/crop transform, bind exact `detector_id` and `training_domain_id`.
3. Compare `none`, `node_mutual`, `port_greedy`, `global_milp` using the **same** source data and frozen scorer. Include corrected S69 synthetic-only evaluation and confidence intervals by whole independent scene, not by pixel/crop.
4. Validate on unrelated physical runs with expert-confirmed joins. Until then, **shadow mode only** and no physical auto joins.
5. Keep all private photos, measured source arrays, model checkpoints and native track coordinates **outside public Git**.

## Summary decision

| Deployment mode | Status |
|---|---|
| Plug in a different detector and inspect advisory 2D graph | **GO for research/shadow mode** |
| Swap detector but reuse ANZA/S72 score or CAL unchanged | **NO** |
| Add S80/S85 extra fragments on top of new detector | **NO by default; detector+augmenter retraining required** |
| Automatic H50Z track reconstruction / physics claims | **NO: independent expert physical GT missing** |

This is a reusable, conservative 2D hypothesis engine, not a completed physical reconstruction pipeline.
