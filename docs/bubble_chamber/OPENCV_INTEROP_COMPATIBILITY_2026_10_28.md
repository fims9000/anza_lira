# OpenCV LSD/Hough shape compatibility without changing frozen scientific evidence

**28 Oct 2026 — stable legacy v1, new detector IDs for v2.**

The historical `bubble_chamber/interop_detectors.py` is preserved **byte-identical**, SHA-256
`656bf257407b99254ffabcc4a54aa428e337d99105ac5a06effacc84a9e819f0`.
That exact file is referenced by historical generated-only S85 source-locked
benchmark receipts; changing it without replay would invalidate their hashes.

A portability issue was found in GitHub Actions: the legacy LSD/Hough code
assumed OpenCV produced `(N,1,4)` line arrays; some OpenCV 4.x wheels produce
`(N,4)`, leading to `IndexError`. The original CI also omitted torch, which
was imported by the frozen synthetic test pipeline. The legacy CI therefore
uses the **historical OpenCV 4.13.0.92** layout and explicitly installs CPU
torch. Historic published generated-only numbers and source SHA remain intact.

**The supported cross-version path for new integrators** is
`bubble_chamber/interop_cv_compat.py`: it normalizes the two OpenCV array
layouts before applying the original segment-processing method. Detector IDs
are intentionally NEW:

- `opencv-lsd-cvshape-v2`
- `opencv-hough-p-cvshape-v2`

Run the same *geometry data contract* through the public API, but do not use
old v1 source-trained scores, thresholds, calibration receipts or historical
numbers to claim v2 scientific benefit. The code exercises both array shapes
with explicit contract tests and a generated-only CLI.

```bash
python scripts/research/bubble_chamber/example_cv_compatible_detector.py
python -m pytest -q tests/test_interop_cvshape_v2.py
```

The new v2 modules do NOT use real H50Z photographs, operator film070
coordinates, physicist identity labels or trained model weights. They do not
measure physical accuracy. Before using v2 in any H50Z or seismic analysis,
obtain a new detector-specific TRAIN, CAL, untouched TEST and expert audit.
