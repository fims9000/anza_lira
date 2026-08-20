---
name: scientific-figures
description: Generate article-ready GeoCrack figures from frozen outputs without retraining.
---

# Scientific Figures

Use master-spec sections 28–29. Generate neutral-background SVG/PDF and 300-dpi
PNG from stored predictions/metrics/traces. Keep legends outside image content.
Choose median, best, and worst delta-Dice examples automatically; the main figure
uses median. Re-render figures independently from training.

For CRACKS, preserve the native 255x701 aspect ratio and select the primary real
example as the deterministic median section under the declared model/metric.
Best/worst examples may appear only in explicitly labeled diagnostic material.
Export white-background 300-dpi PNG plus SVG and PDF, and verify bounds after
rendering so labels, legends, and axes are not clipped.
