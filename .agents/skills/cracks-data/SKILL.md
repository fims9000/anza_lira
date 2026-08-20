---
name: cracks-data
description: Verify, extract, map, split, normalize, load, and audit CRACKS without spatial leakage or guessed label semantics.
---

# CRACKS Data

Use `docs/research/anza_v2_master_spec.md` sections 47-68. Accept only the two
official archives with the frozen MD5 values. Inspect the actual archive and
expert palette before implementing targets. Official semantic colors are
orange=no-fault certain, green=fault uncertain, and blue=fault certain; any
other observed color remains ignored unless primary-source evidence proves its
meaning.

The released 40 expert masks are an available expert subset, not a blocker.
Setting A trains only from novice/practitioner annotations and evaluates against
all available expert masks after architecture and parameters are frozen. Hold
out at least one practitioner and two novices by a deterministic annotator-ID
hash and never choose them by model performance. Setting B uses frozen 5-fold
expert splits of 28 train, 4 validation, and 8 test sections. Setting C removes
the held-out image, all of its annotations, and preferably its +/-2 neighbors
from training. Keep A/B/C metrics separate and never call Setting A unseen-image
generalization.
