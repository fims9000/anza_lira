# CRACKS v2: real-domain ANZA-LIRA evaluation

Official dataset: https://zenodo.org/records/13926822 (DOI 10.5281/zenodo.13926822). The two ZIP archives together are about 114.4 MB; they belong under `data/cracks/raw/`, never in Git history. Pin and verify their published MD5 checksums. Verify image and expert mask section IDs before constructing interventions.

Protocol: create artificial gaps inside confidently annotated, branch-free expert lines while preserving the original seismic image. Keep the full expert mask and hidden path out of *all* inference inputs. Use spatially blocked splits and distinct training, calibration, and test data. Report candidate availability, accepted true links, false links on truly negative controls, successful recovered paths, and model provenance. Do not use a held-out test to select thresholds or architectures.

The historical ANZA, SBPP V3-B, and LIRA checkpoints are not part of this repository checkout. A newly trained CRACKS pair model is a *different* model; the 6-channel synthetic model is not compatible with the real-domain 8-channel pair representation. If a checkpoint or frozen calibration is missing, issue a STOP receipt and do not publish purported end-to-end metrics. Explicitly distinguish any oracle-mask diagnostic from a fully predicted end-to-end evaluation.

The dataset's expert masks are semantic color annotations, not instance identity labels for naturally disconnected faults. Only locally unambiguous, deliberately masked single lines can have known positive correspondence. Distinct disconnected components must not automatically be treated as negative fault identities.

## Executed GitHub Actions evidence (11 November 2026)

- [Run 43094291001](https://github.com/fims9000/anza_lira/actions/runs/43094291001): `success`, provenance artifact `cracks-real-gap-audit` (ID 13889234117).
- Official `images.zip`: **94,191,832 bytes**, published MD5 verified.
- Official `Fault segmentations.zip`: **20,227,782 bytes**, published MD5 verified.
- The expert folder contains 40 section IDs matching the real images; remaining crowd annotations are not treated as expert labels.
- Expert-mask topology: 12,282 skeleton pixels, 167 connected traces with length >=30 pixels, 156 with length >=50 pixels.
- Original gap pilot produced zero cases because the selection of anchor endpoints accidentally placed the anchors inside the padded occlusion region. This was corrected by finding the first surviving original-line points *outside* the occlusion rectangle.
- Exploratory masked-gap audit V2 produced **49 suitable positive examples**: development 23 (23 sections), calibration 16 (10 sections), test 10 (7 sections). Section split changed after initial inspection from 20-section hashed blocks to 5-section deterministic blocks. Hence it is **post-hoc exploratory only** and MUST NOT be advertised as an unseen confirmatory model test.
- Conditional straight-line interpolation using **oracle endpoints** obtained from the expert mask meets the stated 2-pixel tolerance on 49/49 cases. This number is only a geometry sanity check and says nothing about predicted endpoint detection, ANZA, SBPP, CNN-LIRA or overall end-to-end accuracy. No artificial false-positive negatives were fabricated from disconnected semantic masks.
- Three pure-program unit tests passed.
- Model availability receipt: **no frozen ANZA checkpoint** and **no original trained LIRA checkpoint** in this GitHub checkout. The historical ANZA → SBPP V3-B → LIRA → maximin cascade **has not been executed**.

## Interpretation and next admissible experiment

Do not report `49/49` as model success. The current strong limitation is the missing verified trained-model artifacts and compatible frozen decision thresholds, not the size of the CRACKS dataset. Once available, independently evaluate on untouched, pre-declared spatial-block groups without exposing the complete expert centerline to any model's inputs. The F3 sections share a 3D seismic volume; blocked sections are not a domain-shift/external-field test. For full end-to-end evaluation candidate extraction must derive from predicted ANZA probability masks, not from expert endpoints. Negative/ambiguous pairs require defensible LINK/REJECT/AMBIGUOUS identity ground truth.
