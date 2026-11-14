# Tver V31: crowd-validated long-line cohort in NEW numeric F3 section bands

**Fixed 14 November 2026, before inspecting any V31 heldout section annotation or model decision.** Exploratory real-pixel experiment, NOT proof of independent physical seismic-volume generalization.

## Rationale

V27–V30 evaluated only two already inspected expert-eligible TEST sections (108 and 124). The model and P0 relation stages refused all correct links, in large part because CAL consisted of only four positive and four *synthetic candidate-absent* negative intervention scenes from two sections. Their 0 observed errors cannot certify a 10% risk bound. We must not relax CAL simply to accept the V30 TEST.

Official CRACKS v2 `Fault segmentations.zip` also contains 12,563 nonexpert annotation images for 393 seismic source sections from 34 annotators (26 novices and eight practitioners), distinct from the 40 expert masks. The CRACKS manuscript and repository distinguish certain BLUE, uncertain GREEN, certain background ORANGE, and unknown WHITE: WHITE must NEVER become an assured geological negative. V31 uses only practitioner BLUE traces as proposed synthetic gap sources and verifies their local trace support against at least two **other** practitioners. Agreement within a three-pixel tolerance is used solely for **candidate cohort eligibility**, not claimed geological fault-instance identity. Disconnected trace IDs are NOT reliable natural fault instance labels.

## Cohort precommitment, numeric ID bands, not physical spatial coordinates

All sections with image and `practitioner*/section_XXX.png` filename are filtered **only by numeric section ID**:
- TRAIN: IDs **140–239** inclusive (100 nominal source IDs);
- numeric source separation: **240–259** excluded, never inspected for model outcomes;
- CAL: IDs **260–319** inclusive (60 nominal IDs);
- BUFFER: IDs **320–339** excluded, never opened;
- V31 TEST: IDs **340–399** inclusive (60 nominal IDs), **not inspected during preflight or model selection**.

These IDs differ from V27–V30 model TRAIN/CAL/TEST expert source IDs 10–124. Source counts, missing nominal IDs, and annotations are recorded. The two official CRACKS files do **NOT** contain physical section coordinates or proof of adjacency, so numeric banding is only an explicit proxy; do not call it verified physical spatial separation. Prior versions of this research used CRACKS images more broadly: the new TEST block is *V31-model-output-blind*, not guaranteed globally untouched or from another 3D field.

## Frozen practitioner agreement and controlled synthetic gap definition

1. Load masks for practitioner annotators only (`practitioner*`), **never expert**. Each mask MUST use official four-color palette; unknown colors mean STOP.
2. For each annotator's certain BLUE pixels, remove GREEN-adjacent pixels (3px) before skeletonization. Use existing `paths_from_mask` to retain only simple unbranched, border-safe paths of length ≥80 px. Require a path is supported, within Euclidean morphological 3 px tolerance, by BLUE pixels of **at least two distinct other practitioners on ≥60% of its own pixels**. This does not make the full curve independently expert-verified. Select no more than two non-overlapping source paths/section by highest peer-supported fraction, then path length, deterministic annotator/path order. Do not change thresholds after checking TRAIN/CAL outcomes; TEST annotations must not be accessed until every model and accept threshold is frozen.
3. Choose a controlled artificial gap from the central 12 consecutive points of an accepted practitioner trace, pad a tight rectangle by three pixels, and use original V27 intervention code. Reject truncated edges, ambiguous green overlap, or an endpoint which falls inside rectangle. The source/target inferred by SBPP must still be derived solely from the **predicted ANZA / U-Net probability**, not any label endpoint. Practitioner trace coordinates and identity may be used only to define intervention and evaluate candidate pairs.
4. For each valid LINK gap, generate one *synthetic* candidate-absent NO_LINK counterpart by removing destination neighborhood from the predicted segmentation, as V27. This is NOT a natural geologically confirmed negative.
5. Avoid counting numerous correlated gaps from one section as independent data: record metrics by source-section ID and use a section-level false-link event as the CAL risk unit. Do not claim 60 nominal section IDs are 60 independent geological volumes.

## Candidate diagnostic and feasibility criteria before new model training

Source-only preflight reads practitioner TRAIN/CAL mask contents to record per-section supported lines; it can list V31 TEST and BUFFER **filenames only**, without decoding image/mask contents. Require at least **15 usable TRAIN sections** and **29 usable CAL sections**; if fewer, STOP with exact counts and do not alter eligibility based on TEST. The CAL minimum of 29 permits a one-sided exact binomial 95%-upper bound below 10% when no errors occur, under *independent section* sampling; correlation within the same F3 volume makes even this assumption conditional and possibly optimistic.

## Candidate follow-up E2E, once source supports it

Use exact SHA-frozen successful V28 AZConv and U-Net segmentation weights; compare predicted SBPP source/candidate availability on new TRAIN/CAL. If sufficient positive and hard-negative TRAIN pairs for a new logistic and original 5-convolution P0 relation head, train those on TRAIN only; no oracle candidate injection. CAL selects link/no-link threshold from an explicitly frozen grid by maximizing accepted correct links **subject to one-sided section-level Clopper–Pearson upper bounds ≤10%** separately for wrong accepted positive and accepted synthetic NO_LINK; abstain if unsupported. Only then evaluate the entire **340–399 numeric TEST band** once, counting every source with absent usable practitioner-positive trace and every missing predicted port/candidate. Preserve per-section denominators, source eligibility and whole-scene false-link events. No ad hoc TEST score tuning, and no subsequent attribution of these numbers to historical P0 weights.

## Claim boundary

This protocol can improve statistical support and test learned relation selection on more real F3 intensity sections with cross-practitioner agreement, but it is STILL artificial missing-segmentation, nonexpert annotation and a single volume. It does not establish geological instance identity, natural no-link ground truth, validation on independent geological surveys, or ANZA/LIRA superiority over modern published models. If source eligibility is insufficient, that negative feasibility result is a valid research outcome.