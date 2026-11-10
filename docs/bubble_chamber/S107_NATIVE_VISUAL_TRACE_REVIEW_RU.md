# S107: source-locked visual trajectory review for original H50Z frames

S107 supports expert review of actual 8-bit H50Z 005/006 photo pixels,
without resizing or adding fabricated track centerlines. It is preparation
for measurable track-level research, not a new accuracy claim.

## Supported workflow

1. Verify SHA-256 of the two original BMPs and original labels.json.
2. Extract 12 pixel-identical PNG crops:
   - 8 deterministic 2-by-2 blind-stratum samples chosen independently
     from S105 or any source-mask output;
   - 2 source-bound operator START areas (not GT);
   - 2 fiducial reference areas (not proof of background).
3. Open the local HTML editor, inspect original pixels at 100% zoom,
   and manually trace visible track, artifact or unknown line fragments.
4. Preserve an identical visual instance ID for separate visible fragments
   only when the reviewer judges they are continuations. Do not draw a solid
   line across an unobserved gap; limit point spacing to 32 original pixels.
5. Export human strokes into a source-coordinate and manifest-hash-bound JSON.
   S105 comparison is OFF initially; if shown, preview exposure is recorded.
6. Run the evaluator to obtain partial centerline coverage of explicitly
   marked clear-positive track strokes and overlap on explicitly marked
   artifacts, with per-ID missed-run length in original pixels.

Unmarked areas are UNKNOWN, not negative examples. Uncertain / conflicting
strokes remain UNKNOWN. Visual IDs cannot prove physical particle identity.

## Run from repository root

    python -m pip install -r docs/bubble_chamber/requirements.txt
    python scripts/research/bubble_chamber/physicists_s107_review.py build \
      --source-dir /PRIVATE/H50Z \
      --out-dir /PRIVATE/H50Z_S107_review

Optional stored S105 source-locked overlays can be attached with
--s105-root /PRIVATE/H50Z_original_runs. The blind sample is unchanged.
Open /PRIVATE/H50Z_S107_review/index.html in your browser.

To evaluate real expert-reviewed JSON:

    python scripts/research/bubble_chamber/physicists_s107_review.py evaluate \
      --pack-dir /PRIVATE/H50Z_S107_review \
      --review-json /PRIVATE/H50Z_S107_review_YYYY-MM-DD.json \
      --out-dir /PRIVATE/H50Z_S107_evaluation

Output partial labels:
- 0: UNKNOWN (unmarked);
- 1: manually observed certain track centerline neighborhood;
- 2: manually observed certain artifact stroke neighborhood;
- 3: UNKNOWN, conflicting or uncertain.

## Limitations

H50Z 005 and 006 come from ONE physical run; 12 tiles are NOT independent
test events and cannot establish cross-run transfer. Prior S105 numbers are
not real-photo precision/recall. Missing human lines produce null metrics,
not perfect accuracy. The UI does not itself identify particles.

Raw H50Z BMPs are not stored in ordinary Git; after confirmed distribution
rights they belong as verified external scientific assets (GitHub Release,
Git LFS, or restricted data storage as appropriate).
