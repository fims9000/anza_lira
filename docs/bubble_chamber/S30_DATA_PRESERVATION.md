# S30 — provenance-safe dataset preservation (2026-10-25)

**Status:** source and experiment archives duplicated to the user's private ChatGPT Library. Original source files were kept unchanged. The GitHub repository `fims9000/anza_lira` is **public**, so do NOT commit the camera BMP, raw 7z, `labels.json`, visualMetro numerical measurements, or review HTML containing original image pixels.

## Exactly preserved source archives

| Archive | Contents | Integrity |
|---|---|---|
| `H50Z_030_01_006.7z` | Two original H50Z BMP files and one untouched `labels.json` | SHA-256 in `s30_dataset_integrity.json` |
| `1mH2Data.7z` | Original visualMetro measured coordinates and other source files, 5,627 entries | SHA-256 in `s30_dataset_integrity.json` |
| `070_008.7z` | Supplemental operator-measured frame, 223 entries | SHA-256 in `s30_dataset_integrity.json` |

`labels.json` has 105 photo metadata entries with 570 `start` and 837 fiducial marks. Only two original H50Z BMPs are present in this archive, with 9 `start` and 15 fiducial marks in total. The first archive and labels were preserved both together and as an independently accessible exact JSON copy in the private library. Archive byte checksums were recorded *before* backup. Every file above was copied to a dedicated private Library archive; the repo carries only hashes and schema.

## Other persisted research material

The private archival folder contains original and derived research releases S17, S19, S20, S21–S29, the original S29 operator review HTML, S29 provisional 2D quadratic and width estimates, an S30 source bundle and the S30 source patch, plus a detailed SHA-256 index. These files are intentionally distinct from *new* expert-reviewed track centerline/segment relation labels.

**Do not mislabel:**

- visualMetro clicked trajectories ≠ H50Z pixel-level track contours unless a verified image mapping is available;
- operator `start` and fiducial points ≠ 0/1/255 foreground/background masks;
- S29 photo-tracing proposals / S30 decisions ≠ expert-confirmed tracks or joins;
- data from one physical run ≠ independent medium/run test.

No additional expert-confirmed track pixels or pair relations were present in the materials inventoried for this archive. Those labels cannot be fabricated to produce a positive result.

## Restore and verify

Download the three source archives from the private Library and place them in one directory. Then run:

```bash
python scripts/research/bubble_chamber/s30_verify_dataset_archive.py \
  --manifest docs/bubble_chamber/s30_dataset_integrity.json \
  --source-dir /path/to/your/private/raw
```

The tool checks all three SHA-256 and file sizes and exits with a failure if any source was lost or modified. The private archive also has `metadata/s30_private_dataset_index.json`, `metadata/RAW_SHA256SUMS.txt`, and a Russian README with instructions. The public repository receives no raw experimental data.

**Future scientific gate:** the reviewer must manually export a genuine certified JSON of track positions, artifacts and same/different fragment relations from the private S29 interface. Until then, S30 abstention on competing line hypotheses is a tested engineering gate but not a validated physics detector.