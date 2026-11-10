# H50Z 005/006 — verified original BMP archive

This file records two original, **unaltered** 8-bit grayscale H50Z photographs.
The source images are not committed to regular Git because the verified ZIP
is about 320 MiB, outside normal source-history limits. On 2026-11-10 the
user stated that they have redistribution rights for these BMPs. That rights
statement is user-provided and has not been independently verified.

## Source files

| Source BMP | Size (bytes) | Image size | SHA-256 |
| --- | ---: | --- | --- |
| `H50Z_030_01_005.bmp` | 235883098 | 9020 × 26151 | `e78a90e72b80e4b166be2c0b2a53a9e70558c265c2a932fb1efc52783d5352e6` |
| `H50Z_030_01_006.bmp` | 234703910 | 9268 × 25324 | `d1251c30b460e31f9b8b3e8fb4b9849b920d5210dcf94eb133fd814b1d8cbea3` |

Source archive: `H50Z_030_01_006.7z` (private dataset). `labels.json`
exists in that original archive, but is **not** included in the ZIP described
here, because the requested package covers the two BMPs only.

Verified standalone ZIP assembled on 2026-11-10:
- Filename: `H50Z_005_006_original_BMP_verified.zip`
- Bytes: `335795937`
- ZIP SHA-256: `3825fca4d5a178064c7ec6b3399d92783b8b1fb7b68996832a93fcd560b60c7c`
- Members: the two BMPs, `MANIFEST.json`, `README.txt`
- ZIP CRC test: passed; SHA-256 for each BMP extracted from ZIP: passed.

**Archive upload status: NOT UPLOADED TO GITHUB.**
The GitHub connector currently available for this repository can create
text files and commits but has no binary Release-asset upload action.
GitHub regular Git blocks objects larger than 100 MiB and does not
recommend putting scientific datasets in the source history.

## Recommended GitHub Release upload

Upload to a **draft release** first; after verifying the asset and description,
the repository owner can publish the release:

```bash
gh auth login
gh release create h50z-005-006-data \
  --repo fims9000/anza_lira \
  --draft \
  --title "H50Z source frames 005 and 006" \
  --notes "Source photographs. Distribution rights to be confirmed before publishing."
gh release upload h50z-005-006-data \
  H50Z_005_006_original_BMP_verified.zip \
  --repo fims9000/anza_lira
```

The user stated redistribution permission; the owner may publish the verified
release after the upload has been checked. For private storage, use a private repository
or a controlled-access data store. Do not add the full ZIP or unpacked BMPs
to regular Git, and do not split and commit large binaries merely to evade the
per-file Git size limit.

The images are not ground-truth track segmentations. H50Z `start`/fiducial
point labels are not dense instance masks, and film-070 measurements have
not been verified as corresponding to these H50Z frames.
