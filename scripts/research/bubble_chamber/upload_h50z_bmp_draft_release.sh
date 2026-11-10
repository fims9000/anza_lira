#!/usr/bin/env bash
# Stage verified raw H50Z BMP ZIP as a DRAFT GitHub Release asset.
# Never publish until the owner of the original film confirms redistribution rights.
set -euo pipefail

ARCHIVE="${1:?Usage: bash upload_h50z_bmp_draft_release.sh /path/to/H50Z_005_006_original_BMP_verified.zip [owner/repo]}"
REPO="${2:-fims9000/anza_lira}"
TAG="h50z-005-006-data"
EXPECTED_SHA="3825fca4d5a178064c7ec6b3399d92783b8b1fb7b68996832a93fcd560b60c7c"

test -f "$ARCHIVE" || { echo "Archive does not exist: $ARCHIVE" >&2; exit 1; }
command -v gh >/dev/null || { echo "Install GitHub CLI (gh) and login with: gh auth login" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum required" >&2; exit 1; }
printf '%s  %s\n' "$EXPECTED_SHA" "$ARCHIVE" | sha256sum -c -
gh auth status >/dev/null

if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    if ! gh release view "$TAG" --repo "$REPO" --json isDraft --jq '.isDraft' | grep -qx true; then
        echo "Refusing to change non-draft release; contact repository maintainer." >&2
        exit 1
    fi
    gh release upload "$TAG" "$ARCHIVE" --repo "$REPO"
else
    gh release create "$TAG" "$ARCHIVE" --repo "$REPO" --draft \
      --title "H50Z original 005 and 006 (private review)" \
      --notes "Two original film BMPs; source SHA-256 verified. Rights review required before publication."
fi
echo "Archive staged in a DRAFT release. Do not publish without explicit image rights."
