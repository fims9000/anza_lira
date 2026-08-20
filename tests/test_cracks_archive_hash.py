from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_cracks_archives import EXPECTED_MD5, digest_file


@pytest.mark.parametrize("filename", ["images.zip", "Fault segmentations.zip"])
def test_local_official_cracks_archive_md5_when_present(filename: str) -> None:
    path = Path(filename)
    if not path.is_file():
        pytest.skip("Official CRACKS archive is intentionally not stored in Git")
    assert digest_file(path, "md5") == EXPECTED_MD5[filename]
