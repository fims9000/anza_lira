"""Release integrity contracts; no real data or weights required for these tests."""
from pathlib import Path
from hashlib import sha256
import pytest
from scripts.research.bubble_chamber.s9_replay_release import validate_checksums

def make_fixture(root:Path,n:int):
    entries=[]
    for i in range(n):
        path=root/f"file_{i:03d}.txt"
        value=f"fixture {i}".encode()
        path.write_bytes(value)
        entries.append(f"{sha256(value).hexdigest()}  {path.name}")
    (root/"SHA256SUMS.txt").write_text("\n".join(entries)+"\n")

def test_complete_release_sha_inventory(tmp_path):
    make_fixture(tmp_path,45)
    assert validate_checksums(tmp_path)==45

def test_tampering_fails_release_validation(tmp_path):
    make_fixture(tmp_path,45)
    (tmp_path/"file_003.txt").write_text("changed")
    with pytest.raises(ValueError,match="Checksum mismatch"):
        validate_checksums(tmp_path)

def test_incomplete_release_is_rejected(tmp_path):
    make_fixture(tmp_path,4)
    with pytest.raises(ValueError,match="Incomplete"):
        validate_checksums(tmp_path)
