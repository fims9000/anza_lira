import csv
import json

from scripts.validate_anza_v2_study import _finite_artifact


def test_validator_rejects_nonfinite_json_and_csv(tmp_path) -> None:
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    table = tmp_path / "bad.csv"
    good.write_text(json.dumps({"value": 1.0}))
    bad.write_text('{"value": NaN}')
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["value"])
        writer.writeheader()
        writer.writerow({"value": "Inf"})
    assert _finite_artifact(good)
    assert not _finite_artifact(bad)
    assert not _finite_artifact(table)
