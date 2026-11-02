from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/ccta_graph_lira_safe_repair"))
from ct28_selected_statistics import summarize, as_markdown


def sample_rows():
    rows = []
    for i, patient in enumerate((957, 961, 965, 966, 974)):
        for model in ("geometry_only", "ct_conditioned"):
            for scene, relation in (("a", "PAIR"), ("b", "NONE")):
                accepted = int(scene == "a" and (model == "ct_conditioned" or i == 0))
                rows.append(dict(model=model, scan_id=str(patient), scene_id=f"{patient}:{scene}",
                                 relation_class=relation, accepted=str(accepted), exact=str(accepted),
                                 false_scene="0", pre_consistency_exact="1",
                                 pre_consistency_false="0", selected_relation_before_consistency="PAIR",
                                 tau_relation="PAIR"))
    return rows


def test_summarize_paired_patient_bootstrap_and_report():
    report = summarize(sample_rows(), draws=101, seed=20261102)
    assert report["n_independent_patients"] == 5
    assert report["models"]["ct_conditioned"]["accepted_exact"] == 5
    assert report["models"]["geometry_only"]["accepted_exact"] == 1
    assert report["models"]["ct_conditioned"]["accepted_false"] == 0
    assert abs(report["ct_minus_geometry"]["repair_recall"]["estimate"] - .8) < 1e-12
    assert "selection bias" in as_markdown(report)


def test_duplicate_scenes_and_cross_model_mismatch_rejected():
    x = sample_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(x + [x[0]], draws=101)
    x = sample_rows()
    x[1]["relation_class"] = "PAIR"
    with pytest.raises(ValueError, match="Scene mismatch"):
        summarize(x, draws=101)


def test_heldout_patient_is_never_in_validation_report():
    x = sample_rows()
    x[0]["scan_id"] = "954"
    with pytest.raises(ValueError, match="Held-out"):
        summarize(x, draws=101)


def test_selected_validation_artifact_is_recomputed_not_historical():
    from ct28_selected_statistics import read_predictions
    path = Path(__file__).resolve().parents[1] / (
        "artifacts/varvara/ct28_graph_lira/selected_validation/graph_lira_predictions_val.csv"
    )
    report = summarize(read_predictions(path), draws=101, seed=20261102)
    geometry = report["models"]["geometry_only"]
    ct = report["models"]["ct_conditioned"]
    assert report["patient_ids"] == [957, 961, 965, 966, 974]
    assert (geometry["scenes"], geometry["repair_needed"]) == (368, 206)
    assert (geometry["pre_exact"], geometry["pre_false"], geometry["accepted_exact"], geometry["accepted_false"]) == (157, 20, 16, 0)
    assert (ct["pre_exact"], ct["pre_false"], ct["accepted_exact"], ct["accepted_false"]) == (155, 11, 52, 0)
    assert [report["per_patient"][str(p)]["ct_conditioned"]["accepted_exact"] for p in report["patient_ids"]] == [11, 15, 7, 13, 6]
    assert [ct["by_class"][t]["accepted_exact"] for t in ("PAIR", "JUNCTION", "BOTH")] == [48, 2, 2]
