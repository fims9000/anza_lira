"""Scientific safeguards: train-only, monotone, patient-crossfit calibration."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_neural_calibration import (
    fit_train_only_platt,patient_crossfit_platt,load_calibrator,save_calibrator,
)


def evidence():
    rng=np.random.default_rng(17)
    patients=np.repeat([953,955,956,959],25)
    y=np.tile(np.asarray([0,0,0,1,0],dtype=float),20)
    # Underconfident but useful ranking, e.g. logits below 0.85.
    raw=np.clip(.3+.17*y+rng.normal(0,.06,len(y)),.01,.8)
    return raw,y,patients


def test_monotone_calibration_changes_scale_not_auroc_and_reopens(tmp_path):
    x,y,p=evidence()
    fitted,provenance=fit_train_only_platt(x,y,p)
    calibrated=fitted.predict(x)
    assert np.std(calibrated)>0
    assert abs(roc_auc_score(y,x)-roc_auc_score(y,calibrated))<1e-10
    assert provenance["held_out_CT28_test_accessed"] is False
    assert provenance["development_validation_fit_accessed"] is False
    assert provenance["n_train_patients"]==4
    assert fitted.slope>.001
    file=tmp_path/"calibrator.json"
    checksum=save_calibrator(file,provenance)
    assert len(checksum)==64
    assert np.allclose(load_calibrator(file).predict(x),calibrated)
    with pytest.raises(FileExistsError):
        save_calibrator(file,provenance)


def test_patient_crossfit_excludes_own_labels_and_is_finite():
    x,y,p=evidence()
    oof,receipts=patient_crossfit_platt(x,y,p)
    assert np.isfinite(oof).all() and np.all((oof>0)&(oof<1))
    assert len(receipts)==4
    for receipt in receipts:
        assert receipt["held_patient_id"] not in receipt["calibration_train_ids"]


@pytest.mark.parametrize("bad_patient",[957,966,980,984])
def test_calibration_fails_closed_for_dev_val_and_heldout(bad_patient):
    x,y,p=evidence()
    p=p.copy()
    p[0]=bad_patient
    with pytest.raises(ValueError,match="train patients|val/test"):
        fit_train_only_platt(x,y,p)
    with pytest.raises(ValueError,match="train patients|val/test"):
        patient_crossfit_platt(x,y,p)


def test_negative_model_cannot_flip_bad_ranking_by_calibration():
    x,y,p=evidence()
    fitted,_=fit_train_only_platt(1-x,y,p)
    assert fitted.slope>0
    assert abs(roc_auc_score(y,1-x)-
               roc_auc_score(y,fitted.predict(1-x)))<1e-10


def test_unsealed_or_relabelled_calibration_json_rejected(tmp_path):
    x,y,p=evidence()
    _,receipt=fit_train_only_platt(x,y,p)
    for key,value in (
        ("held_out_CT28_test_accessed",True),
        ("development_validation_fit_accessed",True),
        ("train_patient_ids",[957]),
    ):
        broken={**receipt,key:value}
        path=tmp_path/"fake.json"
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            load_calibrator(path)
    with pytest.raises(ValueError,match="finite"):
        fit_train_only_platt(np.full(len(x),np.nan),y,p)


def test_independent_phantom_calibration_split_keeps_val_labels_out_of_fit():
    from benchmark_structural_tubes import fit,make_phantoms
    train=make_phantoms(16,20261111,"JUNCTION",separate_endpoints=True)
    calibration=make_phantoms(16,20262027,"JUNCTION",separate_endpoints=True)
    val=make_phantoms(16,20261911,"JUNCTION",separate_endpoints=True)
    raw=fit(
        "conv3d_sep_structure","JUNCTION",train,val,epochs=1,batch_size=8,
        seed=17,device="cpu",calibration=calibration,
    )
    audit=raw["calibration"]
    assert audit["status"]=="independent_synthetic_calibration_subset_not_medical"
    assert audit["calibration_seed"]==20262027
    assert audit["n_calibration_samples"]==16
    assert audit["calibration_uses_validation_labels"] is False
    assert raw["val_AUROC"]==pytest.approx(audit["val_calibrated_AUROC"])
    assert audit["val_calibrated_tau085_true_accepted"]<=8
    assert audit["val_calibrated_tau085_false_accepted"]<=8
    assert audit["val_calibrated_tau085_abstained"]<=16
