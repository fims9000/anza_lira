"""V34 frozen V33 ANZA-only admission before relation CAL and unseen TEST."""
from __future__ import annotations

import json

import pytest

from scripts.tver_v34_anza_p0_relation_cal import (
    _state_report,FROZEN_ANZA_SHA,FROZEN_UNET_SHA,
    FROZEN_ANZA_TOPOLOGY,UNET_V33_POLICY,
)


def _receipt():
    return {
        "status":"V33_STOP_THEBE_NEW_FRONTEND_PREDICTED_SBPP_INSUFFICIENT",
        "SEG_CAL_section_indices":list(range(30)),
        "REL_CAL_section_indices_locked_unopened":list(range(40,100)),
        "heldout_THEBE_TEST7_NOT_OPENED":True,
        "frontends":{
            "anza_v1":{
                "SHA256":FROZEN_ANZA_SHA,
                "topology_threshold_frozen":FROZEN_ANZA_TOPOLOGY,
                "topology_status":"TRAIN_CAL_FEASIBLE",
            },
            "unet":{
                "SHA256":FROZEN_UNET_SHA,
                "topology_threshold_frozen":None,
                "topology_status":UNET_V33_POLICY,
            },
        }
    }


def test_admit_only_exact_original_new_thebe_v33_anza(tmp_path):
    p=tmp_path/"v33_THEBE_frontends_topology_TRAIN_CAL_frozen.json"
    p.write_text(json.dumps(_receipt()))
    loaded=_state_report(tmp_path)
    assert loaded["frontends"]["anza_v1"]["topology_threshold_frozen"]==.65
    assert loaded["frontends"]["unet"]["topology_threshold_frozen"] is None


@pytest.mark.parametrize("field,wrong",[
    (("frontends","anza_v1","SHA256"),"FAKE"),
    (("frontends","unet","SHA256"),"FAKE"),
    (("frontends","anza_v1","topology_threshold_frozen"),.6),
    (("frontends","unet","topology_threshold_frozen"),.6),
    (("heldout_THEBE_TEST7_NOT_OPENED",),False),
    (("SEG_CAL_section_indices",),[1,2,3]),
    (("REL_CAL_section_indices_locked_unopened",),[108,124]),
])
def test_stop_if_any_source_or_segmentation_provenance_is_changed(tmp_path,field,wrong):
    data=_receipt()
    node=data
    for k in field[:-1]:node=node[k]
    node[field[-1]]=wrong
    p=tmp_path/"v33_THEBE_frontends_topology_TRAIN_CAL_frozen.json"
    p.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match="provenance changed"):
        _state_report(tmp_path)
