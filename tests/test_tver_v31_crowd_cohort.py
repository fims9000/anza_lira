"""V31 source-only crowd BLUE line admission, frozen section IDs and TEST guard."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from scripts.tver_v31_crowd_cohort import (
    TRAIN,CAL,BUFFER,TEST,MIN_BLUE_PEERS,
    MIN_PATH_LENGTH,SUPPORT_RADIUS_PX,
    crowd_practitioner_gaps,practitioner_index,execute,
)

def _mask(offset=0,*,unknown=False):
    rgb=np.full((255,701,3),255,dtype=np.uint8)
    rgb[100+offset,105:225]=[31,119,180]
    if unknown:
        rgb[33,33]=[1,2,3]
    return rgb


def _png(arr):
    b=io.BytesIO()
    Image.fromarray(arr).save(b,format="PNG")
    return b.getvalue()


def test_frozen_id_bands_and_exact_cross_practitioner_contract():
    assert TRAIN==(140,239)
    assert CAL==(260,319)
    assert BUFFER==(320,339)
    assert TEST==(340,399)
    assert MIN_BLUE_PEERS==2
    assert MIN_PATH_LENGTH==80
    assert SUPPORT_RADIUS_PX==3.


def test_peer_supported_single_original_practitioner_path():
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w") as z:
        for person in range(1,5):
            z.writestr(
                f"Fault segmentations/practitioner{person}/section_140.png",
                _png(_mask(offset=0 if person<=3 else 1)))
    b.seek(0)
    with zipfile.ZipFile(b) as z:
        index=practitioner_index(z)
        gaps,receipt=crowd_practitioner_gaps(z,140,index)
    assert receipt["practitioners"]==4
    assert receipt["accepted_paths"]==1
    assert len(gaps)==1
    assert gaps[0]["source_annotation_kind"]=="individually_drawn_practitioner_BLUE"
    assert gaps[0]["at_least_two_other_practitioners_support_fraction"]>=.6
    assert len(gaps[0]["hidden_centerline"])==12
    assert all(140==g["section"] for g in gaps)


def test_one_unvalidated_long_path_does_not_enter_cohort():
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w") as z:
        for person in range(1,5):
            rgb=_mask() if person==1 else np.full((255,701,3),255,dtype=np.uint8)
            z.writestr(f"Fault segmentations/practitioner{person}/section_140.png",
                       _png(rgb))
    b.seek(0)
    with zipfile.ZipFile(b) as z:
        gaps,row=crowd_practitioner_gaps(z,140,practitioner_index(z))
    assert not gaps
    assert row["accepted_paths"]==0


def test_palette_violations_fail_closed():
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w") as z:
        for i in (1,2,3):
            z.writestr(
                f"Fault segmentations/practitioner{i}/section_140.png",
                _png(_mask(unknown=(i==2))))
    b.seek(0)
    with zipfile.ZipFile(b) as z:
        with pytest.raises(RuntimeError,match="palette"):
            crowd_practitioner_gaps(z,140,practitioner_index(z))


def test_source_preflight_does_not_decode_corrupt_TEST_or_buffer(tmp_path):
    images=tmp_path/"images.zip"
    labels=tmp_path/"Fault segmentations.zip"
    with zipfile.ZipFile(images,"w") as z:
        for sid in (140,260,340,330):
            z.writestr(f"images/section_{sid:03d}.png",b"IMAGE NAME ONLY")
    with zipfile.ZipFile(labels,"w") as z:
        for sid in (140,260):
            for practitioner in (1,2,3):
                z.writestr(
                    f"Fault segmentations/practitioner{practitioner}/section_{sid:03d}.png",
                    _png(_mask()))
        for sid in (330,340):
            for practitioner in (1,2,3):
                z.writestr(
                    f"Fault segmentations/practitioner{practitioner}/section_{sid:03d}.png",
                    b"THIS IS CORRUPT PNG AND MUST NOT BE DECODED")
        z.writestr("Fault segmentations/expert/section_140.png",
                   b"EXPERT ALWAYS FORBIDDEN")
    result=execute(images,labels,tmp_path/"cohort.json")
    assert result["status"]=="V31_STOP_INSUFFICIENT_CROWD_AGREEMENT_TRAIN_OR_CAL"
    assert result["no_TEST_or_BUFFER_decoding"] is True
    assert result["real_expert_annotations_not_decoded"] is True
    assert result["split_metrics"]["TRAIN"]["n_eligible_distinct_source_sections"]==1
    assert result["split_metrics"]["CAL"]["n_eligible_distinct_source_sections"]==1
    assert json.loads((tmp_path/"cohort.json").read_text())["status"]==result["status"]
