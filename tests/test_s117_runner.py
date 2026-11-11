"""Smoke test for the S117 real-data runner, without private H50Z source."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from scripts.research.bubble_chamber.run_s117_long_gap import run


def test_cli_runner_reproducible_provenance_and_no_overwrite(tmp_path):
    gray = np.full((150, 500), 190, np.uint8)
    image = tmp_path / 'src.png'
    cv2.imwrite(str(image), gray)
    curves = {'curves': [
        {'candidate_id':'a', 'observed_pixel_path_xy':[[x,50] for x in range(20,81)]},
        {'candidate_id':'b', 'observed_pixel_path_xy':[[x,50] for x in range(330,411)]}]}
    source = tmp_path / 'curves.json'
    source.write_text(json.dumps(curves),encoding='utf8')
    result = run(source,image,tmp_path/'out')
    assert result['provenance']['manual_GT_used'] is False
    assert (tmp_path/'out'/'S117_LINKS_GROUPS_NOT_GT.json').exists()
    assert (tmp_path/'out'/'S117_OBSERVED_VS_INFERRED_REVIEW.png').exists()
    assert result['report']['added_observed_pixel_count'] == 0
    with pytest.raises(FileExistsError):
        run(source,image,tmp_path/'out')


def test_runner_rejects_scaled_directional_response_png(tmp_path):
    from scripts.research.bubble_chamber.run_s117_long_gap import run
    p = tmp_path/'r.png'
    cv2.imwrite(str(p), np.zeros((100,100),np.uint8))
    src = tmp_path/'curves.json'
    src.write_text('{"curves": []}')
    with pytest.raises(ValueError,match='float32'):
        run(src,p,tmp_path/'out',response=p)
