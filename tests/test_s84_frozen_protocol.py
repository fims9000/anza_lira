import inspect
from pathlib import Path
import numpy as np
import pytest
from scripts.research.bubble_chamber import s84_calibrate_then_fresh_test_graph_context as run
from bubble_chamber.s84_graph_context_candidates import gate_additional_lines


def test_S84_new_seeds_and_locked_scene_budget():
    earlier={20268301,20268311,20268303,20268304,20268313,20268314}
    cal=set(run.CAL_SEEDS.values());test=set(sum((list(v) for v in run.TEST_SEEDS.values()),[]))
    assert len(cal)==2 and len(test)==4 and not (cal&test) and not (test&earlier)
    assert (run.CAL_SCENES,run.TEST_SCENES)==(24,24)
    assert run.CAL_CONDITIONS==('clean','blur_sigma_11')
    assert run.TEST_CONDITIONS==('clean','blur_sigma_11','blur_sigma_16')


def test_inference_boundary_does_not_accept_source_truth():
    signature=inspect.signature(run.infer_from_pixels)
    assert tuple(signature.parameters)==('gray','scorer','cutoff','graph_arm','thresholds')
    assert 'source_nodes' not in inspect.getsource(gate_additional_lines)
    with pytest.raises(ValueError):
        run.infer_from_pixels(np.zeros((192,192),float),lambda x:None,.8,'s39_original',{'q25':0.,'q40':0.})
    with pytest.raises(TypeError):
        run.infer_from_pixels(np.zeros((192,192),np.uint8),lambda x:None,.8,'s39_original',
                              {'q25':0.,'q40':0.},reference_tracks=[])


def test_frozen_protocol_and_ids_exist():
    assert run.PROTOCOL.is_file() and '20268403' in run.PROTOCOL.read_text()
    assert run.EXPERIMENT_ID=='ANZA-LIRA-BC-S84-20261026'
    assert run.SCHEMA.endswith('_v1')
