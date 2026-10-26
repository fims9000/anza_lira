from inspect import signature
from scripts.research.bubble_chamber.s81_native_quality_gate_readonly import run


def test_native_qa_does_not_accept_physics_gt():
    parameters=signature(run).parameters
    assert set(parameters)=={'visualmetro','tiles','calibration'}
    assert 'true_particle_id' not in parameters
