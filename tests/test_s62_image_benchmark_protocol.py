"""S62 frozen protocol cannot silently switch to oracle geometry for detector input."""
import inspect
from bubble_chamber.s62_image_candidate_diagnostics import image_only_predictions
from scripts.research.bubble_chamber.s62_benchmark_image_only_candidates import (
    SEEDS,N_SCENES_PER_SEED,ARMS)

def test_detector_interface_is_image_only_and_freeze_is_fresh():
    assert list(inspect.signature(image_only_predictions).parameters)==['gray']
    assert SEEDS=={'005':(20266203,20266204),'006':(20266213,20266214)}
    assert N_SCENES_PER_SEED==24
    assert len(ARMS)==4
