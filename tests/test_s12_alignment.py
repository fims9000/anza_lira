import pytest
from scripts.research.bubble_chamber.s12_evaluate_orientation_alignment import evaluate_model_alignment
from scripts.research.bubble_chamber import s7_original_retrain as s7
import torch


def test_non_az_model_rejected():
    cfg=s7.RunConfig(size=32,train_limit=2)
    x=s7.TrackDataset(s7.procedural_examples(2,32,seed=cfg.seed,split='train'))
    with pytest.raises(ValueError,match='Missing requested'):
        evaluate_model_alignment(torch.nn.Conv2d(1,1,3,padding=1),x,cfg)
