from pathlib import Path
import torch
import json
import numpy as np
import pytest
from scripts.research.bubble_chamber.s12_evaluate_quadratics import evaluate
from scripts.research.bubble_chamber import s7_original_retrain as s7


def test_missing_metrics_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate(tmp_path,tmp_path,tmp_path/'out',model_names=['az_base'])


def test_curves_do_not_leak_test_labels_into_inference(monkeypatch,tmp_path):
    # Evaluation of paired synthetic masks must never feed test truth to a model.
    from bubble_chamber.s12_curve_geometry import evaluate_curves
    x=np.zeros((1,64,64),bool)
    x[:,24,8:50]=True
    metrics=evaluate_curves(x,x)
    assert metrics['fitted_pairs']==1
    assert metrics['paired_eligible_coverage']==1
    fragmented=x.copy();fragmented[:,:,20:35]=False
    result=evaluate_curves(fragmented,x)
    assert result['fitted_pairs']==0
