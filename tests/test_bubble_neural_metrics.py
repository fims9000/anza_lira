import numpy as np
import pytest
from bubble_chamber.neural_metrics import topology_for_image, aggregate_topology


def test_identical_tracks_have_perfect_topology():
    x = np.zeros((40, 40), bool)
    x[12, 3:34] = True
    x[21, 3:34] = True
    assert topology_for_image(x, x)['cldice'] == 1.
    assert topology_for_image(x, x)['centerline_f1'] == 1.


def test_missing_long_section_worsens_skeleton_recall():
    truth = np.zeros((50, 50), bool)
    truth[25, 2:48] = True
    pred = truth.copy()
    pred[25, 15:35] = False
    m = topology_for_image(pred, truth)
    assert 0.0 < m['cldice'] < 1.0
    assert m['centerline_recall'] < 1.0
    assert m['pred_components_8'] == 2 and m['true_components_8'] == 1


def test_false_merging_two_tracks_not_mistaken_for_good_dice():
    y = np.zeros((50, 50), bool)
    y[15, 4:40] = True
    y[27, 4:40] = True
    p = y.copy()
    p[15:28, 23] = True  # a false bridge
    m = topology_for_image(p, y)
    assert m['cldice'] < 1.0
    assert m['pred_components_8'] == 1 and m['true_components_8'] == 2
    assert aggregate_topology(p[None], y[None])['instance_link_accuracy_available'] is False


def test_zero_masks_and_bad_input():
    p = np.zeros((10, 10), np.uint8)
    assert topology_for_image(p, p)['cldice'] == 1.0
    assert topology_for_image(p, p)['centerline_recall'] == 1.0
    with pytest.raises(ValueError):
        topology_for_image(np.zeros((4, 2)), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        topology_for_image(p, p, tolerance_px=-1)
    with pytest.raises(ValueError):
        aggregate_topology(p, p)
