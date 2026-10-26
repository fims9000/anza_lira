"""S85 EVALUATOR-ONLY synthetic candidate labels for TRAIN scenes.

This module is deliberately NOT imported by image inference. It never
supplies artificial truth to the candidate classifier. A candidate matching
neither an injected true fragment nor the decoy is generally ABSTAIN.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

LABEL_SCHEMA = 'S85_SYNTHETIC_TRAIN_LABELS_V1'
DISTANT_RADIUS_PX = 6.0
MIN_DISTANT_FRACTION = .90


def label_extras_after_inference(all_predicted, original_count, synthetic_reference):
    """Return {extra-index: 0/1/None} after blind extraction has finished.

    Positive: strict endpoint-blind global shape association to any one of
    six injected target fragments. Negative: confidently matched injected
    decoy, OR geometrically far from every artificial reference curve.
    Otherwise abstain. This is NOT native H50Z ground truth.
    """
    from bubble_chamber.s76_shape_endpoint_diagnostics import shape_only_match
    if not isinstance(original_count, int) or not 0 <= original_count <= len(all_predicted):
        raise ValueError('Original line-count malformed')
    if len(synthetic_reference) != 7:
        raise ValueError('Synthetic TRAIN reference requires exactly 7 injected fragments')
    if len(all_predicted) > 32:
        raise ValueError('Too many pixel-derived candidate curves')
    matched, _, _ = shape_only_match(all_predicted, synthetic_reference)
    ref_trees = [cKDTree(np.asarray(q, dtype=float)) for q in synthetic_reference]
    labels = {}
    for j in range(original_count, len(all_predicted)):
        if j in matched:
            labels[j-original_count] = 1 if matched[j] < 6 else 0
            continue
        p = np.asarray(all_predicted[j], dtype=float)
        distances = np.column_stack([tree.query(p)[0] for tree in ref_trees])
        far_fraction = np.mean(np.min(distances, axis=1) >= DISTANT_RADIUS_PX)
        labels[j-original_count] = 0 if far_fraction >= MIN_DISTANT_FRACTION else None
    assert len(labels) == len(all_predicted)-original_count
    return labels
