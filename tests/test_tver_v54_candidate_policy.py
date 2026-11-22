"""V54 SBPP candidate search tests. Synthetic fixtures, no Thebe/test labels."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import tver_v26_predicted_cracks_e2e as v26


def _thin_scene():
    p = np.zeros((96, 96), np.float32)
    p[48, 12:45] = .93
    p[48, 53:86] = .91
    gray = np.zeros_like(p)
    gray[48, 12:86] = .8
    return p, gray, (45, 52, 44, 54)


def test_default_candidate_policy_is_identical_to_explicit_original_budgets():
    p, gray, rect = _thin_scene()
    frozen = v26.sbpp_proposals(p, gray, rect, .5)
    explicit = v26.sbpp_proposals(p, gray, rect, .5,
                                  max_sources=4, per_source_k=12,
                                  max_pairs=48, max_source_radius=29.,
                                  min_alignment=.10)
    diagnosed = v26.sbpp_proposals(p, gray, rect, .5,
                                   diagnostics=True)
    assert frozen == explicit
    assert {k: v for k, v in diagnosed.items() if k != "diagnostics"} == frozen
    for pair in diagnosed["proposals"]:
        assert set(pair) == {"source", "target", "features", "geometry"}


def test_expansion_unlocks_later_source_without_oracle(monkeypatch):
    p, gray, rect = _thin_scene()
    ports = [
        SimpleNamespace(point_yx=(48., 28. + float(i)),
                        tangent_yx=(0., 1.), confidence=.8,
                        branch_id=i)
        for i in range(7)
    ]

    def fake_propose(port, clusters):
        return tuple(
            SimpleNamespace(landing_point_yx=(48., 60. + float(j)),
                            geometric_score=float(j + port.branch_id / 100),
                            mean_probability=.9, source_angle=.1,
                            destination_angle=.1, landing_tangent_yx=(0., -1.))
            for j in range(15)
        )

    monkeypatch.setattr(v26, "extract_branches", lambda *a, **k: ())
    monkeypatch.setattr(v26, "source_ports", lambda *a, **k: ports)
    monkeypatch.setattr(v26, "extract_soft_branches", lambda *a, **k: ())
    monkeypatch.setattr(v26, "cluster_branches", lambda *a, **k: ())
    monkeypatch.setattr(v26, "propose_cluster_candidates", fake_propose)
    monkeypatch.setattr(v26, "_features", lambda *a: np.zeros(12))

    original = v26.sbpp_proposals(p, gray, rect, .5, diagnostics=True)
    wide = v26.sbpp_proposals(p, gray, rect, .5,
                              max_sources=8, per_source_k=24,
                              max_pairs=192, diagnostics=True)
    assert original["source_count"] == 4
    assert wide["source_count"] > original["source_count"]
    assert len(wide["proposals"]) > len(original["proposals"])
    assert original["diagnostics"]["ports_after_direction_filter"] > 4
    assert wide["diagnostics"]["ports_used"] > original["diagnostics"]["ports_used"]
    assert wide["diagnostics"]["proposals_before_per_source_k"] > 0
    assert all(set(r) == {"source", "target", "features", "geometry"}
               for r in wide["proposals"])


@pytest.mark.parametrize("kwargs", [
    {"max_sources": 0}, {"per_source_k": 0}, {"max_pairs": 0},
    {"max_source_radius": 2.}, {"max_source_radius": 90.},
    {"min_alignment": -2.}, {"min_alignment": 2.},
    {"max_sources": 2.5},
])
def test_invalid_search_policy_rejected(kwargs):
    p, gray, rect = _thin_scene()
    with pytest.raises(ValueError, match="Invalid explicit SBPP"):
        v26.sbpp_proposals(p, gray, rect, .5, **kwargs)
