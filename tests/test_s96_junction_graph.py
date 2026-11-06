"""S96 adversarial 2D crossing/branch/bundle geometry with no physical GT."""
from __future__ import annotations
import math
import numpy as np
import pytest

from bubble_chamber.s96_junction_graph import (
    JunctionConfig,propose_four_arm_continuations,
    grouped_2d_instance_hypotheses,
)


def _line(a,b,n=41):
    return np.linspace(np.asarray(a,float),np.asarray(b,float),n)


def cross(*,center=(100,100),radius=5.):
    x,y=center
    return [
        {"component_id":"W","points_xy":_line((x-40,y),(x-radius,y))},
        {"component_id":"E","points_xy":_line((x+radius,y),(x+40,y))},
        {"component_id":"N","points_xy":_line((x,y-40),(x,y-radius))},
        {"component_id":"S","points_xy":_line((x,y+radius),(x,y+40))},
    ]


def _pairs(rows):
    return {frozenset((r["component_a"],r["component_b"])) for r in rows}


def test_four_arm_crossing_yields_two_disjoint_visual_alternatives():
    fragments=cross()
    saved=[x["points_xy"].copy() for x in fragments]
    junctions,pairs,report=propose_four_arm_continuations(fragments)
    assert len(junctions)==1 and len(pairs)==2
    assert _pairs(pairs)=={frozenset(("W","E")),frozenset(("N","S"))}
    assert report["candidate_junctions"]==1
    assert report["four_arm_junctions_with_two_visual_pairs"]==1
    assert report["physical_auto_joins_authorized"] is False
    assert all(p["physical_auto_join"] is False and p["status"]=="UNKNOWN"
               for p in pairs)
    assert all(p["inferred_only"] is True for p in pairs)
    assert len(junctions[0]["all_compatible_pair_options"])>=2
    assert junctions[0]["junction_status"]=="TWO_ALTERNATIVE_VISUAL_CROSSING_CONTINUATIONS_UNVERIFIED"
    chains,inventory=grouped_2d_instance_hypotheses(fragments,pairs)
    assert len(chains)==2
    assert {frozenset(x["ordered_source_components"]) for x in chains}==_pairs(pairs)
    assert all(len(x["observed_source_sections_xy"])==2 for x in chains)
    assert all(len(x["unobserved_junction_interpolations"])==1 for x in chains)
    assert all(x["same_particle_identity_verified"] is False for x in chains)
    assert inventory["unassociated_source_components"]==[]
    for before,after in zip(saved,fragments):
        assert np.array_equal(before,after["points_xy"])


def test_t_branch_abstains_and_preserves_every_observed_arm():
    c=cross()
    three=c[:3]
    j,e,r=propose_four_arm_continuations(three)
    assert len(j)==1
    assert j[0]["junction_status"]=="THREE_ARM_BRANCH_ABSTAIN"
    assert e==[] and r["three_arm_branch_abstentions"]==1
    chains,idx=grouped_2d_instance_hypotheses(three,e)
    assert chains==[]
    assert idx["unassociated_source_components"]==sorted(x["component_id"] for x in three)


def test_five_arm_bundle_is_not_forced_into_two_tracks():
    data=cross()
    data.append({
        "component_id":"DIAGONAL",
        "points_xy":_line((70,70),(96,96))
    })
    j,e,r=propose_four_arm_continuations(data)
    assert len(j)==1
    assert j[0]["junction_status"]=="DENSE_BUNDLE_ABSTAIN"
    assert not e
    assert r["source_components_preserved"]==len(data)


def test_two_parts_of_one_dashed_line_are_not_a_four_arm_collision():
    lines=[
        {"component_id":"a","points_xy":_line((10,45),(46,45))},
        {"component_id":"b","points_xy":_line((54,45),(90,45))},
    ]
    j,e,r=propose_four_arm_continuations(lines)
    assert not j and not e
    assert r["source_components"]==2


def test_wrong_direction_four_arm_group_abstains_when_matching_not_unique():
    # Four arms point in approximately same direction, not into each
    # other: there is no supported two-line crossing.
    data=[]
    for i in range(4):
        data.append({
            "component_id":f"r{i}",
            "points_xy":_line((64,90+i*2),(96,90+i*2)),
        })
    j,e,r=propose_four_arm_continuations(data)
    assert not e
    assert j[0]["junction_status"] in (
        "INCOMPLETE_OR_AMBIGUOUS_FOUR_ARM_PAIRING",
        "CLUSTER_SPREAD_ABSTAIN",
    )


def test_translation_rotation_mirror_and_source_orientation_invariance():
    base=cross()
    _,pairs,_=propose_four_arm_continuations(base)
    assert len(pairs)==2
    t=.38
    R=np.array([[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]])
    for M in (R,np.diag([-1.,1.]),np.eye(2)):
        transformed=[
            {"component_id":r["component_id"],
             "points_xy":(r["points_xy"]@M.T)+np.array([400,180])}
            for r in base
        ]
        j,edges,_=propose_four_arm_continuations(transformed)
        assert len(edges)==2
        assert _pairs(edges)==_pairs(pairs)
    # Arbitrary source polyline ordering does not change endpoint matching.
    reversed_paths=[
        {"component_id":r["component_id"],"points_xy":r["points_xy"][::-1]}
        for r in base
    ]
    _,edges,_=propose_four_arm_continuations(reversed_paths)
    assert _pairs(edges)==_pairs(pairs)


def test_unbounded_dense_neighborhood_abstains_entirely():
    data=cross()
    _,edges,r=propose_four_arm_continuations(
        data,JunctionConfig(max_neighbor_pairs=2)
    )
    assert not edges
    assert r["status"]=="DENSE_ENDPOINT_NEIGHBORHOOD"
    assert r["source_components_preserved"]==4


def test_invalid_identity_input_is_rejected_and_no_invalid_source_is_hidden():
    data=cross()
    with pytest.raises(ValueError,match="Duplicate"):
        propose_four_arm_continuations(data+[data[0]])
    with pytest.raises(ValueError,match="Degenerate"):
        propose_four_arm_continuations([{
            "component_id":"x","points_xy":np.array([[0,0],[0,0],[1,0],[2,0]])}])
    with pytest.raises(ValueError,match="bounded"):
        propose_four_arm_continuations(data,JunctionConfig(max_endpoint_count=3))
    j,e,_=propose_four_arm_continuations(data)
    corrupted=e+[e[0]]
    with pytest.raises(ValueError,match="double-occupied"):
        grouped_2d_instance_hypotheses(data,corrupted)
    bad=[{**e[0],"physical_auto_join":True},e[1]]
    with pytest.raises(ValueError,match="unsafe"):
        grouped_2d_instance_hypotheses(data,bad)
