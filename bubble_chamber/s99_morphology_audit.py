"""S99 independently annotated partial visual morphology agreement for S95.

Match each confidently human-drawn S97 visual line (track, coil, fiducial,
possible artifact, crossing/bundle) to OBSERVED original S86/S87 source pixels,
then look up the S95 MACHINE class at those observed pixels.

Abstain where S86 observed fragments compete spatially; count missing S86
coverage separately from wrong S95 labels; do not call unannotated pixels
background; do not infer true neutron interactions or physical particle IDs.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from math import isfinite

from .s97_manual_reference import (
    KINDS,POSITIVE_KINDS,_point_array,_resample_polyline
)
from .s98_trace_continuity import _observed_index,_sample_assignment

SCHEMA="S99_BLIND_INDEPENDENT_PARTIAL_VISUAL_MORPHOLOGY_AUDIT_V1"
EXPECTED_S95={
    "TRACK_LIKE":"TRACK_LIKE_CANDIDATE",
    "COILING_TRACK_LIKE":"COILING_TRACE_CANDIDATE",
    "FIDUCIAL_CROSS":"CALIBRATION_CROSS_CONTEXT",
    "ARTIFACT":"ARTIFACT_LIKE_CANDIDATE",
    "CROSSING_OR_BUNDLE":"BUNDLE_OR_INTERSECTION_REVIEW",
}
VALID_MACHINE_TYPES=frozenset((
    "TRACK_LIKE_CANDIDATE","COILING_TRACE_CANDIDATE",
    "CALIBRATION_CROSS_CONTEXT","ARTIFACT_LIKE_CANDIDATE",
    "BUNDLE_OR_INTERSECTION_REVIEW","UNKNOWN",
))


def audit_s95_against_blind_source_strokes(
    human_strokes:list[dict],
    observed_components:list[dict],
    s95_typed_components:list[dict],
    *,
    radius_px:float=2.,
    ambiguity_margin_px:float=.45,
    sampling_step_px:float=1.,
):
    if (not all(isfinite(v) for v in (
            radius_px,ambiguity_margin_px,sampling_step_px))
        or not(.5<=radius_px<=8.)
        or not(.05<=ambiguity_margin_px<=2.)
        or not(.5<=sampling_step_px<=2.)):
        raise ValueError("S99 fixed geometric parameters invalid")

    tree,tags,source_ids=_observed_index(observed_components)
    typed={}
    for row in s95_typed_components:
        cid=row.get("component_id")
        model=row.get("primary_review_category")
        if cid in typed or cid not in source_ids or model not in VALID_MACHINE_TYPES:
            raise ValueError("S99 S95 duplicate, unknown source component or type")
        if (row.get("preserved_source_observed") is not True
            or row.get("physical_auto_join_authorized") is not False):
            raise ValueError("S99 S95 source class may not assert physical GT")
        typed[cid]=model
    if set(typed)!=source_ids:
        raise ValueError("S99 S95 labels must cover every exact original S86 component")

    # Same HUMAN reference denominator for every S86/S95 candidate.
    by_kind=defaultdict(Counter)
    class_confusion=defaultdict(Counter)
    per_stroke=[]
    skipped_uncertain=Counter()
    seen_strokes=set()
    total_sample_points=0
    for human in human_strokes:
        sid=human["stroke_id"]
        if sid in seen_strokes:
            raise ValueError("Duplicate original independent human stroke ID")
        seen_strokes.add(sid)
        kind=human["kind"]
        if kind not in KINDS:
            raise ValueError("Unknown independent human morphology")
        if human["certainty"]!="CLEAR" or kind=="UNKNOWN":
            skipped_uncertain[kind]+=1
            continue
        pts,length=_resample_polyline(
            _point_array(human["points_xy_source_px"]),step_px=sampling_step_px
        )
        if len(pts)>200_000 or total_sample_points+len(pts)>2_000_000:
            raise ValueError("S99 human source sample budget exceeded; split review by photo")
        total_sample_points+=len(pts)
        assigned,ambiguous=_sample_assignment(
            pts,tree,tags,radius_px=radius_px,
            ambiguity_margin_px=ambiguity_margin_px
        )
        found=Counter()
        no_source=tied=0
        models=Counter()
        samples=len(pts)
        for cid,tie in zip(assigned,ambiguous):
            if tie:
                tied+=1
                continue
            if cid is None:
                no_source+=1
                continue
            label=typed[cid]
            models[label]+=1
            found[cid]+=1
        assigned_n=sum(models.values())
        matches=models[EXPECTED_S95[kind]]
        unresolved=models["UNKNOWN"]
        differs=assigned_n-matches-unresolved
        row={
            "human_stroke_id":sid,
            "human_visual_instance_id":human["visual_instance_id"],
            "human_original_2d_morphology":kind,
            "expected_machine_review_class":EXPECTED_S95[kind],
            "human_reference_length_px":round(length,4),
            "human_reference_sampled_points":samples,
            "source_S86_observed_component_assigned_samples":assigned_n,
            "source_S86_unobserved_samples":no_source,
            "source_S86_competing_source_component_abstentions":tied,
            "human_samples_assigned_plus_unobserved_plus_ambiguous":assigned_n+no_source+tied,
            "machine_S95_category_samples_from_observed_only":dict(models),
            "S95_matches_human_expected_morphology_samples":matches,
            "S95_known_other_category_samples":differs,
            "S95_machine_unknown_samples":unresolved,
            "observed_centerline_coverage_excluding_competition":(
                assigned_n/samples
            ),
            "S95_agreement_conditional_on_observed_and_unambiguous":(
                matches/assigned_n if assigned_n else None
            ),
            "sample_points_not_independent_films":True,
            "one_2D_visual_trace_not_certified_physical_particle":True,
        }
        per_stroke.append(row)
        stats=by_kind[kind]
        stats.update({
            "human_strokes":1,"human_samples":samples,
            "detector_observed_samples":assigned_n,
            "detector_unobserved_samples":no_source,
            "geometrically_ambiguous_abstained_samples":tied,
            "expected_morphology_samples":matches,
            "machine_other_known_morphology_samples":differs,
            "machine_unknown_morphology_samples":unresolved,
        })
        class_confusion[kind].update(models)
    summary={}
    for kind in EXPECTED_S95:
        row=by_kind[kind]
        total=row["human_samples"]
        detected=row["detector_observed_samples"]
        summary[kind]={
            **dict(row),
            "expected_S95_machine_review_class":EXPECTED_S95[kind],
            "reference_S86_observed_coverage_conditional_on_this_kind":(
                detected/total if total else None
            ),
            "S95_morphology_agreement_among_assigned_source_points":(
                row["expected_morphology_samples"]/detected if detected else None
            ),
            "S95_machine_unresolved_among_assigned_source_points":(
                row["machine_unknown_morphology_samples"]/detected if detected else None
            ),
            "NOT_true_physical_precision_recall_or_F1":True,
        }
    return {
        "schema":SCHEMA,
        "status":"S99_HUMAN_FIRST_CONDITIONAL_SOURCE_MORPHOLOGY_AGREEMENT",
        "source_component_inventory_count":len(source_ids),
        "S95_labeled_component_inventory_count":len(typed),
        "source_human_clear_classified_strokes":len(per_stroke),
        "human_uncertain_or_unknown_strokes_excluded":dict(skipped_uncertain),
        "human_samples_total_across_explicit_positive_and_explicit_artifact_strokes":total_sample_points,
        "per_human_stroke":per_stroke,
        "per_human_morphology":summary,
        "human_vs_machine_S95_category_sample_confusion_not_global_confusion_matrix":{
            k:dict(class_confusion[k]) for k in EXPECTED_S95
        },
        "conditioned_on_independent_explicit_strokes_not_dense_image_GT":True,
        "unreviewed_and_unmarked_original_pixels_not_negative_labels":True,
        "operator_ARTIFACT_class_is_human_visual_judgment_not_physics_proof":True,
        "reconstruction_of_particle_or_neutron_causality_not_established":True,
        "no_per_sample_pseudoreplication_significance":True,
        "fixed_matching":{
            "radius_px":radius_px,
            "ambiguity_margin_px":ambiguity_margin_px,
            "human_sampling_step_px":sampling_step_px
        },
    }


def descriptive_same_photo_morphology_contrasts(trials:list[dict]):
    """Compare each candidate to the FIRST, never claim IID sample p-values."""
    if not trials:
        raise ValueError("No S99 independent candidate models")
    baseline=trials[0]
    assert baseline["schema"]==SCHEMA
    baseline_ref={
        (r["human_stroke_id"],r["human_original_2d_morphology"],r["human_reference_sampled_points"])
        for r in baseline["per_human_stroke"]
    }
    comparisons=[]
    for trial in trials[1:]:
        if trial["schema"]!=SCHEMA or {
            (r["human_stroke_id"],r["human_original_2d_morphology"],r["human_reference_sampled_points"])
            for r in trial["per_human_stroke"]
        }!=baseline_ref:
            raise ValueError("S99 candidate comparison requires SAME original human positive/negative-visible strokes")
        differences={}
        for kind in EXPECTED_S95:
            b=baseline["per_human_morphology"][kind]
            t=trial["per_human_morphology"][kind]
            if b["human_samples"]!=t["human_samples"]:
                raise ValueError("S99 same-photo human class denominators differ")
            differences[kind]={
                "human_source_samples":b["human_samples"],
                "delta_observed_source_coverage_percentage_points":(
                    100*(t["reference_S86_observed_coverage_conditional_on_this_kind"]-
                         b["reference_S86_observed_coverage_conditional_on_this_kind"])
                    if b["human_samples"] else None
                ),
                "delta_S95_conditional_class_agreement_percentage_points":(
                    100*(t["S95_morphology_agreement_among_assigned_source_points"]-
                         b["S95_morphology_agreement_among_assigned_source_points"])
                    if (b["detector_observed_samples"] and t["detector_observed_samples"]) else None
                ),
                "not_population_effect_estimate":True,
            }
        comparisons.append({
            "comparison_to_first_candidate":True,
            "human_morphology_conditional_differences":differences,
            "independent_event_level_CI_not_available":True,
            "no_claim_improvement_without_real_photo_reference":True,
        })
    return comparisons
