"""S38 blinded, independent local annotation workflow for H50Z proposals.

Source-only code; neither S31 machine drafts nor reviewer judgments are physical GT.
No source pixels, identities, original coordinates or item-level answers are published.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import re

from scripts.research.bubble_chamber.s37_validate_local_corrections import validate_reviews

LINE_DECISIONS = {"USER_ACCEPTED_PROVISIONAL", "USER_REJECTED_PROVISIONAL"}
PAIR_DECISIONS = {"LOCAL_SAME_PROVISIONAL", "LOCAL_DIFFERENT_PROVISIONAL"}
ROUND_NAMES = ("A", "B")
CROP_STATES = {"REVIEWED_FOR_MISSED_TRACKS_LOCAL", "NO_ADDITIONAL_LOCAL", "UNSURE_LOCAL"}
MAX_EDIT_POINTS = 128


def _rank(seed: str, key: str) -> str:
    return sha256((seed + ":" + key).encode("utf-8")).hexdigest()


def _interleave(groups, round_name: str):
    """Deterministic round-robin sampling across photographic and source strata.

    No model probability, class label, or test ground truth is involved.
    """
    if round_name not in ROUND_NAMES:
        raise ValueError("Round must be A or B")
    remaining = {key: sorted(value, key=lambda p: _rank(round_name, p))
                 for key, value in groups.items()}
    ordered = []
    while any(remaining.values()):
        order = sorted(remaining, key=lambda g: (_rank(round_name + ':group', repr(g))))
        for group in order:
            if remaining[group]:
                ordered.append(remaining[group].pop(0))
    return ordered


def review_queues(drafts: dict, evidence: dict) -> dict:
    photos = drafts.get("photos")
    relations = evidence.get("rows")
    if not isinstance(photos, list) or len(photos) != 39:
        raise ValueError("Must have exactly 39 source H50Z crops")
    if not isinstance(relations, list) or len(relations) != 28:
        raise ValueError("Must have 28 real candidate pair questions")
    line_groups = defaultdict(list)
    crop_groups = defaultdict(list)
    known_lines = {}
    photo_by_crop = {}
    for photo in photos:
        cid = photo["crop_id"]
        if cid in photo_by_crop:
            raise ValueError("Duplicate crop id")
        photo_by_crop[cid] = photo
        crop_groups[(str(photo["photo_id"]), str(photo.get("stratum", "unknown")))].append(cid)
        for draft in photo["drafts"]:
            did = draft["draft_id"]
            if did in known_lines or not isinstance(draft["points_xy"], list):
                raise ValueError("Duplicate/invalid line proposal")
            known_lines[did] = cid
            group = (str(photo["photo_id"]), str(draft["source_method"]))
            line_groups[group].append(did)
    if len(known_lines) != 133:
        raise ValueError("Expected 133 machine-only line proposals")
    pair_groups = defaultdict(list)
    seen_pairs = set()
    for r in relations:
        rid = r["relation_id"]
        if rid in seen_pairs or r["crop_id"] not in photo_by_crop:
            raise ValueError("Unknown crop or duplicate relation")
        seen_pairs.add(rid)
        if any(known_lines.get(r[k]) != r["crop_id"] for k in ("candidate_a", "candidate_b")):
            raise ValueError("Both relation fragments must belong to the same crop")
        # Input scores intentionally excluded from sorting. Pre-specified coarse
        # endpoint-gap bins help expose small and large gaps early.
        gap = float(r["measured_endpoint_gap_px"])
        if not math.isfinite(gap) or gap < 0:
            raise ValueError("Bad measured gap")
        gap_group = "short" if gap < 8 else "medium" if gap < 24 else "long"
        pair_groups[(str(r["photo_id"]), gap_group)].append(rid)
    return {round_name: {
        "lines": _interleave(line_groups, round_name),
        "pairs": _interleave(pair_groups, round_name),
        "crops": _interleave(crop_groups, round_name),
    } for round_name in ROUND_NAMES}


def validate_session(template: dict, drafts: dict, evidence: dict, session: dict) -> dict:
    """Check S31 compatibility + independent review discipline; never certify GT."""
    base = validate_reviews(template, drafts, session)
    info = session.get("s38_review_session")
    if not isinstance(info, dict) or info.get("protocol") != "s38_independent_blind_v1":
        raise ValueError("S38 session metadata missing or incompatible")
    who = info.get("reviewer_id")
    if not isinstance(who, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{2,40}", who):
        raise ValueError("Pseudonymous reviewer ID 2..40 ASCII characters required")
    if info.get("round") not in ROUND_NAMES:
        raise ValueError("Unknown annotation round")
    if info.get("scores_revealed") not in (True, False):
        raise ValueError("Missing blind-state audit")
    if info.get("physical_gt_certified") is not False:
        raise ValueError("Review cannot certify physical ground truth")
    line_by_id = {d["draft_id"]:d for p in drafts["photos"] for d in p["drafts"]}
    relation_by_id = {r["relation_id"]: r for r in evidence["rows"]}
    if len(relation_by_id) != 28:
        raise ValueError("Invalid relation source")
    for did, answer in session["decisions"].items():
        if answer.get("review_origin") not in (None,"LOCAL_USER_PROVISIONAL_NOT_PHYSICIST_CERTIFIED"):
            raise ValueError("Unrecognized attempt to promote line-review provenance")
        edited = answer.get("edited_points_xy")
        if edited is not None:
            if len(edited) > MAX_EDIT_POINTS:
                raise ValueError("Too many edited points")
            if any(not isinstance(p,list) or len(p)!=2 or
                   any(not isinstance(x,(float,int)) or isinstance(x,bool) or not math.isfinite(x) or x<0 or x>511 for x in p)
                   for p in edited):
                raise ValueError("Bad edited coordinates")
            # Source IDs and image bounds are checked by S37. Require non-degenerate arc.
            length = sum(math.dist(p,q) for p,q in zip(edited, edited[1:]))
            if length < 2 or not math.isfinite(length):
                raise ValueError("Degenerate edited curve")
            if answer["status"] != "USER_ACCEPTED_PROVISIONAL":
                raise ValueError("Edits require accepting the proposal provisionally")
        if answer["status"] not in LINE_DECISIONS | {"USER_UNCERTAIN"}:
            raise ValueError("Unknown line status")
        if did not in line_by_id:
            raise ValueError("Foreign line")
    for rid, answer in session["relation_decisions"].items():
        if answer.get("review_origin") not in (None,"LOCAL_USER_NOT_PHYSICIST_CERTIFIED"):
            raise ValueError("Unrecognized attempt to promote pair-review provenance")
        if rid not in relation_by_id:
            raise ValueError("Unknown relation ID")
        if answer["status"] in PAIR_DECISIONS:
            r = relation_by_id[rid]
            for k in ("candidate_a", "candidate_b"):
                if session["decisions"].get(r[k], {}).get("status") != "USER_ACCEPTED_PROVISIONAL":
                    raise ValueError("Pair identity judgment requires both lines reviewed as visible")
    # Manual missed tracks and blank-crop inspections are their own local
    # observations. In particular, no machine proposal does NOT imply background.
    crops = {p["crop_id"] for p in drafts["photos"]}
    assessments=session.get("s38_crop_assessments",{})
    manual=session.get("s38_manual_lines",[])
    if not isinstance(assessments,dict) or not isinstance(manual,list):
        raise ValueError("Invalid crop review or manual line collection")
    if len(manual)>512 or len(assessments)>39:
        raise ValueError("Oversized crop review collection")
    for crop,answer in assessments.items():
        if crop not in crops or not isinstance(answer,dict) or answer.get("status") not in CROP_STATES:
            raise ValueError("Invalid crop visual review")
        if answer.get("review_origin") not in (None,"LOCAL_VISUAL_REVIEW_NOT_EXPERT_CERTIFIED"):
            raise ValueError("Invalid crop review origin")
    seen_manual=set()
    for proposal in manual:
        if not isinstance(proposal,dict) or proposal.get("crop_id") not in crops:
            raise ValueError("Manual line has unknown source crop")
        mid=proposal.get("manual_id")
        if not isinstance(mid,str) or not re.fullmatch(r"manual_[0-9]{4}",mid) or mid in seen_manual:
            raise ValueError("Duplicate or malformed local manual line identifier")
        seen_manual.add(mid)
        if proposal.get("status")!="USER_DRAWN_PROVISIONAL":
            raise ValueError("Manual trace is not an approved physics label")
        pts=proposal.get("points_xy")
        if not isinstance(pts,list) or not 2<=len(pts)<=MAX_EDIT_POINTS:
            raise ValueError("Bad manually added centerline")
        if any(not isinstance(p,list) or len(p)!=2 or
               any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) or x<0 or x>511 for x in p)
               for p in pts):
            raise ValueError("Nonfinite or out-of-bounds manual centerline")
        length=sum(math.dist(a,b) for a,b in zip(pts,pts[1:]))
        if length<2 or not math.isfinite(length):
            raise ValueError("Degenerate manually drawn trace")
        if assessments.get(proposal["crop_id"],{}).get("status")=="NO_ADDITIONAL_LOCAL":
            raise ValueError("No-additional-mark contradicts new traced candidates")
    return {**base, "blinded_review": not info["scores_revealed"],
            "manual_provisional_new_lines":len(manual),
            "crop_visual_reviews":len(assessments),
            "physical_auto_join_authorized":False,
            "source_machine_labels_remain_unverified":True}


def _binary_reviewer_agreement(ids, left: dict, right: dict, valid: set) -> dict:
    """Conditional two-label agreement; no physical truth, no independent CI."""
    both = [(left.get(i, {}).get("status"), right.get(i, {}).get("status"))
            for i in ids]
    both = [(x,y) for x,y in both if x in valid and y in valid]
    n = len(both)
    if not n:
        return {"co_reviewed_two_label_n":0,"observed_agreement":None,"cohens_kappa":None}
    po = sum(x==y for x,y in both)/n
    chance = sum((sum(x==v for x,y in both)/n) *
                 (sum(y==v for x,y in both)/n) for v in valid)
    kappa = (po-chance)/(1-chance) if 1-chance>1e-12 else None
    return {"co_reviewed_two_label_n":n,
            "observed_agreement":round(po,6),
            "cohens_kappa":round(kappa,6) if kappa is not None else None}


def compare_sessions(template: dict, drafts: dict, evidence: dict, a: dict, b: dict):
    """Build PRIVATE individual adjudication queue and PUBLIC aggregate counts.

    Two coincident provisional opinions are *not* an independent physical truth.
    The audit does not calculate real sensitivity, precision, or GT coverage.
    """
    va = validate_session(template, drafts, evidence, a)
    vb = validate_session(template, drafts, evidence, b)
    meta_a, meta_b = a["s38_review_session"], b["s38_review_session"]
    if meta_a["reviewer_id"] == meta_b["reviewer_id"]:
        raise ValueError("Independent review needs different pseudonymous reviewer IDs")
    if meta_a["round"] != "A" or meta_b["round"] != "B":
        raise ValueError("Independent rounds A and B are both required")
    all_lines = {d["draft_id"]:d for p in drafts["photos"] for d in p["drafts"]}
    all_pairs = {r["relation_id"]:r for r in evidence["rows"]}
    tasks = []
    counts = Counter()
    for kind, items, property_name, valid in (("line",all_lines,"decisions",LINE_DECISIONS),
                                               ("pair",all_pairs,"relation_decisions",PAIR_DECISIONS)):
        for iid in sorted(items):
            aa = a[property_name].get(iid, {})
            bb = b[property_name].get(iid, {})
            sa, sb = aa.get("status"), bb.get("status")
            if sa in valid and sb in valid and sa == sb:
                # A matching label does not make two different centerlines identical.
                if kind=="line" and aa.get("edited_points_xy") != bb.get("edited_points_xy"):
                    category="edit_disagreement"
                else:
                    category="provisional_agreement"
            elif sa in valid and sb in valid:
                category="label_disagreement"
            elif sa is None and sb is None:
                category="unreviewed_both"
            else:
                category="incomplete_or_unsure"
            counts[(kind,category)] += 1
            if category != "provisional_agreement":
                tasks.append({"kind":kind,"item_id":iid,"adjudication_status":category,
                              "round_a_status":sa or "UNKNOWN", "round_b_status":sb or "UNKNOWN",
                              "requires_independent_human_resolution":True})
    ac=a.get("s38_crop_assessments",{})
    bc=b.get("s38_crop_assessments",{})
    for photo in drafts["photos"]:
        crop=photo["crop_id"]
        sa=ac.get(crop,{}).get("status")
        sb=bc.get(crop,{}).get("status")
        if sa in CROP_STATES and sb in CROP_STATES:
            counts[("crop","both_reviewed")] += 1
            if sa!=sb:
                tasks.append({"kind":"crop","item_id":crop,
                              "adjudication_status":"visual_review_disagreement",
                              "round_a_status":sa,"round_b_status":sb,
                              "requires_independent_human_resolution":True})
        elif sa or sb:
            counts[("crop","only_one_reviewed")] += 1
            tasks.append({"kind":"crop","item_id":crop,
                          "adjudication_status":"unpaired_visual_review",
                          "round_a_status":sa or "UNKNOWN","round_b_status":sb or "UNKNOWN",
                          "requires_independent_human_resolution":True})
    for rnd,session in (("A",a),("B",b)):
        # Added traces do not have stable cross-review identities: no automatic
        # curve matching or invented same-track labels can follow from them.
        for m in session.get("s38_manual_lines",[]):
            tasks.append({"kind":"new_manual_line","item_id":m["manual_id"],
                          "crop_id":m["crop_id"],"round":rnd,
                          "adjudication_status":"needs_independent_matching",
                          "requires_independent_human_resolution":True})
    line_agree=_binary_reviewer_agreement(all_lines,a["decisions"],b["decisions"],LINE_DECISIONS)
    pair_agree=_binary_reviewer_agreement(all_pairs,a["relation_decisions"],b["relation_decisions"],PAIR_DECISIONS)
    public = {
        "schema":"s38_double_review_aggregate_only_v1",
        "line_two_label_agreement":line_agree,
        "pair_two_label_agreement":pair_agree,
        "agreement_is_not_accuracy":True,
        "source_crops":39,"machine_line_questions":len(all_lines),"machine_pair_questions":len(all_pairs),
        "rounds_distinct":True,
        "scores_hidden_in_both_rounds":va["blinded_review"] and vb["blinded_review"],
        "line_provisional_agreements":counts[("line","provisional_agreement")],
        "line_label_disagreements":counts[("line","label_disagreement")],
        "line_edit_disagreements":counts[("line","edit_disagreement")],
        "line_incomplete_or_unsure":counts[("line","incomplete_or_unsure")],
        "line_unreviewed_both":counts[("line","unreviewed_both")],
        "pair_provisional_agreements":counts[("pair","provisional_agreement")],
        "pair_label_disagreements":counts[("pair","label_disagreement")],
        "pair_incomplete_or_unsure":counts[("pair","incomplete_or_unsure")],
        "pair_unreviewed_both":counts[("pair","unreviewed_both")],
        "ready_for_physics_adjudication":counts[("pair","provisional_agreement")] > 0,
        "crops_reviewed_in_both_rounds":counts[("crop","both_reviewed")],
        "crops_reviewed_in_one_round_only":counts[("crop","only_one_reviewed")],
        "manual_new_line_proposals_round_a":len(a.get("s38_manual_lines",[])),
        "manual_new_line_proposals_round_b":len(b.get("s38_manual_lines",[])),
        "manual_new_line_proposals_auto_matched":0,
        "human_certified_native_track_masks":0,
        "physical_identity_gt_verified":False,
        "automatic_particle_track_joins":False,
        "physics_reconstruction_authorized":False,
        "note":"Two local provisional opinions never substitute for independent physicist-certified native GT",
    }
    # Independent adjudication must explicitly resolve disputes and can reject
    # any existing agreement; no automatic training or production promotions.
    private = {"schema":"s38_private_adjudication_worklist_v1",
               "source_scope":"H50Z two original photos same physical run",
               "reviewers":{"round_a":meta_a["reviewer_id"],"round_b":meta_b["reviewer_id"]},
               "tasks":tasks,"physical_auto_join_authorized":False}
    return private,public


def source_statistics(drafts:dict, evidence:dict, queues:dict) -> dict:
    photos = drafts["photos"]
    by_photo=Counter(p["photo_id"] for p in photos)
    methods=Counter(d["source_method"] for p in photos for d in p["drafts"])
    pairs=Counter(r["photo_id"] for r in evidence["rows"])
    out={"schema":"s38_machine_source_inventory_no_gt_v1",
         "source_photos":2,"physical_runs":1,"cropped_images":len(photos),
         "machine_lines":sum(len(p["drafts"]) for p in photos),
         "machine_pairs":len(evidence["rows"]),
         "crops_without_machine_line_proposals":sum(not p["drafts"] for p in photos),
         "crops_by_photo":dict(sorted(by_photo.items())),
         "line_methods":dict(sorted(methods.items())),
         "pairs_by_photo":dict(sorted(pairs.items())),
         "blinded_staggered_queues":{
             name:{"line_count":len(queue["lines"]),"pair_count":len(queue["pairs"]),"crop_count":len(queue["crops"])}
             for name,queue in queues.items()},
         "expert_certified_real_track_masks":0,"expert_certified_real_links":0,
         "native_pixel_accuracy_measurable":False,
         "note":"Source inventory only; machine proposals and reviewer votes are not physical GT"}
    return out


def canonical_json_sha(obj:dict)->str:
    return sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,
                            separators=(',',':')).encode('utf-8')).hexdigest()
