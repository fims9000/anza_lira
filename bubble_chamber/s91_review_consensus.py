"""S91 private, read-only analysis of S90 *appearance* reviews.

No segment, human vote or inter-rater agreement in this module is physical GT.
Only previously S90-verified, S89-provenance-bound files are accepted. Outputs
are private and never modify a detector mask, source photograph or human vote.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

from .bitmap_tiles import sha256_file
from .s89_private_candidate_audit import SCHEMA as S89_SCHEMA
from .s90_physicist_review import (
    CASE_RE, CHOICES, PACK_SCHEMA, REVIEWER_RE, SUMMARY_SCHEMA,
    _json_bytes, _private_html, _read_ndjson, _safe_private_destination,
    _write_once,
)

REPORT_SCHEMA = 'ANZA_LIRA_S91_PRIVATE_APPEARANCE_CONSENSUS_V1'
QUEUE_SCHEMA = 'ANZA_LIRA_S91_PRIVATE_BLINDED_FOLLOWUP_V1'
STRATA = ('REVIEW_FIRST', 'REVIEW_LATER_NOT_VERIFIED')
REVIEW_STATES = (
    'INSUFFICIENT_INDEPENDENT_REVIEWS',
    'DISAGREEMENT_OR_UNDECIDABLE_REVIEW',
    'IMAGE_APPEARANCE_TRACK_LIKE_AGREEMENT_NOT_GT',
    'IMAGE_APPEARANCE_ARTIFACT_LIKE_AGREEMENT_NOT_GT',
)


@dataclass(frozen=True)
class ConsensusConfig:
    max_followup_cases: int = 24
    build_blinded_followup: bool = True

    def validate(self) -> None:
        if not 0 <= self.max_followup_cases <= 200:
            raise ValueError('max_followup_cases must be within 0..200')


def _json(path: Path) -> dict:
    if path.stat().st_size > 10_000_000:
        raise ValueError('PRIVATE JSON input too large')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('expected a JSON object')
    return data


def _expected_state(votes: list[dict]) -> str:
    if len(votes) < 2:
        return REVIEW_STATES[0]
    if all(v['choice'] == 'TRACK_LIKE' for v in votes):
        return REVIEW_STATES[2]
    if all(v['choice'] == 'ARTIFACT_LIKE' for v in votes):
        return REVIEW_STATES[3]
    return REVIEW_STATES[1]


def _check_inputs(pack_dir: Path, verified_dir: Path, s89_dir: Path):
    pack_path = pack_dir / 'review_pack.json'
    review_path = verified_dir / 'review_summary.json'
    vote_path = verified_dir / 'case_votes.ndjson'
    audit_path = s89_dir / 'candidate_audit.ndjson'
    report_path = s89_dir / 'report.json'
    pack = _json(pack_path)
    verified = _json(review_path)
    audit = _read_ndjson(audit_path)
    audit_report = _json(report_path)
    if (pack.get('schema') != PACK_SCHEMA or
            pack.get('status') != 'PRIVATE_BLINDED_MACHINE_PROPOSAL_REVIEW' or
            pack.get('native_track_GT_count') != 0 or
            pack.get('physical_auto_joins_authorized') is not False or
            pack.get('image_space_reviews_not_physical_truth') is not True or
            verified.get('schema') != SUMMARY_SCHEMA or
            verified.get('native_physics_track_GT_count') != 0 or
            verified.get('physical_track_accuracy_measured') is not False or
            verified.get('no_training_labels_or_detector_mask_modifications') is not True or
            audit_report.get('schema') != S89_SCHEMA or
            audit_report.get('status') != 'COMPLETE_MACHINE_ONLY' or
            audit_report.get('physics_track_GT_count') != 0 or
            audit_report.get('physical_auto_joins_authorized') is not False):
        raise ValueError('untrusted S89/S90 schema or physical-truth semantics')
    if (sha256_file(pack_path) != verified.get('review_pack_sha256') or
            sha256_file(audit_path) != pack.get('source_S89_audit_sha256') or
            sha256_file(report_path) != pack.get('source_S89_report_sha256') or
            pack.get('source_BMP_sha256') != verified.get('source_BMP_sha256') or
            pack.get('source_BMP_sha256') != audit_report.get('original_BMP_sha256') or
            pack.get('source_S86_manifest_sha256') != verified.get('source_S86_manifest_sha256') or
            pack.get('source_S86_manifest_sha256') != audit_report.get('input_s86_manifest_sha256') or
            verified.get('source_S89_report_sha256') != pack.get('source_S89_report_sha256') or
            audit_report.get('sampled_fragment_count') != len(audit)):
        raise ValueError('S89/S90 source/manifest/response SHA256 mismatch')

    by_fragment = {}
    for item in audit:
        fid = item.get('fragment_id')
        if (not isinstance(fid, str) or fid in by_fragment or
                item.get('review_priority') not in STRATA or
                item.get('status') != 'UNKNOWN' or
                item.get('physical_auto_accept') is not False or
                item.get('physical_auto_reject') is not False):
            raise ValueError('invalid or duplicated S89 fragment/priority')
        by_fragment[fid] = item
    cases = pack.get('cases')
    if not isinstance(cases, list) or not cases or len(cases) > 200 or pack.get('case_count') != len(cases):
        raise ValueError('review pack has invalid case count')
    case_map = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('invalid S90 case row')
        cid, fid = case.get('case_id'), case.get('source_fragment_id')
        if (not isinstance(cid, str) or CASE_RE.fullmatch(cid) is None or cid in case_map or
                fid not in by_fragment or
                cid != 'c_' + sha256((pack['source_BMP_sha256'] + ':S90:' + fid).encode()).hexdigest()[:20]):
            raise ValueError('duplicate/unmatched S90 case identity')
        case_map[cid] = case
        for kind, suffix in (('original_preview', '_original.png'), ('overlay_preview', '_overlay.png')):
            rel = f'images/{cid}{suffix}'
            if case.get(kind) != rel or sha256_file(pack_dir / rel) != case.get(kind + '_sha256'):
                raise ValueError('modified or missing S90 original/overlay preview image')

    rows = _read_ndjson(vote_path)
    if len(rows) != len(cases):
        raise ValueError('S90 verified case count mismatch')
    by_case = {}
    unique_reviewers = set()
    states = Counter()
    for row in rows:
        cid = row.get('case_id')
        if not isinstance(cid, str) or cid not in case_map or cid in by_case:
            raise ValueError('unknown/duplicate S90 verified case')
        if (row.get('source_fragment_id') != case_map[cid]['source_fragment_id'] or
                row.get('native_track_GT') is not False or
                row.get('physical_auto_accept') is not False or
                row.get('physical_auto_reject') is not False):
            raise ValueError('inconsistent S90 case identity or physical semantics')
        votes = row.get('votes')
        if not isinstance(votes, list) or len(votes) > 200:
            raise ValueError('invalid S90 votes')
        seen = set()
        for vote in votes:
            if not isinstance(vote, dict) or set(vote) != {'reviewer_id', 'choice', 'note'}:
                raise ValueError('invalid S90 reviewer vote schema')
            reviewer = vote['reviewer_id']
            if (not isinstance(reviewer, str) or REVIEWER_RE.fullmatch(reviewer) is None or
                    reviewer in seen or vote['choice'] not in CHOICES or
                    not isinstance(vote['note'], str) or len(vote['note']) > 300):
                raise ValueError('invalid or repeated reviewer vote')
            seen.add(reviewer)
            unique_reviewers.add(reviewer)
        status = _expected_state(votes)
        if row.get('status') != status:
            raise ValueError('S90 verified vote status does not match votes')
        states[status] += 1
        by_case[cid] = row
    if (set(by_case) != set(case_map) or
            verified.get('selected_candidate_count') != len(cases) or
            verified.get('unique_reviewer_count') != len(unique_reviewers) or
            verified.get('cases_reviewed_by_at_least_one') != sum(bool(row['votes']) for row in rows) or
            verified.get('cases_reviewed_by_at_least_two') != sum(len(row['votes']) >= 2 for row in rows) or
            verified.get('review_state_counts') != dict(sorted(states.items())) or
            verified.get('physical_auto_joins_authorized') is not False):
        raise ValueError('S90 verified summary does not match independently replayed votes')
    return pack, verified, by_case, by_fragment


def pairwise_agreement(by_case: dict) -> list[dict]:
    """Descriptive overlapping-reviewer agreement; never a physical label metric."""
    by_reviewer = defaultdict(dict)
    for cid, row in by_case.items():
        for vote in row['votes']:
            by_reviewer[vote['reviewer_id']][cid] = vote['choice']
    results = []
    for a, b in combinations(sorted(by_reviewer), 2):
        common = sorted(by_reviewer[a].keys() & by_reviewer[b].keys())
        n = len(common)
        matches = sum(by_reviewer[a][c] == by_reviewer[b][c] for c in common)
        count_a = Counter(by_reviewer[a][c] for c in common)
        count_b = Counter(by_reviewer[b][c] for c in common)
        expected = (sum(count_a[x] * count_b[x] for x in CHOICES) / n**2 if n else None)
        observed = (matches / n if n else None)
        kappa = ((observed - expected) / (1 - expected)
                 if n >= 2 and expected is not None and expected < 1 - 1e-12 else None)
        results.append({'reviewer_A': a, 'reviewer_B': b, 'overlap_cases': n,
                        'matching_votes': matches, 'observed_agreement': observed,
                        'cohen_kappa': kappa,
                        'kappa_status': 'DESCRIPTIVE_ONLY' if kappa is not None else
                        'UNDEFINED_INSUFFICIENT_OVERLAP_OR_SINGLE_CLASS',
                        'not_physical_accuracy': True})
    return results


def _followup_queue(pack: dict, by_case: dict, by_fragment: dict, cfg: ConsensusConfig) -> list[dict]:
    eligible = []
    for case in pack['cases']:
        row = by_case[case['case_id']]
        if row['status'] not in REVIEW_STATES[:2]:
            continue
        status_priority = 0 if row['status'] == REVIEW_STATES[1] else 1 if row['votes'] else 2
        s89 = by_fragment[case['source_fragment_id']]['review_priority']
        sub_priority = 0 if s89 == 'REVIEW_FIRST' else 1
        eligible.append((status_priority, sub_priority,
                         sha256((pack['source_BMP_sha256'] + ':S91:' + case['case_id']).encode()).hexdigest(),
                         {'case_id': case['case_id'], 'status': row['status'],
                          'reviewer_vote_count': len(row['votes']), 's89_priority': s89}))
    eligible.sort(key=lambda entry: entry[:3])
    return [row for _, _, _, row in eligible[:cfg.max_followup_cases]]


def analyze_reviews(pack_dir: Path, verified_dir: Path, s89_dir: Path, out_dir: Path,
                    cfg: ConsensusConfig = ConsensusConfig()) -> dict:
    """Produce a PRIVATE source-locked report and blinded subset for fresh reviewers.

    Additional answers export the **original** S90 pack hash. Merge them by
    rerunning S90 verify with all independent responses against that original.
    """
    cfg.validate()
    pack_dir, verified_dir, s89_dir = (Path(x).resolve() for x in (pack_dir, verified_dir, s89_dir))
    out = _safe_private_destination(out_dir, pack_dir, verified_dir, s89_dir)
    pack, verified, by_case, by_fragment = _check_inputs(pack_dir, verified_dir, s89_dir)
    source_cases = pack['cases']
    reviewed = Counter()
    stratified = {s: Counter() for s in STRATA}
    for case in source_cases:
        cid, fid = case['case_id'], case['source_fragment_id']
        row = by_case[cid]
        s = by_fragment[fid]['review_priority']
        c = stratified[s]
        c['cases'] += 1
        c['reviewer_decisions'] += len(row['votes'])
        c['at_least_one_review'] += bool(row['votes'])
        c['at_least_two_reviews'] += len(row['votes']) >= 2
        c[row['status']] += 1
        reviewed['decisions'] += len(row['votes'])
        reviewed['UNDECIDABLE_votes'] += sum(v['choice'] == 'UNDECIDABLE' for v in row['votes'])
    queue = _followup_queue(pack, by_case, by_fragment, cfg)
    report = {
        'schema': REPORT_SCHEMA,
        'status': 'PRIVATE_IMAGE_APPEARANCE_CONSENSUS_ONLY',
        'input_S90_pack_sha256': sha256_file(pack_dir / 'review_pack.json'),
        'input_S90_verified_summary_sha256': sha256_file(verified_dir / 'review_summary.json'),
        'input_S90_case_votes_sha256': sha256_file(verified_dir / 'case_votes.ndjson'),
        'input_S89_report_sha256': sha256_file(s89_dir / 'report.json'),
        'input_S89_audit_sha256': sha256_file(s89_dir / 'candidate_audit.ndjson'),
        'source_BMP_sha256': pack['source_BMP_sha256'],
        'unique_reviewer_count': verified['unique_reviewer_count'],
        'selected_candidate_count': len(source_cases),
        'reviewer_decision_count': reviewed['decisions'],
        'undecidable_decision_count': reviewed['UNDECIDABLE_votes'],
        'cases_reviewed_by_at_least_two': verified['cases_reviewed_by_at_least_two'],
        'image_appearance_state_counts': verified['review_state_counts'],
        'selected_candidate_strata': {k: dict(sorted(v.items())) for k, v in stratified.items()},
        'pairwise_agreement': pairwise_agreement(by_case),
        'unresolved_selected_case_count': sum(by_case[c['case_id']]['status'] in REVIEW_STATES[:2]
                                              for c in source_cases),
        'followup_queue_selected_count': len(queue),
        'followup_pack_available': bool(queue) and cfg.build_blinded_followup,
        'sampling_bias_warning': 'S89_TILE_SAMPLING_AND_S90_FLAG_BALANCING_NOT_RANDOM_PHYSICAL_RUN_SAMPLE',
        'no_representative_physical_precision_or_recall': True,
        'native_physics_track_GT_count': 0,
        'independent_physical_run_GT_count': 0,
        'physical_auto_join_or_reject_authorized': False,
        'no_training_labels_produced': True,
        'original_masks_and_votes_unchanged': True,
        'next_round_instruction': 'Use original S90 pack for --verify with ALL distinct reviewer JSON responses.',
    }
    # Outputs are created only AFTER provenance, vote and image checks pass.
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    _write_once(out / 'report.json', _json_bytes(report))
    _write_once(out / 'followup_queue_private.json', _json_bytes({
        'schema': QUEUE_SCHEMA, 'source_S90_pack_sha256': report['input_S90_pack_sha256'],
        'source_S90_verified_summary_sha256': report['input_S90_verified_summary_sha256'],
        'selection_policy': 'DISAGREEMENT_THEN_INCOMPLETE_HASH_ORDER_ONLY_NO_GT',
        'case_count': len(queue), 'cases': queue,
        'must_not_be_shown_to_blinded_reviewers': True,
    }))
    if cfg.build_blinded_followup and queue:
        followup = out / 'followup_blinded'
        images = followup / 'images'
        images.mkdir(parents=True, exist_ok=False)
        followup.chmod(0o700)
        images.chmod(0o700)
        selected = {x['case_id'] for x in queue}
        cases = [c for c in source_cases if c['case_id'] in selected]
        # Independent reviewer sees exactly the same pixels as in S90, with no
        # prior-vote counts, S89 flag, status, or rater name in HTML or images.
        for case in cases:
            for key in ('original_preview', 'overlay_preview'):
                rel = case[key]
                _write_once(followup / rel, (pack_dir / rel).read_bytes())
        page = _private_html(cases, report['input_S90_pack_sha256']).replace(
            b'PRIVATE S90:', b'PRIVATE S91:').replace(
            b'placeholder="physicist_01"', b'placeholder="new_reviewer_id"')
        _write_once(followup / 'index.html', page)
    return report
