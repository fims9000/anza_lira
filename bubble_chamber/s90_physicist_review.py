"""S90 private, blinded, provenance-locked visual review of machine line proposals.

Machine candidates are UNKNOWN. Human judgments concern *image appearance* only;
review agreement does not establish physical particle identity, native GT masks,
or calibrated track detection accuracy. Original S86/S88 masks are immutable.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

import cv2
import numpy as np

from .bitmap_tiles import GrayBMP8, sha256_file
from .s87_seam_continuity import _read_source
from .s89_private_candidate_audit import SCHEMA as S89_SCHEMA

PACK_SCHEMA = 'ANZA_LIRA_S90_PRIVATE_BLIND_REVIEW_PACK_V1'
RESPONSE_SCHEMA = 'ANZA_LIRA_S90_IMAGE_APPEARANCE_REVIEW_V1'
SUMMARY_SCHEMA = 'ANZA_LIRA_S90_PRIVATE_REVIEW_SUMMARY_V1'
CHOICES = ('TRACK_LIKE', 'ARTIFACT_LIKE', 'UNDECIDABLE')
CASE_RE = re.compile(r'^c_[0-9a-f]{20}$')
REVIEWER_RE = re.compile(r'^[A-Za-z0-9_.-]{1,64}$')


@dataclass(frozen=True)
class ReviewConfig:
    max_cases: int = 24
    max_per_tile: int = 5
    crop_size: int = 192

    def validate(self) -> None:
        if not (1 <= self.max_cases <= 200 and 1 <= self.max_per_tile <= 50):
            raise ValueError('max_cases and max_per_tile must be positive and bounded')
        if self.crop_size < 96 or self.crop_size > 512 or self.crop_size % 16:
            raise ValueError('crop_size must be 96..512 and a multiple of 16')


def _safe_private_destination(directory: Path, *sources: Path) -> Path:
    target = Path(directory).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if target == repo_root or repo_root in target.parents:
        raise ValueError('PRIVATE output cannot be inside public source repository')
    for original in sources:
        source = Path(original).resolve()
        if target == source or source in target.parents or target in source.parents:
            raise ValueError('PRIVATE output cannot overwrite or contain source directories')
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('PRIVATE output must be a new empty directory')
    return target


def _write_once(path: Path, data: bytes) -> None:
    # No replacement of a review, mask, source, or existing snapshot.
    with path.open('xb') as handle:
        handle.write(data)
    path.chmod(0o600)


def _json_bytes(data: dict | list) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def _read_ndjson(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_candidate_rows(rows: list[dict], source_sha: str,
                          cfg: ReviewConfig) -> list[dict]:
    """Deterministic stratification by S89 flag and source tile, not track GT.

    Blind order is independently source-hash shuffled. The source pixel layout
    and S89 diagnostic rank cannot be recovered from display colors or labels.
    """
    cfg.validate()
    unique = set()
    groups: dict[str, list[dict]] = {'REVIEW_FIRST': [], 'REVIEW_LATER_NOT_VERIFIED': []}
    for row in rows:
        fid = row.get('fragment_id')
        if not isinstance(fid, str) or fid in unique:
            raise ValueError('missing/duplicate source fragment identity')
        unique.add(fid)
        category = row.get('review_priority')
        if row.get('status') != 'UNKNOWN' or category not in groups:
            raise ValueError('invalid S89 review-only semantics')
        groups[category].append(row)
    for category in groups:
        groups[category].sort(key=lambda row: sha256((source_sha + ':select:'
                                                      + row['fragment_id']).encode()).hexdigest())
    selected: list[dict] = []
    counts = Counter()
    while len(selected) < cfg.max_cases:
        made_progress = False
        for category in groups:
            remaining = groups[category]
            while remaining:
                candidate = remaining.pop(0)
                tid = candidate['tile_id']
                if counts[tid] >= cfg.max_per_tile:
                    continue
                selected.append(candidate)
                counts[tid] += 1
                made_progress = True
                break
            if len(selected) >= cfg.max_cases:
                break
        if not made_progress:
            break
    return sorted(selected, key=lambda row: sha256((source_sha + ':blind-order:'
                                                    + row['fragment_id']).encode()).hexdigest())


def _crop_around_fragment(bmp: GrayBMP8, original_xy: np.ndarray,
                          size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if original_xy.ndim != 2 or original_xy.shape[1] != 2 or len(original_xy) < 4:
        raise ValueError('invalid original-pixel centerline')
    if not np.isfinite(original_xy).all():
        raise ValueError('non-finite source coordinates')
    center = original_xy[len(original_xy) // 2]
    w, h = min(bmp.width, size), min(bmp.height, size)
    x0 = max(0, min(int(round(float(center[0]))) - w // 2, bmp.width - w))
    y0 = max(0, min(int(round(float(center[1]))) - h // 2, bmp.height - h))
    crop = np.asarray(bmp._data[y0:y0 + h, x0:x0 + w], dtype=np.uint8).copy()
    return crop, (x0, y0, x0 + w, y0 + h)


def _png_bytes(pixels: np.ndarray) -> bytes:
    success, buffer = cv2.imencode('.png', pixels)
    if not success:
        raise OSError('cannot encode private preview')
    return buffer.tobytes()


def _render_case(bmp: GrayBMP8, fragment: dict, crop_size: int) -> tuple[bytes, bytes, list[int]]:
    points = np.asarray(fragment['centerline_xy_source_px'], dtype=np.float64)
    raw, rect = _crop_around_fragment(bmp, points, crop_size)
    overlay = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    # One neutral color for all classes: reviewers must not see S89 priorities.
    local = np.rint(points - np.asarray(rect[:2])).astype(np.int32)
    cv2.polylines(overlay, [local.reshape(-1, 1, 2)], False,
                  (255, 210, 35), 2, cv2.LINE_AA)
    for point in (local[0], local[-1]):
        if 0 <= point[0] < raw.shape[1] and 0 <= point[1] < raw.shape[0]:
            cv2.circle(overlay, tuple(map(int, point)), 3, (255, 210, 35), 1)
    return _png_bytes(raw), _png_bytes(overlay), list(rect)


def _private_html(cases: list[dict], pack_sha: str) -> bytes:
    safe_cases = [{'id': c['case_id'], 'original': c['original_preview'],
                   'overlay': c['overlay_preview']} for c in cases]
    payload = json.dumps(safe_cases, ensure_ascii=True).replace('<', '\\u003c')
    # This is a static offline HTML program. It neither uploads data nor uses
    # browser storage. Explicit JSON download is required to persist decisions.
    body = '''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>PRIVATE S90 — экспертный просмотр кандидатов</title><style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 15px;background:#161d26;color:#f4f6f8}
h1{font-size:1.5rem} .pictures{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.pictures img{width:100%;image-rendering:auto;background:#050608} .choices{display:flex;gap:8px;flex-wrap:wrap;margin:15px 0}
button,input,textarea{font:inherit;padding:9px;border-radius:5px} button{background:#334b60;color:#fff;border:1px solid #6b8698;cursor:pointer}
button.selected{outline:3px solid #e9dfb0} textarea{width:95%;display:block;min-height:45px}
label{display:block;margin:10px 0} .small{font-size:.9rem;color:#b9c5d2}
@media(max-width:700px){.pictures{grid-template-columns:1fr}}
</style></head><body>
<h1>PRIVATE S90: визуальная оценка фрагментов</h1>
<p class="small">Одинаковое оформление для всех примеров. Слева исходные пиксели, справа выделение фрагмента.
Это НЕ подтвержденные физические треки, НЕ полные маски, НЕ 3D и НЕ калиброванная ионизация.
Не публикуйте фотографии или результаты. Решения сохраняются только после скачивания JSON.</p>
<p id="progress" aria-live="polite"></p><div class="pictures">
<figure><img id="original" alt="Оригинальный фрагмент BMP"><figcaption>Исходный пиксельный участок</figcaption></figure>
<figure><img id="overlay" alt="Машинный кандидат"><figcaption>Машинная линия UNKNOWN</figcaption></figure>
</div>
<div class="choices">
<button type="button" data-choice="TRACK_LIKE">1 — Похоже на трек</button>
<button type="button" data-choice="ARTIFACT_LIKE">2 — Похоже на артефакт</button>
<button type="button" data-choice="UNDECIDABLE">3 — Не определить</button>
</div>
<label>Короткое замечание (необязательно, до 300 символов)
<textarea id="note" maxlength="300" placeholder="Например: пересечение / размытый участок"></textarea></label>
<div class="choices"><button id="previous">← Назад</button><button id="next">Далее →</button></div>
<hr><label>Идентификатор рецензента (латиница/цифры/._-, без ФИО или email)
<input id="reviewer" maxlength="64" pattern="[A-Za-z0-9_.-]+" placeholder="physicist_01"></label>
<button id="export">Скачать PRIVATЕ JSON с ответами</button>
<p class="small" id="status">Программа не отправляет данные в сеть и не сохраняет их в браузере.</p>
<script>
'use strict';
const cases = __CASES__;
const packHash = '__HASH__';
const votes = new Map();
let index = 0;
const byId = id => document.getElementById(id);
function keepNote() {
  const id = cases[index].id;
  if (votes.has(id)) votes.get(id).note = byId('note').value.slice(0,300);
}
function render() {
  const c = cases[index];
  byId('original').src = c.original;
  byId('overlay').src = c.overlay;
  const v = votes.get(c.id);
  byId('note').value = v ? v.note : '';
  const done = votes.size;
  byId('progress').textContent = `Кандидат ${index+1}/${cases.length}; оценено ${done}/${cases.length}`;
  document.querySelectorAll('[data-choice]').forEach(b =>
    b.classList.toggle('selected', !!v && v.choice === b.dataset.choice));
  byId('previous').disabled = index === 0;
  byId('next').disabled = index === cases.length - 1;
}
function move(delta){keepNote();index=Math.max(0,Math.min(cases.length-1,index+delta));render();}
document.querySelectorAll('[data-choice]').forEach(b => b.addEventListener('click', () => {
  const c = cases[index];
  votes.set(c.id, {choice:b.dataset.choice,note:byId('note').value.slice(0,300)});
  render();
}));
byId('previous').addEventListener('click',()=>move(-1));
byId('next').addEventListener('click',()=>move(1));
byId('note').addEventListener('input',keepNote);
byId('export').addEventListener('click', () => {
  keepNote();
  const reviewer = byId('reviewer').value.trim();
  if (!/^[A-Za-z0-9_.-]{1,64}$/.test(reviewer)) {
    byId('status').textContent = 'Введите короткий неперсональный ID рецензента.'; return;
  }
  const decisions = cases.filter(c=>votes.has(c.id)).map(c=>({case_id:c.id,...votes.get(c.id)}));
  if (!decisions.length) {byId('status').textContent = 'Сначала оцените хотя бы один кандидат.';return;}
  const payload = {schema:'ANZA_LIRA_S90_IMAGE_APPEARANCE_REVIEW_V1',
    review_pack_sha256:packHash,reviewer_id:reviewer,decisions};
  const bytes = new Blob([JSON.stringify(payload,null,2)+'\\n'],{type:'application/json'});
  const url = URL.createObjectURL(bytes);
  const a = document.createElement('a'); a.href = url;
  a.download = 'PRIVATE_S90_'+reviewer+'_decisions.json'; document.body.appendChild(a);
  a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1500);
  byId('status').textContent = 'Файл скачан локально. Не публикуйте его в GitHub.';
});
render();
</script></body></html>'''
    return body.replace('__CASES__', payload).replace('__HASH__', pack_sha).encode('utf-8')


def build_review_pack(s86_dir: Path, s89_dir: Path, bmp_path: Path, out_dir: Path,
                      cfg: ReviewConfig = ReviewConfig()) -> dict:
    cfg.validate()
    s86_dir, s89_dir, bmp_path = (Path(p).resolve() for p in (s86_dir, s89_dir, bmp_path))
    target = _safe_private_destination(out_dir, s86_dir, s89_dir, bmp_path)
    source, records, manifest_sha = _read_source(s86_dir)
    s89_report_path = s89_dir / 'report.json'
    audit_path = s89_dir / 'candidate_audit.ndjson'
    report_sha = sha256_file(s89_report_path)
    audit_sha = sha256_file(audit_path)
    report = json.loads(s89_report_path.read_text(encoding='utf-8'))
    if (report.get('schema') != S89_SCHEMA or report.get('status') != 'COMPLETE_MACHINE_ONLY'
            or report.get('input_s86_manifest_sha256') != manifest_sha
            or report.get('original_BMP_sha256') != source['source_bmp_sha256']
            or report.get('physics_track_GT_count') != 0
            or report.get('physical_auto_joins_authorized') is not False):
        raise ValueError('S89 source provenance is not a complete machine-only audit of S86')
    if sha256_file(bmp_path) != source['source_bmp_sha256']:
        raise ValueError('original BMP SHA256 does not match the S86/S89 provenance')
    bmp = GrayBMP8(bmp_path)
    if [bmp.height, bmp.width] != source['image_shape_yx']:
        raise ValueError('BMP dimensions mismatch')
    by_fragment = {}
    for record in records:
        for fragment in record['fragments']:
            fid = fragment['fragment_id']
            if fid in by_fragment:
                raise ValueError('duplicate S86 fragment identity')
            by_fragment[fid] = fragment
    rows = _read_ndjson(audit_path)
    if len(rows) != report['sampled_fragment_count']:
        raise ValueError('S89 candidate audit count mismatch')
    for row in rows:
        fid = row.get('fragment_id')
        if fid not in by_fragment or row.get('tile_id') != fid.split('_f')[0]:
            raise ValueError('S89 candidate has no matching observed S86 fragment')
        if row.get('physical_auto_accept') is not False or row.get('physical_auto_reject') is not False:
            raise ValueError('cannot review auto-accepted or auto-rejected physical identities')
    chosen = select_candidate_rows(rows, source['source_bmp_sha256'], cfg)
    if not chosen:
        raise ValueError('S89 sampled no fragments; no expert review pack can be made')
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(0o700)
    image_dir = target / 'images'
    image_dir.mkdir(mode=0o700, exist_ok=False)
    cases, selection = [], []
    for row in chosen:
        fid = row['fragment_id']
        case_id = 'c_' + sha256((source['source_bmp_sha256'] + ':S90:' + fid).encode()).hexdigest()[:20]
        original, overlay, rect = _render_case(bmp, by_fragment[fid], cfg.crop_size)
        original_path = f'images/{case_id}_original.png'
        overlay_path = f'images/{case_id}_overlay.png'
        _write_once(target / original_path, original)
        _write_once(target / overlay_path, overlay)
        cases.append({'case_id': case_id, 'original_preview': original_path,
                      'overlay_preview': overlay_path,
                      'original_preview_sha256': sha256(original).hexdigest(),
                      'overlay_preview_sha256': sha256(overlay).hexdigest(),
                      'source_fragment_id': fid, 'crop_xyxy_source_px': rect,
                      'question': 'IMAGE_APPEARANCE_ONLY_NOT_PARTICLE_ID'})
        selection.append({'case_id': case_id, 's89_review_priority': row['review_priority'],
                          's89_review_flags': row['review_flags']})
    # Detect any race/change of the untrusted raw inputs before completing pack.
    if (sha256_file(bmp_path) != source['source_bmp_sha256']
            or sha256_file(s86_dir / 'manifest.json') != manifest_sha
            or sha256_file(s89_report_path) != report_sha
            or sha256_file(audit_path) != audit_sha):
        raise ValueError('S86 or original BMP changed during review pack generation')
    pack = {'schema': PACK_SCHEMA, 'status': 'PRIVATE_BLINDED_MACHINE_PROPOSAL_REVIEW',
            'config': asdict(cfg), 'cases': cases, 'case_count': len(cases),
            'source_S86_manifest_sha256': manifest_sha,
            'source_S89_report_sha256': report_sha,
            'source_S89_audit_sha256': audit_sha,
            'source_BMP_sha256': source['source_bmp_sha256'],
            'selection': 'S89_FLAG_BALANCED_TILE_CAPPED_HASH_ORDER_NO_LABELS',
            'S89_review_flags_hidden_from_review_UI': True,
            'native_track_GT_count': 0, 'physical_auto_joins_authorized': False,
            'does_not_change_detector_masks': True,
            'image_space_reviews_not_physical_truth': True}
    pack_bytes = _json_bytes(pack)
    pack_sha = sha256(pack_bytes).hexdigest()
    _write_once(target / 'review_pack.json', pack_bytes)
    _write_once(target / 'selection_private.json', _json_bytes(selection))
    _write_once(target / 'index.html', _private_html(cases, pack_sha))
    return {'schema': PACK_SCHEMA, 'status': pack['status'],
            'case_count': len(cases), 'review_pack_sha256': pack_sha,
            'physical_track_GT_count': 0, 'physical_auto_joins_authorized': False,
            'private_output': str(target)}


def _read_response(file_path: Path, pack_sha: str, known_ids: set[str]) -> dict:
    file_path = Path(file_path)
    if file_path.stat().st_size > 2_000_000:
        raise ValueError('review JSON too large')
    content = json.loads(file_path.read_text(encoding='utf-8'))
    if (not isinstance(content, dict) or set(content) != {
            'schema', 'review_pack_sha256', 'reviewer_id', 'decisions'}
            or content['schema'] != RESPONSE_SCHEMA
            or content['review_pack_sha256'] != pack_sha):
        raise ValueError('review schema or pack SHA256 does not match this private review session')
    reviewer = content['reviewer_id']
    if not isinstance(reviewer, str) or REVIEWER_RE.fullmatch(reviewer) is None:
        raise ValueError('reviewer_id must be an opaque 1..64-char identifier')
    decisions = content['decisions']
    if not isinstance(decisions, list) or not 1 <= len(decisions) <= len(known_ids):
        raise ValueError('review decision count invalid')
    seen = set()
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != {'case_id', 'choice', 'note'}:
            raise ValueError('invalid response row schema')
        case_id, choice, note = decision['case_id'], decision['choice'], decision['note']
        if (not isinstance(case_id, str) or case_id not in known_ids or case_id in seen
                or not isinstance(choice, str) or choice not in CHOICES
                or not isinstance(note, str) or len(note) > 300):
            raise ValueError('duplicate/unknown case, invalid decision, or overlong note')
        seen.add(case_id)
    return content


def verify_reviews(pack_dir: Path, response_paths: list[Path], out_dir: Path) -> dict:
    pack_dir = Path(pack_dir).resolve()
    target = _safe_private_destination(out_dir, pack_dir, *response_paths)
    path = pack_dir / 'review_pack.json'
    pack = json.loads(path.read_text(encoding='utf-8'))
    if (pack.get('schema') != PACK_SCHEMA or pack.get('status') != 'PRIVATE_BLINDED_MACHINE_PROPOSAL_REVIEW'
            or pack.get('native_track_GT_count') != 0
            or pack.get('physical_auto_joins_authorized') is not False
            or pack.get('case_count') != len(pack.get('cases', []))):
        raise ValueError('untrusted review pack schema or physical semantics')
    cases = pack['cases']
    known_ids = set()
    for case in cases:
        cid = case.get('case_id')
        if not isinstance(cid, str) or not CASE_RE.fullmatch(cid) or cid in known_ids:
            raise ValueError('duplicate or malformed case ID in review pack')
        known_ids.add(cid)
        for key, suffix in (('original_preview', '_original.png'),
                            ('overlay_preview', '_overlay.png')):
            rel = f'images/{cid}{suffix}'
            if case.get(key) != rel or sha256_file(pack_dir / rel) != case[key + '_sha256']:
                raise ValueError('modified or missing private preview image')
    if not response_paths:
        raise ValueError('at least one independently exported review JSON is required')
    digest = sha256_file(path)
    reviews = [_read_response(p, digest, known_ids) for p in response_paths]
    reviewers = [r['reviewer_id'] for r in reviews]
    if len(set(reviewers)) != len(reviewers):
        raise ValueError('duplicate reviewer_id; multiple exports by one reviewer are not independent')
    by_case = defaultdict(list)
    for r in reviews:
        for d in r['decisions']:
            by_case[d['case_id']].append({'reviewer_id': r['reviewer_id'],
                                         'choice': d['choice'], 'note': d['note']})
    per_case = []
    states = Counter()
    for c in cases:
        cid = c['case_id']
        votes = by_case.get(cid, [])
        categorical = [v['choice'] for v in votes]
        if len(votes) < 2:
            status = 'INSUFFICIENT_INDEPENDENT_REVIEWS'
        elif all(v == 'TRACK_LIKE' for v in categorical):
            status = 'IMAGE_APPEARANCE_TRACK_LIKE_AGREEMENT_NOT_GT'
        elif all(v == 'ARTIFACT_LIKE' for v in categorical):
            status = 'IMAGE_APPEARANCE_ARTIFACT_LIKE_AGREEMENT_NOT_GT'
        else:
            status = 'DISAGREEMENT_OR_UNDECIDABLE_REVIEW'
        states[status] += 1
        per_case.append({'case_id': cid, 'source_fragment_id': c['source_fragment_id'],
                         'status': status, 'votes': votes, 'native_track_GT': False,
                         'physical_auto_reject': False, 'physical_auto_accept': False})
    summary = {'schema': SUMMARY_SCHEMA, 'review_pack_sha256': digest,
               'source_BMP_sha256': pack['source_BMP_sha256'],
               'source_S86_manifest_sha256': pack['source_S86_manifest_sha256'],
               'source_S89_report_sha256': pack['source_S89_report_sha256'],
               'unique_reviewer_count': len(reviewers),
               'selected_candidate_count': len(cases),
               'cases_reviewed_by_at_least_one': sum(bool(x['votes']) for x in per_case),
               'cases_reviewed_by_at_least_two': sum(len(x['votes']) >= 2 for x in per_case),
               'review_state_counts': dict(sorted(states.items())),
               'native_physics_track_GT_count': 0, 'physical_track_accuracy_measured': False,
               'physical_auto_joins_authorized': False,
               'no_training_labels_or_detector_mask_modifications': True,
               'result_interpretation': 'IMAGE_APPEARANCE_ONLY_REVIEW_NOT_PHYSICAL_GROUND_TRUTH'}
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(0o700)
    _write_once(target / 'review_summary.json', _json_bytes(summary))
    _write_once(target / 'case_votes.ndjson',
                ''.join(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n'
                        for x in per_case).encode('utf-8'))
    return summary
