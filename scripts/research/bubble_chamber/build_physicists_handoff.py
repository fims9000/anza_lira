#!/usr/bin/env python3
"""Produce an auditable, strictly public-safe physics handoff from frozen JSON.

No H50Z photographs, operator measurement coordinates, weights or model output
polylines are exported. All illustrations are synthetic *schematics*, never
H50Z evidence. The script deliberately fails if historical counters drift.

Usage:
    python scripts/research/bubble_chamber/build_physicists_handoff.py --write
    python scripts/research/bubble_chamber/build_physicists_handoff.py --check
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / 'docs/bubble_chamber'
OUT = DOCS / 'physicists'
SRC = {
    's39': 'S39_PUBLIC_RESULTS.json',
    's56': 'S56_PUBLIC_GLOBAL_PATH_RESULTS.json',
    's58': 'S58_PUBLIC_MACHINE_PATH_COMPONENTS.json',
    's72': 'S72_PUBLIC_FRESH_TEST_2026_10_26.json',
    's84': 'S84_PUBLIC_FRESH_TEST_2026_10_26.json',
    's28': 's27_s29_public_metrics.json',
    's33': 's33_s34_public_metrics.json',
    's85': 'S85_INTEROP_CALIBRATED_GENERATED_TOY_RESULTS_2026_10_28.json',
}
REPO_URL = 'https://github.com/fims9000/anza_lira/blob/research/bubble-chamber-graph-lira/docs/bubble_chamber/'
LEDGER_NAME = 'results_manifest.json'
TITLE = 'ANZA–LIRA / машинные гипотезы треков для физиков'


def require(condition: bool, msg: str) -> None:
    if not condition:
        raise ValueError('PHYSICS HANDOFF SOURCE MISMATCH: ' + msg)


def _sum_arms(s56, arm, field):
    return sum(r['arms'][arm][field] for d in s56['directions'].values() for r in d['tests'].values())


def _sum_s72_selected(s72, field):
    return sum(r['arms'][d['CAL_selected']][field] for d in s72['directions'].values() for r in d['TEST_seeds'].values())


def load_results(root=ROOT):
    source_dir = root/'docs/bubble_chamber'
    data = {}
    hashes = {}
    for key, relative in SRC.items():
        p = source_dir/relative
        if not p.is_file():
            raise FileNotFoundError('Frozen, source-only research record missing: '+str(p))
        raw = p.read_bytes()
        hashes[relative] = sha256(raw).hexdigest()
        data[key] = json.loads(raw)

    s39, s56, s58 = (data[key] for key in ('s39','s56','s58'))
    s72, s84, s28 = (data[key] for key in ('s72','s84','s28'))
    s33, s85 = (data[key] for key in ('s33','s85'))
    real = s39['real_H50Z_selected_machine_candidates']
    spatial = s39['unbiased_full_BMP_machine_audit']['results']
    graph = s39['native_graph_hypothesis_ranking']['totals']
    path = s56['native_machine_advisory']['counts']
    require(real['genuine_H50Z_photos']==2 and real['independent_physical_runs']==1,'only two original H50Z photographs, one physical run')
    require(real['native_physics_GT_labels']==0 and not real['automatic_physical_joins_authorized'], 'zero native physics labels/auto-joins')
    require(real['original_line_proposals']+real['additional_new_ridge_proposals']==real['all_machine_proposals']==567, '567 machine line *proposals*, not verified tracks')
    require(real['original_empty_crops_with_new_ridge_proposals']==real['original_empty_crops']==17,'all 17 formerly empty selected crops now have candidates')
    require(spatial['tiles']==128 and spatial['tiles_with_any_proposal']==127 and spatial['novel_proposals']==1166,'unbiased spatial machine-audit counters')
    require(graph['graph_nodes']==511 and graph['possible_endpoint_links']==182,'native candidates 511 nodes /182 edges')
    require(path['advisory_selected_edges']==37 and path['possible_edges_scored']==graph['possible_endpoint_links'],'native provisional 37/182')
    require(s58['input_machine_lines']==real['all_machine_proposals'] and s58['input_graph_nodes_after_nms']==graph['graph_nodes'],'S58 shares exact S39 source')
    require(s58['S56_selected_machine_only_joins']==path['advisory_selected_edges'],'native S56/S58 joins')
    require(s58['connected_machine_path_hypotheses']==35 and s58['component_counts_by_fragment_count']=={'2':34,'4':1},'native components')
    require(s58['confirmed_native_physical_tracks']==0 and s58['confirmed_native_physical_joins']==0 and s58['predicted_bridge_pixels_not_observed'],'native chains are hypotheses, not observed physics')
    require(s56['native_verified_joins']==0 and s56['native_verified_trajectories']==0 and s56['synthetic_GT_not_native_physics'],'S56 native GT unavailable')
    require(s56['source_frame_split_counts']=={'train':79,'cal':30,'test':36},'visualMetro frame splits')
    require(s56['operator_source_frames_complete']==145 and s56['source_visualmetro_does_not_match_H50Z_images'],'visualMetro not native H50Z GT')
    require(sum(s56['protocol']['effective']['test_per_seed']*len(v['tests']) for v in s56['directions'].values())==256,'S56 complete sim scene count')
    s56_summary = {arm:{'correct':_sum_arms(s56,arm,'true'),'other':_sum_arms(s56,arm,'wrong'),
        'full_chains':_sum_arms(s56,arm,'full_trajectories')}
        for arm in ('node_mutual','port_greedy','global_forest')}
    require(s56_summary=={'node_mutual':{'correct':88,'other':15,'full_chains':0},
        'port_greedy':{'correct':441,'other':36,'full_chains':129},
        'global_forest':{'correct':246,'other':13,'full_chains':66}},'S56 synthetic controls')

    require(s72['physical_verified_H50Z_GT']==0 and not s72['physical_auto_join_authorized'] and s72['all_metrics_are_SYNTHETIC_ONLY'],'S72 synthetic-only output')
    base=s72['all_TEST_arms']['s39_original']
    chosen={k:_sum_s72_selected(s72,k) for k in ('selected_synthetic_correct_links','synthetic_noncorrect_selected','complete_three_piece_paths_synthetic')}
    require([base[k] for k in chosen]==[56,52,12] and list(chosen.values())==[56,41,13], 'S72 held-out baseline vs selected')
    require(set(d['CAL_selected'] for d in s72['directions'].values())=={'s39_original','s68_branch_endpoint_refine'},'S72 frozen CAL arms')
    require(s84['physical_GT_count']==0 and not s84['physical_auto_joins_authorized'] and s84['synthetic_only_metrics'], 'S84 synthetic-only')
    s84_agg={}
    for condition in ('clean','blur_sigma_11','blur_sigma_16'):
        old=s84['all_TEST_aggregates'][condition]['baseline']
        new=s84['frozen_CAL_selected_TEST'][condition]
        require(old['scenes']==new['scenes']==96,'S84 paired TEST scene count')
        s84_agg[condition]={
          'baseline_correct':old['selected_synthetic_correct_links'],
          'selected_correct':new['selected_synthetic_correct_links'],
          'baseline_other':old['synthetic_noncorrect_selected'],
          'selected_other':new['synthetic_noncorrect_selected'],
          'baseline_complete':old['complete_three_piece_paths_synthetic'],
          'selected_complete':new['complete_three_piece_paths_synthetic']}
    require(s84_agg['clean']=={'baseline_correct':43,'selected_correct':45,'baseline_other':35,'selected_other':42,'baseline_complete':10,'selected_complete':11}, 'S84 CAL-selected clean negative transfer')
    require(s84_agg['blur_sigma_11']['selected_other']==24 and s84_agg['blur_sigma_11']['baseline_other']==19, 'S84 blurred risky extra links')
    require(s85['mode']=='GENERATED_ONLY_NOT_H50Z' and s85['physical_GT_count']==0 and not s85['physical_auto_joins_authorized'],'S85 generated-only, no expert GT')
    require(len(s85['detectors'])==3 and all(s85['models'][d]['trained'] and s85['CAL_selected'][d]=='none' for d in s85['detectors']), 'S85 detector-specific CAL all abstain')
    anchor=s28['S28_injected_known_anchor_prefix_recovery']
    s28_summary={name:{'fraction':round(a['top1']['mean_injected_prefix_coverage'],8),
                      'successes':a['top1']['successful_at_70pct_prefix_coverage_count'],
                      'samples':a['top1']['n']}
      for name,a in anchor.items()}
    require(s28_summary['curvature_constrained_evidence']=={'fraction':0.96666667,'successes':16,'samples':16},'S28 synthetic curve continuation')
    require(s28_summary['original_AZ_plus_curvature']['successes']==15 and s28_summary['straight_evidence']['successes']==0,'S28 ANZA and straight baseline')
    require(s28['real_expert_contour_track_masks']==0 and s28['real_operator_start_points']==9 and s28['S29_real_photo_candidate_provisional_measurements']==72,'S28 real starts but unknown curves')
    gap=s28['S29_anchor_offset_plus_erased_synthetic_gaps']['gap1_offset0_top1']
    require(gap['n']==8 and gap['successes_70pct']==7,'S29 synthetic gap case')
    require(s33['real_expert_confirmed_track_masks']==0 and s33['training_machine_marks_only'] and not s33['physics_autonomous_join_permitted'],'S33 machine weak masks only')
    s33_summary={}
    for d in ('005_train_006_test','006_train_005_test'):
        m=s33['directions'][d]['methods']
        s33_summary[d]={k:round(m[k]['mean_AUROC'],6) for k in ('azunet_raw','azunet_quantile','unet_local')}

    summary={
      'schema':'ANZA_LIRA_PHYSICISTS_PUBLIC_EVIDENCE_V1',
      'updated_on':'2026-10-29',
      'source_branch':'research/bubble-chamber-graph-lira',
      'source_file_sha256':hashes,
      'original_photographs':2,'physical_runs':1,'H50Z_selected_512px_crops':39,
      'H50Z_selected_crops_photo_005':21,'H50Z_selected_crops_photo_006':18,
      'visualMetro_film_070_frames':145,
      'visualMetro_train_cal_test_whole_frames':s56['source_frame_split_counts'],
      'visualMetro_is_NOT_H50Z_photo_GT':True,
      'real_H50Z_machine_only':{
          'old_line_proposals':real['original_line_proposals'],
          'additional_ridge_proposals':real['additional_new_ridge_proposals'],
          'total_line_proposals_not_tracks':real['all_machine_proposals'],
          'selected_crops_filled_with_new_candidates':real['original_empty_crops_with_new_ridge_proposals'],
          'sampled_nonselect_full_frame_tiles':spatial['tiles'],
          'full_frame_tiles_with_machine_proposals':spatial['tiles_with_any_proposal'],
          'full_frame_line_proposals':spatial['novel_proposals'],
          'after_NMS_graph_nodes':graph['graph_nodes'],
          'possible_join_proposals':graph['possible_endpoint_links'],
          'selected_join_hypotheses_UNKNOWN':path['advisory_selected_edges'],
          'machine_connected_components':s58['connected_machine_path_hypotheses'],
          'components_of_two_lines':s58['component_counts_by_fragment_count']['2'],
          'components_of_four_lines':s58['component_counts_by_fragment_count']['4'],
          'native_machine_draft_start_hypotheses':s28['S29_real_photo_candidate_provisional_measurements'],
          'native_real_operator_starts':s28['real_operator_start_points'],
          'machine_pseudo_positive_centerline_pixels_not_GT':real['pseudo_positive_pixels_not_gt'],
          'machine_pseudo_negative_neighbor_pixels_not_GT':real['assumed_offset_negative_pixels_not_gt'],
          'machine_unknown_pixels':real['unknown_pixels']},
      'synthetic_injection_only':{
          'S28_known_anchor_curve_coverage':s28_summary,
          'S29_one_erased_gap_0px_anchor':{'n':gap['n'],'fraction':round(gap['mean_prefix_coverage'],8),'successes':gap['successes_70pct']},
          'S33_photo_disjoint_AUROC_of_injected_trace_vs_scratch':s33_summary,
          'S56_oracle_geometry_not_pixel_end_to_end':{'scene_count':256,'potential_true_joins':1024,'potential_full_paths':512,'arms':s56_summary},
          'S72_pixel_to_graph_synthetic_TEST':{'scene_count':96,'S39_original':{
              'correct':base['selected_synthetic_correct_links'],'other':base['synthetic_noncorrect_selected'],
              'full_chains':base['complete_three_piece_paths_synthetic']},
              'frozen_CAL_selected':{'correct':chosen['selected_synthetic_correct_links'],
                'other':chosen['synthetic_noncorrect_selected'],'full_chains':chosen['complete_three_piece_paths_synthetic']},
              'possible_three_piece_chains':192},
          'S84_additional_candidate_CAL_selected_failed_safety':s84_agg,
          'S85_detector_swaps_generated_only':{'trained_detectors':len(s85['detectors']),
             'CAL_selected_per_detector':s85['CAL_selected'],'safe_improvement_proven':False}},
      'expert_verified_H50Z_line_masks':0,
      'expert_verified_H50Z_same_track_pairs':0,
      'expert_verified_H50Z_particle_ids':0,
      'native_GT_count':0,
      'physical_auto_joins_authorized':False,
      'real_native_accuracy_precision_recall_available':False,
      'allowed_use':'read-only machine hypotheses / private expert review, no auto joins',
      'viewer_examples_are':'GENERATED SCHEMATICS, NEVER REAL H50Z OUTPUT',
      'private_original_BMP_or_measured_point_coordinates_in_repo':False,
    }
    return summary


def svg_panel(kind):
    """Deliberately labeled synthetic drawing. Not an output of ANZA on H50Z."""
    require(kind in {'raw','candidates','graph','mask'}, 'unknown schematic panel')
    bg='#131d2b';stroke='#33465c';cyan='#63dcf2';violet='#ce9aff';red='#ff8b8b';green='#6be2b2'
    variants={
      'raw':('Искусственная 2D-сцена','Схематическое изображение: кривые + царапины','Нет вывода модели'),
      'candidates':('Машинные линейные кандидаты','Пиксельный детектор предлагает отрезки','Голубое: предложенные линии; красное: риск артефакта'),
      'graph':('Граф предполагаемых продолжений','Стыки выбираются из конкурирующих концов','Фиолетовый пунктир: UNKNOWN, не наблюдавшийся трек'),
      'mask':('Машинная псевдомаска','Пример кодирования 1 / 0 / 255','255: неизвестно, не отрицательная экспертная разметка'),
    }
    a,b,c=variants[kind]
    body=[]
    if kind!='mask':
        # Fixed geometries across all three views, including visual gaps.
        natural=['M105 318 C160 292 197 287 261 300','M301 304 C357 319 390 324 434 304',
          'M475 284 C541 256 566 263 633 277','M113 397 C187 407 220 388 270 365',
          'M310 350 C360 331 399 335 451 358','M499 378 C556 402 600 395 652 366']
        scratch=['M224 265 L252 421','M588 231 L562 418']
        colors=[stroke,stroke,stroke,stroke,stroke,stroke]
        if kind!='raw':colors=[cyan]*6
        for idx,path in enumerate(natural):
            body.append(f'<path d="{path}" fill="none" stroke="{colors[idx]}" stroke-width="{3.8 if kind!="raw" else 3.2}" stroke-linecap="round"/>')
        for path in scratch:
            body.append(f'<path d="{path}" fill="none" stroke="{red if kind=="candidates" else stroke}" stroke-width="2" opacity="{.8 if kind=="candidates" else .5}"/>')
        if kind in {'candidates','graph'}:
            for x,y in ((261,300),(301,304),(434,304),(475,284),(270,365),(310,350),(451,358),(499,378)):
                body.append(f'<circle cx="{x}" cy="{y}" r="4.8" fill="{cyan}" stroke="#142639" stroke-width="1.5"/>')
            body.append('<circle cx="105" cy="318" r="7" fill="#f6b85c" stroke="#152639" stroke-width="2"/>')
        if kind=='graph':
            body.append(f'<path d="M261 300 Q282 301 301 304 M434 304 Q455 292 475 284 M270 365 Q285 351 310 350" fill="none" stroke="{violet}" stroke-width="4" stroke-dasharray="9 7" stroke-linecap="round"/>')
            body.append(f'<path d="M451 358 Q475 370 499 378" fill="none" stroke="{red}" stroke-width="4" stroke-dasharray="4 9"/>')
            body.append(f'<path d="M434 304 Q472 335 499 378" fill="none" stroke="{red}" stroke-width="2" stroke-dasharray="3 10" opacity=".65"/>')
            body.append(f'<text x="476" y="332" fill="{red}" font-size="15">конкурирующие связи</text>')
    else:
        body.append('<rect x="65" y="229" width="680" height="228" rx="18" fill="#38465a" stroke="#66758b"/>')
        for path in ('M105 318 C170 288 235 293 265 302','M301 304 C367 320 409 319 434 304','M475 284 C540 257 600 259 633 277'):
            body.append(f'<path d="{path}" fill="none" stroke="{green}" stroke-width="10" stroke-linecap="round"/>')
            body.append(f'<path d="{path}" fill="none" stroke="#0e1926" stroke-width="1" stroke-linecap="round"/>')
        for path in ('M115 342 C182 328 225 335 259 343','M322 328 C365 342 394 346 429 330'):
            body.append(f'<path d="{path}" fill="none" stroke="#7591ad" stroke-width="4" opacity=".70"/>')
        body.append(f'<text x="775" y="295" fill="{green}" font-size="17" font-weight="bold">1</text><text x="775" y="325" fill="#9fb6cf" font-size="17">0?</text><text x="775" y="355" fill="{violet}" font-size="17">255</text>')
    texture=''.join(f'<circle cx="{85+(i*47)%645}" cy="{238+(i*67)%200}" r="{1+(i%3)*.35}" fill="#6e879b" opacity=".2"/>' for i in range(120)) if kind=='raw' else ''
    legend={'raw':'Искусственный фотоподобный фон — никаких реальных H50Z пикселей',
            'candidates':'Детектор ещё не знает, какие линии принадлежат частицам',
            'graph':'Ни один пунктирный участок не является измеренным физическим треком',
            'mask':'Условный фон (0) не является экспертным отрицательным классом'}[kind]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 560" role="img" '
       f'aria-label="{html.escape(a)}. Исключительно искусственная схема, не H50Z">'
       '<rect width="960" height="560" fill="#0c1422"/>'
       '<rect x="22" y="18" width="916" height="525" rx="24" fill="#142236" stroke="#334a63"/>'
       f'<text x="64" y="85" fill="#edf3fa" font-size="28" font-weight="bold">{html.escape(a)}</text>'
       f'<text x="64" y="121" fill="#aebdd0" font-size="18">{html.escape(b)}</text>'
       '<rect x="65" y="151" width="830" height="45" rx="12" fill="#402e37" stroke="#bd8793"/>'
       '<text x="90" y="181" fill="#ffcccc" font-size="17" font-weight="bold">'
       'ИСКУССТВЕННАЯ ИЛЛЮСТРАЦИЯ · НЕ ФОТО И НЕ ВЫВОД МОДЕЛИ H50Z</text>'
       '<rect x="65" y="218" width="830" height="245" rx="14" fill="#1f2f42" stroke="#435b73"/>'
       +texture+''.join(body)+
       f'<text x="70" y="500" fill="#c9d8e6" font-size="17">{html.escape(legend)}</text>'
       f'<text x="70" y="526" fill="#849eb5" font-size="14">{html.escape(c)}</text>'
       '</svg>\n')


def _html_view(ledger):
    r=ledger['real_H50Z_machine_only']; s=ledger['synthetic_injection_only']
    title=html.escape(TITLE)
    metrics=[('567','Машинных предложений линий','Не доказанные реальные физические треки'),
          ('182 → 37','Возможные → отобранные связи','Все имеют статус UNKNOWN'),
          ('35','Связанных компонентов','34 пары + одна цепочка из четырёх'),
          ('0','Подтверждений физиков','Нет реальной precision/recall')]
    cards=''.join(f'<article class="metric"><strong>{v}</strong><h3>{html.escape(label)}</h3><p>{html.escape(note)}</p></article>' for v,label,note in metrics)
    s28=s['S28_known_anchor_curve_coverage']; base=s28['curvature_constrained_evidence']; anza=s28['original_AZ_plus_curvature']; straight=s28['straight_evidence']
    def bar(name,value,suffix,klass):
        return (f'<div class="barentry"><span>{html.escape(name)}</span><div class="track"><div class="fill {klass}" style="width:{value:.1f}%"></div></div><b>{value:.1f}{suffix}</b></div>')
    bars=''.join((bar('Прямая',straight['fraction']*100,'%','muted'),bar('Кривизна + фотометрия',base['fraction']*100,'%','cyan'),bar('ANZA + кривизна',anza['fraction']*100,'%','violet')))
    images=''.join(f'<button class="tab" data-panel="{k}" type="button" aria-selected="{str(i==0).lower()}">{x}</button>' for i,(k,x) in enumerate([('raw','Искусственный фон'),('candidates','Кандидаты'),('graph','Стыки UNKNOWN'),('mask','Псевдомаска')]))
    panels=''.join(f'<img id="panel-{k}" class="stage" src="figures/{k}.svg" alt="{x}: схематический рисунок, НЕ реальные H50Z" {'hidden' if k!='raw' else ''}/>' for k,x in [('raw','Исходное изображение'),('candidates','Кандидаты тонких линий'),('graph','Графовые мостики'),('mask','Машинная псевдомаска')])
    return f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · исходники и ограничения</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{font:16px/1.6 system-ui,-apple-system,sans-serif;background:#0c1422;color:#e7eef7;margin:0}}
main{{max-width:1100px;margin:auto;padding:30px 20px 90px}}h1{{font-size:clamp(26px,3.3vw,40px);line-height:1.2}}h2{{font-size:24px;margin-top:36px}}h3{{font-size:16px;margin:0}}
p{{color:#bdcbdc}}a{{color:#8adaeb}}.eyebrow{{letter-spacing:2px;text-transform:uppercase;color:#8fc3d1;font-size:12px}}
.notice{{padding:14px 17px;border:1px solid #70655a;background:#272638;border-radius:12px}}.notice strong{{color:#fdd2aa}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:13px;margin-top:25px}}.metric{{padding:18px;border:1px solid #38506b;border-radius:12px;background:#17263b}}.metric strong{{font-size:32px;color:#a1e9fa}}.metric p{{margin:8px 0 0;font-size:13px}}.tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:15px 0}}button.tab{{border:1px solid #4a627e;border-radius:8px;background:#263a52;color:#d4e5f5;padding:10px 14px;cursor:pointer}}button[aria-selected="true"]{{background:#a4dbea;color:#142033;font-weight:700}}.stage{{display:block;width:100%;max-width:960px;border-radius:12px}}.stage[hidden]{{display:none}}
.barentry{{display:grid;grid-template-columns:minmax(185px,255px) 1fr 65px;gap:12px;align-items:center;margin:12px 0}}.track{{background:#273a52;border-radius:8px;height:15px;overflow:hidden}}.fill{{height:100%;border-radius:8px}}.fill.cyan{{background:#78dcf0}}.fill.violet{{background:#c8a3ff}}.fill.muted{{background:#657489}}.barentry b{{text-align:right}}table{{width:100%;border-collapse:collapse}}td,th{{padding:11px 10px;border-bottom:1px solid #31445c;text-align:left}}small{{color:#9eaec4}}code{{color:#e9c6fa}}@media(max-width:650px){{.barentry{{grid-template-columns:1fr 55px}}.barentry span{{grid-column:span 2}}}}
</style></head><body><main>
<div class="eyebrow">ANZA–LIRA · отчёт для физиков · 29 октября 2026</div><h1>{title}</h1>
<p>Две исходные H50Z-фотографии одного физического запуска, 39 выбранных участков и отдельные измеренные формы visualMetro другой плёнки.</p>
<div class="notice"><strong>Научная граница:</strong> настоящая физическая GT отсутствует. Все соединения исходных H50Z имеют статус UNKNOWN. Иллюстрации ниже <strong>полностью сгенерированы</strong> и не являются фотографиями или выводами модели на H50Z. Автоматическое физическое соединение выключено.</div>
<h2>Результат машинной обработки настоящих H50Z</h2><div class="grid">{cards}</div>
<p>127 из 128 пространственно выбранных участков исходных BMP получили хотя бы один машинный линейный кандидат (1166 предложений). Это охват работы детектора, <strong>не</strong> полнота обнаружения физических частиц.</p>
<h2>Как выглядят типы выходных слоёв</h2><p>Переключайте примеры схемы. Аналогичные слои существуют в приватных демонстраторах S39/S57. <strong>Здесь не использованы исходные пиксели H50Z.</strong></p>
<div class="tabs" role="group" aria-label="Схематические слои">{images}</div><div class="panels">{panels}</div>
<h2>Контролируемое продолжение искусственных треков (S28)</h2><p>16 искусственно внесённых траекторий поверх настоящего H50Z-фона; процент точек, найденных в радиусе 3 пикселей. Это <strong>не</strong> показатель точности на настоящих треках.</p>{bars}
<h2>Графовая сборка, 96 искусственных сцен S72</h2><table><thead><tr><th>Вариант</th><th>Верных стыков</th><th>Прочих</th><th>Целых цепочек / 192</th></tr></thead><tbody>
<tr><td>S39</td><td>56</td><td>52</td><td>12</td></tr><tr><td>CAL-выбранный S72</td><td>56</td><td>41</td><td>13</td></tr></tbody></table>
<p>Алгоритм уменьшил число неправильных либо несопоставленных искусственных соединений, не потеряв правильные, но собрал лишь 13 из 192 потенциальных цепочек.</p>
<h2>Без экспертной разметки</h2><p>Автоматически создаются машинные предложения линий, 2D-маски и кандидаты связности; возможны пропуски, пересечения, царапины и ложные мостики. Система не знает достоверную идентичность частиц. Эксперты могут проверять уже предложенные пары и фрагменты, не обводя весь архив заново.</p>
<p><strong>Что ещё нельзя заявлять:</strong> физическую precision/recall, тип/массу частицы, 3D-траектории, перенос на другую заливку или безопасные автоматические связи. S84 и detector-specific S85 не прошли установленные критерии безопасности.</p>
<p><a href="../PHYSICISTS_RESULTS_2026_10_29.md">Научный отчёт и ссылки на исходные эксперименты</a> · <a href="results_manifest.json">Агрегаты и SHA-256 исходников</a></p>
<small>Источник только: публичные научные агрегаты S27–S85 из этой исследовательской ветки; без raw BMP, visualMetro координат, весов и настоящих приватных скриншотов.</small>
</main><script>
for (const b of document.querySelectorAll('button[data-panel]')) b.addEventListener('click', () => {{
 for(const x of document.querySelectorAll('button[data-panel]')) x.setAttribute('aria-selected', String(x===b));
 for(const x of document.querySelectorAll('.stage')) x.hidden=x.id!==`panel-${{b.dataset.panel}}`;
}});
</script></body></html>\n'''


def markdown_brief(ledger):
    r=ledger['real_H50Z_machine_only'];s=ledger['synthetic_injection_only']
    methods=s['S28_known_anchor_curve_coverage'];s29=s['S29_one_erased_gap_0px_anchor']
    s56=s['S56_oracle_geometry_not_pixel_end_to_end']['arms'];s72=s['S72_pixel_to_graph_synthetic_TEST']
    s84=s['S84_additional_candidate_CAL_selected_failed_safety']
    return f'''# ANZA–LIRA на фотографиях пузырьковой камеры: что уже можно показать физикам

**Срез: 29 октября 2026 · статус: RESEARCH / READ-ONLY · реальные физические соединения не подтверждены.**

> **Главный результат:** построен работающий автоматический конвейер поиска 2D-линейных структур и ранжирования гипотез соединений на **двух настоящих фотографиях H50Z одного запуска**. Числа по исходным H50Z подтверждают **количество машинных предложений**, а не их физическую правильность. На искусственно добавленных линиях поверх H50Z получены отдельные измеряемые результаты. Веса моделей, частные BMP и измеренные координаты **не опубликованы**.

**[Открыть иллюстрированный офлайн-просмотрщик](physicists/demo.html)** · [Схема машинных линий](physicists/figures/candidates.svg) · [Схема графовых соединений](physicists/figures/graph.svg) · [Проверяемые агрегаты JSON](physicists/results_manifest.json)

## 1. С какими данными работали

- **Реальные источники:** `H50Z_030_01_005.bmp` (**9020 × 26151**, 8-bit grayscale), `H50Z_030_01_006.bmp` (**9268 × 25324**). Это **две фотографии одного физического запуска**, а не независимые серии. Для углублённой проверки выбраны **39 кропов 512×512**: 21 из 005 и 18 из 006.
- **Другой материал:** visualMetro, фильм `070`: **145 пригодных кадров** с измеренными оператором формами треков, целые кадры разделены на **79 TRAIN / 30 CAL / 36 TEST**. Его точки не зарегистрированы с настоящими H50Z, поэтому являются источником **форм искусственных тестовых вставок**, но не ground truth физических треков H50Z.
- `labels.json` исходных H50Z содержит операторские `start` и реперные `f1…f8`: **не** целые размеченные треки. У нас **0** экспертных масок, **0** подтверждённых пар `same_track/different_track`, **0** подтверждённых физических соединений.

Источник: [аудит исходных BMP и плёнки](REAL_H50Z_AND_070_DATA_AUDIT_2026_10_24.md), [S39](S39_MACHINE_ONLY_RESEARCH_2026_10_25.md), [S56](S56_S57_MACHINE_GLOBAL_PATH_RESEARCH_2026_10_26.md).

## 2. Что машина уже действительно выделила на настоящих H50Z (без новой разметки)

| Результат | Получено | Что это означает / чего не означает |
|---|---:|---|
| Старые машинные черновики 2D-линий | **{r['old_line_proposals']}** | Неподтверждённые фрагменты |
| Дополнительные хребтовые кандидаты | **{r['additional_ridge_proposals']}** | Возможны царапины, реперы и шум |
| Все кандидаты до NMS | **{r['total_line_proposals_not_tracks']}** | НЕ число физических треков |
| Прежде пустых кропов, теперь с кандидатами | **{r['selected_crops_filled_with_new_candidates']}/17** | Проверка охвата, НЕ recall |
| Дополнительная пространственно-сбалансированная выборка полных BMP | **{r['full_frame_tiles_with_machine_proposals']}/{r['sampled_nonselect_full_frame_tiles']}** участков с предложениями | **{r['full_frame_line_proposals']}** машинных фрагментов; реакция на артефакты тоже возможна |
| Узлы после подавления дубликатов | **{r['after_NMS_graph_nodes']}** | Схематические 2D-линии |
| Возможные стыки | **{r['possible_join_proposals']}** | Геометрически рассмотренные, не физически подтверждённые |
| Отобранные S56 графом связи | **{r['selected_join_hypotheses_UNKNOWN']}** | Все имеют `UNKNOWN` |
| Компоненты цепочек S58 | **{r['machine_connected_components']}** | **{r['components_of_two_lines']} пары** и **{r['components_of_four_lines']} цепочка из 4**; соединительные промежутки **предсказаны**, а не увидены |

Дополнительно алгоритм автоматически сформировал машинную псевдомаску: **{r['machine_pseudo_positive_centerline_pixels_not_GT']}** предположительно линейных пикселей, **{r['machine_pseudo_negative_neighbor_pixels_not_GT']}** предположительно фоновых и **{r['machine_unknown_pixels']}** `UNKNOWN=255`. Эти числа **не представляют экспертную разметку** и не годятся для подсчёта точности настоящих треков.

Имеются **{r['native_real_operator_starts']} реальных операторских `start`-точек**; на их основе S28 сформировал **{r['native_machine_draft_start_hypotheses']} вариантов продолжения** — 4 классических и 4 с ANZA для каждой точки. Их реальная правильность ещё не оценена.

Источники: [S39 machine-only](S39_MACHINE_ONLY_RESEARCH_2026_10_25.md), [S27–S29](S27_S29_ANZA_H50Z_ROBUST_TRACK_RESEARCH_2026_10_25.md), [S56–S58](S58_MACHINE_PATH_COMPONENTS_2026_10_26.md).

## 3. Что удалось количественно проверить на искусственных линиях поверх настоящего фона

### 3.1. S28: продолжение искусственного трека от известной точки (16 примеров)

| Метод | Средняя доля правильно найденных точек вставки (± 3px) | Успешных ≥70% |
|---|---:|---:|
| Продолжение по прямой | **{100*methods['straight_evidence']['fraction']:.2f}%** | {methods['straight_evidence']['successes']}/16 |
| Кривизна + фотометрия | **{100*methods['curvature_constrained_evidence']['fraction']:.2f}%** | **{methods['curvature_constrained_evidence']['successes']}/16** |
| ANZA + регуляризация кривизны | **{100*methods['original_AZ_plus_curvature']['fraction']:.2f}%** | {methods['original_AZ_plus_curvature']['successes']}/16 |

**Факт:** на данном искусственном контроле классическая геометрия оказалась сильнее ANZA, поэтому нельзя заявлять превосходство нейросети. В S29 при одном сценарии программного удаления участков, смещении начала 0px, покрытие составило **{100*s29['fraction']:.1f}%**, успешных **{s29['successes']}/{s29['n']}**. Не означает успех реальной идентификации частиц.

### 3.2. S56: графовый оптимизатор, которому заранее даны искусственные фрагменты (НЕ от изображения)

256 искусственных сцен, 1024 потенциально верных стыка, 512 возможных трёхчастных цепочек:

| Сборка | Искусственно правильные | Прочие | Полные цепочки |
|---|---:|---:|---:|
| Взаимно-лучший выбор S39 | {s56['node_mutual']['correct']} | {s56['node_mutual']['other']} | {s56['node_mutual']['full_chains']} |
| Жадный выбор по концам | **{s56['port_greedy']['correct']}** | {s56['port_greedy']['other']} | **{s56['port_greedy']['full_chains']}** |
| Глобальная сборка MILP + отказ | {s56['global_forest']['correct']} | **{s56['global_forest']['other']}** | {s56['global_forest']['full_chains']} |

Глобальная сборка уменьшает число дополнительных неверных стыков, но также теряет правильные цепочки по сравнению с жадным выбором. **Важно:** тут исходные полилинии получены из генератора с ошибками, а не выведены из фотопикселей. Это *не* end-to-end оценка точности фотографической реконструкции.

### 3.3. S72: уже путь «пиксели → кандидаты → граф», 96 искусственных сцен

| Метод | Правильные синтетические стыки | Прочие / несопоставленные | Целые 3-фрагментные пути (из 192) |
|---|---:|---:|---:|
| Исходный S39 | {s72['S39_original']['correct']} | {s72['S39_original']['other']} | {s72['S39_original']['full_chains']} |
| **S72 (CAL-выбранный по двум направлениям)** | **{s72['frozen_CAL_selected']['correct']}** | **{s72['frozen_CAL_selected']['other']}** | **{s72['frozen_CAL_selected']['full_chains']}** |

На той же отложенной синтетической выборке сохранено количество правильных стыков и уменьшено число прочих с 52 до 41. Но восстановлено лишь **13/192** полных искусственных цепочек. **Нельзя говорить, что получено 13 настоящих частиц.**

### 3.4. Почему новые алгоритмы пока не заменили исходный S72

- **S84 на чистых синтетических сценах:** правильные стыки **{s84['clean']['baseline_correct']}→{s84['clean']['selected_correct']}**, прочие **{s84['clean']['baseline_other']}→{s84['clean']['selected_other']}**. На размытии σ1.1 прочие **{s84['blur_sigma_11']['baseline_other']}→{s84['blur_sigma_11']['selected_other']}**. Выигрыш по правильным сопровождается нежелательным ростом непроверенных/неправильных связей; автоматическое внедрение отклонено.
- **S85 при замене детектора:** в новом *полностью искусственном* benchmark три detector-specific модели обучились, но **все три** не прошли консервативный критерий на CAL и выбрали `none` — **0 соединений на TEST по безопасной политике**. Это отрицательный результат для готовности автоматической сборки, а не доказательство отсутствия смысла у графов.
- **ANZA vs U-Net:** на synthetic-on-H50Z исследовании S33 нормализация увеличила AUROC некоторых версий AZUNet, но обычная U-Net оставалась конкурентоспособной. Превосходство ANZA в физической точности не доказано. Подробно: [S33–S34](S33_S34_PHOTOMETRIC_AND_AXIAL_RESEARCH_2026_10_25.md).

## 4. Какие визуальные слои можно показать

**Реальные приватные результаты:** исторические `ANZA_LIRA_S39_PRIVATE_MACHINE_DEMO.zip` и `ANZA_LIRA_S56_S57_PRIVATE_MACHINE_DEMO.zip` (локальные исходные фотографии + предложенные линии + частные координаты). В GitHub их **нет**; при наличии разрешённых приватных входов можно заново собрать через существующие [S39](../../scripts/research/bubble_chamber/s39_make_physicist_demo.py) и [S57](../../scripts/research/bubble_chamber/s57_build_global_paths_viewer.py).

**Повторный запуск в авторизованной закрытой директории** (пути `/PRIVATE/...` — обязательные существующие локальные входы, а НЕ файлы GitHub):

```bash
# При наличии подготовленных приватных S49/S51/S56 гипотез:
python scripts/research/bubble_chamber/s57_build_global_paths_viewer.py \
  --s49-private /PRIVATE/S49_MACHINE_HYPOTHESES_PRIVATE.json \
  --s51-private /PRIVATE/S51_NATIVE_TOPOLOGY_ADVISORIES_PRIVATE.json \
  --s56-private /PRIVATE/s56_output/S56_NATIVE_PATH_GRAPH_PRIVATE.json \
  --s49-images /PRIVATE/s49_images \
  --s56-public docs/bubble_chamber/S56_PUBLIC_GLOBAL_PATH_RESULTS.json \
  --out-dir /PRIVATE/ANZA_LIRA_S57_MACHINE_ONLY_SHOWCASE \
  --out-zip /PRIVATE/ANZA_LIRA_S56_S57_PRIVATE_MACHINE_DEMO.zip
```

Скрипт S57 **не** создаёт исходные S49/S51/S56 артефакты из ничего; сначала воспроизведите их по [S39](S39_MACHINE_ONLY_RESEARCH_2026_10_25.md) и [S56](S56_S57_MACHINE_GLOBAL_PATH_RESEARCH_2026_10_26.md). Для S85 нужен отдельный приватный TRAIN/CAL, который в текущем состоянии не выполнен. `--help` обоих скриптов проверен на полном исходном дереве.

**Публично в GitHub — только схемы, не реальные H50Z:**

| Слой | Схематическая иллюстрация |
|---|---|
| Изображение, ещё без машинных выводов | ![ИСКУССТВЕННАЯ схема, не H50Z](physicists/figures/raw.svg) |
| Найденные центральные линии и возможный артефакт | ![ИСКУССТВЕННАЯ схема, не H50Z](physicists/figures/candidates.svg) |
| Варианты графовых стыков, неопределённые мостики | ![ИСКУССТВЕННАЯ схема, не H50Z](physicists/figures/graph.svg) |
| Псевдомаска: 1/0/255 | ![ИСКУССТВЕННАЯ схема, не H50Z](physicists/figures/mask.svg) |

Схемы показывают **типы** слоёв интерфейса; они **не** свидетельствуют, что конкретные нарисованные линии действительно были найдены на H50Z. Существующие приватные просмотрщики содержат реальные фотоналожения с сохранёнными статусами `UNKNOWN`.

## 5. Если не будет экспертной разметки — что система умеет и чего не узнает

1. **Может без разметки:** находить кандидаты 2D-линий на исходных фото, строить вспомогательные 0/1/255-псевдомаски, выбирать геометрически допустимые потенциальные мостики, группировать гипотезы, экспортировать локальную HTML-галерею для физика.
2. **Может использовать машинную псевдоразметку только осторожно:** обучение на предложениях модели зависит от качества самих предложений. 0 — *условный* фон, 255 — неизвестно. Самосогласие двух моделей не становится GT.
3. **Не может объективно установить без независимой проверки:** настоящую физическую precision/recall, правильность particle identity, 3D, импульс/массу, долю ложных мостиков на другом запуске.
4. **Безопасное поведение:** сохраняет оригинальные наблюдаемые линии, мостики — отдельный непроверенный слой, может отказаться от неоднозначной сборки. `physical_auto_joins_authorized = false`.

## 6. Что имеет смысл показать физикам и запросить у них

**Пятиминутная демонстрация:** настоящие фотопиксели в приватном просмотрщике → машинные центральные линии S39 → продолжения от настоящего `start` S28 → 37 выбранных UNKNOWN-стыков S56 → одна четырёхфрагментная UNKNOWN-цепочка S58 → сопоставление с искусственным эталоном S28/S72.

**Минимальный запрос специалистам:** посмотреть ограниченный набор уже сформированных кандидатов на парных 2D-проекциях, указать несколько заведомо реальных/нереальных линий и пары `same_track`/`different_track`, предоставить **новые независимые физические запуски** и, для 3D, систему согласования проекций и параметры поля. Это не просьба вручную разметить тысячи фотографий.

**Следующее направление:** улучшать именно **instance-aware извлечение изогнутых фрагментов и локализацию их концов**, а затем проверять ложные стыки на экспертных данных. [S62](S62_IMAGE_ONLY_CANDIDATE_BOTTLENECK_2026_10_26.md) и [S76](S76_SHAPE_VS_ENDPOINT_SCIENTIFIC_REPORT_2026_10_26.md) показывают, что многие искусственные фрагменты теряются *до* оптимизации графа.

## 7. Научная граница и ресурсы проекта

- **Никаких независимых экспертно подтверждённых физических треков H50Z нет.** Все цифры по нативным фотографиям — машинные предложения. Все проценты полноты на добавленных линиях — *синтетические*.
- Фотографии H50Z `005/006` из **одного физического запуска**; 39 кропов и 96/256 генераторных сцен не дают 39/96/256 независимых физических экспериментов.
- Не говорить «система обнаружила 567 физических треков», «проверила 37 правильных физических связей» или «ANZA достигла 96,7% реальной полноты».
- Приватные исходники изображений, веса, film070 измеренные точки и настоящий фотодемонстратор хранятся **только в авторизованной закрытой директории**.

**Основные первичные записи:** [S39 JSON](S39_PUBLIC_RESULTS.json) · [S56 JSON](S56_PUBLIC_GLOBAL_PATH_RESULTS.json) · [S58 JSON](S58_PUBLIC_MACHINE_PATH_COMPONENTS.json) · [S72 JSON](S72_PUBLIC_FRESH_TEST_2026_10_26.json) · [S84 JSON](S84_PUBLIC_FRESH_TEST_2026_10_26.json) · [S85 generated-only JSON](S85_INTEROP_CALIBRATED_GENERATED_TOY_RESULTS_2026_10_28.json). JSON-манифест содержит SHA-256 каждого оригинального агрегата и проверяется CI.
'''


def readme(ledger):
    r=ledger['real_H50Z_machine_only']
    return f'''# Материалы ANZA–LIRA для встречи с физиками

**Статус: исследовательские результаты · 29 октября 2026 · read-only / без автоматического соединения физических треков.**

**[Научный отчёт с первичными источниками](../PHYSICISTS_RESULTS_2026_10_29.md)** · [HTML-демонстрация (скачать папку целиком и открыть `demo.html`)](demo.html) · [Машинно-проверяемые результаты](results_manifest.json)

## Что есть сейчас

- **2 настоящих H50Z-фотографии из 1 физического запуска:** 39 выбранных участков 512×512.
- **{r['total_line_proposals_not_tracks']} машинных гипотез линий**, после NMS {r['after_NMS_graph_nodes']} узлов, {r['possible_join_proposals']} кандидата соединения, {r['selected_join_hypotheses_UNKNOWN']} выбранных S56 связей.
- **{r['machine_connected_components']} машинных компонентов**: {r['components_of_two_lines']} пары и 1 цепочка из четырёх частей. Все имеют статус `UNKNOWN`.
- Сильный **контролируемый синтетический** S28: классическое продолжение линии от известной точки — 96,67% покрытия, 16/16 успешных примеров; не результат по настоящей particle GT.
- **0 экспертно подтверждённых масок и связей; 0 разрешённых автоматических физических объединений**.

## Как выглядят выходы — схемы, НЕ реальные H50Z

![ИСКУССТВЕННЫЙ пример фотофона, не H50Z](figures/raw.svg)
![ИСКУССТВЕННЫЙ пример машинных линий, не H50Z](figures/candidates.svg)
![ИСКУССТВЕННЫЙ пример UNKNOWN соединений, не H50Z](figures/graph.svg)
![ИСКУССТВЕННАЯ псевдомаска, не H50Z](figures/mask.svg)

## Как получить просмотр на **настоящих** H50Z

Реальные H50Z-пиксели нельзя публиковать без разрешения. В существующих приватных архивах `ANZA_LIRA_S39_PRIVATE_MACHINE_DEMO.zip` и `ANZA_LIRA_S56_S57_PRIVATE_MACHINE_DEMO.zip` есть реальное изображение, наложенные машинные кандидаты, пунктирные мостики, предупреждения и офлайн HTML. Инструкция по повторной сборке и требуемым приватным входам: [проверенный pipeline](../S39_MACHINE_ONLY_RESEARCH_2026_10_25.md), [сборка S57](../S56_S57_MACHINE_GLOBAL_PATH_RESEARCH_2026_10_26.md).

**Нельзя:** выдавать SVG-схемы за реальные снимки, машинные предложения за физические треки, а искусственную GT — за экспертное подтверждение. Без экспертной разметки можно продолжать генерировать предложения; нельзя измерить реальную precision/recall.
'''


def root_entrypoint(ledger):
    r=ledger['real_H50Z_machine_only']
    return f'''# Для физиков: результаты ANZA–LIRA на H50Z

**[Основной отчёт по результатам и их достоверности](docs/bubble_chamber/PHYSICISTS_RESULTS_2026_10_29.md)** · [Машинно-проверяемый JSON](docs/bubble_chamber/physicists/results_manifest.json) · [Иллюстрированная галерея с учебными схемами](docs/bubble_chamber/physicists/README.md)

В архиве исследований: **{r['total_line_proposals_not_tracks']} машинных линейных предложений, {r['selected_join_hypotheses_UNKNOWN']} выбранных графовых стыков и {r['machine_connected_components']} компонентов** на настоящих фотопикселях H50Z из **двух изображений одного запуска**. Правильность реальных треков не проверена физиками; все машинные мостики `UNKNOWN`.

На искусственно добавленных треках отдельно проверено продолжение, поиск фрагментов и графовая сборка. **Синтетическая метрика не является физической precision/recall.**

**Реальные фото, координаты и контрольные веса не публикуются в этом GitHub.** Для просмотра настоящих H50Z используйте только разрешённые приватные демонстраторы S39 и S57 и команды сборки из основного отчёта. Учебные SVG/HTML в GitHub исключительно схематические.

**Автоматические физические соединения не разрешены.**
'''


def files_to_generate(root=ROOT):
    ledger=load_results(root)
    out={
     'docs/bubble_chamber/physicists/results_manifest.json':json.dumps(ledger,ensure_ascii=False,sort_keys=True,indent=2)+'\n',
     'docs/bubble_chamber/PHYSICISTS_RESULTS_2026_10_29.md':markdown_brief(ledger),
     'docs/bubble_chamber/physicists/README.md':readme(ledger),
     'PHYSICISTS_HANDOFF.md':root_entrypoint(ledger),
     'docs/bubble_chamber/physicists/demo.html':_html_view(ledger),
    }
    for kind in ('raw','candidates','graph','mask'):
        out[f'docs/bubble_chamber/physicists/figures/{kind}.svg']=svg_panel(kind)
    return out


def main():
    p=argparse.ArgumentParser(description='Deterministic, source-locked public physics handoff; no private photo data')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--write',action='store_true',help='Write/rewrite only the NEW physics documentation bundle')
    group.add_argument('--check',action='store_true',help='Fail if public handoff is stale or differs from source JSON')
    args=p.parse_args()
    generated=files_to_generate()
    for rel,content in generated.items():
        file=ROOT/rel
        if args.check:
            if not file.is_file() or file.read_bytes()!=content.encode('utf8'):
                raise SystemExit(f'STALE or MISSING: {rel}; use --write and review its scientific meaning')
        else:
            file.parent.mkdir(parents=True,exist_ok=True)
            file.write_text(content,encoding='utf8')
    print(('VERIFIED' if args.check else 'GENERATED'),len(generated),'public-only physics handoff files; zero private H50Z pixels or native GT')


if __name__=='__main__': main()
