"""Whole S85 runner smoke against generated, non-H50Z, non-film070 fixtures.

This verifies TRAIN -> frozen CAL -> TEST -> TEST replay *code wiring* only.
Never quote these examples or metrics as research/physics accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
from scripts.research.bubble_chamber import s85_train_calibrate_test_learned as s85


def test_entire_runner_on_generated_photos_and_geometry(tmp_path,monkeypatch):
    # All source pixels and all template coordinates are created by this test.
    from bubble_chamber.s56_three_piece_scenes import SCENARIOS,make_three_piece_scene
    from bubble_chamber.s84_graph_context_candidates import _score_original_contacts
    from bubble_chamber.s85_train_reference_labels import label_extras_after_inference
    from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
    from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
    from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75
    from scripts.research.bubble_chamber import s80_calibrate_then_test_multiscale as s80
    from scripts.research.bubble_chamber import s84_calibrate_then_fresh_test_graph_context as s84

    photos={}
    for i,source in enumerate(('005','006')):
        rng=np.random.default_rng(85500+i)
        yy,xx=np.indices((512,512))
        texture=rng.normal(0,5,size=(512,512))
        gray=np.clip(196+11*np.sin(xx/27)+7*np.cos(yy/33)+texture,0,255).astype(np.uint8)
        path=tmp_path/(f'GENERATED_NOT_H50Z_{source}.png')
        assert cv2.imwrite(str(path),gray)
        photos[source]=[path]

    def fake_templates(offset):
        ts=[]
        for i in range(28):
            t=np.linspace(0.,1.,64)
            points=np.column_stack((100.+700*t,300.+12*np.sin(2*np.pi*t+i*.14+offset)))
            ts.append({'frame':'SYNTHETIC_FAKE_NOT_FILM070','points':points,'id':str(i)})
        return ts
    templates={k:fake_templates(i) for i,k in enumerate(('train','cal','test'))}
    frozen={'directions':{f'{s}_to_{t}':{'CAL_selected':s68.ARMS[0],
                  'S56_frozen_CAL_cutoff':.60}
         for s,t in [('005','006'),('006','005')]}}
    def fake_scorer(rows):
        return np.full((len(rows),),.67,dtype=float)
    monkeypatch.setattr(s68,'_fit_source',lambda *args:(fake_scorer,.60))
    deps=(SCENARIOS,make_three_piece_scene,_score_original_contacts,
          label_extras_after_inference,s68,s72,s75,s80,s84)
    context={'locks':s85.code_locks(),
             'source_split_sha256':'SYNTHETIC_QA_NOT_FILM070',
             's72_source_code_locks':s72.LOCKS,
             'H50Z_independent_physical_runs':0,
             'native_particle_GT_count':0,
             'H50Z_070_physical_event_correspondence_proven':False,
             'physical_auto_joins_authorized':False}
    monkeypatch.setattr(s85,'_validate_private',lambda *_:(deps,templates,photos,frozen,context))
    # Reduced scenario counts belong ONLY to test execution, never scientific S85.
    monkeypatch.setattr(s85,'N_TRAIN',1)
    monkeypatch.setattr(s85,'N_CAL',1)
    monkeypatch.setattr(s85,'N_TEST',1)
    path_train=tmp_path/'TEST_ONLY_FAKE_train.json'
    path_cal=tmp_path/'TEST_ONLY_FAKE_cal.json'
    for phase,path,params in (
      ('train',path_train,{}),
      ('cal',path_cal,{'train_models':path_train})):
        record=s85.run(phase,tmp_path/'FAKE_V',tmp_path/'FAKE_P',**params)
        assert record['native_particle_GT_count']==0
        assert not record['smoke_NOT_CONFIRMATORY'] # real phase code with synthetic-only in-memory fixtures
        path.write_text(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n')
    train=json.loads(path_train.read_text())
    assert all(not m['model']['trained'] for m in train['models'].values())
    cal=json.loads(path_cal.read_text())
    assert all(d['CAL_decision']['CAL_selected']=='baseline' for d in cal['directions'].values())
    first=s85.run('test',tmp_path/'FAKE_V',tmp_path/'FAKE_P',
                   train_models=path_train,calibration=path_cal)
    repeated=s85.run('test',tmp_path/'FAKE_V',tmp_path/'FAKE_P',
                   train_models=path_train,calibration=path_cal)
    assert json.dumps(first,sort_keys=True)==json.dumps(repeated,sort_keys=True)
    assert first['frozen_CAL_sha256']==s85.digest(path_cal)
    assert first['frozen_TRAIN_models_sha256']==s85.digest(path_train)
    assert all(first['CAL_selected_TEST'][k]==first['all_TEST_aggregates'][k]['baseline']
               for k in s85.TEST_CONDITIONS)
    assert first['physical_auto_joins_authorized'] is False
    assert sum(first['all_TEST_aggregates'][c]['baseline']['scenes'] for c in s85.TEST_CONDITIONS)==16
    # NOTHING from these mock synthetic pixels/geometry is promoted to source or public evidence.
