"""Tests for turnkey S85 private-release plumbing; no physical data or GT."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.research.bubble_chamber import s85_release_manager as release


def test_source_complete_and_no_private_data_is_honestly_blocked(tmp_path):
    status=release.preflight(tmp_path/'ABSENT_FILM070',tmp_path/'ABSENT_H50Z')
    assert status['source']['source_locks_valid'] is True
    assert status['source']['source_bundle_complete'] is True
    assert status['ready_for_private_TRAIN'] is False
    assert set(status['blockers'])=={'VISUALMETRO_FILM070_NOT_MOUNTED','PRIVATE_H50Z_PHOTO_TILES_NOT_MOUNTED'}
    assert status['native_expert_GT_count']==0 and status['physical_auto_joins_authorized'] is False
    assert status['TRAIN_CAL_TEST_executed_by_preflight'] is False


def test_source_only_missing_dependencies_are_visible(monkeypatch,tmp_path):
    monkeypatch.setattr(release,'REQUIRED_PUBLIC',(*release.REQUIRED_PUBLIC,'bubble_chamber/missing_research_dependency.py'))
    j=release.preflight(tmp_path/'nothing',tmp_path/'nothing')
    assert j['source']['source_bundle_complete'] is False
    assert 'bubble_chamber/missing_research_dependency.py' in j['source']['missing_public_source_files']
    assert 'MISSING_OR_CHANGED_FROZEN_PUBLIC_SOURCE' in j['blockers']


def test_private_output_cannot_contaminate_git_and_is_immutable(tmp_path):
    with pytest.raises(ValueError,match='MUST NOT'):
        release._safe_directory(release.ROOT/'docs/bubble_chamber/private_models')
    with pytest.raises(ValueError,match='MUST NOT'):
        release._safe_directory(release.ROOT)
    p=tmp_path/'PRIVATE_TRAIN.json'
    release._write_once(p,{'coef':[1,2], 'private_model':True})
    assert (p.stat().st_mode & 0o777)==0o600
    with pytest.raises(FileExistsError):release._write_once(p,{'coef':[3]})
    assert json.loads(p.read_text())['coef']==[1,2]


def _dummy_row(correct=1,wrong=2):
    return {'selected_synthetic_correct_links':correct,
            'synthetic_noncorrect_selected':wrong,
            'complete_three_piece_paths_synthetic':0,
            'shape_true_fragments_matched':0}


def fake_execute(phase,photo,tiles,*,train_models=None,calibration=None):
    data={'schema':release.s85.SCHEMA,'phase':phase,
          'native_particle_GT_count':0,'smoke_NOT_CONFIRMATORY':False,
          'locks':{'fixed':'source'},'source_split_sha256':'source_only_test',
          's72_source_code_locks':{'S72':'frozen'},
          'directions':{}}
    if phase=='train':
        data['models']={k:{'model':{'trained':False,'reason':'ABSTAIN'},
             'model_sha256':'synthetic_test_hash',
             'label_counts':{'positive':0,'negative':0,'abstain':1}}
             for k in ('005_to_006','006_to_005')}
    if phase=='cal':
        data['directions']={k:{'CAL_decision':{'CAL_selected':'baseline'}, 'train_model_sha256':'synthetic_test_hash'} for k in ('005_to_006','006_to_005')}
        data['frozen_TRAIN_models_sha256']=release.digest(train_models)
    if phase=='test':
        data['directions']={k:{'CAL_selected':'baseline','train_model_sha256':'synthetic_test_hash'} for k in ('005_to_006','006_to_005')}
        data['frozen_TRAIN_models_sha256']=release.digest(train_models)
        data['frozen_CAL_sha256']=release.digest(calibration)
        data['all_TEST_aggregates']={cond:{arm:_dummy_row() for arm in release.s85.ARMS}
              for cond in release.s85.TEST_CONDITIONS}
        data['CAL_selected_TEST']={cond:_dummy_row() for cond in release.s85.TEST_CONDITIONS}
    return data


def fake_native(visualmetro,tiles,models,cal):
    return {'schema':'S85_REAL_39_H50Z_NO_NATIVE_PHYSICS_GT_V1',
            'independent_physics_GT_count':0,
            'frozen_TRAIN_sha256':release.digest(models),
            'frozen_CAL_sha256':release.digest(cal),
            'all_machine_joins':'UNKNOWN',
            'aggregates':{'baseline':{'line_instances':0,'graph_advisory_joins_UNKNOWN':0}}}


def _mock_private_ready(monkeypatch):
    monkeypatch.setattr(release,'preflight',lambda *a,**kw:{'ready_for_private_TRAIN':True,'blockers':[]})
    monkeypatch.setattr(release,'inspection_private_context',lambda *a:{'locks':{'fixed':'source'}})
    monkeypatch.setattr(release.s85,'_model_receipt',lambda *_args:True)


def test_one_command_private_run_repeats_test_and_exports_only_aggregates(tmp_path,monkeypatch):
    _mock_private_ready(monkeypatch)
    result=release.run_all(tmp_path/'photo',tmp_path/'film',tmp_path/'PRIVATE',
                           runner=fake_execute,native_auditor=fake_native)
    assert result['TEST_replay_byte_identical']
    assert result['physical_auto_joins_authorized'] is False
    assert not result['full_physics_deployment_ready']
    assert result['CAL_frozen_selections']=={'005_to_006':'baseline','006_to_005':'baseline'}
    assert result['native_particle_GT_count']==0
    names=release.PRIVATE_OUTPUTS
    assert all((tmp_path/'PRIVATE'/f).is_file() for f in names)
    assert (tmp_path/'PRIVATE'/names[2]).read_bytes()==(tmp_path/'PRIVATE'/names[3]).read_bytes()
    report=(tmp_path/'PRIVATE'/release.SUMMARY_NAME).read_text()
    assert all(k not in report for k in ('"coef"','"intercept"','"points_xy"','"scaler_mean"'))
    with pytest.raises(FileExistsError):
        release.run_all(tmp_path/'photo',tmp_path/'film',tmp_path/'PRIVATE',
                        runner=fake_execute,native_auditor=fake_native)
    resumed=release.run_all(tmp_path/'photo',tmp_path/'film',tmp_path/'PRIVATE',
                        runner=fake_execute,native_auditor=fake_native,resume=True)
    assert resumed==result


def test_replay_mismatch_blocks_summary(tmp_path,monkeypatch):
    _mock_private_ready(monkeypatch)
    count={'test':0}
    def unstable(phase,*a,**kw):
        rec=fake_execute(phase,*a,**kw)
        if phase=='test':
            count['test']+=1
            rec['mismatched_random_seed']=count['test']
        return rec
    with pytest.raises(ValueError,match='NOT byte-identical'):
        release.run_all(tmp_path/'p',tmp_path/'v',tmp_path/'PRIVATE',
                        runner=unstable,native_auditor=fake_native)
    assert not (tmp_path/'PRIVATE'/release.SUMMARY_NAME).exists()


def test_synthetic_test_provenance_cannot_be_spoofed(tmp_path,monkeypatch):
    _mock_private_ready(monkeypatch)
    def wrong_CAL(phase,*a,**kw):
        rec=fake_execute(phase,*a,**kw)
        if phase=='cal':rec['frozen_TRAIN_models_sha256']='not-actual-train-weights'
        return rec
    with pytest.raises(ValueError,match='train-model SHA'):
        release.run_all(tmp_path/'p',tmp_path/'v',tmp_path/'PRIV',
                        runner=wrong_CAL,native_auditor=fake_native)
