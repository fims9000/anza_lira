"""Private-data-free security/provenance and machine-first workflow tests."""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import json,cv2,numpy as np,pytest
from bubble_chamber.s31_machine_first import validate_draft_pack,file_sha,actual_pixel_sha,lsd_candidates
from bubble_chamber.s31_weak_supervision import build_weak_masks,validate_corrections
from bubble_chamber.s31_review_workbench import embed_machine_drafts


def fixture(tmp_path):
    folder=tmp_path/'tiles';folder.mkdir()
    photos=[];pack={'schema':'s31_machine_first_h50z_draft_v1',
       'physical_joins_authorized':False,'expert_track_labels_generated':0,
       'model_drafts_are_ground_truth':False,
       'source_S28_proposals_sha256':'b'*64,'source_S29_reviewer_sha256':'c'*64,
       'photos':[]}
    for i in range(39):
        photo='005' if i<22 else '006';anchor=i<6 or (22<=i<25)
        name=f'H50Z_030_01_{photo}_'+(f'start{i:02d}' if anchor else f'mined{i:02d}')
        gray=np.full((512,512),193,np.uint8)
        cv2.line(gray,(100,220),(350,230),80,2)
        p=folder/(name+'.png');assert cv2.imwrite(str(p),gray)
        pts=[[round(z,2),round(230+10*(z-200)/150,2)] for z in np.linspace(100,350,96)]
        stratum='start_anchor_nearby' if anchor else 'fiducial_control' if i%5==0 else 'dense'
        dr={'draft_id':f'draft_{i:03d}','source_method':'classical' if anchor else 'LSD_line_piece',
           'source_hypothesis_index':0,'points_xy':pts,'cross_method_agreement_UNCALIBRATED':True,
           'duplicate_of_draft_id':None,'photometry_support_fraction_UNCALIBRATED':.9,'model_score_UNCALIBRATED':.8,
           'status':'MACHINE_DRAFT_UNREVIEWED','label_source':'MODEL_NOT_HUMAN','physics_permission':False,
           'geometry':{'pixel_arclength':250.2},'observed_contrast_8bit':12.}
        rec={'crop_id':name,'photo_id':photo,'stratum':stratum,
             'source_original_png_sha256_from_S29':file_sha(p),'derived_png_sha256_from_BMP':file_sha(p),
             'source_gray_pixels_sha256':actual_pixel_sha(gray),
             'image_shape':[512,512],'drafts':[dr],
             'relation_drafts':[]}
        if anchor:rec['start_xy_OPERATOR_REAL']=[256.,256.]
        pack['photos'].append(rec)
        photos.append({'crop_id':name,'photo':photo,'sha256':file_sha(p),'image_modes':{'original':'data:image/png;base64,AA=='}})
    pack['photos'][0]['relation_drafts']=[{'relation_id':'rel_001','candidate_a':'draft_000','candidate_b':'draft_000_B',
      'status':'MACHINE_POSSIBLE_LINK_UNKNOWN','physical_join_allowed':False}]
    # Additional draft needed to form valid relation without introducing physics truth.
    pack['photos'][0]['drafts'].append(dict(pack['photos'][0]['drafts'][0],draft_id='draft_000_B',source_method='original_AZUNet_assisted'))
    return folder,pack,photos


def corrections(pack):
    return {'schema':'s31_machine_corrections_v1',
            'machine_source_s28_sha256':pack['source_S28_proposals_sha256'],
            'machine_source_s29_sha256':pack['source_S29_reviewer_sha256'],
            'physical_joins_authorized':False,'all_unreviewed_stay_unknown':True,
            'crops':[{'crop_id':p['crop_id'],'source_crop_sha256':p['source_original_png_sha256_from_S29'],
                      'source_gray_pixels_sha256':p['source_gray_pixels_sha256']} for p in pack['photos']],
            'decisions':{},'relation_decisions':{}}


def test_39_crop_provenance_and_no_imaginary_expert(tmp_path):
    _,pack,_=fixture(tmp_path)
    assert validate_draft_pack(pack)
    assert sum('start_xy_OPERATOR_REAL' in x for x in pack['photos'])==9
    assert sum(len(x['drafts']) for x in pack['photos'])==40
    bad=json.loads(json.dumps(pack));bad['photos'][0]['drafts'][0]['label_source']='EXPERT'
    with pytest.raises(ValueError,match='expert|physical'):validate_draft_pack(bad)
    bad=json.loads(json.dumps(pack));bad['photos'][0]['relation_drafts'][0]['physical_join_allowed']=True
    with pytest.raises(ValueError,match='relation'):validate_draft_pack(bad)


def test_machine_masks_and_unknown_pixels(tmp_path):
    folder,pack,_=fixture(tmp_path)
    r=build_weak_masks(pack,folder,tmp_path/'out',negative_points_per_crop=100)
    assert r['counts']['crops']==39
    assert r['human_certified_track_pixels']==0
    assert r['physics_auto_join_allowed'] is False
    assert r['counts']['weak_positive_px']>0
    assert r['counts']['unlabeled_px']>0
    for entry in r['entries']:
        m=cv2.imread(str(tmp_path/'out'/entry['mask']),cv2.IMREAD_UNCHANGED)
        w=np.load(tmp_path/'out'/entry['weight'],allow_pickle=False)
        assert set(np.unique(m))<={0,1,255}
        assert np.all(w[m==255]==0)
        if entry['stratum']=='fiducial_control':assert entry['provisional_positive_count']==0


def test_user_corrections_override_machine_and_invalid_relations_blocked(tmp_path):
    folder,pack,_=fixture(tmp_path);r=corrections(pack)
    r['decisions']['draft_000']={'status':'USER_REJECTED_PROVISIONAL','edited_points_xy':None}
    r['relation_decisions']['rel_001']={'status':'LOCAL_DIFFERENT_PROVISIONAL'}
    assert len(validate_corrections(pack,r))==1
    o=build_weak_masks(pack,folder,tmp_path/'with-correction',corrections=r)
    assert o['entries'][0]['provisional_positive_count']==0
    r['decisions']['draft_000']={'status':'USER_ACCEPTED_PROVISIONAL','edited_points_xy':None}
    o2=build_weak_masks(pack,folder,tmp_path/'accepted',corrections=r)
    assert o2['entries'][0]['provisional_positive_count']>0
    r['relation_decisions']['FAKE_REL']={'status':'LOCAL_SAME_PROVISIONAL'}
    with pytest.raises(ValueError,match='invented'):validate_corrections(pack,r)
    r['relation_decisions'].pop('FAKE_REL');r['crops'][0]['source_gray_pixels_sha256']='0'*64
    with pytest.raises(ValueError,match='Source photo'):validate_corrections(pack,r)


def test_original_photo_mutation_fails_before_masks(tmp_path):
    folder,pack,_=fixture(tmp_path)
    file=folder/(pack['photos'][0]['crop_id']+'.png');file.write_bytes(b'forged photo')
    with pytest.raises(ValueError,match='checksum'):build_weak_masks(pack,folder,tmp_path/'out')


def test_user_ui_keeps_existing_manual_export_untouched(tmp_path):
    _,pack,photos=fixture(tmp_path)
    payload=json.dumps({'schema':'s26_review_visual_payload_v1','photos':photos})
    old="function update(){byId('background').src=entries[active].image_modes[byId('mode').value];redraw();if(typeof window.s28RenderGhost==='function')window.s28RenderGhost();}"
    original='<html><body><p id="s29-values"></p><svg id="s28-ghost"></svg><canvas id="annotations" width="512" height="512"></canvas><div id="scene"></div><script type="application/json" id="payload">'+payload+'</script><script>'+old+'</script></body></html>'
    view=embed_machine_drafts(original,pack)
    assert 's31-machine' not in original
    assert 's31-draft-payload' in view and 's31-rel-same' in view and 's31-edit' in view
    assert 's31ExportCorrections' in view
    assert view.count('s31-draft-payload')>=2
    assert original not in view
    bad=json.loads(json.dumps(pack));bad['photos'][1]['source_original_png_sha256_from_S29']='0'*64
    with pytest.raises(ValueError,match='SHA'):embed_machine_drafts(original,bad)


def test_image_lsd_proposals_are_valid_and_not_truth(tmp_path):
    gray=np.full((512,512),210,np.uint8)
    cv2.line(gray,(30,150),(450,200),40,4,cv2.LINE_8)
    lines=lsd_candidates(gray,3)
    assert isinstance(lines,list)
    for item in lines:
        points=np.asarray(item['points'])
        assert points.ndim==2 and points.shape[1]==2
        assert np.isfinite(points).all() and np.all((points>=0)&(points<=511))
