"""S118 synthetic identity tests; no claimed real-photo particle accuracy."""
import json
import pytest
from pathlib import Path
from bubble_chamber.s118_tile_stitch import stitch_tiles, StitchConfig
from scripts.research.bubble_chamber.run_s118_tile_stitch import run

SHA='a'*64


def line(tag, lo, hi, y, rect):
    return {'candidate_id':tag,'observed_pixel_path_xy':[[x-rect[0],y-rect[1]] for x in range(lo,hi+1)]}


def tile(name, rect, specs, grouped=None):
    curves=[line(c,x0,x1,y,rect) for c,x0,x1,y in specs]
    groups = grouped or [[x[0]] for x in specs]
    return {'tile_id':name, 'source_image_id':'sample.bmp','source_image_sha256':SHA,
            'xyxy':list(rect),'curves':curves,
            's117':{'groups':[{'group_id':'local'+str(i),'source_curve_ids':g} for i,g in enumerate(groups)]}}


def test_source_overlap_alias_same_original_path_one_identity():
    a=tile('A',(0,0,130,120),[('a',20,129,40)])
    b=tile('B',(70,0,200,120),[('b',70,170,40)])
    v=stitch_tiles([a,b])
    assert v['report']['global_group_count']==1
    assert v['report']['confirmed_overlap_aliases']==1
    assert v['overlap_aliases'][0]['kind']=='SOURCE_OBSERVATION_ALIAS'
    assert v['overlap_aliases'][0]['median_px']==0
    assert v['groups'][0]['source_curve_keys']==['A:a','B:b']


def test_aliases_are_idempotent_and_order_invariant():
    a=tile('A',(0,0,140,120),[('a',20,139,40)])
    b=tile('B',(60,0,200,120),[('b',60,190,40)])
    c=tile('C',(90,0,220,120),[('c',90,219,40)])
    p=stitch_tiles([a,b,c]);q=stitch_tiles([c,b,a])
    assert p==q
    assert p['report']['global_group_count']==1
    assert p['report']['confirmed_overlap_aliases']==2


def test_near_parallel_different_traces_never_aliased():
    a=tile('A',(0,0,150,120),[('a',20,149,40)])
    b=tile('B',(70,0,200,120),[('b',70,190,43)])
    v=stitch_tiles([a,b])
    assert v['report']['confirmed_overlap_aliases']==0
    assert v['report']['global_group_count']==2


def test_crossing_only_one_point_not_aliased():
    a=tile('A',(0,0,160,160),[('a',20,149,60)])
    # B horizontal trace 1 px off except a true crossing at x=80 needs actual angled points.
    b=tile('B',(60,0,180,160),[('b',60,169,62)])
    v=stitch_tiles([a,b])
    assert v['report']['confirmed_overlap_aliases']==0


def test_auto_tiny_seam_does_not_fill_observed_pixels():
    a=tile('A',(0,0,100,120),[('a',20,99,40)])
    b=tile('B',(100,0,200,120),[('b',100,170,40)])
    v=stitch_tiles([a,b])
    assert v['report']['global_group_count']==1
    assert v['report']['automatic_tiny_seam_joins']==1
    assert v['seam_candidates'][0]['connector_is_observed'] is False
    assert v['seam_candidates'][0]['observed_connector_pixels']==0


def test_long_gap_review_not_automatically_merged():
    a=tile('A',(0,0,100,120),[('a',20,99,40)])
    b=tile('B',(220,0,350,120),[('b',220,280,40)])
    v=stitch_tiles([a,b])
    assert v['report']['global_group_count']==2
    assert v['report']['review_only_gaps']>=1
    assert v['seam_candidates'][0]['gap_px']==121.
    assert v['seam_candidates'][0]['decision']=='REVIEW_ONLY_NO_PHOTO_FOR_GAP'
    forced=stitch_tiles([a,b],decisions={v['seam_candidates'][0]['candidate_id']:'same'})
    assert forced['report']['expert_seam_joins']==1
    assert forced['report']['global_group_count']==1


def test_ambiguous_competing_seam_not_auto_linked():
    a=tile('A',(0,0,100,120),[('a',20,99,40)])
    b=tile('B',(100,0,200,120),[('b',100,180,40),('c',100,180,41)])
    v=stitch_tiles([a,b])
    assert v['report']['automatic_tiny_seam_joins']==0
    assert v['report']['review_only_gaps']>=1


def test_local_groups_preserved_after_alias():
    a=tile('A',(0,0,130,120),[('a',20,129,40),('a2',20,60,75)],grouped=[['a','a2']])
    b=tile('B',(70,0,200,120),[('b',70,170,40)])
    v=stitch_tiles([a,b])
    assert v['report']['global_group_count']==1
    assert len(v['groups'][0]['source_curve_keys'])==3


def test_reject_disagreeing_sha_invalid_coords_and_duplicate_local_membership():
    a=tile('A',(0,0,130,120),[('a',20,129,40)])
    b=tile('B',(70,0,200,120),[('b',70,170,40)])
    b['source_image_sha256']='b'*64
    with pytest.raises(ValueError, match='different source'):
        stitch_tiles([a,b])
    b['source_image_sha256']=SHA
    b['curves'][0]['observed_pixel_path_xy'][0]=[999,999]
    with pytest.raises(ValueError, match='outside'):
        stitch_tiles([a,b])
    b=tile('B',(70,0,200,120),[('b',70,170,40)])
    b['s117']['groups'].append({'group_id':'dup','source_curve_ids':['b']})
    with pytest.raises(ValueError, match='double-assign'):
        stitch_tiles([a,b])


def test_reject_invalid_unknown_decisions():
    a=tile('A',(0,0,100,120),[('a',20,99,40)])
    b=tile('B',(220,0,350,120),[('b',220,280,40)])
    with pytest.raises(ValueError, match='Invalid expert'):
        stitch_tiles([a,b],decisions={'unknown':'yes'})
    with pytest.raises(ValueError, match='missing seam'):
        stitch_tiles([a,b],decisions={'unknown':'same'})


def test_runner_manifest_roundtrip_and_refuses_overwrite(tmp_path):
    a=tile('A',(0,0,130,120),[('a',20,129,40)])
    b=tile('B',(70,0,200,120),[('b',70,170,40)])
    manifest={'source_image_id':'sample.bmp','source_image_sha256':SHA,'tiles':[]}
    for t in (a,b):
        fn=t['tile_id']
        (tmp_path/(fn+'-curves.json')).write_text(json.dumps({'curves':t['curves']}))
        (tmp_path/(fn+'-groups.json')).write_text(json.dumps(t['s117']))
        manifest['tiles'].append({'tile_id':fn,'xyxy':t['xyxy'],'curves':fn+'-curves.json','s117':fn+'-groups.json'})
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    out=tmp_path/'out';result=run(path,out)
    assert result['report']['global_group_count']==1
    assert (out/'S118_STITCH_NOT_GT.json').is_file()
    assert 'polyline' in (out/'S118_VECTOR_REVIEW.svg').read_text()
    with pytest.raises(FileExistsError):run(path,out)


def test_runner_rejects_mixed_provenance_and_invalid_photo_hash(tmp_path):
    a=tile('A',(0,0,130,120),[('a',20,129,40)])
    curves_path=tmp_path/'curves.json'
    curves_path.write_text(json.dumps({'curves':a['curves'],'xyxy':[0,0,130,120]}))
    groups_path=tmp_path/'groups.json'
    groups_path.write_text(json.dumps(a['s117']))
    manifest={'source_image_id':'sample.bmp','source_image_sha256':SHA,
              'tiles':[{'tile_id':'A','xyxy':[0,0,130,120],
                        'curves':'curves.json','s117':'groups.json'}]}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    bad=tmp_path/'sample.bmp';bad.write_bytes(b'not the BMP')
    with pytest.raises(ValueError,match='SHA256 disagrees'):
        run(path,tmp_path/'out',verify_bmp=bad)
    groups=a['s117'].copy()
    groups['provenance']={'source_curves_sha256':'f'*64}
    groups_path.write_text(json.dumps(groups))
    with pytest.raises(ValueError,match='different curve file'):
        run(path,tmp_path/'out')
    groups_path.write_text(json.dumps(a['s117']))
    manifest['tiles'][0]['xyxy']=[1,0,131,120]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='coordinates disagree'):
        run(path,tmp_path/'out')
