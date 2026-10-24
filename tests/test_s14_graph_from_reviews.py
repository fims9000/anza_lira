from pathlib import Path
import importlib.util,sys,hashlib,json
import pytest
from bubble_chamber.review import pair_key,result_digest,SCHEMA

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/research/bubble_chamber/s14_graph_from_reviews.py'
spec=importlib.util.spec_from_file_location('s14_review_script',SCRIPT)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def make_result(i):
    return {'source_sha256_grayscale_pixels':hashlib.sha256(str(i).encode()).hexdigest(),
            'pipeline':'p0',
            'segments': [{'segment_id':1,'measurement_xy_px':[[j,j*.04] for j in range(6)]},
                         {'segment_id':2,'measurement_xy_px':[[j+7, .04*(j+7)**2] for j in range(6)]},
                         {'segment_id':3,'measurement_xy_px':[[j+7, .04*(j+7)**2+10] for j in range(6)]}],
            'join_proposals':[{'segment_A':1,'endpoint_A':'B','segment_B':2,'endpoint_B':'A'},
                              {'segment_A':1,'endpoint_A':'B','segment_B':3,'endpoint_B':'A'}]}


def make_manifest(tmp_path):
    entries=[]
    for split,run,count in [('train','source1',6),('inference','future2',1)]:
        for idx in range(count):
            result=make_result(f'{run}:{idx}')
            p=f'{run}_{idx}.json';(tmp_path/p).write_text(json.dumps(result))
            ent={'film_id':run,'run_id':run,'frame_id':f'{run}_{idx}','split':split,'p0_result':p}
            if split!='inference':
                rev={'schema':SCHEMA,'source_sha256_grayscale_pixels':result['source_sha256_grayscale_pixels'],
                     'pipeline':result['pipeline'],'result_sha256':result_digest(result),
                     'labels':[{'pair_key':pair_key(r),'label':'same_track' if j==0 else 'different_track','comment':''}
                               for j,r in enumerate(result['join_proposals'])]}
                v=f'{run}_{idx}_reviews.json';(tmp_path/v).write_text(json.dumps(rev));ent['review_json']=v
            entries.append(ent)
    f=tmp_path/'reviews_manifest.json';f.write_text(json.dumps({'schema':'s14_p0_pair_review_manifest_v1','entries':entries}))
    return f


def test_review_manifest_inference_without_expert(tmp_path):
    path=make_manifest(tmp_path)
    sets=m.load_review_manifest(path)
    assert len(sets['train'])==12 and len(sets['inference'])==2
    assert all(r['expert_label'] is None for r in sets['inference'])
    out=m.run(path,tmp_path/'out',source_run_id='source1')
    assert out['inference']['status']=='CALIBRATION_MISSING_ABSTAIN_ALL'
    assert not out['inference']['unattended_release_allowed']
    assert (tmp_path/'out/relation_model.json').exists()


def test_unknown_or_corrupt_review_rejected(tmp_path):
    path=make_manifest(tmp_path)
    d=json.loads(path.read_text())
    d['entries'][0]['run_id']='future2';d['entries'][0]['frame_id']='future2_0'
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='duplicate frame'):m.load_review_manifest(path)
