import pytest,json
from bubble_chamber.s28_ghost_review import add_path_ghosts

def data():
    photos=[{'crop_id':f'C{i}', 'sha256':'a'*64} for i in range(39)]
    p={'schema':'s28_genuine_h50z_operator_anchor_to_unverified_2D_path_v1','records':[]}
    for i in range(9):
        item={'path_points_original_xy':[[5,5],[8,9]],'human_confirmed_track':False,'automatic_physical_join_allowed':False}
        p['records'].append({'crop_id':f'C{i}','source_crop_sha256':'a'*64,
           'genuine_OPERATOR_start_xy_in_crop':[255,256],
           'classical_candidate_paths':[item],'original_AZUNet_assisted_candidate_paths':[item]})
    payload={'schema':'s26_review_visual_payload_v1','photos':photos}
    code="function update(){byId('background').src=entries[active].image_modes[byId('mode').value];redraw();}"
    html='<html><body><div id="scene"><canvas id="annotations"></canvas></div><script type="application/json" id="payload">'+json.dumps(payload)+'</script><script type="application/json" id="s27-queue-json">{}</script><script>'+code+'</script></body></html>'
    return html,p

def test_dashed_ghost_cannot_be_serialized_as_human():
    html,p=data();out=add_path_ghosts(html,p)
    assert 'id="s28-ghost"' in out
    assert 's28-payload' in out
    assert 'manually' not in out  # original test template has no real reviewer export
    assert "if(typeof window.s28RenderGhost" in out
    assert 'annotations[active].push' not in out
    assert 'stroke-dasharray' in out

def test_bad_source_rejected():
    html,p=data();p['records'][0]['source_crop_sha256']='f'*64
    with pytest.raises(ValueError,match='SHA'):add_path_ghosts(html,p)
    html,p=data();p['records'][0]['classical_candidate_paths'][0]['human_confirmed_track']=True
    with pytest.raises(ValueError,match='pre-certified'):add_path_ghosts(html,p)
