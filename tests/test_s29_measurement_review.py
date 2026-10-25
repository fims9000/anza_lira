import json,pytest
from bubble_chamber.s29_measurement_review import add_candidate_metrics_panel

def test_s29_panel_preserves_canonical_source_and_human_review():
    base='''<html><div id="scene"></div><select id="s28-choice"></select>
    <script type="application/json" id="s28-payload">{"f":{"photo_sha256":"sha"}}</script>
    <script>window.s28RenderGhost=function(){};</script></body></html>'''
    rows=[]
    for i in range(4):
        for method in ('classical','original_AZ_assisted'):
            rows.append({'crop_id':'f','source_photo_crop_sha256':'sha',
                'method':method,'hypothesis_rank':i+1,'expert_approved':False,'physics_use':'REVIEW_ONLY',
                'photometry':{'fraction_with_dark_center':.55,'median_apparent_width_px':3,
                   'candidate_bubble_peaks_unverified':2},
                'geometry':{'rms_centerline_residual_px':1.,'source_start_xy':[2,3],
                'source_end_xy':[20,40],'x_coefficients_c_b_a':[1,2,3],
                'y_coefficients_c_b_a':[4,5,6]}})
    payload={'schema':'s29_source_locked_unverified_h50z_path_measures_v1','expert_confirmed_paths':0,'rows':rows}
    # 9 cropped scenes expected so use full list.
    ghosts={str(i):{'photo_sha256':'sha'} for i in range(9)}
    expanded=[]
    for i in range(9):
        expanded.extend([dict(r,crop_id=str(i)) for r in rows])
    payload['rows']=expanded
    base=base.replace('{"f":{"photo_sha256":"sha"}}',json.dumps(ghosts))
    out=add_candidate_metrics_panel(base,payload)
    assert 's29-values' in out and 's28-choice' in out
    assert 'expert_labels_generated' not in out
    assert out.count('</body></html>')==1
    payload['rows'][0]['expert_approved']=True
    with pytest.raises(ValueError):add_candidate_metrics_panel(base,payload)
