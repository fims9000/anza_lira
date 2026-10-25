#!/usr/bin/env python3
"""Source-locked geometry/second-order curve extraction from UNVERIFIED ANZA/U-Net masks.

No physical 3D inference, no track ID, no momentum/mass/dE/dx calibration.
Curve coefficients and continuation links are review suggestions ONLY.
"""
from __future__ import annotations
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np,cv2,torch
from skimage.morphology import skeletonize
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s26_visual_inspection import model_for_checkpoint,tiled_predict
from bubble_chamber.pilot import extract_skeleton_segments,fit_quadratic,arc_sample,propose_joins
from bubble_chamber.photometry import sample_track_profiles


def segment_consensus(gray:np.ndarray,preds:dict[str,np.ndarray],thresholds:dict[str,float],min_len=16):
    if gray.shape!=(512,512):raise ValueError('Only original 512-crop coordinate system')
    if set(preds)!={'unet','azunet'}:raise ValueError('two independent source-locked models required')
    for k in ('unet','azunet'):
        if preds[k].shape!=gray.shape or not np.isfinite(preds[k]).all():raise ValueError('invalid probability map')
        if not 0<=thresholds[k]<=1.000001:raise ValueError('threshold invalid')
    # No dilation/closing: that would silently merge unrelated fragments.
    mask=np.asarray((preds['unet']>=thresholds['unet'])&(preds['azunet']>=thresholds['azunet']),bool)
    n,components,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),connectivity=8)
    sizes=stats[:,cv2.CC_STAT_AREA].copy()
    sizes[0]=0
    cleaned=(mask&(sizes[components]>=9))
    skeleton=skeletonize(cleaned)
    pieces=extract_skeleton_segments(skeleton,min_length_px=min_len)
    pieces=sorted(pieces,key=lambda z:-float(np.linalg.norm(np.diff(z,axis=0),axis=1).sum()))
    pieces=pieces[:200]
    distance=cv2.distanceTransform(cleaned.astype('uint8'),cv2.DIST_L2,3)
    local_bkg=cv2.GaussianBlur(gray,(0,0),5)
    records=[];raw=[]
    for chain in pieces:
        if len(chain)<5:continue
        fitted=fit_quadratic(chain)
        arc=arc_sample(chain,min(40,max(14,len(chain)//2)))
        sy=np.clip(chain[:,0].round().astype(int),0,511)
        sx=np.clip(chain[:,1].round().astype(int),0,511)
        profiles=sample_track_profiles(gray,arc[:,::-1],radius_px=8,min_darkness=6.)
        valid=[p for p in profiles if p['status']=='valid']
        records.append({'candidate_id':len(records),'length_px':round(float(np.linalg.norm(np.diff(chain,axis=0),axis=1).sum()),2),
            'measurement_xy_px':arc[:,::-1].tolist(),'quadratic_pca_local':fitted,
            'mean_width_px_uncalibrated':round(float((2*distance[sy,sx]).mean()),3),
            'mean_darkness_contrast_uncalibrated':round(float(np.maximum(local_bkg[sy,sx].astype(float)-gray[sy,sx].astype(float),0).mean()),3),
            'observed_transverse_photo_profiles':profiles,
            'valid_transverse_profile_fraction':round(len(valid)/max(len(profiles),1),4),
            'real_photo_fwhm_px_median_if_valid':round(float(np.median([p['width_fwhm_px'] for p in valid])),4) if valid else None,
            'width_cannot_be_interpreted_as_dedx':True,
            'human_confirmed_trace':False,'reconstruction':'PROPOSAL_ONLY'})
        raw.append(chain)
    questions=propose_joins(raw,max_gap_px=32,max_join_angle_deg=32,max_candidates_per_endpoint=3)
    return records,questions,cleaned


def uncertain_junctions(mask):
    """Pixel-space crossing/junction candidates, NEVER verified secondary stars."""
    skel=skeletonize(np.asarray(mask,bool))
    kern=np.ones((3,3),np.uint8);kern[1,1]=0
    deg=cv2.filter2D(skel.astype('uint8'),cv2.CV_16S,kern,borderType=cv2.BORDER_CONSTANT)
    nodes=(skel & (deg>=3)).astype('uint8')
    n,lab,stat,cent=cv2.connectedComponentsWithStats(nodes,8)
    out=[]
    for i in range(1,n):
        x,y=cent[i]
        if 4<=x<mask.shape[1]-4 and 4<=y<mask.shape[0]-4:
            out.append({'x_px':round(float(x),2),'y_px':round(float(y),2),
                        'degree_8connected_proxy':int(deg[lab==i].max()),
                        'human_verified_secondary_vertex':False,
                        'semantic_status':'CROSSING_OR_JUNCTION_ONLY_NOT_PROVEN_INTERACTION'})
    return out


def run(args):
    torch.set_num_threads(2)
    models={k:model_for_checkpoint(getattr(args,k)) for k in ('unet','azunet')}
    r=json.loads(args.report.read_text())
    gc=json.loads(args.guided_report.read_text())
    th={'unet':r['methods']['unet_local']['validation']['threshold'],
        'azunet':gc['val_005_calibration']['threshold']}
    args.out.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in sorted(args.crops.glob('H50Z_030_01_*_start*.png')):
        im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        preds={k:tiled_predict(im,*mod) for k,mod in models.items()}
        fragments,links,mask=segment_consensus(im,preds,th)
        alternatives={}
        for key in ('unet','azunet'):
            # Independent low-confidence routing for crops where consensus is empty.
            aux,aux_links,_=segment_consensus(im,{'unet':preds[key],'azunet':preds[key]},
                                            {'unet':th[key],'azunet':th[key]})
            alternatives[key]={'status':'SOLO_MODEL_REVIEW_ONLY','fragments':aux,'join_questions':aux_links,
                               'automatic_joins':0}
        junctions=uncertain_junctions(mask)
        id=path.stem;result={'schema':'s26_real_photo_fragment_proposals_v1',
              'crop_id':id,'source_crop_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'model_score_thresholds_photo005_synthetic_marks':th,
              'pixels_are_real_photo':True,'model_supervision_is_synthetic':True,
              'real_human_confirmed_track_pixels':0,'real_human_confirmed_pairs':0,
              'physical_links_authorized':0,'fits_in_pixel_coordinates_not_3D':True,
              'segment_count':len(fragments),'join_question_count':len(links),
              'segments':fragments,'join_questions':links,'possible_crossing_or_secondary_star_questions':junctions,
              'fallback_unverified_single_model_candidates':alternatives}
        (args.out/f'{id}_geometry_questions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        rgb=cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
        for f in fragments:
            coords=np.rint(np.asarray(f['measurement_xy_px'])).astype('int32')
            cv2.polylines(rgb,[coords.reshape(-1,1,2)],False,(50,225,90),1,cv2.LINE_AA)
        for q in links[:30]:
            a=fragments[q['segment_A']]['quadratic_pca_local'];b=fragments[q['segment_B']]['quadratic_pca_local']
            end1=a['endpoint_A_xy_px'] if q['endpoint_A']=='A' else a['endpoint_B_xy_px']
            end2=b['endpoint_A_xy_px'] if q['endpoint_B']=='A' else b['endpoint_B_xy_px']
            cv2.line(rgb,tuple(np.rint(end1).astype(int)),tuple(np.rint(end2).astype(int)),(255,165,80),1,cv2.LINE_AA)
        cv2.imwrite(str(args.out/f'{id}_geometry_overlay.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        records.append({'crop_id':id,'segments':len(fragments),'possible_links_review_only':len(links),
           'fallback_unet_segments':len(alternatives['unet']['fragments']),
           'fallback_azunet_segments':len(alternatives['azunet']['fragments']),
           'unverified_junction_candidates':len(junctions),
           'selected_fraction':round(float(mask.mean()),5)})
        print('GEOMETRY',id,len(fragments),len(links),flush=True)
    (args.out/'all_crops_summary.json').write_text(json.dumps({'schema':'s26_real_track_geometry_status_v1',
             'source_crops':records,'physical_connections_completed':0,
             'human_confirmed_lines':0,'human_confirmed_pairs':0},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--crops',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--guided-report',type=Path,required=True);p.add_argument('--unet',type=Path,required=True)
    p.add_argument('--azunet',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a)
