from pathlib import Path
import sys, importlib.util,pickle,joblib,math,numpy as np,pandas as pd
ROOT=Path('/mnt/data/imagecasx_graphlira_bundle'); PDIR=ROOT/'graph_lira_mask_shape_context'/'presence_oof'; OUT=PDIR/'combo_cache';OUT.mkdir(exist_ok=True)
split=sys.argv[1]; angle=float(sys.argv[2]); jitter=float(sys.argv[3]); label=sys.argv[4]
spec=importlib.util.spec_from_file_location('c',ROOT/'run_graph_lira_large_scale.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
sc=pickle.load(open(ROOT/'graph_lira_large_scale'/'scenes_full.pkl','rb'));local=joblib.load(ROOT/'graph_lira_selective'/'models.joblib');pm,jm=local['pair_model'],local['junction_model'];S=sc[split]
Z=np.load(ROOT/'graph_lira_mask_shape_context'/f'mask_features_{label}.npz',allow_pickle=True);assert list(map(str,Z['scene_ids']))==[str(s['scene_id']) for s in S]
models={}
for v in ['geometry','geometry_plus_mask']:
 X=Z['Xgeom'] if v=='geometry' else np.c_[Z['Xgeom'],Z['Xmask']]
 models[v]=(joblib.load(PDIR/f'pair_{v}.joblib').predict_proba(X)[:,1],joblib.load(PDIR/f'junction_{v}.joblib').predict_proba(X)[:,1])
cache,*_=c.score_all(S,pm,jm,angle,jitter);print('scored',label,len(S),flush=True)
rows=[]
for i,s in enumerate(S):
 pp,jp,_=cache[s['scene_id']]; row={'scene_id':s['scene_id'],'scan_id':s['scan_id'],'kind':s['kind'],'junction_degree':s['junction_degree'],'junction_segment':s['junction_segment']}
 for v,(ep,ej) in models.items():row[f'p_{v}']=ep[i];row[f'j_{v}']=ej[i]
 for wp,wj in [(0,0),(1,0),(0,1),(1,1)]:
  selP=selJ=None
  if wp and pp:selP=max(pp.items(),key=lambda z:z[1])[0]
  if wj and jp:selJ=max(jp.items(),key=lambda z:z[1])[0]
  if selP is not None and selJ is not None and set(selP)&set(selJ):
   best=None
   for pe,ps in pp.items():
    for je,js in jp.items():
     if set(pe)&set(je):continue
     score=math.log(max(ps,1e-8))+math.log(max(js,1e-8))
     if best is None or score>best[0]:best=(score,pe,je)
   if best is not None:selP,selJ=best[1],best[2]
   else:
    if (max(pp.values()) if pp else -1)>=(max(jp.values()) if jp else -1):selJ=None
    else:selP=None
  ev=c.evaluate_pred(s,selJ,selP);key=f'{wp}{wj}';row[f'exact_{key}']=ev['exact'];row[f'false_{key}']=ev['false_scene'];row[f'incomplete_{key}']=ev['incomplete']
 rows.append(row)
pd.DataFrame(rows).to_csv(OUT/f'{label}.csv',index=False);print('saved',label,flush=True)
