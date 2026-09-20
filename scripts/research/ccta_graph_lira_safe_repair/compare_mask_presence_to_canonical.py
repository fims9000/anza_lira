from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/mnt/data/imagecasx_graphlira_bundle'); C=ROOT/'graph_lira_mask_shape_context'/'presence_oof'/'combo_cache'; OUT=ROOT/'graph_lira_mask_shape_context'/'presence_oof'/'structural_robust';OUT.mkdir(exist_ok=True)
pt,jt=.96,.90

def eval_arr(D):
 p=D['p_geometry_plus_mask'].to_numpy()>=pt;j=D['j_geometry_plus_mask'].to_numpy()>=jt;key=p.astype(int)*10+j.astype(int);ex=np.zeros(len(D));fa=np.zeros(len(D));inc=np.zeros(len(D))
 for k,s in [(0,'00'),(10,'10'),(1,'01'),(11,'11')]:
  m=key==k;ex[m]=D.loc[m,f'exact_{s}'];fa[m]=D.loc[m,f'false_{s}'];inc[m]=D.loc[m,f'incomplete_{s}']
 return ex,fa,inc
rows=[];rng=np.random.default_rng(20260920)
for ev in ['test30','test45']:
 D=pd.read_csv(C/f'{ev}.csv');ex,fa,inc=eval_arr(D);M=D[['scene_id','scan_id','kind']].copy();M['exact_mask']=ex;M['false_mask']=fa;M['incomplete_mask']=inc;M.to_csv(OUT/f'mask_presence_5pct_{ev}_predictions.csv',index=False)
 canon=pd.read_csv(ROOT/'graph_lira_relation_type'/f'{ev}_predictions.csv')[['scene_id','scan_id','exact','false_scene','incomplete']].rename(columns={'exact':'exact_canon','false_scene':'false_canon','incomplete':'incomplete_canon'})
 Z=canon.merge(M,on=['scene_id','scan_id']);pts=sorted(Z.scan_id.unique());A=[]
 for pid in pts:
  q=Z[Z.scan_id==pid];A.append([len(q),q.exact_canon.sum(),q.exact_mask.sum(),q.false_canon.sum(),q.false_mask.sum(),q.incomplete_canon.sum(),q.incomplete_mask.sum()])
 A=np.asarray(A,float);P=len(A);den=A[:,0].sum()
 for metric,ic,im in [('exact',1,2),('false_scene',3,4),('incomplete',5,6)]:
  obs=A[:,im].sum()/den-A[:,ic].sum()/den;s=np.empty(10000)
  for k in range(len(s)):
   W=A[rng.integers(0,P,size=P)];d=W[:,0].sum();s[k]=W[:,im].sum()/d-W[:,ic].sum()/d
  lo,hi=np.quantile(s,[.025,.975]);rows.append({'eval':ev,'metric':metric,'mask_minus_canonical':obs,'ci_low':lo,'ci_high':hi,'n_patients':P})
B=pd.DataFrame(rows);B.to_csv(OUT/'mask_presence_5pct_vs_canonical_bootstrap.csv',index=False);print(B.to_string(index=False))
