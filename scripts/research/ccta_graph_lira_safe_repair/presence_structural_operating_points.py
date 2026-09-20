from pathlib import Path
import numpy as np,pandas as pd,itertools
ROOT=Path('/mnt/data/imagecasx_graphlira_bundle/graph_lira_mask_shape_context/presence_oof'); C=ROOT/'combo_cache'; OUT=ROOT/'structural_robust';OUT.mkdir(exist_ok=True)
V30=pd.read_csv(C/'val30.csv');V45=pd.read_csv(C/'val45.csv');T30=pd.read_csv(C/'test30.csv');T45=pd.read_csv(C/'test45.csv')

def eval_thr(D,v,pt,jt):
 p=D[f'p_{v}'].to_numpy()>=pt;j=D[f'j_{v}'].to_numpy()>=jt;key=p.astype(int)*10+j.astype(int)
 ex=np.zeros(len(D));fa=np.zeros(len(D));inc=np.zeros(len(D))
 for k,s in [(0,'00'),(10,'10'),(1,'01'),(11,'11')]:
  m=key==k;ex[m]=D.loc[m,f'exact_{s}'];fa[m]=D.loc[m,f'false_{s}'];inc[m]=D.loc[m,f'incomplete_{s}']
 return ex,fa,inc
ths=np.unique(np.r_[np.arange(.50,.951,.025),np.arange(.96,.991,.01),.995])
rows=[];settings=[]
for budget in [.05,.075,.10]:
 for v in ['geometry','geometry_plus_mask']:
  best=None
  for pt,jt in itertools.product(ths,ths):
   e30,f30,i30=eval_thr(V30,v,pt,jt);e45,f45,i45=eval_thr(V45,v,pt,jt)
   a={'exact':e30.mean(),'false':f30.mean(),'inc':i30.mean()};b={'exact':e45.mean(),'false':f45.mean(),'inc':i45.mean()}
   if a['false']<=budget and b['false']<=budget:
    key=(min(a['exact'],b['exact']),(a['exact']+b['exact'])/2,-max(a['false'],b['false']))
    if best is None or key>best[0]:best=(key,float(pt),float(jt),a,b)
  if best is None:continue
  _,pt,jt,a,b=best;settings.append((budget,v,pt,jt))
  for ev,D in [('val30',V30),('val45',V45),('test30',T30),('test45',T45)]:
   ex,fa,inc=eval_thr(D,v,pt,jt);rows.append({'budget':budget,'variant':v,'pair_thr':pt,'junction_thr':jt,'eval':ev,'exact':ex.mean(),'false_scene':fa.mean(),'incomplete':inc.mean(),'n':len(D)})
R=pd.DataFrame(rows);R.to_csv(OUT/'operating_points.csv',index=False);print(R.to_string(index=False))
