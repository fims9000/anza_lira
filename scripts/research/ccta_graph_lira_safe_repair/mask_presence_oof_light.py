from pathlib import Path
import importlib.util,pickle,gc,joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path('/mnt/data/imagecasx_graphlira_bundle'); OUT=ROOT/'graph_lira_mask_shape_context'/'presence_oof';OUT.mkdir(parents=True,exist_ok=True); SEED=20260920
spec=importlib.util.spec_from_file_location('c',ROOT/'run_graph_lira_large_scale.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
sc=pickle.load(open(ROOT/'graph_lira_large_scale'/'scenes_full.pkl','rb'))
train_ids=np.array(c.SPLIT_IDS['train']);fo={int(x):i%3 for i,x in enumerate(train_ids)};Xo=[];scene_order=[]
for fold in range(3):
 z=np.load(ROOT/'graph_lira_presence_crossfit'/f'oof_fold{fold}.npz');hold=[s for s in sc['train'] if fo[s['scan_id']]==fold];Xp,Xj=z['Xp'],z['Xj'];extra=np.c_[Xp[:,0]-Xj[:,0],Xp[:,0]+Xj[:,0],Xp[:,7],Xj[:,7],Xp[:,16],Xj[:,16]].astype(np.float32);Xo.append(np.c_[Xp,Xj,extra].astype(np.float32));scene_order.extend(hold)
Xo=np.concatenate(Xo);ids=[str(s['scene_id']) for s in scene_order]
parts={}
for i in range(4):
 z=np.load(ROOT/'graph_lira_mask_shape_context'/f'mask_features_train30_chunk{i}.npz',allow_pickle=True)
 for k,sid in enumerate(z['scene_ids']):parts[str(sid)]=z['Xmask'][k].astype(np.float32)
Xm=np.stack([parts[sid] for sid in ids]).astype(np.float32);del parts
yp=np.array([int(s['true_pair'] is not None) for s in scene_order],np.uint8);yj=np.array([int(s['true_junction'] is not None) for s in scene_order],np.uint8)
E={}
for lab,file in [('val30','mask_features_val30.npz'),('val45','mask_features_val45.npz'),('test30','mask_features_test30.npz'),('test45','mask_features_test45.npz')]:
 z=np.load(ROOT/'graph_lira_mask_shape_context'/file,allow_pickle=True);S=sc['val'] if lab.startswith('val') else sc['test'];E[lab]=(z['Xgeom'].astype(np.float32),z['Xmask'].astype(np.float32),np.array([int(s['true_pair'] is not None) for s in S],np.uint8),np.array([int(s['true_junction'] is not None) for s in S],np.uint8))
rows=[];ops=[]
for rel,ytr in [('pair',yp),('junction',yj)]:
 for v in ['geometry','mask','geometry_plus_mask']:
  A=Xo if v=='geometry' else Xm if v=='mask' else np.concatenate([Xo,Xm],axis=1)
  print('fit',rel,v,A.shape,flush=True)
  mdl=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,learning_rate=.06,l2_regularization=2.0,class_weight='balanced',random_state=SEED);mdl.fit(A,ytr)
  probs={}
  for lab,(xg,xm,ypv,yjv) in E.items():
   X=xg if v=='geometry' else xm if v=='mask' else np.concatenate([xg,xm],axis=1);y=ypv if rel=='pair' else yjv;p=mdl.predict_proba(X)[:,1];probs[lab]=(y,p)
   rows.append(dict(relation=rel,variant=v,eval=lab,auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),n=len(y),positive_rate=float(y.mean())))
  cand=np.unique(np.r_[np.linspace(.05,.99,95),np.quantile(probs['val30'][1],np.linspace(.05,.99,40)),np.quantile(probs['val45'][1],np.linspace(.05,.99,40))]);best=None
  for th in cand:
   ms=[];ok=True
   for lab in ['val30','val45']:
    y,p=probs[lab];pr=p>=th;neg=y==0;pos=y==1;fpr=float(pr[neg].mean());tpr=float(pr[pos].mean());prec=float(y[pr].mean()) if pr.any() else 1.;ms.append((fpr,tpr,prec));ok &= fpr<=.05
   if ok:
    key=(min(ms[0][1],ms[1][1]),(ms[0][1]+ms[1][1])/2,(ms[0][2]+ms[1][2])/2,-th)
    if best is None or key>best[0]:best=(key,float(th))
  th=best[1]
  for lab in ['val30','val45','test30','test45']:
   y,p=probs[lab];pr=p>=th;neg=y==0;pos=y==1;ops.append(dict(relation=rel,variant=v,threshold=th,eval=lab,fpr=float(pr[neg].mean()),tpr=float(pr[pos].mean()),precision=float(y[pr].mean()) if pr.any() else 1.,accept_rate=float(pr.mean()),n=len(y)))
  joblib.dump(mdl,OUT/f'{rel}_{v}.joblib');del mdl,A,probs;gc.collect()
R=pd.DataFrame(rows);O=pd.DataFrame(ops);R.to_csv(OUT/'ranking_metrics.csv',index=False);O.to_csv(OUT/'operating_metrics.csv',index=False);print('\nRANK\n',R.to_string(index=False));print('\nOPS\n',O.to_string(index=False))
