#!/usr/bin/env python3
"""Controlled artificial gaps on REAL CRACKS expert masks: oracle-mask diagnostic.

Does not test trained ANZA/LIRA; never reports this as complete model E2E.
"""
import argparse, hashlib, io, json, re, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

COLORS={"blue":(31,119,180),"green":(44,160,44),"orange":(255,127,14),"white":(255,255,255)}
def png(b):
    with Image.open(io.BytesIO(b)) as im:return np.asarray(im.convert("RGB"))
def sections(z,expert=False):
    out={}
    for item in z.namelist():
        if item.endswith("/"):continue
        m=re.search(r"section_(\d+)\.png$",item,re.I)
        if m and (not expert or "expert" in item.lower().split("/")):out.setdefault(int(m.group(1)),item)
    return out
def paths_from_mask(mask):
    sk=skeletonize(mask)
    kernel=np.ones((3,3),np.int8);kernel[1,1]=0
    deg=ndi.convolve(sk.astype(np.int8),kernel,mode="constant")
    branched=ndi.binary_dilation(sk & (deg>=3),iterations=5)
    sk[branched]=False
    sk[:8]=False;sk[-8:]=False;sk[:,:8]=False;sk[:,-8:]=False
    labels,n=ndi.label(sk,structure=np.ones((3,3),np.int8))
    for ix,box in enumerate(ndi.find_objects(labels),1):
        if box is None:continue
        region=(labels[box]==ix)
        if region.sum()<50:continue
        rd=ndi.convolve(region.astype(np.int8),kernel,mode="constant")
        ends=np.argwhere(region & (rd==1))
        if len(ends)!=2 or (rd[region]>=3).any():continue
        path=[tuple(map(int,ends[0]))];prev=None
        for _ in range(int(region.sum())-1):
            y,x=path[-1]
            opts=[(y+a,x+b) for a in (-1,0,1) for b in (-1,0,1)
                  if (a or b) and 0<=y+a<region.shape[0] and 0<=x+b<region.shape[1]
                  and region[y+a,x+b] and (y+a,x+b)!=prev]
            if len(opts)!=1 or opts[0] in path:break
            prev=path[-1];path.append(opts[0])
        if len(path)==int(region.sum()):
            oy,ox=box[0].start,box[1].start
            yield [(y+oy,x+ox) for y,x in path]
def split(section):
    block=section//20
    bucket=int.from_bytes(hashlib.sha256(f"cracks-real-gap-v1:{block}".encode()).digest()[:8],"big")%10
    return ("test" if bucket in (0,1) else "calibration" if bucket==2 else "development",block)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",default="data/cracks/raw/images.zip")
    p.add_argument("--annotations",default="data/cracks/raw/Fault segmentations.zip")
    p.add_argument("--out",default="results/cracks_real_controlled_v1/oracle_mask_audit.json")
    p.add_argument("--max-sections",type=int,default=400)
    args=p.parse_args();out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    report={"protocol":"CRACKS_REAL_ORACLE_MASK_GAP_AUDIT_V1","full_ANZA_LIRA_e2e":False,
            "real_seismic_sections_used":False,"note":"Ground-truth mask used to choose and evaluate artificial gaps. Not a learned-model evaluation.",
            "counts_by_split":{},"samples":[]}
    try:
        with zipfile.ZipFile(args.images) as im,zipfile.ZipFile(args.annotations) as an:
            images=sections(im);labels=sections(an,expert=True)
            ids=sorted(set(images)&set(labels))[:args.max_sections]
            if not ids:raise RuntimeError("No matching section_XXX.png expert and seismic image pairs")
            counters=Counter()
            for sid in ids:
                rgb=png(an.read(labels[sid]));image=png(im.read(images[sid]))
                if image.shape!=rgb.shape:raise RuntimeError(f"section {sid}: image-mask shape mismatch")
                if any(tuple(x) not in COLORS.values() for x in np.unique(rgb.reshape(-1,3),axis=0)):raise RuntimeError(f"section {sid}: unrecognized colors")
                confident=np.all(rgb==COLORS["blue"],axis=-1)
                uncertain=np.all(rgb==COLORS["green"],axis=-1)
                split_name,block=split(sid)
                counters[(split_name,"sections")]+=1
                count=0
                for path in paths_from_mask(confident & ~ndi.binary_dilation(uncertain,iterations=5)):
                    if len(path)<60:
                        counters[("audit","short_line")]+=1
                        continue
                    middle=len(path)//2;middle_path=path[middle-6:middle+6]
                    coords=np.asarray(middle_path)
                    ymin,ymax=coords[:,0].min()-3,coords[:,0].max()+4
                    xmin,xmax=coords[:,1].min()-3,coords[:,1].max()+4
                    if ymin<0 or xmin<0 or ymax>confident.shape[0] or xmax>confident.shape[1]:
                        counters[("audit","out_of_image")]+=1
                        continue
                    outside=confident[ymin:ymax,xmin:xmax].copy()
                    if uncertain[ymin:ymax,xmin:xmax].any():
                        counters[("audit","ambiguous_annotation_near_gap")]+=1
                        continue
                    masked=confident.copy();masked[ymin:ymax,xmin:xmax]=False
                    # Anchor endpoints must remain outside the whole padded rectangle.
                    inside=lambda pt:ymin<=pt[0]<ymax and xmin<=pt[1]<xmax
                    left=next((pt for pt in reversed(path[:middle]) if not inside(pt)),None)
                    right=next((pt for pt in path[middle:] if not inside(pt)),None)
                    if left is None or right is None:
                        counters[("audit","anchors_missing")]+=1
                        continue
                    e0,e1=left,right
                    path_support=np.zeros_like(confident)
                    for point in path:path_support[point]=True
                    permitted=ndi.binary_dilation(path_support,iterations=5)
                    if np.any(outside & ~permitted[ymin:ymax,xmin:xmax]):
                        counters[("audit","other_trace_in_gap")]+=1
                        continue
                    components,_=ndi.label(masked,structure=np.ones((3,3),np.uint8))
                    if not components[e0] or not components[e1]:
                        counters[("audit","anchors_not_visible")]+=1
                        continue
                    if components[e0]==components[e1]:
                        counters[("audit","still_connected_after_gap")]+=1
                        continue
                    counters[(split_name,"gaps")]+=1;count+=1
                    report["samples"].append({"section":sid,"block":block,"split":split_name,"gap_pixels":len(middle_path),
                      "intervention_rectangle":[int(ymin),int(ymax),int(xmin),int(xmax)],
                      "oracle_endpoints_distance":round(float(np.linalg.norm(np.array(e0)-np.array(e1))),3)})
                    if count>=2:break
            report["sections_with_image_and_expert_mask"]=len(ids)
            report["audit_rejections"]={k[1]:int(v) for k,v in counters.items() if k[0]=="audit"}
            report["real_seismic_sections_used"]=True
            report["counts_by_split"]={s:{"sections":counters[(s,"sections")],"eligible_gaps":counters[(s,"gaps")]} for s in ["development","calibration","test"]}
            report["status"]="ORACLE_MASK_AUDIT_COMPLETE"
    except Exception as exc:report["status"]="STOP_REAL_CRACKS_AUDIT";report["error"]=str(exc)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in report.items() if k!="samples"},ensure_ascii=False,indent=2))
    return 0 if report["status"]=="ORACLE_MASK_AUDIT_COMPLETE" else 2
if __name__=="__main__":raise SystemExit(main())
