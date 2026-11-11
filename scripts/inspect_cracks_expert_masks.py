#!/usr/bin/env python3
"""Inspect official CRACKS expert mask topology before choosing gap criteria."""
import io,json,zipfile,re,collections
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from cracks_real_oracle_gap_audit import sections,COLORS

def load(z,name):
    with Image.open(io.BytesIO(z.read(name))) as im:return np.asarray(im.convert("RGB"))
def main():
    p=Path("results/cracks_real_controlled_v1/expert_topology_diagnostics.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile("data/cracks/raw/images.zip") as iz,zipfile.ZipFile("data/cracks/raw/Fault segmentations.zip") as az:
        images=sections(iz);annotations=sections(az,expert=True);ids=sorted(images.keys()&annotations.keys())
        results=[]
        for sid in ids:
            rgb=load(az,annotations[sid]);positive=np.all(rgb==COLORS["blue"],axis=2)
            uncertain=np.all(rgb==COLORS["green"],axis=2)
            binary=skeletonize(positive)
            k=np.ones((3,3),dtype=np.uint8);k[1,1]=0
            deg=ndi.convolve(binary.astype(np.uint8),k,mode="constant")
            lab,n=ndi.label(binary,structure=np.ones((3,3),np.uint8))
            comps=np.bincount(lab[binary],minlength=n+1)[1:]
            clean=binary & ~ndi.binary_dilation(binary&(deg>=3),iterations=5)
            clab,cn=ndi.label(clean,structure=np.ones((3,3),np.uint8))
            lens=np.bincount(clab[clean],minlength=cn+1)[1:]
            results.append({"section":sid,"shape":list(rgb.shape),"blue_pixels":int(positive.sum()),"green_pixels":int(uncertain.sum()),
              "skeleton_pixels":int(binary.sum()),"skeleton_degree1":int(np.count_nonzero(binary&(deg==1))),
              "skeleton_degree2":int(np.count_nonzero(binary&(deg==2))),
              "skeleton_degree3plus":int(np.count_nonzero(binary&(deg>=3))),
              "components_original":n,"components_ge20":int((comps>=20).sum()),"components_ge50":int((comps>=50).sum()),
              "clean_components_ge10":int((lens>=10).sum()),"clean_components_ge20":int((lens>=20).sum()),
              "clean_components_ge30":int((lens>=30).sum()),"clean_components_ge50":int((lens>=50).sum()),
              "largest_clean_component":int(lens.max()) if len(lens) else 0})
    result={"status":"REAL_EXPERT_MASK_TOPOLOGY_MEASURED","sections":len(results),
       "totals":{k:sum(row[k] for row in results) for k in results[0] if k not in ("section","shape")},
       "per_section":results}
    p.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"sections":result["sections"],"totals":result["totals"],
      "top_longest":sorted(results,key=lambda x:x["largest_clean_component"],reverse=True)[:8]},indent=2))
if __name__=="__main__":main()
