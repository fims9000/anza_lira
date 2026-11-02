"""Experimental mask-only nearest-surface portal proposals in 3D CCTA.

A broken coronary branch can have a valid gap between connected-component
surfaces but NO nearby degree-1 skeleton tips. CT28 train patient 956 exposed
that failure: 3D vessel masks separated by 0.81 mm, while extracted skeleton
tips were 14.24 mm apart and no candidate scene was generated under 12 mm.

This detector is an OPTIONAL, explicit candidate generator, NOT part of frozen
K10/15 selected validation. It never reads target segmentation, CT-derived
ground truth, image labels or centerlines. It proposes local surface endpoints
for downstream CT scoring and selective topology veto, never auto-connects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ccta_3d_endpoints import CONNECTIVITY, Endpoint


LPS_SIGN=np.asarray([-1.,-1.,1.])


@dataclass(frozen=True)
class SurfacePortalSettings:
    max_gap_mm:float=3.
    tangent_radius_mm:float=4.
    min_component_voxels:int=100
    min_neighborhood_voxels:int=12
    min_forward_alignment:float=.25
    max_components:int=20
    max_portals:int=32

    def __post_init__(self):
        if not (
            np.isfinite([self.max_gap_mm,self.tangent_radius_mm,
                         self.min_forward_alignment]).all()
            and 0<self.max_gap_mm<=12
            and 1<=self.tangent_radius_mm<=10
            and 0<=self.min_forward_alignment<=1
            and self.min_component_voxels>=12
            and self.min_neighborhood_voxels>=3
            and self.max_components>=2
            and self.max_portals>=2
        ):
            raise ValueError("Unsafe surface portal configuration")


def _physical_ras(voxels:np.ndarray,affine:np.ndarray)->np.ndarray:
    return (affine[:3,:3]@voxels.T).T+affine[:3,3]


def _surface_voxels(
    labels:np.ndarray,
    component:int,
    roi:tuple[slice,slice,slice],
)->np.ndarray:
    sub=labels[roi]==component
    boundary=sub & ~ndimage.binary_erosion(sub,structure=CONNECTIVITY)
    return np.argwhere(boundary).astype(int)+np.asarray([part.start for part in roi])


def _outward_tangent(
    point_ras:np.ndarray,
    opposite_ras:np.ndarray,
    component_ras:np.ndarray,
    component_tree:cKDTree,
    settings:SurfacePortalSettings,
)->tuple[np.ndarray,float]|None:
    indices=component_tree.query_ball_point(point_ras,r=settings.tangent_radius_mm)
    if len(indices)<settings.min_neighborhood_voxels:
        return None
    local=component_ras[np.asarray(indices,dtype=int)]
    # Local mask centroid estimates the inward direction, including at a
    # branch that has no degree-1 skeleton endpoint. It is NOT a learned
    # ground-truth direction, and can be unreliable near broad junctions.
    outward=point_ras-local.mean(axis=0)
    gap=opposite_ras-point_ras
    norm=float(np.linalg.norm(outward))
    gapnorm=float(np.linalg.norm(gap))
    if norm<1e-4 or gapnorm<1e-4:
        return None
    unit=outward/norm
    alignment=float(unit@(gap/gapnorm))
    if alignment<settings.min_forward_alignment:
        return None
    return unit,alignment


def propose_surface_portals(
    predicted_mask:np.ndarray,
    affine:np.ndarray,
    *,
    existing_endpoints:list[Endpoint],
    settings:SurfacePortalSettings=SurfacePortalSettings(),
)->tuple[list[Endpoint],dict]:
    """Identify unlabelled near-surface endpoint pairs between components.

    The component labels must match extract_endpoints() 26-connectivity. Every
    accepted component-pair produces exactly two new synthetic endpoints.
    No merge or path is performed. The downstream relation head is expected
    to abstain if the portal-like input is outside its training distribution.
    """
    if predicted_mask.ndim!=3 or predicted_mask.dtype!=bool:
        raise ValueError("Expected 3D predicted binary mask")
    if affine.shape!=(4,4) or not np.isfinite(affine).all() or (
        not np.allclose(affine[3],[0,0,0,1])
    ) or abs(np.linalg.det(affine[:3,:3]))<1e-10:
        raise ValueError("Nondegenerate physical CCTA affine required")
    if not predicted_mask.any():
        return [],{"candidate_component_pairs":0,"accepted_component_pairs":0,
                   "portal_endpoints":0,"reason":"empty_mask"}
    from scipy import ndimage
    roi=tuple(slice(int(v.min()),int(v.max())+1) for v in np.where(predicted_mask))
    # Limit label, KD-tree and erosion memory to foreground ROI.
    labeled,number=ndimage.label(predicted_mask[roi],structure=CONNECTIVITY)
    size=np.bincount(labeled.ravel(),minlength=number+1)
    usable=[c for c in range(1,number+1) if size[c]>=settings.min_component_voxels]
    if len(usable)>settings.max_components:
        raise ValueError("Too many disconnected vessel components for bounded surface search")
    stats={
        "protocol":"experimental_mask_only_surface_portal_v1",
        "input_nonzero_voxels":int(predicted_mask.sum()),
        "original_components":int(number),
        "usable_components":len(usable),
        "tiny_components_ignored":int(number-len(usable)),
        "candidate_component_pairs":0,
        "accepted_component_pairs":0,
        "portal_endpoints":0,
        "max_gap_mm":settings.max_gap_mm,
        "min_forward_alignment":settings.min_forward_alignment,
        "hypothesis":"surface portals expose broken branch fronts missed by skeleton degree-1 extraction",
        "warnings":[
            "Not part of original CT28 K10/15 frozen validation generator.",
            "Local centroid tangent is a proxy; could generate false bridges between close native arterial trees.",
            "Geometry/image relation head may be OOD; no automatic repair or clinical safety claim.",
        ],
        "pair_examples":[],
    }
    if len(usable)<2:
        return [],stats
    origin=np.asarray([part.start for part in roi])
    local_mask=predicted_mask[roi]
    data={}
    for component in usable:
        inside_local=np.argwhere(labeled==component)
        world_ijk=inside_local+origin
        world=_physical_ras(world_ijk,affine)
        surface=_surface_voxels(labeled,component,tuple(slice(0,n) for n in labeled.shape))
        surface+=origin
        surface_ras=_physical_ras(surface,affine)
        data[component]={
            "voxels":world_ijk,
            "world":world,
            "tree":cKDTree(world),
            "surface_ijk":surface,
            "surface_ras":surface_ras,
            "surface_tree":cKDTree(surface_ras),
        }

    proposals=[]
    for i,first in enumerate(usable):
        for second in usable[i+1:]:
            stats["candidate_component_pairs"]+=1
            a=data[first]; b=data[second]
            distances,nearest=b["surface_tree"].query(a["surface_ras"],k=1)
            index=int(np.argmin(distances))
            dist=float(distances[index])
            if not (0<dist<=settings.max_gap_mm):
                continue
            ia=int(index); ib=int(nearest[index])
            points=(a["surface_ras"][ia],b["surface_ras"][ib])
            tangent_a=_outward_tangent(points[0],points[1],a["world"],a["tree"],settings)
            tangent_b=_outward_tangent(points[1],points[0],b["world"],b["tree"],settings)
            if tangent_a is None or tangent_b is None:
                continue
            proposals.append({
                "first":first,"second":second,"distance":dist,
                "ijk_a":a["surface_ijk"][ia],
                "ijk_b":b["surface_ijk"][ib],
                "ras_a":points[0],"ras_b":points[1],
                "tangent_a":tangent_a[0],"tangent_b":tangent_b[0],
                "forward_a":tangent_a[1],"forward_b":tangent_b[1],
                "size_a":int(size[first]),"size_b":int(size[second]),
            })
    if 2*len(proposals)>settings.max_portals:
        raise ValueError("Too many close surface connections for bounded endpoint proposal")
    proposals.sort(key=lambda item:(item["distance"],item["first"],item["second"]))
    used_ids={e.endpoint_id for e in existing_endpoints}
    ports=[]
    for i,candidate in enumerate(proposals):
        ids=(f"P{i:04d}A",f"P{i:04d}B")
        if used_ids.intersection(ids):
            raise ValueError("Duplicate endpoint between skeleton and surface proposals")
        for k,component in enumerate((candidate["first"],candidate["second"])):
            voxel=candidate["ijk_a" if k==0 else "ijk_b"]
            pos=candidate["ras_a" if k==0 else "ras_b"]*LPS_SIGN
            tangent=candidate["tangent_a" if k==0 else "tangent_b"]*LPS_SIGN
            ports.append(Endpoint(
                endpoint_id=ids[k],component=int(component),
                voxel=tuple(int(x) for x in voxel),
                position_lps=tuple(float(x) for x in pos),
                tangent_lps=tuple(float(x) for x in tangent),
            ))
        stats["pair_examples"].append({
            "endpoint_ids":list(ids),
            "component_ids":[candidate["first"],candidate["second"]],
            "component_voxel_count":[candidate["size_a"],candidate["size_b"]],
            "minimum_surface_distance_mm":candidate["distance"],
            "outward_tangent_alignment":[candidate["forward_a"],candidate["forward_b"]],
            "voxel_ijk":[candidate["ijk_a"].tolist(),candidate["ijk_b"].tolist()],
        })
    stats["accepted_component_pairs"]=len(proposals)
    stats["portal_endpoints"]=len(ports)
    return ports,stats
