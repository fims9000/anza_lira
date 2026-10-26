"""S35 audit-friendly fragment-link benchmark on AUTHENTIC H50Z photo backgrounds.

2D tracks are artificially overlaid using shapes from unrelated visualMetro
film-070 operator measurements. Synthetic left/right relation truth is known
ONLY for these newly drawn strokes. The native H50Z photo is unlabeled.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class SyntheticFragmentScene:
    gray: np.ndarray                    # cropped native H50Z + injected dark fragments
    left: list[np.ndarray]              # candidate polyline fragment centerlines
    right: list[np.ndarray]
    correct_pairs: set[tuple[int, int]] # SYNTHETIC ONLY, never native physical truth
    background_file: str
    scenario: str
    source_frame_ids: tuple[str, str]
    injection_seed: int
    simulated_measurement_error_px: float
    extra_masks: dict[str,np.ndarray]


def extract_track_templates(manifest: dict, *, chosen_frames: set[str]) -> list[dict]:
    """Take operator-clicked 2D view 1 tracks; no false BMP correspondence."""
    rows=[]
    for frame in manifest['frames']:
        if frame['frame_id'] not in chosen_frames: continue
        for t in frame['tracks']:
            pts=np.asarray(t['measured_pixel_xy']['1'],dtype=np.float64)
            if t['n_points']>=9 and pts.shape[0]>=9 and np.isfinite(pts).all():
                delta=np.linalg.norm(np.diff(pts,axis=0),axis=1)
                if np.sum(delta)>400 and np.all(delta>0):
                    rows.append({'points':pts,'frame':frame['frame_id'],'id':t['track_id']})
    if len(rows)<24:raise ValueError('Insufficient measured track templates for frame-disjoint protocol')
    return rows


def frame_split(manifest: dict) -> tuple[set[str],set[str]]:
    frames=sorted((f['frame_id'] for f in manifest['frames']),
                  key=lambda name:sha256(name.encode()).hexdigest())
    split=max(15,int(round(.25*len(frames))))
    test=set(frames[:split]);train=set(frames[split:])
    if len(train)<50 or len(test)<15:raise ValueError('Invalid frame-disjoint visualMetro split')
    return train,test


def _shape_curve(template:dict, *, length:float=138.,center_y:float=100., tilt:float=0.) -> np.ndarray:
    """Normalize genuinely measured (but unrelated) 2D curvature to 192px crop."""
    pts=template['points']
    steps=np.linalg.norm(np.diff(pts,axis=0),axis=1)
    arclength=np.concatenate(([0.],np.cumsum(steps)))
    if arclength[-1]<1:raise ValueError('Degenerate source geometry')
    times=arclength/arclength[-1]
    dense=PchipInterpolator(times,pts,axis=0)(np.linspace(0.,1.,121))
    d=dense[-1]-dense[0];theta=np.arctan2(d[1],d[0]);c,s=np.cos(-theta),np.sin(-theta)
    rot=(dense-dense[0])@np.array([[c,-s],[s,c]]).T
    scale=length/max(float(rot[-1,0]),1.)
    rot=rot*scale
    u=np.linspace(0.,1.,121)
    # Scale transverse deviations to avoid unrealistic stereo-pixel curvature.
    off=(rot[:,1] - np.linspace(rot[0,1],rot[-1,1],len(rot)))
    off=np.clip(off,-20.,20.)
    return np.column_stack((25.+length*u, center_y+off*.65+tilt*(u-.5)))


def _draw_stroke(canvas:np.ndarray, mask:np.ndarray, pts:np.ndarray, *, ink:float, width:int):
    x=np.rint(pts).astype('int32').reshape(-1,1,2)
    stroke=np.zeros_like(mask,dtype=np.uint8)
    cv2.polylines(stroke,[x],False,1,width,lineType=cv2.LINE_8)
    canvas -= float(ink)*stroke
    mask |= stroke.astype(bool)


def make_scene(tile_path:Path,templates:list[dict],*,seed:int,scenario:str='parallel',size:int=192) -> SyntheticFragmentScene:
    """Two separate tracks each broken in half; extra false scratch candidate.

    Returns 2 true and 4 false pair labels. The `correct_pairs` belong only
    to the freshly injected synthetic strokes (not background native photo).
    """
    if scenario not in {'parallel','crossing','scratched','empty_gap'} or size!=192:
        raise ValueError('Unsupported preregistered S35 synthetic scenario')
    photo=Path(tile_path)
    im=cv2.imread(str(photo),cv2.IMREAD_GRAYSCALE)
    if im is None or im.shape!=(512,512):raise ValueError('Need original authenticated 512px H50Z crop')
    rng=np.random.default_rng(seed)
    x=int(rng.integers(0,512-size+1));y=int(rng.integers(0,512-size+1))
    crop=im[y:y+size,x:x+size].astype(np.float32)
    fg=np.zeros((size,size),dtype=bool);scratchmask=np.zeros_like(fg)
    t0=templates[int(rng.integers(len(templates)))];t1=templates[int(rng.integers(len(templates)))]
    length=float(rng.uniform(113,143))
    c_y=float(rng.uniform(72,114))
    first=_shape_curve(t0,length=length,center_y=c_y,tilt=float(rng.uniform(-12,12)))
    second=_shape_curve(t1,length=length,center_y=c_y,tilt=float(rng.uniform(-12,12)))
    u=np.linspace(0,1,121)
    # Distinct photometric profiles provide a possible (not guaranteed) cue.
    offset=float(rng.choice([-1,1])*rng.uniform(5,14))
    offsetfield=offset if scenario!='crossing' else offset*(1.5-3*u)
    second[:,1] += offsetfield
    # Source 2D curvature is used; neither geometry is registered to photo H50Z.
    if not (np.all((first>12)&(first<180)) and np.all((second>12)&(second<180))):
        raise ValueError('Source track projected beyond synthetic crop')
    tracks=[first,second]
    gap_start=int(rng.integers(48,55));gap_end=int(gap_start+rng.integers(9,20))
    gap_end=min(gap_end,77)
    left=[];right=[]
    for k,curve in enumerate(tracks):
        ink=float(rng.uniform(27,75));width=int(rng.integers(1,4))
        a=curve[:gap_start].copy();b=curve[gap_end:].copy()
        _draw_stroke(crop,fg,a,ink=ink,width=width)
        _draw_stroke(crop,fg,b,ink=ink,width=width)
        if scenario!='empty_gap' and rng.random()<.70:
            # Weak bubble residues in missing interval, NOT the original trace.
            middle=curve[gap_start:gap_end]
            for pt in middle[::int(rng.integers(3,6))]:
                cv2.circle(crop,tuple(np.rint(pt).astype(int)),1,float(ink*rng.uniform(.04,.25)),-1)
        left.append(a);right.append(b)
    # Near-track distractor can look like a plausible continuation.
    target=tracks[int(rng.integers(0,2))]
    shift=float(rng.choice([-1,1])*rng.uniform(1.5,11.))
    decoy=target[gap_end:gap_end+31].copy()
    decoy[:,1]+=shift + rng.uniform(-4,4)*np.linspace(0,1,len(decoy))
    if scenario=='scratched':decoy[:,1]+=rng.uniform(-2,2)
    _draw_stroke(crop,scratchmask,decoy,ink=float(rng.uniform(35,90)),width=int(rng.integers(1,4)))
    right.append(decoy)
    for _ in range(5):
        a=rng.integers(8,size-8,size=2);b=np.clip(a+rng.integers(-22,23,size=2),3,size-4)
        cv2.line(crop,tuple(a),tuple(b),-float(rng.uniform(5,35)),1,lineType=cv2.LINE_8)
    gray=cv2.GaussianBlur(np.clip(crop,0,255).astype('uint8'),(0,0),.35)
    if len(left)!=2 or len(right)!=3:raise AssertionError('Bad controlled candidate topology')
    # Simulate IMPERFECT centerline extraction on real photographs. Scoring
    # perfect known injection centerlines makes geometry trivially AUROC=1;
    # estimates must instead have endpoint uncertainty and noisy direction.
    jitter=float(rng.choice([.6,1.3,2.0,3.0]))
    def observed(poly,at_end):
        pts=poly.copy()
        # Slow drift along the local normal plus a random endpoint bias.
        tangent=np.gradient(pts,axis=0)
        tangnorm=np.maximum(np.linalg.norm(tangent,axis=1),1e-8)
        normal=np.stack((-tangent[:,1],tangent[:,0]),axis=1)/tangnorm[:,None]
        white=rng.normal(0,jitter,size=len(pts)).astype('float32')
        smooth=cv2.GaussianBlur(white[None,:],(0,0),2.,borderType=cv2.BORDER_REFLECT_101)[0]
        endpoint_bias=float(rng.normal(0,jitter*1.8))
        u=np.linspace(0,1,len(pts))
        kernel=(u**3 if at_end=='last' else (1-u)**3)
        noisy=pts+normal*(smooth+endpoint_bias*kernel)[:,None]
        return np.clip(noisy,0,191)
    noisy_left=[observed(p,'last') for p in left]
    noisy_right=[observed(p,'first') for p in right]
    return SyntheticFragmentScene(gray,noisy_left,noisy_right,{(0,0),(1,1)},photo.name,scenario,
                                  (t0['frame'],t1['frame']),seed,jitter,
                                  {'injected_visible_strokes':fg,'injected_false_decoy':scratchmask})
