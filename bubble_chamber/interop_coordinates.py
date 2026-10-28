"""Explicit crop-local <-> original-image coordinates for external 2D viewers.

The graph operates in an EXACT 192x192 grayscale crop. Its (x,y) points
must never silently be interpreted as source-tile coordinates, physical
camera coordinates, or 3D particle positions.

The functions below are geometry transforms only, not evidence of true joins.
Any serialization containing source-line coordinates is PRIVATE by design.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .interop_api import IMAGE_SHAPE,ResearchResult,validate_lines


@dataclass(frozen=True)
class CropTransform:
    """Unscaled 192x192 grayscale ROI specified explicitly in source pixels."""
    x0: int
    y0: int
    full_width: int
    full_height: int
    crop_width: int=IMAGE_SHAPE[1]
    crop_height: int=IMAGE_SHAPE[0]

    def __post_init__(self):
        values=(self.x0,self.y0,self.full_width,self.full_height,self.crop_width,self.crop_height)
        if any(type(v) is not int for v in values):
            raise ValueError('Crop offsets and pixel dimensions must be integers')
        if (self.crop_width,self.crop_height)!=(IMAGE_SHAPE[1],IMAGE_SHAPE[0]):
            raise ValueError('This source interface accepts only unresized 192x192 crops')
        if self.x0<0 or self.y0<0 or self.full_width<IMAGE_SHAPE[1] or self.full_height<IMAGE_SHAPE[0]:
            raise ValueError('Crop position/frame dimensions outside source image')
        if self.x0+self.crop_width>self.full_width or self.y0+self.crop_height>self.full_height:
            raise ValueError('Crop rectangle exceeds source image')

    def extract(self,gray:np.ndarray)->np.ndarray:
        if not isinstance(gray,np.ndarray) or gray.dtype!=np.uint8 or gray.shape!=(self.full_height,self.full_width):
            raise ValueError('Source photo does not match declared uint8 grayscale geometry')
        crop=gray[self.y0:self.y0+self.crop_height,self.x0:self.x0+self.crop_width].copy()
        crop.setflags(write=False)
        return crop

    def to_source(self,lines)->tuple[np.ndarray,...]:
        """Make immutable coordinate copies; source offset is (x,y), not (row,col)."""
        checked=validate_lines(tuple(lines))
        offset=np.array([self.x0,self.y0],dtype=float)
        translated=[]
        for p in checked:
            q=p+offset
            q.setflags(write=False)
            translated.append(q)
        return tuple(translated)

    def to_crop(self,source_lines)->tuple[np.ndarray,...]:
        """Reject source curves outside this ROI; never silently clip lines."""
        src=tuple(source_lines)
        offset=np.array([self.x0,self.y0],dtype=float)
        checked=[]
        for p in src:
            if not isinstance(p,np.ndarray) or p.ndim!=2 or p.shape[1]!=2 or len(p)<8:
                raise ValueError('Expected source-pixel polyline array Nx2')
            q=np.asarray(p,dtype=float)-offset
            checked.append(q)
        return validate_lines(tuple(checked))


def private_review_payload(result:ResearchResult,crop:CropTransform,*,source_image_id:str)->dict:
    """PRIVATE ONLY: pixel coordinates for the caller's review UI, NOT Git.

    `result.as_public_dict()` remains aggregate-only and is the only safe
    default for publication. This opt-in method exports source pixel points.
    """
    if not isinstance(result,ResearchResult) or not isinstance(crop,CropTransform):
        raise ValueError('Explicit result and crop transform needed for private viewer')
    if not isinstance(source_image_id,str) or not source_image_id or len(source_image_id)>256:
        raise ValueError('Need a caller-local private source image identifier')
    src=crop.to_source(result.lines)
    return {
        'schema':'ANZA-LIRA-PRIVATE-REVIEW-ONLY-V1',
        'publication_safe':False,
        'source_image_id_PRIVATE':source_image_id,
        'coordinate_system':'original_source_image_pixels_xy',
        'crop':{'x0':crop.x0,'y0':crop.y0,'width':crop.crop_width,'height':crop.crop_height},
        'original_frame_dimensions_wh':[crop.full_width,crop.full_height],
        'lines_PRIVATE': [x.tolist() for x in src],
        'joins_UNKNOWN':[{'a':e.source_line,'b':e.target_line,'source_end':e.source_end,
                          'target_end':e.target_end,'score_UNVERIFIED':e.score,'status':'UNKNOWN'}
                         for e in result.joins],
        'physical_auto_joins_authorized':False,
        'native_physics_gt_count':0,
    }
