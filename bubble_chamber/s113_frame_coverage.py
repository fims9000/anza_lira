"""Declare FULL-frame coverage honestly and render ROI spatial map.

Optional libarchive-streamed color BMP thumbnail of an annotated legacy .7z
is for VISUAL ORIENTATION only, never raw photo GT or model performance.
"""
from __future__ import annotations
import ctypes
import json
from pathlib import Path
import cv2
import numpy as np
from .s113_review_output import FRAME_SIZES


def frame_roi_coverage(annotations: list[dict]) -> dict:
    result={}
    for fid,(width,height) in FRAME_SIZES.items():
        rectangles = []
        for a in annotations:
            if a['frame_id']!=fid: continue
            x0,y0,x1,y1=map(int,a['original_roi_source_coordinates'])
            if not(0<=x0<x1<=width and 0<=y0<y1<=height):raise ValueError('Invalid frame ROI bounds')
            rectangles.append((x0,y0,x1,y1))
        area = 0
        # Exact covered union via sweep on the (at most a few dozen) ROI borders.
        xs=sorted({0,width}|{x for x0,y0,x1,y1 in rectangles for x in (x0,x1)})
        for xleft,xright in zip(xs[:-1],xs[1:]):
            spans=sorted((y0,y1) for x0,y0,x1,y1 in rectangles if x0<=xleft and x1>=xright)
            if not spans:continue
            merged=[]
            for a,b in spans:
                if not merged or a>merged[-1][1]:merged.append([a,b])
                else:merged[-1][1]=max(merged[-1][1],b)
            area+=(xright-xleft)*sum(b-a for a,b in merged)
        result[fid]={'frame_size_wh':[width,height],'frame_pixels':width*height,
                     'reviewed_ROIs':len(rectangles),'processed_roi_pixels_union':area,
                     'reviewed_area_fraction':round(area/(width*height),8),
                     'unassessed_area_fraction':round(1-area/(width*height),8),
                     'all_parts_of_original_have_been_recognized':False,
                     'rectangles_xyxy':rectangles}
    return result


def render_coverage(coverage:dict,out:Path):
    panels=[]
    for fid,rep in coverage.items():
        width,height=rep['frame_size_wh'];scale=680/height;W=int(round(width*scale));H=680
        img=np.full((H,W,3),242,np.uint8)
        for x0,y0,x1,y1 in rep['rectangles_xyxy']:
            a=(int(x0*scale),int(y0*scale));b=(int(x1*scale),int(y1*scale))
            cv2.rectangle(img,a,b,(40,155,235),-1)
            cv2.rectangle(img,a,b,(0,75,140),1)
        cv2.putText(img,f'{fid} assessed {rep["reviewed_area_fraction"]*100:.2f}%',(5,30),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA)
        panels.append(img)
    canvas=np.full((720,sum(p.shape[1] for p in panels)+40,3),250,np.uint8)
    cursor=20
    for a in panels:
        canvas[20:20+len(a),cursor:cursor+a.shape[1]]=a
        cursor+=a.shape[1]
    cv2.imwrite(str(out),canvas)
    return canvas


def lowres_annotated_bmp_from_7z(path:Path, stride:int=12)->tuple[np.ndarray,dict]:
    """Read 24-bit BMP inside a .7z without extracting ~700 MiB to disk.

    Thumbnail is still COLOR-PAINTED source. Processing/training on it as clean
    ground truth is not allowed; the 8-bit originals are not silently substituted.
    """
    if stride<2 or stride>64:raise ValueError('downsampling stride must be in [2,64]')
    L=ctypes.CDLL('libarchive.so.13')
    L.archive_read_new.restype=ctypes.c_void_p
    for fn in ('archive_read_support_filter_all','archive_read_support_format_all'):
        getattr(L,fn).argtypes=[ctypes.c_void_p]
    L.archive_read_open_filename.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_size_t]
    L.archive_read_open_filename.restype=ctypes.c_int
    L.archive_read_next_header.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)]
    L.archive_read_next_header.restype=ctypes.c_int
    L.archive_read_data.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
    L.archive_read_data.restype=ctypes.c_ssize_t
    L.archive_read_free.argtypes=[ctypes.c_void_p]
    obj=L.archive_read_new()
    try:
        L.archive_read_support_filter_all(obj);L.archive_read_support_format_all(obj)
        if L.archive_read_open_filename(obj,str(path).encode(),10240)!=0:raise IOError('Open failed')
        h=ctypes.c_void_p()
        if L.archive_read_next_header(obj,ctypes.byref(h))!=0:raise IOError('7z has no first BMP')
        buf=ctypes.create_string_buffer(2**20)
        def get(n):
            size=L.archive_read_data(obj,buf,n)
            if size<=0:raise IOError('Truncated archived BMP')
            return bytes(buf[:size])
        import struct
        head=get(128)
        if head[:2]!=b'BM':raise ValueError('Not BMP')
        off=struct.unpack_from('<I',head,10)[0]
        w,hs=struct.unpack_from('<ii',head,18)
        bpp=struct.unpack_from('<H',head,28)[0]
        compression=struct.unpack_from('<I',head,30)[0]
        if bpp!=24 or compression:raise ValueError('Need uncompressed 24-bit source BMP')
        H=abs(hs); rowbytes=((w*3+3)//4)*4
        thumb=np.empty(((H+stride-1)//stride,(w+stride-1)//stride,3),np.uint8)
        acc=bytearray(head)
        offset=0
        for row in range(H):
            rowend=off+(row+1)*rowbytes
            while offset+len(acc)<rowend:
                acc.extend(get(len(buf)))
            if row%stride==0:
                start=off+row*rowbytes-offset
                samples=np.frombuffer(bytes(acc[start:start+w*3]),dtype=np.uint8,count=w*3).reshape(w,3)[::stride]
                y=(H-1-row)//stride if hs>0 else row//stride
                thumb[y,:len(samples)]=samples
            amount=off+(row+1)*rowbytes-offset
            del acc[:amount];offset+=amount
        return thumb,{'width':w,'height':H,'bpp':bpp,'historical_annotations_overpaint_original':True,'not_clean_model_input':True}
    finally:
        L.archive_read_free(obj)