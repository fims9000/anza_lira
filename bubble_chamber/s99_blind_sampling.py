"""S99 spatially balanced, detector-independent original-BMP annotation sampling.

A reproducible precommitted choice of 2D regions using ONLY source dimensions,
source SHA and an explicitly stored seed. No S86/S92/S95 predictions, label
locations, pixel strengths, or prior model error scores are consulted. It
exports native-resolution *lossless PNG* via the independently blinded S97
human annotation viewer, not machine predictions. Source H50Z BMP stays local.

Spatially balanced random cell sampling is a design for *reviewable coverage*.
Expert partial positive-stroke annotations are NOT an exhaustive label of
every track/background pixel in a sampled tile. A population-level precision,
recall, accuracy or statistical CI is not inferred from this selection.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import math

from .bitmap_tiles import GrayBMP8,sha256_file
from .s92_curve_review import _write_private,_json
from .s97_reference_viewer import prepare_independent_source_rois

SCHEMA="S99_SOURCE_ONLY_SPATIALLY_BALANCED_BLIND_SAMPLE_V1"
STAGE="PRIVATE_BLIND_ORIGINAL_PHOTO_HUMAN_REVIEW_NOT_MACHINE_SCORES"


def select_blind_original_source_rois(
    *,height:int,width:int,source_sha256:str,seed:int=20261106,
    tile_size:int=512,roi_count:int=18,strata_rows:int=3,
    strata_cols:int=3
):
    """Reproducibly choose image-grid cells from ALL source photo regions.

    Avoid picking the most flattering/difficult S86 output after inspection.
    If at least one tile per stratum is requested, each spatial stratum gets
    one before any stratum receives a second. Seed and BMP SHA are recorded;
    fixing a seed AFTER looking at model outputs is still cherry-picking.
    """
    if not all(type(x) is int for x in (
        height,width,seed,tile_size,roi_count,strata_rows,strata_cols
    )):
        raise ValueError("S99 geometry, seed and ROI count must be integers")
    if (height<1 or width<1 or seed<0 or seed>2**63-1
        or not 128<=tile_size<=1024 or tile_size%32
        or not 1<=roi_count<=512
        or not 1<=strata_rows<=8 or not 1<=strata_cols<=8):
        raise ValueError("S99 invalid ROI sampling dimensions or budget")
    if not isinstance(source_sha256,str) or len(source_sha256)!=64:
        raise ValueError("S99 exact source BMP SHA256 required")
    try:
        int(source_sha256,16)
    except ValueError as e:
        raise ValueError("S99 source SHA must be hexadecimal") from e
    cols=math.ceil(width/tile_size)
    rows=math.ceil(height/tile_size)
    if roi_count>cols*rows:
        raise ValueError("Requested more independent source cells than exist")
    ncols=min(strata_cols,cols)
    nrows=min(strata_rows,rows)
    buckets=defaultdict(list)
    for y in range(rows):
        for x in range(cols):
            group=(min(nrows-1,y*nrows//rows),
                   min(ncols-1,x*ncols//cols))
            x0=x*tile_size
            y0=y*tile_size
            box=[x0,y0,min(x0+tile_size,width),min(y0+tile_size,height)]
            # This pseudorandom *cell ordering* sees NO actual image pixels
            # except the fixed full-BMP SHA-256 salt.
            entropy=f"{SCHEMA}|{seed}|{source_sha256}|{x}|{y}"
            priority=sha256(entropy.encode("ascii")).hexdigest()
            buckets[group].append((priority,box,x,y))
    for key in buckets:
        buckets[key].sort(key=lambda z:z[0])
    if roi_count<len(buckets):
        # We do not hide an unreviewed spatial stratum while advertising
        # spatially balanced sampling.
        raise ValueError("S99 ROI budget must cover every nonempty spatial stratum")
    keys=sorted(buckets,key=lambda group:
        sha256(f"{seed}|{source_sha256}|stratum|{group}".encode()).hexdigest()
    )
    out=[]
    idx=0
    while len(out)<roi_count:
        key=keys[idx%len(keys)]
        if buckets[key]:
            _,rectangle,x,y=buckets[key].pop(0)
            out.append({
                "roi_index":len(out),
                "xyxy_source":rectangle,
                "source_grid_cell_xy":[x,y],
                "spatial_stratum_yx":list(key),
                "image_pixels_used_to_select_ROI":False,
                "model_scores_used_to_select_ROI":False,
            })
        idx+=1
        if idx>roi_count*(len(keys)+1)+len(keys)*max(rows,cols):
            raise RuntimeError("S99 spatial ROI sampler did not terminate")
    if len({tuple(x["xyxy_source"]) for x in out})!=len(out):
        raise AssertionError("S99 sample erroneously selected an ROI twice")
    coverage={}
    for row in out:
        key=f"{row['spatial_stratum_yx'][0]},{row['spatial_stratum_yx'][1]}"
        coverage[key]=coverage.get(key,0)+1
    return {
        "schema":SCHEMA,
        "status":"SOURCE_ONLY_BLIND_PRESELECTED_SAMPLE_NO_MODEL_ACCESS",
        "selection_algorithm":"BALANCED_FULL_SOURCE_GRID_HASH_PRIORITY_V1",
        "source_bmp_sha256":source_sha256,
        "image_shape_yx":[height,width],
        "sampling_seed_predeclared":seed,
        "source_full_grid_shape_yx":[rows,cols],
        "tile_size_px":tile_size,
        "source_grid_count":rows*cols,
        "strata_grid_yx":[nrows,ncols],
        "sampled_source_roi_count":len(out),
        "sampled_cell_selection":out,
        "selected_spatial_stratum_counts":coverage,
        "full_photograph_was_analyzed_for_model_errors":False,
        "no_exhaustive_negative_annotations_from_unsampled_cells":True,
        "edge_cells_may_have_smaller_area":True,
        "not_uniform_random_source_pixel_sampling":True,
        "post_selection_tuning_of_seed_would_invalidate_blind_design":True,
    }


def prepare_blind_s99_review(
    bmp:Path,
    private_out_dir:Path,
    *,
    seed:int=20261106,tile_size:int=512,roi_count:int=18,
    strata_rows:int=3,strata_cols:int=3
):
    bmp=Path(bmp).resolve()
    private_out_dir=Path(private_out_dir).resolve()
    if not bmp.is_file():
        raise FileNotFoundError("S99 original BMP missing")
    original=GrayBMP8(bmp)
    sha=sha256_file(bmp)
    selection=select_blind_original_source_rois(
        height=original.height,width=original.width,
        source_sha256=sha,seed=seed,tile_size=tile_size,
        roi_count=roi_count,strata_rows=strata_rows,
        strata_cols=strata_cols
    )
    rois=[x["xyxy_source"] for x in selection["sampled_cell_selection"]]
    pack=prepare_independent_source_rois(
        bmp,private_out_dir,roi_xyxy_source=rois,tile_size=tile_size,
        max_tiles=roi_count
    )
    if (pack["source_bmp_sha256"]!=sha
        or pack["roi_xyxy_source"]!=rois
        or len(pack["source_image_tiles"])!=roi_count):
        raise RuntimeError("S99 independent original-photo tile export lost sample")
    selection.update({
        "source_bmp_name":bmp.name,
        "source_reference_pack_sha256":sha256_file(
            private_out_dir/"reference_pack.json"
        ),
        "reference_editor":"index.html",
        "image_tiles_original_unmodified_lossless":True,
        "human_editor_shows_machine_prediction":False,
        "independent_physics_particle_ground_truth":False,
        "reference_not_yet_annotated":True,
    })
    _write_private(private_out_dir/"s99_blind_selection.json",_json(selection))
    return selection
