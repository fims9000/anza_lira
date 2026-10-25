"""Verified optional transfer of *unlabeled* reconstruction features to sparse ANZA.

Only image encoders/bottleneck are transferred; no reconstruction head can be
mistaken for a certified particle segmentation head. Expert masks are still
required for fine-tuning and for any downstream evaluation.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import torch

CHECKPOINT_SCHEMA='s20_real_unlabeled_masked_reconstruction_v1'
MODEL_BLOBS={'models/azconv.py':'f4217ada00024927506fa8ebe520aa1f68ae5854',
             'models/segmentation.py':'0b971c117f3d56cc588ebade6991c65faceb8a2a'}
ENCODER_ROOTS=('enc1.','enc2.','enc3.','bottleneck.')


def transfer_unlabeled_encoder(model:torch.nn.Module, checkpoint:Path, *, architecture:str,
                               widths:tuple[int,int,int,int],rules:int):
    checkpoint=Path(checkpoint).resolve()
    if not checkpoint.is_file():raise FileNotFoundError(f'Missing unlabeled checkpoint: {checkpoint}')
    raw=checkpoint.read_bytes()
    # weights_only avoids arbitrary pickle execution for this user-supplied file.
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck.get('schema')!=CHECKPOINT_SCHEMA or ck.get('source_blobs')!=MODEL_BLOBS:
        raise ValueError('Untrusted or wrong-source self-supervised checkpoint')
    if ck.get('architecture')!=architecture or tuple(ck.get('widths',()))!=tuple(widths) or (architecture=='azunet' and ck.get('rules')!=rules):
        raise ValueError('Architecture/width/rules mismatch in unlabeled checkpoint')
    if ck.get('human_verified_track_labels') is not False or ck.get('training_is_unlabeled') is not True:
        raise ValueError('Pretext checkpoint misrepresents human label provenance')
    if ck.get('model_is_not_calibrated_for_real_track_segmentation') is not True:
        raise ValueError('Missing no-automatic-segmentation certificate')
    state=ck['weights'];target=model.state_dict();selected={}
    for name,t in target.items():
        if not name.startswith(ENCODER_ROOTS):continue
        if name not in state or tuple(t.shape)!=tuple(state[name].shape) or t.dtype!=state[name].dtype:
            raise ValueError(f'Mismatched encoder parameter {name}')
        selected[name]=state[name]
    if len(selected)<10:
        raise ValueError('Suspiciously small transfer set')
    # Copy only the approved state subset and preserve a fresh segmentation head.
    model.load_state_dict(selected,strict=False)
    return {'schema':'s20_verified_unlabeled_encoder_transfer_v1',
            'checkpoint_sha256':sha256(raw).hexdigest(),'architecture':architecture,
            'transferred_tensors':len(selected),
            'transferred_encoder_groups':list(ENCODER_ROOTS),
            'reconstruction_head_imported':False,
            'segmentation_metrics_from_pretraining':False,
            'requires_human_sparse_labels':True}
