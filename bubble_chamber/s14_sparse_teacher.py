"""Low-label training primitive for original AZUNet/UNet using verified scribbles.

Expert labels are exactly 1=track, 0=certified artifact, 255=UNKNOWN.
Teacher proposals NEVER become expert ground truth and cannot overwrite a human
label. Only training pixels may receive teacher consistency supervision.
Two teacher maps must be spatially aligned to the student input by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SparseTeacherProtocol:
    teacher_positive_min: float = 0.98
    teacher_negative_max: float = 0.02
    max_teacher_disagreement: float = 0.02
    consistency_weight: float = 0.20
    max_positive_weight: float = 20.0
    min_reliable_pseudo_per_class: int = 1

    def check(self):
        if not (0.5 < self.teacher_positive_min <= 1.0 and 0.0 <= self.teacher_negative_max < 0.5 and
                self.teacher_negative_max < self.teacher_positive_min and
                0 <= self.max_teacher_disagreement <= 1.0 and
                0 <= self.consistency_weight <= 0.5 and
                1 <= self.max_positive_weight <= 100 and
                self.min_reliable_pseudo_per_class >= 1):
            raise ValueError('invalid consistency/teacher supervision settings')


def build_teacher_targets(expert_labels:torch.Tensor,
                          teacher_weak:torch.Tensor,teacher_aug_aligned:torch.Tensor,
                          cfg:SparseTeacherProtocol=SparseTeacherProtocol(), *, split:str='train'):
    """Return pseudo labels + mask ONLY on unknown training pixels.

    Confidence does not mean calibrated physical reliability. A conservative
    intersection and BOTH classes are required before pseudo supervision.
    """
    cfg.check()
    if split!='train':raise ValueError('teacher pseudo labels are forbidden on validation/test')
    if (expert_labels.shape != teacher_weak.shape or expert_labels.shape != teacher_aug_aligned.shape
        or expert_labels.ndim!=4 or expert_labels.shape[1]!=1):
        raise ValueError('all inputs must share [N,1,H,W] shape')
    if not bool(torch.all((expert_labels==0)|(expert_labels==1)|(expert_labels==255))):
        raise ValueError('expert labels must be exactly 0, 1, or 255')
    # All targets are probabilities output by two frozen teacher evaluations.
    with torch.no_grad():
        t1=teacher_weak.detach()
        t2=teacher_aug_aligned.detach()
        if not bool(torch.isfinite(t1).all() and torch.isfinite(t2).all()):
            raise ValueError('non-finite teacher output')
        if not bool(((t1>=0)&(t1<=1)&(t2>=0)&(t2<=1)).all()):
            raise ValueError('teacher output must be sigmoid probabilities')
        reliable=(t1-t2).abs()<=cfg.max_teacher_disagreement
        unknown=expert_labels==255
        positive=unknown & reliable & (t1>=cfg.teacher_positive_min) & (t2>=cfg.teacher_positive_min)
        negative=unknown & reliable & (t1<=cfg.teacher_negative_max) & (t2<=cfg.teacher_negative_max)
        counts=(int(positive.sum()),int(negative.sum()))
        # Do not let a teacher invent a training set of only negatives.
        if min(counts)<cfg.min_reliable_pseudo_per_class:
            positive=torch.zeros_like(positive)
            negative=torch.zeros_like(negative)
        selected=positive|negative
        pseudo=torch.full(expert_labels.shape,255,device=expert_labels.device,dtype=torch.uint8)
        pseudo[positive]=1
        pseudo[negative]=0
        return pseudo,selected,{'candidate_positive_count':counts[0],
                                'candidate_negative_count':counts[1],
                                'used_positive_count':int(positive.sum()),
                                'used_negative_count':int(negative.sum()),
                                'unknown_source_pixels':int(unknown.sum()),
                                'expert_pixels_overwritten':0,
                                'pseudo_label_persisted_as_expert':False}


def sparse_teacher_loss(student_logits:torch.Tensor,expert_labels:torch.Tensor,
                        teacher_weak:torch.Tensor, teacher_aug_aligned:torch.Tensor,
                        cfg:SparseTeacherProtocol=SparseTeacherProtocol(), *, split:str='train'):
    """Sparse human BCE + bounded teacher consistency on separate unknown pixels.

    The loss requires both expert classes across the batch. If there are only
    positive clicks, obtain *certified* artifact examples; don't turn the
    unreviewed rest into negative background.
    """
    if student_logits.shape!=expert_labels.shape or student_logits.ndim!=4 or student_logits.shape[1]!=1:
        raise ValueError('student/labels dimension mismatch')
    if not bool(torch.isfinite(student_logits).all()):raise ValueError('non-finite student logits')
    pseudo,mask,stats=build_teacher_targets(expert_labels,teacher_weak,teacher_aug_aligned,cfg,split=split)
    fg=expert_labels==1
    bg=expert_labels==0
    if not bool(fg.any() and bg.any()):
        raise ValueError('both expert-confirmed track and artifact pixels are mandatory')
    known=fg|bg
    pos_weight=(bg.sum().to(dtype=student_logits.dtype)/fg.sum().to(dtype=student_logits.dtype)).clamp(1,cfg.max_positive_weight)
    point_weight=torch.where(fg,pos_weight,torch.ones_like(student_logits))
    human=F.binary_cross_entropy_with_logits(student_logits[known],
            expert_labels[known].to(student_logits.dtype),weight=point_weight[known],reduction='mean')
    consistency=student_logits.sum()*0.0
    if bool(mask.any()) and cfg.consistency_weight:
        pos=mask&(pseudo==1)
        neg=mask&(pseudo==0)
        # Mean of each class, so prolific predicted background cannot dominate.
        pos_loss=F.binary_cross_entropy_with_logits(student_logits[pos],torch.ones_like(student_logits[pos]))
        neg_loss=F.binary_cross_entropy_with_logits(student_logits[neg],torch.zeros_like(student_logits[neg]))
        consistency=(pos_loss+neg_loss)*.5
    total=human+cfg.consistency_weight*consistency
    details={**stats,'human_positive_pixels':int(fg.sum()),'human_negative_pixels':int(bg.sum()),
             'unreviewed_pixels_not_pseudo':int(((expert_labels==255)&~mask).sum()),
             'human_bce':float(human.detach()),'pseudo_bce':float(consistency.detach()),
             'pseudo_weight':cfg.consistency_weight,
             'all_labels_are_simulated_until_real_review':True}
    return total,details
