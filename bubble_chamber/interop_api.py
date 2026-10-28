"""ANZA-LIRA 2-D research integration boundary, stable API v1.

Allows other detectors to propose 2-D polylines while reusing the S59/S56
geometry graph. The source-specific S56 learned scorer must be rebound to
*each detector*; we NEVER silently reuse ANZA/S72-specific CAL parameters.

The result is an advisory, reversible forest of proposed joins, NOT physics GT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from .s56_global_path_assembly import pair_rows, scored_edges, select_global_forest, select_port_greedy, select_node_mutual
from .s59_candidate_retrieval import filter_rows

IMAGE_SHAPE = (192, 192)
INTERFACE_VERSION = 'ANZA-LIRA-2D-INTEROP-V1'
MAX_LINES = 32
MIN_POINTS = 8
STRATEGIES = ('global_milp','port_greedy','node_mutual','none')


class Detector(Protocol):
    detector_id: str
    def detect(self, gray: np.ndarray) -> Sequence[np.ndarray]: ...


@dataclass(frozen=True)
class ScoreBinding:
    """Developer-provided pair-probability scorer + detector-specific policy.

    A scorer from S72/S85 must not be passed with a different detector ID. A
    newly trained scorer must have its own id and source-CAL cutoff. Hashes
    and software versions should be recorded by the embedding application.
    """
    scorer_id: str
    trained_for_detector: str
    score_pairs: Callable[[list[dict]], np.ndarray]
    cutoff: float
    margin: float = .04
    source_cal_id: str = ''
    strategy: str = 'global_milp'

    def __post_init__(self):
        if not self.scorer_id or not self.trained_for_detector or not callable(self.score_pairs):
            raise ValueError('Nonempty scorer and detector binding are mandatory')
        if not np.isfinite(self.cutoff) or not 0 <= self.cutoff <= 1:
            raise ValueError('Detector-specific source-CAL cutoff must be finite in [0,1]')
        if not np.isfinite(self.margin) or self.margin < 0:
            raise ValueError('Invalid nonnegative ambiguity margin')
        if self.strategy not in STRATEGIES:
            raise ValueError('Unknown solver strategy; choose one of the four explicit methods')


@dataclass(frozen=True)
class ProposedJoin:
    source_line: int
    target_line: int
    source_end: int
    target_end: int
    score: float
    status: str = 'UNKNOWN'


@dataclass(frozen=True)
class ResearchResult:
    interface_version: str
    detector_id: str
    scorer_id: str
    source_cal_id: str
    lines: tuple[np.ndarray, ...]
    joins: tuple[ProposedJoin, ...]
    pairs_considered: int
    solver_success: bool
    ambiguity_margin: float
    solver_strategy: str
    native_physics_gt_count: int = 0
    physical_auto_joins_authorized: bool = False

    def as_public_dict(self):
        """No pixels, source coordinates or private model coefficient export."""
        return {
            'interface_version': self.interface_version,
            'detector_id': self.detector_id,
            'scorer_id': self.scorer_id,
            'source_cal_id': self.source_cal_id,
            'detected_lines': len(self.lines),
            'pairs_considered': self.pairs_considered,
            'advisory_joins_UNKNOWN': len(self.joins),
            'solver_success': self.solver_success,
            'ambiguity_margin': self.ambiguity_margin,
            'solver_strategy': self.solver_strategy,
            'native_physics_gt_count': 0,
            'physical_auto_joins_authorized': False,
        }


def validate_pixels(gray):
    if not isinstance(gray, np.ndarray) or gray.shape != IMAGE_SHAPE or gray.dtype != np.uint8:
        raise ValueError('Expected exactly one uint8 192x192 grayscale tile')
    return gray


def validate_lines(raw: Sequence[np.ndarray], *, max_lines: int = MAX_LINES) -> tuple[np.ndarray,...]:
    if isinstance(raw, np.ndarray) or isinstance(raw, (str,bytes)):
        raise ValueError('Detector output must be a sequence of Nx2 line arrays')
    if not isinstance(max_lines,int) or not 1 <= max_lines <= MAX_LINES:
        raise ValueError('Candidate budget must be an integer in 1..32')
    if not isinstance(raw,(tuple,list)) or len(raw)>max_lines:
        raise ValueError('Detector exceeded finite candidate budget (fail closed)')
    result=[]
    for line in raw:
        if not isinstance(line,np.ndarray) or line.ndim!=2 or line.shape[1]!=2 or len(line)<MIN_POINTS:
            raise ValueError('Each line must be an Nx2 ndarray with at least 8 points')
        p=np.asarray(line,dtype=np.float64)
        if not np.isfinite(p).all() or np.any(p<0) or np.any(p>191):
            raise ValueError('Non-finite or out-of-tile (x,y) coordinates')
        if np.linalg.norm(p[-1]-p[0])<4 or np.linalg.norm(np.diff(p,axis=0),axis=1).sum()<6:
            raise ValueError('Degenerate/open contour is not a candidate line')
        # Copy only after validation; do not mutate input detector objects.
        q=p.copy();q.setflags(write=False);result.append(q)
    return tuple(result)


def assemble_lines(lines: Sequence[np.ndarray], binding: ScoreBinding, *, detector_id: str,
                   solver_timeout: float = 3.) -> ResearchResult:
    """Assemble detector output through exactly the original strict S59 + S56.

    No correspondence/ground truth is accepted; only a developer-bound scorer
    sees the geometric pair feature rows. On solver failure, return no joins.
    """
    if not detector_id or detector_id!=binding.trained_for_detector:
        raise ValueError('Detector/score calibration mismatch; retrain and recalibrate first')
    if not np.isfinite(solver_timeout) or solver_timeout <= 0 or solver_timeout > 30:
        raise ValueError('Finite positive solver budget required (<=30 seconds)')
    pieces=validate_lines(list(lines))
    edges=[];pairs=[];solver_ok=True
    if len(pieces)>=3:
        pairs=filter_rows(pair_rows(list(pieces)),'strict_s39')
        if pairs and binding.strategy != 'none':
            scores=np.asarray(binding.score_pairs(pairs),dtype=float)
            if scores.shape!=(len(pairs),) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
                raise ValueError('Scorer contract: one finite [0,1] probability per pair')
            scored=scored_edges(pairs,scores)
            if binding.strategy=='global_milp':
                edges,qa=select_global_forest(scored,threshold=binding.cutoff,
                                              margin=binding.margin,timeout=float(solver_timeout))
                solver_ok=bool(qa['solver_success'])
                if not solver_ok:edges=[]
            elif binding.strategy=='port_greedy':
                edges=select_port_greedy(scored,threshold=binding.cutoff,margin=binding.margin)
            elif binding.strategy=='node_mutual':
                edges=select_node_mutual(scored,threshold=binding.cutoff,margin=binding.margin)
    joins=tuple(ProposedJoin(e.i,e.j,e.end_i,e.end_j,float(e.score)) for e in edges)
    return ResearchResult(INTERFACE_VERSION,detector_id,binding.scorer_id,binding.source_cal_id,
                          pieces,joins,len(pairs),solver_ok,binding.margin,binding.strategy)


def process_image(gray:np.ndarray, detector:Detector, binding:ScoreBinding, *,
                  solver_timeout:float=3.) -> ResearchResult:
    validate_pixels(gray)
    detector_id=getattr(detector,'detector_id',None)
    if not isinstance(detector_id,str) or not detector_id:
        raise ValueError('Detector needs a stable, nonempty detector_id')
    if detector_id!=binding.trained_for_detector:
        # Never even call a detector if its source-trained scorer is mismatched.
        raise ValueError('Detector/score binding differs; no silent transfer of S72 policy')
    # Readonly image copy guards against accidental detector mutations.
    safe_image=gray.copy();safe_image.setflags(write=False)
    lines=detector.detect(safe_image)
    return assemble_lines(lines,binding,detector_id=detector_id,solver_timeout=solver_timeout)


@dataclass(frozen=True)
class CandidateComparison:
    """A/B detector-only vs optional extra lines. BOTH are advisory."""
    original: ResearchResult
    extended: ResearchResult
    augmenter_id: str
    extra_accepted: int
    native_physics_gt_count: int = 0
    physical_auto_joins_authorized: bool = False


def process_with_extras(gray: np.ndarray, detector: Detector, original_binding: ScoreBinding,
                        augmenter, extended_binding: ScoreBinding, *,
                        solver_timeout: float = 3.) -> CandidateComparison:
    """Research-only S80/other enhancement without silent calibration transfer.

    `extended_binding` must be learned/calibrated for the *composite* detector
    id (`base_id+augmenter_id`), NOT copied from the original detector score.
    No learned S85 threshold or physical GT is implied by this interface.
    """
    validate_pixels(gray)
    base_id=getattr(detector,'detector_id',None)
    augmenter_id=getattr(augmenter,'augmenter_id',None)
    if not isinstance(base_id,str) or not base_id or not isinstance(augmenter_id,str) or not augmenter_id:
        raise ValueError('External detector and augmenter require stable IDs')
    if original_binding.trained_for_detector!=base_id or \
       extended_binding.trained_for_detector!=base_id+'+'+augmenter_id:
        raise ValueError('Separate detector-only and detector+augmented TRAIN/CAL bindings required')
    safe=gray.copy();safe.setflags(write=False)
    original=validate_lines(list(detector.detect(safe)))
    base=assemble_lines(original,original_binding,detector_id=base_id,solver_timeout=solver_timeout)
    proposed=augmenter.propose(safe,original)
    if not isinstance(proposed,(tuple,list)) or len(proposed)<len(original):
        raise ValueError('Augmenter must preserve all originals before its optional candidates')
    if any(x is not y for x,y in zip(proposed,original)):
        raise ValueError('Augmenter changed original line object identity/order')
    enhanced=assemble_lines(proposed,extended_binding,detector_id=base_id+'+'+augmenter_id,
                            solver_timeout=solver_timeout)
    return CandidateComparison(base,enhanced,augmenter_id,len(enhanced.lines)-len(base.lines))
