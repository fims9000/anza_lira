"""Developer-facing interface contract: no source GT, no implicit calibration reuse."""
from __future__ import annotations
from dataclasses import FrozenInstanceError
import numpy as np
import cv2
import pytest
from bubble_chamber.interop_api import (
    INTERFACE_VERSION,ScoreBinding,validate_lines,process_image,assemble_lines)
from bubble_chamber.interop_detectors import (
    HessianRidgeDetector,OpenCVLSDDetector,OpenCVHoughDetector)


def xy(y):return np.column_stack((np.linspace(20.,80.,23),np.full(23,y)))


def scorer(rows):return np.full((len(rows),),.85)


def bind(detector_id):return ScoreBinding('synthetic-heuristic-for-QA',detector_id,scorer,cutoff=.6,
                                           source_cal_id='QA-NOT-H50Z')


def blank():return np.full((192,192),210,np.uint8)


@pytest.mark.parametrize('cls', [HessianRidgeDetector,OpenCVLSDDetector,OpenCVHoughDetector])
def test_built_in_baselines_follow_image_and_line_contract(cls):
    image=blank();cv2.line(image,(22,60),(165,66),15,2);cv2.line(image,(20,120),(170,140),30,2)
    detector=cls(max_lines=12)
    a=process_image(image,detector,bind(detector.detector_id))
    b=process_image(image,detector,bind(detector.detector_id))
    assert a.detector_id==b.detector_id and a.interface_version==INTERFACE_VERSION
    assert a.as_public_dict()==b.as_public_dict()
    assert len(a.lines)<=12 and all(len(line)>=8 for line in a.lines)
    assert all(line.flags.writeable is False for line in a.lines)
    assert not a.physical_auto_joins_authorized
    assert all(e.status=='UNKNOWN' for e in a.joins)
    assert np.array_equal(image,blank()) is False


def test_no_silent_source_scorer_transfer_or_mutation_of_input():
    class DangerousDetector:
        detector_id='other-detector'
        def detect(self,gray):
            gray[:]=0
            return []
    img=blank()
    with pytest.raises(ValueError,match='Detector/score'):
        process_image(img,DangerousDetector(),bind('s72-original'))
    assert np.array_equal(img,blank())
    with pytest.raises(ValueError):process_image(img,DangerousDetector(),bind('other-detector'))
    assert np.array_equal(img,blank())


def test_handles_zero_two_or_many_lines_without_unverified_auto_joins():
    for lines in ([],[xy(70)],[xy(70),xy(100)]):
        out=assemble_lines(lines,bind('local-QA'),detector_id='local-QA')
        assert not out.joins and out.pairs_considered==0
        assert out.as_public_dict()['native_physics_gt_count']==0
    with pytest.raises(ValueError):assemble_lines([xy(50)]*33,bind('local-QA'),detector_id='local-QA')
    with pytest.raises(ValueError):validate_lines([xy(40)[:4]])
    with pytest.raises(ValueError):validate_lines([xy(40)+np.array([192,0])])


def test_probability_and_timeout_invalid_fail_closed():
    lines=[np.column_stack((np.linspace(x,x+23,23),np.full(23,90))) for x in (20,55,90)]
    with pytest.raises(ValueError):assemble_lines(lines,ScoreBinding('bad','det',lambda rows:np.full(len(rows),2.),.7),detector_id='det')
    with pytest.raises(ValueError):assemble_lines(lines,bind('det'),detector_id='det',solver_timeout=float('nan'))
    with pytest.raises(ValueError):ScoreBinding('x','y',scorer,float('nan'))


def test_readonly_public_aggregate_contains_no_line_geometry_or_weights():
    a=assemble_lines([xy(60),xy(95),xy(110)],bind('QA'),detector_id='QA')
    j=a.as_public_dict()
    assert 'lines' not in j and 'joins' not in j and 'points' not in str(j)
    assert j['physical_auto_joins_authorized'] is False


def test_solver_is_explicitly_replaceable_without_changing_line_inputs():
    lines=[np.column_stack((np.linspace(x,x+23,25),np.full(25,90))) for x in (20,55,90)]
    results={}
    for strategy in ('none','node_mutual','port_greedy','global_milp'):
        bound=ScoreBinding('toy-only','det',lambda rows:np.full(len(rows),.95),cutoff=.65,
                           source_cal_id='TOY_ONLY',strategy=strategy)
        results[strategy]=assemble_lines(lines,bound,detector_id='det')
        assert results[strategy].solver_strategy==strategy
        assert results[strategy].as_public_dict()['solver_strategy']==strategy
        assert all(e.status=='UNKNOWN' for e in results[strategy].joins)
    assert not results['none'].joins
    assert all(len(x.lines)==3 for x in results.values())
    with pytest.raises(ValueError):ScoreBinding('x','det',scorer,.6,strategy='make-believe-oracle')


def test_augmenter_needs_new_binding_and_does_not_rewrite_original_curves():
    from bubble_chamber.interop_api import process_with_extras
    class ToyDetector:
        detector_id='toy-raw'
        def detect(self,gray):return [xy(40),xy(90)]
    class ToyAugmenter:
        augmenter_id='toy-extra'
        def propose(self,gray,base):return [*base,xy(145)]
    original=bind('toy-raw');extended=bind('toy-raw+toy-extra')
    img=blank()
    with pytest.raises(ValueError,match='Separate'):
        process_with_extras(img,ToyDetector(),original,ToyAugmenter(),original)
    r=process_with_extras(img,ToyDetector(),original,ToyAugmenter(),extended)
    assert r.extra_accepted==1 and len(r.original.lines)==2 and len(r.extended.lines)==3
    assert r.extended.detector_id=='toy-raw+toy-extra'
    assert not r.physical_auto_joins_authorized
    class BadAugmenter:
        augmenter_id='toy-extra'
        def propose(self,gray,base):return [base[0].copy(),base[1],xy(145)]
    with pytest.raises(ValueError,match='identity'):
        process_with_extras(img,ToyDetector(),original,BadAugmenter(),extended)
