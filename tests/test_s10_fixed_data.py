"""Fixed-source split and photometric noise across neural initialization seeds."""
import numpy as np
import torch
from dataclasses import replace

from scripts.research.bubble_chamber.s7_original_retrain import (
    RunConfig, effective_data_seed, measured_examples, procedural_examples,
    robust_augment, apply_stress, split_fingerprints,
)


def test_fixed_data_seed_controls_procedural_images():
    a=RunConfig(seed=3,data_seed=19,source='procedural')
    b=replace(a,seed=80)
    assert effective_data_seed(a)==effective_data_seed(b)==19
    x=procedural_examples(3,64,seed=effective_data_seed(a),split='train')
    y=procedural_examples(3,64,seed=effective_data_seed(b),split='train')
    assert all(np.array_equal(u[0],v[0]) and np.array_equal(u[1],v[1]) for u,v in zip(x,y))


def test_fixed_data_seed_freezes_photo_stress_and_augmentation():
    a=RunConfig(seed=7,data_seed=21,robust_train=True,robust_probability=1)
    b=replace(a,seed=200)
    x=torch.full((3,1,32,32),0.65)
    ids=torch.tensor([0,1,2])
    assert torch.equal(apply_stress(x,ids,a),apply_stress(x,ids,b))
    assert torch.equal(robust_augment(x,ids,a,2),robust_augment(x,ids,b,2))
    assert not torch.equal(robust_augment(x,ids,a,2),robust_augment(x,ids,a,3))


def test_fingerprint_catches_pixel_mutation():
    rows={'train':[(np.zeros((4,4),dtype=np.float32),np.ones((4,4),dtype=np.float32),'frame1')]}
    original=split_fingerprints(rows)
    modified={'train':[(np.ones((4,4),dtype=np.float32),np.ones((4,4),dtype=np.float32),'frame1')]}
    assert original['train']['sha256']!=split_fingerprints(modified)['train']['sha256']
