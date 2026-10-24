import cv2
import numpy as np
import torch
import pytest
from bubble_chamber.anza_segmentation_adapter import mask_loss,DatasetNotReadyError,inspect_reviewed_dataset


def test_ignore_unlabeled_pixels_and_backprop():
 logits=torch.tensor([[[[.3,-.4,10.]]]],requires_grad=True)
 labels=torch.tensor([[[[1.,0.,255.]]]])
 value=mask_loss(logits,labels)
 value.backward()
 assert logits.grad[0,0,0,2]==0.0
 assert torch.isfinite(value)


def test_refuse_positive_only_training():
 with pytest.raises(DatasetNotReadyError):
  mask_loss(torch.ones(1,1,3,3),torch.full((1,1,3,3),255.))
 with pytest.raises(DatasetNotReadyError):
  mask_loss(torch.ones(1,1,3,3),torch.ones(1,1,3,3))
 with pytest.raises(ValueError):
  mask_loss(torch.ones(1,1,3,3),torch.ones(1,1,2,3))


def test_not_real_training_set_if_single_film(tmp_path):
 image=np.full((40,40),220,dtype=np.uint8)
 mask=np.full((40,40),255,dtype=np.uint8)
 mask[20:22,20:25]=1
 mask[15,11:19]=0
 ip=tmp_path/'image.png';mp=tmp_path/'mask.png'
 cv2.imwrite(str(ip),image);cv2.imwrite(str(mp),mask)
 entries=[{'film_id':'H50Z_030_01','frame_id':f'frame{i}','image_path':str(ip),'target_path':str(mp)} for i in range(2)]
 with pytest.raises(DatasetNotReadyError,match='three independent films'):
  inspect_reviewed_dataset(entries)
 report=inspect_reviewed_dataset(entries,require_film_holdout=False)
 assert report['class_pixel_counts'][255]>0
 assert report['film_count']==1
 assert report['image_count']==2


def test_wrong_mask_values_refused(tmp_path):
 image=np.full((16,16),255,np.uint8)
 mask=np.full((16,16),17,np.uint8)
 ip=tmp_path/'i.png';mp=tmp_path/'m.png'
 cv2.imwrite(str(ip),image);cv2.imwrite(str(mp),mask)
 with pytest.raises(DatasetNotReadyError,match='Unknown raster'):
  inspect_reviewed_dataset([{'film_id':'1','frame_id':'2','image_path':str(ip),'target_path':str(mp)}])
