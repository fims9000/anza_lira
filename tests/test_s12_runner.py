from scripts.research.bubble_chamber import s7_original_retrain as s7
from scripts.research.bubble_chamber.s12_train_orientation import run,ARMS


def test_s12_native_models_and_original_kernel_unmodified(tmp_path):
    cfg=s7.RunConfig(source='procedural',seed=22,data_seed=33,size=32,
         epochs=1,batch_size=2,train_limit=6,val_limit=5,test_limit=6,
         widths=(4,4,4,4),num_rules=2,cpu_threads=2,az_residual=True,robust_train=True)
    report=run(cfg,None,tmp_path,arms=('unet_base','az_base','az_direct05'))
    assert report['source_git_blobs']==s7.SOURCE_BLOBS
    assert report['real_photo_dice'] is None
    assert set(report['results'])=={'unet_base','az_base','az_direct05'}
    assert (tmp_path/'az_direct05'/'best_final.pt').exists()
    assert report['comparisons']['clean_az_direct05_minus_az_base']['frames']==6
