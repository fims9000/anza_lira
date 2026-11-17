"""Publication illustration from original CC BY 4.0 Thebe survey, no model prediction.
Run only in GitHub Actions with verified source memmaps and frozen TRAIN normalization.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.ndimage import binary_dilation
from scripts.tver_v33_thebe_train_segmenters import read_memmap, _gaps


def main(root, out):
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    x=read_memmap(root,'val_seismic_float32.npy')
    y=read_memmap(root,'val_fault_bool.npy')
    found=None
    for sid in list(range(40,100)):
        gaps=_gaps(np.asarray(y[sid],bool),sid)
        if gaps:
            found=(sid,gaps[0]); break
    if found is None: raise RuntimeError('No genuine annotated gap in original Thebe VAL1')
    sid,gap=found
    ry0,ry1,rx0,rx1=gap['rectangle']
    center_y=(ry0+ry1)//2;center_x=(rx0+rx1)//2
    width=240
    sy=int(np.clip(center_y-width//2,0,3174-width))
    sx=int(np.clip(center_x-width//2,0,1537-width))
    gray=np.asarray(x[sid,sy:sy+width,sx:sx+width],dtype=float)
    lo,hi=np.percentile(gray,[2,98]);gray=np.clip((gray-lo)/(hi-lo+1e-8),0,1)
    truth=np.asarray(y[sid,sy:sy+width,sx:sx+width],bool)
    missing=truth.copy();missing[ry0-sy:ry1-sy,rx0-sx:rx1-sx]=False
    if not truth.any() or truth.sum()<=missing.sum():
        raise RuntimeError('Invalid true annotations and artificial removed region')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(11.2,4),dpi=200)
    for ax in axes:
        ax.imshow(gray,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
    def annotate(ax, mask):
        # Wide line FOR DISPLAY ONLY: truth comes from raw 1-pixel expert label.
        wide=binary_dilation(mask,iterations=1)
        colored=np.zeros((*wide.shape,4),dtype=float)
        colored[wide]=[0.92,0.2,0.12,0.8]
        ax.imshow(colored,interpolation='nearest')
    annotate(axes[1],truth)
    annotate(axes[2],missing)
    rect=Rectangle((rx0-sx,ry0-sy),rx1-rx0,ry1-ry0,
                   fill=False,linewidth=2,edgecolor='#00b5d8',linestyle='--')
    axes[2].add_patch(rect)
    axes[0].set_title('а) Сейсмический срез',fontweight='normal')
    axes[1].set_title('б) Экспертная разметка',fontweight='normal')
    axes[2].set_title('в) Заданный разрыв',fontweight='normal')
    fig.subplots_adjust(left=.01,right=.99,top=.91,bottom=.02,wspace=.04)
    fig.savefig(out/'thebe_real_gap_example.jpg',dpi=600,format='jpeg',pil_kwargs={'quality':93},facecolor='white')
    plt.close(fig)
    rec={'source':'An et al. 2021 Thebe field real seismic plus expert interpreted mask',
         'license':'CC BY 4.0', 'dataset_doi':'10.7910/DVN/YBYGBK',
         'validation_file':'seisval1.npz/faultval1.npz',
         'source_section_within_VAL1':int(sid),
         'global_rectangle':list(map(int,gap['rectangle'])),
         'crop_origin': [sy,sx], 'crop_size':width,
         'expert_mask_pixels':int(truth.sum()),
         'display_line_dilated_one_pixel':True,
         'real_amplitude_inside_gap_kept_unmodified':True,
         'not_a_model_prediction':True,
         'source_file_test7_never_opened':True}
    (out/'thebe_figure_provenance.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('ORIGINAL_THEBE_FIGURE_READY',json.dumps(rec,ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mmap-root',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();main(a.mmap_root,a.out_dir)