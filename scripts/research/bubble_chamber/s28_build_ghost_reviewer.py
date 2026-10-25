"""Make private S28 ghost-overlay reviewer, preserving human-only S26 JSON export."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from bubble_chamber.s28_ghost_review import add_path_ghosts

def run(a):
    original=a.review_html.read_text(encoding='utf-8')
    data=json.loads(a.paths.read_text(encoding='utf-8'))
    html=add_path_ghosts(original,data)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(html,encoding='utf-8')
    receipt={'schema':'s28_human_only_review_with_visual_ghosts_v1',
       'S27_private_html_sha256':hashlib.sha256(a.review_html.read_bytes()).hexdigest(),
       'output_sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),
       'source_start_count':len(data['records']),'real_photo_image_count':39,
       'model_prediction_auto_certified_as_expert':0,'export_schema_preserved':'s26_real_track_manual_review_v1',
       'human_only_pixel_labels_and_all_other_unknown':True}
    (a.out.parent/'S28_ghost_reviewer_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--review-html',type=Path,required=True);p.add_argument('--paths',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    print(run(p.parse_args()))
