"""Losslessly preserve authentic source PNG; compact only model-overlay visual previews."""
import argparse,base64,json,re
from pathlib import Path
import cv2,numpy as np

def compact(path):
    s=path.read_text()
    m=re.search(r'(<script type="application/json" id="payload">)(.*?)(</script>)',s,re.S)
    if not m:raise ValueError('review payload missing')
    pack=json.loads(m.group(2))
    count=0
    for p in pack['photos']:
        for name,data in p['image_modes'].items():
            if name=='original':continue
            raw=base64.b64decode(data.split(',',1)[1]);im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            assert im.shape==(512,512,3)
            ok,enc=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,88]);assert ok
            p['image_modes'][name]='data:image/jpeg;base64,'+base64.b64encode(enc).decode('ascii')
            count+=1
    payload=json.dumps(pack,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    path.write_text(s[:m.start(2)]+payload+s[m.end(2):])
    print('converted overlays',count,'bytes',path.stat().st_size)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--html',type=Path,required=True)
 args=p.parse_args();compact(args.html)
