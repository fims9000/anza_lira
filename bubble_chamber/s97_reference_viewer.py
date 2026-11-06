"""S97 private offline manual trace editor over ORIGINAL high-res 8-bit BMP ROIs.

The HTML NEVER overlays machine S86/S92/S96 predictions during annotation.
Full-res source PNG tiles (optionally selected ROI only) are exported read-only.
Human draws points and explicit same-visual-instance IDs and assigns morphology.
No outside-Git original BMP/image pixels, models or labels are uploaded.
"""
from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np

from .bitmap_tiles import GrayBMP8,sha256_file
from .s92_curve_review import _write_private,_json
from .s93_compact_real_overlay import low_memory_gray_preview
from .s97_manual_reference import SCHEMA,KINDS,CERTAINTY

RESULT_STATUS="INDEPENDENT_ORIGINAL_BMP_HUMAN_POSITIVE_REFERENCE_PACK"


def _js(value):
    return json.dumps(value,ensure_ascii=False,separators=(",",":")).replace(
        "<","\\u003c").replace(">","\\u003e").replace("&","\\u0026")


def _validate_rois(rois,h,w):
    if not isinstance(rois,list) or not rois:
        raise ValueError("Provide at least one original-pixel annotation ROI")
    seen=set()
    for rect in rois:
        if (not isinstance(rect,list) or len(rect)!=4
            or any(type(z) is not int for z in rect)
            or not (0<=rect[0]<rect[2]<=w and 0<=rect[1]<rect[3]<=h)):
            raise ValueError("Invalid original pixel ROI bounds")
        if tuple(rect) in seen:
            raise ValueError("Duplicate S97 photo ROI")
        seen.add(tuple(rect))
    return rois


def _review_html(pack,pack_sha):
    tiles=[
        {"id":x["tile_id"],"roi":x["roi_index"],
         "xyxy":x["xyxy_source"],"file":x["source_original_png"]}
        for x in pack["source_image_tiles"]
    ]
    context={
        "schema":pack["schema"],
        "source_bmp_name":pack["source_bmp_name"],
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "image_shape_yx":pack["image_shape_yx"],
        "roi_xyxy_source":pack["roi_xyxy_source"],
        "source_reference_pack_sha256":pack_sha,
        "image_was_unmodified_original":True,
        "machine_proposals_visible_during_annotation":False,
        "full_exhaustive_negative_segmentation":False,
        "data_source":"INDEPENDENT_OPERATOR_ORIGINAL_BMP_ONLY",
        "reference_semantics":"PARTIAL_VISIBLE_2D_STROKES_NO_PARTICLE_GT",
    }
    doc=r"""<!doctype html><html lang="ru"><meta charset="utf-8">
<title>S97: независимая разметка треков на исходном BMP</title>
<style>
body{font:14px system-ui,sans-serif;color:#e8eef5;background:#141922;margin:20px}
h1{font-size:23px}p{max-width:1100px;line-height:1.48;color:#cdd6e2}
main{display:grid;grid-template-columns:minmax(450px,3fr) minmax(320px,1fr);gap:20px}
aside{padding:16px;border:1px solid #48576a;border-radius:6px;min-width:270px}
#viewer{position:relative;display:inline-block;max-width:100%;border:1px solid #6a7686}
#original{max-width:100%;display:block;image-rendering:auto}
#drawing{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;touch-action:none}
input,select,button,textarea{background:#28374c;color:#eef3f8;border:1px solid #74859a;
border-radius:4px;padding:8px;margin:4px;font:14px system-ui}
input[type=text]{width:94%}label{display:block;margin:7px 0}
button{cursor:pointer}button:disabled{opacity:.4}
textarea{width:94%;height:65px}#warning{color:#ffca89}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style>
<h1>S97 — независимый эталон видимых 2D-линий, а не ответ машины</h1>
<p><b>Рисуйте непосредственно на исходном снимке.</b> Выберите участок
изображения; щелчками мыши пройдите по центру линии.
Ставьте точки каждые ~5–20 исходных пикселей, особенно на изгибах и
спиралях (максимальный промежуток между соседними точками 32 px).
Классы: трек, закрученный трек, реперный крест, артефакт, пересечение/пучок,
неясно. Каждой независимой визуальной линии дайте собственный ID;
одинаковый ID можно использовать для продолжения линии на соседней плитке.</p>
<p><b>Важно:</b> это НЕ полная пиксельная маска и не физически подтверждённая
идентификация частиц. Незамеченные и неразмеченные пиксели НЕ являются фоном.
Система оценит только полноту исходного детектора на явно отмеченных
и уверенно видимых экспертных треках. Изображение плитки — точные исходные
пиксели в PNG без масштабирования; увеличение на экране только для просмотра.</p>
<p id="warning">Никаких машинных масок и предсказанных кривых в редакторе нет.</p>
<main><section>
<label>Участок исходного фото <select id="tile"></select></label>
<div id="viewer"><img id="original" alt="Независимый исходный полноразмерный кадр"/>
<svg id="drawing" xmlns="http://www.w3.org/2000/svg"></svg></div>
<div><button id="undo">Удалить последнюю точку</button>
<button id="discard">Очистить незаписанную линию</button></div>
<p id="progress"></p>
</section><aside>
<h3>Новая линия</h3>
<label>ID визуальной линии (одинаковый через плитки)
<input id="instance" type="text" maxlength="96" placeholder="например, T001" /></label>
<label>Морфология <select id="kind"></select></label>
<label>Уверенность <select id="certainty"></select></label>
<label>Комментарий <textarea id="note" maxlength="600"></textarea></label>
<button id="save">Записать линию</button>
<h3>Сохранённые линии на выбранной плитке</h3>
<select id="saved" size="7" style="width:96%"></select>
<button id="remove">Удалить выделенную запись</button>
<hr/>
<label>Имя рецензента
<input id="reviewer" type="text" maxlength="80" placeholder="physicist_A" /></label>
<button id="export">Скачать независимую разметку JSON</button>
<label>Восстановить предыдущую сессию
<input id="import" type="file" accept=".json"/></label>
<p id="status"></p>
</aside></main>
<script>
'use strict';
const META=__META__,TILES=__TILES__,KINDS=__KINDS__,CERTAINTY=__CERTAINTY__;
const allStrokes=new Map(),current=[];
let currentTile=0,serial=1;
const el=id=>document.getElementById(id);
for(const [id,values] of [['kind',KINDS],['certainty',CERTAINTY]]){
 for(const value of values){
  const o=document.createElement('option');o.value=value;
  o.textContent=value;el(id).appendChild(o);
 }
}
for(let i=0;i<TILES.length;i++){
 const t=TILES[i],o=document.createElement('option');
 o.value=String(i);o.textContent=String(i+1)+'/'+TILES.length+' · ROI '+(t.roi+1)
 +' · ('+t.xyxy[0]+','+t.xyxy[1]+')';
 el('tile').appendChild(o);
}
function contains(tile,stroke){
 return stroke.points_xy_source_px.every(([x,y])=>
 x>=tile.xyxy[0]&&x<tile.xyxy[2]&&y>=tile.xyxy[1]&&y<tile.xyxy[3]);
}
function makePolyline(points,color,dashed){
 const svg=document.createElementNS('http://www.w3.org/2000/svg','polyline');
 const t=TILES[currentTile];
 svg.setAttribute('points',points.map(([x,y])=>
   (x-t.xyxy[0])+','+(y-t.xyxy[1])).join(' '));
 svg.setAttribute('fill','none');svg.setAttribute('stroke',color);
 svg.setAttribute('stroke-width','2');
 if(dashed)svg.setAttribute('stroke-dasharray','5 3');
 svg.setAttribute('stroke-linecap','round');svg.setAttribute('stroke-linejoin','round');
 return svg;
}
function updateDrawing(){
 const s=el('drawing'),t=TILES[currentTile];
 const w=t.xyxy[2]-t.xyxy[0],h=t.xyxy[3]-t.xyxy[1];
 s.setAttribute('viewBox','0 0 '+w+' '+h);
 s.replaceChildren();
 for(const row of allStrokes.values()){
  if(!contains(t,row))continue;
  const color=row.kind==='COILING_TRACK_LIKE'?'#67efad':
   row.kind==='TRACK_LIKE'?'#37cff1':row.kind==='ARTIFACT'?'#ff8c8c':
   row.kind==='FIDUCIAL_CROSS'?'#ffc178':'#c9adfb';
  s.appendChild(makePolyline(row.points_xy_source_px,color,false));
 }
 if(current.length) s.appendChild(makePolyline(current,'#fffb55',true));
 const saved=el('saved');saved.replaceChildren();
 for(const row of allStrokes.values()){
  if(!contains(t,row))continue;
  const opt=document.createElement('option');opt.value=row.stroke_id;
  opt.textContent=row.visual_instance_id+' / '+row.kind+' / '+row.stroke_id;
  saved.appendChild(opt);
 }
 el('progress').textContent='Текущих точек: '+current.length+
 '. Сохранено линий: '+allStrokes.size;
}
function render(){
 current.splice(0);
 const t=TILES[currentTile];
 el('original').src=t.file;
 updateDrawing();
}
function setMessage(message){el('status').textContent=message;}
el('tile').onchange=()=>{currentTile=Number(el('tile').value);render();};
el('drawing').addEventListener('click',event=>{
 const bounds=event.currentTarget.getBoundingClientRect();
 const t=TILES[currentTile];
 const w=t.xyxy[2]-t.xyxy[0],h=t.xyxy[3]-t.xyxy[1];
 const x=t.xyxy[0]+Math.min(w-.01,Math.max(0,(event.clientX-bounds.left)/bounds.width*w));
 const y=t.xyxy[1]+Math.min(h-.01,Math.max(0,(event.clientY-bounds.top)/bounds.height*h));
 const pt=[Math.round(x*100)/100,Math.round(y*100)/100];
 if(current.length){
  const q=current[current.length-1],d=Math.hypot(pt[0]-q[0],pt[1]-q[1]);
  if(d>32){setMessage('Точки дальше 32 исходных пикселей; добавьте промежуточные');return;}
  if(d<.4){return;}
 }
 current.push(pt);setMessage('');
 updateDrawing();
});
el('undo').onclick=()=>{current.pop();updateDrawing();};
el('discard').onclick=()=>{current.splice(0);updateDrawing();};
el('save').onclick=()=>{
 const instance=el('instance').value.trim();
 if(!instance||instance.length>96||current.length<2){
  setMessage('Укажите ID линии и не менее двух точек на оригинальном кадре');
  return;
 }
 const t=TILES[currentTile];
 if(!current.every(([x,y])=>x>=t.xyxy[0]&&x<t.xyxy[2]&&y>=t.xyxy[1]&&y<t.xyxy[3])){
  setMessage('Выбранная линия выходит за пределы исходной плитки');return;
 }
 const id='h'+String(serial++).padStart(6,'0');
 allStrokes.set(id,{
  stroke_id:id,visual_instance_id:instance,
  kind:el('kind').value,certainty:el('certainty').value,
  points_xy_source_px:[...current],
  note:el('note').value.slice(0,600)
 });
 current.splice(0);el('note').value='';
 setMessage('Независимая линия записана: '+id);updateDrawing();
};
el('remove').onclick=()=>{
 const id=el('saved').value;
 if(!id)return;
 if(!confirm('Удалить независимую разметку '+id+'?'))return;
 allStrokes.delete(id);updateDrawing();
};
el('export').onclick=()=>{
 const reviewer=el('reviewer').value.trim();
 if(!reviewer||reviewer.length>80){
  setMessage('Для сохранения укажите идентификатор рецензента');return;
 }
 const payload={...META,reviewer_id:reviewer,strokes:[...allStrokes.values()]};
 const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});
 const u=URL.createObjectURL(blob),a=document.createElement('a');
 a.href=u;a.download='S97_independent_human_trace_reference_PRIVATE.json';
 document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(u);
 setMessage('Файл экспортирован. Храните его приватно.');
};
el('import').onchange=async event=>{
 try{
  const input=JSON.parse(await event.target.files[0].text());
  for(const key of ['schema','source_bmp_name','source_bmp_sha256',
   'source_reference_pack_sha256','image_shape_yx','roi_xyxy_source']){
   if(JSON.stringify(input[key])!==JSON.stringify(META[key]))
    throw Error('Фото, ROI или версия исходного пакета отличаются');
  }
  if(input.image_was_unmodified_original!==true
      ||input.machine_proposals_visible_during_annotation!==false
      ||input.full_exhaustive_negative_segmentation!==false
      ||input.data_source!==META.data_source||input.reference_semantics!==META.reference_semantics
      ||!Array.isArray(input.strokes)||input.strokes.length>100000){
    throw Error('Неверная семантика независимой разметки');
  }
  const next=new Map();
  for(const row of input.strokes){
   if(typeof row.stroke_id!=='string'||next.has(row.stroke_id)
      ||!KINDS.includes(row.kind)||!CERTAINTY.includes(row.certainty)
      ||typeof row.visual_instance_id!=='string'
      ||!Array.isArray(row.points_xy_source_px)
      ||row.points_xy_source_px.length<2
      ||typeof row.note!=='string'||row.note.length>600)
     throw Error('Неверная запись кривой или повтор ID');
   if(!TILES.some(t=>contains(t,row)))throw Error('Кривая выходит за выбранную область');
   for(let j=1;j<row.points_xy_source_px.length;j++){
    const a=row.points_xy_source_px[j-1],b=row.points_xy_source_px[j];
    if(!Array.isArray(a)||!Array.isArray(b)||a.length!==2||b.length!==2
       ||a.some(x=>typeof x!=='number'||!Number.isFinite(x))
       ||b.some(x=>typeof x!=='number'||!Number.isFinite(x))
       ||Math.hypot(a[0]-b[0],a[1]-b[1])>32)
     throw Error('Некорректная геометрия или слишком редкие точки');
   }
   next.set(row.stroke_id,row);
  }
  allStrokes.clear();
  for(const [id,row] of next)allStrokes.set(id,row);
  const ids=[...allStrokes.keys()].map(x=>Number(x.replace(/^h/,''))).filter(Number.isFinite);
  serial=Math.max(0,...ids)+1;
  el('reviewer').value=input.reviewer_id||'';
  updateDrawing();setMessage('Независимые исходные линии восстановлены из JSON');
 }catch(error){setMessage('Ошибка импорта: '+String(error.message||error));}
};
if(TILES.length){render();}else{setMessage('Пустая плиточная выборка');}
</script></html>"""
    return (doc.replace("__META__",_js(context))
               .replace("__TILES__",_js(tiles))
               .replace("__KINDS__",_js(KINDS))
               .replace("__CERTAINTY__",_js(CERTAINTY)))


def prepare_independent_source_rois(
    bmp_path:Path,
    out_dir:Path,
    *,
    roi_xyxy_source:list[list[int]]|None=None,
    tile_size:int=512,
    max_tiles:int=256,
    full_photo:bool=False,
    max_preview_side:int=1024,
):
    bmp_path=Path(bmp_path).resolve()
    out_dir=Path(out_dir).resolve()
    repo=Path(__file__).resolve().parents[1]
    if out_dir.exists() or out_dir==repo or repo in out_dir.parents:
        raise ValueError("S97 source-photo reference requires new PRIVATE output outside Git")
    if (type(tile_size) is not int or not 128<=tile_size<=1024 or tile_size%32
        or type(max_tiles) is not int or not 1<=max_tiles<=4096):
        raise ValueError("S97 tile size or maximum tile count invalid")
    bmp=GrayBMP8(bmp_path)
    if roi_xyxy_source is None:
        if not full_photo and bmp.width*bmp.height>1024*1024:
            raise ValueError("Choose explicit --roi x0 y0 x1 y1 for large photo, or --whole-photo")
        roi_xyxy_source=[[0,0,bmp.width,bmp.height]]
    rois=_validate_rois(roi_xyxy_source,bmp.height,bmp.width)
    tiles=[]
    for roi_index,(x0,y0,x1,y1) in enumerate(rois):
        for y in range(y0,y1,tile_size):
            for x in range(x0,x1,tile_size):
                rect=[x,y,min(x+tile_size,x1),min(y+tile_size,y1)]
                tiles.append((roi_index,rect))
                if len(tiles)>max_tiles:
                    raise ValueError("S97 ROI tile count exceeds explicit budget; split into smaller ROIs")
    source_sha=sha256_file(bmp_path)
    out_dir.mkdir(mode=0o700,parents=True)
    out_dir.chmod(0o700)
    (out_dir/"tiles").mkdir(mode=0o700)
    rows=[]
    for idx,(roi_index,(x0,y0,x1,y1)) in enumerate(tiles):
        # EXACT original BMP pixel values, losslessly encoded as PNG.
        original=np.ascontiguousarray(
            bmp._data[y0:y1,x0:x1],dtype=np.uint8
        )
        if original.shape!=(y1-y0,x1-x0):
            raise ValueError("Source BMP tile out of bounds")
        ok,png=cv2.imencode(".png",original,[cv2.IMWRITE_PNG_COMPRESSION,3])
        if not ok:
            raise OSError("Failed to encode lossless-source pixel ROI")
        tid=f"t{idx:05d}"
        relative=f"tiles/{tid}_original.png"
        _write_private(out_dir/relative,png.tobytes())
        rows.append({
            "tile_id":tid,"roi_index":roi_index,
            "xyxy_source":[x0,y0,x1,y1],
            "source_original_png":relative,
            "tile_preview_sha256":sha256_file(out_dir/relative),
            "tile_pixels_lossless_original":True,
            "coordinate_system":"EXACT_ORIGINAL_SOURCE_BMP_PIXELS",
        })
    overview,_=low_memory_gray_preview(bmp,max_side=max_preview_side)
    ok,blob=cv2.imencode(".jpg",overview,[cv2.IMWRITE_JPEG_QUALITY,90])
    if not ok:
        raise RuntimeError("Cannot generate S97 overview")
    _write_private(out_dir/"overview_navigation_only.jpg",blob.tobytes())
    if sha256_file(bmp_path)!=source_sha:
        raise ValueError("Original S97 source BMP changed during tile export")
    pack={
        "schema":SCHEMA,
        "status":RESULT_STATUS,
        "source_bmp_name":bmp_path.name,
        "source_bmp_sha256":source_sha,
        "image_shape_yx":[bmp.height,bmp.width],
        "roi_xyxy_source":rois,
        "source_image_tiles":rows,
        "source_tile_size_px":tile_size,
        "source_photo_coverage_complete":bool(
            len(rois)==1 and rois[0]==[0,0,bmp.width,bmp.height]
        ),
        "tile_pixels_from_unmodified_original_bmp":True,
        "source_zoom_navigational_overview_downsampled":True,
        "human_edit_does_not_show_machine_predictions":True,
        "annotation_is_partial_positive_strokes_only":True,
        "unreviewed_photo_regions_not_negative":True,
        "no_physics_particle_GT":True,
    }
    _write_private(out_dir/"reference_pack.json",_json(pack))
    pack_sha=sha256_file(out_dir/"reference_pack.json")
    _write_private(out_dir/"index.html",_review_html(pack,pack_sha).encode("utf8"))
    return pack
