#!/usr/bin/env python3
"""Create a self-contained *private* folder/ZIP for S31/S35 human review.

Unlike the historical 38 MB base64-embedded HTML, original photo PNGs are
stored once and referenced by relative filenames. The bundle has real source
pixels/coordinates: NEVER commit it into a public repository. This generator
itself uses no private input in the source and is safe for code review.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path
import html,json,zipfile
import cv2
import numpy as np

HTML='''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S37 | Private bubble chamber fragment review</title>
<style>
:root{font-family:system-ui,sans-serif;color-scheme:dark;}
*{box-sizing:border-box}body{margin:0;background:#10161f;color:#e9eef5}header{padding:16px 20px;background:#182332;border-bottom:1px solid #35445a}
h1{font-size:20px;margin:0 0 5px}small{color:#adbacb}main{display:grid;grid-template-columns:minmax(420px,600px) minmax(300px,1fr);gap:18px;padding:18px;max-width:1300px;margin:auto}
@media(max-width:850px){main{grid-template-columns:1fr}}.viewer{position:relative;width:100%;max-width:600px;aspect-ratio:1;background:#293241;overflow:hidden}
.viewer img,.viewer svg{width:100%;height:100%;position:absolute;inset:0}.viewer img{object-fit:contain;image-rendering:auto}.viewer svg{pointer-events:none}
.panel{background:#1b2735;border:1px solid #34495e;border-radius:8px;padding:17px;margin-bottom:12px}label{display:block;margin-bottom:6px;color:#bed1ec}
button{background:#30445d;color:#fff;border:1px solid #69809a;border-radius:5px;padding:9px 12px;margin:3px;cursor:pointer}
button:hover{background:#435d7d}.accept{background:#1d6044}.reject{background:#763739}.neutral{background:#665b30}
.muted{color:#b0c0cf;font-size:13px}strong{color:#f5fbff}.flex{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
select{background:#101924;color:#fff;border:1px solid #51647a;border-radius:4px;padding:8px;max-width:100%}
pre{font-size:12px;white-space:pre-wrap;word-break:break-word;max-height:150px;overflow-y:auto;color:#d0d7e1}
.notice{padding:8px 12px;background:#382e24;border:1px solid #805b35;border-radius:6px;margin:8px 0}
</style></head><body>
<header><h1>ANZA / Graph-LIRA | PRIVATE expert-correction workspace</h1>
<small>Authentic source images; 133 unverified machine lines and 28 unverified pairs. No model score is a probability of physical identity.</small>
<div class="notice">All decisions are LOCAL PROVISIONAL ONLY. Nothing is a certified physical link or a momentum / charge measurement. Export JSON before closing.</div></header>
<main><section><div class="viewer"><img id="photo" alt="Private source crop"><svg id="overlay" viewBox="0 0 512 512"></svg></div>
<div class="panel"><strong id="cropTitle"></strong><p class="muted" id="imgStatus"></p><div class="flex"><label for="showAll"><input id="showAll" type="checkbox" checked> Draw other draft lines in grey</label></div></div></section>
<section><div class="panel"><strong>Relation candidates</strong><p id="progress" class="muted"></p>
<div class="flex"><button id="prev">Previous</button><button id="next">Next unreviewed</button><button id="step">Next in order</button></div>
<p id="pairTitle"></p><p id="pairDetails" class="muted"></p><pre id="evidence"></pre>
<div class="flex"><button class="accept" id="same">1 - Same fragment track?</button><button class="reject" id="diff">2 - Different tracks?</button><button class="neutral" id="unsure">3 - Unsure</button><button id="clearPair">Clear pair</button></div>
<p id="pairStatus" class="muted"></p></div>
<div class="panel"><strong>Individual line proposal</strong><p class="muted">Every line is machine-generated. Accept/reject here means a provisional local annotation, never physical certification.</p>
<label for="draftSelect">Proposal on this crop</label><select id="draftSelect"></select>
<p id="draftInfo" class="muted"></p><div class="flex"><button class="accept" id="acceptLine">Accept</button><button class="reject" id="rejectLine">Reject</button><button class="neutral" id="unsureLine">Unsure</button><button id="clearLine">Clear</button></div><p id="lineStatus" class="muted"></p></div>
<div class="panel"><strong>Export and import review</strong><div class="flex"><button id="export">Export S31-compatible corrections JSON</button><label for="importFile"><input id="importFile" type="file" accept=".json,application/json"></label></div>
<p class="muted">Source SHA-256 and complete crop identities are checked on import. Edits of centerline coordinates remain available in the original S35 full reviewer.</p>
<p id="message" class="muted"></p></div></section></main>
<script type="application/json" id="privateData">__PRIVATE_JSON__</script>
<script>
'use strict';
const B=JSON.parse(document.getElementById('privateData').textContent), P=B.photos, R=B.relations;
const $=id=>document.getElementById(id);
const DEC={same:'LOCAL_SAME_PROVISIONAL',diff:'LOCAL_DIFFERENT_PROVISIONAL',unsure:'LOCAL_UNSURE'};
const DL={accept:'USER_ACCEPTED_PROVISIONAL',reject:'USER_REJECTED_PROVISIONAL',unsure:'USER_UNCERTAIN'};
let idx=0,decisions={},relation_decisions={},lineId=null;
const crops=new Map(P.map(p=>[p.crop_id,p]));const drafts=new Map(P.flatMap(p=>p.drafts.map(d=>[d.draft_id,d])));
const relations=new Set(R.map(r=>r.relation_id));
function text(id,value){$(id).textContent=value===null?'':String(value)}
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function draw(){
 const r=R[idx],p=crops.get(r.crop_id);if(!p)throw Error('Unknown crop id');
 $('photo').src='crops/'+encodeURIComponent(r.crop_id)+'.png';
 text('cropTitle',r.crop_id+' | H50Z '+r.photo_id);
 text('imgStatus',p.drafts.length+' proposals on source crop; original image SHA-256='+p.source_original_png_sha256_from_S29.slice(0,12)+'...');
 const svg=$('overlay');while(svg.firstChild)svg.removeChild(svg.firstChild);
 for(const d of p.drafts){
  const selected=d.draft_id===r.candidate_a||d.draft_id===r.candidate_b||d.draft_id===lineId;
  if(!selected && !$('showAll').checked)continue;
  if(!Array.isArray(d.points_xy)||d.points_xy.length<2)continue;
  const col=d.draft_id===r.candidate_a?'#40ff9e':d.draft_id===r.candidate_b?'#ff9d4d':d.draft_id===lineId?'#f0ff00':'#c4cfdb';
  const poly=document.createElementNS('http://www.w3.org/2000/svg','polyline');
  poly.setAttribute('points',d.points_xy.map(v=>v[0]+','+v[1]).join(' '));
  poly.setAttribute('fill','none');poly.setAttribute('stroke',col);
  poly.setAttribute('stroke-width',selected?'3':'1');poly.setAttribute('stroke-opacity',selected?'1':'.28');
  svg.append(poly);
 }
}
function current(){return R[idx]}
function refreshLine(){
 const r=current(),p=crops.get(r.crop_id),select=$('draftSelect'),last=lineId;
 select.replaceChildren();for(const d of p.drafts){const o=document.createElement('option');o.value=d.draft_id;o.textContent=d.source_method+' | '+d.draft_id.slice(-8);select.add(o)}
 lineId=p.drafts.some(d=>d.draft_id===last)?last:p.drafts.some(d=>d.draft_id===r.candidate_a)?r.candidate_a:p.drafts[0]?.draft_id||null;
 if(lineId)select.value=lineId;
 const d=drafts.get(lineId),saved=decisions[lineId];
 text('draftInfo',d?('method='+d.source_method+', points='+d.points_xy.length+', pseudo-score='+String(d.model_score_UNCALIBRATED)):'No proposals');
 text('lineStatus',saved?.status||'UNREVIEWED');
}
function refresh(){
 const r=current();if(!r)return;
 text('pairTitle',(idx+1)+' / '+R.length+' | '+r.relation_id.slice(-12));
 text('pairDetails','Endpoint gap '+Number(r.measured_endpoint_gap_px).toFixed(2)+' px | Machine disagreement='+Number(r.local_vs_ANZA_score_disagreement).toFixed(3));
 text('evidence',JSON.stringify(r.provisional_scores_NOT_PHYSICAL_PROBABILITIES,null,2));
 text('pairStatus',relation_decisions[r.relation_id]?.status||'UNREVIEWED');
 const reviewed=Object.keys(relation_decisions).length;
 text('progress','Reviewed pairs '+reviewed+'/'+R.length+' | reviewed lines '+Object.keys(decisions).length+'/'+drafts.size);
 refreshLine();draw();
}
function choose(key){const id=current().relation_id;if(key==='clear')delete relation_decisions[id];else relation_decisions[id]={status:DEC[key],review_origin:'LOCAL_USER_NOT_PHYSICIST_CERTIFIED'};refresh()}
function chooseLine(key){if(!lineId)return;if(key==='clear')delete decisions[lineId];else decisions[lineId]={status:DL[key],edited_points_xy:null,review_origin:'LOCAL_USER_PROVISIONAL_NOT_PHYSICIST_CERTIFIED'};refresh()}
function forward(){idx=(idx+1)%R.length;lineId=null;refresh()}
$('step').onclick=forward;$('prev').onclick=()=>{idx=(idx+R.length-1)%R.length;lineId=null;refresh()};
$('next').onclick=()=>{for(let k=1;k<=R.length;k++){const i=(idx+k)%R.length;if(!relation_decisions[R[i].relation_id]){idx=i;lineId=null;refresh();return}}text('message','All relation questions have an answer. Export your JSON.')};
$('showAll').onchange=draw;$('draftSelect').onchange=()=>{lineId=$('draftSelect').value;refresh()};
$('same').onclick=()=>choose('same');$('diff').onclick=()=>choose('diff');$('unsure').onclick=()=>choose('unsure');$('clearPair').onclick=()=>choose('clear');
$('acceptLine').onclick=()=>chooseLine('accept');$('rejectLine').onclick=()=>chooseLine('reject');$('unsureLine').onclick=()=>chooseLine('unsure');$('clearLine').onclick=()=>chooseLine('clear');
function exportState(){return {...B.template,provenance:'LOCAL_USER_CORRECTIONS_NOT_PHYSICIST_CERTIFIED',decisions:JSON.parse(JSON.stringify(decisions)),relation_decisions:JSON.parse(JSON.stringify(relation_decisions))}}
$('export').onclick=()=>{const txt=JSON.stringify(exportState(),null,2),url=URL.createObjectURL(new Blob([txt],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='S31_machine_labels_local_corrections.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);text('message','Corrections exported. Keep this JSON private.')};
$('importFile').onchange=async event=>{try{
 const f=event.target.files?.[0];if(!f)return;const obj=JSON.parse(await f.text());
 if(obj.schema!==B.template.schema||obj.machine_source_s28_sha256!==B.template.machine_source_s28_sha256||obj.machine_source_s29_sha256!==B.template.machine_source_s29_sha256||obj.all_unreviewed_stay_unknown!==true||obj.physical_joins_authorized!==false)throw Error('Incompatible provenance or unsafe policy');
 if(JSON.stringify(obj.crops)!==JSON.stringify(B.template.crops))throw Error('Original source images changed');
 if(!obj.decisions||Array.isArray(obj.decisions)||!obj.relation_decisions||Array.isArray(obj.relation_decisions))throw Error('Invalid correction maps');
 for(const [key,val] of Object.entries(obj.decisions)){if(!drafts.has(key)||!Object.values(DL).includes(val.status))throw Error('Invalid machine-line decision');if(val.edited_points_xy!==null&&val.edited_points_xy!==undefined){if(!Array.isArray(val.edited_points_xy)||val.edited_points_xy.some(p=>!Array.isArray(p)||p.length!==2||p.some(x=>!Number.isFinite(x)||x<0||x>511)))throw Error('Invalid edited path')}}
 for(const [key,val] of Object.entries(obj.relation_decisions)){if(!relations.has(key)||!Object.values(DEC).includes(val.status))throw Error('Invalid pair decision')}
 decisions=obj.decisions;relation_decisions=obj.relation_decisions;refresh();text('message','Imported verified corrections.');
 }catch(err){text('message','Import blocked: '+String(err.message))}};
window.onkeydown=e=>{if(e.target instanceof HTMLInputElement||e.target instanceof HTMLSelectElement)return;if(e.key==='1')choose('same');if(e.key==='2')choose('diff');if(e.key==='3')choose('unsure');if(e.key==='ArrowRight')forward();if(e.key==='ArrowLeft')$('prev').click()};
refresh();
</script></body></html>'''


def build(private_zip:Path,output_zip:Path):
    with zipfile.ZipFile(private_zip) as z:
        drafts=json.loads(z.read('annotations/S31_MACHINE_DRAFTS.json'))
        evidence=json.loads(z.read('review/S35_28_REAL_H50Z_PAIR_QUESTIONS_PRIVATE.json'))
        template=json.loads(z.read('annotations/S31_EMPTY_CORRECTIONS_TEMPLATE.json'))
        assert len(drafts['photos'])==39 and len(evidence['rows'])==28
        assert sum(len(p['drafts']) for p in drafts['photos'])==133
        assert sum(len(p['relation_drafts']) for p in drafts['photos'])==28
        payload={'photos':drafts['photos'],'relations':evidence['rows'],'template':template}
        public=False
        if public:raise AssertionError('Cannot export private reviewer as public file')
        data=json.dumps(payload,ensure_ascii=True,separators=(',',':')).replace('<','\\u003c')
        assert '__PRIVATE_JSON__' in HTML
        page=HTML.replace('__PRIVATE_JSON__',data)
        output_zip.parent.mkdir(exist_ok=True,parents=True)
        checks={}
        with zipfile.ZipFile(output_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as out:
            out.writestr('S37_REVIEWER/index.html',page.encode())
            for p in drafts['photos']:
                name=p['crop_id']+'.png'
                photo=z.read('source_39_H50Z_crops/'+name)
                digest=sha256(photo).hexdigest()
                pixel_array=cv2.imdecode(np.frombuffer(photo,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
                if pixel_array is None or pixel_array.shape!=(512,512):
                    raise ValueError('Missing source grayscale pixels: '+name)
                pixel_sha=sha256(pixel_array.tobytes()).hexdigest()
                if pixel_sha!=p['source_gray_pixels_sha256']:
                    raise ValueError('Crop pixel SHA-256 mismatch: '+name)
                # Nine start PNGs were losslessly re-encoded; original PNG-bytes
                # hash may differ while source raw uint8 pixel hash is identical.
                checks[name]={'actual_encoded_sha256':digest,
                    'source_png_sha256':p['source_original_png_sha256_from_S29'],
                    'matching_grayscale_pixels_sha256':pixel_sha,
                    'original_png_encoding_identical':digest==p['source_original_png_sha256_from_S29']}
                out.writestr('S37_REVIEWER/crops/'+name,photo)
            out.writestr('S37_REVIEWER/README.txt',
                'PRIVATE source photos. Unzip all files. Open index.html locally.\n'
                'Review pairs/lines and EXPORT S31 correction JSON before closing.\n'
                'No browser server is needed; no network calls. No physical auto joins.\n'
                'Never commit this bundle or exported correction file to public GitHub.\n')
            out.writestr('S37_REVIEWER/integrity.json',json.dumps({
                'schema':'s37_review_bundle_integrity_v1','source_archive_sha256':sha256(private_zip.read_bytes()).hexdigest(),
                'images':checks,'original_draft_count':133,'relation_count':28,
                'physical_join_authorized':False},indent=2))
    with zipfile.ZipFile(output_zip) as z:
        assert z.testzip() is None
    print('offline reviewer',output_zip,'size',output_zip.stat().st_size,'source images',len(checks))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private-archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    build(a.private_archive,a.out)
