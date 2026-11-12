#!/usr/bin/env python3
"""Download and hash-check official CRACKS v2 archives; never commit images."""
from __future__ import annotations
import argparse, hashlib, json, time, zipfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

RECORD="13926822"
EXPECTED={"images.zip":"6557236191763af7bd8298ecb136d41e","Fault segmentations.zip":"01e1697e886da2079ff3c1967334a7ca"}
def hash_file(path):
    h=hashlib.md5()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(1048576),b""): h.update(part)
    return h.hexdigest()
def fetch(url,target):
    req=Request(url,headers={"User-Agent":"ANZA-LIRA-CRACKS-reproduction/1.0"})
    tmp=target.with_name(target.name+".partial")
    for attempt in range(4):
        try:
            with urlopen(req,timeout=120) as stream,tmp.open("wb") as out:
                while chunk:=stream.read(1048576):out.write(chunk)
            tmp.replace(target);return
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt==3:raise
            time.sleep(min(2**attempt,8))
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="data/cracks/raw")
    parser.add_argument("--receipt",default="results/cracks_real_controlled_v1/download_receipt.json")
    args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    receipt=Path(args.receipt);receipt.parent.mkdir(parents=True,exist_ok=True)
    result={"source":"https://zenodo.org/records/13926822","files":{}}
    try:
        with urlopen(Request(f"https://zenodo.org/api/records/{RECORD}",headers={"User-Agent":"ANZA-LIRA-CRACKS-reproduction/1.0"}),timeout=45) as resp:
            record=json.load(resp)
        urls={x["key"]:x["links"]["self"] for x in record["files"]}
        for name,expected in EXPECTED.items():
            target=root/name
            if not target.exists() or hash_file(target)!=expected:
                errors=[]
                for url in filter(None,[urls.get(name),f"https://zenodo.org/records/{RECORD}/files/{quote(name)}?download=1"]):
                    try:
                        fetch(url,target)
                        if hash_file(target)==expected:break
                        errors.append("checksum mismatch "+url);target.unlink(missing_ok=True)
                    except Exception as error:errors.append(str(error))
                else:raise RuntimeError(f"failed to obtain {name}: "+"; ".join(errors))
            if not zipfile.is_zipfile(target):raise RuntimeError("non-zip "+name)
            with zipfile.ZipFile(target) as archive:
                bad=archive.testzip()
                if bad:raise RuntimeError("damaged ZIP member "+bad)
                names=[x.filename for x in archive.infolist() if not x.is_dir()]
                if any(x.startswith("/") or ".." in Path(x).parts for x in names):raise RuntimeError("unsafe ZIP member")
                result["files"][name]={"md5":hash_file(target),"bytes":target.stat().st_size,"members":len(names),"example_names":names[:15]}
        result["status"]="VERIFIED_REAL_CRACKS_INPUTS"
    except Exception as error:
        result["status"]="STOP_DATA_DOWNLOAD";result["error"]=f"{type(error).__name__}: {error}"
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result["status"]=="VERIFIED_REAL_CRACKS_INPUTS" else 2
if __name__=="__main__":raise SystemExit(main())
