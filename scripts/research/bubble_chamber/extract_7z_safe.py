"""Extract only visualMetro TXT files from a local 7z archive using system libarchive."""
from __future__ import annotations
import argparse
import ctypes as C
import ctypes.util
import hashlib
import json
from pathlib import Path, PurePosixPath

lib=C.CDLL(ctypes.util.find_library("archive"))
lib.archive_read_new.restype=C.c_void_p
lib.archive_read_support_format_all.argtypes=[C.c_void_p]
lib.archive_read_support_filter_all.argtypes=[C.c_void_p]
lib.archive_read_open_filename.argtypes=[C.c_void_p,C.c_char_p,C.c_size_t]
lib.archive_read_next_header.argtypes=[C.c_void_p,C.POINTER(C.c_void_p)]
lib.archive_entry_pathname.argtypes=[C.c_void_p]
lib.archive_entry_pathname.restype=C.c_char_p
lib.archive_entry_size.argtypes=[C.c_void_p]
lib.archive_entry_size.restype=C.c_int64
lib.archive_read_data.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t]
lib.archive_read_data.restype=C.c_ssize_t
lib.archive_read_free.argtypes=[C.c_void_p]
lib.archive_error_string.argtypes=[C.c_void_p]
lib.archive_error_string.restype=C.c_char_p

def extract(archive:Path,dest:Path)->list[dict]:
    if not archive.is_file():raise FileNotFoundError(archive)
    r=lib.archive_read_new()
    if not r:raise RuntimeError("libarchive allocation failed")
    lib.archive_read_support_format_all(r)
    lib.archive_read_support_filter_all(r)
    if lib.archive_read_open_filename(r,str(archive).encode(),10240)!=0:
        error=lib.archive_error_string(r)
        lib.archive_read_free(r)
        raise RuntimeError(error)
    root=dest.resolve()
    root.mkdir(parents=True,exist_ok=True)
    manifest=[]
    header=C.c_void_p()
    try:
        while True:
            status=lib.archive_read_next_header(r,C.byref(header))
            if status==1:break
            if status!=0:raise RuntimeError(lib.archive_error_string(r))
            name=lib.archive_entry_pathname(header).decode("utf-8","surrogateescape").replace("\\","/")
            parts=PurePosixPath(name).parts
            if not parts or any(s in ("","..") for s in parts) or name.startswith("/"):
                raise ValueError("Unsafe archive path: "+name)
            size=lib.archive_entry_size(header)
            if size<0 or size>20_000_000:raise ValueError("Unexpected entry size: "+name)
            path=root.joinpath(*parts)
            if not path.resolve().is_relative_to(root):raise ValueError("Path escape: "+name)
            if name.endswith("/") or (size==0 and not PurePosixPath(name).suffix):
                path.mkdir(parents=True,exist_ok=True)
                continue
            if not name.lower().endswith(".txt"):continue
            path.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256()
            got=0
            with path.open("wb") as f:
                buf=C.create_string_buffer(65536)
                while True:
                    n=lib.archive_read_data(r,buf,len(buf))
                    if n<0:raise RuntimeError(lib.archive_error_string(r))
                    if n==0:break
                    f.write(buf.raw[:n]);digest.update(buf.raw[:n]);got+=n
            if got!=size:raise ValueError("Unexpected file byte count: "+name)
            manifest.append({"path":name,"bytes":got,"sha256":digest.hexdigest()})
    finally:lib.archive_read_free(r)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("archive",type=Path);p.add_argument("dest",type=Path)
    a=p.parse_args();m=extract(a.archive,a.dest)
    (a.dest/"safe_extraction_manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print(json.dumps({"files":len(m),"bytes":sum(x["bytes"] for x in m)}))
