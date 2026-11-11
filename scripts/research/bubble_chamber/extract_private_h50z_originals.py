"""Private extraction of known H50Z original BMP/labels from 7z archive.

Permitted paths only, no overwrite, no extraction into public Git checkout.
"""
from __future__ import annotations
import argparse,ctypes as c,json
from pathlib import Path

ALLOWED = {'H50Z_030_01_005.bmp','H50Z_030_01_006.bmp','labels.json'}

def extract(archive:Path, output:Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refuse overwriting original private H50Z extraction')
    output.mkdir(parents=True,exist_ok=True)
    L=c.CDLL('libarchive.so.13')
    L.archive_read_new.restype=c.c_void_p
    for name in ('archive_read_support_filter_all','archive_read_support_format_all'):
        getattr(L,name).argtypes=[c.c_void_p]
    L.archive_read_open_filename.argtypes=[c.c_void_p,c.c_char_p,c.c_size_t]
    L.archive_read_open_filename.restype=c.c_int
    L.archive_read_next_header.argtypes=[c.c_void_p,c.POINTER(c.c_void_p)]
    L.archive_read_next_header.restype=c.c_int
    L.archive_entry_pathname.argtypes=[c.c_void_p];L.archive_entry_pathname.restype=c.c_char_p
    L.archive_entry_size.argtypes=[c.c_void_p];L.archive_entry_size.restype=c.c_int64
    L.archive_read_data.argtypes=[c.c_void_p,c.c_void_p,c.c_size_t];L.archive_read_data.restype=c.c_ssize_t
    L.archive_read_free.argtypes=[c.c_void_p]
    a=L.archive_read_new();names=[]
    try:
        L.archive_read_support_filter_all(a);L.archive_read_support_format_all(a)
        if L.archive_read_open_filename(a,str(archive).encode(),10240)!=0:
            raise IOError('Unable to open original archive')
        entry=c.c_void_p();buf=c.create_string_buffer(4<<20)
        import hashlib
        while L.archive_read_next_header(a,c.byref(entry))==0:
            name=L.archive_entry_pathname(entry).decode('utf-8')
            if name not in ALLOWED or Path(name).name!=name:
                raise ValueError('Unexpected path in archive: '+name)
            size=L.archive_entry_size(entry)
            dest=output/name
            if dest.exists():raise FileExistsError(dest)
            written=0;h=hashlib.sha256()
            with dest.open('xb') as f:
                while True:
                    count=L.archive_read_data(a,buf,len(buf))
                    if count<0:raise IOError('Archive read error')
                    if count==0:break
                    data=bytes(buf[:count]);f.write(data);h.update(data);written+=count
            if written!=size:raise IOError('Size mismatch for '+name)
            names.append({'name':name,'size':written,'sha256':h.hexdigest()})
        (output/'EXTRACTION_MANIFEST_PRIVATE.json').write_text(json.dumps(names,indent=2),encoding='utf-8')
        return names
    finally:L.archive_read_free(a)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--private-out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(extract(a.archive,a.private_out),indent=2))