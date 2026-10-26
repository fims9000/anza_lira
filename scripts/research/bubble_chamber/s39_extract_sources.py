#!/usr/bin/env python3
"""Offline source-locked H50Z extraction with system libarchive, not Git content.

No private BMP/JSON or credentials are committed. Files are created only under
an explicitly passed private output path after SHA-256 and member checks.
"""
from __future__ import annotations
import argparse
from ctypes import CDLL, c_void_p,c_int,c_size_t,c_char_p,c_longlong,create_string_buffer,POINTER,byref
from ctypes.util import find_library
from hashlib import sha256
from pathlib import Path

EXPECTED_ARCHIVE='041a89f5b71a1ed4945729208c64e9064b9ebce3e0400f5191972a2f1e06558a'
EXPECTED_MEMBERS={'H50Z_030_01_006.bmp':234703910,
                  'labels.json':348015,
                  'H50Z_030_01_005.bmp':235883098}


def extract_sources(archive:Path,out:Path):
    archive=Path(archive);out=Path(out)
    if sha256(archive.read_bytes()).hexdigest()!=EXPECTED_ARCHIVE:
        raise ValueError('H50Z original archive SHA-256 mismatch')
    lib=find_library('archive')
    if lib is None:raise RuntimeError('System libarchive is required to read .7z offline')
    a=CDLL(lib)
    def bind(name,restype,args):
        f=getattr(a,name);f.restype=restype;f.argtypes=args;return f
    new=bind('archive_read_new',c_void_p,[])
    filters=bind('archive_read_support_filter_all',c_int,[c_void_p])
    formats=bind('archive_read_support_format_all',c_int,[c_void_p])
    openfile=bind('archive_read_open_filename',c_int,[c_void_p,c_char_p,c_size_t])
    nextheader=bind('archive_read_next_header',c_int,[c_void_p,POINTER(c_void_p)])
    namefn=bind('archive_entry_pathname',c_char_p,[c_void_p])
    sizefn=bind('archive_entry_size',c_longlong,[c_void_p])
    read=bind('archive_read_data',c_longlong,[c_void_p,c_void_p,c_size_t])
    close=bind('archive_read_free',c_int,[c_void_p])
    error=bind('archive_error_string',c_char_p,[c_void_p])
    p=new();seen=set()
    try:
        if filters(p)!=0 or formats(p)!=0 or openfile(p,str(archive).encode(),10240)!=0:
            raise RuntimeError('Archive open failure')
        out.mkdir(parents=True,exist_ok=True)
        buf=create_string_buffer(1<<20)
        while True:
            header=c_void_p();status=nextheader(p,byref(header))
            if status==1:break
            if status!=0:raise RuntimeError(str(error(p)))
            member=namefn(header).decode('utf-8');size=sizefn(header)
            if member not in EXPECTED_MEMBERS or member in seen or size!=EXPECTED_MEMBERS[member]:
                raise ValueError('Unapproved 7z member, duplicate or untrusted size')
            target=out/member;tmp=out/(member+'.tmp')
            done=0
            with tmp.open('wb') as f:
                while True:
                    k=read(p,buf,len(buf))
                    if k<0:raise RuntimeError('libarchive decompression failure')
                    if not k:break
                    done+=k
                    if done>size:raise ValueError('Archive member exceeds declared size')
                    f.write(buf.raw[:k])
            if done!=size:raise IOError('Truncated extracted member')
            tmp.replace(target);seen.add(member)
        if seen!=set(EXPECTED_MEMBERS):raise ValueError('Missing required H50Z source members')
    finally:
        close(p)
    return {name:(out/name).stat().st_size for name in sorted(seen)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-7z',type=Path,required=True)
    p.add_argument('--private-out-dir',type=Path,required=True)
    a=p.parse_args();print(extract_sources(a.source_7z,a.private_out_dir))
if __name__=='__main__':main()
