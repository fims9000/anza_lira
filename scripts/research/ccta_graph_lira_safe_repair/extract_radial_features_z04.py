#!/usr/bin/env python3
"""Graph-LIRA matched-CCTA expansion without moving raw CT off the workstation.

Reads original ImageCAS CT entries directly from the already downloaded Kaggle
801-1000.z04 + 801-1000.zip split archive, verifies each CT against frozen
ImageCAS-X grid metadata, samples the frozen radial HU corridor representation
for a precomputed geometry-matched relation plan, and writes a compact result
bundle.

Raw CT bytes are never written to the output bundle.
"""
from __future__ import annotations

import argparse
import binascii
import csv
import gzip
import hashlib
import io
import json
import math
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

SCRIPT_VERSION = "graphlira-z04-radial-expansion-v1"
PROFILE_N = 17
RING_N = 8
RADII_MM = (1.0, 2.0, 3.0)
FPR_BUDGET = 0.05

DTYPE_MAP = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument(
        "--archive-dir",
        type=Path,
        default=Path.home() / "GraphLIRA_CT_subset" / "kaggle_801_1000",
        help="directory containing 801-1000.z04 and 801-1000.zip",
    )
    p.add_argument(
        "--plan",
        type=Path,
        default=here / "relation_pair_plan.csv",
    )
    p.add_argument(
        "--expected-geometry",
        type=Path,
        default=here / "expected_ct_geometry.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "GraphLIRA_CT_expanded_radial",
    )
    return p.parse_args()


class SplitZipReader:
    """Minimal reader for entries beginning on z04 of the 801-1000 split ZIP."""

    def __init__(self, z04: Path, final_zip: Path):
        self.parts = {3: z04, 4: final_zip}  # raw ZIP disk numbers are zero-based
        for p in self.parts.values():
            if not p.exists():
                raise FileNotFoundError(p)
        self.final = final_zip
        self.entries = self._read_central_directory()

    def _read_central_directory(self):
        with self.final.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            tail_n = min(size, 22 + 65535 + 4096)
            f.seek(size - tail_n)
            tail = f.read(tail_n)
        pos = tail.rfind(b"PK\x05\x06")
        if pos < 0:
            raise RuntimeError("ZIP EOCD not found in final 801-1000.zip")
        eocd = tail[pos : pos + 22]
        sig, this_disk, cd_disk, n_this, n_total, cd_size, cd_offset, comment_len = struct.unpack(
            "<4s4H2IH", eocd
        )
        if cd_disk != this_disk:
            raise RuntimeError(
                f"central directory is on disk {cd_disk}, final disk is {this_disk}"
            )
        with self.final.open("rb") as f:
            f.seek(cd_offset)
            cd = f.read(cd_size)
        entries = {}
        p = 0
        while p + 46 <= len(cd):
            if cd[p : p + 4] != b"PK\x01\x02":
                break
            vals = struct.unpack_from("<4s6H3I5H2I", cd, p)
            (
                _sig,
                ver_made,
                ver_need,
                flags,
                method,
                mtime,
                mdate,
                crc32,
                csize,
                usize,
                fn_len,
                ex_len,
                comment_len,
                disk_start,
                int_attr,
                ext_attr,
                local_off,
            ) = vals
            name_b = cd[p + 46 : p + 46 + fn_len]
            enc = "utf-8" if flags & 0x800 else "cp437"
            name = name_b.decode(enc, errors="replace")
            entries[name] = {
                "flags": flags,
                "method": method,
                "crc32": crc32,
                "csize": csize,
                "usize": usize,
                "disk_start": disk_start,
                "local_off": local_off,
            }
            p += 46 + fn_len + ex_len + comment_len
        if not entries:
            raise RuntimeError("central directory parsed but no entries were found")
        return entries

    def _read_across(self, disk: int, offset: int, count: int) -> bytes:
        out = bytearray()
        remaining = count
        d = disk
        off = offset
        while remaining:
            path = self.parts.get(d)
            if path is None:
                raise RuntimeError(
                    f"entry needs raw ZIP disk {d}; only z04(raw 3) + final(raw 4) are available"
                )
            size = path.stat().st_size
            if off > size:
                raise RuntimeError(f"offset {off} exceeds {path} size {size}")
            take = min(remaining, size - off)
            if take <= 0:
                d += 1
                off = 0
                continue
            with path.open("rb") as f:
                f.seek(off)
                chunk = f.read(take)
            if len(chunk) != take:
                raise RuntimeError(f"short read from {path}")
            out.extend(chunk)
            remaining -= take
            d += 1
            off = 0
        return bytes(out)

    def _local_header(self, disk, off):
        # Some split writers include a 4-byte split marker; tolerate only +/-4.
        candidates = [off, off + 4]
        if off >= 4:
            candidates.append(off - 4)
        for q in candidates:
            h = self._read_across(disk, q, 30)
            if h[:4] == b"PK\x03\x04":
                return q, h
        raise RuntimeError(f"local header not found disk={disk} off={off}")

    def extract_entry_bytes(self, name: str):
        if name not in self.entries:
            raise KeyError(f"archive entry not found: {name}")
        e = self.entries[name]
        if e["disk_start"] != 3:
            raise RuntimeError(
                f"{name} starts on raw ZIP disk {e['disk_start']}; expected z04/raw disk 3"
            )
        if e["flags"] & 0x1:
            raise RuntimeError(f"encrypted ZIP entry unsupported: {name}")
        local_off, h = self._local_header(e["disk_start"], e["local_off"])
        (
            _sig,
            ver_need,
            flags,
            method,
            mtime,
            mdate,
            local_crc,
            local_csize,
            local_usize,
            fn_len,
            ex_len,
        ) = struct.unpack("<4s5H3I2H", h)
        data_off = local_off + 30 + fn_len + ex_len
        comp = self._read_across(e["disk_start"], data_off, e["csize"])
        if method == 0:
            raw = comp
        elif method == 8:
            raw = zlib.decompress(comp, -15)
        else:
            raise RuntimeError(f"unsupported ZIP compression method {method}: {name}")
        if len(raw) != e["usize"]:
            raise RuntimeError(f"size mismatch for {name}: {len(raw)} != {e['usize']}")
        crc = binascii.crc32(raw) & 0xFFFFFFFF
        if crc != e["crc32"]:
            raise RuntimeError(f"CRC mismatch for {name}: {crc:08x} != {e['crc32']:08x}")
        if not raw.startswith(b"\x1f\x8b"):
            raise RuntimeError(f"{name} is not gzip-compressed NIfTI bytes")
        return raw, e


def nifti_from_gz_bytes(raw: bytes):
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as f:
        blob = f.read()
    if len(blob) < 352:
        raise RuntimeError("truncated NIfTI")
    le = struct.unpack("<i", blob[:4])[0]
    be = struct.unpack(">i", blob[:4])[0]
    if le == 348:
        endian = "<"
    elif be == 348:
        endian = ">"
    else:
        raise RuntimeError(f"invalid NIfTI sizeof_hdr: {le}/{be}")
    dim = struct.unpack(endian + "8h", blob[40:56])
    ndim = int(dim[0])
    shape = tuple(int(x) for x in dim[1 : 1 + ndim])
    datatype = struct.unpack(endian + "h", blob[70:72])[0]
    pixdim = struct.unpack(endian + "8f", blob[76:108])
    vox_offset = int(round(struct.unpack(endian + "f", blob[108:112])[0]))
    slope = float(struct.unpack(endian + "f", blob[112:116])[0])
    inter = float(struct.unpack(endian + "f", blob[116:120])[0])
    sform_code = struct.unpack(endian + "h", blob[254:256])[0]
    sx = struct.unpack(endian + "4f", blob[280:296])
    sy = struct.unpack(endian + "4f", blob[296:312])
    sz = struct.unpack(endian + "4f", blob[312:328])
    affine = np.array([sx, sy, sz, [0.0, 0.0, 0.0, 1.0]], dtype=float)
    if sform_code <= 0:
        raise RuntimeError("NIfTI sform_code is zero; frozen ImageCAS protocol expects sform")
    if datatype not in DTYPE_MAP:
        raise RuntimeError(f"unsupported NIfTI datatype {datatype}")
    dtype = np.dtype(DTYPE_MAP[datatype]).newbyteorder(endian)
    n = int(np.prod(shape))
    arr = np.frombuffer(blob, dtype=dtype, count=n, offset=vox_offset).reshape(shape, order="F")
    scale = slope if slope not in (0.0, -0.0) else 1.0
    if scale != 1.0 or inter != 0.0:
        arr = arr.astype(np.float32) * scale + inter
    meta = {
        "shape": shape[:3],
        "spacing": tuple(abs(float(x)) for x in pixdim[1:4]),
        "affine": affine,
        "datatype": datatype,
        "sform_code": int(sform_code),
    }
    return meta, arr, blob


def orthobasis(d):
    d = np.asarray(d, float)
    d /= max(np.linalg.norm(d), 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, ref)
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(d, u)
    v /= max(np.linalg.norm(v), 1e-12)
    return u, v


def sample_world(ct, inv_affine, xyz_lps):
    xyz = np.array(xyz_lps, dtype=float, copy=True)
    # ImageCAS-X VTK is LPS; original ImageCAS NIfTI uses RAS world coordinates.
    xyz[..., 0] *= -1.0
    xyz[..., 1] *= -1.0
    flat = xyz.reshape(-1, 3)
    vox = (inv_affine @ np.c_[flat, np.ones(len(flat))].T).T[:, :3]
    vals = map_coordinates(
        ct,
        [vox[:, 0], vox[:, 1], vox[:, 2]],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return vals.reshape(xyz.shape[:-1])


def radial_features(row, ct, inv_affine):
    p1 = np.array([row.p1_0, row.p1_1, row.p1_2], float)
    p2 = np.array([row.p2_0, row.p2_1, row.p2_2], float)
    v = p2 - p1
    dist = float(np.linalg.norm(v))
    d = v / max(dist, 1e-12)
    u, w = orthobasis(d)
    ts = np.linspace(0.05, 0.95, PROFILE_N)
    centers = p1[None, :] + ts[:, None] * v[None, :]
    center = sample_world(ct, inv_affine, centers)
    angles = np.linspace(0, 2 * np.pi, RING_N, endpoint=False)
    rings = []
    for radius in RADII_MM:
        pts = np.stack(
            [centers + radius * (math.cos(a) * u + math.sin(a) * w)[None, :] for a in angles],
            axis=1,
        )
        vals = sample_world(ct, inv_affine, pts)
        rings.append(vals.mean(axis=1))

    cc = np.clip(center.astype(float), -300, 1200)
    feats = [
        cc.mean(), cc.std(), cc.min(), np.percentile(cc, 10), np.median(cc),
        np.percentile(cc, 90), cc.max(), np.mean(cc > 150), np.mean(cc > 250),
        np.mean(np.abs(np.diff(cc))),
    ]
    for rr0 in rings:
        rr = np.clip(rr0.astype(float), -300, 1200)
        contrast = cc - rr
        feats += [
            rr.mean(), rr.std(), contrast.mean(), contrast.min(), np.percentile(contrast, 10),
            np.median(contrast), np.mean(contrast > 25), np.mean(contrast > 75),
        ]
    outer = np.clip(rings[-1].astype(float), -300, 1200)
    contrast = cc - outer
    core = slice(3, -3) if len(cc) > 6 else slice(None)
    feats += [
        np.min(cc[core]),
        np.min(contrast[core]),
        np.mean((cc > 150) & (contrast > 25)),
        np.mean((cc > 200) & (contrast > 50)),
    ]
    return np.asarray(feats, float)


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path)


def select_threshold(y, scores, budget=FPR_BUDGET):
    y = np.asarray(y, int)
    scores = np.asarray(scores, float)
    best = None
    for t in np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]:
        pred = scores >= t
        fp = int(np.sum(pred & (y == 0)))
        tp = int(np.sum(pred & (y == 1)))
        nneg = int(np.sum(y == 0))
        npos = int(np.sum(y == 1))
        fpr = fp / max(nneg, 1)
        rec = tp / max(npos, 1)
        prec = tp / max(int(np.sum(pred)), 1)
        if fpr <= budget + 1e-12:
            key = (rec, prec, -fpr, -float(t))
            if best is None or key > best[0]:
                best = (key, float(t))
    return best[1]


def evaluate_models(df, out_dir: Path):
    try:
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        print(f"[WARN] sklearn evaluation skipped: {exc}")
        return None

    geom = [f"g{i}" for i in range(7)]
    ctcols = [c for c in df.columns if c.startswith("ct") and c[2:].isdigit()]
    feature_sets = {
        "geometry": geom,
        "radial_hu_summary_v1": ctcols,
        "geometry_plus_radial_v1": geom + ctcols,
    }
    rows = []
    preds = df[["scan_id", "split", "pair_id", "y", "kind", "name1", "name2"]].copy()

    for model_name, cols in feature_sets.items():
        tr = df[df.split == "train"]
        va = df[df.split == "val"]
        te = df[df.split == "test"]
        model = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, solver="lbfgs")),
        ])
        model.fit(tr[cols], tr.y)
        score_map = {
            "train": model.predict_proba(tr[cols])[:, 1],
            "val": model.predict_proba(va[cols])[:, 1],
            "test": model.predict_proba(te[cols])[:, 1],
        }
        thr = select_threshold(va.y.to_numpy(), score_map["val"])
        allscores = np.empty(len(df), float)
        for split, sub in [("train", tr), ("val", va), ("test", te)]:
            s = score_map[split]
            y = sub.y.to_numpy(int)
            pred = s >= thr
            tp = int(np.sum(pred & (y == 1)))
            fp = int(np.sum(pred & (y == 0)))
            fn = int(np.sum((~pred) & (y == 1)))
            tn = int(np.sum((~pred) & (y == 0)))
            rows.append({
                "model": model_name,
                "split": split,
                "n": len(sub),
                "patients": int(sub.scan_id.nunique()),
                "auroc": float(roc_auc_score(y, s)),
                "auprc": float(average_precision_score(y, s)),
                "threshold": thr,
                "recall": tp / max(tp + fn, 1),
                "fpr": fp / max(fp + tn, 1),
                "precision": tp / max(tp + fp, 1),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })
            allscores[df.split.to_numpy() == split] = s
            for sid, g in sub.groupby("scan_id"):
                idx = sub.index.get_indexer(g.index)
                sg = s[idx]
                yg = g.y.to_numpy(int)
                pg = sg >= thr
                tp2 = int(np.sum(pg & (yg == 1)))
                fp2 = int(np.sum(pg & (yg == 0)))
                fn2 = int(np.sum((~pg) & (yg == 1)))
                tn2 = int(np.sum((~pg) & (yg == 0)))
                rows.append({
                    "model": model_name,
                    "split": f"{split}_patient",
                    "scan_id": int(sid),
                    "n": len(g),
                    "patients": 1,
                    "auroc": float(roc_auc_score(yg, sg)),
                    "auprc": float(average_precision_score(yg, sg)),
                    "threshold": thr,
                    "recall": tp2 / max(tp2 + fn2, 1),
                    "fpr": fp2 / max(fp2 + tn2, 1),
                    "precision": tp2 / max(tp2 + fp2, 1),
                    "tp": tp2, "fp": fp2, "fn": fn2, "tn": tn2,
                })
        preds[model_name + "_score"] = allscores
        preds[model_name + "_pred"] = allscores >= thr

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "expanded_relation_summary.csv", index=False)
    preds.to_csv(out_dir / "expanded_relation_predictions.csv", index=False)
    print("\nExpanded patient-level result:")
    print(summary[summary.split.isin(["val", "test"])][[
        "model", "split", "patients", "n", "auroc", "auprc", "recall", "fpr", "precision", "tp", "fp"
    ]].to_string(index=False))
    return summary


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan = read_csv(args.plan)
    expected = read_csv(args.expected_geometry).set_index("scan_id")
    ids = sorted(int(x) for x in plan.scan_id.unique())

    z04 = args.archive_dir / "801-1000.z04"
    final_zip = args.archive_dir / "801-1000.zip"
    print(SCRIPT_VERSION)
    print("archive:", args.archive_dir)
    print("patients:", len(ids), ids)
    reader = SplitZipReader(z04, final_zip)

    feature_rows = []
    align_rows = []
    for ix, sid in enumerate(ids, 1):
        name = f"801-1000/{sid}.img.nii.gz"
        raw, ze = reader.extract_entry_bytes(name)
        sha = hashlib.sha256(raw).hexdigest()
        meta, ct, uncompressed_blob = nifti_from_gz_bytes(raw)
        exp = expected.loc[sid]
        exp_shape = (int(exp.shape_x), int(exp.shape_y), int(exp.shape_z))
        exp_spacing = (float(exp.spacing_x), float(exp.spacing_y), float(exp.spacing_z))
        exp_aff = np.eye(4, dtype=float)
        for r in range(3):
            for c in range(4):
                exp_aff[r, c] = float(exp[f"affine_{r}{c}"])
        shape_ok = tuple(meta["shape"]) == exp_shape
        spacing_ok = bool(np.allclose(meta["spacing"], exp_spacing, atol=1e-7, rtol=0))
        aff_err = float(np.max(np.abs(meta["affine"] - exp_aff)))
        affine_ok = aff_err <= 1e-6
        if not (shape_ok and spacing_ok and affine_ok):
            raise RuntimeError(
                f"CT/ImageCAS-X grid mismatch for {sid}: shape={shape_ok}, spacing={spacing_ok}, affine_err={aff_err}"
            )
        align_rows.append({
            "scan_id": sid,
            "split": str(exp["split"]),
            "ct_sha256": sha,
            "archive_disk_start_raw": ze["disk_start"],
            "archive_compressed_bytes": ze["csize"],
            "shape": "x".join(map(str, meta["shape"])),
            "spacing": "x".join(f"{x:.9g}" for x in meta["spacing"]),
            "shape_match": shape_ok,
            "spacing_match": spacing_ok,
            "affine_max_abs_error": aff_err,
            "affine_match": affine_ok,
        })
        inv = np.linalg.inv(meta["affine"])
        sub = plan[plan.scan_id == sid]
        print(f"[{ix:02d}/{len(ids)}] {sid} {exp['split']}: {len(sub)} relation rows, {len(raw)/1024**2:.1f} MiB CT")
        for row in sub.itertuples(index=False):
            out = row._asdict()
            feats = radial_features(row, ct, inv)
            for j, v in enumerate(feats):
                out[f"ct{j}"] = float(v)
            feature_rows.append(out)
        # Drop references before next CT.
        del ct, raw, uncompressed_blob

    import pandas as pd
    features = pd.DataFrame(feature_rows)
    alignment = pd.DataFrame(align_rows)
    features.to_csv(args.out_dir / "expanded_relation_features.csv", index=False)
    alignment.to_csv(args.out_dir / "ct_alignment.csv", index=False)

    protocol = {
        "script_version": SCRIPT_VERSION,
        "patient_ids": ids,
        "split_counts": {k: int(features[features.split == k].scan_id.nunique()) for k in ["train", "val", "test"]},
        "relation_rows": {k: int((features.split == k).sum()) for k in ["train", "val", "test"]},
        "radial": {"profile_n": PROFILE_N, "ring_n": RING_N, "radii_mm": list(RADII_MM), "hu_clip": [-300, 1200]},
        "coordinate_conversion": "ImageCAS-X VTK LPS -> ImageCAS NIfTI RAS by negating x,y before inverse affine",
        "raw_ct_in_output": False,
        "threshold_selection": "validation only; maximize recall subject to FPR <= 0.05",
        "test_threshold_tuning": False,
    }
    (args.out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    evaluate_models(features, args.out_dir)

    zip_path = args.out_dir.parent / "GraphLIRA_CT_expanded_radial_results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(args.out_dir.iterdir()):
            if p.is_file():
                z.write(p, arcname=p.name)
        z.write(args.plan, arcname="frozen_relation_pair_plan.csv")
        z.write(args.expected_geometry, arcname="frozen_expected_ct_geometry.csv")
    print("\nDONE")
    print(zip_path)
    print(f"bundle size: {zip_path.stat().st_size/1024**2:.2f} MiB")
    print("Upload only this ZIP; it contains no raw CT volumes.")


if __name__ == "__main__":
    main()
