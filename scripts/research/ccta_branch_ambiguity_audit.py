"""Controlled branch-ambiguity audit for ImageCAS-X coronary centerlines.

The audit creates short positive continuation gaps along each labelled centerline
and counts geometrically plausible endpoints from a different anatomical segment.
It is a geometry stress diagnostic, not a clinical error-rate estimate.
"""
from pathlib import Path
import collections
import csv
import json

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def load_centerlines(case: str, root: Path):
    records = []
    for side in ("left", "right"):
        path = root / f"{case}.coronary_{side}_centerline.vtk"
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        poly = reader.GetOutput()
        points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
        pdata = poly.GetPointData()
        labels = vtk_to_numpy(pdata.GetArray("segment_label")).astype(int)
        names_array = pdata.GetAbstractArray("segment_name")
        names = np.array(
            [names_array.GetValue(i) for i in range(poly.GetNumberOfPoints())],
            dtype=object,
        )

        lines = poly.GetLines()
        lines.InitTraversal()
        ids = vtk.vtkIdList()
        line_index = 0
        while lines.GetNextCell(ids):
            idx = np.array([ids.GetId(j) for j in range(ids.GetNumberOfIds())], dtype=int)
            xyz = points[idx]
            tangents = np.zeros_like(xyz)
            for k in range(len(idx)):
                k0 = max(0, k - 2)
                k1 = min(len(idx) - 1, k + 2)
                v = xyz[k1] - xyz[k0]
                tangents[k] = v / (np.linalg.norm(v) + 1e-12)
            for k, pid in enumerate(idx):
                records.append(
                    dict(
                        side=side[0].upper(),
                        line=line_index,
                        k=k,
                        pid=int(pid),
                        xyz=xyz[k],
                        tangent=tangents[k],
                        label=int(labels[pid]),
                        name=str(names[pid]),
                    )
                )
            line_index += 1
    return records


def audit_case(case: str, root: Path, outdir: Path):
    records = load_centerlines(case, root)
    lines = collections.defaultdict(list)
    for row in records:
        lines[(row["side"], row["line"])].append(row)

    positives = []
    for _, line in lines.items():
        xyz = np.array([r["xyz"] for r in line])
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        for i in range(0, len(line), 4):
            for target_mm in (3.0, 4.0, 5.0):
                j = int(np.argmin(np.abs(arc - (arc[i] + target_mm))))
                if j <= i + 1 or j >= len(line):
                    continue
                if line[i]["label"] != line[j]["label"]:
                    continue
                distance = float(np.linalg.norm(xyz[j] - xyz[i]))
                if 2.0 <= distance <= 6.5:
                    positives.append((line[i], line[j], distance))

    unique = {}
    for a, b, distance in positives:
        unique[(a["side"], a["line"], a["pid"], b["pid"])] = (a, b, distance)
    positives = list(unique.values())

    wrong_pair_counts = collections.Counter()
    rows = []
    for a, _, positive_distance in positives:
        hard_count = 0
        for q in records:
            if q["label"] == a["label"]:
                continue
            v = q["xyz"] - a["xyz"]
            distance = float(np.linalg.norm(v))
            if not 2.0 <= distance <= 6.5:
                continue
            u = v / (distance + 1e-12)
            align_a = abs(float(np.dot(a["tangent"], u)))
            align_b = abs(float(np.dot(q["tangent"], u)))
            collinear = abs(float(np.dot(a["tangent"], q["tangent"])))
            if align_a >= 0.65 and align_b >= 0.65 and collinear >= 0.60:
                hard_count += 1
                wrong_pair_counts[(a["name"], q["name"])] += 1
        rows.append(
            dict(
                case=case,
                segment=a["name"],
                hard_decoys=hard_count,
                has_hard_decoy=int(hard_count > 0),
                gap_distance_mm=positive_distance,
            )
        )

    by_segment = []
    for segment in sorted(set(r["segment"] for r in rows)):
        values = [r for r in rows if r["segment"] == segment]
        hard = np.array([r["hard_decoys"] for r in values])
        by_segment.append(
            dict(
                case=case,
                segment=segment,
                n_positive_gaps=len(values),
                fraction_with_hard_decoy=float(np.mean(hard > 0)),
                mean_hard_decoys=float(np.mean(hard)),
                median_hard_decoys=float(np.median(hard)),
                max_hard_decoys=int(hard.max()),
            )
        )

    summary = dict(
        case=case,
        n_centerline_records=len(records),
        n_controlled_positive_gaps=len(positives),
        fraction_with_geometrically_plausible_wrong_branch=float(
            np.mean([r["has_hard_decoy"] for r in rows])
        ),
        median_hard_decoys=float(np.median([r["hard_decoys"] for r in rows])),
        top_wrong_branch_pairs=[
            dict(source=a, decoy=b, count=int(count))
            for (a, b), count in wrong_pair_counts.most_common(20)
        ],
    )

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{case}_gap_rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(outdir / f"{case}_by_segment.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_segment[0]))
        writer.writeheader()
        writer.writerows(by_segment)
    (outdir / f"{case}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    root = Path("/mnt/data")
    outdir = root / "ccta_branch_ambiguity_audit"
    summaries = [audit_case(case, root, outdir) for case in ("921", "953")]
    (outdir / "cross_patient_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))
