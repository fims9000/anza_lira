from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class PairInfo:
    sat_path: Path
    gt_path: Path
    fg_ratio: float


def _collect_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for sat in sorted(split_dir.rglob("*_sat.png")):
        gt = sat.with_name(sat.name.replace("_sat.png", "_gt.png"))
        if gt.exists():
            pairs.append((sat, gt))
    return pairs


def _fg_ratio(mask_path: Path) -> float:
    arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return float((arr > 127).mean())


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def _select_subset(
    pairs: list[tuple[Path, Path]],
    min_fg: float,
    max_fg: float,
    limit: int,
    seed: int,
) -> list[PairInfo]:
    candidates: list[PairInfo] = []
    for sat, gt in pairs:
        ratio = _fg_ratio(gt)
        if min_fg <= ratio <= max_fg:
            candidates.append(PairInfo(sat_path=sat, gt_path=gt, fg_ratio=ratio))

    if len(candidates) <= limit:
        return sorted(candidates, key=lambda x: str(x.sat_path))

    rng = random.Random(seed)
    chosen = rng.sample(candidates, k=limit)
    return sorted(chosen, key=lambda x: str(x.sat_path))


def _summarize(rows: list[PairInfo]) -> dict[str, float]:
    if not rows:
        return {"count": 0, "fg_mean": 0.0, "fg_p10": 0.0, "fg_p50": 0.0, "fg_p90": 0.0}
    vals = np.array([r.fg_ratio for r in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "fg_mean": float(vals.mean()),
        "fg_p10": float(np.percentile(vals, 10)),
        "fg_p50": float(np.percentile(vals, 50)),
        "fg_p90": float(np.percentile(vals, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curated subset from GlobalScaleRoad by foreground road ratio.")
    parser.add_argument("--src-root", type=str, default="data/GlobalScaleRoad")
    parser.add_argument("--dst-root", type=str, default="data_subsets/GlobalScaleRoad")
    parser.add_argument("--min-fg", type=float, default=0.015)
    parser.add_argument("--max-fg", type=float, default=0.08)
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--val-limit", type=int, default=240)
    parser.add_argument("--test-limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    if not src_root.exists():
        raise SystemExit(f"Source root not found: {src_root}")

    if args.force and dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    split_plan = [
        ("train", int(args.train_limit), int(args.seed) + 0),
        ("val", int(args.val_limit), int(args.seed) + 1),
        ("in-domain-test", int(args.test_limit), int(args.seed) + 2),
    ]

    manifest: dict[str, object] = {
        "source_root": str(src_root),
        "destination_root": str(dst_root),
        "criteria": {
            "min_fg": float(args.min_fg),
            "max_fg": float(args.max_fg),
            "train_limit": int(args.train_limit),
            "val_limit": int(args.val_limit),
            "test_limit": int(args.test_limit),
            "seed": int(args.seed),
        },
        "splits": {},
    }

    for split, limit, seed in split_plan:
        src_split = src_root / split
        dst_split = dst_root / split
        pairs = _collect_pairs(src_split)
        chosen = _select_subset(
            pairs=pairs,
            min_fg=float(args.min_fg),
            max_fg=float(args.max_fg),
            limit=limit,
            seed=seed,
        )

        files_meta: list[dict[str, object]] = []
        for idx, row in enumerate(chosen):
            prefix = f"{idx:05d}"
            sat_dst = dst_split / f"{prefix}_sat.png"
            gt_dst = dst_split / f"{prefix}_gt.png"
            _safe_link_or_copy(row.sat_path, sat_dst)
            _safe_link_or_copy(row.gt_path, gt_dst)
            files_meta.append(
                {
                    "index": idx,
                    "sat_source": str(row.sat_path),
                    "gt_source": str(row.gt_path),
                    "sat_target": str(sat_dst),
                    "gt_target": str(gt_dst),
                    "fg_ratio": float(row.fg_ratio),
                }
            )

        manifest["splits"][split] = {
            "source_pair_count": int(len(pairs)),
            "selected_count": int(len(chosen)),
            "summary": _summarize(chosen),
            "files": files_meta,
        }
        print(
            f"{split}: selected {len(chosen)}/{len(pairs)} "
            f"(fg mean={_summarize(chosen)['fg_mean']:.4f})",
            flush=True,
        )

    manifest_path = dst_root / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md_lines = [
        "# GlobalScaleRoad Curated Subset",
        "",
        f"- Source: `{src_root}`",
        f"- Destination: `{dst_root}`",
        f"- Foreground ratio filter: `{float(args.min_fg):.4f} <= fg <= {float(args.max_fg):.4f}`",
        "",
        "| Split | Source pairs | Selected | fg mean | fg p10 | fg p50 | fg p90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "in-domain-test"):
        row = manifest["splits"][split]  # type: ignore[index]
        summary = row["summary"]  # type: ignore[index]
        md_lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    str(int(row["source_pair_count"])),  # type: ignore[index]
                    str(int(row["selected_count"])),  # type: ignore[index]
                    f"{float(summary['fg_mean']):.4f}",  # type: ignore[index]
                    f"{float(summary['fg_p10']):.4f}",  # type: ignore[index]
                    f"{float(summary['fg_p50']):.4f}",  # type: ignore[index]
                    f"{float(summary['fg_p90']):.4f}",  # type: ignore[index]
                ]
            )
            + " |"
        )
    md_lines.append("")
    md_lines.append(f"- Manifest: `{manifest_path}`")
    (dst_root / "subset_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved manifest: {manifest_path}", flush=True)
    print(f"Saved summary: {dst_root / 'subset_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
