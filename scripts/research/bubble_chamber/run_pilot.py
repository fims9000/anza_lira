#!/usr/bin/env python3
"""Run a reproducible classical pilot on a real BMP/JPEG or synthetic scenes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2

# Permit launching from the repository root without editable installation.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bubble_chamber.pilot import PilotConfig, analyze_frame
from bubble_chamber.synthetic import centerline_scores, generate_case


def write_case(image, output: Path, cfg: PilotConfig, roi, *, ground_truth=None, provenance=None):
    result, mask, overlay = analyze_frame(image, cfg, roi_xyxy=roi)
    if provenance is not None:
        result["provenance"] = provenance
    if ground_truth is not None:
        result["synthetic_centerline_metrics"] = centerline_scores(mask, ground_truth)
        result["no_ground_truth"] = False
        cv2.imwrite(str(output / "synthetic_ground_truth.png"), ground_truth.astype("uint8") * 255)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "foreground_mask.png"), mask)
    cv2.imwrite(str(output / "fragment_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "measurement_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fragment_id", "index", "x_source_px", "y_source_px", "physical_direction_known"])
        for item in result["segments"]:
            for idx, (x, y) in enumerate(item["measurement_xy_px"]):
                writer.writerow([item["segment_id"], idx, f"{x:.5f}", f"{y:.5f}", "false"])
    if any("transverse_profile_samples" in segment for segment in result["segments"]):
        with (output / "transverse_photometry.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["fragment_id", "point_index", "x_source_px", "y_source_px", "status", "width_fwhm_px", "darkness_px", "calibrated_dedx"])
            for segment in result["segments"]:
                for point in segment.get("transverse_profile_samples", []):
                    writer.writerow([segment["segment_id"], point["index"], f'{point["x_source_px"]:.5f}', f'{point["y_source_px"]:.5f}', point["status"],
                                     "" if point["width_fwhm_px"] is None else point["width_fwhm_px"],
                                     "" if point["darkness_px"] is None else point["darkness_px"], "false"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="Real experimental image. Not auto-downloaded.")
    parser.add_argument("--synthetic", choices=["continuous", "gap", "crossing", "parallel", "weak_noisy", "all"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--threshold", type=float, default=18.0)
    parser.add_argument("--segmentation", choices=("contrast", "frangi"), default="contrast")
    parser.add_argument("--ridge-fraction", type=float, default=0.12)
    parser.add_argument("--min-length", type=float, default=28.0)
    args = parser.parse_args()
    if bool(args.image) == bool(args.synthetic):
        parser.error("provide exactly one of --image or --synthetic")
    cfg = PilotConfig(threshold=args.threshold, min_segment_length_px=args.min_length,
                      segmentation_method=args.segmentation, ridge_response_fraction=args.ridge_fraction)
    if args.image:
        image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
        if image is None:
            parser.error("cannot read image")
        result = write_case(image, args.out_dir, cfg, tuple(args.roi) if args.roi else None,
                            provenance={"kind": "provided_real_photo", "path_name": args.image.name, "label_status": "unannotated"})
        print(json.dumps({"fragments": result["fragment_count"], "review_joins": result["proposed_joins_count"], "out": str(args.out_dir)}))
    else:
        cases = ("continuous", "gap", "crossing", "parallel", "weak_noisy") if args.synthetic == "all" else (args.synthetic,)
        records = {}
        for case in cases:
            image, truth, meta = generate_case(case)
            dest = args.out_dir / case
            dest.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest / "synthetic_observed.png"), image)
            result = write_case(image, dest, cfg, None, ground_truth=truth,
                                provenance={"kind": "synthetic", "params": meta})
            records[case] = {"fragments": result["fragment_count"], "proposals": result["proposed_joins_count"], "metrics": result["synthetic_centerline_metrics"]}
        (args.out_dir / "summary.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
