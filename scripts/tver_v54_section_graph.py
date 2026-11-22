"""Export whole-section predicted trace graphs and unaccepted link hypotheses.

Input: any frozen segmenter's 2-D probability array (NumPy .npy).
The optional expert mask is used AFTER inference for annotated-line coverage
only, never to construct ports or connect faults.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from trace_extraction.section_candidates import full_section_candidate_graph
from trace_extraction.skeleton import skeletonize_mask
from trace_extraction.coverage import mode_layers


def export_section_graph(
    probability_path, output_dir, *,
    expert_mask_path=None, thresholds=(.65, .50, .35),
    max_gap_px=None, min_alignment=None, max_neighbors_per_port=None,
    mode="coverage",
):
    pred = np.load(probability_path, allow_pickle=False, mmap_mode="r")
    if pred.ndim != 2:
        raise ValueError("Expected an entire 2-D probability section")
    presets={
        "coverage": dict(max_gap_px=60.,min_alignment=-.1,max_neighbors_per_port=32),
        "balanced": dict(max_gap_px=40.,min_alignment=.2,max_neighbors_per_port=16),
        "strict": dict(max_gap_px=25.,min_alignment=.6,max_neighbors_per_port=8),
    }
    if mode not in presets:
        raise ValueError("Unknown requested output mode")
    policy=presets[mode].copy()
    if max_gap_px is not None: policy["max_gap_px"]=max_gap_px
    if min_alignment is not None: policy["min_alignment"]=min_alignment
    if max_neighbors_per_port is not None:
        policy["max_neighbors_per_port"]=max_neighbors_per_port
    evidence=mode_layers(pred,mode=mode)
    # Save all weak evidence even if a dense mask is unsuitable for a graph.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output/"V54_high_confidence.npy",evidence.high_confidence)
    np.save(output/"V54_supported_weak.npy",evidence.supported_weak)
    np.save(output/"V54_review_weak.npy",evidence.review_weak)
    np.save(output/"V54_all_possible_trace.npy",evidence.possible_trace)
    layers=[]
    skipped=[]
    for threshold in thresholds:
        try:
            layers.append(full_section_candidate_graph(
                pred,threshold=float(threshold),**policy))
        except ValueError as exc:
            if "Dense segmentation" not in str(exc):
                raise
            skipped.append({
                "threshold":float(threshold),
                "reason":"GRAPH_SKIPPED_DENSE_BUT_ALL_TRACE_EVIDENCE_SAVED",
                "foreground_fraction":float(np.mean(pred>=float(threshold))),
            })
    # Correctness reference is loaded only after every inference layer exists.
    expert = None
    if expert_mask_path is not None:
        label = np.load(expert_mask_path, allow_pickle=False, mmap_mode="r")
        if label.shape != pred.shape:
            raise ValueError("Expert reference shape differs from prediction")
        expert = skeletonize_mask(np.asarray(label, bool))
    reports = []
    for layer in layers:
        token = str(layer.threshold).replace(".", "p")
        np.save(output / f"V54_skeleton_{token}.npy", layer.graph.skeleton,
                allow_pickle=False)
        with (output / f"V54_trace_ports_{token}.csv").open(
                "w", newline="", encoding="utf8") as file:
            writer = csv.writer(file)
            writer.writerow(["port_id", "trace_id", "side", "row", "col",
                             "outward_row", "outward_col"])
            for port in layer.ports:
                writer.writerow([port.port_id, port.trace_id, port.side,
                                 *port.pixel, *port.outward_yx])
        with (output / f"V54_candidate_links_{token}.csv").open(
                "w", newline="", encoding="utf8") as file:
            writer = csv.writer(file)
            writer.writerow(["first_port", "second_port", "first_trace",
                             "second_trace", "gap_px", "source_alignment",
                             "target_alignment", "geometric_cost"])
            for pair in layer.pairs:
                writer.writerow([pair.first_port, pair.second_port,
                                 pair.first_trace, pair.second_trace,
                                 pair.gap_px, pair.source_alignment,
                                 pair.target_alignment, pair.geometric_cost])
        annotated_coverage = None
        if expert is not None and expert.any():
            nearest = distance_transform_edt(~layer.graph.skeleton)
            annotated_coverage = float(np.mean(nearest[expert] <= 2.))
        reports.append({
            "threshold": layer.threshold,
            "foreground_fraction": layer.foreground_fraction,
            "n_trace_segments": len(layer.graph.segments),
            "n_trace_end_ports": len(layer.ports),
            "n_unaccepted_link_candidates": len(layer.pairs),
            "annotated_centerline_coverage_2px": annotated_coverage,
        })
    report = {
        "status": "V54_FULL_SECTION_PROPOSALS_NOT_IDENTITY_DECISIONS",
        "input_probability_shape": list(pred.shape),
        "no_expert_gap_rectangle_required": True,
        "expert_optional_for_post_inference_evaluation_only": True,
        "unlabeled_pixels_not_treated_as_confident_background": True,
        "separate_threshold_graphs_not_merged": True,
        "outputs_are_unaccepted_candidate_links": True,
        "no_real_fault_instance_identity_ground_truth_claimed": True,
        "mode":mode,
        "policy":policy,
        "mode_trace_counts":{
            "high_confidence_pixels":int(evidence.high_confidence.sum()),
            "supported_weak_pixels":int(evidence.supported_weak.sum()),
            "review_weak_pixels":int(evidence.review_weak.sum()),
            "all_possible_trace_pixels":int(evidence.possible_trace.sum()),
        },
        "skipped_dense_graph_layers":skipped,
        "layers": reports,
    }
    (output / "V54_SECTION_TRACE_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf8",
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probability", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expert-mask", type=Path)
    parser.add_argument("--mode",choices=("coverage","balanced","strict"),default="coverage")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.65, .5, .35])
    parser.add_argument("--max-gap-px", type=float, default=40.)
    args = parser.parse_args()
    result = export_section_graph(
        args.probability, args.out_dir,
        expert_mask_path=args.expert_mask,
        thresholds=args.thresholds, max_gap_px=args.max_gap_px,
        mode=args.mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
