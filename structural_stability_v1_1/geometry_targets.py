"""Train-only crowd-derived tangent/coherence targets for B2 and B3."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from lira_final.protocol import TRAIN_ANNOTATORS
from structural_stability_v1.agreement import crowd_agreement
from structural_stability_v1_1.protocol import PROTOCOL, ROOT, canonical_hash, protocol_hash


def geometry_target(masks_rgb: list[np.ndarray]) -> dict[str, np.ndarray]:
    crowd = crowd_agreement(masks_rgb)
    positive = crowd["agreement"] * crowd["crowd_probability"]
    smooth = gaussian_filter(positive.astype(np.float64), sigma=1.5, mode="reflect")
    grad_y, grad_x = np.gradient(smooth)
    t_xx = gaussian_filter(grad_x * grad_x, sigma=2.0, mode="reflect")
    t_xy = gaussian_filter(grad_x * grad_y, sigma=2.0, mode="reflect")
    t_yy = gaussian_filter(grad_y * grad_y, sigma=2.0, mode="reflect")
    delta = np.sqrt(np.maximum(np.square(t_xx - t_yy) + 4.0 * np.square(t_xy), 0.0))
    nu1 = 0.5 * (t_xx + t_yy + delta)
    nu2 = np.maximum(0.5 * (t_xx + t_yy - delta), 0.0)
    coherence = np.clip((nu1 - nu2) / (nu1 + nu2 + 1e-8), 0.0, 1.0)
    # The major tensor eigenvector is the ridge normal. Adding pi/2 negates its
    # doubled-angle representation and yields the axial fault tangent.
    normal_c2 = (t_xx - t_yy) / (delta + 1e-8)
    normal_s2 = (2.0 * t_xy) / (delta + 1e-8)
    tangent_c2 = -normal_c2
    tangent_s2 = -normal_s2
    magnitude = np.sqrt(np.square(tangent_c2) + np.square(tangent_s2))
    tangent_c2 = np.divide(tangent_c2, magnitude, out=np.ones_like(tangent_c2), where=magnitude > 1e-6)
    tangent_s2 = np.divide(tangent_s2, magnitude, out=np.zeros_like(tangent_s2), where=magnitude > 1e-6)
    supervision = (
        (crowd["crowd_probability"] >= 0.75)
        & (crowd["agreement"] >= 0.25)
        & (coherence >= 0.20)
    )
    strength = np.where(supervision, 0.35 * coherence, 0.0)
    geometry_weight = np.where(supervision, crowd["agreement"] * coherence, 0.0)
    return {
        **crowd,
        "positive_field": positive.astype(np.float32),
        "smoothed_positive": smooth.astype(np.float32),
        "coherence": coherence.astype(np.float32),
        "target_c2": tangent_c2.astype(np.float32),
        "target_s2": tangent_s2.astype(np.float32),
        "target_d": strength.astype(np.float32),
        "geometry_weight": geometry_weight.astype(np.float32),
        "supervision": supervision,
    }


def _load_train_masks(section_id: int) -> tuple[list[str], list[np.ndarray]]:
    names, masks = [], []
    for annotator in TRAIN_ANNOTATORS:
        path = ROOT / "data/cracks/annotations" / annotator / f"section_{section_id:03d}.png"
        if path.is_file():
            with Image.open(path) as handle:
                masks.append(np.asarray(handle.convert("RGB"), dtype=np.uint8))
            names.append(annotator)
    if not masks:
        raise FileNotFoundError(f"no train nonexpert annotations for section {section_id}")
    return names, masks


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if not values.size:
        return {key: 0.0 for key in ("q05", "q25", "q50", "q75", "q95")}
    result = np.quantile(values, (0.05, 0.25, 0.50, 0.75, 0.95))
    return {key: float(value) for key, value in zip(("q05", "q25", "q50", "q75", "q95"), result)}


def _save_example(section_id: int, target: dict[str, np.ndarray], output: Path) -> None:
    field = target["positive_field"]
    mask = target["supervision"]
    yy, xx = np.mgrid[: field.shape[0], : field.shape[1]]
    stride = 12
    selected = mask[::stride, ::stride]
    theta = 0.5 * np.arctan2(target["target_s2"], target["target_c2"])
    u = np.cos(theta)[::stride, ::stride]
    v = np.sin(theta)[::stride, ::stride]
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    axis.imshow(field, cmap="magma", aspect="auto")
    axis.quiver(xx[::stride, ::stride][selected], yy[::stride, ::stride][selected], u[selected], v[selected], color="cyan", pivot="middle", scale=35)
    axis.set_title(f"Train-only crowd tangent target, section {section_id}")
    axis.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def audit_geometry_targets(section_ids: Sequence[int], output: Path) -> dict[str, Any]:
    ordered = [int(value) for value in section_ids]
    rows: list[dict[str, Any]] = []
    all_kappa: list[np.ndarray] = []
    all_strength: list[np.ndarray] = []
    supervised_pixels = 0
    total_pixels = 0
    for index, section_id in enumerate(ordered):
        names, masks = _load_train_masks(section_id)
        target = geometry_target(masks)
        selected = target["supervision"]
        kappa = target["coherence"][selected]
        strength = target["target_d"][selected]
        supervised_pixels += int(selected.sum())
        total_pixels += int(selected.size)
        all_kappa.append(kappa)
        all_strength.append(strength)
        rows.append({
            "section_id": section_id,
            "annotator_count": len(names),
            "supervised_pixels": int(selected.sum()),
            "supervised_fraction": float(selected.mean()),
            **{f"kappa_{key}": value for key, value in _quantiles(kappa).items()},
            **{f"d_{key}": value for key, value in _quantiles(strength).items()},
        })
        if index < 3:
            _save_example(section_id, target, output / "figures" / f"tangent_target_section_{section_id:03d}.png")
        if (index + 1) % 25 == 0 or index + 1 == len(ordered):
            print(f"phase=SS1.5_GEOMETRY section={index + 1}/{len(ordered)} expert=LOCKED", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "GEOMETRY_TARGET_AUDIT.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    combined_kappa = np.concatenate(all_kappa) if all_kappa else np.asarray([])
    combined_strength = np.concatenate(all_strength) if all_strength else np.asarray([])
    summary = {
        "status": "GEOMETRY_TARGET_AUDIT_PASS",
        "protocol_sha256": protocol_hash(),
        "train_section_ids_sha256": canonical_hash(ordered),
        "train_sections": len(ordered),
        "sections_with_supervision": sum(row["supervised_pixels"] > 0 for row in rows),
        "supervised_pixels": supervised_pixels,
        "total_train_pixels": total_pixels,
        "supervised_fraction": float(supervised_pixels / total_pixels),
        "kappa": _quantiles(combined_kappa),
        "target_d": _quantiles(combined_strength),
        "expert_data_accessed": False,
        "calibration_development_confirm_labels_accessed": False,
        "training_annotators": list(TRAIN_ANNOTATORS),
    }
    (output / "GEOMETRY_TARGET_PROTOCOL.json").write_text(json.dumps(PROTOCOL["geometry_target"], indent=2, sort_keys=True) + "\n")
    (output / "GEOMETRY_TARGET_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "GEOMETRY_TARGET_AUDIT.md").write_text(
        "# V1.1 train-only geometry target audit\n\n"
        f"- SS_TRAIN sections: `{len(ordered)}`; supervised sections: `{summary['sections_with_supervision']}`.\n"
        f"- Geometry pixels: `{supervised_pixels}` / `{total_pixels}` (`{summary['supervised_fraction']:.6f}`).\n"
        f"- Kappa quantiles: `{summary['kappa']}`.\n- d* quantiles: `{summary['target_d']}`.\n"
        "- Only frozen TRAIN_ANNOTATORS nonexpert masks were decoded. Expert and calibration/development/confirm labels were not accessed.\n"
        "- The same target is prescribed for B2 and B3 and is not evidence of Anosov value.\n\n"
        "GEOMETRY_TARGET_AUDIT_PASS\n"
    )
    return summary
