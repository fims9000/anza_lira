"""Numerically bounded, label-preserving perturbation implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates, zoom
from scipy.signal import hilbert

from datasets.cracks import BLUE, GREEN, ORANGE, WHITE
from structural_stability_v1.perturb.seeds import perturbation_seed
from structural_stability_v1.protocol import FAMILIES, PROTOCOL, SEVERITIES


@dataclass(frozen=True)
class PerturbationResult:
    image: np.ndarray
    family: str
    severity: int
    seed: int
    metadata: dict[str, float | int | str]
    displacement_yx: np.ndarray | None = None


def _direction(rng: np.random.Generator) -> int:
    return -1 if int(rng.integers(0, 2)) == 0 else 1


def _resize_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    factors = (1.0, shape[0] / field.shape[1], shape[1] / field.shape[2])
    resized = zoom(field, factors, order=3, mode="reflect", prefilter=True)
    return resized[:, : shape[0], : shape[1]].astype(np.float32)


def warp_jacobian(displacement_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(displacement_yx, dtype=np.float64)
    dy_dy, dy_dx = np.gradient(field[0])
    dx_dy, dx_dx = np.gradient(field[1])
    a = 1.0 + dy_dy
    b = dy_dx
    c = dx_dy
    d = 1.0 + dx_dx
    determinant = a * d - b * c
    trace_jtj = a * a + b * b + c * c + d * d
    det_jtj = determinant * determinant
    discriminant = np.maximum(trace_jtj * trace_jtj - 4.0 * det_jtj, 0.0)
    sigma_max2 = 0.5 * (trace_jtj + np.sqrt(discriminant))
    sigma_min2 = np.maximum(0.5 * (trace_jtj - np.sqrt(discriminant)), 1e-12)
    condition = np.sqrt(sigma_max2 / sigma_min2)
    return determinant.astype(np.float32), condition.astype(np.float32)


def _smooth_warp(seed: int, shape: tuple[int, int], maximum: float) -> tuple[np.ndarray, dict[str, float | int | str]]:
    limits = PROTOCOL["warp_validity"]
    for attempt in range(int(limits["maximum_attempts"])):
        rng = np.random.default_rng((seed + attempt * 104729) % (2**64))
        coarse_shape = (2, max(4, int(np.ceil(shape[0] / 48))), max(4, int(np.ceil(shape[1] / 48))))
        coarse = rng.normal(size=coarse_shape).astype(np.float32)
        field = _resize_field(coarse, shape)
        magnitude = np.sqrt(np.sum(field * field, axis=0))
        field *= float(maximum) / max(float(magnitude.max()), 1e-8)
        determinant, condition = warp_jacobian(field)
        if float(determinant.min()) >= float(limits["det_min"]) and float(determinant.max()) <= float(limits["det_max"]) and float(condition.max()) <= float(limits["condition_max"]):
            return field, {
                "maximum_displacement_px": float(np.sqrt(np.sum(field * field, axis=0)).max()),
                "jacobian_det_min": float(determinant.min()), "jacobian_det_max": float(determinant.max()),
                "jacobian_condition_max": float(condition.max()), "warp_attempt": attempt,
            }
    raise RuntimeError("unable to generate a valid smooth warp within the frozen attempt budget")


def _warp_array(array: np.ndarray, displacement_yx: np.ndarray, order: int) -> np.ndarray:
    values = np.asarray(array)
    height, width = values.shape[-2:]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    coordinates = (yy + displacement_yx[0, :height, :width], xx + displacement_yx[1, :height, :width])
    if values.ndim == 2:
        return map_coordinates(values, coordinates, order=order, mode="reflect", prefilter=order > 1)
    return np.stack([map_coordinates(channel, coordinates, order=order, mode="reflect", prefilter=order > 1) for channel in values], axis=0)


def apply_perturbation(image_chw: np.ndarray, section_id: int, crop_id: str, family: str, severity: int, view_index: int = 0) -> PerturbationResult:
    if family not in FAMILIES or severity not in SEVERITIES:
        raise ValueError("unknown perturbation")
    image = np.asarray(image_chw, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError("normalized image must be CHW")
    seed = perturbation_seed(section_id, crop_id, family, severity, view_index)
    rng = np.random.default_rng(seed)
    metadata: dict[str, float | int | str] = {"coordinate_transform": "identity"}
    displacement = None
    if family == "gain":
        low, high = PROTOCOL["severity_values"]["gain"][str(severity)]
        factor = float(low if _direction(rng) < 0 else high)
        output = image * factor
        metadata["gain"] = factor
    elif family == "noise":
        snr = float(PROTOCOL["severity_values"]["noise"][str(severity)])
        power = max(float(np.mean(image.astype(np.float64) ** 2)), 1e-12)
        sigma = float(np.sqrt(power / (10.0 ** (snr / 10.0))))
        output = image + rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
        metadata.update({"snr_db": snr, "noise_sigma": sigma})
    elif family == "bandlimit":
        sigma = float(PROTOCOL["severity_values"]["bandlimit"][str(severity)])
        output = gaussian_filter1d(image, sigma=sigma, axis=-2, mode="reflect").astype(np.float32)
        metadata.update({"sigma_px": sigma, "axis": "depth_y"})
    elif family == "phase":
        degrees = float(PROTOCOL["severity_values"]["phase"][str(severity)]) * _direction(rng)
        analytic = hilbert(image.astype(np.float64), axis=-2)
        output = np.real(analytic * np.exp(1j * np.deg2rad(degrees))).astype(np.float32)
        metadata.update({"phase_degrees": degrees, "axis": "depth_y"})
    else:
        maximum = float(PROTOCOL["severity_values"]["warp"][str(severity)])
        displacement, warp_meta = _smooth_warp(seed, image.shape[-2:], maximum)
        output = _warp_array(image, displacement, order=1).astype(np.float32)
        metadata = {"coordinate_transform": "smooth_output_to_input", **warp_meta}
    if not np.isfinite(output).all():
        raise ValueError("perturbation produced NaN/Inf")
    return PerturbationResult(output, family, severity, seed, metadata, displacement)


def transform_rgb_mask(mask_rgb: np.ndarray, result: PerturbationResult) -> np.ndarray:
    rgb = np.asarray(mask_rgb, dtype=np.uint8)
    if result.family != "warp":
        return rgb.copy()
    warped = _warp_array(rgb.transpose(2, 0, 1), result.displacement_yx, order=0).transpose(1, 2, 0).astype(np.uint8)
    allowed = np.asarray((BLUE, GREEN, ORANGE, WHITE), dtype=np.uint8)
    valid = np.any(np.all(warped[..., None, :] == allowed[None, None, :, :], axis=-1), axis=-1)
    if not valid.all():
        raise AssertionError("nearest-neighbor warp changed CRACKS palette semantics")
    return warped

