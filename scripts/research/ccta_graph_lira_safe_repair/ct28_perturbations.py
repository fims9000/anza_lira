from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


class PerturbationError(RuntimeError):
    pass


def stable_seed(seed: int, scene_id: str, repetition: int) -> int:
    payload = f"{seed}|{scene_id}|{repetition}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def perturb_direction(
    vector: np.ndarray, sigma_degrees: float, generator: np.random.Generator
) -> np.ndarray:
    direction = np.array(vector, dtype=float, copy=True)
    direction /= np.linalg.norm(direction) + 1e-12
    normal = generator.normal(size=3)
    normal -= direction * float(normal @ direction)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return direction.copy()
    normal /= norm
    angle = np.deg2rad(sigma_degrees) * generator.normal()
    perturbed = direction * np.cos(angle) + normal * np.sin(angle)
    return perturbed / (np.linalg.norm(perturbed) + 1e-12)


def perturb_scene_endpoints(
    endpoints: pd.DataFrame,
    *,
    seed: int,
    repetition: int,
    position_noise_mm: float,
    tangent_noise_deg: float,
) -> pd.DataFrame:
    scene_ids = set(endpoints["scene_id"])
    if len(scene_ids) != 1:
        raise PerturbationError("endpoint perturbation expects one scene")
    scene_id = str(next(iter(scene_ids)))
    generator = np.random.default_rng(stable_seed(seed, scene_id, repetition))
    perturbed = endpoints.copy()
    positions = perturbed[["x_lps", "y_lps", "z_lps"]].to_numpy(
        dtype=float, copy=True
    )
    positions += generator.normal(scale=position_noise_mm, size=positions.shape)
    tangents = perturbed[["tangent_x", "tangent_y", "tangent_z"]].to_numpy(
        dtype=float
    )
    tangents = np.stack(
        [
            perturb_direction(tangent, tangent_noise_deg, generator)
            for tangent in tangents
        ]
    )
    perturbed.loc[:, ["x_lps", "y_lps", "z_lps"]] = positions
    perturbed.loc[:, ["tangent_x", "tangent_y", "tangent_z"]] = tangents
    return perturbed
