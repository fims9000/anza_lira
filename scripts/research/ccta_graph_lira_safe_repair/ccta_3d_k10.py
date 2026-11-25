"""K10/15 inference adapter for unlabeled 3D vessel-mask scenes.

Uses the exact frozen CT28 local geometry/CT models, score summaries,
PAIR/JUNCTION identity policy, tau=0.85 and 15x perturbation consistency.
Unlike the original runner it accepts generated *unlabeled* scenes; this
distribution shift is experimental and is NOT a validated clinical result.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import joblib
import numpy as np
import pandas as pd

import build_ct28_scene_plan as scene_builder
import ct28_perturbations as perturbations
import extract_ct28_scene_candidate_features as ct_features
import run_ct28_graph_lira as graph
from ct28_safety_policy import should_accept
from ccta_3d_endpoints import Endpoint, Scene


K10_15_SHA256 = "1693b6e9b00fd74f2d443905351f534e6bdf745a5cdc00d0057823646aa09630"
IDENTITY = {
    "geometry_only": "geometry_score",
    "ct_conditioned": {
        "PAIR": "geometry_plus_ct_score",
        "JUNCTION": "geometry_score",
    },
}
EVAL_POSITION_NOISE_MM = 0.8
EVAL_TANGENT_NOISE_DEG = 20.0
EVAL_PERTURBATIONS = 15
EVAL_SEED = 20260920
EVAL_TAU = 0.85
EVAL_CONSISTENCY = 0.60


@dataclass(frozen=True)
class ScenePrediction:
    scene_id: str
    raw_relation: str
    proposed_relation: str
    selected_relation: str
    confidence: float
    consistency: float
    accepted: bool
    pair_candidate_id: str | None
    junction_candidate_id: str | None
    identity_log_utility: float
    candidate_count_pair: int
    candidate_count_junction: int
    relation_distribution: tuple[float, float, float, float]
    decision_keys_under_perturbation: tuple[str, ...]
    structural_abstention_reason: str | None = None


def verified_model_pack(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing local K10/15 model pack: {path}")
    hashvalue = hashlib.sha256(path.read_bytes()).hexdigest()
    if hashvalue != K10_15_SHA256:
        raise ValueError(
            "K10/15 SHA256 mismatch: model pack not selected freeze. "
            f"Expected {K10_15_SHA256}; found {hashvalue}"
        )
    # joblib is pickle-based: only read trusted local pack after hash check.
    return joblib.load(path), hashvalue


class K10Inference:
    def __init__(self, models: dict, *, pack_sha: str = "injected_for_test") -> None:
        self.models = models
        self.pack_sha = pack_sha
        if not {"local_models", "relation_heads", "ct_conditioned_relation_features"}.issubset(models):
            raise ValueError("K10/15 model pack missing required stages")
        if len(models["ct_conditioned_relation_features"]) != 84:
            raise ValueError("Expected CT-conditioned 84-feature head")
        for kind in ("PAIR", "JUNCTION"):
            if kind not in models["local_models"] or not set(
                ("geometry", "ct", "geometry_plus_ct")
            ).issubset(models["local_models"][kind]):
                raise ValueError(f"Missing local score branches for {kind}")
        for model in ("geometry_only", "ct_conditioned"):
            if model not in models["relation_heads"]:
                raise ValueError(f"Missing relation head {model}")
            labels = getattr(models["relation_heads"][model], "classes_", None)
            if labels is not None and list(labels) != [0, 1, 2, 3]:
                raise ValueError("Relation class order differs from frozen NONE/PAIR/JUNCTION/BOTH")
        self.arm_names = ct_features.arm_feature_names(ct_features.RADII_MM)

    @classmethod
    def from_file(cls, path: Path) -> "K10Inference":
        models, sha = verified_model_pack(path)
        return cls(models, pack_sha=sha)

    @staticmethod
    def _frames(
        scene: Scene, endpoints: list[Endpoint],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        lookup = {endpoint.endpoint_id: endpoint for endpoint in endpoints}
        if any(endpoint_id not in lookup for endpoint_id in scene.endpoint_ids):
            raise ValueError("Scene references unknown endpoint")
        data = []
        for endpoint_id in scene.endpoint_ids:
            endpoint = lookup[endpoint_id]
            data.append({
                "scene_id": scene.scene_id,
                "endpoint_id": endpoint.endpoint_id,
                "component": str(endpoint.component),
                **dict(zip(("x_lps", "y_lps", "z_lps"), endpoint.position_lps)),
                **dict(zip(("tangent_x", "tangent_y", "tangent_z"), endpoint.tangent_lps)),
            })
        plan = pd.DataFrame([{
            "scene_id": scene.scene_id, "scan_id": -1,
            "split": "unlabelled_runtime", "scene_family": "unlabelled_runtime",
            "relation_class": "UNLABELLED",
        }])
        candidates = pd.DataFrame([
            {
                "scene_id": scene.scene_id,
                "candidate_id": candidate.candidate_id,
                "candidate_type": candidate.kind,
                "endpoint_ids": "|".join(candidate.endpoint_ids),
            }
            for candidate in scene.candidates
        ])
        return plan, pd.DataFrame(data), candidates

    @staticmethod
    def _score_sparse_candidate_types(
        features: pd.DataFrame, local_models: dict,
    ) -> pd.DataFrame:
        """Score only candidate kinds present; never fabricate dummy candidates.

        The historical CT28 runner always has both candidate types in its
        controlled scenes and therefore calls each branch on every scene.
        A real local five-endpoint window can contain PAIR without JUNCTION.
        """
        if features.empty:
            raise ValueError("Cannot score empty feature table")
        output = features[["scene_id", "candidate_id", "candidate_type"]].copy()
        for stream in ("geometry", "ct", "geometry_plus_ct"):
            output[stream + "_score"] = np.nan
        for kind in ("PAIR", "JUNCTION"):
            mask = features.candidate_type.eq(kind)
            if not mask.any():
                continue
            for stream in ("geometry", "ct", "geometry_plus_ct"):
                model = local_models[kind][stream]
                columns = list(model.feature_names_in_)
                if not set(columns).issubset(features.columns):
                    raise ValueError(f"Missing {kind}/{stream} model input features")
                scores = model.predict_proba(features.loc[mask, columns])[:, 1]
                output.loc[mask, stream + "_score"] = scores
        if not np.isfinite(
            output[["geometry_score", "ct_score", "geometry_plus_ct_score"]].to_numpy(dtype=float)
        ).all():
            raise ValueError("Incomplete local score streams")
        return output

    def _score(
        self, scene: Scene, endpoints: pd.DataFrame, candidates: pd.DataFrame,
        ct: np.ndarray, inverse_affine: np.ndarray,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
        lookup = endpoints.set_index("endpoint_id")
        features = pd.DataFrame([
            graph.perturbed_candidate_features(
                candidate, lookup, ct, inverse_affine, self.arm_names
            )
            for _, candidate in candidates.iterrows()
        ])
        scores = self._score_sparse_candidate_types(
            features, self.models["local_models"]
        )
        plan = pd.DataFrame([{
            "scene_id": scene.scene_id, "scan_id": -1,
            "split": "unlabelled_runtime", "scene_family": "unlabelled_runtime",
            "relation_class": "UNLABELLED",
        }])
        predictions = graph.predict_relations(plan, candidates, scores, self.models, EVAL_TAU)
        decisions = graph.select_scene_decisions(plan, candidates, scores, predictions, IDENTITY)
        return scores, {key: val[scene.scene_id] for key, val in decisions.items()}

    def predict(
        self, scene: Scene, endpoints: list[Endpoint], ct: np.ndarray,
        affine: np.ndarray,
    ) -> ScenePrediction:
        if not scene.candidates:
            raise ValueError("Cannot score empty candidate pool")
        if ct.ndim != 3 or not np.isfinite(affine).all() or affine.shape != (4, 4):
            raise ValueError("Invalid CT/affine inputs")
        _, endpoint_frame, candidate_frame = self._frames(scene, endpoints)
        inverse = np.linalg.inv(affine)
        base_scores, decisions = self._score(
            scene, endpoint_frame, candidate_frame, ct, inverse
        )
        base = decisions["ct_conditioned"]
        repeats = []
        for repetition in range(EVAL_PERTURBATIONS):
            perturbed = perturbations.perturb_scene_endpoints(
                endpoint_frame, seed=EVAL_SEED, repetition=repetition,
                position_noise_mm=EVAL_POSITION_NOISE_MM,
                tangent_noise_deg=EVAL_TANGENT_NOISE_DEG,
            )
            _, decision = self._score(scene, perturbed, candidate_frame, ct, inverse)
            repeats.append(str(decision["ct_conditioned"]["relation_key"]))
        consistency = graph.consistency_value(str(base["relation_key"]), repeats)
        selected = str(base["selected_relation"])
        proposed = str(base["relation"])
        accepted = should_accept(
            proposed_relation=proposed,
            selected_relation=selected,
            compatibility_satisfied=bool(base["compatibility_satisfied"]),
            consistency=float(consistency),
            threshold=EVAL_CONSISTENCY,
        )
        abstention_reason = None
        if not accepted:
            if base.get("endpoint_patch_fov_abstain"):
                abstention_reason = "endpoint_outside_neural_patch"
            elif float(base["confidence"]) < EVAL_TAU:
                abstention_reason = "relation_confidence_below_tau"
            elif proposed == "NONE":
                abstention_reason = "predicted_none"
            elif selected != proposed or not base["compatibility_satisfied"]:
                abstention_reason = "incompatible_or_missing_candidate_set"
            else:
                abstention_reason = "consistency_below_threshold"
        score_lookup = base_scores.set_index("candidate_id")
        utility = 0.0
        # For frozen K10/15 this is EXACTLY the historical PAIR=fusion and
        # JUNCTION=geometry utility. The experimental neural subclass may
        # explicitly opt into ANN JUNCTION identity, in which case the
        # reported utility MUST use the same stream as candidate selection.
        identity_fields = getattr(self, "identity_utility_columns", {
            "PAIR": "geometry_plus_ct_score",
            "JUNCTION": "geometry_score",
        })
        for kind, key in (("PAIR", base["pair_candidate_id"]),
                          ("JUNCTION", base["junction_candidate_id"])):
            if key is not None:
                utility += float(np.log(max(
                    1e-8,score_lookup.loc[key,identity_fields[kind]]
                )))
        return ScenePrediction(
            scene_id=scene.scene_id,
            raw_relation=str(base["raw_relation"]),
            proposed_relation=proposed,
            selected_relation=selected,
            confidence=float(base["confidence"]),
            consistency=float(consistency),
            accepted=bool(accepted),
            pair_candidate_id=(str(base["pair_candidate_id"]) if accepted and
                               base["pair_candidate_id"] is not None else None),
            junction_candidate_id=(str(base["junction_candidate_id"]) if accepted and
                                   base["junction_candidate_id"] is not None else None),
            identity_log_utility=utility,
            candidate_count_pair=int((candidate_frame.candidate_type == "PAIR").sum()),
            candidate_count_junction=int((candidate_frame.candidate_type == "JUNCTION").sum()),
            relation_distribution=tuple(float(v) for v in base["probabilities"]),
            decision_keys_under_perturbation=tuple(repeats),
            structural_abstention_reason=abstention_reason,
        )
