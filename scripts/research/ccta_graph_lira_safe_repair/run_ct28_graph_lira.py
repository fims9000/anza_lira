#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import build_ct28_relation_features as relation_features
import build_ct28_scene_plan as scene_builder
from ct28_perturbations import perturb_scene_endpoints
import extract_ct28_scene_candidate_features as ct_features
import train_ct28_relation_head as relation_training


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_DIR = REPO_ROOT / "artifacts/varvara/ct28_graph_lira"
DEFAULT_ARCHIVE_DIR = REPO_ROOT / "varvara/anza_lira_local_data/imagecas_801_1000"
RELATION_CLASSES = relation_training.RELATION_CLASSES
MODEL_CONFIGS = {
    "geometry_only": {
        "score_columns": ("geometry_score",),
        "relation_feature_key": "geometry_relation_features",
    },
    "ct_conditioned": {
        "score_columns": ("geometry_score", "ct_score"),
        "relation_feature_key": "ct_conditioned_relation_features",
    },
}
IDENTITY_SCORE_COLUMNS = ("geometry_score", "geometry_plus_ct_score")
EXPECTED_VALIDATION_PATIENTS = {957, 961, 965, 966, 974}
BOOTSTRAP_METRICS = (
    "repair_needed_exact",
    "false_structural_repair",
    "incomplete",
    "coverage",
    "false_among_accepted",
    "exact_among_accepted",
)


class GraphLiraError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen CT28 geometry-only and CT-conditioned Graph-LIRA "
            "through structural repair/abstain (path construction excluded)."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="Local-score and relation-model pack; defaults to --scene-dir.",
    )
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--consistency", type=float, default=0.60)
    parser.add_argument(
        "--ct-conditioned-identity-score",
        choices=IDENTITY_SCORE_COLUMNS,
        default="geometry_score",
        help=(
            "Candidate score used only for CT-conditioned PAIR/JUNCTION identity "
            "selection; relation-type probabilities are unchanged."
        ),
    )
    parser.add_argument(
        "--ct-conditioned-junction-identity-score",
        choices=IDENTITY_SCORE_COLUMNS,
        help="Optional JUNCTION-only identity score; PAIR keeps --ct-conditioned-identity-score.",
    )
    parser.add_argument("--perturbations", type=int, default=15)
    parser.add_argument("--position-noise-mm", type=float, default=0.8)
    parser.add_argument("--tangent-noise-deg", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument(
        "--save-consistency-details",
        action="store_true",
        help="Save one diagnostic row per model, scene and perturbation repeat.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def endpoint_item(row: pd.Series) -> dict[str, np.ndarray]:
    return {
        "position": row[["x_lps", "y_lps", "z_lps"]].to_numpy(dtype=float),
        "tangent": row[["tangent_x", "tangent_y", "tangent_z"]].to_numpy(
            dtype=float
        ),
    }


def perturbed_candidate_features(
    candidate: pd.Series,
    endpoint_lookup: pd.DataFrame,
    ct: np.ndarray,
    inverse_affine: np.ndarray,
    junction_arm_names: Sequence[str],
) -> dict[str, object]:
    endpoint_ids = str(candidate["endpoint_ids"]).split("|")
    rows = endpoint_lookup.loc[endpoint_ids].reset_index()
    items = [endpoint_item(row) for _, row in rows.iterrows()]
    output: dict[str, object] = {
        "scene_id": candidate["scene_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_type": candidate["candidate_type"],
    }
    if candidate["candidate_type"] == "PAIR":
        output.update(
            dict(zip(relation_training.PAIR_GEOMETRY_FEATURES, scene_builder.pair_geometry(*items)))
        )
        try:
            values = ct_features.extract_pair_features(rows, ct, inverse_affine)
        except ct_features.SceneFeatureError as error:
            if "leaves the CT volume" not in str(error):
                raise
            values = np.full(38, np.nan)
        output.update({f"pair_ct{index}": value for index, value in enumerate(values)})
    else:
        output.update(
            dict(
                zip(
                    relation_training.JUNCTION_GEOMETRY_FEATURES,
                    scene_builder.oriented_junction_geometry(items),
                )
            )
        )
        try:
            values = ct_features.extract_junction_features(
                rows, ct, inverse_affine, junction_arm_names
            )
        except (ct_features.SceneFeatureError, ValueError) as error:
            if "leaves the CT volume" not in str(error):
                raise
            values = np.full(152, np.nan)
        output.update(
            {
                f"junction_{aggregation}_{name}": values[index]
                for index, (aggregation, name) in enumerate(
                    (aggregation, name)
                    for aggregation in ct_features.ARM_AGGREGATIONS
                    for name in junction_arm_names
                )
            }
        )
    return output


def load_verified_ct(
    scan_id: int,
    archive_dir: Path,
    expected_geometry: pd.DataFrame,
    expected_alignment: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    reader = ct_features.SplitZipReader(
        archive_dir / "801-1000.z04", archive_dir / "801-1000.zip"
    )
    raw, _ = reader.extract_entry_bytes(f"801-1000/{scan_id}.img.nii.gz")
    metadata, ct, _ = ct_features.nifti_from_gz_bytes(raw)
    geometry = expected_geometry.loc[scan_id]
    alignment = expected_alignment.loc[scan_id]
    expected_shape = tuple(int(geometry[f"shape_{axis}"]) for axis in "xyz")
    expected_spacing = tuple(float(geometry[f"spacing_{axis}"]) for axis in "xyz")
    checks = (
        tuple(metadata["shape"]) == expected_shape,
        np.allclose(metadata["spacing"], expected_spacing, atol=1e-7, rtol=0),
        np.max(np.abs(metadata["affine"] - ct_features.expected_affine(geometry)))
        <= 1e-6,
        hashlib.sha256(raw).hexdigest() == str(alignment["ct_sha256"]),
    )
    if not all(checks):
        raise GraphLiraError(f"CT alignment failed during perturbation run: {scan_id}")
    return ct, np.linalg.inv(metadata["affine"])


def score_feature_frame(
    frame: pd.DataFrame, local_models: dict[str, dict[str, object]]
) -> pd.DataFrame:
    output = frame[["scene_id", "candidate_id", "candidate_type"]].copy()
    score_columns = [
        f"{stream}_score" for stream in ("geometry", "ct", "geometry_plus_ct")
    ]
    for column in score_columns:
        output[column] = np.nan
    for candidate_type in ("PAIR", "JUNCTION"):
        mask = frame["candidate_type"].eq(candidate_type)
        for stream in ("geometry", "ct", "geometry_plus_ct"):
            model = local_models[candidate_type][stream]
            columns = list(model.feature_names_in_)
            output.loc[mask, f"{stream}_score"] = model.predict_proba(
                frame.loc[mask, columns]
            )[:, 1]
    if not np.isfinite(output[score_columns]).all().all():
        raise GraphLiraError("perturbed local scores are incomplete")
    return output


def relation_feature_rows(
    plans: pd.DataFrame,
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    score_columns: Sequence[str],
) -> tuple[list[dict[str, object]], list[str]]:
    plan_by_scene = {
        row["scene_id"]: row for row in plans.to_dict(orient="records")
    }
    score_by_key = {
        (row["scene_id"], row["candidate_id"]): {
            column: float(row[column]) for column in score_columns
        }
        for row in scores.to_dict(orient="records")
    }
    keys_by_scene_type: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for row in candidates.to_dict(orient="records"):
        key = (row["scene_id"], row["candidate_id"])
        keys_by_scene_type[(row["scene_id"], row["candidate_type"])].append(key)
    return relation_features.build_historical_features(
        plan_by_scene, score_by_key, keys_by_scene_type, list(score_columns)
    )


def predict_relations(
    plans: pd.DataFrame,
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    models: dict[str, object],
    tau: float,
) -> dict[str, dict[str, dict[str, object]]]:
    predictions: dict[str, dict[str, dict[str, object]]] = {}
    for model_name, config in MODEL_CONFIGS.items():
        rows, columns = relation_feature_rows(
            plans, candidates, scores, config["score_columns"]
        )
        expected_columns = models[config["relation_feature_key"]]
        expected_set = set(expected_columns)
        selected_columns = [column for column in columns if column in expected_set]
        if selected_columns != expected_columns or len(expected_set) != len(expected_columns):
            raise GraphLiraError(f"relation feature order changed for {model_name}")
        values = np.asarray(
            [[float(row[column]) for column in expected_columns] for row in rows]
        )
        probabilities = models["relation_heads"][model_name].predict_proba(values)
        predictions[model_name] = {}
        for row, probability in zip(rows, probabilities):
            raw_index = int(np.argmax(probability))
            confidence = float(np.max(probability))
            selected_index = raw_index if confidence >= tau else 0
            predictions[model_name][str(row["scene_id"])] = {
                "raw_relation": RELATION_CLASSES[raw_index],
                "relation": RELATION_CLASSES[selected_index],
                "confidence": confidence,
                "probabilities": probability.tolist(),
            }
    return predictions


def choose_compatible_candidates(
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    relation: str,
    identity_score_column: str | Mapping[str, str] = "geometry_score",
) -> tuple[str | None, str | None]:
    by_type = (
        {"PAIR": identity_score_column, "JUNCTION": identity_score_column}
        if isinstance(identity_score_column, str)
        else dict(identity_score_column)
    )
    if set(by_type) != {"PAIR", "JUNCTION"} or any(
        value not in IDENTITY_SCORE_COLUMNS for value in by_type.values()
    ):
        raise GraphLiraError(f"unsupported candidate identity scores: {by_type}")
    if relation not in RELATION_CLASSES:
        raise GraphLiraError(f"unsupported relation: {relation}")
    needed = list(dict.fromkeys(by_type.values()))
    merged = candidates[["candidate_id", "candidate_type", "endpoint_ids"]].merge(
        scores[["candidate_id", *needed]],
        on="candidate_id",
        validate="one_to_one",
    )
    if len(merged) != len(candidates) or not np.isfinite(merged[needed].to_numpy(dtype=float)).all():
        raise GraphLiraError("candidate identity scores are missing or non-finite")
    merged["identity_score"] = [
        getattr(row, by_type[row.candidate_type]) for row in merged.itertuples()
    ]
    pair_rows = list(merged[merged["candidate_type"] == "PAIR"].itertuples())
    junction_rows = list(merged[merged["candidate_type"] == "JUNCTION"].itertuples())
    pair = (
        max(pair_rows, key=lambda row: row.identity_score, default=None)
        if relation in {"PAIR", "BOTH"}
        else None
    )
    junction = (
        max(junction_rows, key=lambda row: row.identity_score, default=None)
        if relation in {"JUNCTION", "BOTH"}
        else None
    )
    if pair is not None and junction is not None:
        if set(str(pair.endpoint_ids).split("|")) & set(str(junction.endpoint_ids).split("|")):
            best = None
            for pair_row in pair_rows:
                for junction_row in junction_rows:
                    if set(str(pair_row.endpoint_ids).split("|")) & set(
                        str(junction_row.endpoint_ids).split("|")
                    ):
                        continue
                    objective = np.log(max(pair_row.identity_score, 1e-8)) + np.log(
                        max(junction_row.identity_score, 1e-8)
                    )
                    if best is None or objective > best[0]:
                        best = (objective, pair_row, junction_row)
            if best is not None:
                pair, junction = best[1], best[2]
            elif max(row.identity_score for row in pair_rows) >= max(
                row.identity_score for row in junction_rows
            ):
                junction = None
            else:
                pair = None
    return (
        None if pair is None else str(pair.candidate_id),
        None if junction is None else str(junction.candidate_id),
    )


def relation_key(pair_id: str | None, junction_id: str | None) -> str:
    if pair_id is not None and junction_id is not None:
        return f"BOTH|{pair_id}|{junction_id}"
    if pair_id is not None:
        return f"PAIR|{pair_id}"
    if junction_id is not None:
        return f"JUNCTION|{junction_id}"
    return "NONE"


def relation_class(pair_id: str | None, junction_id: str | None) -> str:
    if pair_id is not None and junction_id is not None:
        return "BOTH"
    if pair_id is not None:
        return "PAIR"
    if junction_id is not None:
        return "JUNCTION"
    return "NONE"


def select_scene_decisions(
    plans: pd.DataFrame,
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    relation_predictions: dict[str, dict[str, dict[str, object]]],
    identity_score_columns: dict[str, str | Mapping[str, str]] | None = None,
) -> dict[str, dict[str, dict[str, object]]]:
    identity_score_columns = identity_score_columns or {
        model_name: "geometry_score" for model_name in MODEL_CONFIGS
    }
    output: dict[str, dict[str, dict[str, object]]] = {}
    for model_name in MODEL_CONFIGS:
        output[model_name] = {}
        for scene_id in plans["scene_id"]:
            scene_candidates = candidates[candidates["scene_id"] == scene_id]
            scene_scores = scores[scores["scene_id"] == scene_id]
            prediction = relation_predictions[model_name][scene_id]
            pair_id, junction_id = choose_compatible_candidates(
                scene_candidates,
                scene_scores,
                str(prediction["relation"]),
                identity_score_columns[model_name],
            )
            selected_relation = relation_class(pair_id, junction_id)
            compatibility_satisfied = not (
                prediction["relation"] == "BOTH" and selected_relation != "BOTH"
            )
            output[model_name][scene_id] = {
                **prediction,
                "pair_candidate_id": pair_id,
                "junction_candidate_id": junction_id,
                "selected_relation": selected_relation,
                "relation_key": relation_key(pair_id, junction_id),
                "compatibility_satisfied": compatibility_satisfied,
            }
    return output


def consistency_value(base_key: str, perturbed_keys: Sequence[str]) -> float:
    if not perturbed_keys:
        return 0.0
    counts = Counter(perturbed_keys)
    mode, mode_count = counts.most_common(1)[0]
    if mode != base_key:
        return 0.0
    agree = counts.get(base_key, 0)
    return min(agree / len(perturbed_keys), mode_count / len(perturbed_keys))


def evaluate_structure(
    plan: pd.Series,
    candidates: pd.DataFrame,
    pair_id: str | None,
    junction_id: str | None,
) -> tuple[int, int, int]:
    positives = candidates[candidates["y"].astype(int) == 1]
    true_pair_rows = positives[positives["candidate_type"] == "PAIR"]
    true_junction_rows = positives[positives["candidate_type"] == "JUNCTION"]
    true_pair = None if true_pair_rows.empty else str(true_pair_rows.iloc[0]["candidate_id"])
    true_junction = (
        None if true_junction_rows.empty else str(true_junction_rows.iloc[0]["candidate_id"])
    )
    expected = {
        "PAIR": (true_pair, None),
        "JUNCTION": (None, true_junction),
        "BOTH": (true_pair, true_junction),
        "NONE": (None, None),
    }[str(plan["relation_class"])]
    predicted = (pair_id, junction_id)
    exact = int(predicted == expected)
    false = int(
        (pair_id is not None and pair_id != true_pair)
        or (junction_id is not None and junction_id != true_junction)
    )
    incomplete = int(not exact and not false)
    return exact, false, incomplete


def metric_summary(frame: pd.DataFrame) -> dict[str, object]:
    repair = frame["relation_class"].ne("NONE")
    accepted = frame["accepted"].astype(bool)
    labels = list(RELATION_CLASSES)
    matrix = confusion_matrix(
        frame["relation_class"], frame["final_relation"], labels=labels
    ).tolist()

    def safe_mean(series: pd.Series) -> float:
        return float(series.mean()) if len(series) else float("nan")

    return {
        "scenes": int(len(frame)),
        "repair_needed_scenes": int(repair.sum()),
        "accepted_scenes": int(accepted.sum()),
        "pre_consistency_repair_needed_exact": safe_mean(
            frame.loc[repair, "pre_consistency_exact"]
        ),
        "pre_consistency_false_structural_repair": safe_mean(
            frame["pre_consistency_false"]
        ),
        "pre_consistency_incomplete": safe_mean(frame["pre_consistency_incomplete"]),
        "pre_consistency_proposed_coverage": safe_mean(
            frame["selected_relation_before_consistency"].ne("NONE")
        ),
        "repair_needed_exact": safe_mean(frame.loc[repair, "exact"]),
        "false_structural_repair": safe_mean(frame["false_scene"]),
        "incomplete": safe_mean(frame["incomplete"]),
        "abstain": safe_mean(~accepted),
        "repair_needed_abstain": safe_mean(~accepted[repair]),
        "coverage": safe_mean(accepted),
        "repair_needed_coverage": safe_mean(accepted[repair]),
        "false_among_accepted": safe_mean(frame.loc[accepted, "false_scene"]),
        "exact_among_accepted": safe_mean(frame.loc[accepted, "exact"]),
        "relation_type_accuracy": safe_mean(
            frame["final_relation"].eq(frame["relation_class"])
        ),
        "tau_relation_type_accuracy": safe_mean(
            frame["tau_relation"].eq(frame["relation_class"])
        ),
        "selected_relation_type_accuracy_before_consistency": safe_mean(
            frame["selected_relation_before_consistency"].eq(
                frame["relation_class"]
            )
        ),
        "relation_confusion_matrix": json.dumps(matrix, separators=(",", ":")),
    }


def grouped_metrics(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for (model, group), part in frame.groupby(["model", group_column], sort=True):
        rows.append({"model": model, group_column: group, **metric_summary(part)})
    return pd.DataFrame(rows)


def finite_bootstrap_summary(values: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(finite)),
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def patient_bootstrap(
    frame: pd.DataFrame, repetitions: int, seed: int
) -> pd.DataFrame:
    patients = sorted(frame["scan_id"].astype(int).unique())
    if set(patients) != EXPECTED_VALIDATION_PATIENTS:
        raise GraphLiraError("bootstrap patient set changed")
    generator = np.random.default_rng(seed)
    draws: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _ in range(repetitions):
        sampled = generator.choice(patients, size=len(patients), replace=True)
        for model in MODEL_CONFIGS:
            source = frame[frame["model"] == model]
            parts = [source[source["scan_id"].astype(int) == patient] for patient in sampled]
            summary = metric_summary(pd.concat(parts, ignore_index=True))
            for metric in BOOTSTRAP_METRICS:
                draws[(model, metric)].append(float(summary[metric]))
    rows = []
    estimates = {
        model: metric_summary(frame[frame["model"] == model])
        for model in MODEL_CONFIGS
    }
    for model in MODEL_CONFIGS:
        for metric in BOOTSTRAP_METRICS:
            values = np.asarray(draws[(model, metric)], dtype=float)
            bootstrap_mean, ci_low, ci_high = finite_bootstrap_summary(values)
            rows.append(
                {
                    "comparison": model,
                    "metric": metric,
                    "estimate": estimates[model][metric],
                    "bootstrap_mean": bootstrap_mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "repetitions": repetitions,
                    "cluster": "patient",
                }
            )
    for metric in BOOTSTRAP_METRICS:
        values = np.asarray(draws[("ct_conditioned", metric)]) - np.asarray(
            draws[("geometry_only", metric)]
        )
        bootstrap_mean, ci_low, ci_high = finite_bootstrap_summary(values)
        rows.append(
            {
                "comparison": "ct_conditioned_minus_geometry_only",
                "metric": metric,
                "estimate": (
                    estimates["ct_conditioned"][metric]
                    - estimates["geometry_only"][metric]
                ),
                "bootstrap_mean": bootstrap_mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "repetitions": repetitions,
                "cluster": "patient",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    evidence_dir = args.evidence_dir or args.scene_dir
    if args.tau != 0.85 or args.consistency != 0.60:
        raise GraphLiraError("frozen tau=0.85 and consistency=0.60 must not change")
    if args.perturbations != 15:
        raise GraphLiraError("frozen consistency protocol uses 15 perturbations")
    identity_score_columns = {
        "geometry_only": "geometry_score",
        "ct_conditioned": (
            {"PAIR": args.ct_conditioned_identity_score,
             "JUNCTION": args.ct_conditioned_junction_identity_score}
            if args.ct_conditioned_junction_identity_score
            else args.ct_conditioned_identity_score
        ),
    }
    source_paths = {
        "scene_plan": args.scene_dir / "scene_plan_train_val.csv",
        "scene_endpoints": args.scene_dir / "scene_endpoints_train_val.csv",
        "scene_candidates": args.scene_dir / "scene_candidates_train_val.csv",
        "local_scores": evidence_dir / "relation_local_scores_train_val.csv",
        "relation_predictions": evidence_dir / "relation_predictions_val.csv",
        "relation_models": evidence_dir / "relation_models.joblib",
        "junction_plan": REPO_ROOT
        / "artifacts/varvara/ct28_junction/junction_relation_plan_train_val.csv",
        "junction_audit": REPO_ROOT
        / "artifacts/varvara/ct28_junction/junction_selection_audit_train_val.csv",
        "pair_plan": REPO_ROOT / "artifacts/varvara/ct28_pair/relation_pair_plan.csv",
    }
    plans = pd.read_csv(source_paths["scene_plan"])
    endpoints = pd.read_csv(source_paths["scene_endpoints"])
    candidates = pd.read_csv(source_paths["scene_candidates"])
    local_scores = pd.read_csv(source_paths["local_scores"])
    saved_relation_predictions = pd.read_csv(source_paths["relation_predictions"])
    models = joblib.load(source_paths["relation_models"])
    junction_plan = pd.read_csv(source_paths["junction_plan"])
    junction_audit = pd.read_csv(source_paths["junction_audit"])
    pair_plan = pd.read_csv(source_paths["pair_plan"])
    scene_feature_protocol_path = args.scene_dir / "scene_candidate_feature_protocol.json"
    scene_feature_protocol = (
        json.loads(scene_feature_protocol_path.read_text())
        if scene_feature_protocol_path.exists()
        else {}
    )
    evidence_protocol_path = evidence_dir / "relation_model_protocol.json"
    evidence_protocol = (
        json.loads(evidence_protocol_path.read_text())
        if evidence_protocol_path.exists()
        else {}
    )
    evidence_hashes = evidence_protocol.get("source_sha256", {})
    scene_feature_path = args.scene_dir / "scene_candidate_features_train_val.csv"
    comparable_sources = {
        "scene_plan": source_paths["scene_plan"],
        "scene_candidates": source_paths["scene_candidates"],
        "scene_candidate_features": scene_feature_path,
    }
    mismatched_sources = [
        name
        for name, path in comparable_sources.items()
        if evidence_hashes.get(name)
        and (not path.exists() or evidence_hashes[name] != sha256(path))
    ]
    if mismatched_sources:
        raise GraphLiraError(
            "relation evidence and --scene-dir use different frozen inputs: "
            f"{mismatched_sources}"
        )
    base_stress = scene_feature_protocol.get(
        "base_stress", {"enabled": False, "role": "clean controlled scene"}
    )
    local_model_family = models.get("local_model_family", "logistic")
    if local_model_family not in {"logistic", "hgb"}:
        raise GraphLiraError(f"unsupported local model family: {local_model_family}")
    if (
        evidence_dir.resolve() != args.scene_dir.resolve()
        and args.out_dir.resolve() == args.scene_dir.resolve()
    ):
        raise GraphLiraError(
            "ablation evidence requires a separate --out-dir"
        )

    val_plans = plans[plans["split"] == "val"].copy()
    val_scene_ids = set(val_plans["scene_id"])
    val_endpoints = endpoints[endpoints["scene_id"].isin(val_scene_ids)].copy()
    val_candidates = candidates[candidates["scene_id"].isin(val_scene_ids)].copy()
    val_scores = local_scores[local_scores["scene_id"].isin(val_scene_ids)].copy()
    selected_identity_columns = set()
    for choice in identity_score_columns.values():
        selected_identity_columns.update(choice.values() if isinstance(choice, dict) else [choice])
    required_score_columns = selected_identity_columns | {
        "geometry_score",
        "ct_score",
    }
    missing_score_columns = required_score_columns - set(val_scores.columns)
    if missing_score_columns:
        raise GraphLiraError(
            f"local-score table is missing columns: {sorted(missing_score_columns)}"
        )
    if set(val_plans["scan_id"].astype(int)) != EXPECTED_VALIDATION_PATIENTS:
        raise GraphLiraError("validation patient set changed")
    if set(plans["scan_id"].astype(int)) & {954, 958, 972, 973, 980, 984}:
        raise GraphLiraError("held-out test patient entered the scene pack")

    baseline_relation = predict_relations(
        val_plans, val_candidates, val_scores, models, args.tau
    )
    saved = saved_relation_predictions[
        saved_relation_predictions["model"].isin(MODEL_CONFIGS)
    ]
    for row in saved.itertuples(index=False):
        current = baseline_relation[row.model][row.scene_id]
        if current["raw_relation"] != row.raw_prediction or current["relation"] != row.tau_prediction:
            raise GraphLiraError("baseline relation prediction does not reproduce")
        for index, label in enumerate(RELATION_CLASSES):
            if not np.isclose(
                current["probabilities"][index], getattr(row, f"probability_{label.lower()}"), atol=1e-12
            ):
                raise GraphLiraError("baseline relation probability does not reproduce")
    baseline_decisions = select_scene_decisions(
        val_plans,
        val_candidates,
        val_scores,
        baseline_relation,
        identity_score_columns,
    )

    expected_geometry = pd.read_csv(ct_features.EXPECTED_GEOMETRY).set_index("scan_id")
    expected_alignment = pd.read_csv(ct_features.EXPECTED_ALIGNMENT).set_index("scan_id")
    junction_arm_names = ct_features.arm_feature_names(ct_features.RADII_MM)
    perturbed_keys: dict[str, dict[str, list[str]]] = {
        model: {scene_id: [] for scene_id in val_scene_ids} for model in MODEL_CONFIGS
    }
    perturbation_detail_rows: list[dict[str, object]] = []
    for scan_id in sorted(EXPECTED_VALIDATION_PATIENTS):
        patient_plans = val_plans[val_plans["scan_id"].astype(int) == scan_id]
        patient_scene_ids = set(patient_plans["scene_id"])
        patient_candidates = val_candidates[
            val_candidates["scene_id"].isin(patient_scene_ids)
        ]
        patient_endpoints = val_endpoints[val_endpoints["scene_id"].isin(patient_scene_ids)]
        ct, inverse_affine = load_verified_ct(
            scan_id, args.archive_dir, expected_geometry, expected_alignment
        )
        for repetition in range(args.perturbations):
            feature_rows = []
            for scene_id, scene_endpoint_rows in patient_endpoints.groupby(
                "scene_id", sort=False
            ):
                perturbed = perturb_scene_endpoints(
                    scene_endpoint_rows,
                    seed=args.seed,
                    repetition=repetition,
                    position_noise_mm=args.position_noise_mm,
                    tangent_noise_deg=args.tangent_noise_deg,
                )
                endpoint_lookup = perturbed.set_index("endpoint_id")
                for _, candidate in patient_candidates[
                    patient_candidates["scene_id"] == scene_id
                ].iterrows():
                    feature_rows.append(
                        perturbed_candidate_features(
                            candidate,
                            endpoint_lookup,
                            ct,
                            inverse_affine,
                            junction_arm_names,
                        )
                    )
            feature_frame = pd.DataFrame(feature_rows)
            scores = score_feature_frame(feature_frame, models["local_models"])
            relation_predictions = predict_relations(
                patient_plans, patient_candidates, scores, models, args.tau
            )
            decisions = select_scene_decisions(
                patient_plans,
                patient_candidates,
                scores,
                relation_predictions,
                identity_score_columns,
            )
            for model_name in MODEL_CONFIGS:
                for scene_id in sorted(patient_scene_ids):
                    decision = decisions[model_name][scene_id]
                    perturbed_keys[model_name][scene_id].append(
                        str(decision["relation_key"])
                    )
                    if args.save_consistency_details:
                        perturbation_detail_rows.append(
                            {
                                "model": model_name,
                                "scene_id": scene_id,
                                "scan_id": scan_id,
                                "repetition": repetition,
                                "raw_relation": decision["raw_relation"],
                                "tau_relation": decision["relation"],
                                "selected_relation": decision["selected_relation"],
                                "confidence": decision["confidence"],
                                "pair_candidate_id": decision["pair_candidate_id"]
                                or "",
                                "junction_candidate_id": decision[
                                    "junction_candidate_id"
                                ]
                                or "",
                                "relation_key": decision["relation_key"],
                                "compatibility_satisfied": int(
                                    decision["compatibility_satisfied"]
                                ),
                            }
                        )
        del ct
        print(f"consistency: patient {scan_id} complete", flush=True)

    prediction_rows = []
    consistency_rows = []
    plan_lookup = val_plans.set_index("scene_id")
    junction_group = junction_audit.set_index("junction_id")["benchmark_group"].to_dict()
    negative_junctions = junction_plan[junction_plan["y"].astype(int) == 0]
    min_distance = negative_junctions.groupby("source_junction_id")[
        "geometry_match_distance"
    ].min()
    train_competitive_ids = junction_audit.loc[
        (junction_audit["split"] == "train")
        & (junction_audit["benchmark_group"] == "competitive"),
        "junction_id",
    ]
    ambiguity_cutoff = float(
        min_distance.reindex(train_competitive_ids).dropna().quantile(0.25)
    )
    positive_pairs = pair_plan[pair_plan["y"].astype(int) == 1].set_index("pair_id")
    for model_name in MODEL_CONFIGS:
        for scene_id in val_plans["scene_id"]:
            plan = plan_lookup.loc[scene_id]
            candidates_for_scene = val_candidates[val_candidates["scene_id"] == scene_id]
            base = baseline_decisions[model_name][scene_id]
            consistency = consistency_value(
                str(base["relation_key"]), perturbed_keys[model_name][scene_id]
            )
            proposed = base["selected_relation"] != "NONE"
            accepted = bool(proposed and consistency >= args.consistency)
            final_pair = base["pair_candidate_id"] if accepted else None
            final_junction = base["junction_candidate_id"] if accepted else None
            final_relation = relation_class(final_pair, final_junction)
            pre_exact, pre_false, pre_incomplete = evaluate_structure(
                plan,
                candidates_for_scene,
                base["pair_candidate_id"],
                base["junction_candidate_id"],
            )
            exact, false, incomplete = evaluate_structure(
                plan, candidates_for_scene, final_pair, final_junction
            )
            junction_source_id = (
                "" if pd.isna(plan["junction_source_id"]) else str(plan["junction_source_id"])
            )
            pair_source_id = (
                "" if pd.isna(plan["pair_source_id"]) else str(plan["pair_source_id"])
            )
            benchmark_group = junction_group.get(junction_source_id, "")
            source_distance = (
                float(min_distance[junction_source_id])
                if junction_source_id in min_distance
                else float("nan")
            )
            geometry_ambiguous = int(
                benchmark_group == "competitive"
                and np.isfinite(source_distance)
                and source_distance <= ambiguity_cutoff
            )
            pair_name1 = (
                str(positive_pairs.loc[pair_source_id, "name1"])
                if pair_source_id in positive_pairs.index
                else ""
            )
            pair_name2 = (
                str(positive_pairs.loc[pair_source_id, "name2"])
                if pair_source_id in positive_pairs.index
                else ""
            )
            row = {
                "model": model_name,
                "scene_id": scene_id,
                "scan_id": int(plan["scan_id"]),
                "scene_family": plan["scene_family"],
                "relation_class": plan["relation_class"],
                "pair_source_id": pair_source_id,
                "junction_source_id": junction_source_id,
                "junction_benchmark_group": benchmark_group,
                "junction_geometry_match_distance": source_distance,
                "geometry_ambiguous": geometry_ambiguous,
                "pair_name1_audit": pair_name1,
                "pair_name2_audit": pair_name2,
                "raw_relation": base["raw_relation"],
                "tau_relation": base["relation"],
                "selected_relation_before_consistency": base["selected_relation"],
                "confidence": base["confidence"],
                "combined_consistency": consistency,
                "accepted": int(accepted),
                "final_relation": final_relation,
                "proposed_pair_candidate_id": base["pair_candidate_id"] or "",
                "proposed_junction_candidate_id": base["junction_candidate_id"] or "",
                "pair_candidate_id": final_pair or "",
                "junction_candidate_id": final_junction or "",
                "pre_consistency_exact": pre_exact,
                "pre_consistency_false": pre_false,
                "pre_consistency_incomplete": pre_incomplete,
                "exact": exact,
                "false_scene": false,
                "incomplete": incomplete,
            }
            prediction_rows.append(row)
            consistency_rows.append(
                {
                    "model": model_name,
                    "scene_id": scene_id,
                    "scan_id": int(plan["scan_id"]),
                    "baseline_relation_key": base["relation_key"],
                    "perturbation_keys": json.dumps(
                        perturbed_keys[model_name][scene_id], separators=(",", ":")
                    ),
                    "combined_consistency": consistency,
                    "gate": args.consistency,
                    "accepted": int(accepted),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(
        [
            {"model": model, **metric_summary(predictions[predictions["model"] == model])}
            for model in MODEL_CONFIGS
        ]
    )
    per_patient = grouped_metrics(predictions, "scan_id")
    by_scene_type = grouped_metrics(predictions, "scene_family")
    junction_predictions = predictions[
        predictions["junction_source_id"].astype(str).ne("")
    ].copy()
    junction_predictions["junction_evaluation_group"] = np.where(
        junction_predictions["geometry_ambiguous"].astype(bool),
        "geometry_ambiguous",
        np.where(
            junction_predictions["junction_benchmark_group"].eq("competitive"),
            "competitive_non_ambiguous",
            "isolated",
        ),
    )
    by_junction_group = grouped_metrics(
        junction_predictions, "junction_evaluation_group"
    )
    bootstrap = patient_bootstrap(predictions, args.bootstrap_repetitions, args.seed + 77)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "predictions": args.out_dir / "graph_lira_predictions_val.csv",
        "metrics": args.out_dir / "graph_lira_structural_metrics_val.csv",
        "per_patient": args.out_dir / "graph_lira_per_patient_val.csv",
        "by_scene_type": args.out_dir / "graph_lira_by_scene_type_val.csv",
        "by_junction_group": args.out_dir / "graph_lira_by_junction_group_val.csv",
        "bootstrap": args.out_dir / "graph_lira_patient_bootstrap_val.csv",
        "consistency": args.out_dir / "perturbation_consistency_val.csv",
    }
    if args.save_consistency_details:
        output_paths["consistency_details"] = (
            args.out_dir / "perturbation_decisions_val.csv"
        )
    predictions.to_csv(output_paths["predictions"], index=False)
    metrics.to_csv(output_paths["metrics"], index=False)
    per_patient.to_csv(output_paths["per_patient"], index=False)
    by_scene_type.to_csv(output_paths["by_scene_type"], index=False)
    by_junction_group.to_csv(output_paths["by_junction_group"], index=False)
    bootstrap.to_csv(output_paths["bootstrap"], index=False)
    pd.DataFrame(consistency_rows).to_csv(output_paths["consistency"], index=False)
    if args.save_consistency_details:
        pd.DataFrame(perturbation_detail_rows).to_csv(
            output_paths["consistency_details"], index=False
        )

    protocol = {
        "protocol_version": "ct28_graph_lira_structural_v1",
        "status": "development_train_val_only",
        "held_out_test_accessed": False,
        "path_stage_implemented": False,
        "models": list(MODEL_CONFIGS),
        "local_model_family": local_model_family,
        "base_scene": base_stress,
        "base_scene_protocol_sha256": (
            sha256(scene_feature_protocol_path)
            if scene_feature_protocol_path.exists()
            else None
        ),
        "base_scene_description": (
            "single deterministic 30 degree tangent / 1 mm position stressed realization"
            if base_stress.get("enabled")
            else "clean controlled realization"
        ),
        "evidence_dir": str(evidence_dir.resolve()),
        "identity_score_by_model": identity_score_columns,
        "compatibility": (
            "historical endpoint-disjoint PAIR/JUNCTION selection; maximize "
            "log(pair_score)+log(junction_score) on conflicts"
        ),
        "tau": args.tau,
        "consistency_gate": args.consistency,
        "consistency_protocol": "ct28_perturbation_consistency_v1",
        "consistency_historical_reproduction": False,
        "consistency_formula": (
            "baseline identity agreement fraction if the baseline identity is "
            "also the perturbation mode, otherwise zero"
        ),
        "consistency_details_saved": args.save_consistency_details,
        "perturbations": args.perturbations,
        "position_noise_mm": args.position_noise_mm,
        "tangent_noise_deg": args.tangent_noise_deg,
        "candidate_policy_under_perturbation": (
            "fixed frozen candidate identities; recompute geometry and CT evidence"
        ),
        "seed": args.seed,
        "bootstrap": {
            "cluster": "patient",
            "patients": sorted(EXPECTED_VALIDATION_PATIENTS),
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed + 77,
        },
        "geometry_ambiguous": {
            "definition": (
                "validation competitive junction minimum retained-negative "
                "geometry_match_distance <= train-only lower quartile"
            ),
            "train_only_cutoff": ambiguity_cutoff,
        },
        "degree4_included": False,
        "anatomical_names_role": "failure audit only; never model input",
        "exact_run_command": shlex.join(
            [
                "python",
                "scripts/research/ccta_graph_lira_safe_repair/run_ct28_graph_lira.py",
                *(
                    ["--scene-dir", str(args.scene_dir)]
                    if args.scene_dir.resolve() != DEFAULT_SCENE_DIR.resolve()
                    else []
                ),
                *(
                    ["--evidence-dir", str(evidence_dir)]
                    if args.evidence_dir
                    else []
                ),
                *(
                    ["--out-dir", str(args.out_dir)]
                    if args.out_dir.resolve() != DEFAULT_SCENE_DIR.resolve()
                    else []
                ),
                *(
                    [
                        "--ct-conditioned-identity-score",
                        args.ct_conditioned_identity_score,
                    ]
                    if args.ct_conditioned_identity_score != "geometry_score"
                    else []
                ),
                *(
                    ["--ct-conditioned-junction-identity-score",
                     args.ct_conditioned_junction_identity_score]
                    if args.ct_conditioned_junction_identity_score
                    else []
                ),
                *( ["--save-consistency-details"] if args.save_consistency_details else [] ),
                "--position-noise-mm", str(args.position_noise_mm),
                "--tangent-noise-deg", str(args.tangent_noise_deg),
                "--seed", str(args.seed),
                "--bootstrap-repetitions", str(args.bootstrap_repetitions),
            ]
        ),
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "output_sha256": {name: sha256(path) for name, path in output_paths.items()},
    }
    protocol_path = args.out_dir / "graph_lira_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(metrics.to_string(index=False), flush=True)
    print(f"outputs: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
