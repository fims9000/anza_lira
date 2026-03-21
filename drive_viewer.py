#!/usr/bin/env python3
"""Русский интерактивный viewer для DRIVE и AZ-моделей."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from tkinter import messagebox, ttk

import utils
from models.azconv import AZConv2d
from utils import build_model, segmentation_metrics_from_counts

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "drive.yaml"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
SWEEP_THRESHOLDS = [round(x, 2) for x in np.arange(0.25, 0.81, 0.05)]


@dataclass(frozen=True)
class DriveRunInfo:
    name: str
    run_dir: Path
    checkpoint_path: Path
    metrics_path: Path | None
    metrics: dict[str, Any]
    variant: str
    test_dice: float | None
    test_iou: float | None
    test_recall: float | None
    test_precision: float | None
    num_parameters: int | None


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def discover_drive_runs(results_dir: Path) -> list[DriveRunInfo]:
    runs: list[DriveRunInfo] = []
    if not results_dir.exists():
        return runs

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        checkpoint_path = run_dir / "checkpoint_best.pt"
        if not checkpoint_path.exists():
            continue
        metrics_path = run_dir / "metrics.json"
        metrics = _load_json(metrics_path) if metrics_path.exists() else {}
        variant = str(metrics.get("variant", run_dir.name))
        runs.append(
            DriveRunInfo(
                name=run_dir.name,
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                metrics_path=metrics_path if metrics_path.exists() else None,
                metrics=metrics,
                variant=variant,
                test_dice=_safe_float(metrics.get("test_dice")),
                test_iou=_safe_float(metrics.get("test_iou")),
                test_recall=_safe_float(metrics.get("test_recall")),
                test_precision=_safe_float(metrics.get("test_precision")),
                num_parameters=_safe_int(metrics.get("num_parameters")),
            )
        )

    return sorted(
        runs,
        key=lambda run: (run.test_dice if run.test_dice is not None else float("-inf"), run.name),
        reverse=True,
    )


def recommended_threshold_for_run(run: DriveRunInfo, default_threshold: float = 0.6) -> float:
    sweep_path = run.run_dir / "threshold_sweep.json"
    if sweep_path.exists():
        try:
            sweep = _load_json(sweep_path)
            return float(sweep.get("recommended_threshold", default_threshold))
        except Exception:
            return default_threshold
    return default_threshold


def metrics_from_prob_map(
    prob_map: np.ndarray,
    target_mask: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    valid = valid_mask > 0.5
    pred = prob_map >= threshold
    target = target_mask > 0.5

    tp = float(np.logical_and(pred, target)[valid].sum())
    fp = float(np.logical_and(pred, np.logical_not(target))[valid].sum())
    tn = float(np.logical_and(np.logical_not(pred), np.logical_not(target))[valid].sum())
    fn = float(np.logical_and(np.logical_not(pred), target)[valid].sum())
    return segmentation_metrics_from_counts(tp, fp, tn, fn)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def _normalize_for_model(image_rgb: np.ndarray) -> torch.Tensor:
    image = (image_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(image).permute(2, 0, 1)


def _fit_image(image_rgb: np.ndarray, max_size: tuple[int, int] = (430, 430)) -> Image.Image:
    image = Image.fromarray(image_rgb)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val)


def _scalar_map_to_rgb(arr: np.ndarray) -> np.ndarray:
    norm = _normalize_map(arr)
    rgb = np.stack([np.sqrt(norm), norm, 1.0 - norm], axis=-1)
    return _to_uint8_rgb(rgb)


def _heat_overlay(image_rgb: np.ndarray, values: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    heat = _scalar_map_to_rgb(values).astype(np.float32)
    base = image_rgb.astype(np.float32)
    overlay = 0.55 * base + alpha * heat
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _mask_to_overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    base = image_rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    mask3 = mask.astype(np.float32)[..., None]
    blended = base * (1.0 - alpha * mask3) + color_arr * (alpha * mask3)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _error_map(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    target = target > 0.5
    pred = pred > 0.5
    valid = valid > 0.5
    out = np.zeros((*pred.shape, 3), dtype=np.uint8)
    out[np.logical_and(pred, target) & valid] = np.array([40, 220, 80], dtype=np.uint8)
    out[np.logical_and(pred, np.logical_not(target)) & valid] = np.array([235, 80, 70], dtype=np.uint8)
    out[np.logical_and(np.logical_not(pred), target) & valid] = np.array([70, 130, 255], dtype=np.uint8)
    out[np.logical_not(valid)] = np.array([25, 25, 25], dtype=np.uint8)
    return out


def _tensor_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _draw_arrow(draw: ImageDraw.ImageDraw, center: tuple[float, float], vec: np.ndarray, color: tuple[int, int, int], scale: float) -> None:
    vx = float(vec[0]) * scale
    vy = float(vec[1]) * scale
    x0, y0 = center
    x1, y1 = x0 + vx, y0 + vy
    draw.line((x0, y0, x1, y1), fill=color, width=5)
    angle = np.arctan2(vy, vx)
    head_len = 14.0
    left = (x1 - head_len * np.cos(angle - np.pi / 6), y1 - head_len * np.sin(angle - np.pi / 6))
    right = (x1 - head_len * np.cos(angle + np.pi / 6), y1 - head_len * np.sin(angle + np.pi / 6))
    draw.polygon([left, (x1, y1), right], fill=color)


def _geometry_visualization(image_rgb: np.ndarray, interp: dict[str, Any], rule_idx: int) -> np.ndarray:
    theta_map = _tensor_to_numpy(interp.get("theta_map"))
    if theta_map is not None:
        theta = theta_map[rule_idx].astype(np.float32)
        hyper = _tensor_to_numpy(interp.get("hyper_map"))
        hyper_rule = hyper[rule_idx].astype(np.float32) if hyper is not None else np.zeros_like(theta)
        r = (np.cos(theta) + 1.0) * 0.5
        g = (np.sin(theta) + 1.0) * 0.5
        b = _normalize_map(hyper_rule)
        return _to_uint8_rgb(np.stack([r, g, b], axis=-1))

    out = image_rgb.copy()
    pil_image = Image.fromarray(out)
    draw = ImageDraw.Draw(pil_image)
    center = (pil_image.width * 0.18, pil_image.height * 0.2)
    u_vec = _tensor_to_numpy(interp.get("u_vec"))
    s_vec = _tensor_to_numpy(interp.get("s_vec"))
    if u_vec is not None:
        _draw_arrow(draw, center, u_vec[rule_idx], (240, 70, 60), scale=90.0)
    if s_vec is not None:
        _draw_arrow(draw, center, s_vec[rule_idx], (60, 120, 255), scale=90.0)
    draw.rectangle((20, pil_image.height - 72, 320, pil_image.height - 16), fill=(0, 0, 0))
    draw.text((30, pil_image.height - 64), "Красный: unstable", fill=(240, 70, 60))
    draw.text((30, pil_image.height - 40), "Синий: stable", fill=(60, 120, 255))
    return np.array(pil_image, dtype=np.uint8)


class DriveInspectorApp:
    def __init__(self, root: tk.Tk, results_dir: Path, config_path: Path) -> None:
        self.root = root
        self.results_dir = results_dir
        self.config_path = config_path
        self.device_default = "cuda" if torch.cuda.is_available() else "cpu"

        self.runs: list[DriveRunInfo] = []
        self.run_lookup: dict[str, DriveRunInfo] = {}
        self.model_cache: dict[tuple[str, str], tuple[torch.nn.Module, dict[str, Any], torch.device]] = {}
        self.dataset_cache: dict[tuple[str, str], utils.DriveDataset] = {}
        self.prediction_cache: dict[tuple[str, str, int, str], dict[str, Any]] = {}
        self.photo_refs: dict[str, ImageTk.PhotoImage] = {}
        self.layer_display_to_index: dict[str, int] = {}

        self.run_var = tk.StringVar()
        self.device_var = tk.StringVar(value=self.device_default)
        self.split_var = tk.StringVar(value="test")
        self.sample_var = tk.IntVar(value=0)
        self.threshold_var = tk.DoubleVar(value=0.6)
        self.layer_var = tk.StringVar()
        self.rule_var = tk.IntVar(value=1)
        self.sample_info_var = tk.StringVar(value="Образец: -")
        self.status_var = tk.StringVar(value="Готово.")

        self.runs_tree: ttk.Treeview
        self.run_combo: ttk.Combobox
        self.sample_spin: ttk.Spinbox
        self.layer_combo: ttk.Combobox
        self.rule_spin: ttk.Spinbox
        self.threshold_value_label: ttk.Label
        self.run_text: tk.Text
        self.sample_text: tk.Text
        self.interpret_text: tk.Text
        self.sweep_text: tk.Text
        self.seg_labels: dict[str, ttk.Label] = {}
        self.interpret_labels: dict[str, ttk.Label] = {}

        self._build_ui()
        self.refresh_runs(select_best=True)

    def _build_ui(self) -> None:
        self.root.title("Просмотрщик DRIVE: Аносов-Заде")
        self.root.geometry("1680x1020")

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=10)
        right = ttk.Frame(main, padding=10)
        main.add(left, weight=0)
        main.add(right, weight=1)

        controls = ttk.LabelFrame(left, text="Управление", padding=10)
        controls.pack(fill=tk.X)
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Прогон").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        self.run_combo = ttk.Combobox(controls, textvariable=self.run_var, state="readonly", width=40)
        self.run_combo.grid(row=0, column=1, sticky="ew", pady=3)
        self.run_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_run_change())

        ttk.Label(controls, text="Устройство").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        device_combo = ttk.Combobox(controls, textvariable=self.device_var, state="readonly", values=self._device_options(), width=10)
        device_combo.grid(row=1, column=1, sticky="ew", pady=3)
        device_combo.bind("<<ComboboxSelected>>", lambda _event: self.render_selected_sample(force_reload=True))

        ttk.Label(controls, text="Сплит").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        split_combo = ttk.Combobox(controls, textvariable=self.split_var, state="readonly", values=("test", "training"), width=10)
        split_combo.grid(row=2, column=1, sticky="ew", pady=3)
        split_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_split_change())

        ttk.Label(controls, text="Индекс изображения").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=3)
        self.sample_spin = ttk.Spinbox(controls, from_=0, to=0, textvariable=self.sample_var, width=10, command=self.render_selected_sample)
        self.sample_spin.grid(row=3, column=1, sticky="ew", pady=3)
        self.sample_spin.bind("<Return>", lambda _event: self.render_selected_sample())

        ttk.Label(controls, text="Порог").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=3)
        threshold_frame = ttk.Frame(controls)
        threshold_frame.grid(row=4, column=1, sticky="ew", pady=3)
        threshold_frame.columnconfigure(0, weight=1)
        threshold_scale = ttk.Scale(
            threshold_frame,
            from_=0.1,
            to=0.9,
            variable=self.threshold_var,
            command=lambda _value: self.on_threshold_change(),
        )
        threshold_scale.grid(row=0, column=0, sticky="ew")
        self.threshold_value_label = ttk.Label(threshold_frame, width=5, text="0.60")
        self.threshold_value_label.grid(row=0, column=1, padx=(8, 0))

        ttk.Label(controls, text="AZ-слой").grid(row=5, column=0, sticky="w", padx=(0, 8), pady=3)
        self.layer_combo = ttk.Combobox(controls, textvariable=self.layer_var, state="readonly", width=40)
        self.layer_combo.grid(row=5, column=1, sticky="ew", pady=3)
        self.layer_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_layer_change())

        ttk.Label(controls, text="Правило").grid(row=6, column=0, sticky="w", padx=(0, 8), pady=3)
        self.rule_spin = ttk.Spinbox(controls, from_=1, to=1, textvariable=self.rule_var, width=10, command=self.on_rule_change)
        self.rule_spin.grid(row=6, column=1, sticky="ew", pady=3)
        self.rule_spin.bind("<Return>", lambda _event: self.on_rule_change())

        buttons = ttk.Frame(controls)
        buttons.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for col in range(3):
            buttons.columnconfigure(col, weight=1)
        ttk.Button(buttons, text="Назад", command=self.prev_sample).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="Вперед", command=self.next_sample).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(buttons, text="Обновить", command=self.refresh_runs).grid(row=0, column=2, sticky="ew", padx=(4, 0))
        ttk.Button(buttons, text="Оценить сплит", command=self.evaluate_current_split).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0), padx=(0, 4))
        ttk.Button(buttons, text="Sweep порога", command=self.threshold_sweep_current_split).grid(row=1, column=2, sticky="ew", pady=(6, 0), padx=(4, 0))

        ttk.Label(controls, textvariable=self.sample_info_var).grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))

        runs_frame = ttk.LabelFrame(left, text="Сохраненные прогоны", padding=6)
        runs_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.runs_tree = ttk.Treeview(
            runs_frame,
            columns=("variant", "dice", "iou", "recall"),
            show="headings",
            height=9,
        )
        self.runs_tree.heading("variant", text="Вариант")
        self.runs_tree.heading("dice", text="Dice")
        self.runs_tree.heading("iou", text="IoU")
        self.runs_tree.heading("recall", text="Recall")
        self.runs_tree.column("variant", width=105, anchor="w")
        self.runs_tree.column("dice", width=70, anchor="e")
        self.runs_tree.column("iou", width=70, anchor="e")
        self.runs_tree.column("recall", width=70, anchor="e")
        self.runs_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.runs_tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        tree_scroll = ttk.Scrollbar(runs_frame, orient="vertical", command=self.runs_tree.yview)
        tree_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        self.runs_tree.configure(yscrollcommand=tree_scroll.set)

        run_metrics = ttk.LabelFrame(left, text="Метрики прогона", padding=6)
        run_metrics.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.run_text = self._make_readonly_text(run_metrics, height=11)

        sample_metrics = ttk.LabelFrame(left, text="Метрики текущего изображения", padding=6)
        sample_metrics.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.sample_text = self._make_readonly_text(sample_metrics, height=10)

        interpret_metrics = ttk.LabelFrame(left, text="Интерпретация AZ-слоя", padding=6)
        interpret_metrics.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.interpret_text = self._make_readonly_text(interpret_metrics, height=12)

        sweep_metrics = ttk.LabelFrame(left, text="Результат sweep / оценки", padding=6)
        sweep_metrics.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.sweep_text = self._make_readonly_text(sweep_metrics, height=10)

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        seg_tab = ttk.Frame(notebook, padding=6)
        interp_tab = ttk.Frame(notebook, padding=6)
        notebook.add(seg_tab, text="Сегментация")
        notebook.add(interp_tab, text="Интерпретации")

        self._build_panel_grid(seg_tab, self.seg_labels, [
            ("original", "Исходное изображение"),
            ("ground_truth", "Разметка"),
            ("prediction", "Предсказание"),
            ("error_map", "TP / FP / FN"),
        ])
        self._build_panel_grid(interp_tab, self.interpret_labels, [
            ("mu_map", "Карта принадлежности mu"),
            ("kernel", "Ядро правила"),
            ("compat", "Совместимость"),
            ("geometry", "Геометрия Аносова"),
        ])

        status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=(8, 4))
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_panel_grid(self, parent: ttk.Frame, labels: dict[str, ttk.Label], panels: list[tuple[str, str]]) -> None:
        for row in range(2):
            parent.rowconfigure(row, weight=1)
        for col in range(2):
            parent.columnconfigure(col, weight=1)
        for idx, (key, title) in enumerate(panels):
            panel = ttk.LabelFrame(parent, text=title, padding=6)
            panel.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=6, pady=6)
            label = ttk.Label(panel, anchor="center")
            label.pack(fill=tk.BOTH, expand=True)
            labels[key] = label

    def _make_readonly_text(self, parent: ttk.Widget, height: int) -> tk.Text:
        text = tk.Text(parent, height=height, wrap="word")
        text.pack(fill=tk.BOTH, expand=True)
        text.configure(state="disabled")
        return text

    def _device_options(self) -> tuple[str, ...]:
        return ("cuda", "cpu") if torch.cuda.is_available() else ("cpu",)

    def _set_text(self, widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", content.rstrip() + "\n")
        widget.configure(state="disabled")

    def _set_panel(self, label_map: dict[str, ttk.Label], key: str, image_rgb: np.ndarray) -> None:
        photo = ImageTk.PhotoImage(_fit_image(image_rgb))
        self.photo_refs[f"{id(label_map)}:{key}"] = photo
        label_map[key].configure(image=photo)

    def refresh_runs(self, select_best: bool = False) -> None:
        self.runs = discover_drive_runs(self.results_dir)
        self.run_lookup = {run.name: run for run in self.runs}
        run_names = [run.name for run in self.runs]
        self.run_combo.configure(values=run_names)

        for row in self.runs_tree.get_children():
            self.runs_tree.delete(row)
        for run in self.runs:
            self.runs_tree.insert(
                "",
                tk.END,
                iid=run.name,
                values=(run.variant, _format_metric(run.test_dice), _format_metric(run.test_iou), _format_metric(run.test_recall)),
            )

        if not run_names:
            self.run_var.set("")
            self._set_text(self.run_text, "Чекпоинты не найдены.")
            self._set_text(self.sample_text, "Сначала обучите модель или положите чекпоинты в results/.")
            self._set_text(self.interpret_text, "")
            self._set_text(self.sweep_text, "")
            return

        target_name = self.run_var.get()
        if select_best or target_name not in self.run_lookup:
            target_name = run_names[0]
        self.run_var.set(target_name)
        self.runs_tree.selection_set(target_name)
        self.runs_tree.focus(target_name)
        self.on_run_change()

    def current_run(self) -> DriveRunInfo | None:
        return self.run_lookup.get(self.run_var.get())

    def current_dataset(self) -> utils.DriveDataset | None:
        run = self.current_run()
        if run is None:
            return None
        _model, cfg, _device = self._model_for_run(run)
        split = self.split_var.get()
        data_root = _resolve_project_path(cfg.get("data_root", "./data")) / "DRIVE"
        cache_key = (str(data_root), split)
        if cache_key not in self.dataset_cache:
            self.dataset_cache[cache_key] = utils.DriveDataset(
                root=data_root,
                split=split,
                augment=False,
                use_fov_mask=bool(cfg.get("use_fov_mask", True)),
            )
        return self.dataset_cache[cache_key]

    def _checkpoint_payload(self, checkpoint_path: Path) -> dict[str, Any]:
        try:
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location="cpu")

    def _model_for_run(self, run: DriveRunInfo) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
        device_name = self.device_var.get() or self.device_default
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
            self.device_var.set("cpu")
        cache_key = (str(run.checkpoint_path), device_name)
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        payload = self._checkpoint_payload(run.checkpoint_path)
        cfg = payload.get("cfg") or utils.load_config(str(self.config_path))
        utils.set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))
        in_channels, num_outputs = utils.dataset_channels_and_outputs(cfg["dataset"])
        device = torch.device(device_name)
        model = build_model(
            payload.get("variant", run.variant),
            num_outputs=num_outputs,
            in_channels=in_channels,
            num_rules=int(cfg.get("num_rules", 4)),
            task=utils.task_for_dataset(cfg["dataset"]),
        )
        model.load_state_dict(payload["model"])
        model.to(device)
        model.eval()
        self.model_cache[cache_key] = (model, cfg, device)
        return self.model_cache[cache_key]

    def _prediction_for_sample(
        self,
        run: DriveRunInfo,
        dataset: utils.DriveDataset,
        idx: int,
        force_reload: bool = False,
    ) -> dict[str, Any]:
        cache_key = (str(run.checkpoint_path), dataset.split, idx, self.device_var.get())
        if not force_reload and cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        model, _cfg, device = self._model_for_run(run)
        image_path, mask_path, fov_path = dataset.samples[idx]
        image_tensor = dataset._load_rgb(image_path)
        mask_tensor = dataset._load_mask(mask_path)
        valid_tensor = dataset._load_mask(fov_path) if dataset.use_fov_mask else torch.ones_like(mask_tensor)

        image_rgb = np.transpose(image_tensor.numpy(), (1, 2, 0))
        image_rgb_u8 = _to_uint8_rgb(image_rgb)
        normalized = _normalize_for_model(image_rgb_u8).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(normalized)
            logits, _aux_logits, _boundary_logits = utils.unpack_segmentation_outputs(output)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

        layer_infos: list[dict[str, Any]] = []
        for name, module in model.named_modules():
            if isinstance(module, AZConv2d):
                snapshot = module.interpretation_snapshot()
                if snapshot:
                    layer_infos.append({"name": name, "snapshot": snapshot})

        sample = {
            "sample_id": image_path.stem,
            "image": image_rgb_u8,
            "mask": mask_tensor[0].numpy().astype(np.float32),
            "valid": valid_tensor[0].numpy().astype(np.float32),
            "prob": prob,
            "layers": layer_infos,
        }
        self.prediction_cache[cache_key] = sample
        return sample

    def _update_sample_bounds(self) -> None:
        dataset = self.current_dataset()
        if dataset is None or not dataset.samples:
            self.sample_spin.configure(from_=0, to=0)
            self.sample_var.set(0)
            return
        max_index = len(dataset.samples) - 1
        self.sample_spin.configure(from_=0, to=max_index)
        self.sample_var.set(min(max(int(self.sample_var.get()), 0), max_index))

    def on_run_change(self) -> None:
        run = self.current_run()
        if run is None:
            return
        self.threshold_var.set(recommended_threshold_for_run(run, default_threshold=0.6))
        self.threshold_value_label.configure(text=f"{self.threshold_var.get():.2f}")
        self._update_run_text(run)
        self._update_sample_bounds()
        self.render_selected_sample(force_reload=True)

    def on_tree_select(self, _event: Any) -> None:
        selection = self.runs_tree.selection()
        if not selection:
            return
        selected = selection[0]
        if selected != self.run_var.get():
            self.run_var.set(selected)
            self.on_run_change()

    def on_split_change(self) -> None:
        self._update_sample_bounds()
        self.render_selected_sample(force_reload=True)

    def on_threshold_change(self) -> None:
        self.threshold_value_label.configure(text=f"{self.threshold_var.get():.2f}")
        self.render_selected_sample(force_reload=False)

    def on_layer_change(self) -> None:
        self.render_selected_sample(force_reload=False)

    def on_rule_change(self) -> None:
        self.render_selected_sample(force_reload=False)

    def prev_sample(self) -> None:
        if self.sample_var.get() > 0:
            self.sample_var.set(self.sample_var.get() - 1)
            self.render_selected_sample()

    def next_sample(self) -> None:
        dataset = self.current_dataset()
        if dataset is None:
            return
        if self.sample_var.get() < len(dataset.samples) - 1:
            self.sample_var.set(self.sample_var.get() + 1)
            self.render_selected_sample()

    def _update_run_text(self, run: DriveRunInfo) -> None:
        metrics = run.metrics
        lines = [
            f"Прогон: {run.name}",
            f"Вариант: {run.variant}",
            f"Число параметров: {run.num_parameters if run.num_parameters is not None else '-'}",
        ]
        if metrics:
            lines.extend(
                [
                    "",
                    f"Dice: {_format_metric(run.test_dice)}",
                    f"IoU: {_format_metric(run.test_iou)}",
                    f"Precision: {_format_metric(run.test_precision)}",
                    f"Recall: {_format_metric(run.test_recall)}",
                    f"Balanced Accuracy: {_format_metric(_safe_float(metrics.get('test_balanced_accuracy')))}",
                    f"Accuracy: {_format_metric(_safe_float(metrics.get('test_accuracy')))}",
                    f"Время forward batch: {_format_metric(_safe_float(metrics.get('seconds_per_forward_batch')), digits=5)} с",
                ]
            )
        self._set_text(self.run_text, "\n".join(lines))

    def _current_layer_info(self, sample: dict[str, Any]) -> tuple[dict[str, Any] | None, int | None]:
        layers = sample.get("layers", [])
        if not layers:
            return None, None
        layer_display = self.layer_var.get()
        idx = self.layer_display_to_index.get(layer_display, 0)
        idx = min(max(idx, 0), len(layers) - 1)
        return layers[idx], idx

    def _update_layer_controls(self, sample: dict[str, Any]) -> None:
        layers = sample.get("layers", [])
        layer_names = [f"{idx + 1}. {item['name']}" for idx, item in enumerate(layers)]
        self.layer_display_to_index = {name: idx for idx, name in enumerate(layer_names)}
        self.layer_combo.configure(values=layer_names)

        if not layer_names:
            self.layer_var.set("")
            self.rule_var.set(1)
            self.rule_spin.configure(from_=1, to=1)
            return

        if self.layer_var.get() not in self.layer_display_to_index:
            self.layer_var.set(layer_names[0])

        layer_info, _ = self._current_layer_info(sample)
        num_rules = int(layer_info["snapshot"].get("num_rules", 1)) if layer_info is not None else 1
        current_rule = min(max(int(self.rule_var.get()), 1), num_rules)
        self.rule_var.set(current_rule)
        self.rule_spin.configure(from_=1, to=num_rules)

    def _update_sample_text(self, sample: dict[str, Any], metrics: dict[str, float], threshold: float) -> None:
        vessel_fraction = float(((sample["mask"] > 0.5) & (sample["valid"] > 0.5)).sum() / max((sample["valid"] > 0.5).sum(), 1))
        lines = [
            f"Изображение: {sample['sample_id']}",
            f"Порог: {threshold:.2f}",
            f"Доля сосудов в FOV: {vessel_fraction:.4f}",
            "",
            f"Dice: {metrics['dice']:.4f}",
            f"IoU: {metrics['iou']:.4f}",
            f"Precision: {metrics['precision']:.4f}",
            f"Recall: {metrics['recall']:.4f}",
            f"Specificity: {metrics['specificity']:.4f}",
            f"Accuracy: {metrics['accuracy']:.4f}",
            f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}",
        ]
        self._set_text(self.sample_text, "\n".join(lines))

    def _update_interpretation_text(self, layer_info: dict[str, Any] | None, rule_idx: int) -> None:
        if layer_info is None:
            self._set_text(self.interpret_text, "В этой модели нет AZ-слоев для интерпретации.")
            return

        snap = layer_info["snapshot"]
        rule_means = _tensor_to_numpy(snap.get("mu_rule_mean"))
        sigma_u = _tensor_to_numpy(snap.get("sigma_u"))
        sigma_s = _tensor_to_numpy(snap.get("sigma_s"))
        gap = _tensor_to_numpy(snap.get("gap"))
        reg = snap.get("regularization", {})

        lines = [
            f"Слой: {layer_info['name']}",
            f"Режим геометрии: {snap.get('geometry_mode', '-')}",
            f"Выбранное правило: {rule_idx + 1} из {snap.get('num_rules', 1)}",
            "",
        ]
        if rule_means is not None:
            lines.append("Средняя принадлежность mu по правилам:")
            for idx, value in enumerate(rule_means.tolist(), start=1):
                lines.append(f"  правило {idx}: {float(value):.4f}")
            lines.append("")
        if sigma_u is not None and sigma_s is not None:
            lines.append(f"sigma_u(rule {rule_idx + 1}) = {float(sigma_u[rule_idx]):.4f}")
            lines.append(f"sigma_s(rule {rule_idx + 1}) = {float(sigma_s[rule_idx]):.4f}")
        if gap is not None:
            gap_val = float(gap[rule_idx]) if gap.ndim > 0 else float(gap)
            lines.append(f"anisotropy gap(rule {rule_idx + 1}) = {gap_val:.4f}")
        if reg:
            lines.append("")
            lines.append("Регуляризации последнего forward:")
            for key, value in reg.items():
                lines.append(f"  {key}: {float(value):.6f}")
        lines.extend(
            [
                "",
                "Как читать вкладку интерпретаций:",
                "  mu-карта показывает, где правило активно.",
                "  ядро показывает форму локального фильтра.",
                "  совместимость показывает итоговый локальный вес.",
                "  геометрия показывает stable/unstable структуру.",
            ]
        )
        self._set_text(self.interpret_text, "\n".join(lines))

    def _render_interpretation_panels(self, sample: dict[str, Any]) -> None:
        layer_info, _ = self._current_layer_info(sample)
        if layer_info is None:
            blank = np.zeros_like(sample["image"])
            for key in self.interpret_labels:
                self._set_panel(self.interpret_labels, key, blank)
            self._update_interpretation_text(None, 0)
            return

        snap = layer_info["snapshot"]
        num_rules = int(snap.get("num_rules", 1))
        rule_idx = min(max(int(self.rule_var.get()) - 1, 0), num_rules - 1)
        self.rule_var.set(rule_idx + 1)

        mu_map = _tensor_to_numpy(snap.get("mu_map"))
        kernel_map = _tensor_to_numpy(snap.get("kernel_map"))
        compat_map = _tensor_to_numpy(snap.get("compat_map"))

        mu_overlay = _heat_overlay(sample["image"], mu_map[rule_idx]) if mu_map is not None else np.zeros_like(sample["image"])
        kernel_rgb = _scalar_map_to_rgb(kernel_map[rule_idx]) if kernel_map is not None else np.zeros((64, 64, 3), dtype=np.uint8)
        compat_rgb = _scalar_map_to_rgb(compat_map[rule_idx]) if compat_map is not None else np.zeros((64, 64, 3), dtype=np.uint8)
        geometry_rgb = _geometry_visualization(sample["image"], snap, rule_idx)

        self._set_panel(self.interpret_labels, "mu_map", mu_overlay)
        self._set_panel(self.interpret_labels, "kernel", kernel_rgb)
        self._set_panel(self.interpret_labels, "compat", compat_rgb)
        self._set_panel(self.interpret_labels, "geometry", geometry_rgb)
        self._update_interpretation_text(layer_info, rule_idx)

    def render_selected_sample(self, force_reload: bool = False) -> None:
        run = self.current_run()
        dataset = self.current_dataset()
        if run is None or dataset is None:
            return
        if not dataset.samples:
            self._set_text(self.sample_text, "Сплит пуст.")
            return

        idx = min(max(int(self.sample_var.get()), 0), len(dataset.samples) - 1)
        self.sample_var.set(idx)
        self.sample_spin.configure(to=max(len(dataset.samples) - 1, 0))
        self.status_var.set(f"Загрузка изображения {idx} из {self.split_var.get()}...")
        self.root.update_idletasks()

        try:
            sample = self._prediction_for_sample(run, dataset, idx, force_reload=force_reload)
        except Exception as exc:
            self.status_var.set("Ошибка рендера.")
            messagebox.showerror("Просмотрщик DRIVE", str(exc))
            return

        threshold = float(self.threshold_var.get())
        pred = sample["prob"] >= threshold
        gt = sample["mask"] > 0.5
        valid = sample["valid"] > 0.5

        self._set_panel(self.seg_labels, "original", sample["image"])
        self._set_panel(self.seg_labels, "ground_truth", _mask_to_overlay(sample["image"], gt & valid, color=(50, 220, 80)))
        self._set_panel(self.seg_labels, "prediction", _mask_to_overlay(sample["image"], pred & valid, color=(255, 180, 40)))
        self._set_panel(self.seg_labels, "error_map", _error_map(pred, gt, valid))

        self._update_layer_controls(sample)
        self._render_interpretation_panels(sample)

        metrics = metrics_from_prob_map(sample["prob"], sample["mask"], sample["valid"], threshold=threshold)
        self.sample_info_var.set(f"Образец: {sample['sample_id']} | индекс: {idx} | сплит: {self.split_var.get()}")
        self._update_sample_text(sample, metrics, threshold)
        self.status_var.set(f"Готово: {sample['sample_id']} при пороге {threshold:.2f}.")

    def evaluate_current_split(self) -> None:
        run = self.current_run()
        dataset = self.current_dataset()
        if run is None or dataset is None:
            return

        threshold = float(self.threshold_var.get())
        self.status_var.set(f"Оценка сплита {self.split_var.get()} при пороге {threshold:.2f}...")
        self.root.update_idletasks()

        tp = fp = tn = fn = 0.0
        for idx in range(len(dataset.samples)):
            sample = self._prediction_for_sample(run, dataset, idx, force_reload=False)
            valid = sample["valid"] > 0.5
            pred = sample["prob"] >= threshold
            target = sample["mask"] > 0.5
            tp += float(np.logical_and(pred, target)[valid].sum())
            fp += float(np.logical_and(pred, np.logical_not(target))[valid].sum())
            tn += float(np.logical_and(np.logical_not(pred), np.logical_not(target))[valid].sum())
            fn += float(np.logical_and(np.logical_not(pred), target)[valid].sum())
            self.status_var.set(f"Оценка сплита... {idx + 1}/{len(dataset.samples)}")
            self.root.update_idletasks()

        metrics = segmentation_metrics_from_counts(tp, fp, tn, fn)
        lines = [
            f"Оценка прогона: {run.name}",
            f"Сплит: {self.split_var.get()}",
            f"Порог: {threshold:.2f}",
            "",
            f"Dice: {metrics['dice']:.4f}",
            f"IoU: {metrics['iou']:.4f}",
            f"Precision: {metrics['precision']:.4f}",
            f"Recall: {metrics['recall']:.4f}",
            f"Specificity: {metrics['specificity']:.4f}",
            f"Accuracy: {metrics['accuracy']:.4f}",
            f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}",
        ]
        self._set_text(self.sweep_text, "\n".join(lines))
        self.status_var.set("Оценка сплита завершена.")

    def threshold_sweep_current_split(self) -> None:
        run = self.current_run()
        dataset = self.current_dataset()
        if run is None or dataset is None:
            return

        self.status_var.set(f"Sweep порога для {self.split_var.get()}...")
        self.root.update_idletasks()

        samples = [self._prediction_for_sample(run, dataset, idx, force_reload=False) for idx in range(len(dataset.samples))]
        best: tuple[float, dict[str, float]] | None = None
        rows: list[str] = []
        for threshold in SWEEP_THRESHOLDS:
            tp = fp = tn = fn = 0.0
            for sample in samples:
                valid = sample["valid"] > 0.5
                pred = sample["prob"] >= threshold
                target = sample["mask"] > 0.5
                tp += float(np.logical_and(pred, target)[valid].sum())
                fp += float(np.logical_and(pred, np.logical_not(target))[valid].sum())
                tn += float(np.logical_and(np.logical_not(pred), np.logical_not(target))[valid].sum())
                fn += float(np.logical_and(np.logical_not(pred), target)[valid].sum())
            metrics = segmentation_metrics_from_counts(tp, fp, tn, fn)
            rows.append(
                f"thr={threshold:.2f} | dice={metrics['dice']:.4f} | iou={metrics['iou']:.4f} | "
                f"prec={metrics['precision']:.4f} | rec={metrics['recall']:.4f}"
            )
            if best is None or (metrics["dice"], metrics["recall"]) > (best[1]["dice"], best[1]["recall"]):
                best = (threshold, metrics)

        assert best is not None
        self.threshold_var.set(best[0])
        self.threshold_value_label.configure(text=f"{best[0]:.2f}")
        self.render_selected_sample(force_reload=False)
        summary = [
            f"Sweep порога для прогона: {run.name}",
            f"Сплит: {self.split_var.get()}",
            f"Лучший порог: {best[0]:.2f}",
            f"Лучший Dice: {best[1]['dice']:.4f}",
            f"Лучший IoU: {best[1]['iou']:.4f}",
            f"Лучший Precision: {best[1]['precision']:.4f}",
            f"Лучший Recall: {best[1]['recall']:.4f}",
            "",
            *rows,
        ]
        self._set_text(self.sweep_text, "\n".join(summary))
        self.status_var.set(f"Sweep завершен. Лучший порог = {best[0]:.2f}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Русский viewer для DRIVE и Anosov-Zadeh моделей.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    root = tk.Tk()
    app = DriveInspectorApp(
        root=root,
        results_dir=_resolve_project_path(args.results_dir),
        config_path=_resolve_project_path(args.config),
    )
    if not app.runs:
        messagebox.showwarning("Просмотрщик DRIVE", f"Под каталогом {app.results_dir} не найдено чекпоинтов.")
    root.mainloop()


if __name__ == "__main__":
    main()
