from __future__ import annotations

import torch

import train
import utils
from models.azconv import AZConv2d


def _capture_shapes(model: torch.nn.Module, module_names: list[str], x: torch.Tensor) -> dict[str, object]:
    shapes: dict[str, object] = {}
    name_to_module = dict(model.named_modules())
    handles = []

    def _shape_of(value):
        if isinstance(value, torch.Tensor):
            return tuple(value.shape)
        if isinstance(value, dict):
            return {key: _shape_of(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_shape_of(item) for item in value]
        return type(value).__name__

    for name in module_names:
        module = name_to_module[name]

        def hook(_module, _inputs, output, *, name=name):
            shapes[name] = _shape_of(output)

        handles.append(module.register_forward_hook(hook))

    try:
        with torch.no_grad():
            model.eval()
            _ = model(x)
    finally:
        for handle in handles:
            handle.remove()
    return shapes


def test_az_thesis_stage_shapes_match_encoder_decoder_contract():
    model = utils.build_model(
        "az_thesis",
        num_outputs=1,
        in_channels=3,
        num_rules=3,
        task="segmentation",
        widths=(8, 16, 24, 32),
        model_kwargs={
            "encoder_az_stages": 2,
            "encoder_block_mode": "hybrid",
            "hybrid_mix_init": 0.25,
            "bottleneck_mode": "aspp",
            "decoder_mode": "residual",
            "boundary_mode": "conv",
        },
        az_cfg_kwargs={
            "geometry_mode": "local_hyperbolic",
            "learn_directions": True,
            "geometry_kernel_size": 3,
            "normalize_mode": "per_rule",
            "compatibility_floor": 1e-4,
            "use_input_residual": True,
            "residual_init": 0.1,
        },
    )
    x = torch.randn(1, 3, 64, 80)

    shapes = _capture_shapes(
        model,
        ["enc1", "enc2", "enc3", "bottleneck", "up3", "up2", "up1", "main_head", "boundary_head"],
        x,
    )
    out = model.eval()(x)

    assert shapes["enc1"] == (1, 8, 64, 80)
    assert shapes["enc2"] == (1, 16, 32, 40)
    assert shapes["enc3"] == (1, 24, 16, 20)
    assert shapes["bottleneck"] == (1, 32, 8, 10)
    assert shapes["up3"] == (1, 24, 16, 20)
    assert shapes["up2"] == (1, 16, 32, 40)
    assert shapes["up1"] == (1, 8, 64, 80)
    assert shapes["main_head"] == (1, 1, 64, 80)
    assert shapes["boundary_head"] == (1, 1, 64, 80)
    assert out["logits"].shape == (1, 1, 64, 80)
    assert [tuple(aux.shape) for aux in out["aux_logits"]] == [(1, 1, 64, 80), (1, 1, 64, 80)]
    assert out["boundary_logits"].shape == (1, 1, 64, 80)
    assert torch.isfinite(out["logits"]).all()


def test_build_model_az_overrides_reach_nested_az_layers():
    model = utils.build_model(
        "az_thesis",
        num_outputs=1,
        in_channels=3,
        num_rules=5,
        task="segmentation",
        widths=(8, 16, 24, 32),
        model_kwargs={
            "encoder_az_stages": 1,
            "encoder_block_mode": "az",
            "bottleneck_mode": "az_single",
            "decoder_mode": "az",
            "boundary_mode": "az",
        },
        az_cfg_kwargs={
            "geometry_mode": "local_hyperbolic",
            "learn_directions": True,
            "geometry_kernel_size": 3,
            "normalize_mode": "per_rule",
            "fuzzy_temperature": 1.7,
            "min_hyperbolicity": 0.2,
            "compatibility_floor": 2e-4,
        },
    )

    az_layers = [module for module in model.modules() if isinstance(module, AZConv2d)]
    assert az_layers
    assert all(layer.R == 5 for layer in az_layers)
    assert all(layer.cfg.geometry_mode == "local_hyperbolic" for layer in az_layers)
    assert all(layer.cfg.geometry_kernel_size == 3 for layer in az_layers)
    assert all(layer.cfg.normalize_mode == "per_rule" for layer in az_layers)
    assert all(layer.cfg.fuzzy_temperature == 1.7 for layer in az_layers)
    assert all(layer.cfg.min_hyperbolicity == 0.2 for layer in az_layers)
    assert all(layer.cfg.compatibility_floor == 2e-4 for layer in az_layers)
    assert all(layer.geometry_conv is not None and layer.geometry_conv.kernel_size == (3, 3) for layer in az_layers)


def test_per_rule_az_kernel_normalization_preserves_fuzzy_rule_mass():
    x = torch.randn(1, 4, 11, 13)
    layer = AZConv2d(
        4,
        6,
        kernel_size=3,
        num_rules=4,
        cfg=utils.az_config_from_variant_and_overrides(
            "az_full",
            {
                "normalize_mode": "per_rule",
                "geometry_mode": "local_hyperbolic",
                "geometry_kernel_size": 3,
            },
        ),
    )

    y = layer(x)
    snapshot = layer.interpretation_snapshot()
    per_rule_mass = snapshot["compat_map"].float().sum(dim=(1, 2))
    mu_rule_mean = snapshot["mu_rule_mean"].float()

    assert y.shape == (1, 6, 11, 13)
    assert torch.isfinite(y).all()
    assert torch.allclose(per_rule_mass, mu_rule_mean, atol=2e-3)


def test_architecture_state_reports_active_az_math_after_forward():
    model = utils.build_model(
        "az_cat",
        num_outputs=1,
        in_channels=3,
        num_rules=6,
        task="segmentation",
        widths=(8, 16, 24, 32),
    )
    x = torch.randn(1, 3, 48, 48)

    with torch.no_grad():
        _ = model.eval()(x)
    state = train.collect_architecture_state(model)

    assert state["az_layer_count"] > 0
    assert state["az_geometry_mode_counts"]["fixed_cat_map"] == state["az_layer_count"]
    assert state["az_metric_min_eig_min"] > 0.0
    assert state["az_metric_condition_max"] >= 1.0
    assert 0.0 <= state["az_rule_usage_entropy_norm_mean"] <= 1.0
    assert abs(state["az_compat_mass_mean"] - 1.0) < 1e-3


def test_one_segmentation_batch_runs_from_model_to_loss_metrics_and_backward():
    model = utils.build_model(
        "az_thesis",
        num_outputs=1,
        in_channels=3,
        num_rules=3,
        task="segmentation",
        widths=(8, 16, 24, 32),
        model_kwargs={
            "encoder_az_stages": 1,
            "encoder_block_mode": "hybrid",
            "hybrid_mix_init": 0.3,
            "bottleneck_mode": "aspp",
            "decoder_mode": "residual",
            "boundary_mode": "conv",
        },
        az_cfg_kwargs={
            "geometry_mode": "local_hyperbolic",
            "learn_directions": True,
            "geometry_kernel_size": 3,
        },
    )
    x = torch.randn(2, 3, 64, 64)
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 18:46, 30:34] = 1.0
    valid_mask = torch.ones_like(target)

    output = model(x)
    loss, logs, main_logits = utils.segmentation_objective(
        output,
        target,
        valid_mask,
        bce_weight=1.0,
        dice_weight=1.0,
        aux_weight=0.2,
        boundary_weight=0.1,
        topology_weight=0.01,
        topology_num_iters=5,
    )
    loss.backward()
    tp, fp, tn, fn = utils.binary_confusion_counts(main_logits.detach(), target, valid_mask, threshold=0.5)
    metrics = utils.segmentation_metrics_from_counts(tp, fp, tn, fn)
    skel_counts = utils.skeleton_confusion_counts(main_logits.detach(), target, valid_mask, threshold=0.5, num_iters=5)
    skel_metrics = utils.skeleton_metrics_from_counts(*skel_counts)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(main_logits).all()
    assert {"bce_loss", "dice_loss", "aux_loss", "boundary_loss", "topology_loss"} <= set(logs)
    assert all(0.0 <= metrics[key] <= 1.0 for key in ("dice", "iou", "precision", "recall", "balanced_accuracy"))
    assert all(0.0 <= skel_metrics[key] <= 1.0 for key in ("cldice", "skeleton_precision", "skeleton_recall"))
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())
