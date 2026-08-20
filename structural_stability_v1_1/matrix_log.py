"""Stable differentiable logarithm for 2x2 SPD metric tensors."""

from __future__ import annotations

import torch


def spd_matrix_log(metric: torch.Tensor, minimum: float = 1e-4, maximum: float = 1e4) -> torch.Tensor:
    if metric.shape[-4:-2] != (2, 2):
        raise ValueError("metric tensor must have shape Bx2x2xHxW")
    matrix_last = metric.permute(0, 3, 4, 1, 2)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix_last)
    log_values = torch.log(eigenvalues.clamp(min=float(minimum), max=float(maximum)))
    logged = eigenvectors @ torch.diag_embed(log_values) @ eigenvectors.transpose(-1, -2)
    return logged.permute(0, 3, 4, 1, 2)


def spd_matrix_exp(log_metric: torch.Tensor) -> torch.Tensor:
    if log_metric.shape[-4:-2] != (2, 2):
        raise ValueError("metric tensor must have shape Bx2x2xHxW")
    matrix_last = log_metric.permute(0, 3, 4, 1, 2)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix_last)
    reconstructed = eigenvectors @ torch.diag_embed(torch.exp(eigenvalues)) @ eigenvectors.transpose(-1, -2)
    return reconstructed.permute(0, 3, 4, 1, 2)
