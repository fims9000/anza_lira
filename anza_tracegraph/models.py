"""P0 independent corridor baseline and P1/P2 relation transformers."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .corridor import anza_q_bias, extract_corridors


K_MAX = 8
VARIANTS = ("P0_pair", "P1_tracegraph", "P2_anza_tracegraph")


class CorridorEncoder(nn.Module):
    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, output_dim, 3, padding=1), nn.GroupNorm(8, output_dim), nn.GELU(),
        )

    def forward(self, corridor: torch.Tensor) -> torch.Tensor: return self.features(corridor)


class PairClassifierP0(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.encoder = CorridorEncoder(96); self.head = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, batch: dict[str, torch.Tensor], *, return_aux: bool = False) -> torch.Tensor | dict[str, Any]:
        corridors, _ = extract_corridors(batch["dense"], batch["source_point"], batch["destination_points"]); features = self.encoder(corridors); pooled = torch.cat((features.mean((-2, -1)), features.amax((-2, -1))), dim=-1); logits = self.head(pooled).reshape(len(batch["dense"]), K_MAX); logits = logits.masked_fill(~batch["candidate_mask"], -30.0)
        return {"pair_logits": logits, "corridors": corridors} if return_aux else logits


class BiasedTransformerLayer(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4, ffn: int = 256, dropout: float = 0.1) -> None:
        super().__init__(); self.heads = heads; self.norm1 = nn.LayerNorm(dim); self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True); self.norm2 = nn.LayerNorm(dim); self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, dim)); self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, source_to_corridor_bias: torch.Tensor | None = None) -> torch.Tensor:
        value = self.norm1(x); mask = None
        if source_to_corridor_bias is not None:
            batch, corridor_tokens = source_to_corridor_bias.shape; mask = value.new_zeros(batch, x.shape[1], x.shape[1]); mask[:, 0, 1 : 1 + corridor_tokens] = source_to_corridor_bias; mask = mask[:, None].expand(-1, self.heads, -1, -1).reshape(batch * self.heads, x.shape[1], x.shape[1])
        attended = self.attention(value, value, value, attn_mask=mask, need_weights=False)[0]; x = x + self.dropout(attended); return x + self.dropout(self.ffn(self.norm2(x)))


class RelationTransformer(nn.Module):
    def __init__(self, *, use_anza_bias: bool) -> None:
        super().__init__(); self.use_anza_bias = bool(use_anza_bias); dim = 128
        self.corridor_encoder = nn.Sequential(nn.Conv2d(10, dim, kernel_size=8, stride=8), nn.GELU())
        self.source_projection = nn.Linear(16, dim); self.destination_projection = nn.Linear(16, dim); self.geometry_projection = nn.Linear(8, dim)
        self.layers = nn.ModuleList([BiasedTransformerLayer() for _ in range(2)]); self.pair_head = nn.Sequential(nn.Linear(4 * dim, 128), nn.GELU(), nn.Linear(128, 1)); self.none_head = nn.Linear(dim, 1)
        if self.use_anza_bias: self.beta_raw = nn.Parameter(torch.tensor(-4.0))

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_raw) if self.use_anza_bias else torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, batch: dict[str, torch.Tensor], *, return_aux: bool = False) -> torch.Tensor | dict[str, Any]:
        corridors, _ = extract_corridors(batch["dense"], batch["source_point"], batch["destination_points"]); batch_size = len(batch["dense"]); encoded = self.corridor_encoder(corridors); corridor_tokens = encoded.flatten(2).transpose(1, 2); token_count = corridor_tokens.shape[1]
        source = self.source_projection(batch["source_token"])[:, None].expand(-1, K_MAX, -1).reshape(-1, 1, 128); destination = self.destination_projection(batch["destination_tokens"].reshape(-1, 16))[:, None]; geometry = self.geometry_projection(batch["geometry"].reshape(-1, 8))[:, None]
        sequence = torch.cat((source, corridor_tokens, destination, geometry), dim=1); q = anza_q_bias(batch["source_tangent"], batch["source_point"], batch["destination_points"], h=0.35, token_hw=encoded.shape[-2:]).reshape(-1, token_count)
        bias = -self.beta * q if self.use_anza_bias else None
        for layer in self.layers: sequence = layer(sequence, bias)
        source_out = sequence[:, 0]; corridor_out = sequence[:, 1 : 1 + token_count].mean(1); destination_out = sequence[:, 1 + token_count]; geometry_out = sequence[:, 2 + token_count]
        pair = self.pair_head(torch.cat((source_out, destination_out, geometry_out, corridor_out), dim=-1)).reshape(batch_size, K_MAX); pair = pair.masked_fill(~batch["candidate_mask"], -30.0)
        # Source representation is candidate-independent before biased attention; average makes NONE invariant to K ordering.
        source_scene = source_out.reshape(batch_size, K_MAX, -1).mean(1); none = self.none_head(source_scene); logits = torch.cat((pair, none), dim=-1)
        if not return_aux: return logits
        return {"logits": logits, "pair_logits": pair, "corridors": corridors, "beta": self.beta, "bias_mean_abs": torch.zeros((), device=logits.device) if bias is None else bias.abs().mean(), "bias_active_fraction": torch.zeros((), device=logits.device) if bias is None else (bias.abs() > 0.01).float().mean()}


def build_model(variant: str) -> nn.Module:
    if variant == "P0_pair": return PairClassifierP0()
    if variant == "P1_tracegraph": return RelationTransformer(use_anza_bias=False)
    if variant == "P2_anza_tracegraph": return RelationTransformer(use_anza_bias=True)
    raise ValueError(variant)
