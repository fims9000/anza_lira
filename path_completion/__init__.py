"""Exact max-min and widest-path structural completion oracles."""

from .maxmin import maxmin_closure_reference, maxmin_closure_torch
from .widest_path import widest_path

__all__ = ["maxmin_closure_reference", "maxmin_closure_torch", "widest_path"]

