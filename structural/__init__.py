"""Shared structural graph algebra; ANZA-2 changes edge evidence, not widest-path novelty."""

from .widest_path import domain_restricted_widest_path, exhaustive_widest_path

__all__ = ["domain_restricted_widest_path", "exhaustive_widest_path"]
