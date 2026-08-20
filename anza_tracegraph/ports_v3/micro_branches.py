"""Candidate-only micro-branch view."""

from .branches import Branch


def micro_branches(branches: tuple[Branch, ...]) -> tuple[Branch, ...]:
    return tuple(branch for branch in branches if branch.candidate_only and 4.0 <= branch.length < 8.0)
