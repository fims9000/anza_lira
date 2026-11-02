"""Metamorphic test of the *actual* Graph-LIRA candidate selection function.

Load only the function AST, not the medical-CT module's optional dependencies.
Ground-truth endpoint ID order is never passed to the scorer in this test.
"""
import ast
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts/research/ccta_graph_lira_safe_repair"))
from ct28_safety_policy import should_accept


def selection_function():
    source = (_ROOT / "scripts/research/ccta_graph_lira_safe_repair/run_ct28_graph_lira.py").read_text()
    tree = ast.parse(source)
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef)
             and node.name == "choose_compatible_candidates"]
    assert len(nodes) == 1
    compiled = compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                       "<graph_lira_candidate_selector>", "exec")
    scope = {
        "pd": pd, "np": np, "Mapping": Mapping, "GraphLiraError": ValueError,
        "RELATION_CLASSES": ("NONE", "PAIR", "JUNCTION", "BOTH"),
        "IDENTITY_SCORE_COLUMNS": ("geometry_score", "geometry_plus_ct_score"),
    }
    exec(compiled, scope)
    return scope["choose_compatible_candidates"]


def fixture(renaming=None):
    rename = renaming or {f"E{i:02d}": f"E{i:02d}" for i in range(7)}
    entries = [
        ("PAIR", ["E00", "E01"], .94),
        ("PAIR", ["E02", "E03"], .82),
        ("JUNCTION", ["E01", "E04", "E05"], .95),
        ("JUNCTION", ["E02", "E04", "E06"], .70),
    ]
    candidates, scores = [], []
    for kind, ids, score in entries:
        names = sorted(rename[i] for i in ids)
        cid = kind + ":" + "+".join(names)
        candidates.append({"candidate_id": cid, "candidate_type": kind,
                           "endpoint_ids": "|".join(names)})
        scores.append({"candidate_id": cid, "geometry_score": score,
                       "geometry_plus_ct_score": score})
    return pd.DataFrame(candidates), pd.DataFrame(scores)


def test_candidate_identity_is_endpoint_renaming_invariant():
    choose = selection_function()
    old_candidates, old_scores = fixture()
    selected = choose(old_candidates, old_scores, "BOTH")
    assert selected == ("PAIR:E02+E03", "JUNCTION:E01+E04+E05")
    mapping = {"E00": "E06", "E01": "E04", "E02": "E01",
               "E03": "E05", "E04": "E03", "E05": "E00", "E06": "E02"}
    candidates, scores = fixture(mapping)
    renamed = choose(candidates.iloc[::-1].reset_index(drop=True),
                     scores.iloc[::-1].reset_index(drop=True), "BOTH")
    def rename(cid):
        prefix, values = cid.split(":")
        return prefix + ":" + "+".join(sorted(mapping[x] for x in values.split("+")))
    assert renamed == tuple(rename(cid) for cid in selected)


def test_unavailable_compatible_both_cannot_be_accepted():
    choose = selection_function()
    candidates = pd.DataFrame([
        {"candidate_id": "PAIR:E00+E01", "candidate_type": "PAIR", "endpoint_ids": "E00|E01"},
        {"candidate_id": "JUNCTION:E00+E02+E03", "candidate_type": "JUNCTION", "endpoint_ids": "E00|E02|E03"},
    ])
    scores = pd.DataFrame([
        {"candidate_id": "PAIR:E00+E01", "geometry_score": .9},
        {"candidate_id": "JUNCTION:E00+E02+E03", "geometry_score": .8},
    ])
    pair, junction = choose(candidates, scores, "BOTH")
    assert (pair is None) != (junction is None)
    selected = "PAIR" if pair else "JUNCTION"
    assert not should_accept("BOTH", selected, False, 1.0, .6)
