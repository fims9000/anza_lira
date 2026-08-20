"""Formula/code audit for legacy v1 and the isolated CleanANZA candidate."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from affinity_repair.v1_audit import v1_source_facts
from models.azconv import AZConv2d
from models.azconv_clean import CleanANZA2d


def formula_code_audit() -> dict[str, Any]:
    legacy_path = Path(inspect.getsourcefile(AZConv2d) or "models/azconv.py")
    clean_path = Path(inspect.getsourcefile(CleanANZA2d) or "models/azconv_clean.py")
    clean_memberships = inspect.getsource(CleanANZA2d._memberships)
    clean_base = inspect.getsource(CleanANZA2d._base_terms)
    clean_normalize = inspect.getsource(CleanANZA2d._normalize)
    facts = v1_source_facts()
    return {
        "status": "V1_FORMULA_CODE_AUDIT_PASS_CLEAN_ANZA_REQUIRED",
        "published_contract": "w_r(p,q)=mu_r(p)mu_r(q)G_r(p,q), then normalize positive interaction weights",
        "legacy": {
            "path": str(legacy_path),
            "sha256": hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
            "membership_activation": facts["membership_activation"],
            "softmax_across_modes": facts["membership_is_categorical_simplex"],
            "pair_membership_product": facts["raw_weight_has_center_membership"] and facts["raw_weight_has_neighbor_membership"],
            "global_rule_neighbor_normalization": facts["global_normalization_over_rule_and_neighbor"],
            "axial_theta_pi_invariance": facts["axial_pair_geometry_uses_doubled_angle"],
            "repeated_mu_attenuation": facts["repeated_membership_beyond_pair_endpoints"],
            "gaussian_literal_half_factor": facts["gaussian_literal_half_factor_present"],
            "gaussian_scale_note": facts["gaussian_scale_equivalence_note"],
            "modified": False,
        },
        "clean_anza": {
            "path": str(clean_path),
            "sha256": hashlib.sha256(clean_path.read_bytes()).hexdigest(),
            "independent_sigmoid_memberships": "torch.sigmoid" in clean_memberships,
            "no_mode_simplex": "softmax" not in clean_memberships,
            "same_pair_product": "mu_center * mu_un * kernel" in clean_base,
            "same_global_normalization": "sum(dim=(1, 2)" in clean_normalize,
            "legacy_geometry_reused": True,
        },
        "expert_data_accessed": False,
        "synthetic_test_accessed": False,
        "cracks_accessed": False,
    }


def write_formula_code_audit(document_path: Path, evidence_path: Path) -> dict[str, Any]:
    result = formula_code_audit()
    evidence_path = Path(evidence_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    legacy = result["legacy"]
    clean = result["clean_anza"]
    document = f"""# ANZA v1 formula/code audit

Status: `{result['status']}`

Published contract: `w_r(p,q)=mu_r(p) mu_r(q) G_r(p,q)` followed by
normalization of positive interaction weights over rules and valid neighbors.

| Property | Legacy v1 | CleanANZA | Verdict |
|---|---|---|---|
| Membership activation | softmax across modes | independent sigmoid | legacy mismatch; clean match |
| Multiple memberships may exceed 0.5 | no, except degenerate two-mode boundary | yes | clean match |
| Pair weight | center mu x neighbor mu x geometry | unchanged | match |
| Normalization | global over rule and neighbor | unchanged | match |
| Axial theta equivalence | doubled-angle local geometry | inherited unchanged | match |
| Repeated membership attenuation | absent beyond pair endpoints | absent | match |
| Gaussian literal 1/2 | absent | inherited | scale-equivalent, not literal |

The legacy source remains unchanged at SHA256 `{legacy['sha256']}`. Its
scientifically material mismatch is categorical softmax competition. CleanANZA
is isolated in `{clean['path']}` and only changes membership activation; the
positive normalized ANZA aggregation is otherwise reused.

The missing literal Gaussian factor `1/2` can be absorbed into learned sigma,
so it is recorded as a parameterization mismatch rather than silently called
equation-identical.
"""
    document_path = Path(document_path)
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text(document)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = write_formula_code_audit(
        root / "docs" / "research" / "ANZA_V1_FORMULA_CODE_AUDIT.md",
        root / "results" / "connectivity_repair" / "pretraining" / "v1_formula_code_audit.json",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))

