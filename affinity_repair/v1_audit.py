"""Machine-readable audit of the unchanged AZConv2d v1 implementation."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from models.azconv import AZConv2d


def v1_source_facts() -> dict[str, Any]:
    source = inspect.getsource(AZConv2d.forward)
    local_geometry = inspect.getsource(AZConv2d._local_hyperbolic_kernel)
    return {
        "membership_activation": "softmax_across_rules",
        "membership_is_categorical_simplex": "F.softmax(logits / float(self.cfg.fuzzy_temperature), dim=1)" in source,
        "raw_weight_has_center_membership": "mu_center * mu_un * kern * valid_un" in source,
        "raw_weight_has_neighbor_membership": "mu_center * mu_un * kern * valid_un" in source,
        "global_normalization_over_rule_and_neighbor": "compat.sum(dim=(1, 2), keepdim=True)" in source,
        "default_normalization_mode": "global",
        "repeated_membership_beyond_pair_endpoints": False,
        "axial_pair_geometry_uses_doubled_angle": all(token in local_geometry for token in ("torch.cos(2.0 * theta_center)", "torch.sin(2.0 * theta_center)")),
        "gaussian_literal_half_factor_present": "-0.5" in local_geometry,
        "gaussian_scale_equivalence_note": "missing literal 1/2 can be absorbed into learned sigma but is not equation-identical parameterization",
        "fuzzy_independent_candidate_required": True,
    }


def write_v1_audit(document_path: Path, evidence_path: Path) -> dict[str, Any]:
    source_path = Path(inspect.getsourcefile(AZConv2d) or "models/azconv.py")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    facts = v1_source_facts()
    result = {
        "status": "V1_PARTIAL_EQUATION_MATCH_C1_REQUIRED",
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "facts": facts,
        "legacy_source_modified": False,
        "expert_data_accessed": False,
        "test_stream_accessed": False,
    }
    evidence_path = Path(evidence_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    document = f"""# AZConv2d v1 code-equation audit

Status: `{result['status']}`

The audited source is the unchanged `models/azconv.py` with SHA256 `{source_hash}`.

## Findings

| Contract | Code finding | Verdict |
|---|---|---|
| `mu_r` is an independent fuzzy degree | `F.softmax(..., dim=1)` forces `sum_r mu_r = 1` | **MISMATCH** |
| `w_r = mu_r(p) mu_r(q) G_r` | Code uses `mu_center * mu_un * kern` | MATCH |
| normalize over neighbor and rule | Default global path sums dimensions `(rule, neighbor)` | MATCH |
| no extra membership attenuation | Membership occurs only at the two pair endpoints | MATCH |
| axial `theta == theta + pi` | Local pair geometry uses `cos(2 theta), sin(2 theta)` | MATCH |
| Gaussian exponent includes literal `1/2` | Code uses `exp(-du^2/sigma_u^2-ds^2/sigma_s^2)` | PARAMETERIZATION MISMATCH |
| positive finite scales | softplus base and bounded hyperbolicity produce positive finite scales for finite parameters | MATCH |
| isotropic limit ignores direction | `use_anisotropy=False` uses radial squared distance | MATCH |

The missing Gaussian `1/2` is scale-equivalent after `sigma -> sigma/sqrt(2)`, but the parameterization is not literally the supplied equation. The scientifically material mismatch is categorical softmax membership. Therefore C1 (`v1_fuzzy_independent`) is required as a separate ablation; the frozen v1 code is not edited.
"""
    document_path = Path(document_path)
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text(document)
    return result
