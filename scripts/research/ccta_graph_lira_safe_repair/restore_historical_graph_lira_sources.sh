#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$ROOT/scripts/research/ccta_graph_lira_safe_repair"
OUT="${1:-/tmp/anza_lira_graph_lira_historical}"

mkdir -p "$OUT"

files=(
  ccta_graph_lira_cross_patient.py.gz.b64
  ccta_graph_lira_cross_patient_v2.py.gz.b64
  ct_conditioned_lira_pilot.py.gz.b64
  ct_scene_graph_lira_hybrid.py.gz.b64
  canonical_mask_veto.py.gz.b64
  selective_mask_veto_joint.py.gz.b64
  evaluate_graph_promotion_gate.py.gz.b64
  run_expanded_geometry_baselines.py.gz.b64
)

for name in "${files[@]}"; do
  src="$SRC/$name"
  if [[ ! -f "$src" ]]; then
    echo "MISSING: $src" >&2
    exit 1
  fi
  out_name="${name%.gz.b64}"
  base64 -d "$src" | gzip -d > "$OUT/$out_name"
  echo "RESTORED: $OUT/$out_name"
done

cat <<EOF

These files are historical references only.
Do not edit them in place and do not make the new CT28 runner depend on hidden
historical pickle/joblib state.

Current task:
  docs/varvara/TASK_CT28_CT_CONDITIONED_GRAPH_LIRA_2026-10-06.md
EOF
