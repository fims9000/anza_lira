#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$(dirname "$project_root")/venv/bin/python"

if [[ ! -x "$python_bin" ]]; then
  echo "Existing Code/venv Python was not found: $python_bin" >&2
  exit 2
fi

cd "$project_root"
exec "$python_bin" scripts/geocrack_study.py full "$@"
