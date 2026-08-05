#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

PYTHONPATH=src "$PYTHON_BIN" \
  project/geodata-3d-conditional/scripts/stage4/audit_seismic_identifiability.py \
  --device "$DEVICE" \
  "${OVERWRITE_ARGS[@]}"
