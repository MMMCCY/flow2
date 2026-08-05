#!/usr/bin/env bash
set -euo pipefail

python scripts/stage5/build_acoustic_inversion_posterior.py
python scripts/stage5/audit_acoustic_inversion_posterior.py
