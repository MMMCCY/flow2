#!/usr/bin/env python3
"""CLI for the frozen Stage 12A geology-only benchmark build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
STRUCTURALGEO_SRC = REPO_ROOT / "StructuralGeo-main" / "src"
for import_path in (PROJECT_ROOT, STRUCTURALGEO_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from guidance.full_structuralgeo_benchmark import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    build_benchmark,
    probe_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen full StructuralGeo same-recipe geology benchmark."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--probe-seed", type=int)
    parser.add_argument("--probe-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.probe_seed is not None:
        if args.probe_json is None:
            raise SystemExit("--probe-json is required with --probe-seed")
        write_json(args.probe_json, probe_seed(args.probe_seed))
        return 0
    if args.probe_json is not None:
        raise SystemExit("--probe-json is only valid with --probe-seed")
    decision = build_benchmark(
        output_dir=args.output_dir,
        config_path=args.config,
        script_path=Path(__file__),
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
