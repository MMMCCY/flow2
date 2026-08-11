#!/usr/bin/env python3
"""Generate and validate every main-paper figure in one deterministic command."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.paper_figures import figure01_framework
from scripts.paper_figures import figure02_controllability
from scripts.paper_figures import figure03_seismic_structured
from scripts.paper_figures import figure04_information_hierarchy
from scripts.paper_figures.style import (
    FIGURE_DATA_DIR,
    FIGURES_DIR,
    MANIFESTS_DIR,
    ensure_output_dirs,
    git_head,
    read_json,
    sha256,
    source_record,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
GENERATORS = (
    ("figure01_framework", figure01_framework.generate),
    ("figure02_controllability", figure02_controllability.generate),
    ("figure03_seismic_structured", figure03_seismic_structured.generate),
    ("figure04_information_hierarchy", figure04_information_hierarchy.generate),
)


def _all_deliverable_paths() -> list[Path]:
    paths: list[Path] = []
    for figure_id, _ in GENERATORS:
        paths.extend(FIGURES_DIR / f"{figure_id}.{suffix}" for suffix in ("pdf", "svg", "png"))
        paths.append(MANIFESTS_DIR / f"{figure_id}.json")
        data_matches = sorted(FIGURE_DATA_DIR.glob(f"{figure_id}.*"))
        if len(data_matches) != 1:
            raise ValueError(f"expected exactly one figure-data artifact for {figure_id}: {data_matches}")
        paths.extend(data_matches)
    return paths


def _validate_output_set() -> dict[str, object]:
    checks: dict[str, object] = {}
    for figure_id, _ in GENERATORS:
        paths = {suffix: FIGURES_DIR / f"{figure_id}.{suffix}" for suffix in ("pdf", "svg", "png")}
        for suffix, path in paths.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"missing/empty {suffix} output: {path}")
        if paths["pdf"].read_bytes()[:4] != b"%PDF":
            raise ValueError(f"invalid PDF header: {paths['pdf']}")
        if "<svg" not in paths["svg"].read_text(encoding="utf-8")[:2000]:
            raise ValueError(f"invalid SVG document: {paths['svg']}")
        with Image.open(paths["png"]) as image:
            dpi = image.info.get("dpi")
            if dpi is None or any(abs(float(value) - 600.0) > 1.0 for value in dpi):
                raise ValueError(f"PNG is not 600 dpi: {paths['png']} ({dpi})")
            checks[figure_id] = {"pixel_size": list(image.size), "dpi": [float(value) for value in dpi]}
        manifest = read_json(MANIFESTS_DIR / f"{figure_id}.json")
        if manifest.get("figure_id") != figure_id:
            raise ValueError(f"manifest figure id mismatch: {figure_id}")
        if manifest.get("generation", {}).get("git_head") != git_head():
            raise ValueError(f"manifest git HEAD mismatch: {figure_id}")
        if len(manifest.get("outputs", [])) != 3:
            raise ValueError(f"manifest output inventory is incomplete: {figure_id}")
    return checks


def _write_index(results: list[dict[str, object]], checks: dict[str, object]) -> Path:
    index_path = MANIFESTS_DIR / "paper_figures.json"
    manifests = [MANIFESTS_DIR / f"{figure_id}.json" for figure_id, _ in GENERATORS]
    data_files = [next(iter(sorted(FIGURE_DATA_DIR.glob(f"{figure_id}.*")))) for figure_id, _ in GENERATORS]
    payload = {
        "schema": "paper_figure_collection_manifest_v1",
        "git_head": git_head(),
        "generator": source_record(SCRIPT_PATH, "collection generator"),
        "figures": results,
        "figure_manifests": [source_record(path, "figure manifest") for path in manifests],
        "figure_data": [source_record(path, "figure data") for path in data_files],
        "output_checks": checks,
        "deterministic_policy": {
            "fixed_source_data": True,
            "fixed_camera": True,
            "fixed_robust_percentiles": True,
            "fixed_pdf_svg_png_metadata": True,
            "fixed_npz_member_timestamps": True,
            "pdf_metadata_timestamp_note": "CreationDate and ModDate are fixed at 2000-01-01 UTC.",
        },
    }
    write_json(index_path, payload)
    return index_path


def generate_all() -> tuple[list[dict[str, object]], dict[str, object], Path]:
    ensure_output_dirs()
    results = []
    for figure_id, generator in GENERATORS:
        print(f"[paper figures] generating {figure_id}", flush=True)
        results.append(generator())
    checks = _validate_output_set()
    index = _write_index(results, checks)
    return results, checks, index


def _hash_snapshot() -> dict[str, str]:
    paths = _all_deliverable_paths() + [MANIFESTS_DIR / "paper_figures.json"]
    return {str(path.relative_to(PROJECT_DIR)): sha256(path) for path in sorted(paths)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-deterministic",
        action="store_true",
        help="Generate twice and require byte-identical figures, data and manifests.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate an existing complete output set without regenerating it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        checks = _validate_output_set()
        print({"status": "PASS", "checks": checks})
        return
    results, checks, index = generate_all()
    deterministic = None
    if args.verify_deterministic:
        first = _hash_snapshot()
        generate_all()
        second = _hash_snapshot()
        mismatches = {name: {"first": first.get(name), "second": second.get(name)} for name in sorted(set(first) | set(second)) if first.get(name) != second.get(name)}
        if mismatches:
            raise RuntimeError(f"determinism check failed: {mismatches}")
        deterministic = True
    print(
        {
            "status": "PASS",
            "figures": [item["figure"] for item in results],
            "checks": checks,
            "collection_manifest": str(index.relative_to(PROJECT_DIR)),
            "byte_identical_rerun": deterministic,
        }
    )


if __name__ == "__main__":
    main()

