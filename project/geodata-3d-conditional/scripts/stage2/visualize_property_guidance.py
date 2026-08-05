#!/usr/bin/env python3
"""Render paper-style Phase-2 truth/baseline/guided 3-D comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for import_root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.stage1.visualize_probability_guidance import (
    LABEL_COLORS,
    _choose_sample_id,
    _ensemble_probability,
    _load_json,
    _load_volume,
    _metric_rows_by_id,
    _prepare_output_dir,
    _sample_paths,
    export_vtk_volumes,
    render_paired_changes,
    render_probability_isosurfaces,
    render_triplet,
)
from scripts.stage2.run_property_guidance import paired_property_config_verdict


def _validate_phase2_pair(
    baseline_dir: Path,
    guided_dir: Path,
    target_label: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Require a completed strict Phase-2 property pair before rendering."""
    baseline = _load_json(baseline_dir / "config.json")
    guided = _load_json(guided_dir / "config.json")
    if baseline.get("run_status") != "completed" or guided.get("run_status") != "completed":
        raise ValueError("baseline and guided Phase-2 runs must both be completed")
    if int(baseline["target_label"]) != target_label or int(guided["target_label"]) != target_label:
        raise ValueError("requested target label differs from Phase-2 run configs")
    if int(baseline.get("max_post_projection_condition_violations", -1)) != 0:
        raise ValueError("baseline contains post-projection condition violations")
    if int(guided.get("max_post_projection_condition_violations", -1)) != 0:
        raise ValueError("guided run contains post-projection condition violations")
    paired, reason = paired_property_config_verdict(baseline, guided)
    if not paired:
        raise ValueError(f"strict Phase-2 pair validation failed: {reason}")
    saved_pairing = guided.get("pairing_validation")
    if not isinstance(saved_pairing, dict) or saved_pairing.get("paired") is not True:
        raise ValueError("guided config does not retain the strict pairing verdict")
    if set(_sample_paths(baseline_dir)) != set(_sample_paths(guided_dir)):
        raise ValueError("baseline and guided sample ids differ")
    return baseline, guided


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PyVista Phase-2 property-guidance figures and VTK volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--boreholes", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--guided-dir", type=Path, required=True)
    parser.add_argument("--target-label", type=int, default=9)
    parser.add_argument("--sample-id", type=int, default=None)
    parser.add_argument("--cut-fraction", type=float, default=0.52)
    parser.add_argument(
        "--probability-threshold",
        type=float,
        action="append",
        default=None,
        help="Repeat for empirical ensemble surfaces; defaults to 0.25, 0.5, 0.75.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.cut_fraction < 1.0:
        raise ValueError("cut_fraction must be in (0,1)")
    thresholds = tuple(args.probability_threshold or (0.25, 0.5, 0.75))
    if any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("probability thresholds must be in (0,1]")
    thresholds = tuple(sorted(set(float(value) for value in thresholds)))
    _prepare_output_dir(args.output_dir, args.overwrite)

    baseline_config, guided_config = _validate_phase2_pair(
        args.baseline_dir,
        args.guided_dir,
        args.target_label,
    )
    sample_id = _choose_sample_id(args.guided_dir, args.sample_id)
    truth = _load_volume(args.truth_model).numpy()
    boreholes = _load_volume(args.boreholes).numpy()
    baseline_paths = _sample_paths(args.baseline_dir)
    guided_paths = _sample_paths(args.guided_dir)
    baseline = _load_volume(baseline_paths[sample_id]).numpy()
    guided = _load_volume(guided_paths[sample_id]).numpy()
    target_mask = _load_volume(args.guided_dir / "target_mask.pt").numpy().astype(bool)
    roi_mask = _load_volume(args.guided_dir / "target_roi_mask.pt").numpy().astype(bool)
    shapes = {
        truth.shape,
        boreholes.shape,
        baseline.shape,
        guided.shape,
        target_mask.shape,
        roi_mask.shape,
    }
    if len(shapes) != 1:
        raise ValueError("truth, conditions, samples, target mask and ROI must match")

    baseline_metrics = _metric_rows_by_id(
        args.baseline_dir / "sample_metrics.csv"
    ).get(sample_id)
    guided_metrics = _metric_rows_by_id(
        args.guided_dir / "sample_metrics.csv"
    ).get(sample_id)
    triplet_path = args.output_dir / "truth_baseline_guided_3d.png"
    changes_path = args.output_dir / "paired_changes_3d.png"
    probability_path = args.output_dir / "ensemble_probability_isosurfaces_3d.png"

    render_triplet(
        truth=truth,
        baseline=baseline,
        guided=guided,
        target_mask=target_mask,
        roi_mask=roi_mask,
        boreholes=boreholes,
        target_label=args.target_label,
        sample_id=sample_id,
        alpha=float(guided_config["alpha"]),
        baseline_metrics=baseline_metrics,
        guided_metrics=guided_metrics,
        cut_fraction=args.cut_fraction,
        path=triplet_path,
    )
    changes = render_paired_changes(
        baseline=baseline,
        guided=guided,
        target_mask=target_mask,
        roi_mask=roi_mask,
        target_label=args.target_label,
        sample_id=sample_id,
        path=changes_path,
    )
    baseline_probability = _ensemble_probability(baseline_paths, args.target_label)
    guided_probability = _ensemble_probability(guided_paths, args.target_label)
    render_probability_isosurfaces(
        baseline_probability,
        guided_probability,
        target_mask,
        args.target_label,
        thresholds,
        probability_path,
    )
    vtk_paths = export_vtk_volumes(
        output_dir=args.output_dir / "vtk",
        truth=truth,
        baseline=baseline,
        guided=guided,
        target_mask=target_mask,
        roi_mask=roi_mask,
        boreholes=boreholes,
        target_label=args.target_label,
    )
    manifest = {
        "description": (
            "Paper-style PyVista visualization of a strict Phase-2 paired "
            "truth-derived property-guidance result; not measured geophysics."
        ),
        "ensemble_probability_definition": (
            "Empirical per-voxel occurrence frequency of label 9 across saved "
            "hard-label realizations, not a soft decoder probability."
        ),
        "stage": guided_config["stage"],
        "truth_model": str(args.truth_model),
        "boreholes": str(args.boreholes),
        "baseline_dir": str(args.baseline_dir),
        "guided_dir": str(args.guided_dir),
        "property_config_sha256": guided_config["property_config_sha256"],
        "target_properties_sha256": guided_config["target_properties_sha256"],
        "target_label": int(args.target_label),
        "sample_id": int(sample_id),
        "sample_selection": (
            "explicit --sample-id"
            if args.sample_id is not None
            else "maximum delta_selected_roi_iou from paired_deltas.csv"
        ),
        "alpha": float(guided_config["alpha"]),
        "strict_pairing": True,
        "phase2_protocol_version": guided_config["phase2_protocol_version"],
        "condition_violations": 0,
        "cut_fraction": float(args.cut_fraction),
        "probability_thresholds": list(thresholds),
        "ensemble_size": len(guided_paths),
        "label_colors": {str(key): value for key, value in LABEL_COLORS.items()},
        "baseline_metrics": baseline_metrics,
        "guided_metrics": guided_metrics,
        "paired_change_counts": changes,
        "figures": {
            "truth_baseline_guided_3d": str(triplet_path),
            "paired_changes_3d": str(changes_path),
            "ensemble_probability_isosurfaces_3d": str(probability_path),
        },
        "vtk_volumes": vtk_paths,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Rendered Phase-2 figures and VTK volumes: {args.output_dir}")


if __name__ == "__main__":
    main()
