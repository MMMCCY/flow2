#!/usr/bin/env python3
"""Generate registered noiseless Stage12B observations, then close truth access."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.prior_ensemble import file_sha256, hard_seismic_response
from guidance.seismic import acoustic_tables_from_config, seismic_operator_from_config
from scripts.stage12b.common import (
    CASE_IDS,
    CONFIG_PATH,
    EXPERIMENT_DIR,
    STAGE12A_DIR,
    load_config,
    load_prior_bank,
    resolve_path,
    verify_stage12a_files,
)
from scripts.stage9.common import (
    create_staging_directory,
    file_record,
    publish_staging_directory,
    read_json,
    save_tensor_x,
    utc_now,
    write_json_x,
)


def _load(path: Path) -> torch.Tensor:
    return runtime.load_tensor(path, map_location="cpu").contiguous()


def main() -> None:
    config = load_config()
    verify_stage12a_files(config)
    # This is the critical temporal precondition: the common prior bank exists
    # and is immutable before any Stage12B observation is generated.
    prior_manifest, _ = load_prior_bank()
    if prior_manifest.get("seismic_observation_received") is not False:
        raise ValueError("prior bank was not frozen independently of Stage12B seismic")
    observations = EXPERIMENT_DIR / "observations"
    if observations.exists():
        raise FileExistsError(f"refusing to reuse immutable output: {observations}")
    observations.mkdir(parents=True)

    acoustic_path = resolve_path(config["acoustic_config"]["path"])
    seismic_path = resolve_path(config["seismic_config"]["path"])
    acoustic_config = read_json(acoustic_path)
    seismic_config = read_json(seismic_path)
    tables, acoustic_metadata = acoustic_tables_from_config(acoustic_config, 15)
    property_table = tables.property_table.float()
    operator, seismic_metadata = seismic_operator_from_config(
        seismic_config, grid_shape=tuple(config["grid_shape"])
    )
    case_records: dict[str, object] = {}
    for case_id in CASE_IDS:
        source = STAGE12A_DIR / "cases" / case_id
        truth = _load(source / "truth/true_model.pt").long()
        condition_values = _load(source / "condition/condition_values.pt").long()
        condition_mask = _load(source / "condition/condition_mask.pt").bool()
        surface_mask = _load(source / "condition/surface_mask.pt").bool()
        borehole_mask = _load(source / "condition/borehole_mask.pt").bool()
        if not torch.equal(condition_mask, surface_mask | borehole_mask):
            raise ValueError(f"{case_id}: condition mask is not surface | borehole")
        outside_air = surface_mask & condition_values.eq(-1)
        subsurface = ~outside_air
        # Truth may be consulted only within this synthetic-observation process.
        if not torch.equal(subsurface, truth.ne(-1)):
            raise ValueError(f"{case_id}: inference-visible surface does not identify support")
        if not torch.equal(condition_values[condition_mask], truth[condition_mask]):
            raise ValueError(f"{case_id}: frozen conditions disagree with truth")
        observed = hard_seismic_response(
            truth,
            property_table=property_table,
            subsurface_mask=subsurface,
            forward_operator=operator,
        )
        final = observations / case_id
        staging = create_staging_directory(final)
        tensors = {
            "condition_values": save_tensor_x(staging / "condition_values.pt", condition_values),
            "condition_mask": save_tensor_x(staging / "condition_mask.pt", condition_mask),
            "surface_mask": save_tensor_x(staging / "surface_mask.pt", surface_mask),
            "borehole_mask": save_tensor_x(staging / "borehole_mask.pt", borehole_mask),
            "subsurface_mask": save_tensor_x(staging / "subsurface_mask.pt", subsurface),
            "acoustic_property_table": save_tensor_x(
                staging / "acoustic_property_table.pt", property_table
            ),
            "observation_correct": save_tensor_x(staging / "observation_correct.pt", observed),
        }
        write_json_x(
            staging / "manifest.json",
            {
                "schema": "stage12b_inference_case_v1",
                "status": "complete_truth_generation_process_closed",
                "created_at_utc": utc_now(),
                "case_id": case_id,
                "axis_order": config["axis_order"],
                "grid_shape": config["grid_shape"],
                "acquisition": "noiseless_full_64x64_upper_bound",
                "geological_truth_available_to_inference": False,
                "truth_path_recorded_in_inference_manifest": False,
                "prior_bank_was_frozen_before_observation": True,
                "fixed_nine_well_layout": config["fixed_well_layout"],
                "acoustic_metadata": acoustic_metadata,
                "seismic_metadata": seismic_metadata,
                "tensors": tensors,
                "source_assets": {
                    "protocol": file_record(CONFIG_PATH, relative_to=REPOSITORY_ROOT),
                    "acoustic_config": file_record(acoustic_path, relative_to=REPOSITORY_ROOT),
                    "seismic_config": file_record(seismic_path, relative_to=REPOSITORY_ROOT),
                    "stage12a_case_manifest_sha256": config["stage12a"]["case_hashes"][case_id]["manifest"],
                    "stage12a_truth_tensor_sha256": config["stage12a"]["case_hashes"][case_id]["true_model"],
                    "builder_source": file_record(Path(__file__), relative_to=REPOSITORY_ROOT),
                },
            },
        )
        publish_staging_directory(staging, final)
        case_records[case_id] = file_record(final / "manifest.json", relative_to=REPOSITORY_ROOT)
        print(f"generated frozen Stage12B observation: {case_id}", flush=True)
        del truth, observed
    write_json_x(
        observations / "manifest.json",
        {
            "schema": "stage12b_observation_collection_v1",
            "status": "complete_truth_generation_process_closed",
            "created_at_utc": utc_now(),
            "case_ids": list(CASE_IDS),
            "case_manifests": case_records,
            "property_prior_bank_manifest": file_record(
                EXPERIMENT_DIR / "prior_bank/manifest.json", relative_to=REPOSITORY_ROOT
            ),
            "property_prior_frozen_before_observation": True,
            "geological_truth_available_to_downstream_inference": False,
            "noise_type": "none",
            "trace_grid": "all_xy_columns",
        },
    )
    print("Stage12B truth-generation process closed; downstream assets contain no truth")


if __name__ == "__main__":
    main()
