"""Shared I/O helpers for dike guidance demo post-processing.

The helpers in this file intentionally work only with saved tensors and table
outputs. They do not import the training module, instantiate a model, or change
the inference-time guidance formula.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch


DEFAULT_SAMPLE_PREFIXES = ("sample", "sol", "run")


def numeric_suffix(path: Path) -> int:
    """Return the trailing integer in names like sample_12.pt, or -1."""
    match = re.search(r"_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def sample_id_from_path(path: Path, fallback: int) -> int:
    suffix = numeric_suffix(path)
    return suffix if suffix >= 0 else fallback


def find_sample_files(
    samples_dir: Path,
    prefixes: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Find saved realization tensors under a directory."""
    if not samples_dir.exists():
        raise FileNotFoundError(f"samples directory not found: {samples_dir}")
    search_prefixes = tuple(prefixes or DEFAULT_SAMPLE_PREFIXES)
    paths: List[Path] = []
    for prefix in search_prefixes:
        paths.extend(samples_dir.glob(f"{prefix}_*.pt"))
    unique = sorted(
        set(paths),
        key=lambda path: (
            path.stem.rsplit("_", 1)[0],
            numeric_suffix(path),
            path.name,
        ),
    )
    if not unique:
        searched = ", ".join(f"{prefix}_*.pt" for prefix in search_prefixes)
        raise FileNotFoundError(f"no files matching {searched} found in {samples_dir}")
    return unique


def load_tensor(path: Path, device: str = "cpu") -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"tensor file not found: {path}")
    value = torch.load(path, map_location=device)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected a tensor in {path}, got {type(value).__name__}")
    return value


def as_batched_volume(volume: torch.Tensor, source: object = "volume") -> torch.Tensor:
    """Normalize geology/density tensors to [B, 1, X, Y, Z]."""
    if not isinstance(volume, torch.Tensor):
        raise TypeError(f"{source} must be a torch.Tensor")
    if volume.dim() == 3:
        return volume.unsqueeze(0).unsqueeze(0)
    if volume.dim() == 4:
        return volume.unsqueeze(0) if volume.shape[0] == 1 else volume.unsqueeze(1)
    if volume.dim() == 5 and volume.shape[1] == 1:
        return volume
    raise ValueError(
        f"unsupported shape {tuple(volume.shape)} in {source}; expected "
        "[X,Y,Z], [1,X,Y,Z], [B,X,Y,Z], or [B,1,X,Y,Z]"
    )


def as_single_volume(volume: torch.Tensor, source: object = "volume") -> torch.Tensor:
    """Normalize one geology tensor to [1, 1, X, Y, Z]."""
    batched = as_batched_volume(volume, source)
    if batched.shape[0] != 1:
        raise ValueError(f"{source} must contain exactly one volume, got batch {batched.shape[0]}")
    return batched


def load_volume(path: Path, device: str = "cpu", single: bool = False) -> torch.Tensor:
    value = load_tensor(path, device=device)
    return as_single_volume(value, path) if single else as_batched_volume(value, path)


def load_sample_stack(
    paths: Iterable[Path],
    device: str = "cpu",
) -> tuple[torch.Tensor, List[Dict[str, object]]]:
    """Load sample files and concatenate them to [B, 1, X, Y, Z]."""
    batches = []
    records: List[Dict[str, object]] = []
    expected_shape = None
    offset = 0
    for file_index, path in enumerate(paths):
        batch = load_volume(path, device=device)
        if expected_shape is None:
            expected_shape = tuple(batch.shape[1:])
        elif tuple(batch.shape[1:]) != expected_shape:
            raise ValueError(
                f"shape mismatch in {path}: got {tuple(batch.shape[1:])}, "
                f"expected {expected_shape}"
            )
        base_id = sample_id_from_path(path, file_index)
        for batch_index in range(batch.shape[0]):
            records.append(
                {
                    "sample_id": base_id + batch_index if batch.shape[0] > 1 else base_id,
                    "path": str(path),
                    "batch_index": batch_index,
                    "stack_index": offset + batch_index,
                }
            )
        offset += batch.shape[0]
        batches.append(batch)
    if not batches:
        raise ValueError("at least one sample path is required")
    return torch.cat(batches, dim=0), records


def label_mask(volume: torch.Tensor, target_label: int) -> torch.Tensor:
    """Return a boolean [B, 1, X, Y, Z] mask for a categorical label."""
    return as_batched_volume(volume).long() == int(target_label)


def target_probability(
    realizations: torch.Tensor,
    target_label: int,
) -> torch.Tensor:
    """Return target-label probability volume [1, 1, X, Y, Z]."""
    masks = label_mask(realizations, target_label).float()
    return masks.mean(dim=0, keepdim=True)


def density_value(label: int, properties: Optional[Mapping[int, float]] = None) -> float:
    """Return the default lightweight proxy density value for a categorical label."""
    if properties is None:
        from geophysics import LithologyPropertyMap

        properties = LithologyPropertyMap.DEFAULT_DENSITY_CONTRASTS
    return float(properties.get(int(label), 0.0))


def dominant_label(volume: torch.Tensor, exclude: Sequence[int] = (-1,)) -> int:
    labels, counts = torch.unique(as_batched_volume(volume).long(), return_counts=True)
    excluded = {int(value) for value in exclude}
    pairs = [
        (int(label.item()), int(count.item()))
        for label, count in zip(labels, counts)
        if int(label.item()) not in excluded
    ]
    if not pairs:
        raise ValueError("no labels remain after exclusions")
    return max(pairs, key=lambda item: item[1])[0]


def local_replacement_label(
    volume: torch.Tensor,
    target_mask: torch.Tensor,
    target_label: int,
    ignore_labels: Sequence[int] = (-1,),
) -> tuple[int, str]:
    """Choose the most common non-target label around a target mask."""
    labels = as_single_volume(volume, "volume").long()[0, 0]
    mask = target_mask.detach().to(device=labels.device).bool()
    if mask.dim() == 5:
        mask = mask[0, 0]
    elif mask.dim() == 4:
        mask = mask[0]
    if mask.shape != labels.shape:
        raise ValueError("target_mask and volume must have matching spatial shape")

    neighbor = torch.zeros_like(mask)
    neighbor[1:, :, :] |= mask[:-1, :, :]
    neighbor[:-1, :, :] |= mask[1:, :, :]
    neighbor[:, 1:, :] |= mask[:, :-1, :]
    neighbor[:, :-1, :] |= mask[:, 1:, :]
    neighbor[:, :, 1:] |= mask[:, :, :-1]
    neighbor[:, :, :-1] |= mask[:, :, 1:]
    neighbor &= ~mask

    excluded = {int(target_label), *(int(value) for value in ignore_labels)}
    local_values = labels[neighbor]
    if local_values.numel() > 0:
        unique, counts = torch.unique(local_values, return_counts=True)
        pairs = [
            (int(label.item()), int(count.item()))
            for label, count in zip(unique, counts)
            if int(label.item()) not in excluded
        ]
        if pairs:
            return max(pairs, key=lambda item: item[1])[0], "local_neighbor_mode"
    return dominant_label(labels, exclude=tuple(excluded)), "global_dominant_fallback"


def finite_stats(values: Sequence[float]) -> Dict[str, object]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return {
        "count": len(finite),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(finite),
        "max": max(finite),
    }


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_density_config(path: Optional[Path]) -> Optional[Dict[str, object]]:
    """Load a controlled lightweight gravity-proxy density config."""
    if path is None:
        return None
    config = read_json(path)
    densities = config.get("densities")
    if not isinstance(densities, Mapping):
        raise ValueError(f"density config must contain a 'densities' object: {path}")
    parsed = {int(label): float(value) for label, value in densities.items()}
    config = dict(config)
    config["densities"] = parsed
    if "default_density" in config and config["default_density"] is not None:
        config["default_density"] = float(config["default_density"])
    if "target_label" in config and config["target_label"] is not None:
        config["target_label"] = int(config["target_label"])
    return config


def load_susceptibility_config(path: Optional[Path]) -> Optional[Dict[str, object]]:
    """Load a controlled lightweight magnetic-proxy susceptibility config."""
    if path is None:
        return None
    config = read_json(path)
    values = config.get("susceptibilities", config.get("densities"))
    if not isinstance(values, Mapping):
        raise ValueError(
            f"susceptibility config must contain a 'susceptibilities' object: {path}"
        )
    parsed = {int(label): float(value) for label, value in values.items()}
    config = dict(config)
    config["susceptibilities"] = parsed
    if "default_susceptibility" in config and config["default_susceptibility"] is not None:
        config["default_susceptibility"] = float(config["default_susceptibility"])
    if "target_label" in config and config["target_label"] is not None:
        config["target_label"] = int(config["target_label"])
    return config


def property_map_from_density_config(config: Optional[Mapping[str, object]]):
    """Build LithologyPropertyMap from an optional density config."""
    from geophysics import LithologyPropertyMap

    if config is None:
        return LithologyPropertyMap()
    densities = config.get("densities")
    if not isinstance(densities, Mapping):
        raise ValueError("density config must contain a 'densities' object")
    return LithologyPropertyMap(
        properties={int(label): float(value) for label, value in densities.items()},
        default_value=float(config.get("default_density", 0.0)),
    )


def property_map_from_susceptibility_config(config: Optional[Mapping[str, object]]):
    """Build LithologyPropertyMap from an optional susceptibility config."""
    from geophysics import LithologyPropertyMap

    if config is None:
        return LithologyPropertyMap(
            properties={label: value * 0.01 for label, value in LithologyPropertyMap.DEFAULT_DENSITY_CONTRASTS.items()},
            default_value=0.0,
        )
    values = config.get("susceptibilities")
    if not isinstance(values, Mapping):
        raise ValueError("susceptibility config must contain a 'susceptibilities' object")
    return LithologyPropertyMap(
        properties={int(label): float(value) for label, value in values.items()},
        default_value=float(config.get("default_susceptibility", 0.0)),
    )


def density_config_metadata(config: Optional[Mapping[str, object]]) -> Dict[str, object]:
    if config is None:
        return {"density_config": None, "density_config_description": "default LithologyPropertyMap"}
    densities = config.get("densities", {})
    target_label = config.get("target_label")
    target_density = None
    if target_label is not None and isinstance(densities, Mapping):
        target_density = densities.get(int(target_label))
    return {
        "density_config": config.get("name", "custom_density_config"),
        "density_config_description": config.get("description", ""),
        "target_label": target_label,
        "target_density": target_density,
        "default_density": config.get("default_density", 0.0),
    }


def susceptibility_config_metadata(config: Optional[Mapping[str, object]]) -> Dict[str, object]:
    if config is None:
        return {"susceptibility_config": None, "susceptibility_config_description": "default susceptibility proxy map"}
    values = config.get("susceptibilities", {})
    target_label = config.get("target_label")
    target_susceptibility = None
    if target_label is not None and isinstance(values, Mapping):
        target_susceptibility = values.get(int(target_label))
    return {
        "susceptibility_config": config.get("name", "custom_susceptibility_config"),
        "susceptibility_config_description": config.get("description", ""),
        "target_label": target_label,
        "target_susceptibility": target_susceptibility,
        "default_susceptibility": config.get("default_susceptibility", 0.0),
    }


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def rows_by_sample_id(rows: Sequence[Mapping[str, object]]) -> Dict[int, Mapping[str, object]]:
    indexed: Dict[int, Mapping[str, object]] = {}
    for row in rows:
        if "sample_id" not in row or row["sample_id"] in ("", None):
            continue
        indexed[int(row["sample_id"])] = row
    return indexed


def target_component_stats(mask: torch.Tensor) -> Dict[str, object]:
    """Return connected-component stats for one target mask."""
    mask_3d = mask.detach().cpu().bool()
    if mask_3d.dim() == 5:
        mask_3d = mask_3d[0, 0]
    elif mask_3d.dim() == 4:
        mask_3d = mask_3d[0]
    if mask_3d.dim() != 3:
        raise ValueError("target_component_stats expects a 3D target mask")
    components = connected_components_3d(mask_3d)
    total = int(mask_3d.sum().item())
    largest = max((int(component["voxel_count"]) for component in components), default=0)
    return {
        "target_connected_components": len(components),
        "largest_component_voxels": largest,
        "largest_component_fraction": largest / total if total > 0 else float("nan"),
    }


def infer_paired_by_seed(baseline_dir: Path, guided_dir: Path) -> tuple[bool, str]:
    """Conservatively infer whether sample_i files likely share initial X0 seeds."""
    baseline_config = baseline_dir / "config.json"
    guided_config = guided_dir / "config.json"
    if not baseline_config.exists() or not guided_config.exists():
        return False, "missing baseline or guided config.json"
    baseline = read_json(baseline_config)
    guided = read_json(guided_config)
    required_equal = ("truth_model", "boreholes", "seed", "n_samples", "kernel_size")
    for key in required_equal:
        if baseline.get(key) != guided.get(key):
            return False, f"config field differs: {key}"
    if baseline.get("guidance_mode") != guided.get("guidance_mode"):
        return False, "guidance_mode differs"
    mode = str(guided.get("guidance_mode", ""))
    if mode not in {"absolute", "relative"}:
        return False, "missing or unsupported guidance_mode"
    baseline_alpha = baseline.get("alpha") if baseline.get("alpha") is not None else 1.0
    baseline_mu = baseline.get("mu") if baseline.get("mu") is not None else 1.0
    if mode == "relative" and float(baseline_alpha) != 0.0:
        return False, "relative baseline alpha is not zero"
    if mode == "absolute" and float(baseline_mu) != 0.0:
        return False, "absolute baseline mu is not zero"
    return True, "matching guided-sampler config with zero-guidance baseline"


def connected_components_3d(mask: torch.Tensor) -> List[Dict[str, object]]:
    """Return 6-connected components for one boolean [X, Y, Z] mask."""
    mask_3d = mask.detach().cpu().bool()
    if mask_3d.dim() != 3:
        raise ValueError("connected_components_3d expects a [X,Y,Z] mask")
    visited = torch.zeros_like(mask_3d, dtype=torch.bool)
    components: List[Dict[str, object]] = []
    sizes = mask_3d.shape
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    starts = torch.nonzero(mask_3d, as_tuple=False)
    for start_tensor in starts:
        start = tuple(int(value) for value in start_tensor.tolist())
        if visited[start]:
            continue
        queue: deque[tuple[int, int, int]] = deque([start])
        visited[start] = True
        coords = []
        while queue:
            current = queue.popleft()
            coords.append(current)
            for dx, dy, dz in neighbors:
                nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
                if (
                    0 <= nxt[0] < sizes[0]
                    and 0 <= nxt[1] < sizes[1]
                    and 0 <= nxt[2] < sizes[2]
                    and mask_3d[nxt]
                    and not visited[nxt]
                ):
                    visited[nxt] = True
                    queue.append(nxt)
        coord_tensor = torch.as_tensor(coords, dtype=torch.long)
        mins = coord_tensor.min(dim=0).values
        maxs = coord_tensor.max(dim=0).values
        components.append(
            {
                "component_id": len(components),
                "voxel_count": int(coord_tensor.shape[0]),
                "bbox_min": [int(value) for value in mins.tolist()],
                "bbox_max": [int(value) for value in maxs.tolist()],
                "centroid": [float(value) for value in coord_tensor.float().mean(dim=0).tolist()],
                "coords": coord_tensor,
            }
        )
    return components
