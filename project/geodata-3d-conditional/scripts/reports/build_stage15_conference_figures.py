#!/usr/bin/env python3
"""Build the two compact figures used by the Stage15 conference manuscript.

The truth and generated label9 bodies are deliberately rendered as complete,
opaque surfaces.  No categorical cutaway or retrospective ghost overlay is
used, avoiding the visual truncation seen in earlier exploratory figures.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime

ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
OUTPUT = ROOT / "reports/conference_paper_v1"
SEEDS = (42, 142, 242)
PHASE1 = (
    PROJECT_DIR
    / "experiments/stage1_probability/runs/cond_generation_0/label9/all/phase1b_v4"
    / "calibrated_reference_windowed/seed42_n4_s32"
)
PHASE2 = (
    PROJECT_DIR
    / "experiments/stage2_property/runs/cond_generation_0"
    / "ideal_density_susceptibility_label9_contrast_v1/phase2a_v1"
    / "seed42_n4_s32_a025_c025"
)


def _load(path: Path) -> torch.Tensor:
    return runtime.load_tensor(path, map_location="cpu")


def _figure_geophysics(
    output: Path,
    observed: torch.Tensor,
    score: torch.Tensor,
    boundary: torch.Tensor,
    truth9: torch.Tensor,
) -> None:
    import matplotlib.pyplot as plt

    seismic = observed[0, 0].numpy()
    q = score[0, 0].numpy()
    edge = boundary[0, 0].numpy()
    target = truth9[0, 0].numpy()
    truth_footprint = target.any(axis=2)
    # Section locations are selected by the truth-blind inversion volume.
    # Truth contours are retrospective overlays only.
    x_index = int(np.argmax(q.sum(axis=(1, 2))))
    y_index = int(np.argmax(q.sum(axis=(0, 2))))

    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.1), constrained_layout=True)
    panels = []
    panels.append(axes[0, 0].imshow(seismic[:, y_index, :].T, aspect="auto", cmap="seismic", origin="upper"))
    axes[0, 0].set(xlabel="x", ylabel="time")
    panels.append(axes[0, 1].imshow(np.sqrt(np.mean(seismic**2, axis=2)).T, origin="lower", cmap="magma"))
    axes[0, 1].set(xlabel="x", ylabel="y")
    panels.append(axes[0, 2].imshow(q.max(axis=2).T, origin="lower", cmap="viridis", vmin=0, vmax=1))
    axes[0, 2].contour(truth_footprint.T, levels=[0.5], colors="white", linewidths=0.9)
    axes[0, 2].set(xlabel="x", ylabel="y")
    panels.append(axes[1, 0].imshow(q[x_index].T, origin="lower", cmap="viridis", vmin=0, vmax=1))
    axes[1, 0].contour(target[x_index].T, levels=[0.5], colors="white", linewidths=0.9)
    axes[1, 0].set(xlabel="y", ylabel="z")
    panels.append(axes[1, 1].imshow(q[:, y_index, :].T, origin="lower", cmap="viridis", vmin=0, vmax=1))
    axes[1, 1].contour(target[:, y_index, :].T, levels=[0.5], colors="white", linewidths=0.9)
    axes[1, 1].set(xlabel="x", ylabel="z")
    panels.append(axes[1, 2].imshow(edge.max(axis=2).T, origin="lower", cmap="inferno", vmin=0, vmax=1))
    axes[1, 2].contour(truth_footprint.T, levels=[0.5], colors="cyan", linewidths=0.9)
    axes[1, 2].set(xlabel="x", ylabel="y")

    letters = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    for ax, im, letter in zip(axes.flat, panels, letters):
        ax.text(0.02, 0.96, letter, transform=ax.transAxes, va="top", ha="left", fontsize=10, fontweight="bold", color="black", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5})
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    fig.savefig(output, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _figure_flow(output: Path, truth: torch.Tensor, score: torch.Tensor) -> None:
    import matplotlib.pyplot as plt
    from scripts.paper_figures.style import (
        render_categorical_volume_3d,
        render_target_constraints_panel,
        show_render,
    )

    truth_np = runtime.normalize_single_geology(truth, "truth")[0, 0].numpy()
    target = truth_np == 9
    run_root = ROOT / "trace_boundary/flow_property_seed42_v4"
    stage15_guided = runtime.normalize_single_geology(
        _load(run_root / "guided/sample_0.pt"), "Stage15 guided seed42"
    )[0, 0].numpy()

    phase1_baseline = runtime.normalize_single_geology(
        _load(PHASE1 / "baseline/sample_0.pt"), "Phase1 baseline"
    )[0, 0].numpy()
    phase1_guided = runtime.normalize_single_geology(
        _load(PHASE1 / "alpha025/sample_0.pt"), "Phase1 guided"
    )[0, 0].numpy()
    phase2_baseline = runtime.normalize_single_geology(
        _load(PHASE2 / "baseline/sample_0.pt"), "Phase2 baseline"
    )[0, 0].numpy()
    phase2_guided = runtime.normalize_single_geology(
        _load(PHASE2 / "alpha025/sample_0.pt"), "Phase2 guided"
    )[0, 0].numpy()
    if not np.array_equal(phase1_baseline, phase2_baseline):
        raise ValueError("Phase1 and Phase2 paired Flow-only samples differ")
    phase1_config = json.loads((PHASE1 / "baseline/config.json").read_text(encoding="utf-8"))
    well_xy = tuple(
        tuple(int(value) for value in pair)
        for pair in phase1_config["conditioning_report"]["full_borehole_xy"]
    )

    boreholes = runtime.normalize_single_geology(
        _load(PROJECT_DIR / "samples/jupyter-demo/cond_generation_0/boreholes.pt"),
        "boreholes",
    )[0, 0].numpy()
    condition_mask = boreholes != -1

    top_images = [
        render_categorical_volume_3d(
            volume,
            borehole_xy=well_xy,
            condition_mask=condition_mask,
            context_opacity=1.0,
            target_opacity=1.0,
            window_size=(920, 760),
        )
        for volume in (truth_np, phase1_baseline, phase1_guided, phase2_guided, stage15_guided)
    ]
    constraint_image = render_categorical_volume_3d(
        boreholes,
        borehole_xy=well_xy,
        context_opacity=0.94,
        target_opacity=1.0,
        cutaway=False,
        window_size=(920, 760),
    )
    target_images = [
        render_target_constraints_panel(
            volume == 9,
            target,
            boreholes,
            well_xy=well_xy,
            window_size=(920, 760),
        )
        for volume in (phase1_baseline, phase1_guided, phase2_guided, stage15_guided)
    ]
    rendered = top_images + [constraint_image] + target_images
    titles = [
        "(a) Ture",
        "(b) CFM",
        "(c) Phase1",
        "(d) Phase2",
        "(e) Phase3",
        "(f) Surface & wells",
        "(g) CFM · label9",
        "(h) Phase1 · label9",
        "(i) Phase2 · label9",
        "(j) Phase3 · label9",
    ]
    fig, axes = plt.subplots(2, 5, figsize=(13.8, 6.1), constrained_layout=True)
    for ax, image, title in zip(axes.flat, rendered, titles):
        show_render(ax, image)
        ax.set_title(title, loc="left", fontsize=9.2, pad=1.5)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _figure_prior_limit(output: Path) -> None:
    import matplotlib.pyplot as plt
    from scripts.paper_figures.style import (
        render_target_error_panel,
        show_render,
    )

    truth = runtime.normalize_single_geology(
        _load(PROJECT_DIR / "samples/jupyter-demo/cond_generation_0/true_model.pt"), "truth"
    )[0, 0].numpy()
    baseline = runtime.normalize_single_geology(
        _load(PHASE1 / "baseline/sample_0.pt"), "baseline"
    )[0, 0].numpy()
    guided = [
        runtime.normalize_single_geology(_load(PHASE1 / "alpha025/sample_0.pt"), "Phase1")[0, 0].numpy(),
        runtime.normalize_single_geology(_load(PHASE2 / "alpha025/sample_0.pt"), "Phase2")[0, 0].numpy(),
        runtime.normalize_single_geology(
            _load(ROOT / "trace_boundary/flow_property_seed42_v4/guided/sample_0.pt"), "Stage15-H"
        )[0, 0].numpy(),
    ]
    phase1_config = json.loads((PHASE1 / "baseline/config.json").read_text(encoding="utf-8"))
    well_xy = tuple(tuple(int(v) for v in xy) for xy in phase1_config["conditioning_report"]["full_borehole_xy"])
    target, base = truth == 9, baseline == 9
    top = [
        render_target_error_panel(target, base, base, well_xy=well_xy, mode="omission", window_size=(1000, 800)),
        *[render_target_error_panel(target, base, value == 9, well_xy=well_xy, mode="omission", window_size=(1000, 800)) for value in guided],
    ]
    bottom = [
        render_target_error_panel(target, base, base, well_xy=well_xy, mode="commission", window_size=(1000, 800)),
        *[render_target_error_panel(target, base, value == 9, well_xy=well_xy, mode="commission", window_size=(1000, 800)) for value in guided],
    ]
    titles = (
        "(a) Baseline omissions",
        "(b) Phase1 · 97.0% recovered",
        "(c) Phase2 · 48.4% recovered",
        "(d) Seismic · 42.4% recovered",
        "(e) Baseline commissions",
        "(f) Phase1 · 73.2% removed",
        "(g) Phase2 · 92.1% removed",
        "(h) Seismic · 1.1% removed",
    )
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.25), constrained_layout=True)
    for ax, image, title in zip(axes.flat, top + bottom, titles):
        show_render(ax, image)
        ax.set_title(title, loc="left", fontsize=9.4, pad=1.5)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _figure_five_body(output: Path, *, seed: int = 142) -> None:
    """Render the five-body diagnostic with the same wells in every panel."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scripts.paper_figures.style import (
        LABEL9_COLOR,
        OBSERVATION_COLOR,
        TRUTH_OUTLINE_COLOR,
        _add_bounding_box,
        _add_surface,
        _import_pyvista,
        _new_plotter,
        _set_camera,
        binary_surface,
        show_render,
    )

    case = PROJECT_DIR / "experiments/stage15_five_body_flow/cases_v1/FIVE_BODY"
    runs = PROJECT_DIR / "experiments/stage15_five_body_flow/runs_v1"
    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    well_xy = tuple(tuple(int(v) for v in xy) for xy in manifest["conditioning_report"]["full_borehole_xy"])
    truth = _load(case / "truth_restricted/binary_truth.pt").bool()[0, 0].numpy()
    conditions = _load(case / "condition_values.pt").long()[0, 0].numpy()
    target_wells = {
        tuple(int(v) for v in body["well_xy"])
        for body in manifest["bodies"]
        if body["role"] == "drilled"
    }
    volumes = [
        truth,
        _load(runs / "FLOW_ONLY/FIVE_BODY" / f"seed_{seed}.pt").long().numpy() == 9,
        _load(runs / "SEISMIC_GUIDED/FIVE_BODY" / f"seed_{seed}.pt").long().numpy() == 9,
    ]

    def render(mask: np.ndarray, *, truth_panel: bool) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        plotter = _new_plotter((1050, 820))
        if not truth_panel:
            _add_surface(plotter, binary_surface(truth), TRUTH_OUTLINE_COLOR, 0.13)
        _add_surface(plotter, binary_surface(mask), LABEL9_COLOR, 0.92)
        pv = _import_pyvista()
        for x, y in well_xy:
            line = pv.Line((float(x) + 0.5, float(y) + 0.5, 0.0), (float(x) + 0.5, float(y) + 0.5, 64.0))
            is_target_well = (int(x), int(y)) in target_wells
            _add_surface(
                plotter,
                line.tube(radius=0.43 if is_target_well else 0.22),
                "#B2185B" if is_target_well else OBSERVATION_COLOR,
                0.98 if is_target_well else 0.78,
            )
        hits = np.argwhere(conditions == 9).astype(float) + 0.5
        if hits.size:
            plotter.add_points(
                hits,
                color="#B2185B",
                point_size=13,
                render_points_as_spheres=True,
            )
        _add_bounding_box(plotter, mask.shape)
        camera = {
            "position_direction": [1.20, -1.55, 1.05],
            "focal_point_fraction": [0.50, 0.50, 0.42],
            "view_up": [0.0, 0.0, 1.0],
            "parallel_projection": True,
            "zoom": 1.18,
            "cut_fraction": 0.52,
        }
        _set_camera(plotter, mask.shape, camera)
        image = plotter.screenshot(return_img=True)
        plotter.close()
        return np.asarray(image)[..., :3]

    images = [render(volume, truth_panel=index == 0) for index, volume in enumerate(volumes)]
    titles = (
        "(a) Ture",
        "(B) CFM",
        "(c) Seismic-guided",
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.75), constrained_layout=True)
    for ax, image, title in zip(axes, images, titles):
        show_render(ax, image)
        ax.set_title(title, loc="left", fontsize=9.6, pad=2.0)
    legend = [
        Patch(facecolor=LABEL9_COLOR, edgecolor="none", label="generated / truth label9"),
        Patch(facecolor=TRUTH_OUTLINE_COLOR, alpha=0.25, edgecolor="none", label="truth reference"),
        Line2D([0], [0], color=OBSERVATION_COLOR, lw=2.5, label="background borehole"),
        Line2D([0], [0], color="#B2185B", lw=3.2, label="label9-intersecting borehole"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=8.2, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    truth = _load(PROJECT_DIR / "samples/jupyter-demo/cond_generation_0/true_model.pt").long()
    support = _load(ROOT / "observations/cond_generation_0/subsurface_mask.pt").bool()
    observed = _load(ROOT / "observations/cond_generation_0/observed_seismic.pt").float()
    score = _load(ROOT / "trace_boundary/cond_generation_0_v1/binary_impedance_score.pt").float()
    boundary = _load(ROOT / "trace_boundary/cond_generation_0_v1/vertical_boundary_strength.pt").float()
    truth9 = (truth == 9) & support
    _figure_flow(OUTPUT / "figure1_progressive_guidance.png", truth, score)
    _figure_five_body(OUTPUT / "figure2_five_body_geophysical_guidance.png")


if __name__ == "__main__":
    main()
