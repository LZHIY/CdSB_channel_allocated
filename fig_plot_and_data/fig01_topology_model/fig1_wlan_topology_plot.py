# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle


def find_project_root(start: Path) -> Path:
    """Find the project root regardless of where this script is placed."""
    for candidate in (start, *start.parents):
        if (
            (candidate / "configs" / "default_config.yaml").exists()
            and (candidate / "core" / "environment.py").exists()
        ):
            return candidate
    raise FileNotFoundError(
        f"Cannot locate project root from {start}. Expected configs/default_config.yaml "
        "and core/environment.py."
    )


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "result_picture"  # every generated image goes here
IEEE_FULL_WIDTH_IN = 7.16
IEEE_OUTPUT_DPI = 600

# Manuscript Fig. 1 shows a single deployment, so only this topology is rendered:
# 20 APs x 10 UEs/AP = 200 UEs in the 150 m x 150 m area (seeds from the config).
TARGET_NUM_AP = 20

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.environment import WLANIsingModel


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the YAML config used by the benchmark scripts."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_timestamped_output_dir(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir = output_root / f"{timestamp}_fig01_topology_model"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def build_model_from_config(config: dict, num_ap: int) -> WLANIsingModel:
    env_config = config.get("environment", {})
    return WLANIsingModel(
        num_ap=num_ap,
        num_ch=env_config.get("num_ch", 3),
        num_ue_per_ap=env_config.get("num_ue_per_ap", 5),
        ap_seed=env_config.get("ap_seed"),
        ue_seed=env_config.get("ue_seed"),
        max_dist=env_config.get("max_dist", 150),
        carrier_sense_range_m=env_config.get("carrier_sense_range_m", 70.0),
    )


def finite_ue_positions(model: WLANIsingModel, ap_idx: int) -> np.ndarray:
    ue_pos = model.UE_POSITIONS[ap_idx]
    return ue_pos[np.isfinite(ue_pos).all(axis=1)]


def all_finite_ue_positions(model: WLANIsingModel) -> np.ndarray:
    mask = np.isfinite(model.UE_POSITIONS).all(axis=2)
    return model.UE_POSITIONS[mask]


def select_example_ap(model: WLANIsingModel) -> Optional[int]:
    counts = getattr(model, "UE_COUNTS_PER_AP", None)
    if counts is None:
        return None
    candidates = np.flatnonzero(np.asarray(counts) > 0)
    if candidates.size == 0:
        return None

    center = np.array([model.MAX_DIST / 2.0, model.MAX_DIST / 2.0])
    distances = np.linalg.norm(model.AP_POSITIONS[candidates] - center, axis=1)
    return int(candidates[np.argmin(distances)])


def apply_ieee_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 18,
            "axes.labelsize": 22,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "lines.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def visualize_wlan_topology_paper_style(
    model: WLANIsingModel,
    ap_seed: Optional[int] = None,
    ue_seed: Optional[int] = None,
    figsize: tuple = (IEEE_FULL_WIDTH_IN, IEEE_FULL_WIDTH_IN),
    show_carrier_sense: bool = True,
    show_simulation_box: bool = True,
    save_path: Optional[Path | str] = None,
    dpi: int = IEEE_OUTPUT_DPI,
    show: bool = False,
    save_pdf: bool = True,
    highlight_example: bool = False,
    show_all_carrier_sense: bool = True,
) -> None:
    """Visualize a WLAN AP-UE topology using an IEEE Access friendly style."""
    _ = (ap_seed, ue_seed)
    apply_ieee_figure_style()

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    cmap = plt.get_cmap("tab20", model.NUM_AP)
    sense_color = "#7a7a7a"
    highlight_color = "#c62828"

    total_ue = int(np.isfinite(model.UE_POSITIONS).all(axis=2).sum())
    ap_size = 58 if model.NUM_AP <= 40 else 44 if model.NUM_AP <= 60 else 34
    ue_size = 12 if total_ue <= 500 else 8
    example_ap = select_example_ap(model) if highlight_example else None

    if show_carrier_sense:
        carrier_sense_indices = range(model.NUM_AP)
        if not show_all_carrier_sense and example_ap is not None:
            carrier_sense_indices = [example_ap]
        for ap_idx in carrier_sense_indices:
            ap_x, ap_y = model.AP_POSITIONS[ap_idx]
            is_example = example_ap == ap_idx
            ap_group_color = cmap(ap_idx)
            circle = Circle(
                (ap_x, ap_y),
                model.CARRIER_SENSE_RANGE_M,
                facecolor=highlight_color if is_example else ap_group_color,
                alpha=0.08 if is_example else 0.11,
                edgecolor=highlight_color if is_example else ap_group_color,
                linestyle="--",
                linewidth=0.65 if is_example else 0.45,
                zorder=1,
            )
            ax.add_patch(circle)

    for ap_idx in range(model.NUM_AP):
        ue_positions = finite_ue_positions(model, ap_idx)
        if ue_positions.size == 0:
            continue
        ax.scatter(
            ue_positions[:, 0],
            ue_positions[:, 1],
            color=cmap(ap_idx),
            s=ue_size,
            alpha=0.78,
            edgecolors="none",
            zorder=3,
        )

    ax.scatter(
        model.AP_POSITIONS[:, 0],
        model.AP_POSITIONS[:, 1],
        color=[cmap(i) for i in range(model.NUM_AP)],
        s=ap_size,
        marker="^",
        edgecolors="black",
        linewidth=0.65,
        zorder=4,
    )

    if show_simulation_box:
        box = Rectangle(
            (0, 0),
            model.MAX_DIST,
            model.MAX_DIST,
            fill=False,
            edgecolor="black",
            linewidth=0.9,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(
            model.MAX_DIST / 2,
            -0.07 * model.MAX_DIST,
            f"{model.MAX_DIST:g} x {model.MAX_DIST:g} m",
            ha="center",
            va="top",
            fontsize=22,
            fontweight="normal",
            color="black",
            zorder=6,
        )

    if example_ap is not None:
        example_ues = finite_ue_positions(model, example_ap)
        ap_xy = model.AP_POSITIONS[example_ap]
        for ue_xy in example_ues:
            ax.plot(
                [ap_xy[0], ue_xy[0]],
                [ap_xy[1], ue_xy[1]],
                color=highlight_color,
                linewidth=0.45,
                alpha=0.58,
                zorder=5,
            )
        if example_ues.size:
            ax.scatter(
                example_ues[:, 0],
                example_ues[:, 1],
                color=highlight_color,
                s=ue_size + 8,
                edgecolors="white",
                linewidth=0.25,
                zorder=7,
            )
        ax.scatter(
            [ap_xy[0]],
            [ap_xy[1]],
            color=highlight_color,
            s=ap_size + 28,
            marker="^",
            edgecolors="black",
            linewidth=0.75,
            zorder=8,
        )

    ax.set_xlabel("x-coordinate (m)", fontsize=22, fontweight="bold")
    ax.set_ylabel("y-coordinate (m)", fontsize=22, fontweight="bold")

    margin = max(model.CARRIER_SENSE_RANGE_M * 0.85, model.MAX_DIST * 0.08)
    ax.set_xlim(-margin, model.MAX_DIST + margin)
    ax.set_ylim(-margin, model.MAX_DIST + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d6d6d6", alpha=0.45, linestyle="-", linewidth=0.35)
    ax.tick_params(direction="in", length=3, width=0.7, labelsize=18)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("normal")
    carrier_legend_color = sense_color if show_all_carrier_sense else highlight_color

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#b3b3b3",
            markeredgecolor="black",
            markeredgewidth=0.65,
            markersize=6.5,
            label="AP",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#7f7f7f",
            markersize=4.5,
            label="UE",
        ),
        Line2D(
            [0],
            [0],
            color=carrier_legend_color,
            linestyle="--",
            linewidth=0.8,
            label="Carrier-sense range",
        ),
    ]
    if example_ap is not None:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=highlight_color,
                marker="o",
                markerfacecolor=highlight_color,
                markeredgecolor="white",
                linewidth=0.8,
                markersize=4.5,
                label="Example AP-UE association",
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        ncol=1,
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.45,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved topology figure: {save_path}")
        if save_pdf and save_path.suffix.lower() != ".pdf":
            pdf_path = save_path.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
            print(f"Saved topology figure: {pdf_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    config = load_config()
    output_dir = make_timestamped_output_dir()

    env_config = config.get("environment", {})
    ap_seed = env_config.get("ap_seed")
    ue_seed = env_config.get("ue_seed")
    # Only the deployment used by Fig. 1 is rendered (previously this iterated over
    # experiment.ap_list, which produced one figure per AP count).
    ap_list = [TARGET_NUM_AP]

    for num_ap in ap_list:
        model = build_model_from_config(config, num_ap=int(num_ap))
        save_path = output_dir / f"wlan_topology_ap{model.NUM_AP}_apseed{model.AP_SEED}_ueseed{model.UE_SEED}_ieee.png"
        visualize_wlan_topology_paper_style(
            model,
            ap_seed=ap_seed,
            ue_seed=ue_seed,
            figsize=(IEEE_FULL_WIDTH_IN, IEEE_FULL_WIDTH_IN),
            show_carrier_sense=True,
            show_simulation_box=True,
            save_path=save_path,
            dpi=IEEE_OUTPUT_DPI,
            show=False,
        )

    print("\n=== Current topology config ===")
    print(f"Config file: {DEFAULT_CONFIG_PATH}")
    print(f"Output folder: {output_dir}")
    print(f"AP list: {ap_list}")
    print(f"Channels: {env_config.get('num_ch', 3)}")
    print(f"UE generated per AP: {env_config.get('num_ue_per_ap', 5)}")
    print(f"AP seed: {ap_seed}")
    print(f"UE seed: {ue_seed}")
    print(f"Simulation box: {env_config.get('max_dist', 150)} x {env_config.get('max_dist', 150)} m")
    print(f"Carrier sense range: {env_config.get('carrier_sense_range_m', 70.0)} m")


if __name__ == "__main__":
    main()
