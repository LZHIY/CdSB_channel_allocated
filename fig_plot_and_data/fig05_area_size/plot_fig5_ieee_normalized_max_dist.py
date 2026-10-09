# -*- coding: utf-8 -*-
"""Draw IEEE-style Fig. 5 normalized-throughput max-dist panels.

The script reads ``max_dist_best_solver_summary_plotted.csv`` — the self-contained plotted
table of this figure (its L=150 m rows already hold the values used in the manuscript) —
and saves the 2x2 panel figure plus the four single panels.

Run:
    python fig05_area_size/plot_fig5_ieee_normalized_max_dist.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
# Data folder shipped with this figure: <figure>/data, i.e. side by side with this script.
# SCRIPT_DIR.parent/"data" is kept as a fallback for a <figure>/plot/ layout.
DATA_DIRS = (SCRIPT_DIR / "data", SCRIPT_DIR.parent / "data")


def _find_package_root() -> Path:
    """Folder that holds core/ (the package root, in either layout)."""
    for candidate in (SCRIPT_DIR.parent, SCRIPT_DIR, SCRIPT_DIR.parents[1]):
        if (candidate / "core" / "environment.py").exists():
            return candidate
    return SCRIPT_DIR.parent


# Every generated image goes to <package>/result_picture/<timestamp>_<figure>/
RESULT_PICTURE_DIR = _find_package_root() / "result_picture"
OUTPUT_TAG = "fig05_area_size"
# Self-contained plotted table: its L=150 m rows already carry the numbers used in the
# manuscript, so this figure needs no input from Fig. 2 (see README.md).
SUMMARY_NAME = "max_dist_best_solver_summary_plotted.csv"
OUTPUT_STEM = "fig5_ieee_normalized_max_dist"

SOLVER_ORDER = ["SA", "Greedy", "CIM", "DSB", "BSB", "GBSB", "GDSB"]
DISPLAY_NAMES = {
    "SA": "SA",
    "Greedy": "Greedy",
    "CIM": "CIM",
    "DSB": "dSB",
    "BSB": "bSB",
    "GBSB": "GbSB",
    "GDSB": "CdSB",
}
SOLVER_STYLES = {
    "SA": {"color": "#0072B2", "marker": "o"},
    "Greedy": {"color": "#E69F00", "marker": "o"},
    "CIM": {"color": "#009E73", "marker": "o"},
    "DSB": {"color": "#7A3EB1", "marker": "o"},
    "BSB": {"color": "#56B4E9", "marker": "o"},
    "GBSB": {"color": "#CC79A7", "marker": "o"},
    "GDSB": {"color": "#D55E00", "marker": "*"},
}


def latest_result_dir(base_dir: Path, summary_name: str, *fallback_dirs: Path) -> Path:
    roots = (base_dir, *fallback_dirs)
    candidates = [
        path
        for root in roots
        if root.is_dir()
        for path in root.iterdir()
        if path.is_dir() and (path / summary_name).exists()
    ]
    if not candidates:
        searched = " or ".join(str(root) for root in roots)
        raise FileNotFoundError(
            f"No directory containing {summary_name} found under {searched}. "
            "Pass --result-dir explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def make_timestamped_output_dir(output_root: Path = RESULT_PICTURE_DIR) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir = output_root / f"{timestamp}_{OUTPUT_TAG}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 18,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "axes.linewidth": 1.15,
            "lines.linewidth": 2.25,
            "lines.markersize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def load_fig5_summary(result_dir: Path) -> pd.DataFrame:
    summary_path = result_dir / SUMMARY_NAME
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {SUMMARY_NAME}: {summary_path}")

    df = pd.read_csv(summary_path)
    required = {
        "max_dist",
        "num_ap",
        "solver",
        "theoretical_max_mbps",
        "best_throughput_mbps",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["max_dist"] = pd.to_numeric(df["max_dist"], errors="raise").astype(float)
    df["num_ap"] = pd.to_numeric(df["num_ap"], errors="raise").astype(int)
    df["theoretical_max_mbps"] = pd.to_numeric(
        df["theoretical_max_mbps"], errors="raise"
    )
    df["best_throughput_mbps"] = pd.to_numeric(
        df["best_throughput_mbps"], errors="raise"
    )
    df["best_normalized_throughput"] = (
        df["best_throughput_mbps"] / df["theoretical_max_mbps"]
    )
    df["solver_order"] = df["solver"].map(
        {solver: idx for idx, solver in enumerate(SOLVER_ORDER)}
    ).fillna(999)
    return df[df["solver"].isin(SOLVER_ORDER)].sort_values(
        ["max_dist", "solver_order", "num_ap"]
    ).reset_index(drop=True)


def plot_fig5(summary_df: pd.DataFrame, result_dir: Path) -> tuple[Path, Path, Path]:
    configure_style()

    max_dist_list = sorted(summary_df["max_dist"].astype(float).unique())
    ap_list = sorted(summary_df["num_ap"].astype(int).unique())

    fig, axes = plt.subplots(2, 2, figsize=(15.6, 12.0), sharex=False, sharey=False)
    fig.patch.set_facecolor("white")
    axes_flat = axes.ravel()
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    y_max = float(summary_df["best_normalized_throughput"].max())
    y_upper = max(0.6, np.ceil((y_max + 0.03) / 0.1) * 0.1)

    for ax_idx, max_dist in enumerate(max_dist_list):
        ax = axes_flat[ax_idx]
        ax.set_facecolor("white")
        panel_df = summary_df[np.isclose(summary_df["max_dist"], float(max_dist))]

        for solver in SOLVER_ORDER:
            solver_df = (
                panel_df[panel_df["solver"] == solver]
                .set_index("num_ap")
                .reindex(ap_list)
                .reset_index()
            )
            if solver_df["best_normalized_throughput"].isna().all():
                continue

            style = SOLVER_STYLES.get(solver, {"color": "#333333", "marker": "o"})
            is_cdsb = solver == "GDSB"
            ax.plot(
                solver_df["num_ap"],
                solver_df["best_normalized_throughput"],
                label=DISPLAY_NAMES.get(solver, solver),
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=1.0 if is_cdsb else 0.75,
                markersize=15.5 if is_cdsb else 7.4,
                linewidth=2.15,
                zorder=4 if is_cdsb else 3,
            )

        ax.text(
            0.965,
            0.925,
            f"L = {int(max_dist)} m",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=18,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
        ax.set_xticks(ap_list)
        ax.set_ylim(0.0, y_upper)
        ax.set_yticks(np.arange(0.0, y_upper + 0.001, 0.2))
        ax.set_xlabel("Number of APs", fontsize=22, fontweight="bold", labelpad=2)
        ax.set_ylabel("Normalized throughput", fontsize=22, fontweight="bold", labelpad=2)
        ax.grid(True, which="major", axis="both", color="#E4E4E4", linewidth=0.8)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.15)

        ax.tick_params(
            direction="in",
            top=False,
            right=False,
            width=1.15,
            length=5.0,
            labelsize=18,
        )
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.985, 0.80),
            frameon=False,
            ncol=2,
            fontsize=16,
            handlelength=1.25,
            columnspacing=0.55,
            labelspacing=0.18,
            borderaxespad=0.0,
        )

    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.095,
        top=0.955,
        wspace=0.50,
        hspace=0.50,
    )

    for ax_idx, ax in enumerate(axes_flat):
        pos = ax.get_position()
        fig.text(
            pos.x0,
            pos.y1 + 0.007,
            panel_labels[ax_idx],
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
        )

    png_path = result_dir / f"{OUTPUT_STEM}.png"
    pdf_path = result_dir / f"{OUTPUT_STEM}.pdf"
    eps_path = result_dir / f"{OUTPUT_STEM}.eps"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(eps_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png_path, pdf_path, eps_path


def plot_single_fig5_panels(summary_df: pd.DataFrame, result_dir: Path) -> list[tuple[Path, Path, Path]]:
    configure_style()

    max_dist_list = sorted(summary_df["max_dist"].astype(float).unique())
    ap_list = sorted(summary_df["num_ap"].astype(int).unique())
    y_max = float(summary_df["best_normalized_throughput"].max())
    y_upper = max(0.6, np.ceil((y_max + 0.03) / 0.1) * 0.1)
    outputs: list[tuple[Path, Path, Path]] = []

    for panel_idx, max_dist in enumerate(max_dist_list, start=1):
        panel_df = summary_df[np.isclose(summary_df["max_dist"], float(max_dist))]

        fig, ax = plt.subplots(figsize=(7.8, 6.0))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        for solver in SOLVER_ORDER:
            solver_df = (
                panel_df[panel_df["solver"] == solver]
                .set_index("num_ap")
                .reindex(ap_list)
                .reset_index()
            )
            if solver_df["best_normalized_throughput"].isna().all():
                continue

            style = SOLVER_STYLES.get(solver, {"color": "#333333", "marker": "o"})
            is_cdsb = solver == "GDSB"
            ax.plot(
                solver_df["num_ap"],
                solver_df["best_normalized_throughput"],
                label=DISPLAY_NAMES.get(solver, solver),
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=1.0 if is_cdsb else 0.75,
                markersize=15.5 if is_cdsb else 7.4,
                linewidth=2.15,
                zorder=4 if is_cdsb else 3,
            )

        ax.text(
            0.965,
            0.925,
            f"L = {int(max_dist)} m",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=18,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
        ax.set_xticks(ap_list)
        ax.set_ylim(0.0, y_upper)
        ax.set_yticks(np.arange(0.0, y_upper + 0.001, 0.2))
        ax.set_xlabel("Number of APs", fontsize=22, fontweight="bold", labelpad=2)
        ax.set_ylabel("Normalized throughput", fontsize=22, fontweight="bold", labelpad=2)
        ax.grid(True, which="major", axis="both", color="#E4E4E4", linewidth=0.8)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.15)

        ax.tick_params(
            direction="in",
            top=False,
            right=False,
            width=1.15,
            length=5.0,
            labelsize=18,
        )
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.985, 0.80),
            frameon=False,
            ncol=2,
            fontsize=16,
            handlelength=1.25,
            columnspacing=0.55,
            labelspacing=0.18,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.16, top=0.985)

        png_path = result_dir / f"fig5_{panel_idx}.png"
        pdf_path = result_dir / f"fig5_{panel_idx}.pdf"
        eps_path = result_dir / f"fig5_{panel_idx}.eps"
        fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(eps_path, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        outputs.append((png_path, pdf_path, eps_path))

    return outputs


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    result_dir = args.result_dir.resolve() if args.result_dir else latest_result_dir(
        SCRIPT_DIR, SUMMARY_NAME, *DATA_DIRS
    )
    output_dir = (
        args.output_dir.resolve() if args.output_dir else make_timestamped_output_dir()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_fig5_summary(result_dir)
    outputs = plot_fig5(summary_df, output_dir)
    single_outputs = plot_single_fig5_panels(summary_df, output_dir)

    print(f"Used Fig. 5 result dir: {result_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Saved PNG to: {outputs[0]}")
    print(f"Saved PDF to: {outputs[1]}")
    print(f"Saved EPS to: {outputs[2]}")
    for panel_outputs in single_outputs:
        print(f"Saved single-panel PNG to: {panel_outputs[0]}")
        print(f"Saved single-panel PDF to: {panel_outputs[1]}")
        print(f"Saved single-panel EPS to: {panel_outputs[2]}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw IEEE-style Fig. 5 normalized max-dist panels."
    )
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
