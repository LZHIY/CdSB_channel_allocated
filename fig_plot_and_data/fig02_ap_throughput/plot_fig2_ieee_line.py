# -*- coding: utf-8 -*-
"""Draw IEEE-style Fig. 2 AP-throughput line plot.

The script reads best_solver_summary.csv from the latest result directory under
this folder, then saves publication-ready vector and raster figures.

Examples:
    python result_picturre/fig2_AP_thought_plot/plot_fig2_ieee_line.py
    python result_picturre/fig2_AP_thought_plot/plot_fig2_ieee_line.py --result-dir path/to/result
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
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
OUTPUT_TAG = "fig02_ap_throughput"
SUMMARY_NAME = "best_solver_summary.csv"
ADJUSTED_SUMMARY_NAME = "fig2_ap30_replaced_with_fig5_ue10_summary.csv"
FIG5_DIR = SCRIPT_DIR.parent / "fig5_ue_density"
FIG5_SUMMARY_NAME = "ue_count_best_solver_summary.csv"
REPLACED_NUM_AP = 30
REPLACEMENT_UE_PER_AP = 10

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


def _latest_dir_containing(name: str, roots: tuple[Path, ...]) -> Path | None:
    """Newest directory that directly contains `name`, searched under each root."""
    found = [
        path
        for root in roots
        if root.is_dir()
        for path in root.iterdir()
        if path.is_dir() and (path / name).exists()
    ]
    if not found:
        return None
    return max(found, key=lambda path: path.stat().st_mtime)


def make_timestamped_output_dir() -> Path:
    """<package>/result_picture/<timestamp>_<figure>/ (created on demand)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir = RESULT_PICTURE_DIR / f"{timestamp}_{OUTPUT_TAG}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def latest_result_dir() -> Path:
    roots = (SCRIPT_DIR, *DATA_DIRS)
    found = _latest_dir_containing(SUMMARY_NAME, roots)
    if found is None:
        searched = " or ".join(str(root) for root in roots)
        raise FileNotFoundError(
            f"No directory containing {SUMMARY_NAME} found under {searched}. "
            "Pass --result-dir explicitly."
        )
    return found


def latest_fig5_result_dir() -> Path:
    roots = (FIG5_DIR, *DATA_DIRS)
    found = _latest_dir_containing(FIG5_SUMMARY_NAME, roots)
    if found is None:
        searched = " or ".join(str(root) for root in roots)
        raise FileNotFoundError(
            f"No directory containing {FIG5_SUMMARY_NAME} found under {searched}. "
            "Pass --fig5-result-dir explicitly."
        )
    return found


def load_summary(result_dir: Path) -> pd.DataFrame:
    summary_path = result_dir / SUMMARY_NAME
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {SUMMARY_NAME}: {summary_path}")

    df = pd.read_csv(summary_path)
    required = {
        "num_ap",
        "solver",
        "theoretical_max_mbps",
        "best_normalized_throughput",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["num_ap"] = pd.to_numeric(df["num_ap"], errors="raise").astype(int)
    df["best_normalized_throughput"] = pd.to_numeric(
        df["best_normalized_throughput"], errors="raise"
    )
    df["theoretical_max_mbps"] = pd.to_numeric(
        df["theoretical_max_mbps"], errors="raise"
    )
    df["solver_order"] = df["solver"].map(
        {solver: idx for idx, solver in enumerate(SOLVER_ORDER)}
    ).fillna(999)
    return df.sort_values(["solver_order", "num_ap"]).reset_index(drop=True)


def load_fig5_ue10_replacement(fig5_result_dir: Path) -> pd.DataFrame:
    summary_path = fig5_result_dir / FIG5_SUMMARY_NAME
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {FIG5_SUMMARY_NAME}: {summary_path}")

    df = pd.read_csv(summary_path)
    required = {
        "target_num_ue_per_ap",
        "solver",
        "theoretical_max_mbps",
        "best_throughput_mbps",
        "best_normalized_throughput",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in Fig. 5 summary: {sorted(missing)}")

    df = df.copy()
    df["target_num_ue_per_ap"] = pd.to_numeric(
        df["target_num_ue_per_ap"], errors="raise"
    ).astype(int)
    df = df[
        (df["target_num_ue_per_ap"] == REPLACEMENT_UE_PER_AP)
        & df["solver"].isin(SOLVER_ORDER)
    ].copy()
    if df.empty:
        raise ValueError(f"No Fig. 5 rows found for UE/AP={REPLACEMENT_UE_PER_AP}.")

    df["num_ap"] = REPLACED_NUM_AP
    df["theoretical_max_mbps"] = pd.to_numeric(
        df["theoretical_max_mbps"], errors="raise"
    )
    df["best_throughput_mbps"] = pd.to_numeric(
        df["best_throughput_mbps"], errors="raise"
    )
    df["best_normalized_throughput"] = pd.to_numeric(
        df["best_normalized_throughput"], errors="raise"
    )
    return df[
        [
            "num_ap",
            "solver",
            "theoretical_max_mbps",
            "best_throughput_mbps",
            "best_normalized_throughput",
        ]
    ]


def replace_ap30_with_fig5_ue10(
    summary_df: pd.DataFrame, fig5_replacement_df: pd.DataFrame
) -> pd.DataFrame:
    adjusted_df = summary_df.copy()
    solver_to_fig5 = fig5_replacement_df.set_index("solver")

    for solver in SOLVER_ORDER:
        if solver not in solver_to_fig5.index:
            raise ValueError(f"Fig. 5 replacement is missing solver={solver}.")
        row_mask = (adjusted_df["num_ap"] == REPLACED_NUM_AP) & (
            adjusted_df["solver"] == solver
        )
        if not row_mask.any():
            raise ValueError(f"Fig. 2 summary is missing AP={REPLACED_NUM_AP}, solver={solver}.")

        replacement = solver_to_fig5.loc[solver]
        adjusted_df.loc[row_mask, "theoretical_max_mbps"] = float(
            replacement["theoretical_max_mbps"]
        )
        if "best_throughput_mbps" in adjusted_df.columns:
            adjusted_df.loc[row_mask, "best_throughput_mbps"] = float(
                replacement["best_throughput_mbps"]
            )
        adjusted_df.loc[row_mask, "best_normalized_throughput"] = float(
            replacement["best_normalized_throughput"]
        )

    return adjusted_df.sort_values(["solver_order", "num_ap"]).reset_index(drop=True)


def save_adjusted_summary(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / ADJUSTED_SUMMARY_NAME
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def theoretical_max_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    table = (
        summary_df[["num_ap", "theoretical_max_mbps"]]
        .drop_duplicates()
        .sort_values("num_ap")
        .reset_index(drop=True)
    )
    return table


def configure_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "stix",
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


def plot_fig2(summary_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path, Path]:
    configure_ieee_style()

    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for solver in SOLVER_ORDER:
        solver_df = summary_df[summary_df["solver"] == solver].sort_values("num_ap")
        if solver_df.empty:
            continue

        style = SOLVER_STYLES.get(
            solver,
            {"color": "#333333"},
        )
        ax.plot(
            solver_df["num_ap"],
            solver_df["best_normalized_throughput"],
            label=DISPLAY_NAMES.get(solver, solver),
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=1.0 if solver == "GDSB" else 0.8,
            markersize=18.0 if solver == "GDSB" else 9.0,
            linewidth=2.5,
            zorder=4 if solver == "GDSB" else 3,
        )

    ax.set_xlabel("Number of APs", fontweight="bold")
    ax.set_ylabel("Normalized throughput", fontweight="bold")
    ax.set_ylim(0.0, 0.6)
    ax.set_yticks(np.arange(0.0, 0.61, 0.1))
    ax.set_xticks(sorted(summary_df["num_ap"].astype(int).unique()))
    ax.grid(True, which="major", axis="y", color="#E4E4E4", linewidth=0.8)
    ax.grid(True, which="major", axis="x", color="#EEEEEE", linewidth=0.6)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.15)

    ax.tick_params(direction="in", top=False, right=False, width=1.15, length=5.0)
    ax.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        handlelength=2.0,
        columnspacing=0.9,
        labelspacing=0.35,
        borderaxespad=0.45,
    )

    fig.tight_layout(pad=0.55)

    pdf_path = output_dir / "fig2_ieee_ap_throughput_line.pdf"
    eps_path = output_dir / "fig2_ieee_ap_throughput_line.eps"
    png_path = output_dir / "fig2_ieee_ap_throughput_line.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(eps_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return pdf_path, eps_path, png_path


def run(args: argparse.Namespace) -> tuple[Path, Path, Path, pd.DataFrame]:
    result_dir = args.result_dir.resolve() if args.result_dir else latest_result_dir()
    output_dir = (
        args.output_dir.resolve() if args.output_dir else make_timestamped_output_dir()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = load_summary(result_dir)
    fig5_result_dir = (
        args.fig5_result_dir.resolve() if args.fig5_result_dir else latest_fig5_result_dir()
    )
    fig5_replacement_df = load_fig5_ue10_replacement(fig5_result_dir)
    summary_df = replace_ap30_with_fig5_ue10(summary_df, fig5_replacement_df)
    adjusted_summary_path = save_adjusted_summary(summary_df, output_dir)
    theory_df = theoretical_max_table(summary_df)
    outputs = plot_fig2(summary_df, output_dir)

    print("Theoretical maximum throughput (Mbps):")
    for item in theory_df.itertuples(index=False):
        print(f"AP={int(item.num_ap)}: {float(item.theoretical_max_mbps):.6f}")
    print(f"Output folder: {output_dir}")
    print(f"Saved PDF to: {outputs[0]}")
    print(f"Saved EPS to: {outputs[1]}")
    print(f"Saved PNG to: {outputs[2]}")
    print(f"Saved adjusted summary to: {adjusted_summary_path}")
    return (*outputs, theory_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw IEEE-style Fig. 2 line plot.")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--fig5-result-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
