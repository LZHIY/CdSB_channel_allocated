# -*- coding: utf-8 -*-
"""Draw IEEE-style Fig. 4 with true-theoretical normalized throughput.

The theoretical maximum is computed in the same way as Fig. 2:
``WLANIsingModel.compute_true_theoretical_max()`` for each UE seed.

The plot keeps x=TTS(ms) and uses y=throughput/theoretical maximum.

Run:
    python result_picturre/fig4_randomUE_AP30/plot_fig4_ieee_normalized_tts.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent


def _find_project_root() -> Path:
    """Folder that holds core/ (the package root, in either layout)."""
    for candidate in (SCRIPT_DIR.parent, SCRIPT_DIR, SCRIPT_DIR.parents[1]):
        if (candidate / "core" / "environment.py").exists():
            return candidate
    return SCRIPT_DIR.parent


PROJECT_ROOT = _find_project_root()
# Data folder shipped with this figure: <figure>/data, i.e. side by side with this script.
# SCRIPT_DIR.parent/"data" is kept as a fallback for a <figure>/plot/ layout.
DATA_DIRS = (SCRIPT_DIR / "data", SCRIPT_DIR.parent / "data")

# Every generated image goes to <package>/result_picture/<timestamp>_<figure>/
RESULT_PICTURE_DIR = PROJECT_ROOT / "result_picture"
OUTPUT_TAG = "fig04_random_ue_tts"
SUMMARY_NAME = "ap30_ue_seed_tts_summary.csv"
CONFIG_NAME = "experiment_config.yaml"
OUTPUT_STEM = "fig4_ieee_normalized_throughput_tts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.environment import WLANIsingModel


SOLVER_ORDER = ["CIM", "BSB", "DSB", "GBSB", "GDSB", "SA"]
DISPLAY_NAMES = {
    "CIM": "CIM",
    "BSB": "bSB",
    "DSB": "dSB",
    "GBSB": "GbSB",
    "GDSB": "CdSB",
    "SA": "SA",
}
SOLVER_STYLES = {
    "CIM": {"color": "#009E73", "marker": "o"},
    "DSB": {"color": "#7A3EB1", "marker": "o"},
    "BSB": {"color": "#56B4E9", "marker": "o"},
    "GBSB": {"color": "#CC79A7", "marker": "o"},
    "GDSB": {"color": "#D55E00", "marker": "*"},
    "SA": {"color": "#0072B2", "marker": "o"},
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


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
            "legend.fontsize": 15,
            "axes.linewidth": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def theoretical_max_by_seed(result_dir: Path, summary_df: pd.DataFrame) -> dict[int, float]:
    runtime_config = load_yaml(result_dir / CONFIG_NAME)
    experiment = runtime_config.get("experiment", {})
    env_config = runtime_config.get("project_config_used_for_unspecified_defaults", {}).get(
        "environment", {}
    )

    num_ap = int(experiment.get("num_ap", summary_df["num_ap"].iloc[0]))
    ap_seed = int(experiment.get("ap_seed", summary_df["ap_seed"].iloc[0]))
    ue_seeds = sorted(int(seed) for seed in summary_df["ue_seed"].unique())

    theory: dict[int, float] = {}
    for ue_seed in ue_seeds:
        env = WLANIsingModel(
            num_ap=num_ap,
            num_ch=int(env_config.get("num_ch", 3)),
            num_ue_per_ap=int(env_config.get("num_ue_per_ap", 10)),
            ap_seed=ap_seed,
            ue_seed=int(ue_seed),
            max_dist=float(env_config.get("max_dist", 150)),
            carrier_sense_range_m=float(env_config.get("carrier_sense_range_m", 70.0)),
        )
        theory[int(ue_seed)] = float(env.compute_true_theoretical_max())
    return theory


def load_plot_data(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = result_dir / SUMMARY_NAME
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {SUMMARY_NAME}: {summary_path}")

    df = pd.read_csv(summary_path)
    required = {
        "num_ap",
        "ap_seed",
        "ue_seed",
        "solver",
        "tts_ms",
        "best_solver_sample_mbps",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["ue_seed"] = pd.to_numeric(df["ue_seed"], errors="raise").astype(int)
    df["tts_ms"] = pd.to_numeric(df["tts_ms"], errors="coerce")
    df["best_solver_sample_mbps"] = pd.to_numeric(
        df["best_solver_sample_mbps"], errors="coerce"
    )
    df = df[df["solver"].isin(SOLVER_ORDER)].copy()

    theory = theoretical_max_by_seed(result_dir, df)
    theory_df = pd.DataFrame(
        [
            {"ue_seed": ue_seed, "theoretical_max_mbps": theoretical_max}
            for ue_seed, theoretical_max in sorted(theory.items())
        ]
    )
    df = df.merge(theory_df, on="ue_seed", how="left")
    df["normalized_throughput"] = (
        df["best_solver_sample_mbps"] / df["theoretical_max_mbps"]
    )
    return df, theory_df


def percentile_error(values: np.ndarray) -> tuple[float, np.ndarray]:
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25, 75])
    err = np.array([[max(median - q25, 0.0)], [max(q75 - median, 0.0)]])
    return median, err


def plot_fig4(data_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path, Path]:
    configure_style()

    finite_mask = (
        np.isfinite(data_df["tts_ms"])
        & (data_df["tts_ms"] > 0.0)
        & np.isfinite(data_df["normalized_throughput"])
    )
    plot_df = data_df[finite_mask].copy()
    if plot_df.empty:
        raise ValueError("No finite TTS and normalized-throughput points available.")

    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for solver in SOLVER_ORDER:
        solver_df = plot_df[plot_df["solver"] == solver]
        if solver_df.empty:
            continue
        style = SOLVER_STYLES.get(solver, {"color": "#333333", "marker": "o"})
        is_cdsb = solver == "GDSB"

        ax.scatter(
            solver_df["tts_ms"],
            solver_df["normalized_throughput"],
            s=145 if is_cdsb else 62,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.95 if is_cdsb else 0.75,
            alpha=0.60,
            label=DISPLAY_NAMES.get(solver, solver),
            zorder=4 if is_cdsb else 3,
        )

        x_values = solver_df["tts_ms"].to_numpy(dtype=float)
        y_values = solver_df["normalized_throughput"].to_numpy(dtype=float)
        x_med, xerr = percentile_error(x_values)
        y_med, yerr = percentile_error(y_values)
        ax.errorbar(
            [x_med],
            [y_med],
            xerr=xerr,
            yerr=yerr,
            fmt=style["marker"],
            markersize=15.0 if is_cdsb else 9.2,
            markerfacecolor=style["color"],
            markeredgecolor="black",
            markeredgewidth=1.0,
            ecolor=style["color"],
            elinewidth=1.35,
            capsize=3.6,
            alpha=0.96,
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("TTS (ms)", fontweight="bold")
    ax.set_ylabel("Normalized throughput", fontweight="bold")

    y_min = max(0.0, float(plot_df["normalized_throughput"].min()) - 0.02)
    y_max = min(1.02, float(plot_df["normalized_throughput"].max()) + 0.03)
    ax.set_ylim(y_min, y_max)

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=5))
    ax.xaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12)
    )
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))

    ax.grid(True, which="major", axis="both", color="#E0E0E0", linewidth=0.82)
    ax.grid(True, which="minor", axis="x", color="#EDEDED", linewidth=0.58)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.15)

    ax.tick_params(direction="in", top=False, right=False, width=1.15, length=5.0)
    ax.legend(frameon=False, loc="upper right", handlelength=1.5, borderaxespad=0.4)
    fig.tight_layout(pad=0.55)

    png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    eps_path = output_dir / f"{OUTPUT_STEM}.eps"
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(eps_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, eps_path


def run(args: argparse.Namespace) -> tuple[Path, Path, Path, pd.DataFrame]:
    result_dir = args.result_dir.resolve() if args.result_dir else latest_result_dir()
    output_dir = (
        args.output_dir.resolve() if args.output_dir else make_timestamped_output_dir()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data_df, theory_df = load_plot_data(result_dir)
    outputs = plot_fig4(data_df, output_dir)

    print("Theoretical maximum throughput by UE seed (Mbps):")
    for item in theory_df.itertuples(index=False):
        print(f"UE seed={int(item.ue_seed)}: {float(item.theoretical_max_mbps):.6f}")
    print(f"Output folder: {output_dir}")
    print(f"Saved PNG to: {outputs[0]}")
    print(f"Saved PDF to: {outputs[1]}")
    print(f"Saved EPS to: {outputs[2]}")
    return (*outputs, theory_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw IEEE-style Fig. 4 normalized-throughput TTS plot."
    )
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
