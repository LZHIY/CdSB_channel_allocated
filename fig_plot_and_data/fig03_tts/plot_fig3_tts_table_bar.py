# -*- coding: utf-8 -*-
"""Draw Fig. 3 IEEE-style TTS table-bar plot.

The plot keeps the table-plus-log-bar format while using the visual conventions
of Fig. 2: Arial fonts, bold 22-pt axis label, full frame, and CdSB-colored
shades for the solver bars.

Run:
    python result_picturre/fig3_AP_tts/plot_fig3_tts_table_bar.py
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
OUTPUT_TAG = "fig03_tts"
SUMMARY_NAME = "tts_rbk_summary.csv"
OUTPUT_STEM = "fig3_tts_table_bar"

SOLVER_ORDER = ["SA", "CIM", "BSB", "DSB", "GBSB", "GDSB"]
DISPLAY_NAMES = {
    "CIM": "CIM",
    "BSB": "bSB",
    "DSB": "dSB",
    "GBSB": "GbSB",
    "GDSB": "CdSB",
    "SA": "SA",
}

DEFAULT_BAR_COLOR = "#F6C99F"
BEST_BAR_COLOR = "#D55E00"
X_AXIS_MAX = 2.0e4
CELL_PAD = 1.0
GROUP_RULE_COLOR = "#000000"
GROUP_RULE_WIDTH = 0.9


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


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 18,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.linewidth": 1.15,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def format_tts_ms(value: float) -> str:
    if not np.isfinite(value):
        return "inf"
    return f"{value:.1f}"


def format_x_tick(value: float, _: int) -> str:
    exponent = int(np.round(np.log10(value)))
    if np.isclose(value, 10.0**exponent):
        return rf"$10^{{{exponent}}}$"
    return ""


def load_summary(result_dir: Path) -> pd.DataFrame:
    summary_path = result_dir / SUMMARY_NAME
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {SUMMARY_NAME}: {summary_path}")

    df = pd.read_csv(summary_path)
    required = {"num_ap", "solver", "tts_ms"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["num_ap"] = pd.to_numeric(df["num_ap"], errors="raise").astype(int)
    if "num_bits" in df.columns:
        df["num_bits"] = pd.to_numeric(df["num_bits"], errors="raise").astype(int)
    else:
        df["num_bits"] = df["num_ap"] * 3
    df["tts_ms"] = pd.to_numeric(df["tts_ms"], errors="coerce")
    df["solver_order"] = df["solver"].map(
        {solver: idx for idx, solver in enumerate(SOLVER_ORDER)}
    ).fillna(999)
    return df[df["solver"].isin(SOLVER_ORDER)].sort_values(
        ["num_ap", "solver_order"]
    ).reset_index(drop=True)


def make_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for num_ap, group in summary_df.groupby("num_ap", sort=True):
        group = group.sort_values("solver_order")
        finite = group[np.isfinite(group["tts_ms"].to_numpy(dtype=float))]
        best_tts = float(finite["tts_ms"].min()) if not finite.empty else np.nan
        num_bits = int(group.iloc[0]["num_bits"])

        for item in group.itertuples(index=False):
            tts_ms = float(item.tts_ms)
            rows.append(
                {
                    "num_ap": int(num_ap),
                    "num_bits": num_bits,
                    "machine": DISPLAY_NAMES.get(str(item.solver), str(item.solver)),
                    "solver": str(item.solver),
                    "tts_ms": tts_ms,
                    "tts_text": format_tts_ms(tts_ms),
                    "is_best_tts": bool(np.isfinite(tts_ms) and np.isclose(tts_ms, best_tts)),
                }
            )
    return pd.DataFrame(rows)


def plot_table_bar(rows: pd.DataFrame, output_dir: Path) -> tuple[Path, Path, Path]:
    if rows.empty:
        raise ValueError("No TTS rows to plot.")

    configure_style()

    y_positions = []
    group_bounds = []
    y = 0.95
    group_gap = CELL_PAD
    for num_ap in rows["num_ap"].drop_duplicates():
        group = rows[rows["num_ap"] == num_ap]
        start_y = y
        for _ in range(len(group)):
            y_positions.append(y)
            y += 1.0
        group_bounds.append((int(num_ap), start_y, y - 1.0))
        y += group_gap

    rows = rows.copy()
    rows["y"] = y_positions

    body_top = float(rows["y"].min() - CELL_PAD)
    body_bottom = float(rows["y"].max() + CELL_PAD)

    xmin = 1.0e1
    xmax = X_AXIS_MAX
    inf_bar_value = xmax / 1.03

    header_height = 0.95
    y_max = body_bottom

    fig = plt.figure(figsize=(8.0, 10.0))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.025)
    table_ax = fig.add_subplot(gs[0, 0])
    bar_ax = fig.add_subplot(gs[0, 1])

    table_ax.set_facecolor("white")
    bar_ax.set_facecolor("white")
    table_ax.set_xlim(0.0, 1.0)
    table_ax.set_ylim(y_max, -header_height)
    bar_ax.set_ylim(body_bottom, body_top)
    table_ax.axis("off")

    header_y = -0.58
    columns = [
        ("AP", 0.08, "center"),
        ("N", 0.22, "center"),
        ("Solver", 0.52, "center"),
        ("TTS (ms)", 0.875, "center"),
    ]
    for label, x_pos, ha in columns:
        table_ax.text(
            x_pos,
            header_y,
            label,
            ha=ha,
            va="center",
            fontsize=15.5,
            fontweight="bold",
        )
    header_rules = table_ax.hlines(
        [-1.22, -0.04],
        xmin=0.045,
        xmax=0.985,
        color="black",
        linewidth=[1.8, 1.05],
    )
    header_rules.set_clip_on(False)

    for num_ap, start_y, end_y in group_bounds:
        group = rows[rows["num_ap"] == num_ap]
        center_y = (start_y + end_y) / 2.0

        table_ax.text(0.08, center_y, f"{num_ap}", ha="center", va="center", fontsize=16)
        table_ax.text(0.22, center_y, f"{int(group.iloc[0]['num_bits'])}", ha="center", va="center", fontsize=16)

        for item in group.itertuples(index=False):
            is_cdsb = str(item.solver) == "GDSB"
            weight = "bold" if is_cdsb or bool(item.is_best_tts) else "normal"
            table_ax.text(0.52, item.y, item.machine, ha="center", va="center", fontsize=15.5, fontweight=weight)
            table_ax.text(0.875, item.y, item.tts_text, ha="center", va="center", fontsize=15.5, fontweight=weight)

        table_rule = table_ax.hlines(
            end_y + CELL_PAD,
            xmin=0.045,
            xmax=0.985,
            color=GROUP_RULE_COLOR,
            linewidth=GROUP_RULE_WIDTH,
        )
        table_rule.set_clip_on(False)
        bar_rule = bar_ax.axhline(
            end_y + CELL_PAD,
            color=GROUP_RULE_COLOR,
            linewidth=GROUP_RULE_WIDTH,
            zorder=5,
        )
        bar_rule.set_clip_on(False)

    bar_ax.set_xscale("log")
    bar_ax.set_xlim(xmin, xmax)
    bar_ax.set_xlabel("TTS (ms)", fontsize=22, fontweight="bold")
    major_ticks = [
        multiplier * (10.0**exponent)
        for exponent in range(
            int(math.floor(math.log10(xmin))),
            int(math.ceil(math.log10(xmax))) + 1,
        )
        for multiplier in (1.0, 2.0, 5.0)
        if xmin <= multiplier * (10.0**exponent) <= xmax
    ]
    if not any(np.isclose(tick, X_AXIS_MAX) for tick in major_ticks):
        major_ticks.append(X_AXIS_MAX)
    bar_ax.xaxis.set_major_locator(mticker.FixedLocator(major_ticks))
    bar_ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_x_tick))
    bar_ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    bar_ax.grid(True, which="major", axis="x", color="#E9C9AF", linewidth=0.8)
    bar_ax.grid(True, which="minor", axis="x", color="#F2D9C6", linewidth=0.55)
    bar_ax.tick_params(axis="y", left=False, labelleft=False)
    bar_ax.tick_params(direction="in", top=False, right=False, width=1.15, length=5.0)
    bar_ax.tick_params(axis="x", labelsize=14)

    for spine in bar_ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.15)

    for item in rows.itertuples(index=False):
        color = BEST_BAR_COLOR if bool(item.is_best_tts) else DEFAULT_BAR_COLOR
        value = min(float(item.tts_ms), inf_bar_value) if np.isfinite(item.tts_ms) else inf_bar_value
        width = max(value - xmin, xmin * 0.10)
        bar_ax.barh(
            item.y,
            width,
            left=xmin,
            height=0.36,
            color=color,
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
        if not np.isfinite(item.tts_ms):
            bar_ax.text(
                inf_bar_value,
                item.y,
                "inf",
                ha="right",
                va="center",
                fontsize=15,
                color="black",
                zorder=4,
            )

    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.085, top=0.965, wspace=0.035)
    table_pos = table_ax.get_position()
    bar_pos = bar_ax.get_position()
    header_to_body_ratio = (body_bottom - body_top) / (body_bottom + header_height)
    bar_ax.set_position(
        [
            bar_pos.x0,
            table_pos.y0,
            bar_pos.width,
            table_pos.height * header_to_body_ratio,
        ]
    )

    png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    eps_path = output_dir / f"{OUTPUT_STEM}.eps"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    fig.savefig(eps_path)
    plt.close(fig)
    return png_path, pdf_path, eps_path


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    result_dir = args.result_dir.resolve() if args.result_dir else latest_result_dir()
    output_dir = (
        args.output_dir.resolve() if args.output_dir else make_timestamped_output_dir()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = load_summary(result_dir)
    rows = make_rows(summary_df)
    outputs = plot_table_bar(rows, output_dir)
    print(f"Output folder: {output_dir}")
    print(f"Saved PNG to: {outputs[0]}")
    print(f"Saved PDF to: {outputs[1]}")
    print(f"Saved EPS to: {outputs[2]}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw Fig. 3 IEEE-style TTS table-bar plot.")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
