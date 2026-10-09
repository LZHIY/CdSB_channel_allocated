# -*- coding: utf-8 -*-
"""Compare solver throughput while sweeping WLAN area size max_dist.

Experiment:
    max_dist in [150, 200, 250, 300]
    AP count in [10, 20, 30, 40, 50]

For each (max_dist, AP count), the script builds a WLANIsingModel with fixed
AP/UE seeds, runs each solver, saves CSV results, and visualizes throughput
versus AP count in a 2x2 panel figure where each panel corresponds to one
max_dist value.

Run:
    python max_dist_solver_throughput_experiment.py

Quick smoke test:
    python max_dist_solver_throughput_experiment.py --quick
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"
RESULT_ROOT = PROJECT_ROOT / "resultsMaxDistThroughput"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.environment import WLANIsingModel
from solvers.bsb_solver import AdvancedbSBSolver
from solvers.cim_solver import ParallelCIMSolver
from solvers.dsb_solver import AdvanceddSBSolver
from solvers.gbsb_solver import ParallelGBSBSolver
from solvers.gdsb_solver import ParallelGDSBSolver
from solvers.greedy_solver import GreedyAssignmentSolver
from solvers.sa_solver import SASolver


AP_LIST = [10, 20, 30, 40, 50]
MAX_DIST_LIST = [150, 200, 250, 300]
AP_SEED = 15
UE_SEED = 16
DT = 0.15
STEPS = 900
NUM_AGENTS = 100
RUNS_PER_SOLVER = 1

DEFAULT_SOLVERS = ["SA", "Greedy", "CIM", "DSB", "BSB", "GBSB", "GDSB"]
SOLVER_ORDER = ["SA", "Greedy", "CIM", "DSB", "BSB", "GBSB", "GDSB"]
SOLVER_REGISTRY = {
    "SA": SASolver,
    "Greedy": GreedyAssignmentSolver,
    "CIM": ParallelCIMSolver,
    "DSB": AdvanceddSBSolver,
    "BSB": AdvancedbSBSolver,
    "GBSB": ParallelGBSBSolver,
    "GDSB": ParallelGDSBSolver,
}
A_SOLVERS = {"GBSB", "GDSB"}
A_BY_AP = {
    10: 0.30,
    20: 0.49,
    30: 0.39,
    40: 0.59,
    50: 0.63,
}
SOLVER_COLORS = {
    "SA": "#666666",
    "Greedy": "#588157",
    "CIM": "#0072B2",
    "DSB": "#332288",
    "BSB": "#009E73",
    "GBSB": "#E69F00",
    "GDSB": "#CC6677",
}
SOLVER_MARKERS = {
    "SA": "X",
    "Greedy": "v",
    "CIM": "o",
    "DSB": "^",
    "BSB": "s",
    "GBSB": "D",
    "GDSB": "P",
}
def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_algorithm_keys(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    algorithms = normalized.setdefault("algorithms", {})
    if "GbSB" in algorithms and "GBSB" not in algorithms:
        algorithms["GBSB"] = algorithms["GbSB"]
    return normalized


def make_result_dir() -> Path:
    now = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    ms = int((now - int(now)) * 1000)
    result_dir = RESULT_ROOT / f"{timestamp}_{ms:03d}_max_dist_solver_throughput"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def make_environment(config: dict[str, Any], *, num_ap: int, max_dist: float) -> WLANIsingModel:
    env_config = config.get("environment", {})
    return WLANIsingModel(
        num_ap=int(num_ap),
        num_ch=int(env_config.get("num_ch", 3)),
        num_ue_per_ap=int(env_config.get("num_ue_per_ap", 10)),
        ap_seed=AP_SEED,
        ue_seed=UE_SEED,
        max_dist=float(max_dist),
        carrier_sense_range_m=float(env_config.get("carrier_sense_range_m", 70.0)),
    )


def solver_config_for(
    config: dict[str, Any],
    solver_name: str,
    *,
    num_ap: int,
    dt: float,
    steps: int,
    num_agents: int,
) -> dict[str, Any]:
    algorithms = config.get("algorithms", {})
    algorithm_config = copy.deepcopy(algorithms.get(solver_name, {}))

    if solver_name == "DSB":
        algorithm_config.setdefault("mode", "DSB")
    if solver_name == "BSB":
        algorithm_config.setdefault("mode", "BSB")

    solver_config = {
        **copy.deepcopy(config.get("solvers_common", {})),
        **algorithm_config,
        "num_agents": int(num_agents),
        "steps": int(steps),
        "dt": float(dt),
    }
    if solver_name in A_SOLVERS:
        solver_config["A"] = float(A_BY_AP[int(num_ap)])
    return solver_config


def assignment_repr(assignment: Any) -> str:
    if assignment is None:
        return ""
    arr = np.asarray(assignment, dtype=int).reshape(-1)
    return " ".join(str(int(x)) for x in arr)


def run_solver_once(
    solver_name: str,
    env: WLANIsingModel,
    J: np.ndarray,
    h: np.ndarray,
    theo_max: float,
    config: dict[str, Any],
    *,
    max_dist: float,
    num_ap: int,
    run_idx: int,
    dt: float,
    steps: int,
    num_agents: int,
) -> dict[str, Any]:
    solver_config = solver_config_for(
        config,
        solver_name,
        num_ap=num_ap,
        dt=dt,
        steps=steps,
        num_agents=num_agents,
    )
    solver = SOLVER_REGISTRY[solver_name](J=J, h=h, env=env, config=solver_config)

    tic = time.perf_counter()
    throughput_mbps, assignment, _, _ = solver.solve()
    wall_time_s = time.perf_counter() - tic
    normalized = float(throughput_mbps) / float(theo_max) if theo_max > 1e-12 else np.nan

    return {
        "max_dist": float(max_dist),
        "num_ap": int(num_ap),
        "num_ch": int(env.NUM_CH),
        "num_bits": int(env.NUM_AP * env.NUM_CH),
        "solver": solver_name,
        "run_idx": int(run_idx),
        "A": float(A_BY_AP[int(num_ap)]) if solver_name in A_SOLVERS else np.nan,
        "dt": float(dt),
        "steps": int(steps),
        "num_agents": int(num_agents),
        "ap_seed": int(AP_SEED),
        "ue_seed": int(UE_SEED),
        "theoretical_max_mbps": float(theo_max),
        "throughput_mbps": float(throughput_mbps),
        "normalized_throughput": float(normalized),
        "wall_time_s": float(wall_time_s),
        "assignment": assignment_repr(assignment),
    }


def summarize_best(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, _, _), group in raw_df.groupby(["max_dist", "num_ap", "solver"], sort=False):
        best = group.sort_values(
            ["throughput_mbps", "normalized_throughput"],
            ascending=[False, False],
        ).iloc[0]
        rows.append(
            {
                "max_dist": float(best["max_dist"]),
                "num_ap": int(best["num_ap"]),
                "num_ch": int(best["num_ch"]),
                "num_bits": int(best["num_bits"]),
                "solver": str(best["solver"]),
                "best_run_idx": int(best["run_idx"]),
                "A": best["A"],
                "dt": float(best["dt"]),
                "steps": int(best["steps"]),
                "num_agents": int(best["num_agents"]),
                "theoretical_max_mbps": float(best["theoretical_max_mbps"]),
                "best_throughput_mbps": float(best["throughput_mbps"]),
                "best_normalized_throughput": float(best["normalized_throughput"]),
                "mean_throughput_mbps": float(group["throughput_mbps"].mean()),
                "std_throughput_mbps": float(group["throughput_mbps"].std(ddof=1))
                if len(group) > 1
                else 0.0,
                "mean_normalized_throughput": float(group["normalized_throughput"].mean()),
                "mean_wall_time_s": float(group["wall_time_s"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_throughput_panels(
    summary_df: pd.DataFrame,
    *,
    max_dist_list: list[float],
    ap_list: list[int],
    solvers: list[str],
    result_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.edgecolor": "#111111",
            "axes.linewidth": 0.9,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.6,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    y_values = summary_df["best_throughput_mbps"].to_numpy(dtype=float)
    finite_y = y_values[np.isfinite(y_values)]
    y_max = float(finite_y.max()) * 1.12 if finite_y.size else 1.0

    for ax_idx, max_dist in enumerate(max_dist_list):
        ax = axes_flat[ax_idx]
        panel_df = summary_df[np.isclose(summary_df["max_dist"].astype(float), float(max_dist))]
        for solver_name in solvers:
            solver_df = (
                panel_df[panel_df["solver"] == solver_name]
                .set_index("num_ap")
                .reindex(ap_list)
                .reset_index()
            )
            if solver_df["best_throughput_mbps"].isna().all():
                continue
            ax.plot(
                solver_df["num_ap"],
                solver_df["best_throughput_mbps"],
                marker=SOLVER_MARKERS.get(solver_name, "o"),
                markersize=5.8,
                linewidth=1.65,
                color=SOLVER_COLORS.get(solver_name, "#4E79A7"),
                label=solver_name,
            )

        ax.set_title(f"max_dist = {int(max_dist)} m", fontsize=11.8, pad=7)
        ax.set_xticks(ap_list)
        ax.grid(True, which="major", color="#D0D3D6", linewidth=0.72, alpha=0.78)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0.0, top=y_max)

    for ax in axes[-1, :]:
        ax.set_xlabel("Number of APs")
    for ax in axes[:, 0]:
        ax.set_ylabel("Throughput (Mbps)")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(solvers), 7),
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.suptitle("Solver Throughput under Different Deployment Areas", y=1.055, fontsize=13.6)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    png_path = result_dir / "max_dist_solver_throughput_panels.png"
    pdf_path = result_dir / "max_dist_solver_throughput_panels.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=260)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return png_path, pdf_path


def save_runtime_config(
    result_dir: Path,
    config: dict[str, Any],
    *,
    max_dist_list: list[float],
    ap_list: list[int],
    solvers: list[str],
    runs_per_solver: int,
    dt: float,
    steps: int,
    num_agents: int,
) -> None:
    runtime_config = {
        "script": Path(__file__).name,
        "experiment": {
            "max_dist_list": [float(x) for x in max_dist_list],
            "ap_list": [int(x) for x in ap_list],
            "ap_seed": AP_SEED,
            "ue_seed": UE_SEED,
            "dt": float(dt),
            "steps": int(steps),
            "num_agents": int(num_agents),
            "runs_per_solver": int(runs_per_solver),
            "solvers": solvers,
            "A_by_ap": A_BY_AP,
            "best_policy": "best throughput over all runs for each max_dist/AP/solver",
            "visualization": "2x2 max_dist panels, x=AP count, y=best throughput Mbps",
        },
        "project_config_used_for_unspecified_defaults": config,
    }
    with (result_dir / "experiment_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_config, f, allow_unicode=True, sort_keys=False)


def run_experiment(args: argparse.Namespace) -> Path:
    config = normalize_algorithm_keys(load_config(args.config))
    result_dir = make_result_dir()

    max_dist_list = args.max_dists or MAX_DIST_LIST
    max_dist_list = list(dict.fromkeys(float(x) for x in max_dist_list))
    ap_list = args.ap_list or AP_LIST
    ap_list = list(dict.fromkeys(int(x) for x in ap_list))
    solvers = args.solvers or DEFAULT_SOLVERS
    runs_per_solver = int(args.runs)
    dt = float(args.dt)
    steps = int(args.steps)
    num_agents = int(args.num_agents)

    if args.quick:
        max_dist_list = [150.0, 200.0]
        ap_list = [10]
        solvers = ["Greedy", "CIM", "GDSB"]
        runs_per_solver = 1
        steps = 5
        num_agents = 4

    missing_a = [ap for ap in ap_list if ap not in A_BY_AP]
    if missing_a and any(solver in A_SOLVERS for solver in solvers):
        raise ValueError(f"A is not configured for AP counts: {missing_a}")

    save_runtime_config(
        result_dir,
        config,
        max_dist_list=max_dist_list,
        ap_list=ap_list,
        solvers=solvers,
        runs_per_solver=runs_per_solver,
        dt=dt,
        steps=steps,
        num_agents=num_agents,
    )

    print("=" * 92)
    print("Max-dist solver throughput experiment")
    print(f"Result dir: {result_dir}")
    print(f"max_dist list: {max_dist_list}")
    print(f"AP list: {ap_list}")
    print(f"Solvers: {', '.join(solvers)}")
    print(f"AP seed={AP_SEED}, UE seed={UE_SEED}, dt={dt}, steps={steps}, agents={num_agents}")
    print("=" * 92)

    raw_rows: list[dict[str, Any]] = []
    for max_dist in max_dist_list:
        for num_ap in ap_list:
            print(f"\nBuilding WLAN model: max_dist={max_dist:g}, AP={num_ap}")
            env = make_environment(config, num_ap=num_ap, max_dist=max_dist)
            J, h = env.build_J_h()
            theo_max = env.compute_true_theoretical_max()
            print(f"J{J.shape}, h{h.shape}, theoretical max={theo_max:.6g} Mbps")

            for solver_name in solvers:
                print(f"  {solver_name}", end=" | ", flush=True)
                for run_idx in range(1, runs_per_solver + 1):
                    row = run_solver_once(
                        solver_name,
                        env,
                        J,
                        h,
                        theo_max,
                        config,
                        max_dist=max_dist,
                        num_ap=num_ap,
                        run_idx=run_idx,
                        dt=dt,
                        steps=steps,
                        num_agents=num_agents,
                    )
                    raw_rows.append(row)
                    print(f"run {run_idx}: {row['throughput_mbps']:.3f} Mbps", end="  ")
                print("")

    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize_best(raw_df)

    raw_path = result_dir / "max_dist_raw_solver_runs.csv"
    summary_path = result_dir / "max_dist_best_solver_summary.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    panel_png_path, panel_pdf_path = plot_throughput_panels(
        summary_df,
        max_dist_list=max_dist_list,
        ap_list=ap_list,
        solvers=solvers,
        result_dir=result_dir,
    )
    print("\nBest throughput (Mbps):")
    for max_dist in max_dist_list:
        panel = summary_df[np.isclose(summary_df["max_dist"].astype(float), float(max_dist))]
        printable = panel.pivot(
            index="num_ap",
            columns="solver",
            values="best_throughput_mbps",
        ).reindex(index=ap_list, columns=solvers)
        print(f"\nmax_dist={max_dist:g}")
        print(printable.to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved panel figure PNG to: {panel_png_path}")
    print(f"Saved panel figure PDF to: {panel_pdf_path}")
    return result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep max_dist and compare solver throughput.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-dists", type=float, nargs="+", default=None)
    parser.add_argument("--ap-list", type=int, nargs="+", default=None)
    parser.add_argument("--solvers", choices=SOLVER_ORDER, nargs="+", default=None)
    parser.add_argument("--runs", type=int, default=RUNS_PER_SOLVER)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
