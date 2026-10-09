# -*- coding: utf-8 -*-
"""Grouped bar benchmark for WLAN throughput solvers.

This script fixes the WLAN topology with AP seed 15 and UE seed 16, then
compares multiple solvers by their best normalized throughput over five
unseeded random initializations.

Run from the project root:
    python fig2_solver_throughput_bar_experiment.py

Quick smoke test:
    python fig2_solver_throughput_bar_experiment.py --quick
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
RESULT_ROOT = PROJECT_ROOT / "resultsTHROUGHT"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.environment import WLANIsingModel
from solvers.bsb_solver import AdvancedbSBSolver
from solvers.cim_solver import ParallelCIMSolver
from solvers.dsb_solver import AdvanceddSBSolver
from solvers.gbsb_solver import ParallelGBSBSolver
from solvers.gdsb_solver import ParallelGDSBSolver
from solvers.sa_solver import SASolver
from solvers.greedy_solver import GreedyAssignmentSolver


AP_LIST = [10, 20, 30, 40, 50]
A_BY_AP = {
    10: 0.3,
    20: 0.48,
    30: 0.39,
    40: 0.56,
    50: 0.63,
}
AP_SEED = 15
UE_SEED = 16
DT = 0.15
STEPS = 900
NUM_AGENTS = 300
RUNS_PER_SOLVER = 1

# Include algorithmic baselines for direct comparison.
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
SOLVER_COLORS = {
    "SA": "#6C7A89",
    "Greedy": "#588157",
    "CIM": "#457B9D",
    "DSB": "#1D3557",
    "BSB": "#2A9D8F",
    "GBSB": "#E9C46A",
    "GDSB": "#D1495B",
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
    result_dir = RESULT_ROOT / f"{timestamp}_solver_throughput_bar"
    result_dir = RESULT_ROOT / f"{timestamp}_{ms:03d}_solver_throughput_bar"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def make_environment(config: dict[str, Any], num_ap: int) -> WLANIsingModel:
    env_config = config.get("environment", {})
    return WLANIsingModel(
        num_ap=int(num_ap),
        num_ch=int(env_config.get("num_ch", 3)),
        num_ue_per_ap=int(env_config.get("num_ue_per_ap", 10)),
        ap_seed=AP_SEED,
        ue_seed=UE_SEED,
        max_dist=float(env_config.get("max_dist", 150)),
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


def run_solver_once(
    solver_name: str,
    env: WLANIsingModel,
    J: np.ndarray,
    h: np.ndarray,
    theo_max: float,
    config: dict[str, Any],
    *,
    num_ap: int,
    run_idx: int,
    dt: float,
    steps: int,
    num_agents: int,
) -> dict[str, Any]:
    # Intentionally do not call np.random.seed or random.seed here. Each solve()
    # call starts from the process RNG state, so xi is random and unseeded.
    solver_config = solver_config_for(
        config,
        solver_name,
        num_ap=num_ap,
        dt=dt,
        steps=steps,
        num_agents=num_agents,
    )
    solver_class = SOLVER_REGISTRY[solver_name]
    solver = solver_class(J=J, h=h, env=env, config=solver_config)

    tic = time.perf_counter()
    throughput_mbps, assignment, _, _ = solver.solve()
    wall_time_s = time.perf_counter() - tic
    normalized = float(throughput_mbps) / float(theo_max) if theo_max > 1e-12 else np.nan

    return {
        "num_ap": int(num_ap),
        "num_ch": int(env.NUM_CH),
        "num_bits": int(env.NUM_AP * env.NUM_CH),
        "solver": solver_name,
        "run_idx": int(run_idx),
        "A": float(A_BY_AP[int(num_ap)]) if solver_name in A_SOLVERS else np.nan,
        "dt": float(dt),
        "steps": int(steps),
        "num_agents": int(num_agents),
        "xi_seed": "unseeded",
        "theoretical_max_mbps": float(theo_max),
        "throughput_mbps": float(throughput_mbps),
        "normalized_throughput": float(normalized),
        "wall_time_s": float(wall_time_s),
        "assignment": repr(assignment),
    }


def summarize_best(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, _), group in raw_df.groupby(["num_ap", "solver"], sort=False):
        best = group.sort_values(
            ["normalized_throughput", "throughput_mbps"],
            ascending=[False, False],
        ).iloc[0]
        rows.append(
            {
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
                "mean_normalized_throughput": float(group["normalized_throughput"].mean()),
                "std_normalized_throughput": float(group["normalized_throughput"].std(ddof=1))
                if len(group) > 1
                else 0.0,
                "mean_wall_time_s": float(group["wall_time_s"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_grouped_bar(
    summary_df: pd.DataFrame,
    solvers: list[str],
    ap_list: list[int],
    result_dir: Path,
    show: bool,
) -> None:
    ordered_ap = [int(ap) for ap in ap_list]
    pivot = (
        summary_df.pivot(index="num_ap", columns="solver", values="best_normalized_throughput")
        .reindex(index=ordered_ap, columns=solvers)
        .dropna(how="all")
    )
    if pivot.empty:
        raise ValueError("No summary data available for plotting.")

    ap_values = pivot.index.to_list()
    x = np.arange(len(ap_values), dtype=float)
    width = min(0.82 / max(len(solvers), 1), 0.16)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.labelcolor": "#1F2933",
            "xtick.color": "#1F2933",
            "ytick.color": "#1F2933",
        }
    )
    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    fig.patch.set_facecolor("#F7F7F5")
    ax.set_facecolor("#F7F7F5")
    for solver_idx, solver_name in enumerate(solvers):
        if solver_name not in pivot:
            continue
        offsets = x + (solver_idx - (len(solvers) - 1) / 2.0) * width
        y = pivot[solver_name].to_numpy(dtype=float)
        bars = ax.bar(
            offsets,
            y,
            width=width,
            label=solver_name,
            color=SOLVER_COLORS.get(solver_name, "#4E79A7"),
            edgecolor="#303030",
            linewidth=0.5,
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xlabel("AP count")
    ax.set_ylabel("Throughput / theoretical maximum")
    ax.set_title("Best Normalized Throughput by Solver")
    ax.set_xticks(x)
    ax.set_xticklabels([str(ap) for ap in ap_values])
    y_max = np.nanmax(pivot.to_numpy(dtype=float))
    ax.set_ylim(0, max(1.0, y_max * 1.12))
    ax.grid(axis="y", linestyle="-", alpha=0.28, color="#AAB2BD", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        ncol=min(len(solvers), 4),
        frameon=False,
        fontsize=9.2,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.tight_layout()

    fig.savefig(result_dir / "solver_throughput_bar.png", bbox_inches="tight", dpi=240)
    fig.savefig(result_dir / "solver_throughput_bar.pdf", bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def save_runtime_config(
    result_dir: Path,
    config: dict[str, Any],
    *,
    ap_list: list[int],
    solvers: list[str],
    runs_per_solver: int,
    dt: float,
    steps: int,
    num_agents: int,
) -> None:
    runtime_config = {
        "experiment": {
            "ap_list": ap_list,
            "ap_seed": AP_SEED,
            "ue_seed": UE_SEED,
            "dt": dt,
            "steps": steps,
            "num_agents": num_agents,
            "runs_per_solver": runs_per_solver,
            "A_by_ap": A_BY_AP,
            "solvers": solvers,
            "xi_initialization": "random_uniform_inside_solver_without_fixed_seed",
            "best_policy": "best throughput over all runs for each AP and solver",
        },
        "project_config_used_for_unspecified_defaults": config,
    }
    with (result_dir / "experiment_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_config, f, allow_unicode=True, sort_keys=False)


def run_experiment(args: argparse.Namespace) -> Path:
    config = normalize_algorithm_keys(load_config(args.config))
    result_dir = make_result_dir()

    solvers = args.solvers or DEFAULT_SOLVERS
    ap_list = args.ap_list or AP_LIST
    # Keep user order while removing accidental duplicates.
    ap_list = list(dict.fromkeys(int(ap) for ap in ap_list))
    dt = args.dt
    steps = args.steps
    num_agents = args.num_agents
    runs_per_solver = args.runs

    if args.quick:
        solvers = ["SA", "Greedy", "GDSB"]
        ap_list = [10]
        steps = 10
        num_agents = 4
        runs_per_solver = 2

    missing_a = [ap for ap in ap_list if ap not in A_BY_AP]
    if missing_a:
        raise ValueError(f"A is not configured for AP counts: {missing_a}")

    save_runtime_config(
        result_dir,
        config,
        ap_list=ap_list,
        solvers=solvers,
        runs_per_solver=runs_per_solver,
        dt=dt,
        steps=steps,
        num_agents=num_agents,
    )

    print("=" * 88)
    print("WLAN solver throughput bar experiment")
    print(f"Result dir: {result_dir}")
    print(f"AP seed={AP_SEED}, UE seed={UE_SEED}")
    print(f"dt={dt}, steps={steps}, num_agents={num_agents}, runs={runs_per_solver}")
    print(f"AP list: {ap_list}")
    print(f"Solvers: {', '.join(solvers)}")
    print("RNG policy: solver xi initialization is random and unseeded")
    print("=" * 88)

    raw_rows: list[dict[str, Any]] = []
    for num_ap in ap_list:
        print(f"\nBuilding WLAN model for AP={num_ap}, A={A_BY_AP[num_ap]:.3g}")
        env = make_environment(config, num_ap)
        J, h = env.build_J_h()
        theo_max = env.compute_true_theoretical_max()
        print(f"J{J.shape}, h{h.shape}, theoretical max={theo_max:.6g} Mbps")

        for solver_name in solvers:
            print(f"  {solver_name}", end=" | ")
            for run_idx in range(1, runs_per_solver + 1):
                row = run_solver_once(
                    solver_name,
                    env,
                    J,
                    h,
                    theo_max,
                    config,
                    num_ap=num_ap,
                    run_idx=run_idx,
                    dt=dt,
                    steps=steps,
                    num_agents=num_agents,
                )
                raw_rows.append(row)
                print(f"run {run_idx}: {row['normalized_throughput']:.4f}", end="  ")
            print("")

    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize_best(raw_df)

    raw_df.to_csv(result_dir / "raw_solver_runs.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(result_dir / "best_solver_summary.csv", index=False, encoding="utf-8-sig")
    plot_grouped_bar(summary_df, solvers, ap_list, result_dir, show=args.show)

    print("\nBest normalized throughput:")
    printable = summary_df.pivot(
        index="num_ap",
        columns="solver",
        values="best_normalized_throughput",
    ).reindex(index=ap_list, columns=solvers)
    print(printable.to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved results to: {result_dir}")
    return result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare WLAN solvers with a grouped throughput bar chart.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ap-list", type=int, nargs="+", default=None)
    parser.add_argument("--solvers", choices=SOLVER_ORDER, nargs="+", default=None)
    parser.add_argument("--runs", type=int, default=RUNS_PER_SOLVER)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
