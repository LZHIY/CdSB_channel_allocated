# -*- coding: utf-8 -*-
"""Sweep UE density and compare solver throughput.

Experiment:
    Fix AP count, AP/UE seeds, max_dist, channel count, and solver parameters.
    Sweep num_ue_per_ap in [10, 15, 20, 25, 30].

By default AP=30, matching the recent AP=30 experiments. The AP count can be
changed from the command line if needed.

Run:
    python fig5_ue_count_solver_throughput_experiment.py

Quick smoke test:
    python fig5_ue_count_solver_throughput_experiment.py --quick
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
RESULT_ROOT = PROJECT_ROOT / "resultsUECountThroughput"

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


NUM_AP = 30
UE_PER_AP_LIST = [10, 15, 20, 25, 30]
AP_SEED = 15
UE_SEED = 16
MAX_DIST = 150.0
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
    result_dir = RESULT_ROOT / f"{timestamp}_{ms:03d}_ue_count_solver_throughput"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def make_environment(
    config: dict[str, Any],
    *,
    num_ap: int,
    num_ue_per_ap: int,
    max_dist: float,
) -> WLANIsingModel:
    env_config = config.get("environment", {})
    return WLANIsingModel(
        num_ap=int(num_ap),
        num_ch=int(env_config.get("num_ch", 3)),
        num_ue_per_ap=int(num_ue_per_ap),
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
    num_ap: int,
    num_ue_per_ap: int,
    max_dist: float,
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
        "num_ap": int(num_ap),
        "target_num_ue_per_ap": int(num_ue_per_ap),
        "total_ue": int(env.TOTAL_UE),
        "max_ue_count_on_ap": int(env.MAX_UE_COUNT),
        "mean_actual_ue_per_ap": float(np.mean(env.UE_COUNTS_PER_AP)),
        "max_dist": float(max_dist),
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
    for (_, _), group in raw_df.groupby(["target_num_ue_per_ap", "solver"], sort=False):
        best = group.sort_values(
            ["throughput_mbps", "normalized_throughput"],
            ascending=[False, False],
        ).iloc[0]
        rows.append(
            {
                "num_ap": int(best["num_ap"]),
                "target_num_ue_per_ap": int(best["target_num_ue_per_ap"]),
                "total_ue": int(best["total_ue"]),
                "max_ue_count_on_ap": int(best["max_ue_count_on_ap"]),
                "mean_actual_ue_per_ap": float(best["mean_actual_ue_per_ap"]),
                "max_dist": float(best["max_dist"]),
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
                "mean_wall_time_s": float(group["wall_time_s"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_ue_count_throughput(
    summary_df: pd.DataFrame,
    *,
    ue_per_ap_list: list[int],
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
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 8.8,
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for solver_name in solvers:
        solver_df = (
            summary_df[summary_df["solver"] == solver_name]
            .set_index("target_num_ue_per_ap")
            .reindex(ue_per_ap_list)
            .reset_index()
        )
        if solver_df["best_throughput_mbps"].isna().all():
            continue
        ax.plot(
            solver_df["target_num_ue_per_ap"],
            solver_df["best_throughput_mbps"],
            marker=SOLVER_MARKERS.get(solver_name, "o"),
            markersize=6.0,
            linewidth=1.75,
            color=SOLVER_COLORS.get(solver_name, "#4E79A7"),
            label=solver_name,
        )

    ax.set_xlabel("Target UE count per AP")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Solver Throughput under Increasing UE Density", fontsize=13.2, pad=10)
    ax.set_xticks(ue_per_ap_list)
    ax.grid(True, which="major", color="#D0D3D6", linewidth=0.72, alpha=0.78)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    png_path = result_dir / "ue_count_solver_throughput.png"
    pdf_path = result_dir / "ue_count_solver_throughput.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=260)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return png_path, pdf_path


def save_runtime_config(
    result_dir: Path,
    config: dict[str, Any],
    *,
    num_ap: int,
    ue_per_ap_list: list[int],
    max_dist: float,
    solvers: list[str],
    runs_per_solver: int,
    dt: float,
    steps: int,
    num_agents: int,
) -> None:
    runtime_config = {
        "script": Path(__file__).name,
        "experiment": {
            "num_ap": int(num_ap),
            "ue_per_ap_list": [int(x) for x in ue_per_ap_list],
            "ap_seed": AP_SEED,
            "ue_seed": UE_SEED,
            "max_dist": float(max_dist),
            "dt": float(dt),
            "steps": int(steps),
            "num_agents": int(num_agents),
            "runs_per_solver": int(runs_per_solver),
            "solvers": solvers,
            "A_by_ap": A_BY_AP,
            "best_policy": "best throughput over all runs for each UE density and solver",
            "visualization": "x=target UE count per AP, y=best throughput Mbps",
        },
        "project_config_used_for_unspecified_defaults": config,
    }
    with (result_dir / "experiment_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_config, f, allow_unicode=True, sort_keys=False)


def run_experiment(args: argparse.Namespace) -> Path:
    config = normalize_algorithm_keys(load_config(args.config))
    result_dir = make_result_dir()

    num_ap = int(args.num_ap)
    ue_per_ap_list = args.ue_per_ap_list or UE_PER_AP_LIST
    ue_per_ap_list = list(dict.fromkeys(int(x) for x in ue_per_ap_list))
    solvers = args.solvers or DEFAULT_SOLVERS
    runs_per_solver = int(args.runs)
    dt = float(args.dt)
    steps = int(args.steps)
    num_agents = int(args.num_agents)
    max_dist = float(args.max_dist)

    if args.quick:
        ue_per_ap_list = [10, 15]
        solvers = ["Greedy", "CIM", "GDSB"]
        runs_per_solver = 1
        steps = 5
        num_agents = 4

    if num_ap not in A_BY_AP and any(solver in A_SOLVERS for solver in solvers):
        raise ValueError(f"A is not configured for AP={num_ap}.")

    save_runtime_config(
        result_dir,
        config,
        num_ap=num_ap,
        ue_per_ap_list=ue_per_ap_list,
        max_dist=max_dist,
        solvers=solvers,
        runs_per_solver=runs_per_solver,
        dt=dt,
        steps=steps,
        num_agents=num_agents,
    )

    print("=" * 92)
    print("UE-count solver throughput experiment")
    print(f"Result dir: {result_dir}")
    print(f"AP={num_ap}, max_dist={max_dist:g}, AP seed={AP_SEED}, UE seed={UE_SEED}")
    print(f"UE-per-AP list: {ue_per_ap_list}")
    print(f"Solvers: {', '.join(solvers)}")
    print(f"dt={dt}, steps={steps}, agents={num_agents}, runs={runs_per_solver}")
    print("=" * 92)

    raw_rows: list[dict[str, Any]] = []
    for num_ue_per_ap in ue_per_ap_list:
        print(f"\nBuilding WLAN model: AP={num_ap}, target UE/AP={num_ue_per_ap}")
        env = make_environment(
            config,
            num_ap=num_ap,
            num_ue_per_ap=num_ue_per_ap,
            max_dist=max_dist,
        )
        J, h = env.build_J_h()
        theo_max = env.compute_true_theoretical_max()
        print(
            f"J{J.shape}, h{h.shape}, total UE={env.TOTAL_UE}, "
            f"max AP UE count={env.MAX_UE_COUNT}, theoretical max={theo_max:.6g} Mbps"
        )

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
                    num_ap=num_ap,
                    num_ue_per_ap=num_ue_per_ap,
                    max_dist=max_dist,
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

    raw_path = result_dir / "ue_count_raw_solver_runs.csv"
    summary_path = result_dir / "ue_count_best_solver_summary.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    png_path, pdf_path = plot_ue_count_throughput(
        summary_df,
        ue_per_ap_list=ue_per_ap_list,
        solvers=solvers,
        result_dir=result_dir,
    )

    print("\nBest throughput (Mbps):")
    printable = summary_df.pivot(
        index="target_num_ue_per_ap",
        columns="solver",
        values="best_throughput_mbps",
    ).reindex(index=ue_per_ap_list, columns=solvers)
    print(printable.to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved figure PNG to: {png_path}")
    print(f"Saved figure PDF to: {pdf_path}")
    return result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep UE density and compare solver throughput.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--num-ap", type=int, default=NUM_AP)
    parser.add_argument("--ue-per-ap-list", type=int, nargs="+", default=None)
    parser.add_argument("--max-dist", type=float, default=MAX_DIST)
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
