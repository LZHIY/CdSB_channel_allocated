# -*- coding: utf-8 -*-
"""Rbk-based TTS benchmark for WLAN solvers.

For each AP scenario, this script follows one unified protocol:

1. CIM, BSB, DSB, GBSB, and GDSB each call solve() once with 100 parallel
   agents. Their per-agent throughputs are collected as 100 results each.
2. SA calls solve() 100 times serially, collecting one result after each call.
3. The best CIM throughput for the same AP scenario is defined as Rbk.
4. Each sample succeeds when Ralg / Rbk > success_ratio.
5. The single-trial runtime for parallel solvers is batch_time / 100. The
   single-trial runtime for SA is the average serial solve time.
6. TTS uses Ptarget = 0.99.

Run from project root:
    python fig3_solver_tts_rbk_experiment.py

Quick smoke test:
    python fig3_solver_tts_rbk_experiment.py --quick
"""

from __future__ import annotations

import argparse
import copy
import math
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
RESULT_ROOT = PROJECT_ROOT / "resultsTTS"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.environment import WLANIsingModel
from solvers.bsb_solver import AdvancedbSBSolver
from solvers.cim_solver import ParallelCIMSolver
from solvers.dsb_solver import AdvanceddSBSolver
from solvers.gbsb_solver import ParallelGBSBSolver
from solvers.gdsb_solver import ParallelGDSBSolver
from solvers.sa_solver import SASolver


AP_LIST = [10, 20, 30, 40, 50]
AP_SEED = 15
UE_SEED = 16
DT = 0.15
STEPS = 600
SAMPLES_PER_SOLVER = 100
P_TARGET = 0.99
SUCCESS_RATIO = 0.95
SA_ITER_PER_TEMP = 20

SOLVERS = ["CIM", "BSB", "DSB", "GBSB", "GDSB", "SA"]
PARALLEL_SOLVERS = {"CIM", "BSB", "DSB", "GBSB", "GDSB"}
SOLVER_REGISTRY = {
    "CIM": ParallelCIMSolver,
    "BSB": AdvancedbSBSolver,
    "DSB": AdvanceddSBSolver,
    "GBSB": ParallelGBSBSolver,
    "GDSB": ParallelGDSBSolver,
    "SA": SASolver,
}
A_SOLVERS = {"GBSB", "GDSB"}
A_BY_AP = {10: 0.30, 20: 0.49, 30: 0.39, 40: 0.59, 50: 0.63}
SOLVER_COLORS = {
    "CIM": "#457B9D",
    "BSB": "#2A9D8F",
    "DSB": "#1D3557",
    "GBSB": "#E9C46A",
    "GDSB": "#D1495B",
    "SA": "#6C7A89",
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
    result_dir = RESULT_ROOT / f"{timestamp}_{ms:03d}_solver_tts_rbk_600"
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
    samples: int,
    sa_iter_per_temp: int,
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
        "steps": int(steps),
        "dt": float(dt),
    }

    if solver_name == "SA":
        solver_config["num_agents"] = 1
        solver_config["iter_per_temp"] = int(sa_iter_per_temp)
        t_start = float(solver_config.get("t_start", 10.0))
        t_end = float(solver_config.get("t_end", 1e-5))
        if steps > 0 and t_start > t_end > 0.0:
            solver_config["cooling_rate"] = float((t_end / t_start) ** (1.0 / int(steps)))
    else:
        solver_config["num_agents"] = int(samples)

    if solver_name in A_SOLVERS and int(num_ap) in A_BY_AP:
        solver_config["A"] = float(A_BY_AP[int(num_ap)])

    return solver_config


def assignment_repr(assignment: Any) -> str:
    if assignment is None:
        return ""
    arr = np.asarray(assignment, dtype=int).reshape(-1)
    return " ".join(str(int(x)) for x in arr)


def make_sample_row(
    *,
    env: WLANIsingModel,
    solver_name: str,
    sample_idx: int,
    throughput_mbps: float,
    assignment: Any,
    sample_source: str,
    batch_wall_time_s: float,
    single_run_time_s: float,
    dt: float,
    steps: int,
    samples: int,
) -> dict[str, Any]:
    return {
        "num_ap": int(env.NUM_AP),
        "num_ch": int(env.NUM_CH),
        "num_bits": int(env.NUM_AP * env.NUM_CH),
        "solver": solver_name,
        "sample_idx": int(sample_idx),
        "sample_source": sample_source,
        "samples_per_solver": int(samples),
        "dt": float(dt),
        "steps": int(steps),
        "throughput_mbps": float(throughput_mbps),
        "batch_wall_time_s": float(batch_wall_time_s),
        "single_run_time_s": float(single_run_time_s),
        "assignment": assignment_repr(assignment),
    }


def run_parallel_solver(
    solver_name: str,
    env: WLANIsingModel,
    J: np.ndarray,
    h: np.ndarray,
    config: dict[str, Any],
    *,
    num_ap: int,
    dt: float,
    steps: int,
    samples: int,
    sa_iter_per_temp: int,
) -> list[dict[str, Any]]:
    solver_config = solver_config_for(
        config,
        solver_name,
        num_ap=num_ap,
        dt=dt,
        steps=steps,
        samples=samples,
        sa_iter_per_temp=sa_iter_per_temp,
    )
    solver = SOLVER_REGISTRY[solver_name](J=J, h=h, env=env, config=solver_config)

    tic = time.perf_counter()
    best_tp, best_assignment, _, _ = solver.solve()
    batch_wall_time_s = time.perf_counter() - tic

    sample_throughputs = list(getattr(solver, "last_sample_throughputs", []))
    sample_assignments = list(getattr(solver, "last_sample_assignments", []))
    if not sample_throughputs:
        sample_throughputs = [float(best_tp)]
        sample_assignments = [best_assignment]

    single_run_time_s = batch_wall_time_s / max(1, len(sample_throughputs))
    rows = []
    for sample_idx, throughput_mbps in enumerate(sample_throughputs, start=1):
        assignment = sample_assignments[sample_idx - 1] if sample_idx - 1 < len(sample_assignments) else None
        rows.append(
            make_sample_row(
                env=env,
                solver_name=solver_name,
                sample_idx=sample_idx,
                throughput_mbps=float(throughput_mbps),
                assignment=assignment,
                sample_source="parallel_agent",
                batch_wall_time_s=batch_wall_time_s,
                single_run_time_s=single_run_time_s,
                dt=dt,
                steps=steps,
                samples=samples,
            )
        )

    if len(rows) != samples:
        print(
            f"WARNING: {solver_name} returned {len(rows)} samples, expected {samples}.",
            flush=True,
        )
    return rows


def run_sa_serial(
    env: WLANIsingModel,
    J: np.ndarray,
    h: np.ndarray,
    config: dict[str, Any],
    *,
    num_ap: int,
    dt: float,
    steps: int,
    samples: int,
    sa_iter_per_temp: int,
) -> list[dict[str, Any]]:
    solver_config = solver_config_for(
        config,
        "SA",
        num_ap=num_ap,
        dt=dt,
        steps=steps,
        samples=samples,
        sa_iter_per_temp=sa_iter_per_temp,
    )

    rows = []
    per_sample_times = []
    batch_start = time.perf_counter()
    for sample_idx in range(1, samples + 1):
        solver = SASolver(J=J, h=h, env=env, config=solver_config)
        tic = time.perf_counter()
        throughput_mbps, assignment, _, _ = solver.solve()
        single_run_time_s = time.perf_counter() - tic
        per_sample_times.append(single_run_time_s)
        rows.append(
            make_sample_row(
                env=env,
                solver_name="SA",
                sample_idx=sample_idx,
                throughput_mbps=float(throughput_mbps),
                assignment=assignment,
                sample_source="serial_solve",
                batch_wall_time_s=np.nan,
                single_run_time_s=single_run_time_s,
                dt=dt,
                steps=steps,
                samples=samples,
            )
        )
    batch_wall_time_s = time.perf_counter() - batch_start
    average_single_run_time_s = batch_wall_time_s / max(1, samples)

    for row in rows:
        row["batch_wall_time_s"] = float(batch_wall_time_s)
        row["single_run_time_s"] = float(average_single_run_time_s)
        row["measured_serial_solve_time_s"] = float(per_sample_times[row["sample_idx"] - 1])

    return rows


def tts_seconds(p_success: float, single_run_time_s: float, p_target: float) -> float:
    if p_success <= 0.0:
        return float("inf")
    if p_success >= 1.0:
        return float(single_run_time_s)
    return float(math.log(1.0 - p_target) / math.log(1.0 - p_success) * single_run_time_s)


def mark_rbk_success(raw_df: pd.DataFrame, success_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    rbk_rows = []

    for num_ap, group in raw_df.groupby("num_ap", sort=False):
        max_sample = group.sort_values(["throughput_mbps"], ascending=False).iloc[0]
        cim_group = group[group["solver"] == "CIM"]
        if cim_group.empty:
            raise ValueError(
                f"Rbk is defined as the best CIM result, but AP={num_ap} has no CIM samples."
            )

        rbk_sample = cim_group.sort_values(["throughput_mbps"], ascending=False).iloc[0]
        rbk_mbps = float(rbk_sample["throughput_mbps"])
        rbk_solver = str(rbk_sample["solver"])
        rbk_sample_idx = int(rbk_sample["sample_idx"])
        rbk_assignment = str(rbk_sample["assignment"])

        marked = group.copy()
        marked["rbk_mbps"] = rbk_mbps
        marked["rbk_solver"] = rbk_solver
        marked["rbk_sample_idx"] = rbk_sample_idx
        marked["rbk_ratio"] = marked["throughput_mbps"] / rbk_mbps if rbk_mbps > 1e-12 else np.nan
        marked["success_threshold_ratio"] = float(success_ratio)
        marked["success"] = marked["rbk_ratio"] > float(success_ratio)
        marked["is_rbk"] = marked.index == rbk_sample.name
        frames.append(marked)

        out_row = {
            "num_ap": int(num_ap),
            "rbk_mbps": rbk_mbps,
            "rbk_solver": rbk_solver,
            "rbk_sample_idx": rbk_sample_idx,
            "rbk_assignment": rbk_assignment,
            "success_threshold_ratio": float(success_ratio),
            "max_sample_mbps": float(max_sample["throughput_mbps"]),
            "max_sample_solver": str(max_sample["solver"]),
            "max_sample_idx": int(max_sample["sample_idx"]),
            "max_sample_to_rbk_ratio": float(max_sample["throughput_mbps"]) / rbk_mbps
            if rbk_mbps > 1e-12
            else np.nan,
        }
        rbk_rows.append(out_row)

    return pd.concat(frames, ignore_index=True), pd.DataFrame(rbk_rows)


def summarize_tts(raw_df: pd.DataFrame, *, p_target: float) -> pd.DataFrame:
    rows = []
    for (num_ap, solver_name), group in raw_df.groupby(["num_ap", "solver"], sort=False):
        trials = int(len(group))
        success_count = int(group["success"].sum())
        p_success = success_count / trials if trials else 0.0
        single_run_time_s = float(group["single_run_time_s"].mean())
        batch_wall_time_s = float(group["batch_wall_time_s"].mean())
        tts_s = tts_seconds(p_success, single_run_time_s, p_target)
        best = group.sort_values(["throughput_mbps"], ascending=False).iloc[0]

        rows.append(
            {
                "num_ap": int(num_ap),
                "solver": str(solver_name),
                "solver_order": SOLVERS.index(str(solver_name)) if str(solver_name) in SOLVERS else 999,
                "trials": trials,
                "success_count": success_count,
                "p_success": float(p_success),
                "single_run_time_s": single_run_time_s,
                "batch_wall_time_s": batch_wall_time_s,
                "rbk_mbps": float(group["rbk_mbps"].iloc[0]),
                "best_solver_sample_mbps": float(best["throughput_mbps"]),
                "best_solver_sample_ratio_to_rbk": float(best["rbk_ratio"]),
                "mean_throughput_mbps": float(group["throughput_mbps"].mean()),
                "std_throughput_mbps": float(group["throughput_mbps"].std(ddof=1)) if trials > 1 else 0.0,
                "tts_s": tts_s,
                "tts_ms": float(tts_s * 1000.0) if np.isfinite(tts_s) else float("inf"),
            }
        )

    return pd.DataFrame(rows).sort_values(["num_ap", "solver_order"]).reset_index(drop=True)


def plot_tts(summary_df: pd.DataFrame, solvers: list[str], ap_list: list[int], result_dir: Path) -> None:
    pivot = (
        summary_df.pivot(index="num_ap", columns="solver", values="tts_ms")
        .reindex(index=ap_list, columns=solvers)
        .dropna(how="all")
    )
    if pivot.empty:
        return

    finite_values = pivot.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values) & (finite_values > 0)]
    y_min = max(float(finite_values.min()) * 0.6, 1e-3) if finite_values.size else 1e-3
    y_max = float(finite_values.max()) * 1.8 if finite_values.size else 1.0
    y_max = max(y_max, y_min * 10.0)

    x = np.arange(len(pivot.index), dtype=float)
    width = min(0.82 / max(len(solvers), 1), 0.16)

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    fig.patch.set_facecolor("#F7F7F5")
    ax.set_facecolor("#F7F7F5")

    for solver_idx, solver_name in enumerate(solvers):
        if solver_name not in pivot:
            continue
        offsets = x + (solver_idx - (len(solvers) - 1) / 2.0) * width
        values = pivot[solver_name].to_numpy(dtype=float)
        plot_values = np.where(np.isfinite(values), values, y_max)
        bars = ax.bar(
            offsets,
            plot_values,
            width=width,
            label=solver_name,
            color=SOLVER_COLORS.get(solver_name, "#4E79A7"),
            edgecolor="#303030",
            linewidth=0.5,
        )
        labels = ["inf" if not np.isfinite(v) else f"{v:.3g}" for v in values]
        ax.bar_label(bars, labels=labels, padding=2, fontsize=8)

    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("AP count")
    ax.set_ylabel("TTS (ms), Ptarget=0.99")
    ax.set_title("Rbk-based TTS by Solver")
    ax.set_xticks(x)
    ax.set_xticklabels([str(ap) for ap in pivot.index])
    ax.grid(axis="y", which="both", alpha=0.32, color="#AAB2BD", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=min(len(solvers), 6), frameon=False, fontsize=9.2)
    fig.tight_layout()
    fig.savefig(result_dir / "solver_tts_rbk_bar.png", bbox_inches="tight", dpi=240)
    fig.savefig(result_dir / "solver_tts_rbk_bar.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_runtime_config(
    result_dir: Path,
    config: dict[str, Any],
    *,
    ap_list: list[int],
    solvers: list[str],
    samples: int,
    dt: float,
    steps: int,
    p_target: float,
    success_ratio: float,
    sa_iter_per_temp: int,
) -> None:
    runtime_config = {
        "script": Path(__file__).name,
        "experiment": {
            "ap_list": [int(ap) for ap in ap_list],
            "ap_seed": AP_SEED,
            "ue_seed": UE_SEED,
            "solvers": solvers,
            "samples_per_solver": int(samples),
            "total_samples_per_ap": int(samples * len(solvers)),
            "parallel_solvers": sorted(PARALLEL_SOLVERS),
            "serial_solvers": ["SA"],
            "dt": float(dt),
            "steps": int(steps),
            "sa_iter_per_temp": int(sa_iter_per_temp),
            "p_target": float(p_target),
            "success_rule": "Ralg / Rbk > success_ratio",
            "success_ratio": float(success_ratio),
            "rbk_definition": "best CIM throughput among CIM samples for the same AP scenario",
            "time_rule": "parallel single_run_time = batch_wall_time / samples; SA single_run_time = serial batch_wall_time / samples",
            "A_by_ap": A_BY_AP,
        },
        "project_config_used_for_unspecified_defaults": config,
    }
    with (result_dir / "experiment_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_config, f, allow_unicode=True, sort_keys=False)


def run_experiment(args: argparse.Namespace) -> Path:
    config = normalize_algorithm_keys(load_config(args.config))
    result_dir = make_result_dir()

    ap_list = args.ap_list or AP_LIST
    ap_list = list(dict.fromkeys(int(ap) for ap in ap_list))
    solvers = args.solvers or SOLVERS
    samples = int(args.samples)
    dt = float(args.dt)
    steps = int(args.steps)
    p_target = float(args.p_target)
    success_ratio = float(args.success_ratio)
    sa_iter_per_temp = int(args.sa_iter_per_temp)

    if args.quick:
        ap_list = [4]
        solvers = ["CIM", "BSB", "DSB", "GBSB", "GDSB", "SA"]
        samples = 3
        steps = 5
        sa_iter_per_temp = 1

    save_runtime_config(
        result_dir,
        config,
        ap_list=ap_list,
        solvers=solvers,
        samples=samples,
        dt=dt,
        steps=steps,
        p_target=p_target,
        success_ratio=success_ratio,
        sa_iter_per_temp=sa_iter_per_temp,
    )

    print("=" * 88)
    print("Rbk-based TTS benchmark")
    print(f"Result dir: {result_dir}")
    print(f"AP list: {ap_list}")
    print(f"Solvers: {', '.join(solvers)}")
    print(f"samples_per_solver={samples}, total_per_ap={samples * len(solvers)}")
    print(f"success: Ralg / Rbk > {success_ratio}, Ptarget={p_target}")
    print("=" * 88)

    raw_rows: list[dict[str, Any]] = []
    for num_ap in ap_list:
        print(f"\nBuilding WLAN model for AP={num_ap}")
        env = make_environment(config, num_ap)
        J, h = env.build_J_h()
        print(f"J{J.shape}, h{h.shape}")

        for solver_name in solvers:
            print(f"  Running {solver_name}", end="", flush=True)
            if solver_name in PARALLEL_SOLVERS:
                rows = run_parallel_solver(
                    solver_name,
                    env,
                    J,
                    h,
                    config,
                    num_ap=num_ap,
                    dt=dt,
                    steps=steps,
                    samples=samples,
                    sa_iter_per_temp=sa_iter_per_temp,
                )
            elif solver_name == "SA":
                rows = run_sa_serial(
                    env,
                    J,
                    h,
                    config,
                    num_ap=num_ap,
                    dt=dt,
                    steps=steps,
                    samples=samples,
                    sa_iter_per_temp=sa_iter_per_temp,
                )
            else:
                raise ValueError(f"Unknown solver: {solver_name}")

            raw_rows.extend(rows)
            best_tp = max(row["throughput_mbps"] for row in rows)
            batch_time = float(rows[0]["batch_wall_time_s"]) if rows else 0.0
            single_time = float(rows[0]["single_run_time_s"]) if rows else 0.0
            print(
                f" | samples={len(rows)}, best={best_tp:.6g} Mbps, "
                f"batch={batch_time:.4g}s, single={single_time:.4g}s"
            )

    raw_df = pd.DataFrame(raw_rows)
    raw_df, rbk_df = mark_rbk_success(raw_df, success_ratio)
    summary_df = summarize_tts(raw_df, p_target=p_target)

    raw_path = result_dir / "tts_rbk_raw_600_results.csv"
    rbk_path = result_dir / "tts_rbk_by_ap.csv"
    summary_path = result_dir / "tts_rbk_summary.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    rbk_df.to_csv(rbk_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if not args.no_plot:
        plot_tts(summary_df, solvers, ap_list, result_dir)

    print("\nRbk by AP:")
    print(rbk_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nTTS summary:")
    printable = summary_df[
        [
            "num_ap",
            "solver",
            "trials",
            "success_count",
            "p_success",
            "single_run_time_s",
            "tts_ms",
        ]
    ]
    print(printable.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {summary_path}")
    return result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rbk-based WLAN TTS benchmark.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ap-list", type=int, nargs="+", default=None)
    parser.add_argument("--solvers", choices=SOLVERS, nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_SOLVER)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--sa-iter-per-temp", type=int, default=SA_ITER_PER_TEMP)
    parser.add_argument("--p-target", type=float, default=P_TARGET)
    parser.add_argument("--success-ratio", type=float, default=SUCCESS_RATIO)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
