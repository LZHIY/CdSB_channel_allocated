# -*- coding: utf-8 -*-
"""AP=30 UE-seed TTS/throughput experiment.

Protocol:
1. Fix AP=30 and AP seed=15.
2. Sweep UE seeds. Each UE seed creates a new WLAN topology.
3. For each UE seed, run each solver for 100 samples.
   - CIM/BSB/DSB/GBSB/GDSB run once with num_agents=100 and expose 100
     parallel-agent samples.
   - SA has no parallel-agent implementation in this project, so if included
     it is run 100 independent serial solves to produce 100 samples.
4. For each UE seed, define Rbk as the best CIM throughput among CIM samples.
5. A sample is successful when Ralg / Rbk > 0.95.
6. Compute TTS with Ptarget=0.99.
7. Plot x=TTS(ms), y=throughput(Mbps).

Run:
    python fig4_ap30_ue_seed_tts_throughput.py

Quick smoke test:
    python fig4_ap30_ue_seed_tts_throughput.py --quick
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
RESULT_ROOT = PROJECT_ROOT / "resultsUESeedTTS"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.environment import WLANIsingModel
from solvers.bsb_solver import AdvancedbSBSolver
from solvers.cim_solver import ParallelCIMSolver
from solvers.dsb_solver import AdvanceddSBSolver
from solvers.gbsb_solver import ParallelGBSBSolver
from solvers.gdsb_solver import ParallelGDSBSolver
from solvers.sa_solver import SASolver


AP_COUNT = 30
AP_SEED = 15
UE_SEEDS = [16, 19, 20,34,234,432,1,213,343,453,545,654,73,85,933,1234,2345,3545,6765,6785,]
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
A_BY_AP = {30: 0.39}

SOLVER_COLORS = {
    "CIM": "#0072B2",
    "BSB": "#009E73",
    "DSB": "#332288",
    "GBSB": "#E69F00",
    "GDSB": "#CC6677",
    "SA": "#666666",
}
SOLVER_MARKERS = {
    "CIM": "o",
    "BSB": "s",
    "DSB": "^",
    "GBSB": "D",
    "GDSB": "P",
    "SA": "X",
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
    result_dir = RESULT_ROOT / f"{timestamp}_{ms:03d}_ap30_ue_seed_tts_throughput"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def make_environment(config: dict[str, Any], *, ue_seed: int) -> WLANIsingModel:
    env_config = config.get("environment", {})
    return WLANIsingModel(
        num_ap=AP_COUNT,
        num_ch=int(env_config.get("num_ch", 3)),
        num_ue_per_ap=int(env_config.get("num_ue_per_ap", 10)),
        ap_seed=AP_SEED,
        ue_seed=int(ue_seed),
        max_dist=float(env_config.get("max_dist", 150)),
        carrier_sense_range_m=float(env_config.get("carrier_sense_range_m", 70.0)),
    )


def solver_config_for(
    config: dict[str, Any],
    solver_name: str,
    *,
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

    if solver_name in A_SOLVERS:
        solver_config["A"] = float(A_BY_AP[AP_COUNT])

    return solver_config


def assignment_repr(assignment: Any) -> str:
    if assignment is None:
        return ""
    arr = np.asarray(assignment, dtype=int).reshape(-1)
    return " ".join(str(int(x)) for x in arr)


def make_sample_row(
    *,
    env: WLANIsingModel,
    ue_seed: int,
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
        "ap_seed": int(AP_SEED),
        "ue_seed": int(ue_seed),
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
    ue_seed: int,
    dt: float,
    steps: int,
    samples: int,
    sa_iter_per_temp: int,
) -> list[dict[str, Any]]:
    solver_config = solver_config_for(
        config,
        solver_name,
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
                ue_seed=ue_seed,
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
        print(f"WARNING: {solver_name} returned {len(rows)} samples, expected {samples}.")
    return rows


def run_sa_serial(
    env: WLANIsingModel,
    J: np.ndarray,
    h: np.ndarray,
    config: dict[str, Any],
    *,
    ue_seed: int,
    dt: float,
    steps: int,
    samples: int,
    sa_iter_per_temp: int,
) -> list[dict[str, Any]]:
    solver_config = solver_config_for(
        config,
        "SA",
        dt=dt,
        steps=steps,
        samples=samples,
        sa_iter_per_temp=sa_iter_per_temp,
    )

    rows = []
    measured_times = []
    batch_start = time.perf_counter()
    for sample_idx in range(1, samples + 1):
        solver = SASolver(J=J, h=h, env=env, config=solver_config)
        tic = time.perf_counter()
        throughput_mbps, assignment, _, _ = solver.solve()
        single_time = time.perf_counter() - tic
        measured_times.append(single_time)
        rows.append(
            make_sample_row(
                env=env,
                ue_seed=ue_seed,
                solver_name="SA",
                sample_idx=sample_idx,
                throughput_mbps=float(throughput_mbps),
                assignment=assignment,
                sample_source="serial_solve",
                batch_wall_time_s=np.nan,
                single_run_time_s=single_time,
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
        row["measured_serial_solve_time_s"] = float(measured_times[row["sample_idx"] - 1])
    return rows


def tts_seconds(p_success: float, single_run_time_s: float, p_target: float) -> float:
    if p_success <= 0.0:
        return float("inf")
    if p_success >= 1.0:
        return float(single_run_time_s)
    return float(math.log(1.0 - p_target) / math.log(1.0 - p_success) * single_run_time_s)


def mark_success_with_cim_rbk(raw_df: pd.DataFrame, success_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    marked_groups = []
    rbk_rows = []

    for ue_seed, group in raw_df.groupby("ue_seed", sort=False):
        cim_group = group[group["solver"] == "CIM"]
        if cim_group.empty:
            raise ValueError(f"UE seed={ue_seed} has no CIM samples, cannot define Rbk.")

        rbk_sample = cim_group.sort_values("throughput_mbps", ascending=False).iloc[0]
        max_sample = group.sort_values("throughput_mbps", ascending=False).iloc[0]
        rbk_mbps = float(rbk_sample["throughput_mbps"])

        marked = group.copy()
        marked["rbk_mbps"] = rbk_mbps
        marked["rbk_solver"] = "CIM"
        marked["rbk_sample_idx"] = int(rbk_sample["sample_idx"])
        marked["rbk_assignment"] = str(rbk_sample["assignment"])
        marked["rbk_ratio"] = marked["throughput_mbps"] / rbk_mbps if rbk_mbps > 1e-12 else np.nan
        marked["success_threshold_ratio"] = float(success_ratio)
        marked["success"] = marked["rbk_ratio"] > float(success_ratio)
        marked["is_rbk"] = marked.index == rbk_sample.name
        marked_groups.append(marked)

        rbk_rows.append(
            {
                "num_ap": AP_COUNT,
                "ap_seed": AP_SEED,
                "ue_seed": int(ue_seed),
                "rbk_mbps": rbk_mbps,
                "rbk_solver": "CIM",
                "rbk_sample_idx": int(rbk_sample["sample_idx"]),
                "rbk_assignment": str(rbk_sample["assignment"]),
                "success_threshold_ratio": float(success_ratio),
                "max_sample_mbps": float(max_sample["throughput_mbps"]),
                "max_sample_solver": str(max_sample["solver"]),
                "max_sample_idx": int(max_sample["sample_idx"]),
                "max_sample_to_rbk_ratio": float(max_sample["throughput_mbps"]) / rbk_mbps
                if rbk_mbps > 1e-12
                else np.nan,
            }
        )

    return pd.concat(marked_groups, ignore_index=True), pd.DataFrame(rbk_rows)


def summarize_tts(marked_df: pd.DataFrame, *, p_target: float) -> pd.DataFrame:
    rows = []
    group_cols = ["ue_seed", "solver"]
    for (ue_seed, solver_name), group in marked_df.groupby(group_cols, sort=False):
        trials = int(len(group))
        success_count = int(group["success"].sum())
        p_success = success_count / trials if trials else 0.0
        single_run_time_s = float(group["single_run_time_s"].mean())
        batch_wall_time_s = float(group["batch_wall_time_s"].mean())
        tts_s = tts_seconds(p_success, single_run_time_s, p_target)
        best = group.sort_values("throughput_mbps", ascending=False).iloc[0]

        rows.append(
            {
                "num_ap": AP_COUNT,
                "ap_seed": AP_SEED,
                "ue_seed": int(ue_seed),
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

    return pd.DataFrame(rows).sort_values(["ue_seed", "solver_order"]).reset_index(drop=True)


def plot_tts_vs_throughput(
    summary_df: pd.DataFrame,
    result_dir: Path,
    *,
    throughput_col: str,
    annotate: bool,
) -> tuple[Path, Path]:
    df = summary_df.copy()
    df["tts_ms"] = pd.to_numeric(df["tts_ms"], errors="coerce")
    df[throughput_col] = pd.to_numeric(df[throughput_col], errors="coerce")
    finite_mask = np.isfinite(df["tts_ms"]) & (df["tts_ms"] > 0.0) & np.isfinite(df[throughput_col])
    plot_df = df[finite_mask].copy()
    skipped_df = df[~finite_mask].copy()
    if plot_df.empty:
        raise ValueError("No finite TTS/throughput points available for plotting.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.edgecolor": "#111111",
            "axes.linewidth": 0.9,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
        }
    )
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for solver_name in SOLVERS:
        solver_df = plot_df[plot_df["solver"] == solver_name]
        if solver_df.empty:
            continue
        color = SOLVER_COLORS.get(solver_name, "#4E79A7")
        marker = SOLVER_MARKERS.get(solver_name, "o")

        ax.scatter(
            solver_df["tts_ms"],
            solver_df[throughput_col],
            s=46,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            alpha=0.58,
            label=solver_name,
            zorder=3,
        )

        x_values = solver_df["tts_ms"].to_numpy(dtype=float)
        y_values = solver_df[throughput_col].to_numpy(dtype=float)
        x_med = float(np.median(x_values))
        y_med = float(np.median(y_values))
        x_q25, x_q75 = np.percentile(x_values, [25, 75])
        y_q25, y_q75 = np.percentile(y_values, [25, 75])
        xerr = np.array([[max(x_med - x_q25, 0.0)], [max(x_q75 - x_med, 0.0)]])
        yerr = np.array([[max(y_med - y_q25, 0.0)], [max(y_q75 - y_med, 0.0)]])

        ax.errorbar(
            [x_med],
            [y_med],
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            markersize=8.8,
            markerfacecolor=color,
            markeredgecolor="#111111",
            markeredgewidth=0.8,
            ecolor=color,
            elinewidth=1.15,
            capsize=3.0,
            alpha=0.95,
            zorder=4,
        )

        if annotate:
            ax.annotate(
                solver_name,
                (x_med, y_med),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9.0,
                color="#111111",
            )

    ax.set_xscale("log")
    ax.set_xlabel("TTS (ms)")
    ax.set_ylabel("Throughput (Mbps)" if throughput_col == "best_solver_sample_mbps" else f"{throughput_col.replace('_', ' ')} (Mbps)")
    ax.set_title("AP=30, UE-seed sweep", fontsize=12.5, pad=8)
    ax.grid(True, which="major", axis="both", color="#D0D3D6", linewidth=0.72, alpha=0.78)
    ax.grid(True, which="minor", axis="x", color="#E5E8EB", linewidth=0.45, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Solver",
        frameon=False,
        loc="best",
        handletextpad=0.4,
        borderaxespad=0.6,
    )
    ax.text(
        0.015,
        0.015,
        "Transparent markers: topology trials\nOutlined markers/error bars: median and IQR",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#333333",
    )

    fig.tight_layout()
    suffix = throughput_col.replace("_mbps", "")
    png_path = result_dir / f"ap30_ue_seed_tts_vs_{suffix}_academic.png"
    pdf_path = result_dir / f"ap30_ue_seed_tts_vs_{suffix}_academic.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=260)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    if not skipped_df.empty:
        print(f"Skipped {len(skipped_df)} non-finite points, usually TTS=inf.")
    return png_path, pdf_path


def save_runtime_config(
    result_dir: Path,
    config: dict[str, Any],
    *,
    ue_seeds: list[int],
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
            "num_ap": AP_COUNT,
            "ap_seed": AP_SEED,
            "ue_seeds": [int(seed) for seed in ue_seeds],
            "solvers": solvers,
            "samples_per_solver": int(samples),
            "parallel_solvers": sorted(PARALLEL_SOLVERS),
            "serial_solvers": ["SA"] if "SA" in solvers else [],
            "dt": float(dt),
            "steps": int(steps),
            "sa_iter_per_temp": int(sa_iter_per_temp),
            "p_target": float(p_target),
            "success_rule": "Ralg / Rbk > success_ratio",
            "success_ratio": float(success_ratio),
            "rbk_definition": "best CIM throughput among CIM samples for the same UE seed",
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

    ue_seeds = args.ue_seeds or UE_SEEDS
    ue_seeds = list(dict.fromkeys(int(seed) for seed in ue_seeds))
    solvers = args.solvers or SOLVERS
    samples = int(args.samples)
    dt = float(args.dt)
    steps = int(args.steps)
    p_target = float(args.p_target)
    success_ratio = float(args.success_ratio)
    sa_iter_per_temp = int(args.sa_iter_per_temp)

    if args.quick:
        ue_seeds = [16]
        samples = 3
        steps = 5
        sa_iter_per_temp = 1

    if "CIM" not in solvers:
        raise ValueError("CIM must be included because Rbk is defined as best CIM throughput.")

    save_runtime_config(
        result_dir,
        config,
        ue_seeds=ue_seeds,
        solvers=solvers,
        samples=samples,
        dt=dt,
        steps=steps,
        p_target=p_target,
        success_ratio=success_ratio,
        sa_iter_per_temp=sa_iter_per_temp,
    )

    print("=" * 92)
    print("AP=30 UE-seed TTS/throughput experiment")
    print(f"Result dir: {result_dir}")
    print(f"AP seed={AP_SEED}, UE seeds={ue_seeds}")
    print(f"solvers={', '.join(solvers)}, samples={samples}, steps={steps}, dt={dt}")
    print(f"Rbk: best CIM throughput per UE seed; success: Ralg/Rbk > {success_ratio}")
    print("=" * 92)

    raw_rows: list[dict[str, Any]] = []
    for ue_seed in ue_seeds:
        print(f"\nBuilding WLAN model for AP={AP_COUNT}, UE seed={ue_seed}")
        env = make_environment(config, ue_seed=ue_seed)
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
                    ue_seed=ue_seed,
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
                    ue_seed=ue_seed,
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
    marked_df, rbk_df = mark_success_with_cim_rbk(raw_df, success_ratio)
    summary_df = summarize_tts(marked_df, p_target=p_target)

    raw_path = result_dir / "ap30_ue_seed_raw_solver_samples.csv"
    rbk_path = result_dir / "ap30_ue_seed_rbk.csv"
    summary_path = result_dir / "ap30_ue_seed_tts_summary.csv"
    marked_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    rbk_df.to_csv(rbk_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if not args.no_plot:
        png_path, pdf_path = plot_tts_vs_throughput(
            summary_df,
            result_dir,
            throughput_col=args.throughput_col,
            annotate=args.annotate,
        )
        print(f"\nSaved scatter PNG to: {png_path}")
        print(f"Saved scatter PDF to: {pdf_path}")

    print("\nRbk by UE seed:")
    print(rbk_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nTTS summary:")
    print(
        summary_df[
            ["ue_seed", "solver", "success_count", "p_success", "best_solver_sample_mbps", "tts_ms"]
        ].to_string(index=False, float_format=lambda x: f"{x:.6g}")
    )
    print(f"\nSaved raw samples to: {raw_path}")
    print(f"Saved summary to: {summary_path}")
    return result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AP=30 UE-seed TTS/throughput experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ue-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--solvers", choices=SOLVERS, nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_SOLVER)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--sa-iter-per-temp", type=int, default=SA_ITER_PER_TEMP)
    parser.add_argument("--p-target", type=float, default=P_TARGET)
    parser.add_argument("--success-ratio", type=float, default=SUCCESS_RATIO)
    parser.add_argument(
        "--throughput-col",
        choices=["best_solver_sample_mbps", "mean_throughput_mbps", "rbk_mbps"],
        default="best_solver_sample_mbps",
    )
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
