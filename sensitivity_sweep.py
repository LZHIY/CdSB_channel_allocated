# -*- coding: utf-8 -*-
"""
敏感性分析扫描脚本 —— 基于论文原始 802.11g 代码 (core/environment.py + solvers/gdsb_solver.py)
====================================================================================
目的：验证 CdSB(=GDSB) 结果对关键参数的稳健性，回应 Reviewer 2 Comment 6
      (A sensitivity analysis could be beneficial for the results.)。

基准配置（与论文最终数据 raw_solver_runs.csv 一致）：
    M=30, K=3, UE/AP=10, AP_SEED=15, UE_SEED=16, max_dist=150
    dt=0.15, steps=900, num_agents=300, A=0.39
    zeta=0.6, gamma=0.1, V1_SCALE=1.0, V2_SCALE=100.0, d_CS=70 m

扫描（每次只改一个参数，其余保持基准；每点重复 repeat 次以平滑随机初始化）：
    1) A      ∈ {0.29, 0.34, 0.39, 0.44, 0.49}          (基准 0.39)
    2) dt     ∈ {0.05, 0.10, 0.15, 0.20}                  (基准 0.15)
    3) steps  ∈ {600, 900, 1200}                          (基准 900)
    4) V2/V1  ∈ {50, 100, 200, 500}  (V1=1.0 固定)        (基准 100)
    5) d_CS   ∈ {50, 60, 70, 80, 90} m                    (基准 70 m)

输出：sensitivity_results.csv
    每行：sweep_param, value, repeat, best_mbps_mean, best_mbps_max,
          mean_mbps_mean, std_mbps_mean, wall_s_mean, theo_max_mbps,
          normalized_best_mean

运行（请放到项目根目录，与 main_benchmark.py 同级）：
    python sensitivity_sweep.py
    python sensitivity_sweep.py --quick          # 快速冒烟测试 (agents=50, repeat=1)

预计耗时：完整扫描 21 点 × 3 次 ≈ 20~40 分钟（取决于机器）。
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from core.environment import WLANIsingModel
from solvers.gdsb_solver import ParallelGDSBSolver

# ---------------- 基准配置（与论文最终数据一致）----------------
BASE = dict(
    num_ap=30,
    num_ch=3,
    num_ue_per_ap=10,
    ap_seed=15,
    ue_seed=16,
    max_dist=150,
    A=0.39,
    dt=0.15,
    steps=900,
    num_agents=300,          # 注意：raw_solver_runs.csv 记录为 300；
                             # 若你的生成脚本实际用 100，请改回 100。
    zeta=0.6,
    gamma=0.1,
    V1=1.0,
    V2=100.0,
    d_cs=70.0,
)


def make_env_and_jh(*, V1: float, V2: float, d_cs: float):
    env = WLANIsingModel(
        num_ap=BASE["num_ap"],
        num_ch=BASE["num_ch"],
        num_ue_per_ap=BASE["num_ue_per_ap"],
        ap_seed=BASE["ap_seed"],
        ue_seed=BASE["ue_seed"],
        max_dist=BASE["max_dist"],
        carrier_sense_range_m=d_cs,
    )
    env.V1_SCALE = V1
    env.V2_SCALE = V2
    J, h = env.build_J_h()
    theo = env.compute_true_theoretical_max()
    return env, J, h, theo


def solve_once(env, J, h, theo, *, A: float, dt: float, steps: int, num_agents: int):
    cfg = dict(
        num_agents=num_agents,
        steps=steps,
        dt=dt,
        zeta=BASE["zeta"],
        gamma=BASE["gamma"],
        A=A,
    )
    solver = ParallelGDSBSolver(J, h, env, cfg)

    t0 = time.perf_counter()
    best_tp, _, _, _ = solver.solve()
    wall = time.perf_counter() - t0

    tps = solver.last_sample_throughputs
    mean_tp = float(np.mean(tps))
    std_tp = float(np.std(tps, ddof=1)) if len(tps) > 1 else 0.0
    return best_tp, mean_tp, std_tp, wall


def run_sweep_solver_param(name, values, *, repeat, num_agents):
    """扫描只影响 solver 的参数 (A/dt/steps)：env/J/h 复用。"""
    env, J, h, theo = make_env_and_jh(V1=BASE["V1"], V2=BASE["V2"], d_cs=BASE["d_cs"])
    rows = []
    print(f"\n=== sweep: {name} (env/J/h 复用) ===")
    for v in values:
        bests, means, stds, walls = [], [], [], []
        for _ in range(repeat):
            kw = dict(A=BASE["A"], dt=BASE["dt"], steps=BASE["steps"],
                      num_agents=num_agents)
            kw[name] = v
            b, m, s, w = solve_once(env, J, h, theo, **kw)
            bests.append(b); means.append(m); stds.append(s); walls.append(w)
        rows.append(_summary(name, v, repeat, bests, means, stds, walls, theo))
    return rows


def run_sweep_v2(values, *, repeat, num_agents):
    """扫描 V2 (V2/V1)：只重建 J/h，env 复用。"""
    env, _, _, theo = make_env_and_jh(V1=BASE["V1"], V2=BASE["V2"], d_cs=BASE["d_cs"])
    rows = []
    print(f"\n=== sweep: V2 (只重建 J/h) ===")
    for v in values:
        env.V2_SCALE = v
        J, h = env.build_J_h()
        bests, means, stds, walls = [], [], [], []
        for _ in range(repeat):
            b, m, s, w = solve_once(env, J, h, theo, A=BASE["A"], dt=BASE["dt"],
                                    steps=BASE["steps"], num_agents=num_agents)
            bests.append(b); means.append(m); stds.append(s); walls.append(w)
        rows.append(_summary("V2", v, repeat, bests, means, stds, walls, theo))
    return rows


def run_sweep_dcs(values, *, repeat, num_agents):
    """扫描 d_CS：重建 env + J/h。"""
    rows = []
    print(f"\n=== sweep: d_cs (重建 env+J/h) ===")
    for v in values:
        env, J, h, theo = make_env_and_jh(V1=BASE["V1"], V2=BASE["V2"], d_cs=v)
        bests, means, stds, walls = [], [], [], []
        for _ in range(repeat):
            b, m, s, w = solve_once(env, J, h, theo, A=BASE["A"], dt=BASE["dt"],
                                    steps=BASE["steps"], num_agents=num_agents)
            bests.append(b); means.append(m); stds.append(s); walls.append(w)
        rows.append(_summary("d_cs", v, repeat, bests, means, stds, walls, theo))
    return rows


def _summary(name, v, repeat, bests, means, stds, walls, theo):
    row = {
        "sweep_param": name,
        "value": v,
        "repeat": repeat,
        "best_mbps_mean": float(np.mean(bests)),
        "best_mbps_max": float(np.max(bests)),
        "mean_mbps_mean": float(np.mean(means)),
        "std_mbps_mean": float(np.mean(stds)),
        "wall_s_mean": float(np.mean(walls)),
        "theo_max_mbps": float(theo),
        "normalized_best_mean": float(np.mean(bests) / theo),
    }
    print(f"  {name}={v:>6}: best(mean)={row['best_mbps_mean']:7.2f}  "
          f"best(max)={row['best_mbps_max']:7.2f}  norm={row['normalized_best_mean']:.4f}  "
          f"wall={row['wall_s_mean']:.1f}s")
    return row


def print_robustness(rows, name):
    sub = [r for r in rows if r["sweep_param"] == name]
    if not sub:
        return
    bests = [r["best_mbps_mean"] for r in sub]
    norm = [r["normalized_best_mean"] for r in sub]
    lo, hi = min(bests), max(bests)
    span = hi - lo
    rel = span / ((lo + hi) / 2) * 100 if (lo + hi) > 0 else 0.0
    print(f"\n[{name}] 吞吐量范围 {lo:.2f}~{hi:.2f} Mbps, 跨参数波动 {span:.2f} Mbps "
          f"(相对均值 {rel:.1f}%)")
    print(f"  normalized 范围 {min(norm):.4f}~{max(norm):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="快速冒烟测试：num_agents=50, repeat=1")
    ap.add_argument("--repeat", type=int, default=3,
                    help="每个参数点的重复次数（平滑随机初始化）")
    args = ap.parse_args()

    num_agents = 50 if args.quick else BASE["num_agents"]
    repeat = 1 if args.quick else args.repeat

    print("基准配置：", {k: BASE[k] for k in
          ["num_ap", "num_ch", "num_ue_per_ap", "A", "dt", "steps",
           "num_agents", "V2", "d_cs"]})
    print(f"每点重复 {repeat} 次，agents={num_agents}")

    all_rows = []
    all_rows += run_sweep_solver_param("A", [0.29, 0.34, 0.39, 0.44, 0.49],
                                       repeat=repeat, num_agents=num_agents)
    all_rows += run_sweep_solver_param("dt", [0.05, 0.10, 0.15, 0.20],
                                       repeat=repeat, num_agents=num_agents)
    all_rows += run_sweep_solver_param("steps", [600, 900, 1200],
                                       repeat=repeat, num_agents=num_agents)
    all_rows += run_sweep_v2([50.0, 100.0, 200.0, 500.0],
                             repeat=repeat, num_agents=num_agents)
    all_rows += run_sweep_dcs([50.0, 60.0, 70.0, 80.0, 90.0],
                              repeat=repeat, num_agents=num_agents)

    fn = "sensitivity_results.csv"
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n已保存 {fn}（共 {len(all_rows)} 行）")

    print("\n==================== 稳健性汇总 ====================")
    for name in ["A", "dt", "steps", "V2", "d_cs"]:
        print_robustness(all_rows, name)


if __name__ == "__main__":
    main()
