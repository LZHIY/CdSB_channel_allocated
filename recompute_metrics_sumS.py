# -*- coding: utf-8 -*-
"""
recompute_metrics_sumS.py
=====================================================================
用【原始 environment.py】(sum_S 口径) 重算审稿回复所需的四个指标,
不改动任何模型代码, 只做测量。

依赖 (与本脚本放同一目录):
  - environment.py   (原始 WLANIsingModel, sum_S 口径, 未改动)
  - base_solver.py, bsb_solver.py, dsb_solver.py, gbsb_solver.py,
    gdsb_solver.py, cim_solver.py, greedy_solver.py

四个实验:
  E1  相关性   : QUBO 能量 vs 真实吞吐 的 Pearson/Spearman (M=20,K=3,3000 样本)
  E2  判别实验 : CdSB(真实 J) vs CdSB(高斯噪声 J) vs 随机 best-of-100  (M=20,K=3)
  E3  消融 2x2 : bSB/dSB/GbSB/CdSB 归一化吞吐 (M=30, UE/AP=10)
  E4  经典基线 : 图着色 / 局部搜索 / 禁忌搜索 vs CdSB vs Greedy (M=30,K=3,UE=5)

运行示例:
  python recompute_metrics_sumS.py                        # 全部实验(论文口径)
  python recompute_metrics_sumS.py --only E1              # 只跑相关性(最快)
  python recompute_metrics_sumS.py --samples 500 --steps 500 --trials 3   # 快速冒烟

说明:
  - 本脚本的 Φ = Σ_a Σ_b [S_ijkla + S_kjlab], 与原始 _build_throughput_aware_J_h
    中的 sum_S 完全一致 (同信道 j=l 口径)。
  - E_Q = -V1·Σ_{同信道对}Φ, 即论文 Eq.(14) 的 -V1 项 (可行态上 V2 罚项=0)。
  - 如需与论文拓扑一致, 请用 --seed 指定与论文相同的随机种子。
=====================================================================
"""
import argparse
import csv
import os
import time
import numpy as np

# ---------------------------------------------------------------------
# 兼容 shim: 让 bsb/dsb/gbsb/gdsb/cim/greedy_solver 里的
# `from core.base_solver import BaseSolver` 在本目录也能工作。
# ---------------------------------------------------------------------
try:
    from core.base_solver import BaseSolver  # noqa: F401
except ImportError:
    import importlib.util
    import sys
    import types
    _here = os.path.dirname(os.path.abspath(__file__))
    _bp = os.path.join(_here, 'base_solver.py')
    if os.path.exists(_bp):
        _core = types.ModuleType('core')
        _spec = importlib.util.spec_from_file_location('core.base_solver', _bp)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules['core'] = _core
        sys.modules['core.base_solver'] = _mod
        _spec.loader.exec_module(_mod)
    else:
        raise RuntimeError("找不到 base_solver.py, 请与本脚本放在同一目录。")

from core.environment import WLANIsingModel
from solvers.bsb_solver import AdvancedbSBSolver          # bSB
from solvers.dsb_solver import AdvanceddSBSolver          # dSB
from solvers.gbsb_solver import ParallelGBSBSolver        # GbSB
from solvers.gdsb_solver import ParallelGDSBSolver        # CdSB (GDSB)
from solvers.greedy_solver import GreedyAssignmentSolver  # Greedy


# =====================================================================
# 基础工具
# =====================================================================
def onehot(env, assign):
    x = np.zeros(env.NUM_AP * env.NUM_CH)
    for i, c in enumerate(assign):
        x[i * env.NUM_CH + c] = 1.0
    return x


def compute_phi(env):
    """Φ_{i,k} = Σ_a Σ_b [S(i,0,k,0,a,b) + S(k,0,i,0,b,a)] (同信道口径, 与 sum_S 一致)."""
    M = env.NUM_AP
    phi = np.zeros((M, M))
    for i in range(M):
        ci = int(env.UE_COUNTS_PER_AP[i])
        if ci == 0:
            continue
        for k in range(M):
            if i == k:
                continue
            ck = int(env.UE_COUNTS_PER_AP[k])
            if ck == 0:
                continue
            s = 0.0
            for a in range(ci):
                for b in range(ck):
                    s += env.compute_Sijkla(i, 0, k, 0, a, b) + env.compute_Sijkla(k, 0, i, 0, b, a)
            phi[i, k] = s
    return phi


def compute_phi_orth(env):
    """正交基线 Φ^orth_{i,k} (i,k 不同信道 j=0,l=1), 用于计算退化量 = Φ^orth - Φ."""
    M = env.NUM_AP
    phio = np.zeros((M, M))
    for i in range(M):
        ci = int(env.UE_COUNTS_PER_AP[i])
        if ci == 0:
            continue
        for k in range(M):
            if i == k:
                continue
            ck = int(env.UE_COUNTS_PER_AP[k])
            if ck == 0:
                continue
            s = 0.0
            for a in range(ci):
                for b in range(ck):
                    s += env.compute_Sijkla(i, 0, k, 1, a, b) + env.compute_Sijkla(k, 1, i, 0, b, a)
            phio[i, k] = s
    return phio


def qubo_energy(phi, assign, V1):
    """E_Q = -V1·Σ_{同信道对}Φ (论文 Eq.14 的 -V1 项; 可行态上 V2 罚项=0)."""
    e = 0.0
    for i in range(len(assign)):
        for k in range(i + 1, len(assign)):
            if assign[i] == assign[k]:
                e += phi[i, k]
    return -V1 * e


def ising_energy(env, J, h, assign):
    """H = -0.5 sJs - hs (求解器实际最小化的量), s = 2x - 1."""
    x = onehot(env, assign)
    s = 2.0 * x - 1.0
    return float(-0.5 * (s @ J @ s) - (h @ s))


def random_best_of_N(env, N, seed):
    rng = np.random.default_rng(seed)
    M, K = env.NUM_AP, env.NUM_CH
    best, best_assign = -1.0, None
    for _ in range(N):
        assign = rng.integers(0, K, size=M).astype(int)
        tp = env.compute_true_throughput(onehot(env, assign))
        if tp > best:
            best, best_assign = tp, assign.copy()
    return best, best_assign


def spearman(a, b):
    """Pearson of ranks (无并列修正的 Spearman)."""
    ra = np.argsort(np.argsort(a, kind='mergesort')).astype(float)
    rb = np.argsort(np.argsort(b, kind='mergesort')).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# =====================================================================
# E1 相关性
# =====================================================================
def exp_correlation(args):
    env = WLANIsingModel(num_ap=args.M, num_ch=args.K, num_ue_per_ap=args.ue, seed=None, ap_seed=args.ap_seed, ue_seed=args.ue_seed)
    phi = compute_phi(env)
    J, h = env.build_J_h()
    M, K = env.NUM_AP, env.NUM_CH
    rng = np.random.default_rng(args.seed)
    assign = rng.integers(0, K, size=(args.samples, M))

    R = np.zeros(args.samples)
    EQ = np.zeros(args.samples)   # E_Q = -V1·ΣΦ (Eq.14)
    H = np.zeros(args.samples)    # Ising 能量
    for s in range(args.samples):
        a = assign[s]
        R[s] = env.compute_true_throughput(onehot(env, a))
        EQ[s] = qubo_energy(phi, a, args.V1)
        H[s] = ising_energy(env, J, h, a)
        if (s + 1) % max(1, args.samples // 10) == 0:
            print(f"  [E1] {s + 1}/{args.samples}", flush=True)

    r_eq = float(np.corrcoef(R, EQ)[0, 1])
    rho_eq = spearman(R, EQ)
    r_h = float(np.corrcoef(R, H)[0, 1])

    print(f"\n[E1 相关性] M={M}, K={K}, ue={args.ue}, samples={args.samples}")
    print(f"  R 范围                   : [{R.min():.1f}, {R.max():.1f}] Mbps")
    print(f"  Pearson  (R, E_Q=-V1·ΣΦ) : {r_eq:+.4f}   (论文口径, 预期为负)")
    print(f"  Spearman (R, E_Q)        : {rho_eq:+.4f}")
    print(f"  Pearson  (R, H=-½sJs-hs) : {r_h:+.4f}   (求解器能量, 预期为正)")

    with open('E1_correlation_sumS.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['R_mbps', 'E_Q', 'H_ising'])
        w.writerows(zip(R, EQ, H))
    print("  已保存 E1_correlation_sumS.csv")
    return dict(r_eq=r_eq, rho_eq=rho_eq, r_h=r_h)


# =====================================================================
# E2 判别实验
# =====================================================================
def exp_discrimination(args):
    env = WLANIsingModel(num_ap=args.M, num_ch=args.K, num_ue_per_ap=args.ue, seed=None, ap_seed=args.ap_seed, ue_seed=args.ue_seed)
    J, h = env.build_J_h()
    N = env.NUM_AP * env.NUM_CH
    scale = float(np.std(J))

    cfg = dict(num_agents=args.agents, steps=args.steps, dt=args.dt,
               zeta=args.zeta, gamma=args.gamma, A=args.A)

    def run_cdsb(Jj, hh, seed_offset):
        tps = []
        for t in range(args.trials):
            np.random.seed(args.seed * 1000 + seed_offset + t)
            s = ParallelGDSBSolver(Jj, hh, env, dict(cfg))
            tp, _, _, _ = s.solve()
            tps.append(tp)
        return float(np.mean(tps))

    t_true = run_cdsb(J, h, 0)

    rng = np.random.default_rng(args.seed + 7)
    Jn = rng.normal(0.0, scale, (N, N))
    Jn = (Jn + Jn.T) / 2.0
    np.fill_diagonal(Jn, 0.0)
    hn = rng.normal(0.0, scale, N)
    t_noise = run_cdsb(Jn, hn, 1000)

    rnd = [random_best_of_N(env, args.agents, args.seed * 5000 + t)[0]
           for t in range(args.trials)]
    t_random = float(np.mean(rnd))

    net = (t_true - t_noise) / max(t_noise, 1e-9) * 100.0

    print(f"\n[E2 判别实验] M={args.M}, K={args.K}, agents={args.agents}, "
          f"steps={args.steps}, trials={args.trials}")
    print(f"  CdSB(真实 J/sum_S)      : {t_true:7.1f} Mbps")
    print(f"  CdSB(高斯噪声 J)        : {t_noise:7.1f} Mbps")
    print(f"  随机 best-of-{args.agents}        : {t_random:7.1f} Mbps")
    print(f"  代理净价值 (相对噪声)   : {net:+.1f}%")
    return dict(true=t_true, noise=t_noise, random=t_random, net=net)


# =====================================================================
# E3 消融 2x2
# =====================================================================
def exp_ablation(args):
    env = WLANIsingModel(num_ap=args.M, num_ch=args.K, num_ue_per_ap=args.ue_ab, seed=None, ap_seed=args.ap_seed, ue_seed=args.ue_seed)
    J, h = env.build_J_h()
    tmax = env.compute_true_theoretical_max()

    cfg = dict(num_agents=args.agents, steps=args.steps, dt=args.dt,
               zeta=args.zeta, gamma=args.gamma, A=args.A)
    solvers = {
        'bSB':  AdvancedbSBSolver,
        'dSB':  AdvanceddSBSolver,
        'GbSB': ParallelGBSBSolver,
        'CdSB': ParallelGDSBSolver,
    }
    res = {}
    print(f"\n[E3 消融] M={args.M}, K={args.K}, UE/AP={args.ue_ab}, steps={args.steps}, trials={args.trials}")
    for name, Solver in solvers.items():
        tps = []
        for t in range(args.trials):
            np.random.seed(args.seed * 100 + t)
            s = Solver(J, h, env, dict(cfg))
            tp, _, _, _ = s.solve()
            tps.append(tp)
        mean_tp = float(np.mean(tps))
        res[name] = (mean_tp, mean_tp / max(tmax, 1e-9))
        print(f"  {name:<5} 吞吐={mean_tp:7.1f} Mbps  归一化={mean_tp / max(tmax, 1e-9):.4f}")

    b, gb, d, cd = res['bSB'][1], res['GbSB'][1], res['dSB'][1], res['CdSB'][1]
    print(f"  Δ(sgn | 无P)  = dSB-bSB   = {d - b:+.4f}")
    print(f"  Δ(sgn | 有P)  = CdSB-GbSB = {cd - gb:+.4f}")
    print(f"  Δ(P   | 无sgn) = GbSB-bSB = {gb - b:+.4f}")
    print(f"  Δ(P   | 有sgn) = CdSB-dSB = {cd - d:+.4f}")

    with open('E3_ablation_sumS.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'throughput_mbps', 'normalized'])
        for name in ('bSB', 'dSB', 'GbSB', 'CdSB'):
            w.writerow([name, res[name][0], res[name][1]])
    print("  已保存 E3_ablation_sumS.csv")
    return res


# =====================================================================
# E4 经典基线
# =====================================================================
def exp_baselines(args):
    env = WLANIsingModel(num_ap=args.M, num_ch=args.K, num_ue_per_ap=args.ue, seed=None, ap_seed=args.ap_seed, ue_seed=args.ue_seed)
    phi = compute_phi(env)
    phio = compute_phi_orth(env)
    deg = np.maximum(phio - phi, 0.0)   # 退化量 = 正交基线 - 同信道

    M, K = env.NUM_AP, env.NUM_CH

    def tp_of(assign):
        return env.compute_true_throughput(onehot(env, assign))

    # ---- 图着色 (贪心, 冲突权重=退化量, 最大度优先) ----
    def graph_coloring():
        degree = deg.sum(axis=1)
        order = np.argsort(-degree)
        assign = np.full(M, -1, dtype=int)
        for i in order:
            costs = np.zeros(K)
            for k in range(M):
                if assign[k] >= 0:
                    costs[assign[k]] += deg[i, k]
            assign[i] = int(np.argmin(costs))
        return assign

    # ---- 局部搜索 (1-flip 爬山, 目标=真实吞吐) ----
    def local_search(init, max_iter=2000):
        assign = init.copy()
        cur = tp_of(assign)
        for _ in range(max_iter):
            improved = False
            for i in range(M):
                old = assign[i]
                best_c, best_val = old, cur
                for c in range(K):
                    if c == old:
                        continue
                    assign[i] = c
                    val = tp_of(assign)
                    if val > best_val:
                        best_val, best_c = val, c
                assign[i] = best_c
                if best_val > cur:
                    cur = best_val
                    improved = True
            if not improved:
                break
        return assign

    # ---- 禁忌搜索 (1-flip + 禁忌表 + 特赦准则) ----
    def tabu_search(init, max_iter=500, tabu_len=7):
        assign = init.copy()
        best = assign.copy()
        cur = tp_of(assign)
        best_tp = cur
        tabu = {}
        for it in range(max_iter):
            best_move = None
            best_val = -1.0
            for i in range(M):
                old = assign[i]
                for c in range(K):
                    if c == old:
                        continue
                    assign[i] = c
                    val = tp_of(assign)
                    assign[i] = old
                    key = (i, old, c)
                    allowed = (key not in tabu) or (tabu[key] <= it) or (val > best_tp)
                    if val > best_val and allowed:
                        best_val, best_move = val, (i, old, c)
            if best_move is None:
                break
            i, old, c = best_move
            assign[i] = c
            tabu[(i, c, old)] = it + tabu_len
            cur = best_val
            if cur > best_tp:
                best_tp = cur
                best = assign.copy()
        return best

    rows = []

    def report(name, assign, t0):
        tp = tp_of(assign)
        dt = time.time() - t0
        rows.append((name, tp, dt))
        print(f"  {name:<18} 吞吐={tp:7.1f} Mbps  用时={dt:8.3f} s")

    print(f"\n[E4 经典基线] M={args.M}, K={args.K}, UE={args.ue}")

    t0 = time.time(); a_gc = graph_coloring(); report('Graph coloring', a_gc, t0)
    t0 = time.time(); a_ls = local_search(a_gc, args.ls_iters); report('Local search', a_ls, t0)
    t0 = time.time(); a_ts = tabu_search(a_gc, args.ts_iters, args.ts_tabu); report('Tabu search', a_ts, t0)

    # CdSB (GDSB)
    J, h = env.build_J_h()
    cfg = dict(num_agents=args.agents, steps=args.steps, dt=args.dt,
               zeta=args.zeta, gamma=args.gamma, A=args.A)
    t0 = time.time()
    s = ParallelGDSBSolver(J, h, env, dict(cfg))
    tp_cdsb, a_cdsb, _, _ = s.solve()
    report('CdSB', np.array(a_cdsb), t0)

    # Greedy (Ising)
    t0 = time.time()
    g = GreedyAssignmentSolver(J, h, env, dict(cfg))
    tp_g, a_g, _, _ = g.solve()
    report('Greedy (Ising)', np.array(a_g), t0)

    with open('E4_baselines_sumS.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'throughput_mbps', 'time_s'])
        w.writerows(rows)
    print("  已保存 E4_baselines_sumS.csv")
    return rows


# =====================================================================
# 主入口
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=str, default='', help='只跑某实验, 如 E1/E2/E3/E4 (逗号分隔)')
    ap.add_argument('--M', type=int, default=20)        # E1/E2 的 AP 数
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--ue', type=int, default=5)        # E1/E2/E4 的 UE/AP
    ap.add_argument('--ue_ab', type=int, default=10)    # E3 消融的 UE/AP
    ap.add_argument('--samples', type=int, default=3000)
    ap.add_argument('--agents', type=int, default=100)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--dt', type=float, default=0.05)
    ap.add_argument('--A', type=float, default=0.32)
    ap.add_argument('--zeta', type=float, default=0.6)
    ap.add_argument('--gamma', type=float, default=0.1)
    ap.add_argument('--V1', type=float, default=1.0)
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ap_seed', type=int, default=None)   # AP 拓扑种子 (None=用 env 默认 15)
    ap.add_argument('--ue_seed', type=int, default=None)   # UE 拓扑种子 (None=用 env 默认 16)
    ap.add_argument('--ls_iters', type=int, default=300)   # 局部搜索最大迭代
    ap.add_argument('--ts_iters', type=int, default=150)   # 禁忌搜索最大迭代
    ap.add_argument('--ts_tabu', type=int, default=7)      # 禁忌表长度
    args = ap.parse_args()

    only = {e.strip() for e in args.only.split(',') if e.strip()} if args.only else set()

    def want(e):
        return (not only) or (e in only)

    if want('E1'):
        exp_correlation(args)
    if want('E2'):
        exp_discrimination(args)
    if want('E3'):
        exp_ablation(args)
    if want('E4'):
        exp_baselines(args)

    print("\n完成。")


if __name__ == '__main__':
    main()
