# -*- coding: utf-8 -*-
import numpy as np
from core.base_solver import BaseSolver

class ParallelCIMSolver(BaseSolver):
    def __init__(self, J, h, env, config):
        super().__init__(J, h, env, config, name="CIM")
        self.zeta = self.config.get('zeta', 0.8)
        self.gamma = self.config.get('gamma', 0.1)
        self.pump_min = self.config.get('pump_min', 0.5)
        self.pump_max = self.config.get('pump_max', 1.8)

    def solve(self):
        #初始化所有代理的状态
        #使用极小的随机噪声模拟量子涨落
        X = np.random.uniform(-0.1, 0.1, (self.num_agents, self.N))

        #记录最优代理的轨迹，用于绘图
        x_trace = np.zeros((self.steps, self.N))
        energy_trace = np.zeros(self.steps)

        #动力学演化循环
        for m in range(self.steps):
            p = self.pump_min + (self.pump_max - self.pump_min) * (m / self.steps)
            #矩阵并行运算，计算耦合项（M,N)@(N,N)->(M,N)
            coupling = self.zeta * (X @ self.J) + self.gamma * self.h
            #CIM核心动力学方程
            X += ((p - 1.0) * X - X**3 + coupling) * self.dt

            x_trace[m, :] = X[0, :]
            energy_trace[m] = -0.5 * (X[0] @ self.J @ X[0]) - (self.h @ X[0])

        best_tp, best_assignment = -1.0, None
        sample_throughputs = []
        sample_assignments = []
        for m in range(self.num_agents):
            config_one_hot = np.zeros(self.N)
            m_assign = []
            for i in range(self.env.NUM_AP):
                ap_slice = X[m, i * self.env.NUM_CH : (i + 1) * self.env.NUM_CH]
                ch_idx = int(np.argmax(ap_slice))
                m_assign.append(ch_idx)
                config_one_hot[i * self.env.NUM_CH + ch_idx] = 1

            current_tp = self.env.compute_true_throughput(config_one_hot)
            sample_throughputs.append(float(current_tp))
            sample_assignments.append(m_assign)
            if current_tp > best_tp:
                best_tp = current_tp
                best_assignment = m_assign

        self.last_sample_throughputs = sample_throughputs
        self.last_sample_assignments = sample_assignments
        self.last_best_sample_idx = int(np.argmax(sample_throughputs)) if sample_throughputs else None
        return best_tp, best_assignment, x_trace, energy_trace
