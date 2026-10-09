# -*- coding: utf-8 -*-
import numpy as np
from core.base_solver import BaseSolver


class ParallelGDSBSolver(BaseSolver):
    def __init__(self, J, h, env, config):
        super().__init__(J, h, env, config, name="GDSB")
        self.A = self.config.get('A', 0.32)
        self.zeta = self.config.get('zeta', 0.6)
        self.gamma = self.config.get('gamma', 0.1)

    def solve(self):
        X = np.random.uniform(-0.1, 0.1, (self.num_agents, self.N))
        Y = np.zeros((self.num_agents, self.N))
        P = np.ones((self.num_agents, self.N))

        x_trace = np.zeros((self.steps, self.N))
        energy_trace = np.zeros(self.steps)

        for m in range(self.steps):
            denom = self.steps - m
            P_next = P - (1.0 - self.A * (X ** 2)) * P / denom
            coupling = self.zeta * (np.sign(X)@ self.J) + self.gamma * self.h
            Y += (-P_next * X + coupling) * self.dt
            X += Y * self.dt

            out_of_bounds = np.abs(X) > 1.0
            X = np.clip(X, -1.0, 1.0)
            Y[out_of_bounds] = 0.0
            P = P_next

            x_trace[m, :] = X[0, :]
            #energy_trace[m] = -0.5 * (X[0] @ self.J @ X[0]) - (self.h @ X[0])
            energy_trace[m] = -0.5 * (np.sign(X[0]) @ self.J @ np.sign(X[0])) - (self.h @ np.sign(X[0]))

        best_tp = -1.0
        best_assignment = None
        sample_throughputs = []
        sample_assignments = []
        for m in range(self.num_agents):
            config_one_hot = np.zeros(self.N)
            m_assign = []
            for i in range(self.env.NUM_AP):
                ap_slice = X[m, i * self.env.NUM_CH: (i + 1) * self.env.NUM_CH]
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
