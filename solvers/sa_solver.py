# -*- coding: utf-8 -*-
import numpy as np
import random
import math
from core.base_solver import BaseSolver

class SASolver(BaseSolver):
    def __init__(self, J, h, env, config):
        super().__init__(J, h, env, config, name="SA")
        self.t_start = self.config.get('t_start', 10.0)
        self.t_end = self.config.get('t_end', 1e-5)
        self.cooling_rate = self.config.get('cooling_rate', 0.99)
        self.iter_per_temp = self.config.get('iter_per_temp', 100)

    def _get_energy_and_x(self, state):
        X = np.zeros(self.N)
        for ap_idx, ch_idx in enumerate(state):
            X[ap_idx * self.env.NUM_CH + ch_idx] = 1
        s = 2 * X - 1
        energy = -0.5 * np.dot(s.T, np.dot(self.J, s)) - np.dot(self.h.T, s)
        return energy, X

    def solve(self):
        current_state = np.random.randint(0, self.env.NUM_CH, size=self.env.NUM_AP)
        current_energy, _ = self._get_energy_and_x(current_state)
        best_state, best_energy = np.copy(current_state), current_energy

        t = self.t_start
        while t > self.t_end:
            for _ in range(self.iter_per_temp):
                target_ap = random.randint(0, self.env.NUM_AP - 1)
                new_ch = random.randint(0, self.env.NUM_CH - 1)
                next_state = np.copy(current_state)
                next_state[target_ap] = new_ch
                next_energy, _ = self._get_energy_and_x(next_state)

                delta_e = next_energy - current_energy
                if delta_e < 0 or random.random() < math.exp(-delta_e / (t + 1e-12)):
                    current_state, current_energy = next_state, next_energy
                    if current_energy < best_energy:
                        best_energy, best_state = current_energy, np.copy(current_state)
            t *= self.cooling_rate

        _, best_X = self._get_energy_and_x(best_state)
        return self.env.compute_true_throughput(best_X), best_state, None, None