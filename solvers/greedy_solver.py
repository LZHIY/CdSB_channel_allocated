# -*- coding: utf-8 -*-
import numpy as np

from core.base_solver import BaseSolver


class GreedyAssignmentSolver(BaseSolver):
    """Greedy baseline in Ising space: minimize energy from J/h sequentially."""

    def __init__(self, J, h, env, config):
        super().__init__(J, h, env, config, name="Greedy")

    def _state_to_one_hot(self, state: np.ndarray) -> np.ndarray:
        x = np.zeros(self.N)
        for ap_idx, ch_idx in enumerate(state):
            x[ap_idx * self.env.NUM_CH + ch_idx] = 1.0
        return x

    def _energy_from_state(self, state: np.ndarray) -> float:
        x = self._state_to_one_hot(state)
        s = 2.0 * x - 1.0
        return float(-0.5 * (s @ self.J @ s) - (self.h @ s))

    def solve(self):
        state = np.zeros(self.env.NUM_AP, dtype=int)

        for ap_idx in range(self.env.NUM_AP):
            best_ch = 0
            best_energy = float("inf")
            for ch in range(self.env.NUM_CH):
                state[ap_idx] = ch
                energy = self._energy_from_state(state)
                if energy < best_energy:
                    best_energy = energy
                    best_ch = ch
            state[ap_idx] = best_ch

        x_best = self._state_to_one_hot(state)
        throughput = self.env.compute_true_throughput(x_best)
        return throughput, state.copy(), None, None
