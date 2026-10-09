# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np


class BaseSolver(ABC):
    def __init__(self, J, h, env, config, name="BaseSolver"):
        self.name = name
        self.env = env
        self.config = config

        self.num_agents = config.get('num_agents', 100)
        self.steps = config.get('steps', 5000)
        self.dt = config.get('dt', 0.05)

        self.j_raw = J
        self.h_raw = h
        self.J, self.h = self.env.normalize_and_export(self.j_raw, self.h_raw)
        self.N = len(self.h)

    @abstractmethod
    def solve(self):
        pass