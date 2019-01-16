import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_phase, wgreen, get_state_sumo
from agents.linqagent import LinQAgent


class MultiAgentWrapper:

    def __init__(self, mode="lin", features=None, memory_palace=True):
        self.mode = mode
        self.features = features
        self.memory_palace = memory_palace

        if features is None:
            self.features = ["count_incoming"]

        nodes = ["C2", "D2", "C3", "D3"]

        # parameters
        self.agents = {}
        i = 3
        for node in nodes:
            self.agents[node] = LinQAgent(
                mode=self.mode, features=self.features, node=node, memory_palace=self.memory_palace
            )
            self.agents[node].offline_period = i
            i += 1

    def choose_action(self):
        ans = {}
        for node, agent in self.agents.items():
            ans[node] = agent.choose_action()

        return ans

    def feedback(self, rewards):
        for node, agent in self.agents.items():
            agent.feedback(rewards[node])

    def save(self, name):
        for node, agent in self.agents.items():
            agent.save(name + "_" + node)

    def load(self, name):
        for node, agent in self.agents.items():
            agent.load(name + "_" + node)

    def reset(self):
        for agent in self.agents.values():
            agent.reset()
