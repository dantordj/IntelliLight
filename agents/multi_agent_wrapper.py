import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_phase, wgreen, get_state_sumo
from agents.linqagent import LinQAgent
from agents.fully_offline import OfflineAgent


class MultiAgentWrapper:

    def __init__(self, agent_class=LinQAgent, offsets=None, mode="deep", features=None, memory_palace=True):
        self.features = features
        self.memory_palace = memory_palace

        if features is None:
            self.features = ["count_incoming"]

        nodes = ["C2", "D2", "C3", "D3"]

        # parameters
        self.agents = {}

        if offsets is None:
            offsets = {node: i + 3 for (i, node) in enumerate(nodes)}

        for node in nodes:
            self.agents[node] = agent_class(
                mode=mode, features=self.features, node=node, memory_palace=self.memory_palace
            )

            self.agents[node].offline_period = offsets[node]

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
