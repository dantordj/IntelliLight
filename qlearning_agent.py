from utils import get_phase, wgreen, ngreen, yellow_nw, yellow_wn, get_state_sumo
import os
import numpy as np
from neural_nets import ConvNet, LinearNet, DeepNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn
from learning_agent import LearningAgent
from agents import MyNormalizer

class QLearningAgent(LearningAgent):

    def __init__(self):
        super(QLearningAgent, self).__init__()
        self.mode = "q_learning"

        self.steps = 0
        self.n_states = (4 ** 4) * 2
        self.Q = np.zeros((self.n_states, 2))
        self.T = np.zeros((self.n_states, 2))

        self.observe_steps = 5000

        # parameters
        self.gamma = 0.8
        self.epsilon = 0.1
        self.beta = 0.5
        self.action = 0
        self.last_state = 0

        self.visited_states = np.zeros(self.n_states)

    def save(self, name):
        path = os.path.join("saved_agents", name)
        if not os.path.exists(path):
            os.mkdir(path)

        np.savetxt(os.path.join(path, "Q.txt"), self.Q)
        np.savetxt(os.path.join(path, "visited_states.txt"), self.visited_states)
        np.savetxt(os.path.join(path, "T.txt"), self.T)

    def load(self, name):
        path = os.path.join("saved_agents", name)
        assert os.path.exists(path), "no such saved agent"

        self.Q = np.loadtxt(os.path.join(path, "Q.txt"))
        self.visited_states = np.loadtxt(os.path.join(path, "visited_states.txt"))
        self.T = np.loadtxt(os.path.join(path, "T.txt"))

    def q_value(self, state, action):
        return self.Q[state, action]

    def get_state(self):

        assert get_phase() in [wgreen, ngreen]

        # assign an integer between 1 and 511 to each state
        phase = int(get_phase() == wgreen)
        state = np.array([0, 0, 0, 0])
        count_incoming, speed_incoming, img = get_state_sumo()

        for i, (key, value) in enumerate(count_incoming.items()):
            if value < 3:
                state[i] = 0
            elif value < 5:
                state[i] = 1
            elif value < 7:
                state[i] = 2
            else:
                state[i] = 3

        s = phase + np.dot(state, np.array([4 ** i for i in range(4)]) * 2)
        return s

    def remember(self, state, action, target_q):
        self.T[state, action] += 1
        current_q = self.Q[state, action]
        alpha = (1. / self.T[state, action]) ** self.beta
        self.Q[state, action] = (1 - alpha) * current_q + alpha * target_q
        self.visited_states[state] += 1
