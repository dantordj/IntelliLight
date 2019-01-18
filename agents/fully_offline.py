import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_phase, wgreen, get_state_sumo, yellow_nw, yellow_wn
from agents.neural_nets import DeepNet
from agents.learning_agent import LearningAgent


class OfflineNormalizer:
    def __init__(self):
        pass

    def fit(self):
        pass

    def normalize(self, x):
        return x


class OfflineAgent(LearningAgent):

    def __init__(self, features=None, node="node0"):
        super(OfflineAgent, self).__init__(node=node)

        self.features = features

        if features is None:
            self.features = ["count_incoming"]

        print(self.features)

        # attributes
        self.count = 0
        self.use_img = False

        self.n_inputs = 0

        self.n_inputs = 4 * len(self.features)
        self.n_inputs += 2
        self.network = DeepNet(self.n_inputs)
        self.lr = 1e-2

        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.action_states = []
        self.rewards = []
        self.next_states = []
        self.has_trained = False

        self.epochs_per_train = 30
        self.n_iteration = int(- np.log(1000) / np.log(self.gamma))

    def save(self, name):
        path = os.path.join("saved_agents", name)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.network.state_dict(), os.path.join(path, "weights"))
        np.savetxt(os.path.join(path, "actions_states"), self.action_states)
        np.savetxt(os.path.join(path, "rewards"), self.rewards)
        np.savetxt(os.path.join(path, "next_states"), self.next_states)

    def load(self, name):
        path = os.path.join("saved_agents", name)
        self.network.load_state_dict(torch.load(os.path.join(path, "weights")))

        self.action_states = list(np.loadtxt(os.path.join(path, "actions_states")))
        self.rewards = list(np.loadtxt(os.path.join(path, "rewards")))
        self.next_states = list(np.loadtxt(os.path.join(path, "next_states")))

    def q_value(self, state, action):

        if not self.has_trained:
            return 0

        self.network.eval()

        current = np.zeros(len(state) + 1)
        current[:len(state)] = state
        current[-1] = action

        current = torch.tensor(current, dtype=torch.float)
        return self.network.forward(current).detach().numpy()

    def remember(self, state, action, reward, next_state):

        current = np.zeros(len(state) + 1)
        current[:len(state)] = state
        current[-1] = action

        self.action_states.append(current)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def choose_action(self):

        self.steps += 1

        assert get_phase(self.node) not in [yellow_wn, yellow_nw]

        state = self.get_state()
        self.last_state = state

        if not self.is_online:
            # offline mode, juste change the light every n iterations
            if self.steps % self.offline_period == 0:
                self.action = 1
                return 1
            self.action = 0
            return 0

        q = np.array([self.q_value(state, i) for i in range(2)])

        if np.random.uniform(0, 1) < self.epsilon and self.is_training:
            action = np.random.choice([0, 1])
        else:
            action = np.argmax(q)

        self.action = action

        return action

    def feedback(self, reward):

        next_state = self.get_state()

        self.remember(self.last_state, self.action, reward, next_state)

    def train_network(self):

        rewards = np.array(self.rewards)
        next_states = np.array(self.next_states)

        print("dataset size: ", len(self.action_states))

        print(type(self.next_states))

        next_states0 = np.zeros((next_states.shape[0], next_states.shape[1] + 1))
        next_states1 = np.zeros((next_states.shape[0], next_states.shape[1] + 1))

        next_states0[:, :-1] = next_states.copy()
        next_states1[:, :-1] = next_states.copy()

        next_states0[:, -1] = 0
        next_states1[:, -1] = 1

        q_values = np.zeros((next_states.shape[0], 2))

        for j in range(self.n_iteration):
            print("iteration number ", j)

            if self.has_trained:
                q_values[:, 0] = self.network.forward(torch.tensor(next_states0, dtype=torch.float)).detach().numpy()[:, 0]
                q_values[:, 1] = self.network.forward(torch.tensor(next_states0, dtype=torch.float)).detach().numpy()[:, 0]

            target = rewards + self.gamma * np.max(q_values, axis=1)
            target = np.reshape(target, (len(target), 1))

            self.network.train()

            for i in range(self.epochs_per_train):
                self.optim.zero_grad()

                pred = self.network.forward(torch.tensor(self.action_states, dtype=torch.float))
                loss = self.loss(torch.tensor(target, dtype=torch.float), pred)

                numpy_loss = loss.detach().numpy()

                loss.backward()
                self.optim.step()

                print("Loss: ", numpy_loss / len(self.action_states))

            self.has_trained = True
            print("")
            print("")

    def get_state(self):
        phase = int(get_phase(self.node) == wgreen)
        features = []

        if "img" in self.features:
            self.use_img = True

        sumo_state = get_state_sumo(node=self.node, get_img=self.use_img)

        if "count_incoming" in self.features:
            for key, value in sumo_state["count_incoming"].items():
                features.append(value)

        if "count_upstream" in self.features:
            for key, value in sumo_state["count_upstream"].items():
                features.append(value)

        for key, value in sumo_state["speed_incoming"].items():

            value_array = np.array(value)

            if "max_speed" in self.features:
                features.append(np.max(value_array) if len(value) > 0 else 0)

            if "min_speed" in self.features:
                features.append(np.min(value_array) if len(value) > 0 else 0)

            if "median_speed" in self.features:
                features.append(np.median(value_array) if len(value) > 0 else 0)

            if "mean_speed" in self.features:
                features.append(np.mean(value_array) if len(value) > 0 else 0)

        if self.mode == "lin":
            state = np.zeros(len(features) * 2)
            state[phase * len(features): (phase + 1) * len(features)] = np.array(features)
        else:
            features.append(phase)
            state = np.array(features)

        if "img" in self.features:
            return state, sumo_state["img"]
        return state

    def reset(self):
        pass
