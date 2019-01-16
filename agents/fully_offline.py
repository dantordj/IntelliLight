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
        self.normalizer = OfflineNormalizer()

    def save(self, name):
        path = os.path.join("saved_agents", name)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.network.state_dict(), os.path.join(path, "weights"))

    def load(self, name):
        path = os.path.join("saved_agents", name)
        path = os.path.join(path, "weights")
        self.network.load_state_dict(torch.load(path))

    def q_value(self, state, action):
        self.network.eval()

        current = np.zeros(len(state) + 1)
        current[:len(state)] = state
        current[-1] = action

        current = self.normalizer.normalize(current)
        current = torch.tensor(current, dtype=torch.float)
        return self.network.forward(current).detach().numpy()

    def remember(self, state, action, target_q):

        current = np.zeros(len(state) + 1)
        current[:len(state)] = state
        current[-1] = action

        current = self.normalizer.normalize(current)
        self.cache += [[current, target_q]]

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

        q_next = np.array([self.q_value(next_state, i) for i in range(2)])
        q_next = np.max(q_next)

        q = reward + self.gamma * q_next

        self.remember(self.last_state, self.action, q)

        return

    def train_network(self):

        min_cache = np.min([len(cache) for cache in self.cache.values()])
        if min_cache == 0:
            print("no cache, no training")
            return

        cache = []
        for cache_s_a in self.cache.values():
            indices = np.random.choice(range(len(cache_s_a)), min_cache)
            cache += [cache_s_a[i] for i in indices]

        self.has_trained = True

        self.network.train()
        for i in range(self.epochs_per_train):
            loss_epoch = 0
            data_loader = DataLoader(cache, batch_size=10000, shuffle=True)
            count = 0
            for sample in data_loader:
                count += 1
                self.optim.zero_grad()
                if self.use_img:
                    states = sample[0]
                    imgs = sample[1]
                    temp = np.reshape(sample[2], (len(sample[2]), 1))
                    q_values = torch.tensor(temp, dtype=torch.float)

                    q = self.network.forward(states, imgs)
                else:
                    states = sample[0]
                    temp = np.reshape(sample[1], (len(sample[1]), 1))
                    q_values = torch.tensor(temp, dtype=torch.float)
                    q = self.network.forward(torch.tensor(states, dtype=torch.float))

                loss = self.loss(q_values, q)

                numpy_loss = loss.detach().numpy()
                loss_epoch += numpy_loss

                loss.backward()
                self.optim.step()

            if i == 0:
                l1_crit = nn.L1Loss(reduction="sum")
                reg_loss = 0
                for param in self.network.parameters():
                    reg_loss += l1_crit(param, torch.zeros_like(param))
                print("reg loss: ", reg_loss.detach().numpy())

            print("Loss: ", loss_epoch / len(cache))

        self.reset_cache()

    def reset_cache(self):
        if self.memory_palace:
            self.cache = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
            self.cache_max = 200
        else:
            self.cache = {"all": []}
            self.cache_max = 200

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
        if self.is_training:
            self.train_network()