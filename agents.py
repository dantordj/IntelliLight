from utils import get_phase, wgreen, ngreen, yellow_nw, yellow_wn, get_state_sumo
import os
import numpy as np
from neural_nets import ConvNet, LinearNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn


class Agent(object):
    def __init__(self):
        pass

    def choose_action(self):
        return False

    def feedback(self, reward):
        pass

    def reset(self):
        pass


class ConstantAgent(Agent):
    def __init__(self, period=30):
        super(ConstantAgent, self).__init__()
        self.period = period
        self.count = 0

    def choose_action(self):
        self.count += 1

        if (self.count > self.period) and get_phase() in [wgreen, ngreen]:
            self.count = 0
            return True
        return False


class SimpleAgent(Agent):
    def __init__(self, factor):
        super(SimpleAgent, self).__init__()
        self.factor = factor

    def choose_action(self):
        state = get_state_sumo()

        vertical_cars = 0
        horizontal_cars = 0

        for line, num_vehicles in state.items():
            try:
                i = int(line[4])
            except:
                continue
            if i in [3, 4]:
                vertical_cars += num_vehicles
            elif i in [1, 2]:
                horizontal_cars += num_vehicles

        change = False

        if (vertical_cars > self.factor * horizontal_cars) and get_phase() == wgreen:
            change = True

        if (horizontal_cars > self.factor * vertical_cars) and get_phase() == ngreen:
            change = True

        return change


class QLearningAgent(Agent):

    def __init__(self):
        # super(QLearningAgent, self).__init__()
        self.t = 0
        self.n_states = (4 ** 4) * 2
        self.Q = np.zeros((self.n_states, 2))
        self.T = np.zeros((self.n_states, 2))
        self.gamma = 0.95
        self.epsilon = 0.1
        self.beta = 0.5
        self.action = 0
        self.last_state = 0

        self.acc_reward = 0
        self.acc_count = 0
        self.visited_states = np.zeros(self.n_states)

    def load(self, name):
        path = os.path.join("saved_agents", name)
        assert os.path.exists(path), "no such saved agent"

        self.Q = np.loadtxt(os.path.join(path, "Q.txt"))
        self.visited_states = np.loadtxt(os.path.join(path, "visited_states.txt"))
        self.T = np.loadtxt(os.path.join(path, "T.txt"))

    def encode_state(self):

        assert get_phase() in [wgreen, ngreen]

        # assign an integer between 1 and 511 to each state
        phase = int(get_phase() == wgreen)
        state = np.array([0, 0, 0, 0])

        for line, num_vehicles in get_state_sumo().items():
            try:
                i = int(line[4]) - 1
                if i < 0:
                    continue
            except:
                continue
            if num_vehicles < 3:
                state[i] = 0
            elif num_vehicles < 5:
                state[i] = 1
            elif num_vehicles < 7:
                state[i] = 2
            else:
                state[i] = 3

        s = phase + np.dot(state, np.array([4 ** i for i in range(4)]) * 2)
        return s

    def choose_action(self):

        if get_phase() in [yellow_wn, yellow_nw]:
            return 0

        state = self.encode_state()
        self.last_state = state

        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice([0, 1])
        else:
            action = np.argmax(self.Q[state])

        self.action = action

        if action:
            self.acc_count = 0
            self.acc_reward = 0

        self.T[state, action] += 1

        return action

    def feedback(self, reward):

        if get_phase() in [yellow_nw, yellow_wn]:
            self.acc_reward += reward * (self.gamma ** self.acc_count)
            self.acc_count += 1
            return

        next_state = self.encode_state()

        q = self.Q[self.last_state, self.action]
        q_next = np.max(self.Q[next_state])

        alpha = (1. / self.T[self.last_state, self.action]) ** self.beta

        q = (1 - alpha) * q

        if self.acc_count > 0:
            self.acc_reward += reward * (self.gamma ** self.acc_count)
            q += self.acc_reward
            q += q_next * (self.gamma ** (self.acc_count + 1))
            q *= alpha
            self.acc_count = 0
            self.acc_reward = 0
        else:
            q += alpha * (reward + self.gamma * q_next)

        self.Q[self.last_state, self.action] = q

        self.visited_states[self.last_state] += 1

        return

    def save(self, name):
        path = os.path.join("saved_agents", name)
        if not os.path.exists(path):
            os.mkdir(path)

        np.savetxt(os.path.join(path, "Q.txt"), self.Q)
        np.savetxt(os.path.join(path, "visited_states.txt"), self.visited_states)
        np.savetxt(os.path.join(path, "T.txt"), self.T)

    def reset(self):
        self.acc_count = 0
        self.acc_reward = 0


class LinQAgent(object):

    def __init__(self):
        self.n_states = 4

        self.gamma = 0.9
        self.epsilon = 0.1
        self.action = 0
        self.last_state = 0

        self.network = LinearNet()
        self.lr = 0.05
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.cache = []
        self.cache_max = 90
        self.is_training = True
        self.steps = 0
        self.is_online = True

    def set_is_training(self, is_training):
        self.is_training = is_training

    def set_is_online(self, is_online):
        self.is_online = is_online

    def load(self, name):
        path = os.path.join("saved_agents", name)
        path = os.path.join(path, "weights")

        self.network.load_state_dict(torch.load(path))

    def q_value(self, state, action):
        current = np.zeros(len(state) + 1)
        current[:len(state)] = state

        current[-1] = action
        current = torch.tensor(current, dtype=torch.float)

        return self.network(current).detach().numpy()

    def choose_action(self):

        self.steps += 1

        assert get_phase() not in [yellow_wn, yellow_nw]

        state = self.get_state()
        self.last_state = state

        print(state)

        if not self.is_online:
            if self.steps % 5 == 0:
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

    def remember(self, state, action, target_q):
        if self.is_training:
            current = np.zeros(len(state) + 1)
            current[:len(state)] = state
            current[-1] = action
            self.cache += [[current, target_q]]

            if len(self.cache) >= self.cache_max:
                self.train_network()

    def feedback(self, reward):

        next_state = self.get_state()

        q_next = np.array([self.q_value(next_state, i) for i in range(2)])
        q_next = np.max(q_next)

        q = reward + self.gamma * q_next

        self.remember(self.last_state, self.action, q)

        return

    def train_network(self):

        self.network.train()
        data_loader = DataLoader(self.cache, batch_size=32, shuffle=True)
        for sample in data_loader:
            states = torch.tensor(sample[0], dtype=torch.float)
            q_values = torch.tensor(sample[1], dtype=torch.float)
            q = self.network(states)
            loss = self.loss(q_values, q)
            loss.backward()
            self.optim.step()
        self.cache = []

    def get_state(self):
        phase = int(get_phase() == wgreen)
        state = np.zeros(4 + 1)
        for line, num_vehicles in get_state_sumo().items():
            try:
                i = int(line[4]) - 1
                if i < 0:
                    continue
            except:
                continue
            state[i] = int(num_vehicles)
        state[-1] = phase
        return state

    def save(self, name):
        path = os.path.join("saved_agents", name)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.network.state_dict(), os.path.join(path, "weights"))

    def reset(self):
        self.train_network()

    def plot(self):
        size_array = 15
        matrix = np.zeros((size_array, size_array))

        for i in range(size_array):
            for j in range(size_array):

                incoming_e = i
                incoming_w = 0
                incoming_n = j
                incoming_s = 0

                phase = "WGREEN"
                state = np.array([incoming_n, incoming_s, incoming_e, incoming_w, int(phase == "WGREEN")])

                q = np.array([self.q_value(state, i) for i in range(2)])
                matrix[i, j] = np.argmax(q)

        seaborn.heatmap(matrix, cmap="hot")
        plt.show()


class DeepQLearningAgent(object):

    def __init__(self):
        self.t = 0
        self.n_states = 5

        self.gamma = 0.8
        self.epsilon = 0.05
        self.beta = 0.5
        self.action = 0
        self.last_state = 0
        self.last_q = np.zeros(2)
        self.alpha = 1

        self.network = LinearNet()
        self.lr = 0.01
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.cache = []
        self.cache_max = 100

    def load(self, name):
        path = os.path.join("saved_agents", name)
        path = os.path.join(path, "weights")
        self.network.load_state_dict(torch.load(path))

    def choose_action(self):
        self.t += 1

        assert get_phase() not in [yellow_wn, yellow_nw]

        state = self.get_state()
        print(state)
        self.last_state = state
        q = self.last_q

        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice([0, 1])

        else:
            self.network.eval()
            action = np.argmax(q)

        self.action = action

        return action

    def feedback(self, reward):

        next_state = self.get_state()

        next_state = torch.tensor(next_state, dtype=torch.float)

        self.network.eval()
        q_vec = self.last_q
        q_next_vec = self.network(next_state).detach().numpy()
        q_next = np.max(q_next_vec)

        q = reward + self.gamma * q_next

        q_vec[self.action] = q

        q_vec = torch.tensor(self.alpha * q_vec + (1 - self.alpha) * self.last_q, dtype=torch.float)

        self.cache += [[self.last_state, q_vec]]

        if len(self.cache) == self.cache_max:
            self.train_network()
        self.last_q = q_next_vec
        return

    def train_network(self):
        self.network.train()
        data_loader = DataLoader(self.cache, batch_size=32, shuffle=True)
        for sample in data_loader:
            states = torch.tensor(sample[0], dtype=torch.float)
            q_values = torch.tensor(sample[1], dtype=torch.float)
            q = self.network(states)
            loss = self.loss(q_values, q)
            loss.backward()
            self.optim.step()
        self.cache = []

    def get_state(self):
        # assign an integer between 1 and 511 to each state
        phase = int(get_phase() == wgreen)
        ans = np.zeros(8)
        state = np.array([0, 0, 0, 0])
        for line, num_vehicles in get_state_sumo().items():
            try:
                i = int(line[4]) - 1
                if i < 0:
                    continue
            except:
                continue
            state[i] = int(num_vehicles)
        ans[phase * 4: (phase + 1) * 4] = state
        return ans

    def save(self, name):
        path = os.path.join("saved_agents", name)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.network.state_dict(), os.path.join(path, "weights"))

    def reset(self):

        self.train_network()
        self.acc_count = 0
        self.last_q = np.zeros(2)
        self.t = 0
        self.acc_reward = 0
