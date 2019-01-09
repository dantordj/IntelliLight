from utils import get_phase, wgreen, ngreen, yellow_nw, yellow_wn, get_state_sumo
import os
import numpy as np
from neural_nets import ConvNet, LinearNet, DeepNet
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

        if (self.count > self.period) and (get_phase() in [wgreen, ngreen]):
            self.count = 0
            return 1
        return 0


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


class MyNormalizer(object):
    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.ones(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = np.clip(self.mean_diff/self.n, a_min=1e-2, a_max=1e12)

    def normalize(self, inputs):
        obs_std = np.sqrt(self.var)
        return (inputs - self.mean)/obs_std


class LinQAgent(object):

    def __init__(self, mode="lin"):

        self.mode = mode

        if self.mode == "lin":
            self.network = LinearNet()
            self.lr = 0.1
            self.n_features = 16

        else:
            self.network = DeepNet()
            self.lr = 1e-2
            self.n_features = 6

        self.gamma = 0.8
        self.epsilon = 0.1
        self.action = 0
        self.last_state = 0

        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.cache = []
        self.cache_max = 20000
        self.is_training = True
        self.steps = 0
        self.is_online = True
        self.count = 0
        self.normalizer = MyNormalizer(self.n_features)

    def set_is_training(self, is_training):
        self.is_training = is_training

    def set_is_online(self, is_online):
        self.is_online = is_online

    def load(self, name):
        path = os.path.join("saved_agents", name)
        path = os.path.join(path, "weights")

        self.network.load_state_dict(torch.load(path))

    def q_value(self, state, action):

        if self.mode == "lin":
            current = np.zeros(2 * len(state))
            current[action * len(state): (action + 1) * len(state)] = state.copy()
        else:
            current = np.zeros(len(state) + 1)
            current[:len(state)] = state
            current[-1] = action

        current = self.normalizer.normalize(current)
        current = torch.tensor(current, dtype=torch.float)

        return self.network.forward(current).detach().numpy()

    def choose_action(self):

        self.steps += 1

        assert get_phase() not in [yellow_wn, yellow_nw]

        state = self.get_state()
        self.last_state = state

        if not self.is_online:
            if np.random.rand() < 0.3:
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

            if self.mode == "lin":
                current = np.zeros(2 * len(state))
                current[action * len(state): (action + 1) * len(state)] = state.copy()
            else:
                current = np.zeros(len(state) + 1)
                current[:len(state)] = state
                current[-1] = action

            self.normalizer.observe(current)
            current = self.normalizer.normalize(current)

            if self.steps > 50:
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

        print("len cache: ", len(self.cache))

        self.network.train()
        for i in range(30):
            loss_epoch = 0
            data_loader = DataLoader(self.cache, batch_size=10000, shuffle=True)
            count = 0
            for sample in data_loader:
                count += 1
                self.optim.zero_grad()
                states = torch.tensor(sample[0], dtype=torch.float)
                temp = np.reshape(sample[1], (len(sample[1]), 1))
                q_values = torch.tensor(temp, dtype=torch.float)
                q = self.network.forward(states)

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

            print("Loss: ", loss_epoch / len(self.cache))

        self.cache = []

    def get_state(self):
        phase = int(get_phase() == wgreen)
        incoming_vehicles = np.zeros(4)

        for line, num_vehicles in get_state_sumo().items():
            try:
                i = int(line[4]) - 1
                if i < 0:
                    continue
            except:
                continue
            incoming_vehicles[i] = int(num_vehicles)

        if self.mode == "lin":
            state = np.zeros(8)
            state[phase * 4: (phase + 1) * 4] = incoming_vehicles
        else:
            state = np.zeros(4 + 1)
            state[:-1] = incoming_vehicles
            state[-1] = phase

        return state

    def save(self, name):
        path = os.path.join("saved_agents", name)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.network.state_dict(), os.path.join(path, "weights"))

    def reset(self):
        if self.is_training:
            self.train_network()

    def plot(self):
        size_array = 15
        matrix = np.zeros((size_array, size_array))

        for phase in ["WGREEN", "NGREEN"]:

            for i in range(size_array):
                for j in range(size_array):

                    incoming_e = i
                    incoming_w = i
                    incoming_n = j
                    incoming_s = j

                    if self.mode == "lin":
                        int_phase = int(phase == "WGREEN")
                        temp = np.array([incoming_n, incoming_s, incoming_e, incoming_w])
                        state = np.zeros(8)
                        state[int_phase * 4: (int_phase + 1) * 4] = temp

                    else:
                        state = np.array([incoming_n, incoming_s, incoming_e, incoming_w, int(phase == "WGREEN")])

                    q = np.array([self.q_value(state, k) for k in range(2)])
                    matrix[i, j] = np.argmax(q)

            seaborn.heatmap(matrix, cmap="hot")
            plt.title(phase)
            plt.xlabel("incoming ew")
            plt.ylabel("incoming ns")
            plt.show()

        q_values = np.zeros(10)

        phase = "WGREEN"
        for i in range(10):
            incoming_e = 0
            incoming_w = 0
            incoming_n = i
            incoming_s = i

            if self.mode == "lin":
                int_phase = int(phase == "WGREEN")
                temp = np.array([incoming_n, incoming_s, incoming_e, incoming_w])
                state = np.zeros(8)
                state[int_phase * 4: (int_phase + 1) * 4] = temp

            else:
                state = np.array([incoming_n, incoming_s, incoming_e, incoming_w, int(phase == "WGREEN")])

            q = self.q_value(state, 0)
            q_values[i] = q

        plt.plot(range(10), q_values)
        plt.title(phase)
        plt.xlabel("incoming n")
        plt.ylabel("q value")
        plt.show()

