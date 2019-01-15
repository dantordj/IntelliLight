from utils import get_phase, wgreen, ngreen, get_state_sumo
import os
import numpy as np


class Agent(object):
    def __init__(self, node="node0"):
        self.node = node
        pass

    def choose_action(self):
        return False

    def feedback(self, reward):
        pass

    def reset(self):
        pass


class ConstantAgent(Agent):
    def __init__(self, period=30, node="node0"):
        super(ConstantAgent, self).__init__(node)
        self.period = period
        self.count = 0

    def choose_action(self):
        self.count += 1

        if (self.count > self.period) and (get_phase(self.node) in [wgreen, ngreen]):
            self.count = 0
            return 1
        return 0


class SimpleAgent(Agent):
    def __init__(self, factor, node="node0"):
        super(SimpleAgent, self).__init__(node)
        self.factor = factor

    def choose_action(self):
        count_incoming, speed_incoming, img = get_state_sumo(self.node)

        vertical_cars = 0
        horizontal_cars = 0
        for line, num_vehicles in count_incoming.items():
            try:
                i = int(line[4])
            except:
                continue
            if i in [3, 4]:
                vertical_cars += num_vehicles
            elif i in [1, 2]:
                horizontal_cars += num_vehicles

        change = False

        if (vertical_cars > self.factor * horizontal_cars) and get_phase(self.node) == wgreen:
            change = True

        if (horizontal_cars > self.factor * vertical_cars) and get_phase(self.node) == ngreen:
            change = True

        return change


class MyNormalizer(object):
    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.ones(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()

        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = np.clip(self.mean_diff / self.n, a_min=1e-2, a_max=1e12)

    def normalize(self, inputs):
        obs_std = np.sqrt(self.var)
        return (inputs - self.mean) / obs_std

    def save(self, path):
        to_store = np.array([self.n, self.mean, self.mean_diff, self.var])
        np.savetxt(os.path.join(path, "normalizer.txt"), to_store)

    def load(self, path):
        to_store = np.loadtxt(os.path.join(path, "normalizer.txt"))
        self.n = to_store[0]
        self.mean = to_store[1]
        self.mean_diff = to_store[2]
        self.var = to_store[3]
