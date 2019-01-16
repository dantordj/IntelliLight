import seaborn
import numpy as np
import matplotlib.pyplot as plt

from utils import get_phase, yellow_nw, yellow_wn
from agents.agents import Agent


class LearningAgent(Agent):
    def __init__(self, node="node0"):
        super(LearningAgent, self).__init__(node=node)
        self.is_training = True
        self.is_online = True
        self.steps = 0
        self.mode = "none"
        self.gamma = 0.8
        self.epsilon = 0.1
        self.action = 0
        self.last_state = 0
        self.offline_period = 5

    def set_is_training(self, is_training):
        self.is_training = is_training

    def set_is_online(self, is_online):
        self.is_online = is_online

    def save(self, name):
        pass

    def load(self, name):
        pass

    def q_value(self, state, k):
        return 0

    def reset(self):
        pass

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

    def get_state(self):
        return 0

    def remember(self, state, action, target_q):
        pass

    def feedback(self, reward):

        next_state = self.get_state()

        q_next = np.array([self.q_value(next_state, i) for i in range(2)])
        q_next = np.max(q_next)

        q = reward + self.gamma * q_next

        self.remember(self.last_state, self.action, q)

        return

    def plot(self):
        size_array = 4 if self.mode == "q_learning" else 15
        matrix = np.zeros((size_array, size_array))

        for phase in ["WGREEN", "NGREEN"]:

            for i in range(size_array):
                for j in range(size_array):

                    incoming_e = i
                    incoming_w = i
                    incoming_n = j
                    incoming_s = j
                    int_phase = int(phase == "WGREEN")

                    if self.mode == "lin":
                        temp = np.array([incoming_n, incoming_s, incoming_e, incoming_w])
                        state = np.zeros(8)
                        state[int_phase * 4: (int_phase + 1) * 4] = temp

                    elif self.mode == "deep":
                        state = np.array([incoming_n, incoming_s, incoming_e, incoming_w, int(phase == "WGREEN")])
                    else:
                        state = np.array([incoming_n, incoming_s, incoming_e, incoming_w])
                        state = int_phase + np.dot(state, np.array([4 ** i for i in range(4)]) * 2)

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
            int_phase = int(phase == "WGREEN")

            if self.mode == "lin":
                temp = np.array([incoming_n, incoming_s, incoming_e, incoming_w])
                state = np.zeros(8)
                state[int_phase * 4: (int_phase + 1) * 4] = temp

            elif self.mode == "deep":
                state = np.array([incoming_n, incoming_s, incoming_e, incoming_w, int(phase == "WGREEN")])
            else:
                state = np.array([incoming_n, incoming_s, incoming_e, incoming_w])
                state = int_phase + np.dot(state, np.array([4 ** i for i in range(4)]) * 2)

            q = self.q_value(state, 0)
            q_values[i] = q

        plt.plot(range(10), q_values)
        plt.title(phase)
        plt.xlabel("incoming n")
        plt.ylabel("q value")
        plt.show()
