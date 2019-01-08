from agents import ConstantAgent, SimpleAgent, QLearningAgent, DeepQLearningAgent, LinQAgent
from training import run_agent, train_agent
import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(2)
flow_type = "unequal"
agent = LinQAgent()
agent.load("deep_q_learning")
agent.set_is_training(False)

incoming_e = 5
incoming_w = 5
incoming_n = 5
incoming_s = 5

state1 = np.array([incoming_n, incoming_s, incoming_e, incoming_w])
phase = "WGREEN"
phase = phase == "WGREEN"
state = np.zeros(8)
state[phase * 4: (phase + 1) * 4] = state1
state = torch.tensor(state, dtype=torch.float)


reward, n_switches, avg_travel_time = run_agent(agent, flow_type=flow_type, use_gui=True)
print(reward, n_switches, avg_travel_time)
