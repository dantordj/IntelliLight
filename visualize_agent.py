from agents import ConstantAgent, SimpleAgent, QLearningAgent, DeepQLearningAgent
from training import run_agent, train_agent
import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(2)
flow_type = "equal_big"
agent = DeepQLearningAgent()
agent.epsilon = 0.00
agent.beta = 100000
agent.load("deep_q_learning")

incoming_e = 100
incoming_w = 0
incoming_n = 0
incoming_s = 0

state1 = np.array([incoming_n, incoming_s, incoming_e, incoming_w])
phase = "WGREEN"
phase = phase == "WGREEN"
state = np.zeros(8)
state[phase * 4: (phase + 1) * 4] = state1
state = torch.tensor(state, dtype=torch.float)

print(agent.network(state).detach().numpy())

reward, n_switches, avg_travel_time = run_agent(agent, flow_type=flow_type, use_gui=True)
print(reward, n_switches, avg_travel_time)
