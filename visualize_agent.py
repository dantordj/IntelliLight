from agents import ConstantAgent, SimpleAgent, QLearningAgent, LinQAgent
from training import run_agent, train_agent
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.manual_seed(5)
np.random.seed(2)

flow_type = "my_flow"
agent = ConstantAgent(period=10)

reward, n_switches, avg_travel_time = run_agent(
    agent, flow_type=flow_type, use_gui=True, max_t=5000, lane_type="slow_lane"
)
print(reward, n_switches, avg_travel_time)

