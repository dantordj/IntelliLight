from training import run_agent, train_agent
from agents.linqagent import LinQAgent
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(5)
np.random.seed(2)

flow_type = "multi_agent"
features = ["count_incoming", "count_upstream"]
agent = LinQAgent(mode="deep", node="C2", memory_palace=True, features=features)
agent.set_is_online(True)
agent.set_is_training(False)
agent.load("multi_agent_count_upstream")

reward, n_switches, avg_travel_time = run_agent(
    agent, flow_type=flow_type, use_gui=True, max_t=2000
)
print(reward, n_switches, avg_travel_time)
