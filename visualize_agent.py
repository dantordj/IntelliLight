from agents import ConstantAgent, SimpleAgent
from training import run_agent, train_agent
from linqagent import LinQAgent
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.manual_seed(5)
np.random.seed(2)

flow_type = "my_flow"
features = ["count_incoming", "median_speed", "mean_speed", "img"]
agent = LinQAgent(mode="conv", features=features)
agent.load("lin_q_im_trained29")
agent.set_is_online(True)
agent.set_is_training(False)
agent.use_img = True

reward, n_switches, avg_travel_time = run_agent(
    agent, flow_type=flow_type, use_gui=True, max_t=2000, lane_type="slow_lane"
)
print(reward, n_switches, avg_travel_time)
