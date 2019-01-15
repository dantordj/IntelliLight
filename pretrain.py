from agents.linqagent import LinQAgent
from training import train_agent, run_agent
import numpy as np
import torch

np.random.seed(2)
torch.manual_seed(2)

flow_type = "my_flow"
lane_type = "slow_lane"
features = ["count_incoming", "median_speed", "mean_speed", "img"]
agent = LinQAgent(mode="conv", features=features)
agent.set_is_online(False)
agent.use_img = True

for i in range(10):
    print("i =", i)
    rewards, avg_travel_times = train_agent(agent, flow_type=flow_type, epochs=1, max_t=5000)
    agent.save("lin_q_img_pretrained")
    # agent.plot()

agent.set_is_training(False)
agent.set_is_online(True)
reward, n_switches, avg_travel_time = run_agent(agent, flow_type=flow_type, max_t=5000)
print(reward, n_switches, avg_travel_time)
