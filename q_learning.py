from agents import ConstantAgent, SimpleAgent, LinQAgent, QLearningAgent
from training import train_agent, run_agent
import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(2)
torch.manual_seed(2)

flow_type = "my_flow"
agent = QLearningAgent()
agent.set_is_online(False)

for i in range(3):
    print("i =", i)
    rewards, avg_travel_times = train_agent(agent, flow_type=flow_type, epochs=1, max_t=5000)
    agent.save("q_learning_pretrained")

agent.set_is_online(True)
for i in range(3):
    print("i =", i)
    rewards, avg_travel_times = train_agent(agent, flow_type=flow_type, epochs=1, max_t=5000)
    agent.save("q_learning_pretrained")

agent.set_is_training(False)
agent.set_is_online(True)
reward, n_switches, avg_travel_time = run_agent(agent, flow_type=flow_type, max_t=5000)
print(reward, n_switches, avg_travel_time)