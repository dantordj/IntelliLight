from training import run_agent, train_agent
from agents.linqagent import LinQAgent
import numpy as np
import torch

torch.manual_seed(5)
np.random.seed(2)

features = ["count_incoming", "count_upstream"]
agent = LinQAgent(features=features, mode="deep", node="C2", memory_palace=True)

flow_type = "multi_agent"
agent.set_is_online(False)
agent.set_is_training(True)
agent.has_trained = False
agent.use_img = False
agent.observe_steps = 1000

"""
# train offline
for i in range(10):
    print("i = ", i)
    rewards, avg_travel_times = train_agent(agent, flow_type=flow_type, epochs=1, max_t=5000)
    agent.save("multi_agent_count_upstream")

agent.set_is_training(False)
agent.set_is_online(True)
reward, n_switches, avg_travel_time = run_agent(agent, flow_type=flow_type, max_t=5000)
print("evaluation mode: ", reward, n_switches, avg_travel_time)
"""

agent.set_is_training(True)
agent.set_is_online(True)
for i in range(40):
    print("i = ", i)
    rewards, avg_travel_times = train_agent(agent, flow_type=flow_type, epochs=1, max_t=5000)
    agent.save("multi_agent_count_upstream")

    if i % 5 == 0:
        agent.set_is_training(False)
        reward, n_switches, avg_travel_time = run_agent(agent, flow_type=flow_type, max_t=5000)
        print("evaluation mode: ", reward, n_switches, avg_travel_time)
        agent.set_is_training(True)
