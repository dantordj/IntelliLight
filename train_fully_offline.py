from training import run_four_agents
from agents.fully_offline import OfflineAgent
from agents.multi_agent_wrapper import MultiAgentWrapper
import numpy as np
import torch

torch.manual_seed(5)
np.random.seed(2)

my_agent_class = OfflineAgent

features = ["count_incoming", "count_upstream"]
agent = MultiAgentWrapper(
    agent_class=my_agent_class
)

agent.set_is_online(False)
agent.set_is_training(True)

for i in range(30):

    for node, a in agent.agents.items():
        a.offline_period = np.random.choice([3, 4, 5, 6])

    reward, n_switches, avg_travel_time = run_four_agents(agent=agent, use_gui=False, max_t=5000)
    print("i: ", i)
    print("reward, n_switches, avg_travel_time: ", reward, n_switches, avg_travel_time)
    agent.save("offline")

agent.train()

