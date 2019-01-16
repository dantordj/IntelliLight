from training import run_agent, train_agent, train_four_agents
from agents.linqagent import LinQAgent
from agents.multi_agent_wrapper import MultiAgentWrapper
import numpy as np
import torch

torch.manual_seed(5)
np.random.seed(2)

features = ["count_incoming"]
agent = MultiAgentWrapper(features=features, mode="deep", memory_palace=False)
flow_type = "four_agents"

train_four_agents(agent, use_gui=False, max_t=5000, epochs=100)

