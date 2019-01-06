from agents import ConstantAgent, SimpleAgent, QLearningAgent, DeepQLearningAgent
from training import run_agent, train_agent
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(2)
flow_type = "equal_big"
agent = QLearningAgent()
agent.epsilon = 0.01
agent.beta = 100000
agent.load("q_learning")
reward, n_switches, avg_travel_time = run_agent(agent, flow_type=flow_type, use_gui=True)
print(reward, n_switches, avg_travel_time)
