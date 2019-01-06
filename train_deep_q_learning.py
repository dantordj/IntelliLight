from agents import ConstantAgent, SimpleAgent, DeepQLearningAgent
from training import train_agent, run_agent
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2)

flow_type = "equal_big"
agent = DeepQLearningAgent()
epochs = 50
rewards, avg_travel_times = train_agent(agent, flow_type=flow_type, epochs=epochs)
agent.save("deep_q_learning")
plt.plot(range(epochs), rewards)
plt.show()

plt.plot(range(epochs), avg_travel_times)
plt.show()
