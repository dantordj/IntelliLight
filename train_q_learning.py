from agents import ConstantAgent, SimpleAgent, QLearningAgent
from training import train_agent, run_agent
import matplotlib.pyplot as plt
import numpy as np

np.random.seed()
flow_type = "equal_big"
agent = QLearningAgent()
epochs = 30
rewards, avg_travel_times = train_agent(agent, flow_type=flow_type, epochs=epochs)
agent.save("q_learning")
plt.title("Q-Learning {0} - Rewards".format(flow_type))
plt.xlabel("iterations")
plt.ylabel("reward")
plt.plot(range(epochs), rewards)
plt.show()

plt.title("Q-Learning {0} - Travel Time".format(flow_type))
plt.xlabel("iterations")
plt.ylabel("travel time")
plt.scatter(range(epochs), avg_travel_times)
plt.show()
