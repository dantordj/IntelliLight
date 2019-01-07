from agents import ConstantAgent, SimpleAgent, DeepQLearningAgent, LinQAgent
from training import train_agent, run_agent
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2)

flow_type = "equal_big"
agent = LinQAgent()
epochs = 10
rewards, avg_travel_times = train_agent(agent, flow_type=flow_type, epochs=epochs)

agent.epsilon = 0.
reward, n_switches, avg_travel_time = run_agent(agent, flow_type=flow_type)

print(reward, n_switches, avg_travel_time)

agent.save("deep_q_learning")
plt.title("Deep Q-Learning {0} - Rewards".format(flow_type))
plt.xlabel("iterations")
plt.ylabel("reward")
plt.plot(range(epochs), rewards)
plt.show()

plt.title("Deep Q-Learning {0} - Travel Time".format(flow_type))
plt.xlabel("iterations")
plt.ylabel("travel time")
plt.plot(range(epochs), avg_travel_times)
plt.show()
