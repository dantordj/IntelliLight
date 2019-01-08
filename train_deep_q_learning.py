from agents import ConstantAgent, SimpleAgent, DeepQLearningAgent, LinQAgent
from training import train_agent, run_agent
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2)

flow_type = "unequal_big"
agent = LinQAgent()
agent.set_is_online(False)

for i in range(2):
    print("i = ", i)
    rewards, avg_travel_times = train_agent(agent, flow_type=flow_type, epochs=5)
    agent.save("deep_q_learning_pretrained_offline")
    agent.plot()

agent.set_is_training(False)
agent.set_is_online(False)
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
