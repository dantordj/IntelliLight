from agents import ConstantAgent, SimpleAgent
from training import run_agent
import matplotlib.pyplot as plt

flow_type = "unequal_big"

# for constant, unequal: avg_travel_time: 68

rewards = []
travel_times = []

factors = [1, 2, 3]
for factor in factors:
    print("factor = ", factor)

    agent = SimpleAgent(factor=factor)
    reward, n_switches, avg_travel_time = run_agent(agent, flow_type=flow_type, use_gui=False)
    rewards.append(reward)
    travel_times.append(avg_travel_time)

    print("reward:", reward)
    print("avg_travel_time:", avg_travel_time)
    print("n_switches:", n_switches)
    print()

plt.scatter(periods, rewards)
plt.show()

plt.scatter(periods, travel_times)
plt.show()