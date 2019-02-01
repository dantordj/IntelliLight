from training import run_four_agents
from agents.agents import ConstantAgent, SimpleAgent
from agents.multi_agent_wrapper import MultiAgentWrapper
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2)

rewards = []
travel_times = []

factors = [1, 2, 3]
for factor in factors:
    print("factor = ", factor)

    my_agent_class = SimpleAgent

    agent = MultiAgentWrapper(
        agent_class=my_agent_class, factor=factor
    )

    reward, n_switches, avg_travel_time = run_four_agents(agent=agent, use_gui=True, max_t=5000)
    rewards.append(reward)
    travel_times.append(avg_travel_time)

    print("reward:", reward)
    print("avg_travel_time:", avg_travel_time)
    print("n_switches:", n_switches)
    print()

plt.scatter(factors, travel_times)
plt.title("Simple Agent on four agents")
plt.xlabel("factor")
plt.ylabel("avg_travel_time")
plt.show()


rewards = []
travel_times = []

periods = [3, 4, 5, 6, 7]
for period in periods:
    print("period = ", period)

    my_agent_class = ConstantAgent

    agent = MultiAgentWrapper(
        agent_class=my_agent_class, period=period
    )

    reward, n_switches, avg_travel_time = run_four_agents(agent=agent, use_gui=False, max_t=5000)
    rewards.append(reward)
    travel_times.append(avg_travel_time)

    print("reward:", reward)
    print("avg_travel_time:", avg_travel_time)
    print("n_switches:", n_switches)
    print()

plt.scatter(periods, travel_times)
plt.title("Constant Agent on four agents")
plt.xlabel("periods")
plt.ylabel("avg_travel_time")
plt.show()


