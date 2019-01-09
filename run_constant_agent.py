from agents import ConstantAgent, SimpleAgent
from training import run_agent
import matplotlib.pyplot as plt

flow_type = "my_flow"

rewards = []
travel_times = []

factors = [1, 2, 3]
for factor in factors:
    print("factor = ", factor)

    agent = SimpleAgent(factor=factor)
    reward, n_switches, avg_travel_time = run_agent(agent, flow_type=flow_type, use_gui=False, max_t=5000)
    rewards.append(reward)
    travel_times.append(avg_travel_time)

    print("reward:", reward)
    print("avg_travel_time:", avg_travel_time)
    print("n_switches:", n_switches)
    print()

plt.scatter(factors, travel_times)
plt.title("Simple Agent on flow " + flow_type)
plt.xlabel("factor")
plt.ylabel("avg_travel_time")
plt.show()


rewards = []
travel_times = []

periods = [3, 4, 5, 6, 7]
for period in periods:
    print("period = ", period)

    agent = ConstantAgent(period=period)
    reward, n_switches, avg_travel_time = run_agent(agent, flow_type=flow_type, use_gui=False, max_t=5000)
    rewards.append(reward)
    travel_times.append(avg_travel_time)

    print("reward:", reward)
    print("avg_travel_time:", avg_travel_time)
    print("n_switches:", n_switches)
    print()

plt.scatter(periods, travel_times)
plt.title("Constant Agent on flow " + flow_type)
plt.xlabel("periods")
plt.ylabel("avg_travel_time")
plt.show()

