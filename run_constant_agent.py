from utils import ConstantAgent, run_agent
import matplotlib.pyplot as plt

flow_type = "equal_big"

rewards = []
travel_times = []

periods = [20, 60, 100]
for period in periods:
    print("period = ", period)

    agent = ConstantAgent(period=period)
    reward, n_switches, avg_travel_time = run_agent(agent, flow_type=flow_type)
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