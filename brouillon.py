from utils import q_learning, simple_rule2, ConstantAgent, SimpleAgent, run_agent, QLearningAgent, train_agent
import matplotlib.pyplot as plt

flow_type = "equal_big"
agent = QLearningAgent()
epochs = 10
rewards = train_agent(agent, flow_type=flow_type, epochs=epochs)
agent.save("q_learning")
plt.scatter(range(epochs), rewards)
plt.show()


n_tests = 10
factors = [2, 3, 4, 5]
flow_type = "equal_big"
rewards = []
for factor in factors:
    agent = SimpleAgent(factor=factor)
    reward, n_switches = run_agent(agent, flow_type=flow_type)
    rewards.append(reward)

    print("period = ", factor)
    print("reward:", reward)
    print("n_switches:", n_switches)
    print()

plt.scatter(factors, rewards)
plt.show()

rewards = []
periods = [10, 20, 40, 60, 80, 100, 120, 200]
for period in periods:
    agent = ConstantAgent(period=period)
    reward, n_switches = run_agent(agent, flow_type=flow_type)
    rewards.append(reward)

    print("period = ", period)
    print("reward:", reward)
    print("n_switches:", n_switches)
    print()

plt.scatter(periods, rewards)
plt.show()