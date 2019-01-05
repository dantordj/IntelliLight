from utils import q_learning, simple_rule2, ConstantAgent, SimpleAgent, run_agent, QLearningAgent, train_agent
import matplotlib.pyplot as plt

flow_type = "equal_big"
agent = QLearningAgent()
epochs = 10
rewards, avg_travel_times = train_agent(agent, flow_type=flow_type, epochs=epochs)
agent.save("q_learning")
plt.scatter(range(epochs), rewards)
plt.show()

plt.scatter(range(epochs), avg_travel_times)
plt.show()