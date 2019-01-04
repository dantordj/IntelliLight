from utils import q_learning, simple_rule2
import matplotlib.pyplot as plt

n_tests = 10
# pi, rewards, visited_states = q_learning(n_tests, t_max=1000)

pi, rewards, visited_states = simple_rule2(1, t_max=1000)

plt.plot(range(n_tests), rewards)
plt.show()