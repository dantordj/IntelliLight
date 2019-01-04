from utils import q_learning, simple_rule2, constant_rule
import matplotlib.pyplot as plt

n_tests = 10

for period in [40, 60, 80, 100, 120, 200]:
    avg_reward = constant_rule(t_max=1000, display_freq=1e12, period=period)
    print("avg reward simple rule, period = " + str(period) + ": ", avg_reward)

pi, rewards, visited_states = q_learning(n=n_tests, t_max=1000, display_freq=1e12, epsilon=0)

for factor in [1, 2, 3, 4]:
    avg_reward = simple_rule2(t_max=1000, display_freq=1e12, factor=factor)
    print("avg reward simple rule, factor = " + str(factor) + ": ", avg_reward)


# print("avg reward: ", avg_reward)
plt.plot(range(n_tests), rewards)
plt.show()

#         <phase duration="6" state="grrrgyyygrrrgyyy"/>
#         <phase duration="6" state="gyyygGGGgyyygGGG"/>