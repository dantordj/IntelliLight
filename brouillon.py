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



"""
sinon au niveau du code:
- j'ai touché un peu à la classe sumoEnv, notamment j'ai retiré l'histoire du encode_state car ca doit être fait
du coté de l'agent
- maintenant un agent c'est une classe avec une méthode choose_action et une méthode feedback. comme ca ya juste un 
script run_agent et tu lui passes en argument un agent et il le fait tourner. Le plus simple pour comprendre c'est de 
regarder ConstantAgent
- ce qui veut dire que quand tu run l'agent t'as juste ces trois lignes:
        action = agent.choose_action()
        env.step(action)
        reward += env.get_reward()
        agent.feedback(reward)
"""



















