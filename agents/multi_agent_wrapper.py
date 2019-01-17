from agents.linqagent import LinQAgent


class MultiAgentWrapper:

    def __init__(self, agent_class=LinQAgent, offsets=None, **kwargs):

        nodes = ["C2", "D2", "C3", "D3"]

        # parameters
        self.agents = {}

        if offsets is None:
            offsets = {node: i + 3 for (i, node) in enumerate(nodes)}

        for node in nodes:
            self.agents[node] = agent_class(
                node=node, **kwargs
            )

            self.agents[node].offline_period = offsets[node]

    def choose_action(self):
        ans = {}
        for node, agent in self.agents.items():
            ans[node] = agent.choose_action()

        return ans

    def feedback(self, rewards):
        for node, agent in self.agents.items():
            agent.feedback(rewards[node])

    def save(self, name):
        for node, agent in self.agents.items():
            agent.save(name + "_" + node)

    def load(self, name):
        for node, agent in self.agents.items():
            agent.load(name + "_" + node)

    def reset(self):
        for agent in self.agents.values():
            agent.reset()

    def set_is_online(self, is_online=True):
        for node, agent in self.agents.items():
            agent.set_is_online(is_online)

    def set_is_training(self, is_training=True):
        for node, agent in self.agents.items():
            agent.set_is_training(is_training)

    def train(self):
        for node, agent in self.agents.items():
            print("training agent ", node)
            agent.train_network()
            print("")
            print("")
            print("")
            print("")
            print("")
