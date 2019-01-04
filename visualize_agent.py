from utils import q_learning, simple_rule2, ConstantAgent, SimpleAgent, run_agent, QLearningAgent, train_agent
import matplotlib.pyplot as plt

flow_type = "equal_big"
agent = QLearningAgent()
agent.load("q_learning")
rewards = run_agent(agent, flow_type=flow_type)