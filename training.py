from utils import start_sumo, end_sumo
from environnements import sumoEnv
import time
import numpy as np


def run_agent(agent, max_t=1000, flow_type="unequal", use_gui=False):
    start_sumo(flow_type, use_gui=use_gui)
    env = sumoEnv()
    reward = 0
    n_switches = 0

    for t in range(max_t):
        action = agent.choose_action()
        n_switches += int(action)
        env.step(action)
        reward += env.get_reward()
        agent.feedback(reward)

        if use_gui:
            time.sleep(0.3)

        agent.reset()

    reward /= max_t

    avg_travel_time = env.get_avg_travel_time()

    print("len all vehicles: ", len(env.all_vehicles))
    print("len arrived vehicles: ", len(env.arrived_vehicles))

    end_sumo()

    return reward, n_switches, avg_travel_time


def train_agent(agent, epochs=1, max_t=1000, flow_type="unequal"):
    rewards = []
    avg_travel_times = []

    for i in range(epochs):

        print("epoch: ", i)
        reward, n_switches, avg_travel_time = run_agent(agent, max_t=max_t, flow_type=flow_type)
        rewards.append(reward)
        avg_travel_times.append(avg_travel_time)

        print("reward: ", reward)
        print("avg_travel_time: ", avg_travel_time)
        try:
            print("agent.visited_states: ", np.sum(agent.visited_states > 0))
        except AttributeError:
            pass

    return rewards, avg_travel_times
