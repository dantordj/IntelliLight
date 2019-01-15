from utils import start_sumo, end_sumo
from environnements import sumoEnv
import time
import numpy as np

n_steps = 5


def run_agent(agent, max_t=1000, flow_type="unequal", lane_type="uniform", use_gui=False):
    start_sumo(flow_type, lane_type=lane_type, use_gui=use_gui)

    if flow_type == "multi_agent":
        env = sumoEnv(multi_agent=True)
    else:
        env = sumoEnv()
    reward = 0
    n_switches = 0

    t = 0
    while (t < max_t):
        action = agent.choose_action()
        n_switches += int(action)

        current_reward = 0

        env.step(action)

        for i in range(n_steps):
            if use_gui:
                time.sleep(0.2)
            env.step(0)
            r = env.get_reward()
            current_reward += r
            print(r)
            t += 1
        if t % 100 == 0:
            print("t = ", t)

        agent.feedback(current_reward)
        reward += current_reward

    agent.reset()

    reward /= max_t

    avg_travel_time = env.get_avg_travel_time()

    end_sumo()

    return reward, n_switches, avg_travel_time


def train_agent(agent, epochs=1, max_t=1000, flow_type="unequal"):
    rewards = []
    avg_travel_times = []

    for i in range(epochs):

        print("start epoch: ", i)
        reward, n_switches, avg_travel_time = run_agent(agent, max_t=max_t, flow_type=flow_type)
        rewards.append(reward)
        avg_travel_times.append(avg_travel_time)

        print("end epoch: ", i)
        print("reward: ", reward)
        print("avg_travel_time: ", avg_travel_time)
        print("n switches: ", n_switches)
        try:
            print("agent.visited_states: ", np.sum(agent.visited_states > 0))
        except AttributeError:
            pass

    return rewards, avg_travel_times
