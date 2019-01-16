from utils import start_sumo, end_sumo
from environnements import SumoEnv
import time
import numpy as np

n_steps = 5


def run_agent(agent, max_t=1000, flow_type="unequal", lane_type="uniform", use_gui=False):
    start_sumo(flow_type, lane_type=lane_type, use_gui=use_gui)

    if flow_type == "multi_agent":
        env = SumoEnv(multi_agent=True)
    else:
        env = SumoEnv()
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


def run_four_agents(agent, use_gui=False, max_t=2000):
    start_sumo("four_agents", use_gui=use_gui)

    env = SumoEnv()

    reward = 0
    n_switches = 0

    t = 0
    while (t < max_t):
        actions = agent.choose_action()
        n_switches += sum(actions.values())

        env.step_multi_agents(actions)

        rewards = {key: 0 for key in agent.agents.keys()}

        for i in range(n_steps):
            if use_gui:
                time.sleep(0.2)
            env.step_multi_agents({key: 0 for key in agent.agents.keys()})

            for node in agent.agents.keys():
                rewards[node] += env.get_reward(node)
            t += 1
        if t % 100 == 0:
            print("t = ", t)

        print(rewards)
        agent.feedback(rewards)
        reward += sum(rewards.values())

    agent.reset()
    reward /= max_t

    avg_travel_time = env.get_avg_travel_time()

    end_sumo()

    return reward, n_switches, avg_travel_time


def train_four_agents(
        agent, epochs=10, max_t=1000, use_gui=False, offline_epochs=10, switch_every=5, eval_every=5
):
    rewards = []
    avg_travel_times = []

    for a in agent.agents.values():
        a.set_is_training(True)
        a.set_is_online(False)

    for i in range(offline_epochs):

        print("start offline epoch: ", i)

        reward, n_switches, avg_travel_time = run_four_agents(agent, max_t=max_t, use_gui=use_gui)
        rewards.append(reward)
        avg_travel_times.append(avg_travel_time)

        print("end offline epoch: ", i)
        print("reward: ", reward)
        print("avg_travel_time: ", avg_travel_time)
        print("n switches: ", n_switches)

    for a in agent.agents.values():
        a.set_is_online(True)

    for i in range(epochs):
        print("start offline epoch: ", i)

        for j, a in enumerate(agent.agents.values()):
            a.set_is_training(False)

            if j % 4 == (i // switch_every):
                print("switching to agent")
                a.set_is_training(True)

        reward, n_switches, avg_travel_time = run_four_agents(agent, max_t=max_t)
        rewards.append(reward)
        avg_travel_times.append(avg_travel_time)

        print("end offline epoch: ", i)
        print("reward: ", reward)
        print("avg_travel_time: ", avg_travel_time)
        print("n switches: ", n_switches)

        if i % eval_every == 0:
            for a in agent.agents.values():
                a.set_is_training(False)
                a.set_is_online(True)

        reward, n_switches, avg_travel_time = run_four_agents(agent, max_t=max_t, use_gui=use_gui)

        print("eval number " + str(i % eval_every) + ": ", reward, n_switches, avg_travel_time)

    return rewards, avg_travel_times
