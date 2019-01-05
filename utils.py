import sys
import os

# os.environ["SUMO_HOME"] = "../sumo/"

sys.path.append(
    os.path.join(os.environ["SUMO_HOME"], "tools")
)
import traci
import traci.constants as tc
import time
from collections import Counter
import xml.etree.ElementTree as ET
import numpy as np
import math
import matplotlib.pyplot as plt
from env_name import env_name

use_gui = False

if env_name == "raph":
    sumo_binary_path = os.path.join('c:', os.sep, "Program Files (x86)", "Eclipse", "Sumo", "bin", "sumo")
    sumo_gui_binary_path = os.path.join('c:', os.sep, "Program Files (x86)", "Eclipse", "Sumo", "bin", "sumo-gui")

else:
    sumo_binary_path = '/usr/local/bin/sumo'
    sumo_gui_binary_path = '/usr/local/bin/sumo-gui'

# vertical lanes start with edge 4 and edge 3
area_length = 600
grid_width = 4
listLanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2',
             'edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']

# assgin sumo code to each phase

wgreen = "WGREEN"
ngreen = "NGREEN"
yellow_wn = "YELLOW_WN"
yellow_nw = "YELLOW_NW"

phases = {
    wgreen: "grrr gGGG grrr gGGG".replace(" ", ""),  # index 0
    ngreen: "gGGG grrr gGGG grrr".replace(" ", "")  # index 1
}


def get_id_phase(phase):
    if phase == phases[wgreen]:
        return wgreen
    if phase == phases[ngreen]:
        return ngreen


phase_decoder = {
    0: wgreen,
    1: yellow_wn,
    2: ngreen,
    3: yellow_nw
}

phase_encoder = {
    wgreen: 0,
    yellow_wn: 1,
    ngreen: 2,
    yellow_nw: 3
}


def get_phase():
    return phase_decoder[traci.trafficlight.getPhase("node0")]


def set_phase(phase):
    traci.trafficlight.setPhase("node0", phase_encoder[phase])


""" Functions to interact with sumo """


class Vehicles:
    initial_speed = 5.0

    def __init__(self):
        # add what ever you need to maintain
        self.id = None
        self.speed = None
        self.wait_time = None
        self.stop_count = None
        self.enter_time = None
        self.has_read = False
        self.first_stop_time = -1
        self.entering = True


def set_traffic_file(sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name):
    """ Set the traffic file in the sumo config"""
    # update sumocfg
    sumo_cfg = ET.parse(sumo_config_file_tmp_name)
    config_node = sumo_cfg.getroot()
    input_node = config_node.find("input")
    for route_files in input_node.findall("route-files"):
        input_node.remove(route_files)
    input_node.append(
        ET.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)}))
    sumo_cfg.write(sumo_config_file_output_name)


def start_sumo(traffic, use_gui=False):
    """ Start sumo, 3 config possibles"""
    trafic_files = {"alternate": "cross.2phases_rou1_switch_rou0.xml",
                    "equal": "cross.2phases_rou01_equal_300s.xml",
                    "unequal": "cross.2phases_rou01_unequal_5_300s.xml",
                    "equal_big": "cross.2phases_rou01_equal_300s_big.xml"
                    }
    file = trafic_files[traffic]
    set_traffic_file(
        os.path.join('data/one_run', "cross.sumocfg"),
        os.path.join('data/one_run', "cross.sumocfg"),
        [file])

    file_path = os.path.join("data", "one_run", "cross.sumocfg")

    sumoCmd = [sumo_gui_binary_path if use_gui else sumo_binary_path, "-c", file_path]

    traci.start(sumoCmd)
    for i in range(20):
        traci.simulationStep()


def get_state_sumo():
    """ Put here what we need to define the state. For now only the number of vehicles by lines"""
    vehicle_roads = Counter()

    vehicle_id_list = traci.vehicle.getIDList()
    for vehicle_id in vehicle_id_list:
        road_id = traci.vehicle.getRoadID(vehicle_id)
        vehicle_roads[road_id] += 1
    return vehicle_roads


def is_blocked(lane, phase):
    """ return True if the line is blocked"""
    if "edge3" in lane and phase == wgreen:
        return True
    if "edge4" in lane and phase == wgreen:
        return True
    if "edge1" in lane and phase == ngreen:
        return True
    if "edge2" in lane and phase == ngreen:
        return True
    return False


def get_overall_queue_length(listLanes, blocked_only=False):
    """ return queue length, overall or only the blockes lines """
    overall_queue_length = 0
    sumo_phase = traci.trafficlights.getRedYellowGreenState("node0")
    phase = get_id_phase(sumo_phase)
    for lane in listLanes:

        blocked = is_blocked(lane, phase)
        if not blocked_only or blocked:
            overall_queue_length += traci.lane.getLastStepHaltingNumber(lane)

    return overall_queue_length


def get_overall_waiting_time(listLanes):
    """ Unused for now"""
    overall_waiting_time = 0
    for lane in listLanes:
        overall_waiting_time += traci.lane.getWaitingTime(str(lane)) / 60.0

    return overall_waiting_time


def get_travel_time_duration(vehicle_dict, vehicle_id_list):
    """ Unused for now"""
    travel_time_duration = 0
    for vehicle_id in vehicle_id_list:
        if (vehicle_id in vehicle_dict.keys()):
            travel_time_duration += (traci.simulation.getCurrentTime() / 1000 - vehicle_dict[
                vehicle_id].enter_time) / 60.0
    if len(vehicle_id_list) > 0:
        return travel_time_duration  # /len(vehicle_id_list)
    else:
        return 0


def get_vehicle_id_entering():
    vehicle_id_entering = []
    entering_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2',
                      'edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']

    for lane in entering_lanes:
        vehicle_id_entering.extend(traci.lane.getLastStepVehicleIDs(lane))

    return vehicle_id_entering


def update_vehicles_state(dic_vehicles):
    """ Update a dictionnary with vehicles classed based on the current state of the simulation"""
    vehicle_id_list = traci.vehicle.getIDList()
    vehicle_id_entering_list = get_vehicle_id_entering()
    for vehicle_id in (set(dic_vehicles.keys()) - set(vehicle_id_list)):
        del (dic_vehicles[vehicle_id])

    for vehicle_id in vehicle_id_list:
        if (vehicle_id in dic_vehicles.keys()) == False:
            vehicle = Vehicles()
            vehicle.id = vehicle_id
            traci.vehicle.subscribe(vehicle_id, (tc.VAR_LANE_ID, tc.VAR_SPEED))
            vehicle.speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64)
            current_sumo_time = traci.simulation.getCurrentTime() / 1000
            vehicle.enter_time = current_sumo_time
            # if it enters and stops at the very first
            if (vehicle.speed < 0.1) and (vehicle.first_stop_time == -1):
                vehicle.first_stop_time = current_sumo_time
            dic_vehicles[vehicle_id] = vehicle
        else:
            dic_vehicles[vehicle_id].speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64)
            if (dic_vehicles[vehicle_id].speed < 0.1) and (dic_vehicles[vehicle_id].first_stop_time == -1):
                dic_vehicles[vehicle_id].first_stop_time = traci.simulation.getCurrentTime() / 1000
            if (vehicle_id in vehicle_id_entering_list) == False:
                dic_vehicles[vehicle_id].entering = False

    return dic_vehicles


""" Function to plot state of the traffic as an image"""


def vehicle_location_mapper(coordinate, area_length=600, area_width=600):
    transformX = math.floor(coordinate[0] / grid_width)
    transformY = math.floor((area_length - coordinate[1]) / grid_width)
    length_num_grids = int(area_length / grid_width)
    transformY = length_num_grids - 1 if transformY == length_num_grids else transformY
    transformX = length_num_grids - 1 if transformX == length_num_grids else transformX
    tempTransformTuple = (transformY, transformX)
    return tempTransformTuple


def plotcurrenttrafic():
    """ Plot the curretn state of the traffic"""
    length_num_grids = int(area_length / grid_width)
    mapOfCars = np.zeros((length_num_grids, length_num_grids))

    vehicle_id_list = traci.vehicle.getIDList()
    for vehicle_id in vehicle_id_list:
        vehicle_position = traci.vehicle.getPosition(vehicle_id)  # (double,double),tuple

        transform_tuple = vehicle_location_mapper(vehicle_position)  # call the function
        mapOfCars[transform_tuple[0], transform_tuple[1]] = 1
    plt.imshow(mapOfCars)
    plt.show()


def end_sumo():
    traci.close()


def plottraffic(N):
    """ Plot evolution of the traffic for N steps without changing the phase"""
    for i in range(N):
        length_num_grids = int(area_length / grid_width)
        for i in range(10):
            traci.simulationStep()
        plotcurrenttrafic()
        print(get_state_sumo())


""" Defintion environement and first definition of policy """


class sumoEnv():

    def __init__(self):
        self.vehicle_dict = {}
        # self.n_states = (4 ** 4) * 2
        self.n_actions = 2
        self.all_vehicles = {}
        self.arrived_vehicles = {}

    def get_reward(self, blocked_only=True):
        # queue = get_overall_queue_length(listLanes, blocked_only=blocked_only)
        # w_time = get_overall_waiting_time(listLanes)

        reward = 0
        vehicle_entering = get_vehicle_id_entering()
        for id_ in vehicle_entering:
            reward += traci.vehicle.getSpeed(id_) - traci.lane.getMaxSpeed(traci.vehicle.getLaneID(id_))

        return reward

    def update_vehicles_dic(self):

        running_vehicles = traci.vehicle.getIDList()
        arrived_vehicles = traci.simulation.getArrivedIDList()

        # print("len all vehicles: ", len(running_vehicles))
        # print("len arrived_vehicles: ", len(arrived_vehicles))

        current_time = traci.simulation.getCurrentTime() / 1000

        for vehicle_id in running_vehicles:

            if vehicle_id not in self.all_vehicles.keys():
                self.all_vehicles[vehicle_id] = {
                    "start_time": current_time
                }

        for vehicle_id in arrived_vehicles:
            if vehicle_id not in self.arrived_vehicles.keys():
                self.arrived_vehicles[vehicle_id] = {
                    "stop_time": current_time,
                    "travel_time": current_time - self.all_vehicles[vehicle_id]["start_time"]
                }

    def get_avg_travel_time(self):
        current_time = traci.simulation.getCurrentTime() / 1000

        avg_travel_time = 0

        for key, value in self.all_vehicles.items():
            if key in self.arrived_vehicles.keys():
                avg_travel_time += self.arrived_vehicles[key]["travel_time"]
            else:
                avg_travel_time += current_time - self.all_vehicles[key]["start_time"]

        return avg_travel_time / len(self.all_vehicles)

    def step(self, change=True):

        if change and get_phase() in [yellow_wn, yellow_nw]:
            print("cant change phase since the light is already yellow")

        if change:
            if get_phase() == wgreen:
                set_phase(yellow_wn)

            elif get_phase() == ngreen:
                set_phase(yellow_nw)

        traci.simulationStep()

        self.update_vehicles_dic()


class ConstantAgent(object):
    def __init__(self, period=30):
        self.period = period
        self.count = 0

    def choose_action(self):
        self.count += 1

        if (self.count > self.period) and get_phase() in [wgreen, ngreen]:
            self.count = 0
            return True
        return False

    def feedback(self, reward):
        pass


class SimpleAgent(object):
    def __init__(self, factor):
        self.factor = factor

    def choose_action(self):
        state = get_state_sumo()

        vertical_cars = 0
        horizontal_cars = 0

        for line, num_vehicles in state.items():
            try:
                i = int(line[4])
            except:
                continue
            if i in [3, 4]:
                vertical_cars += num_vehicles
            elif i in [1, 2]:
                horizontal_cars += num_vehicles

        change = False

        if (vertical_cars > self.factor * horizontal_cars) and get_phase() == wgreen:
            change = True

        if (horizontal_cars > self.factor * vertical_cars) and get_phase() == ngreen:
            change = True

        return change

    def feedback(self, reward):
        pass


class QLearningAgent(object):

    def __init__(self):
        self.t = 0
        self.n_states = (4 ** 4) * 2
        self.Q = np.zeros((self.n_states, 2))
        self.T = np.zeros((self.n_states, 2))
        self.gamma = 0.95
        self.epsilon = 0.05
        self.beta = 0.5
        self.action = 0
        self.last_state = 0

        self.acc_reward = 0
        self.acc_count = 0
        self.visited_states = np.zeros(self.n_states)

    def load(self, name):
        path = os.path.join("saved_agents", name)
        assert os.path.exists(path), "no such saved agent"

        self.Q = np.loadtxt(os.path.join(path, "Q.txt"))
        self.visited_states = np.loadtxt(os.path.join(path, "visited_states.txt"))
        self.T = np.loadtxt(os.path.join(path, "T.txt"))

    def encode_state(self):
        # assign an integer between 1 and 511 to each state
        phase = get_phase() == wgreen
        state = np.array([0, 0, 0, 0])

        for line, num_vehicles in get_state_sumo().items():
            try:
                i = int(line[4]) - 1
                if i < 0:
                    continue
            except:
                continue
            if num_vehicles < 3:
                state[i] = 0
            elif num_vehicles < 5:
                state[i] = 1
            elif num_vehicles < 7:
                state[i] = 2
            else:
                state[i] = 3

        s = phase + np.dot(state, np.array([4 ** i for i in range(4)]) * 2)
        return s

    def choose_action(self):
        state = self.encode_state()

        if get_phase() in [yellow_wn, yellow_nw]:
            return 0

        self.last_state = state

        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice([0, 1])
        else:
            action = np.argmax(self.Q[state])

        self.action = action

        if action:
            self.acc_count = 0
            self.acc_reward = 0

        self.T[state, action] += 1

        return action

    def feedback(self, reward):
        next_state = self.encode_state()

        if next_state in [yellow_nw, yellow_wn]:
            self.acc_reward += reward * (self.gamma ** self.acc_count)
            self.acc_count += 1
            return

        q = self.Q[self.last_state, self.action]
        q_next = np.max(self.Q[next_state])

        alpha = (1. / self.T[self.last_state, self.action]) ** self.beta

        q = (1 - alpha) * q

        if self.acc_count > 0:
            self.acc_reward += reward * (self.gamma ** self.acc_count)
            q += self.acc_reward
            q += self.gamma ** (self.acc_count + 1) * q_next
            self.acc_count = 0
            self.acc_reward = 0
        else:
            q += reward + self.gamma * q_next

        self.Q[self.last_state, self.action] = q

        self.visited_states[self.last_state] += 1

        return

    def save(self, name):
        path = os.path.join("saved_agents", name)
        if not os.path.exists(path):
            os.mkdir(path)

        np.savetxt(os.path.join(path, "Q.txt"), self.Q)
        np.savetxt(os.path.join(path, "visited_states.txt"), self.visited_states)
        np.savetxt(os.path.join(path, "T.txt"), self.T)


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

    reward /= max_t

    avg_travel_time = env.get_avg_travel_time()

    # print("agent.visited_states: ", np.sum(agent.visited_states > 0))
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

    return rewards, avg_travel_times


def q_learning(n=10, epsilon=0.01, beta=0.55, t_max=1000, display_freq=1e12):
    start_sumo("unequal")
    env = sumoEnv()
    Q = np.zeros((env.n_states, 2))
    T = np.zeros((env.n_states, 2))
    gamma = 0.99

    end_sumo()

    rewards = []
    for i in range(n):
        print("i = :", i)
        t = 0
        start_sumo("unequal")
        env = sumoEnv()
        s = env.encode_state()
        visited_states[s] += 1

        reward = 0
        while t <= t_max:
            t += 1

            if np.random.uniform(0, 1) < epsilon:
                a = np.random.choice([0, 1])
            else:
                a = np.argmax(Q[s])
            T[s, a] += 1
            next_s, r = env.step(a)

            q = Q[s, a]
            q_next = np.max(Q[next_s])
            alpha = (1. / T[s, a]) ** beta
            q = (1 - alpha) * q + alpha * (r + gamma * q_next)
            Q[s, a] = q

            s = next_s
            visited_states[s] += 1
            reward += r

            if t % display_freq == 0:
                plotcurrenttrafic()

            time.sleep(0.5)

        print("reward = ", reward)

        end_sumo()
        rewards += [reward]
    print("rewards: ", rewards)

    pi = np.argmax(Q, axis=1)
    return pi, rewards, visited_states


def simple_rule2(t_max=1000, display_freq=1e12, factor=3):

    t = 0
    start_sumo("unequal")
    env = sumoEnv()
    reward = 0

    while t <= t_max:
        t += 1

        vertical_cars = 0
        horizontal_cars = 0

        for line, num_vehicles in env.state_sumo.items():
            try:
                i = int(line[4])
                if i <= 0:
                    continue
            except:
                continue
            if i in [3, 4]:
                vertical_cars += num_vehicles
            else:
                horizontal_cars += num_vehicles

        change = False

        if (vertical_cars > factor * horizontal_cars) and env.phase == wgreen:
            change = True

        if (horizontal_cars > factor * vertical_cars) and env.phase == ngreen:
            change = True

        if t % display_freq == 0:
            plotcurrenttrafic()

        if use_gui:
            time.sleep(0.3)

        next_s, r = env.step(change=change)
        reward += r

    end_sumo()

    return reward