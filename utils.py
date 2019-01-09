import sys
import os

sys.path.append(
    os.path.join(os.environ["SUMO_HOME"], "tools")
)
import traci
import traci.constants as tc
from collections import Counter
import xml.etree.ElementTree as ET
import numpy as np
import math
import matplotlib.pyplot as plt
from env_name import env_name

if env_name == "raph":
    sumo_binary_path = os.path.join('c:', os.sep, "Program Files (x86)", "Eclipse", "Sumo", "bin", "sumo")
    sumo_gui_binary_path = os.path.join('c:', os.sep, "Program Files (x86)", "Eclipse", "Sumo", "bin", "sumo-gui")

else:
    sumo_binary_path = '/usr/local/bin/sumo'
    sumo_gui_binary_path = '/usr/local/bin/sumo-gui'

# vertical lanes start with edge 4 and edge 3
area_length = 600
grid_width = 4

entering_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2',
                  'edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']

# assign sumo code to each phase
wgreen = "WGREEN"
ngreen = "NGREEN"
yellow_wn = "YELLOW_WN"
yellow_nw = "YELLOW_NW"

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


def set_traffic_file(
        sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name
):
    """ Set the traffic file in the sumo config"""
    # update sumocfg
    sumo_cfg = ET.parse(sumo_config_file_tmp_name)
    config_node = sumo_cfg.getroot()
    input_node = config_node.find("input")

    for route_files in input_node.findall("route-files"):
        input_node.remove(route_files)
    input_node.append(
        ET.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)})
    )

    sumo_cfg.write(sumo_config_file_output_name)


def start_sumo(traffic, lane_type="uniform", use_gui=False):
    """ Start sumo, 3 config possibles"""
    trafic_files = {
        "alternate": "cross.2phases_rou1_switch_rou0.xml",
        "equal": "cross.2phases_rou01_equal_300s.xml",
        "unequal": "cross.2phases_rou01_unequal_5_300s.xml",
        "equal_big": "cross.2phases_rou01_equal_300s_big.xml",
        "unequal_big": "cross.2phases_rou01_unequal_5_300s_big.xml",
        "my_flow": "my_flow.xml"
    }

    set_traffic_file(
        os.path.join("data", "one_run", "cross.sumocfg"),
        os.path.join("data", "one_run", "cross.temp_config.sumocfg"),
        [trafic_files[traffic]]
    )

    file_path = os.path.join("data", "one_run", "cross.temp_config.sumocfg")

    sumoCmd = [sumo_gui_binary_path if use_gui else sumo_binary_path, "-c", file_path]
    traci.start(sumoCmd)

    speed_dic = {
        "uniform": {"fast_lane": 19.44, "slow_lane": 19.44},
        "slow_lane": {"fast_lane": 19.44, "slow_lane": 10.00}
    }

    slow_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2']
    fast_lanes = ['edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']

    for lane_id in slow_lanes:
        traci.lane.setMaxSpeed(lane_id, speed_dic[lane_type]["slow_lane"])

    for lane_id in fast_lanes:
        traci.lane.setMaxSpeed(lane_id, speed_dic[lane_type]["fast_lane"])

    for i in range(3):
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
    phase = get_phase()

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

    for lane in entering_lanes:
        vehicle_id_entering.extend(traci.lane.getLastStepVehicleIDs(lane))

    return vehicle_id_entering


def get_vehicles_id_incoming():
    vehicles_incoming = []

    for vehicle_id in traci.vehicle.getIDList():
        if traci.vehicle.getLaneID(vehicle_id) in entering_lanes:
            vehicles_incoming.append(vehicle_id)

    return vehicles_incoming


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
