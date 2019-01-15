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

entering_lanes_node0 = [
    'edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2',
    'edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2'
]

entering_lanes = {
    "node0": entering_lanes_node0,
    "C2": ["C3C2_0", "D2C2_0", "C1C2_0", "B2C2_0"]
}

east_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2']  # we should check this
west_lanes = ['edge2-0_0', 'edge2-0_1', 'edge2-0_2']
north_lanes = ['edge3-0_0', 'edge3-0_1', 'edge3-0_2']
south_lanes = ['edge4-0_0', 'edge4-0_1', 'edge4-0_2']

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


def get_phase(node="node0"):
    return phase_decoder[traci.trafficlight.getPhase(node)]


def set_phase(phase, node="node0"):
    traci.trafficlight.setPhase(node, phase_encoder[phase])


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


def start_sumo(traffic, lane_type="uniform", use_gui=False, mutli_agent=False):
    """ Start sumo, 3 config possibles"""
    trafic_files = {
        "alternate": "cross.2phases_rou1_switch_rou0.xml",
        "equal": "cross.2phases_rou01_equal_300s.xml",
        "unequal": "cross.2phases_rou01_unequal_5_300s.xml",
        "equal_big": "cross.2phases_rou01_equal_300s_big.xml",
        "unequal_big": "cross.2phases_rou01_unequal_5_300s_big.xml",
        "my_flow": "my_flow.xml",
        "mutli_agent": "flow.xml"
    }

    if traffic != "multi_agent":
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
            "slow_lane": {"fast_lane": 19.55, "slow_lane": 5.00}
        }
        print("Starting sumo %s"%lane_type)
        slow_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2']
        fast_lanes = ['edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']

        for lane_id in slow_lanes:
            traci.lane.setMaxSpeed(lane_id, speed_dic[lane_type]["slow_lane"])

        for lane_id in fast_lanes:
            traci.lane.setMaxSpeed(lane_id, speed_dic[lane_type]["fast_lane"])
    else:
        file_path = os.path.join("data", "multi_agent", "conf")

        sumoCmd = [sumo_gui_binary_path if use_gui else sumo_binary_path, "-c", file_path]
        traci.start(sumoCmd)

    for i in range(3):
        traci.simulationStep()


def get_state_sumo():
    """ Put here what we need to define the state. For now only the number of vehicles by lines"""
    vehicle_roads = Counter()

    vehicle_id_list = traci.vehicle.getIDList()
    for vehicle_id in vehicle_id_list:
        road_id = traci.vehicle.getRoadID(vehicle_id)
        vehicle_roads[road_id] += 1

    incoming_lanes = {"south": south_lanes, "north": north_lanes, "east": east_lanes, "west": west_lanes}

    count_incoming = {}
    speed_incoming = {}
    for key, value in incoming_lanes.items():
        count_incoming[key] = 0
        speed_incoming[key] = []

    for vehicle_id in traci.vehicle.getIDList():
        current_lane_id = traci.vehicle.getLaneID(vehicle_id)
        for key, value in incoming_lanes.items():
            if current_lane_id in incoming_lanes[key]:
                count_incoming[key] += 1
                speed_incoming[key].append(traci.vehicle.getSpeed(vehicle_id))

    img = get_image_traffic()
    return count_incoming, speed_incoming, img


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
    """
    we keep this for now because we might be using the queue length later
    """
    overall_queue_length = 0
    phase = get_phase()

    for lane in listLanes:
        blocked = is_blocked(lane, phase)
        if not blocked_only or blocked:
            overall_queue_length += traci.lane.getLastStepHaltingNumber(lane)

    return overall_queue_length


def get_vehicles_id_incoming(node="node0"):
    vehicles_incoming = []

    for vehicle_id in traci.vehicle.getIDList():
        if traci.vehicle.getLaneID(vehicle_id) in entering_lanes[node]:
            vehicles_incoming.append(vehicle_id)

    return vehicles_incoming


def vehicle_location_mapper(coordinate, area_length=600, area_width=600):
    """
    Function to plot state of the traffic as an image.
    Not useful but we keep it to create the "image" for the convolutional network
    """
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


def get_image_traffic():
    length_num_grids = int(area_length / grid_width)
    mapOfCars = np.zeros((length_num_grids, length_num_grids))

    vehicle_id_list = traci.vehicle.getIDList()
    for vehicle_id in vehicle_id_list:
        vehicle_position = traci.vehicle.getPosition(vehicle_id)  # (double,double),tuple

        transform_tuple = vehicle_location_mapper(vehicle_position)  # call the function
        mapOfCars[int(transform_tuple[0]), int(transform_tuple[1])] = 1
    return mapOfCars


def end_sumo():
    traci.close()
