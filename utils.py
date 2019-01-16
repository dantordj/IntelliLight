import sys
import os

sys.path.append(
    os.path.join(os.environ["SUMO_HOME"], "tools")
)
import traci
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


east_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2']  # we should check this
west_lanes = ['edge2-0_0', 'edge2-0_1', 'edge2-0_2']
north_lanes = ['edge3-0_0', 'edge3-0_1', 'edge3-0_2']
south_lanes = ['edge4-0_0', 'edge4-0_1', 'edge4-0_2']

incoming_lanes_node0 = {"south": south_lanes, "north": north_lanes, "east": east_lanes, "west": west_lanes}
incoming_lanes_dic = {"node0": incoming_lanes_node0}
upstream_lanes_dic = {}

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
letter_to_int = {}
for u in range(len(alphabet)):
    letter_to_int[alphabet[u]] = u


def build_incoming_lanes(node):
    letter = node[0]
    number = int(node[1])

    letter_int = letter_to_int[letter]

    directions = ["west", "south", "east", "north"]

    ans = {}
    for u, (i, j) in enumerate([(-1, 0), (0, -1), (1, 0), (0, 1)]):
        current_number = number + i
        current_letter_int = letter_int + j
        current_letter = alphabet[current_letter_int]

        ans[directions[u]] = current_letter + str(current_number) + node + "_0"

    return ans


def build_upstream_lanes(node):
    letter = node[0]
    number = int(node[1])

    letter_int = letter_to_int[letter]

    directions = ["west", "south", "east", "north"]

    ans = {}
    for u, (i, j) in enumerate([(-1, 0), (0, -1), (1, 0), (0, 1)]):
        current_number = number + i
        current_letter_int = letter_int + j
        current_letter = alphabet[current_letter_int]

        upstream_number = number + i * 2
        upstream_letter_int = letter_int + j * 2
        upstream_letter = alphabet[upstream_letter_int]

        ans[directions[u]] = upstream_letter + str(upstream_number)
        ans[directions[u]] += current_letter + str(current_number) + "_0"

    return ans


def get_incoming_lanes(node):
    try:
        ans = incoming_lanes_dic[node]
    except:
        ans = build_incoming_lanes(node)
        incoming_lanes_dic[node] = ans
    return ans


def get_upstream_lanes(node):
    try:
        ans = upstream_lanes_dic[node]
    except:
        ans = build_upstream_lanes(node)
        upstream_lanes_dic[node] = ans
    return ans


def get_incoming_lanes_list(node):
    return list(get_incoming_lanes(node).values())


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

    if traffic == "multi_agent":
        file_path = os.path.join("data", "multi_agent", "conf")

        sumoCmd = [sumo_gui_binary_path if use_gui else sumo_binary_path, "-c", file_path]
        traci.start(sumoCmd)

    elif traffic == "four_agents":
        file_path = os.path.join("data", "four_agents", "conf")

        sumoCmd = [sumo_gui_binary_path if use_gui else sumo_binary_path, "-c", file_path]
        traci.start(sumoCmd)

    else:
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
        print("Starting sumo %s" % lane_type)
        slow_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2']
        fast_lanes = ['edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']

        for lane_id in slow_lanes:
            traci.lane.setMaxSpeed(lane_id, speed_dic[lane_type]["slow_lane"])

        for lane_id in fast_lanes:
            traci.lane.setMaxSpeed(lane_id, speed_dic[lane_type]["fast_lane"])

    for i in range(3):
        traci.simulationStep()


def get_state_sumo(node="node0", get_img=False):
    """ Put here what we need to define the state. For now only the number of vehicles by lines"""

    ans = {}
    incoming_lanes = get_incoming_lanes(node)

    if node != "node0":
        upstream_lanes = get_upstream_lanes(node)

    count_incoming = {}
    speed_incoming = {}
    count_upstream = {}

    for key, value in incoming_lanes.items():
        count_incoming[key] = 0
        count_upstream[key] = 0
        speed_incoming[key] = []

    for vehicle_id in traci.vehicle.getIDList():
        current_lane_id = traci.vehicle.getLaneID(vehicle_id)
        for key, value in incoming_lanes.items():
            if current_lane_id in incoming_lanes[key]:
                count_incoming[key] += 1
                speed_incoming[key].append(traci.vehicle.getSpeed(vehicle_id))

        if node != "node0":
            for key, value in upstream_lanes.items():
                if current_lane_id in upstream_lanes[key]:
                    count_upstream[key] += 1

    ans["count_incoming"] = count_incoming
    ans["speed_incoming"] = speed_incoming

    if get_img:
        ans["img"] = get_image_traffic()
    if node != "node0":
        ans["count_upstream"] = count_upstream

    return ans


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
        if traci.vehicle.getLaneID(vehicle_id) in get_incoming_lanes_list(node):
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
