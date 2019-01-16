import traci
from utils import get_phase, set_phase, yellow_wn, yellow_nw, wgreen, ngreen, get_vehicles_id_incoming


class SumoEnv:

    def __init__(self, multi_agent=False):
        self.vehicle_dict = {}
        self.n_actions = 2
        self.all_vehicles = {}
        self.arrived_vehicles = {}
        self.multi_agent = multi_agent

    def get_reward(self, node="node0"):

        if self.multi_agent:
            node = "C2"

        incoming_vehicles = get_vehicles_id_incoming(node)

        reward = 0
        for id_ in incoming_vehicles:
            reward += traci.vehicle.getSpeed(id_) - traci.lane.getMaxSpeed(traci.vehicle.getLaneID(id_))

        return reward / 1000.

    def update_vehicles_dic(self):

        running_vehicles = traci.vehicle.getIDList()
        arrived_vehicles = traci.simulation.getArrivedIDList()

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

    def step(self, change=True, node="node0"):

        if self.multi_agent:
            node = "C2"

        if change and get_phase(node) in [yellow_wn, yellow_nw]:
            print("cant change phase since the light is already yellow")

        if change:
            if get_phase(node) == wgreen:
                set_phase(yellow_wn, node)

            elif get_phase(node) == ngreen:
                set_phase(yellow_nw, node)

        traci.simulationStep()

        self.update_vehicles_dic()

    def step_multi_agents(self, actions):

        for node, change in actions.items():

            if change and get_phase(node) in [yellow_wn, yellow_nw]:
                print("cant change phase since the light is already yellow")

            if change:
                if get_phase(node) == wgreen:
                    set_phase(yellow_wn, node)

                elif get_phase(node) == ngreen:
                    set_phase(yellow_nw, node)

        traci.simulationStep()

        self.update_vehicles_dic()
