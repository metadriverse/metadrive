from gym.spaces import Box


class TargetVehicleManager:
    """
    This class manages all

    Note:
    vehicle name: unique name for each vehicle instance, random string.
    agent name: agent name that exists in the environment, like agent0, agent1, ....
    """
    def __init__(self, ):
        self.agent_to_vehicle = {}
        self.vehicle_to_agent = {}
        self.pending_vehicles = {}
        self.active_vehicles = {}
        self.next_agent_count = 0
        self.observations = {}
        self.observation_spaces = {}
        self.action_spaces = {}
        self.allow_respawn = True

    def reset(self, vehicles, observation_spaces, action_spaces, observations):
        self.agent_to_vehicle = {k: v.name for k, v in vehicles.items()}
        self.vehicle_to_agent = {v.name: k for k, v in vehicles.items()}
        self.active_vehicles = {v.name: v for v in vehicles.values()}
        self.next_agent_count = len(vehicles)
        self.observations = {vehicles[k].name: v for k, v in observations.items()}
        self.observation_spaces = {vehicles[k].name: v for k, v in observation_spaces.items()}
        for o in observation_spaces.values():
            assert isinstance(o, Box)
        self.action_spaces = {vehicles[k].name: v for k, v in action_spaces.items()}
        for o in action_spaces.values():
            assert isinstance(o, Box)
        self.pending_vehicles = {}
        self.allow_respawn = True

    def finish(self, agent_name):
        vehicle_name = self.agent_to_vehicle[agent_name]
        v = self.active_vehicles.pop(vehicle_name)
        assert vehicle_name not in self.active_vehicles
        self.pending_vehicles[vehicle_name] = v
        self._check()

    def _check(self):
        current_keys = sorted(list(self.pending_vehicles.keys()) + list(self.active_vehicles.keys()))
        exist_keys = sorted(list(self.vehicle_to_agent.keys()))
        assert current_keys == exist_keys, "You should confirm_respawn() after request for propose_new_vehicle()!"

    def propose_new_vehicle(self):
        self._check()
        if len(self.pending_vehicles) > 0:
            v_id = list(self.pending_vehicles.keys())[0]
            self._check()
            v = self.pending_vehicles.pop(v_id)
            return self.allow_respawn, dict(
                vehicle=v,
                observation=self.observations[v_id],
                observation_space=self.observation_spaces[v_id],
                action_space=self.action_spaces[v_id],
                old_name=self.vehicle_to_agent[v_id],
                new_name="agent{}".format(self.next_agent_count)
            )
        return None, None

    def confirm_respawn(self, success: bool, vehicle_info):
        vehicle = vehicle_info['vehicle']
        if success:
            self.next_agent_count += 1
            self.active_vehicles[vehicle.name] = vehicle
            self.vehicle_to_agent[vehicle.name] = vehicle_info["new_name"]
            self.agent_to_vehicle.pop(vehicle_info["old_name"])
            self.agent_to_vehicle[vehicle_info["new_name"]] = vehicle.name
        else:
            self.pending_vehicles[vehicle.name] = vehicle
        self._check()

    def set_allow_respawn(self, flag: bool):
        self.allow_respawn = flag

    def _translate(self, d):
        return {self.vehicle_to_agent[k]: v for k, v in d.items()}

    def get_vehicle_list(self):
        return list(self.active_vehicles.values()) + list(self.pending_vehicles.values())

    def get_observations(self):
        return list(self.observations.values())

    def get_observation_spaces(self):
        return list(self.observation_spaces.values())

    def get_action_spaces(self):
        return list(self.action_spaces.values())