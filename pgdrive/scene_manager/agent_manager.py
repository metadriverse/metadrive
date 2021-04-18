from gym.spaces import Box


class AgentManager:
    """
    This class maintain the relationship between active agents in the environment with the underlying instance
    of objects.

    Note:
    agent name: Agent name that exists in the environment, like agent0, agent1, ....
    object name: The unique name for each object, typically be random string.
    """
    def __init__(self, never_allow_respawn, debug=False):
        self.agent_to_object = {}
        self.object_to_agent = {}
        self.pending_object = {}
        self.active_object = {}
        self.next_agent_count = 0
        self.observations = {}
        self.observation_spaces = {}
        self.action_spaces = {}
        self.allow_respawn = True if not never_allow_respawn else False
        self.never_allow_respawn = never_allow_respawn
        self._debug = debug

    def reset(self, vehicles, observation_spaces, action_spaces, observations):
        self.agent_to_object = {k: v.name for k, v in vehicles.items()}
        self.object_to_agent = {v.name: k for k, v in vehicles.items()}
        self.active_object = {v.name: v for v in vehicles.values()}
        self.next_agent_count = len(vehicles)
        self.observations = {vehicles[k].name: v for k, v in observations.items()}
        self.observation_spaces = {vehicles[k].name: v for k, v in observation_spaces.items()}
        for o in observation_spaces.values():
            assert isinstance(o, Box)
        self.action_spaces = {vehicles[k].name: v for k, v in action_spaces.items()}
        for o in action_spaces.values():
            assert isinstance(o, Box)
        self.pending_object = {}
        self.allow_respawn = True if not self.never_allow_respawn else False

    def finish(self, agent_name):
        vehicle_name = self.agent_to_object[agent_name]
        v = self.active_object.pop(vehicle_name)
        v.chassis_np.node().setStatic(True)
        assert vehicle_name not in self.active_object
        self.pending_object[vehicle_name] = v
        self._check()

    def _check(self):
        if self._debug:
            current_keys = sorted(list(self.pending_object.keys()) + list(self.active_object.keys()))
            exist_keys = sorted(list(self.object_to_agent.keys()))
            assert current_keys == exist_keys, "You should confirm_respawn() after request for propose_new_vehicle()!"

    def propose_new_vehicle(self):
        self._check()
        if len(self.pending_object) > 0:
            v_id = list(self.pending_object.keys())[0]
            self._check()
            v = self.pending_object.pop(v_id)
            v.prepare_step([0, -1])
            v.chassis_np.node().setStatic(False)
            return self.allow_respawn, dict(
                vehicle=v,
                observation=self.observations[v_id],
                observation_space=self.observation_spaces[v_id],
                action_space=self.action_spaces[v_id],
                old_name=self.object_to_agent[v_id],
                new_name="agent{}".format(self.next_agent_count)
            )
        return None, None

    def confirm_respawn(self, success: bool, vehicle_info):
        vehicle = vehicle_info['vehicle']
        if success:
            vehicle.set_static(False)
            self.next_agent_count += 1
            self.active_object[vehicle.name] = vehicle
            self.object_to_agent[vehicle.name] = vehicle_info["new_name"]
            self.agent_to_object.pop(vehicle_info["old_name"])
            self.agent_to_object[vehicle_info["new_name"]] = vehicle.name
        else:
            vehicle.set_static(True)
            self.pending_object[vehicle.name] = vehicle
        self._check()

    def set_allow_respawn(self, flag: bool):
        if self.never_allow_respawn:
            self.allow_respawn = False
        else:
            self.allow_respawn = flag

    def _translate(self, d):
        return {self.object_to_agent[k]: v for k, v in d.items()}

    def get_vehicle_list(self):
        return list(self.active_object.values()) + list(self.pending_object.values())

    def get_observations(self):
        return list(self.observations.values())

    def get_observation_spaces(self):
        return list(self.observation_spaces.values())

    def get_action_spaces(self):
        return list(self.action_spaces.values())
