import copy
import logging
from typing import Dict

from gym.spaces import Box, Dict

from pgdrive.manager.base_manager import BaseManager


class AgentManager(BaseManager):
    """
    This class maintain the relationship between active agents in the environment with the underlying instance
    of objects.

    Note:
    agent name: Agent name that exists in the environment, like agent0, agent1, ....
    object name: The unique name for each object, typically be random string.
    """
    INITIALIZED = False  # when vehicles instances are created, it will be set to True
    HELL_POSITION = (-999, -999, -999)  # a place to store pending vehicles

    def __init__(self, init_observations, never_allow_respawn, debug=False, delay_done=0, infinite_agents=False):
        """
        The real init is happened in self.init(), in which super().__init__() will be called
        """
        # when new agent joins in the game, we only change this two maps.
        self._agent_to_object = {}
        self._object_to_agent = {}

        # BaseVehicles which can be controlled by policies when env.step() called
        self._active_objects = {}

        # BaseVehicles which can be respawned
        self._pending_objects = {}
        self._dying_objects = {}

        # Dict[object_id: value], init for **only** once after spawning vehicle
        self.observations = {}
        self.observation_spaces = {}
        self.action_spaces = {}

        self.next_agent_count = 0
        self.next_newly_added_agent_count = -1
        self._allow_respawn = True if not never_allow_respawn else False
        self.never_allow_respawn = never_allow_respawn
        self._debug = debug
        self._delay_done = delay_done
        self._infinite_agents = infinite_agents
        self._init_config_dict = None

        self._init_object_to_agent = None
        self._newly_added_object_to_agent = None
        self._agents_finished_this_frame = dict()  # for

        # fake init. before creating engine and vehicles, it is necessary when all vehicles re-created in runtime
        self.observations = copy.copy(init_observations)  # its value is map<agent_id, obs> before init() is called
        self._init_observations = init_observations  # map <agent_id, observation>
        self._init_observation_spaces = None  # map <agent_id, space>
        self._init_action_spaces = None  # map <agent_id, space>

        # this map will be override when the env.init() is first called and vehicles are made
        self._agent_to_object = {k: k for k in self.observations.keys()}  # no target vehicles created, fake init
        self._object_to_agent = {k: k for k in self.observations.keys()}  # no target vehicles created, fake init

    def _get_vehicles(self, config_dict: dict):
        from pgdrive.component.vehicle.base_vehicle import BaseVehicle
        ret = {
            key: super(AgentManager, self).spawn_object(
                BaseVehicle, v_config, am_i_the_special_one=v_config.get("am_i_the_special_one", False)
            )
            for key, v_config in config_dict.items()
        }
        return ret

    def init_space(self, init_observation_space, init_action_space):
        """
        For getting env.observation_space/action_space before making vehicles
        """
        assert isinstance(init_action_space, dict)
        assert isinstance(init_observation_space, dict)
        self._init_observation_spaces = init_observation_space
        self.observation_spaces = copy.copy(init_observation_space)

        self._init_action_spaces = init_action_space
        self.action_spaces = copy.copy(init_action_space)

    def init(self, config_dict: dict):
        """
        Agent manager is really initialized after the BaseVehicle Instances are created
        """
        super(AgentManager, self).__init__()
        self._init_config_dict = config_dict
        init_vehicles = self._get_vehicles(config_dict=config_dict)
        vehicles_created = set(init_vehicles.keys())
        vehicles_in_config = set(self._init_observations.keys())
        assert vehicles_created == vehicles_in_config, "{} not defined in target vehicles config".format(
            vehicles_created.difference(vehicles_in_config)
        )

        self.INITIALIZED = True
        # it is used when reset() is called to reset its original agent_id
        self._init_object_to_agent = {vehicle.name: agent_id for agent_id, vehicle in init_vehicles.items()}
        self._newly_added_object_to_agent = {}

        self._agent_to_object = {agent_id: vehicle.name for agent_id, vehicle in init_vehicles.items()}
        self._object_to_agent = {vehicle.name: agent_id for agent_id, vehicle in init_vehicles.items()}
        self._active_objects = {v.name: v for v in init_vehicles.values()}
        self._pending_objects = {}
        self._dying_objects = {}

        # real init {obj_name: space} map
        self.observations = dict()
        self.observation_spaces = dict()
        self.action_spaces = dict()
        for agent_id, vehicle in init_vehicles.items():
            self.observations[vehicle.name] = self._init_observations[agent_id]

            obs_space = self._init_observation_spaces[agent_id]
            self.observation_spaces[vehicle.name] = obs_space
            if not vehicle.config["offscreen_render"]:
                assert isinstance(obs_space, Box)
            else:
                assert isinstance(obs_space, Dict), "Multi-agent observation should be gym.Dict"
            action_space = self._init_action_spaces[agent_id]
            self.action_spaces[vehicle.name] = action_space
            assert isinstance(action_space, Box)

    def reset(self):
        self._agents_finished_this_frame = dict()

        # Remove vehicles that are dying.
        for v_name, (v, _) in self._dying_objects.items():
            self._pending_objects[v_name] = v
        self._dying_objects = {}

        # free them in physics world
        vehicles = self.get_vehicle_list()
        assert len(vehicles) == len(self.observations) == len(self.observation_spaces) == len(self.action_spaces)

        self._active_objects.clear()
        self._pending_objects.clear()
        self._agent_to_object.clear()
        self._object_to_agent.clear()
        for v in vehicles:
            if v.name in self._init_object_to_agent:
                self._active_objects[v.name] = v
                self._agent_to_object[self._init_object_to_agent[v.name]] = v.name
                self._object_to_agent[v.name] = self._init_object_to_agent[v.name]
                v.set_static(False)
            elif v.name in self._newly_added_object_to_agent:
                agent_name = self._newly_added_object_to_agent[v.name]
                self._pending_objects[v.name] = v
                self._agent_to_object[agent_name] = v.name
                self._object_to_agent[v.name] = agent_name
                v.set_static(True)
            else:
                raise ValueError()

        self.next_agent_count = len(vehicles)
        # Note: We don't reset next_newly_added_agent_count here! Since it is always counting!

        self._allow_respawn = True if not self.never_allow_respawn else False

    def finish(self, agent_name, ignore_delay_done=False):
        """
        ignore_delay_done: Whether to ignore the delay done. This is not required when the agent success the episode!
        """
        vehicle_name = self._agent_to_object[agent_name]
        v = self._active_objects.pop(vehicle_name)
        if (not ignore_delay_done) and (self._delay_done > 0):
            self._put_to_dying_queue(v, vehicle_name)
        else:
            # move to invisible place
            self._put_to_pending_place(v, vehicle_name)
        self._agents_finished_this_frame[agent_name] = v.name
        self._check()

    def _check(self):
        if self._debug:
            current_keys = sorted(
                list(self._pending_objects.keys()) + list(self._active_objects.keys()) +
                list(self._dying_objects.keys())
            )
            exist_keys = sorted(list(self._object_to_agent.keys()))
            assert current_keys == exist_keys, "You should confirm_respawn() after request for propose_new_vehicle()!"

    def propose_new_vehicle(self):
        if self._infinite_agents and len(self._pending_objects) == 0:  # Create a new vehicle.
            self._add_new_vehicle()
        assert self._pending_objects, "No available agent exists!"
        self._check()
        obj_name = next(iter(self._pending_objects.keys()))
        self._check()
        vehicle = self._pending_objects.pop(obj_name)
        vehicle.before_step([0, -1])
        self.observations[obj_name].reset(vehicle)
        new_agent_id = self.next_agent_id()
        dead_vehicle_id = self._object_to_agent[obj_name]
        vehicle.set_static(False)
        self._active_objects[vehicle.name] = vehicle
        self._object_to_agent[vehicle.name] = new_agent_id
        if dead_vehicle_id in self._agent_to_object:  # dead_vehicle_id might not in self._agent_to_object
            self._agent_to_object.pop(dead_vehicle_id)
        self._agent_to_object[new_agent_id] = vehicle.name
        self.next_agent_count += 1
        self._check()
        logging.debug("{} Dead. {} Respawn!".format(dead_vehicle_id, new_agent_id))
        return new_agent_id, vehicle

    def _add_new_vehicle(self):
        agent_name = "newly_added{}".format(self.next_newly_added_agent_count)
        next_config = self._init_config_dict["agent{}".format(
            ((-self.next_newly_added_agent_count - 1) % len(self._init_object_to_agent))
        )]
        new_v = self._get_vehicles({agent_name: next_config})[agent_name]
        new_v_name = new_v.name
        self._newly_added_object_to_agent[new_v_name] = agent_name
        self._agent_to_object[agent_name] = new_v_name
        self._object_to_agent[new_v_name] = agent_name
        self._pending_objects[new_v_name] = new_v
        self.observations[new_v_name] = self._init_observations["agent0"]
        self.observation_spaces[new_v_name] = self._init_observation_spaces["agent0"]
        self.action_spaces[new_v_name] = self._init_action_spaces["agent0"]
        self.next_newly_added_agent_count += 1

    def next_agent_id(self):
        return "agent{}".format(self.next_agent_count)

    def set_allow_respawn(self, flag: bool):
        if self.never_allow_respawn:
            self._allow_respawn = False
        else:
            self._allow_respawn = flag

    def before_step(self):
        self._agents_finished_this_frame = dict()
        finished = set()
        for v_name in self._dying_objects.keys():
            self._dying_objects[v_name][1] -= 1
            if self._dying_objects[v_name][1] == 0:  # Countdown goes to 0, it's time to remove the vehicles!
                v = self._dying_objects[v_name][0]
                self._put_to_pending_place(v, v_name)
                finished.add(v_name)
        for v_name in finished:
            self._dying_objects.pop(v_name)

    def _translate(self, d):
        return {self._object_to_agent[k]: v for k, v in d.items()}

    def get_vehicle_list(self):
        return list(self._active_objects.values()) + list(self._pending_objects.values()) + \
               [v for (v, _) in self._dying_objects.values()]

    def get_observations(self):
        ret = {
            old_agent_id: self.observations[v_name]
            for old_agent_id, v_name in self._agents_finished_this_frame.items()
        }
        for obj_id, observation in self.observations.items():
            if self.is_active_object(obj_id):
                ret[self.object_to_agent(obj_id)] = observation
        return ret

    def get_observation_spaces(self):
        ret = {
            old_agent_id: self.observation_spaces[v_name]
            for old_agent_id, v_name in self._agents_finished_this_frame.items()
        }
        for obj_id, space in self.observation_spaces.items():
            if self.is_active_object(obj_id):
                ret[self.object_to_agent(obj_id)] = space
        return ret

    def get_action_spaces(self):
        ret = dict()
        for obj_id, space in self.action_spaces.items():
            if self.is_active_object(obj_id):
                ret[self.object_to_agent(obj_id)] = space
        return ret

    def is_active_object(self, object_name):
        if not self.INITIALIZED:
            return True
        return True if object_name in self._active_objects.keys() else False

    @property
    def active_agents(self):
        """
        Return Map<agent_id, BaseVehicle>
        """
        return {self._object_to_agent[k]: v for k, v in self._active_objects.items()}

    @property
    def active_objects(self):
        """
        Return meta-data, a pointer, Caution !
        :return: Map<obj_name, obj>
        """
        return self._active_objects

    @property
    def pending_objects(self):
        """
        Return Map<agent_id, BaseVehicle>
        """
        ret = {self._object_to_agent[k]: v for k, v in self._pending_objects.items()}
        ret.update({self._object_to_agent[k]: v for k, (v, _) in self._dying_objects.items()})
        return ret

    def get_agent(self, agent_name):
        object_name = self.agent_to_object(agent_name)
        return self.get_object(object_name)

    def get_object(self, object_name):
        if object_name in self._active_objects:
            return self._active_objects[object_name]
        elif object_name in self._pending_objects:
            return self._pending_objects[object_name]
        elif object_name in self._dying_objects:
            return self._dying_objects[object_name]
        else:
            raise ValueError("Object {} not found!".format(object_name))

    def object_to_agent(self, obj_name):
        """
        :param obj_name: BaseVehicle name
        :return: agent id
        """
        # if obj_name not in self._active_objects.keys() and self.INITIALIZED:
        #     raise ValueError("You can not access a pending Object(BaseVehicle) outside the agent_manager!")
        return self._object_to_agent[obj_name]

    def agent_to_object(self, agent_id):
        return self._agent_to_object[agent_id]

    def destroy(self):
        # when new agent joins in the game, we only change this two maps.
        if self.INITIALIZED:
            super(AgentManager, self).destroy()
        self._agent_to_object = {}
        self._object_to_agent = {}

        # BaseVehicles which can be controlled by policies when env.step() called
        self._active_objects = {}

        # BaseVehicles which can be respawned
        self._pending_objects = {}
        self._dying_objects = {}

        # Dict[object_id: value], init for **only** once after spawning vehicle
        self.observations = {}
        self.observation_spaces = {}
        self.action_spaces = {}

        self.next_agent_count = 0

    def _put_to_dying_queue(self, v, vehicle_name):
        v.set_static(True)
        self._dying_objects[vehicle_name] = [v, self._delay_done]

    def _put_to_pending_place(self, v, vehicle_name):
        v.set_static(True)
        v.set_position(self.HELL_POSITION[:-1], height=self.HELL_POSITION[-1])
        assert vehicle_name not in self._active_objects
        self._pending_objects[vehicle_name] = v

    def has_pending_objects(self):
        # If infinite_agents, then we pretend always has available agents to be re-spawn.
        return (len(self._pending_objects) > 0) or self._infinite_agents

    @property
    def allow_respawn(self):
        return self.has_pending_objects() and self._allow_respawn

    def for_each_active_agents(self, func, *args, **kwargs):
        """
        This func is a function that take each vehicle as the first argument and *arg and **kwargs as others.
        """
        assert len(self.active_agents) > 0, "Not enough vehicles exist!"
        ret = dict()
        for k, v in self.active_agents.items():
            ret[k] = func(v, *args, **kwargs)
        return ret
