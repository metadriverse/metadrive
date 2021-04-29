import copy
import logging
# from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle
from typing import Dict

from gym.spaces import Box


class AgentManager:
    """
    This class maintain the relationship between active agents in the environment with the underlying instance
    of objects.

    Note:
    agent name: Agent name that exists in the environment, like agent0, agent1, ....
    object name: The unique name for each object, typically be random string.
    """
    INITIALIZED = False  # when vehicles instances are created, it will be set to True
    HELL_POSITION = (-999, -999, -999)  # a place to store pending vehicles

    def __init__(self, init_observations, never_allow_respawn, debug=False, delay_done=0):
        # when new agent joins in the game, we only change this two maps.
        self.__agent_to_object = {}
        self.__object_to_agent = {}

        # BaseVehicles which can be controlled by policies when env.step() called
        self.__active_objects = {}

        # BaseVehicles which can be respawned
        self.__pending_objects = {}
        self.__dying_objects = {}

        # Dict[object_id: value], init for **only** once after spawning vehicle
        self.observations = {}
        self.observation_spaces = {}
        self.action_spaces = {}

        self.next_agent_count = 0
        self._allow_respawn = True if not never_allow_respawn else False
        self.never_allow_respawn = never_allow_respawn
        self._debug = debug
        self._delay_done = delay_done

        self.__init_object_to_agent = None
        self.__agents_finished_this_frame = dict()  # for

        # fake init. before creating pg_world and vehicles, it is necessary when all vehicles re-created in runtime
        self.observations = copy.copy(init_observations)  # its value is map<agent_id, obs> before init() is called
        self.__init_observations = init_observations  # map <agent_id, observation>
        self.__init_observation_spaces = None  # map <agent_id, space>
        self.__init_action_spaces = None  # map <agent_id, space>

        # this map will be override when the env.init() is first called and vehicles are made
        self.__agent_to_object = {k: k for k in self.observations.keys()}  # no target vehicles created, fake init
        self.__object_to_agent = {k: k for k in self.observations.keys()}  # no target vehicles created, fake init

    def init_space(self, init_observation_space, init_action_space):
        """
        For getting env.observation_space/action_space before making vehicles
        """
        assert isinstance(init_action_space, dict)
        assert isinstance(init_observation_space, dict)
        self.__init_observation_spaces = init_observation_space
        self.observation_spaces = copy.copy(init_observation_space)

        self.__init_action_spaces = init_action_space
        self.action_spaces = copy.copy(init_action_space)

    def init(self, init_vehicles: Dict):
        """
        Agent manager is really initialized after the BaseVehicle Instances are created
        """
        vehicles_created = set(init_vehicles.keys())
        vehicles_in_config = set(self.__init_observations.keys())
        assert vehicles_created == vehicles_in_config, "{} not defined in target vehicles config".format(
            vehicles_created.difference(vehicles_in_config)
        )

        self.INITIALIZED = True
        # it is used when reset() is called to reset its original agent_id
        self.__init_object_to_agent = {vehicle.name: agent_id for agent_id, vehicle in init_vehicles.items()}

        self.__agent_to_object = {agent_id: vehicle.name for agent_id, vehicle in init_vehicles.items()}
        self.__object_to_agent = {vehicle.name: agent_id for agent_id, vehicle in init_vehicles.items()}
        self.__active_objects = {v.name: v for v in init_vehicles.values()}
        self.__pending_objects = {}
        self.__dying_objects = {}

        # real init {obj_name: space} map
        self.observations = dict()
        self.observation_spaces = dict()
        self.action_spaces = dict()
        for agent_id, vehicle in init_vehicles.items():
            self.observations[vehicle.name] = self.__init_observations[agent_id]

            obs_space = self.__init_observation_spaces[agent_id]
            self.observation_spaces[vehicle.name] = obs_space
            assert isinstance(obs_space, Box)
            action_space = self.__init_action_spaces[agent_id]
            self.action_spaces[vehicle.name] = action_space
            assert isinstance(action_space, Box)

    def reset(self):
        self.__agents_finished_this_frame = dict()

        # Remove vehicles that are dying.
        for v_name, (v, _) in self.__dying_objects.items():
            self.__pending_objects[v_name] = v
        self.__dying_objects = {}

        # free them in physics world
        vehicles = self.get_vehicle_list()
        for v in vehicles:
            v.set_static(False)
        assert len(vehicles) == len(self.observations) == len(self.observation_spaces) == len(self.action_spaces)
        origin_agent_id_vehicles = {self.__init_object_to_agent[v.name]: v for v in vehicles}

        self.__agent_to_object = {k: v.name for k, v in origin_agent_id_vehicles.items()}
        self.__object_to_agent = {v.name: k for k, v in origin_agent_id_vehicles.items()}
        self.next_agent_count = len(vehicles)
        self.__active_objects = {v.name: v for v in origin_agent_id_vehicles.values()}
        self.__pending_objects = {}
        self._allow_respawn = True if not self.never_allow_respawn else False

    def finish(self, agent_name, ignore_delay_done=False):
        """
        ignore_delay_done: Whether to ignore the delay done. This is not required when the agent success the episode!
        """
        vehicle_name = self.__agent_to_object[agent_name]
        v = self.__active_objects.pop(vehicle_name)
        if (not ignore_delay_done) and (self._delay_done > 0):
            self._put_to_dying_queue(v, vehicle_name)
        else:
            # move to invisible place
            self._put_to_pending_place(v, vehicle_name)
        self.__agents_finished_this_frame[agent_name] = v.name
        self._check()

    def _check(self):
        if self._debug:
            current_keys = sorted(
                list(self.__pending_objects.keys()) + list(self.__active_objects.keys()) +
                list(self.__dying_objects.keys())
            )
            exist_keys = sorted(list(self.__object_to_agent.keys()))
            assert current_keys == exist_keys, "You should confirm_respawn() after request for propose_new_vehicle()!"

    def propose_new_vehicle(self):
        self._check()
        obj_name = list(self.__pending_objects.keys())[0]
        self._check()
        vehicle = self.__pending_objects.pop(obj_name)
        vehicle.prepare_step([0, -1])
        self.observations[obj_name].reset(vehicle)
        new_agent_id = self.next_agent_id()
        dead_vehicle_id = self.__object_to_agent[obj_name]
        vehicle.set_static(False)
        self.__active_objects[vehicle.name] = vehicle
        self.__object_to_agent[vehicle.name] = new_agent_id
        self.__agent_to_object.pop(dead_vehicle_id)
        self.__agent_to_object[new_agent_id] = vehicle.name
        self.next_agent_count += 1
        self._check()
        logging.debug("{} Dead. {} Respawn!".format(dead_vehicle_id, new_agent_id))
        return new_agent_id, vehicle

    def next_agent_id(self):
        return "agent{}".format(self.next_agent_count)

    def set_allow_respawn(self, flag: bool):
        if self.never_allow_respawn:
            self._allow_respawn = False
        else:
            self._allow_respawn = flag

    def prepare_step(self):
        self.__agents_finished_this_frame = dict()
        finished = set()
        for v_name in self.__dying_objects.keys():
            self.__dying_objects[v_name][1] -= 1
            if self.__dying_objects[v_name][1] == 0:  # Countdown goes to 0, it's time to remove the vehicles!
                v = self.__dying_objects[v_name][0]
                self._put_to_pending_place(v, v_name)
                finished.add(v_name)
        for v_name in finished:
            self.__dying_objects.pop(v_name)

    def _translate(self, d):
        return {self.__object_to_agent[k]: v for k, v in d.items()}

    def get_vehicle_list(self):
        return list(self.__active_objects.values()) + list(self.__pending_objects.values()) + \
               [v for (v, _) in self.__dying_objects.values()]

    def get_observations(self):
        ret = {
            old_agent_id: self.observations[v_name]
            for old_agent_id, v_name in self.__agents_finished_this_frame.items()
        }
        for obj_id, observation in self.observations.items():
            if self.is_active_object(obj_id):
                ret[self.object_to_agent(obj_id)] = observation
        return ret

    def get_observation_spaces(self):
        ret = {
            old_agent_id: self.observation_spaces[v_name]
            for old_agent_id, v_name in self.__agents_finished_this_frame.items()
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
        return True if object_name in self.__active_objects.keys() else False

    @property
    def active_objects(self):
        """
        Return Map<agent_id, BaseVehicle>
        """
        return {self.__object_to_agent[k]: v for k, v in self.__active_objects.items()}

    def meta_active_objects(self):
        """
        Return meta-data, a pointer, Caution !
        :return: Map<obj_name, obj>
        """
        return self.__active_objects

    @property
    def pending_objects(self):
        """
        Return Map<agent_id, BaseVehicle>
        """
        ret = {self.__object_to_agent[k]: v for k, v in self.__pending_objects.items()}
        ret.update({self.__object_to_agent[k]: v for k, (v, _) in self.__dying_objects.items()})
        return ret

    def object_to_agent(self, obj_name):
        """
        :param obj_name: BaseVehicle name
        :return: agent id
        """
        # if obj_name not in self.__active_objects.keys() and self.INITIALIZED:
        #     raise ValueError("You can not access a pending Object(BaseVehicle) outside the agent_manager!")
        return self.__object_to_agent[obj_name]

    def agent_to_object(self, agent_id):
        return self.__agent_to_object[agent_id]

    def destroy(self):
        # when new agent joins in the game, we only change this two maps.
        self.__agent_to_object = {}
        self.__object_to_agent = {}

        # BaseVehicles which can be controlled by policies when env.step() called
        self.__active_objects = {}

        # BaseVehicles which can be respawned
        self.__pending_objects = {}
        self.__dying_objects = {}

        # Dict[object_id: value], init for **only** once after spawning vehicle
        self.observations = {}
        self.observation_spaces = {}
        self.action_spaces = {}

        self.next_agent_count = 0

    def _put_to_dying_queue(self, v, vehicle_name):
        v.set_static(True)
        self.__dying_objects[vehicle_name] = [v, self._delay_done]

    def _put_to_pending_place(self, v, vehicle_name):
        v.set_static(True)
        v.set_position(self.HELL_POSITION[:-1], height=self.HELL_POSITION[-1])
        assert vehicle_name not in self.__active_objects
        self.__pending_objects[vehicle_name] = v

    def has_pending_objects(self):
        return (len(self.__pending_objects) > 0)  # or (len(self.__dying_objects) > 0)

    @property
    def allow_respawn(self):
        return self.has_pending_objects() and self._allow_respawn
