from metadrive.constants import DEFAULT_AGENT
from metadrive.engine.logger import get_logger
from metadrive.manager.base_manager import BaseAgentManager
from metadrive.policy.AI_protect_policy import AIProtectPolicy
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy, TakeoverPolicy, TakeoverPolicyWithoutBrake
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy

logger = get_logger()


class VehicleAgentManager(BaseAgentManager):
    """
    This class maintain the relationship between active agents in the environment with the underlying instance
    of objects.

    Note:
    agent name: Agent name that exists in the environment, like default_agent, agent0, agent1, ....
    object name: The unique name for each object, typically be random string.
    """
    INITIALIZED = False  # when vehicles instances are created, it will be set to True

    def __init__(self, init_observations):
        """
        The real init is happened in self.init(), in which super().__init__() will be called
        """
        super(VehicleAgentManager, self).__init__(init_observations)
        # For multi-agent env, None values is updated in init()
        self._allow_respawn = None
        self._delay_done = None
        self._infinite_agents = None

        self._dying_objects = {}  # BaseVehicles which will be recycled after the delay_done time
        self._agents_finished_this_frame = dict()  # for observation space
        self.next_agent_count = 0

    def _create_agents(self, config_dict: dict):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        ret = {}
        for agent_id, v_config in config_dict.items():
            v_type = random_vehicle_type(self.np_random) if self.engine.global_config["random_agent_model"] else \
                vehicle_type[v_config["vehicle_model"] if v_config.get("vehicle_model", False) else "default"]

            obj_name = agent_id if self.engine.global_config["force_reuse_object_name"] else None
            obj = self.spawn_object(v_type, vehicle_config=v_config, name=obj_name)
            ret[agent_id] = obj
            policy_cls = self.agent_policy
            args = [obj, self.generate_seed()]
            if policy_cls == TrajectoryIDMPolicy or issubclass(policy_cls, TrajectoryIDMPolicy):
                args.append(self.engine.map_manager.current_sdc_route)
            self.add_policy(obj.id, policy_cls, *args)
        return ret

    @property
    def agent_policy(self):
        """Get the agent policy class

        Make sure you access the global config via get_global_config() instead of self.engine.global_config

        Returns:
            Agent Policy class
        """
        from metadrive.engine.engine_utils import get_global_config
        # Takeover policy shares the control between RL agent (whose action is input via env.step)
        # and external control device (whose action is input via controller).
        if get_global_config()["agent_policy"] in [TakeoverPolicy, TakeoverPolicyWithoutBrake]:
            return get_global_config()["agent_policy"]
        if get_global_config()["manual_control"]:
            if get_global_config().get("use_AI_protector", False):
                policy = AIProtectPolicy
            else:
                policy = ManualControlPolicy
        else:
            policy = get_global_config()["agent_policy"]
        return policy

    def before_reset(self):
        if not self.INITIALIZED:
            super(BaseAgentManager, self).__init__()
            self.INITIALIZED = True

        self.episode_created_agents = None

        if not self.engine.replay_episode:
            for v in self.dying_agents.values():
                self._remove_vehicle(v)

        for v in list(self._active_objects.values()) + [v for (v, _) in self._dying_objects.values()]:
            if hasattr(v, "before_reset"):
                v.before_reset()

        super(VehicleAgentManager, self).before_reset()

    def reset(self):
        """
        Agent manager is really initialized after the BaseVehicle Instances are created
        """
        self.random_spawn_lane_in_single_agent()
        config = self.engine.global_config
        self._delay_done = config["delay_done"]
        self._infinite_agents = config["num_agents"] == -1
        self._allow_respawn = config["allow_respawn"]
        super(VehicleAgentManager, self).reset()

    def after_reset(self):
        super(VehicleAgentManager, self).after_reset()
        self._dying_objects = {}
        self._agents_finished_this_frame = dict()
        self.next_agent_count = len(self.episode_created_agents)

    def random_spawn_lane_in_single_agent(self):
        if not self.engine.global_config["is_multi_agent"] and \
                self.engine.global_config.get("random_spawn_lane_index", False) and self.engine.current_map is not None:
            spawn_road_start = self.engine.global_config["agent_configs"][DEFAULT_AGENT]["spawn_lane_index"][0]
            spawn_road_end = self.engine.global_config["agent_configs"][DEFAULT_AGENT]["spawn_lane_index"][1]
            index = self.np_random.randint(self.engine.current_map.config["lane_num"])
            self.engine.global_config["agent_configs"][DEFAULT_AGENT]["spawn_lane_index"] = (
                spawn_road_start, spawn_road_end, index
            )

    def _finish(self, agent_name, ignore_delay_done=False):
        """
        ignore_delay_done: Whether to ignore the delay done. This is not required when the agent success the episode!
        """
        if not self.engine.replay_episode:
            vehicle_name = self._agent_to_object[agent_name]
            v = self._active_objects.pop(vehicle_name)
            if (not ignore_delay_done) and (self._delay_done > 0):
                self._put_to_dying_queue(v)
            else:
                # move to invisible place
                self._remove_vehicle(v)
            self._agents_finished_this_frame[agent_name] = v.name
            self._check()

    def _check(self):
        if self._debug:
            current_keys = sorted(list(self._active_objects.keys()) + list(self._dying_objects.keys()))
            exist_keys = sorted(list(self._object_to_agent.keys()))
            assert current_keys == exist_keys, "You should confirm_respawn() after request for propose_new_vehicle()!"

    def propose_new_vehicle(self):
        # Create a new vehicle.
        agent_name = self._next_agent_id()
        next_config = self.engine.global_config["agent_configs"]["agent0"]
        vehicle = self._create_agents({agent_name: next_config})[agent_name]
        new_v_name = vehicle.name
        self._agent_to_object[agent_name] = new_v_name
        self._object_to_agent[new_v_name] = agent_name
        # TODO: this may cause error? Sharing observation
        # logger.warning("Test MARL new agent observation to avoid bug!")
        self.observations[new_v_name] = self._init_observations["agent0"]
        self.observations[new_v_name].reset(vehicle)
        self.observation_spaces[new_v_name] = self._init_observation_spaces["agent0"]
        self.action_spaces[new_v_name] = self._init_action_spaces["agent0"]
        self._active_objects[vehicle.name] = vehicle
        self._check()
        step_info = vehicle.before_step([0, 0])
        vehicle.set_static(False)
        return agent_name, vehicle, step_info

    def _next_agent_id(self):
        ret = "agent{}".format(self.next_agent_count)
        self.next_agent_count += 1
        return ret

    def set_allow_respawn(self, flag: bool):
        self._allow_respawn = flag

    def try_actuate_agent(self, step_infos, stage="before_step"):
        """
        Some policies should make decision before physics world actuation, in particular, those need decision-making
        But other policies like ReplayPolicy should be called in after_step, as they already know the final state and
        exempt the requirement for rolling out the dynamic system to get it.
        """
        assert stage == "before_step" or stage == "after_step"
        for agent_id in self.active_agents.keys():
            policy = self.get_policy(self._agent_to_object[agent_id])
            is_replay = isinstance(policy, ReplayTrafficParticipantPolicy)
            assert policy is not None, "No policy is set for agent {}".format(agent_id)
            if is_replay:
                if stage == "after_step":
                    policy.act(agent_id)
                    step_infos[agent_id] = policy.get_action_info()
                else:
                    step_infos[agent_id] = self.get_agent(agent_id).before_step([0, 0])
            else:
                if stage == "before_step":
                    action = policy.act(agent_id)
                    step_infos[agent_id] = policy.get_action_info()
                    step_infos[agent_id].update(self.get_agent(agent_id).before_step(action))

        return step_infos

    def before_step(self):
        # not in replay mode
        step_infos = super(VehicleAgentManager, self).before_step()
        self._agents_finished_this_frame = dict()
        finished = set()
        for v_name in self._dying_objects.keys():
            self._dying_objects[v_name][1] -= 1
            if self._dying_objects[v_name][1] <= 0:  # Countdown goes to 0, it's time to remove the vehicles!
                v = self._dying_objects[v_name][0]
                self._remove_vehicle(v)
                finished.add(v_name)
        for v_name in finished:
            self._dying_objects.pop(v_name)
        return step_infos

    def get_observations(self):
        if hasattr(self, "engine") and self.engine.replay_episode:
            return self.engine.replay_manager.get_replay_agent_observations()
        else:
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

    @property
    def dying_agents(self):
        assert not self.engine.replay_episode
        return {self._object_to_agent[k]: v for k, (v, _) in self._dying_objects.items()}

    @property
    def just_terminated_agents(self):
        assert not self.engine.replay_episode
        ret = {}
        for agent_name, v_name in self._agents_finished_this_frame.items():
            v = self.get_agent(v_name, raise_error=False)
            ret[agent_name] = v
        return ret

    def get_agent(self, agent_name, raise_error=True):
        try:
            object_name = self.agent_to_object(agent_name)
        except KeyError:
            if raise_error:
                raise ValueError("Object {} not found!".format(object_name))
            else:
                return None
        if object_name in self._active_objects:
            return self._active_objects[object_name]
        elif object_name in self._dying_objects:
            return self._dying_objects[object_name][0]
        else:
            if raise_error:
                raise ValueError("Object {} not found!".format(object_name))
            else:
                return None

    def destroy(self):
        # when new agent joins in the game, we only change this two maps.
        super(VehicleAgentManager, self).destroy()
        self._dying_objects = {}
        self.next_agent_count = 0
        self._agents_finished_this_frame = dict()

    def _put_to_dying_queue(self, v, ignore_delay_done=False):
        vehicle_name = v.name
        v.set_static(True)
        self._dying_objects[vehicle_name] = [v, 0 if ignore_delay_done else self._delay_done]

    def _remove_vehicle(self, vehicle):
        vehicle_name = vehicle.name
        assert vehicle_name not in self._active_objects
        self.clear_objects([vehicle_name])
        self._agent_to_object.pop(self._object_to_agent[vehicle_name])
        self._object_to_agent.pop(vehicle_name)

    @property
    def allow_respawn(self):
        if not self._allow_respawn:
            return False
        if len(self._active_objects) + len(self._dying_objects) < self.engine.global_config["num_agents"] \
                or self._infinite_agents:
            return True
        else:
            return False
