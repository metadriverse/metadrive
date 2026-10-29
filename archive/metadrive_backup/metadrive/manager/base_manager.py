import copy
from metadrive.engine.engine_utils import get_global_config
from metadrive.constants import DEFAULT_AGENT

from gymnasium.spaces import Space

from metadrive.base_class.randomizable import Randomizable


class BaseManager(Randomizable):
    """
    Managers should be created and registered after launching BaseEngine
    """
    PRIORITY = 10  # the engine will call managers according to the priority

    def __init__(self):
        from metadrive.engine.engine_utils import get_engine, engine_initialized
        assert engine_initialized(), "You should not create manager before the initialization of BaseEngine"
        # self.engine = get_engine()
        Randomizable.__init__(self, get_engine().global_random_seed)
        self.spawned_objects = {}

    @property
    def episode_step(self):
        """
        Return how many steps are taken from env.reset() to current step
        Returns:

        """
        return self.engine.episode_step

    def before_step(self, *args, **kwargs) -> dict:
        """
        Usually used to set actions for all elements with their policies
        """
        return dict()

    def step(self, *args, **kwargs):
        pass

    def after_step(self, *args, **kwargs) -> dict:
        """
        Update state for this manager after system advancing dt
        """
        return dict()

    def before_reset(self):
        """
        Update episode level config to this manager and clean element or detach element
        """
        self.clear_objects([object_id for object_id in self.spawned_objects.keys()])
        self.spawned_objects = {}

    def reset(self):
        """
        Generate objects according to some pre-defined rules
        """
        pass

    def after_reset(self):
        """
        Usually used to record information after all managers called reset(),
        Since reset() of managers may influence each other
        """
        pass

    def destroy(self):
        """
        Destroy manager
        """
        # self.engine = None
        super(BaseManager, self).destroy()
        self.clear_objects(list(self.spawned_objects.keys()), force_destroy=True)
        self.spawned_objects = None

    def spawn_object(self, object_class, **kwargs):
        """
        Spawn one objects
        """
        object = self.engine.spawn_object(object_class, **kwargs)
        self.spawned_objects[object.id] = object
        return object

    def clear_objects(self, *args, **kwargs):
        """
        Same as the function in engine, clear objects, Return exclude object ids

        filter: A list of object ids or a function returning a list of object id
        """
        exclude_objects = self.engine.clear_objects(*args, **kwargs)
        for obj in exclude_objects:
            self.spawned_objects.pop(obj)
        return exclude_objects

    def get_objects(self, *args, **kwargs):
        return self.engine.get_objects(*args, *kwargs)

    def change_object_name(self, obj, new_name):
        """
        Change the name of one object, Note: it may bring some bugs if abusing!
        """
        self.engine.change_object_name(obj, new_name)
        obj = self.spawned_objects.pop(obj.name)
        self.spawned_objects[new_name] = obj
        obj.name = new_name

    def add_policy(self, object_id, policy_class, *policy_args, **policy_kwargs):
        return self.engine.add_policy(object_id, policy_class, *policy_args, **policy_kwargs)

    def get_policy(self, object_id):
        return self.engine.get_policy(object_id)

    def has_policy(self, object_id, policy_cls=None):
        return self.engine.has_policy(object_id, policy_cls)

    def get_state(self):
        """This function will be called by RecordManager to collect manager state, usually some mappings"""
        return {"spawned_objects": {name: v.class_name for name, v in self.spawned_objects.items()}}

    def set_state(self, state: dict, old_name_to_current=None):
        """
        A basic function for restoring spawned objects mapping
        """
        assert self.episode_step == 0, "This func can only be called after env.reset() without any env.step() called"
        if old_name_to_current is None:
            old_name_to_current = {key: key for key in state.keys()}
        spawned_objects = state["spawned_objects"]
        ret = {}
        for name, class_name in spawned_objects.items():
            current_name = old_name_to_current[name]
            name_obj = self.engine.get_objects([current_name])
            assert current_name in name_obj and name_obj[current_name
                                                         ].class_name == class_name, "Can not restore mappings!"
            ret[current_name] = name_obj[current_name]
        self.spawned_objects = ret

    @property
    def class_name(self):
        return self.__class__.__name__

    @property
    def engine(self):
        from metadrive.engine.engine_utils import get_engine
        return get_engine()

    def get_metadata(self):
        """
        This function will store the metadata of each manager before the episode start, usually, we put some raw real
        world data in it, so that we won't lose information
        """
        assert self.episode_step == 0, "This func can only be called after env.reset() without any env.step() called"
        return {}

    @property
    def global_config(self):
        return get_global_config()


class BaseAgentManager(BaseManager):
    """
    This manager allows one to use object like vehicles/traffic lights as agent with multi-agent support.
    You would better make your own agent manager based on this class
    """

    INITIALIZED = False  # when the reset() and init() are called, it will be set to True

    def __init__(self, init_observations):
        """
        The real init is happened in self.init(), in which super().__init__() will be called
        """
        # for getting {agent_id: BaseObject}, use agent_manager.active_agents
        self._active_objects = {}  # {object.id: BaseObject}

        # fake init. before creating engine, it is necessary when all objects re-created in runtime
        self.observations = copy.copy(init_observations)  # its value is map<agent_id, obs> before init() is called
        self._init_observations = init_observations  # map <agent_id, observation>

        # init spaces before initializing env.engine
        observation_space = {
            agent_id: single_obs.observation_space
            for agent_id, single_obs in init_observations.items()
        }
        init_action_space = self._get_action_space()
        assert isinstance(init_action_space, dict)
        assert isinstance(observation_space, dict)
        self._init_observation_spaces = observation_space
        self._init_action_spaces = init_action_space
        self.observation_spaces = copy.copy(observation_space)
        self.action_spaces = copy.copy(init_action_space)
        self.episode_created_agents = None

        # this map will be override when the env.init() is first called and objects are made
        self._agent_to_object = {k: k for k in self.observations.keys()}  # no target objects created, fake init
        self._object_to_agent = {k: k for k in self.observations.keys()}  # no target objects created, fake init

        self._debug = None

    def _get_action_space(self):
        from metadrive.engine.engine_utils import get_global_config
        if self.global_config["is_multi_agent"]:
            return {v_id: self.agent_policy.get_input_space() for v_id in get_global_config()["agent_configs"].keys()}
        else:
            return {DEFAULT_AGENT: self.agent_policy.get_input_space()}

    @property
    def agent_policy(self):
        """
        Return the agent policy
        Returns: Agent Poicy class
        Make sure you access the global config via get_global_config() instead of self.engine.global_config
        """
        from metadrive.engine.engine_utils import get_global_config
        return get_global_config()["agent_policy"]

    def before_reset(self):
        if not self.INITIALIZED:
            super(BaseAgentManager, self).__init__()
            self.INITIALIZED = True
        self.episode_created_agents = None

        for v in list(self._active_objects.values()):
            if hasattr(v, "before_reset"):
                v.before_reset()
        super(BaseAgentManager, self).before_reset()

    def _create_agents(self, config_dict):
        """
        It should create a set of vehicles or other objects serving as agents
        Args:
            config_dict:

        Returns:

        """
        raise NotImplementedError

    def reset(self):
        """
        Agent manager is really initialized after the BaseObject Instances are created
        """
        config = self.engine.global_config
        self._debug = config["debug"]
        self.episode_created_agents = self._create_agents(config_dict=self.engine.global_config["agent_configs"])

    def after_reset(self):
        init_objects = self.episode_created_agents
        objects_created = set(init_objects.keys())
        objects_in_config = set(self._init_observations.keys())
        assert objects_created == objects_in_config, "{} not defined in target objects config".format(
            objects_created.difference(objects_in_config)
        )

        # it is used when reset() is called to reset its original agent_id
        self._agent_to_object = {agent_id: object.name for agent_id, object in init_objects.items()}
        self._object_to_agent = {object.name: agent_id for agent_id, object in init_objects.items()}
        self._active_objects = {v.name: v for v in init_objects.values()}

        # real init {obj_name: space} map
        self.observations = dict()
        self.observation_spaces = dict()
        self.action_spaces = dict()
        for agent_id, object in init_objects.items():
            self.observations[object.name] = self._init_observations[agent_id]
            obs_space = self._init_observation_spaces[agent_id]
            self.observation_spaces[object.name] = obs_space
            action_space = self._init_action_spaces[agent_id]
            self.action_spaces[object.name] = action_space
            assert isinstance(action_space, Space)

    def set_state(self, state: dict, old_name_to_current=None):
        super(BaseAgentManager, self).set_state(state, old_name_to_current)
        created_agents = state["created_agents"]
        created_agents = {agent_id: old_name_to_current[obj_name] for agent_id, obj_name in created_agents.items()}
        episode_created_agents = {}
        for a_id, name in created_agents.items():
            episode_created_agents[a_id] = self.engine.get_objects([name])[name]
        self.episode_created_agents = episode_created_agents

    def get_state(self):
        ret = super(BaseAgentManager, self).get_state()
        agent_info = {agent_id: obj.name for agent_id, obj in self.episode_created_agents.items()}
        ret["created_agents"] = agent_info
        return ret

    def try_actuate_agent(self, step_infos, stage="before_step"):
        """
        Some policies should make decision before physics world actuation, in particular, those need decision-making
        But other policies like ReplayPolicy should be called in after_step, as they already know the final state and
        exempt the requirement for rolling out the dynamic system to get it.
        """
        assert stage == "before_step" or stage == "after_step"
        for agent_id in self.active_agents.keys():
            policy = self.get_policy(self._agent_to_object[agent_id])
            assert policy is not None, "No policy is set for agent {}".format(agent_id)
            if stage == "before_step":
                action = policy.act(agent_id)
                step_infos[agent_id] = policy.get_action_info()
                step_infos[agent_id].update(self.get_agent(agent_id).before_step(action))

        return step_infos

    def before_step(self):
        # not in replay mode
        step_infos = self.try_actuate_agent(dict(), stage="before_step")
        return step_infos

    def after_step(self, *args, **kwargs):
        step_infos = self.try_actuate_agent({}, stage="after_step")
        step_infos.update(self.for_each_active_agents(lambda v: v.after_step()))
        return step_infos

    def _translate(self, d):
        return {self._object_to_agent[k]: v for k, v in d.items()}

    def get_observations(self):
        if hasattr(self, "engine") and self.engine.replay_episode:
            return self.engine.replay_manager.get_replay_agent_observations()
        else:
            ret = {}
            for obj_id, observation in self.observations.items():
                if self.is_active_object(obj_id):
                    ret[self.object_to_agent(obj_id)] = observation
            return ret

    def get_observation_spaces(self):
        ret = {}
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
        Return Map<agent_id, BaseObject>
        """
        if hasattr(self, "engine") and self.engine is not None and self.engine.replay_episode:
            return self.engine.replay_manager.replay_agents
        else:
            return {self._object_to_agent[k]: v for k, v in self._active_objects.items()}

    def get_agent(self, agent_name, raise_error=True):
        object_name = self.agent_to_object(agent_name)
        if object_name in self._active_objects:
            return self._active_objects[object_name]
        else:
            if raise_error:
                raise ValueError("Object {} not found!".format(object_name))
            else:
                return None

    def object_to_agent(self, obj_name):
        """
        We recommend to use engine.agent_to_object() or engine.object_to_agent() instead of the ones in agent_manager,
        since this two functions DO NOT work when replaying episode.
        :param obj_name: BaseObject.name
        :return: agent id
        """
        return self._object_to_agent[obj_name]

    def agent_to_object(self, agent_id):
        """
        We recommend to use engine.agent_to_object() or engine.object_to_agent() instead of the ones in agent_manager,
        since this two functions DO NOT work when replaying episode.
        """
        return self._agent_to_object[agent_id]

    def destroy(self):
        # when new agent joins in the game, we only change this two maps.
        if self.INITIALIZED:
            super(BaseAgentManager, self).destroy()
        self._agent_to_object = {}
        self._object_to_agent = {}
        self._active_objects = {}
        for obs in self.observations.values():
            obs.destroy()
        self.observations = {}
        self.observation_spaces = {}
        self.action_spaces = {}

        self.INITIALIZED = False

    def for_each_active_agents(self, func, *args, **kwargs):
        """
        This func is a function that take each object as the first argument and *arg and **kwargs as others.
        """
        assert len(self.active_agents) > 0, "Not enough objects exist!"
        ret = dict()
        for k, v in self.active_agents.items():
            ret[k] = func(v, *args, **kwargs)
        return ret
