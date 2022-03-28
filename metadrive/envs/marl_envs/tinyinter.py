from metadrive.envs.marl_envs.marl_intersection import MultiAgentIntersectionEnv
from metadrive.manager.agent_manager import AgentManager
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import Config
import copy

class MixedIDMAgentManager(AgentManager):
    """In this manager, we can replace part of RL policy by IDM policy"""
    def __init__(self, init_observations, init_action_space, num_RL_agents):
        super(MixedIDMAgentManager, self).__init__(
            init_observations=init_observations, init_action_space=init_action_space
        )
        self.num_RL_agents = num_RL_agents
        self.RL_agents = set()
        self.dying_RL_agents = set()
        self.all_previous_RL_agents = set()

    def filter_RL_agents(self, source_dict, original_done_dict=None):

        # new_ret = {k: v for k, v in source_dict.items() if k in self.RL_agents}

        new_ret = {k: v for k, v in source_dict.items() if k in self.all_previous_RL_agents}

        # if len(new_ret) > self.num_RL_agents:
        #     assert len(self.RL_agents) - len(self.dying_RL_agents) == self.num_RL_agents
        # if len(new_ret) < self.num_RL_agents:
        #     print("Something wrong!")
        return new_ret

    def get_observation_spaces(self):
        ret = self.filter_RL_agents(super(MixedIDMAgentManager, self).get_observation_spaces())
        if len(ret) == 0:
            k, v = list(super(MixedIDMAgentManager, self).get_observation_spaces().items())[0]
            return {k: v}
        else:
            return ret

    def get_action_spaces(self):
        ret = self.filter_RL_agents(super(MixedIDMAgentManager, self).get_action_spaces())
        if len(ret) == 0:
            k, v = list(super(MixedIDMAgentManager, self).get_action_spaces().items())[0]
            return {k: v}
        else:
            return ret

    def finish(self, agent_name, ignore_delay_done=False):
        # ignore_delay_done = True
        if agent_name in self.RL_agents:
            self.dying_RL_agents.add(agent_name)
        super(MixedIDMAgentManager, self).finish(agent_name, ignore_delay_done)

    def _remove_vehicle(self, v):
        agent_name = self.object_to_agent(v.name)
        if agent_name in self.RL_agents:
            self.RL_agents.remove(agent_name)
        if agent_name in self.dying_RL_agents:
            self.dying_RL_agents.remove(agent_name)
        super(MixedIDMAgentManager, self)._remove_vehicle(v)

    def _get_vehicles(self, config_dict: dict):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        ret = {}
        for agent_count, (agent_id, v_config) in enumerate(config_dict.items()):
            if self.engine.global_config["random_agent_model"]:
                v_type = random_vehicle_type(self.np_random)
            else:
                if v_config.get("vehicle_model", False):
                    v_type = vehicle_type[v_config["vehicle_model"]]
                else:
                    v_type = vehicle_type["default"]
            obj = self.spawn_object(v_type, vehicle_config=v_config)
            ret[agent_id] = obj
            if (len(self.RL_agents) - len(self.dying_RL_agents)) >= self.num_RL_agents:
                policy = IDMPolicy(obj, self.generate_seed())
                obj._use_special_color = False
            else:
                policy = self._get_policy(obj)
                self.RL_agents.add(agent_id)
                self.all_previous_RL_agents.add(agent_id)
                obj._use_special_color = True
            self.add_policy(obj.id, policy)
        return ret

    def reset(self):
        self.RL_agents.clear()
        self.dying_RL_agents.clear()
        self.all_previous_RL_agents.clear()
        super(MixedIDMAgentManager, self).reset()

    @property
    def allow_respawn(self):
        """In native implementation, we can only respawn new vehicle when the total vehicles in the scene
        is less then num_agents. But here, since we always have N-K IDM vehicles and K RL vehicles, so
        allow_respawn is always False. We set to True to allow spawning new vehicle.
        """
        if self._allow_respawn is False:
            return False

        if len(self.active_agents) < self.engine.global_config["num_agents"]:
            return True

        if len(self.RL_agents) - len(self.dying_RL_agents) < self.num_RL_agents:
            return True

        return False


class MultiAgentTinyInter(MultiAgentIntersectionEnv):
    @staticmethod
    def default_config() -> Config:
        tiny_config = dict(
            num_agents=8,
            num_RL_agents=1,

            map_config=dict(
                exit_length=30,
                lane_num=1,
                lane_width=4,
            )
        )
        return MultiAgentIntersectionEnv.default_config().update(tiny_config, allow_add_new_key=True)

    def _get_reset_return(self):
        return self.agent_manager.filter_RL_agents(super(MultiAgentTinyInter, self)._get_reset_return())

    def step(self, actions):
        o, r, d, i = super(MultiAgentTinyInter, self).step(actions)
        original_done_dict = copy.deepcopy(d)
        d = self.agent_manager.filter_RL_agents(d, original_done_dict=original_done_dict)
        if "__all__" in d:
            d.pop("__all__")
        # assert len(d) == self.agent_manager.num_RL_agents, d
        d["__all__"] = all(d.values())
        return (
            self.agent_manager.filter_RL_agents(o, original_done_dict=original_done_dict),
            self.agent_manager.filter_RL_agents(r, original_done_dict=original_done_dict),
            d,
            self.agent_manager.filter_RL_agents(i, original_done_dict=original_done_dict),
        )

    def _preprocess_actions(self, actions):
        actions = {v_id: actions[v_id] for v_id in self.vehicles.keys() if v_id in self.agent_manager.RL_agents}
        return actions

    def __init__(self, config=None):
        config = config or {}
        self.num_RL_agents = config.get("num_RL_agents", self.default_config()["num_RL_agents"])
        super(MultiAgentTinyInter, self).__init__(config=config)
        self.agent_manager = MixedIDMAgentManager(
            init_observations=self._get_observations(),
            init_action_space=self._get_action_space(),
            num_RL_agents=self.num_RL_agents
        )


if __name__ == '__main__':
    env = MultiAgentTinyInter(
        config={
            "num_agents": 8,
            "num_RL_agents": 8,

            # "vehicle_config": {
            #     "show_line_to_dest": True,
            #     "lidar": {
            #         "num_others": 2,
            #         "add_others_navi": True
            #     }
            # },
            # "manual_control": True,
            # "use_render": True,
        }
    )
    o = env.reset()
    print("vehicle num", len(env.engine.traffic_manager.vehicles))
    print("RL agent num", len(o))
    for i in range(1, 100000):
        o, r, d, info = env.step({k: [0.01, 1] for k in env.action_space.sample().keys()})
        env.render("top_down", camera_position=(42.5, 0), film_size=(1000, 1000))
        vehicles = env.vehicles
        # if not d["__all__"]:
        #     assert sum(
        #         [env.engine.get_policy(v.name).__class__.__name__ == "EnvInputPolicy" for k, v in vehicles.items()]
        #     ) == env.config["num_RL_agents"]
        if any(d.values()):
            print("Somebody dead.", d, info)
            print("Step {}. Policies: {}".format(i, {k: v['policy'] for k, v in info.items()}))
        if d["__all__"]:
            # assert i >= 1000
            print("Reset. ", i, info)
            break
            env.reset()
    env.close()
