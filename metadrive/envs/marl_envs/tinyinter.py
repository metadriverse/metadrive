from metadrive.envs.marl_envs.marl_intersection import MultiAgentIntersectionEnv
from metadrive.manager.agent_manager import AgentManager
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.utils import Config

import copy

class MixedIDMAgentManager(AgentManager):
    """In this manager, we can replace part of RL policy by IDM policy"""

    def __init__(self, init_observations, init_action_space, num_RL_agents):
        super(MixedIDMAgentManager, self).__init__(init_observations=init_observations, init_action_space=init_action_space)
        self.num_RL_agents = num_RL_agents
        self.RL_agents = set()

    def filter_RL_agents(self, source_dict):
        new_ret = {}
        for k in self.RL_agents:
            assert k in source_dict
            new_ret[k] = source_dict[k]
        assert len(new_ret) <= self.num_RL_agents
        return new_ret

    def get_observation_spaces(self):
        return self.filter_RL_agents(super(MixedIDMAgentManager, self).get_observation_spaces())

    def get_action_spaces(self):
        return self.filter_RL_agents(super(MixedIDMAgentManager, self).get_action_spaces())

    # @property
    # def active_agents(self):
    #     return self.filter_RL_agents(super(MixedIDMAgentManager, self).active_agents)

    def finish(self, agent_name, ignore_delay_done=False):
        if agent_name in self.RL_agents:
            self.RL_agents.remove(agent_name)
        super(MixedIDMAgentManager, self).finish(agent_name=agent_name, ignore_delay_done=ignore_delay_done)

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
            if len(self.RL_agents) >= self.num_RL_agents:
                policy = IDMPolicy(obj, self.generate_seed())
            else:
                policy = self._get_policy(obj)
                self.RL_agents.add(agent_id)
            self.add_policy(obj.id, policy)
        return ret


class MultiAgentTinyInter(MultiAgentIntersectionEnv):
    @staticmethod
    def default_config() -> Config:
        tiny_config = dict(
            num_agents=8, map_config=dict(
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
        d = self.agent_manager.filter_RL_agents(d)
        if "__all__" in d:
            d.pop("__all__")
        d["__all__"] = all(d.values())
        return (
            self.agent_manager.filter_RL_agents(o),
            self.agent_manager.filter_RL_agents(r),
            d,
            self.agent_manager.filter_RL_agents(i),
        )


    def _preprocess_actions(self, actions):
        actions = {v_id: actions[v_id] for v_id in self.vehicles.keys() if v_id in self.agent_manager.RL_agents}
        return actions

    def __init__(self, config=None):
        self.num_RL_agents = 1
        super(MultiAgentTinyInter, self).__init__(config=config)
        self.num_agents = self.num_RL_agents  # Hack


        self.agent_manager = MixedIDMAgentManager(
            init_observations=self._get_observations(), init_action_space=self._get_action_space(), num_RL_agents=self.num_RL_agents
        )




if __name__ == '__main__':
    from metadrive.envs.marl_envs import MultiAgentTinyInter





if __name__ == "__main__":
    env = MultiAgentTinyInter(config={
        "vehicle_config": {"show_line_to_dest": True, "lidar": {"num_others": 2, "add_others_navi": True}},
        # "manual_control": True,
        # "use_render": True,
    })
    o = env.reset()
    print("vehicle num", len(env.engine.traffic_manager.vehicles))
    print("RL agent num", len(o))
    for i in range(1, 100000):
        o, r, d, info = env.step({k: [0, 1] for k in env.action_space.sample().keys()})
        env.render("top_down", camera_position=(50, 0), film_size=(1000, 1000))
        vehicles = env.vehicles

        if not d["__all__"]:
            assert sum([env.engine.get_policy(v.name).__class__.__name__ == "EnvInputPolicy" for k, v in vehicles.items()]) == 1
            assert len(env.observation_space) == 1
            assert len(env.action_space) == 1

        if any(d.values()):
            print("Somebody dead.", d)

        # if True in d.values():
        #     print("Somebody Done. ", info)
            # env.reset()
        if d["__all__"]:
            print("Reset.", info)
            env.reset()
    env.close()
