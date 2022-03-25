from metadrive.envs.marl_envs.marl_intersection import MultiAgentIntersectionEnv
from metadrive.manager.agent_manager import AgentManager
from metadrive.policy.AI_protect_policy import AIProtectPolicy
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.utils import Config


class MixedIDMAgentManager(AgentManager):
    """In this manager, we can replace part of RL policy by IDM policy"""

    def __init__(self, init_observations, init_action_space, num_RL_agents):
        super(MixedIDMAgentManager, self).__init__(init_observations=init_observations, init_action_space=init_action_space)
        self.num_RL_agents = num_RL_agents
        self.RL_agents = set()


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

            if agent_count < self.RL_agents:
                pass
                self.RL_agents.add(agent_id)
            else:
                policy = self._get_policy(obj)
            self.add_policy(obj.id, policy)
        return ret

    def _get_policy(self, obj):
        # note: agent.id = object id
        if self.engine.global_config["agent_policy"] is not None:
            return self.engine.global_config["agent_policy"](obj, self.generate_seed())
        if self.engine.global_config["manual_control"]:
            if self.engine.global_config.get("use_AI_protector", False):
                policy = AIProtectPolicy(obj, self.generate_seed())
            else:
                policy = ManualControlPolicy(obj, self.generate_seed())
        else:
            policy = EnvInputPolicy(obj, self.generate_seed())
        return policy


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

    def __init__(self, config=None):
        super(MultiAgentTinyInter, self).__init__(config=config)
        self.agent_manager = MixedIDMAgentManager(
            init_observations=self._get_observations(), init_action_space=self._get_action_space()
        )


if __name__ == '__main__':
    from metadrive.envs.marl_envs import MultiAgentTinyInter


class TestEnv(MultiAgentTinyInter):
    def __init__(self):
        super(TestEnv, self).__init__(
            config={

                # "num_agents": 8,
                # "map_config": {
                #     "exit_length": 30,
                #     "lane_num": 1,
                #     "lane_width": 4
                # },

                # === Debug ===
                "vehicle_config": {
                    "show_line_to_dest": True
                },
                "manual_control": True,
                # "num_agents": 4,
                "use_render": True,
            }
        )


if __name__ == "__main__":
    env = TestEnv()
    o = env.reset()
    print("vehicle num", len(env.engine.traffic_manager.vehicles))
    for i in range(1, 100000):
        o, r, d, info = env.step(env.action_space.sample())
        if True in d.values():
            print("Somebody Done. ", info)
            # env.reset()
    env.close()
