from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
from metadrive.utils import setup_logger


class TestEnv(MultiAgentIntersectionEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            config={
                "use_render": True,
                "num_agents": 8,
                "map_config": {
                    "exit_length": 30,
                    "lane_num": 1
                }
            }
        )


if __name__ == "__main__":
    env = TestEnv()
    o = env.reset()
    print("vehicle num", len(env.engine.traffic_manager.vehicles))
    for i in range(1, 100000):
        o, r, d, info = env.step(env.action_space.sample())
        if True in d.values():
            print("Reset")
            env.reset()
    env.close()
