from metadrive.envs.marl_envs.marl_inout_roundabout import MultiAgentRoundaboutEnv
from metadrive.utils import setup_logger


class TestEnv(MultiAgentRoundaboutEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            config={
                "use_render": True,
                "map": "SSS",
                "num_agents": 4,
                "force_destroy": True,
                "manual_control": True,
                "target_vehicle_configs": {"agent{}".format(i): {
                    "spawn_longitude": i * 5
                }
                                           for i in range(4)}
            }
        )


if __name__ == "__main__":
    setup_logger(True)
    env = TestEnv()

    o = env.reset()
    print("vehicle num", len(env.engine.traffic_manager.vehicles))
    for i in range(1, 100000):
        o, r, d, info = env.step({key: [0, 0] for key in env.action_space.sample()})
        # o, r, d, info = env.step([0,1])
        env.render(text={"display_regions": len(env.engine.win.getDisplayRegions())})
        if True in d.values():
            print("Reset")
            env.reset()
    env.close()
