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
                "agent_configs": {"agent{}".format(i): {
                    "spawn_longitude": i * 5
                }
                                  for i in range(4)}
            }
        )


if __name__ == "__main__":
    setup_logger(True)
    env = TestEnv()

    o, _ = env.reset()
    # print("vehicle num", len(env.engine.traffic_manager.vehicles))
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step({key: [0, 0] for key in env.action_space.sample()})
        # o, r, tm, tc, info = env.step([0,1])
        env.render(text={"display_regions": len(env.engine.win.getDisplayRegions())})
        if True in tm.values() or True in tc.values():
            # print("Reset")
            env.reset()
    env.close()
