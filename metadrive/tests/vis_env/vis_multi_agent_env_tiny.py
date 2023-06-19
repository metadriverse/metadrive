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
    o, _ = env.reset()
    # print("vehicle num", len(env.engine.traffic_manager.vehicles))
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step(env.action_space.sample())
        if True in tm.values() or True in tc.values():
            print("Somebody Done. ", info)
            # env.reset()
    env.close()
