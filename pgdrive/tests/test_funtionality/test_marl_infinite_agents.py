from pgdrive.envs.marl_envs.marl_inout_roundabout import MultiAgentRoundaboutEnv


def test_infinite_agents():
    env = MultiAgentRoundaboutEnv({"num_agents": -1, "delay_done": 0, "horizon": 50})
    try:
        o = env.reset()
        max_num = old_num_of_vehicles = len(env.vehicles)
        for i in range(1, 300):
            o, r, d, info = env.step({k: [0, 1] for k in env.vehicles})
            # print("Current active agents: ", len(env.vehicles),
            #       ". Objects: ", len(env.agent_manager._object_to_agent))
            max_num = max(len(env.vehicles), max_num)
            # env.render(mode="top_down")
            for kkk, iii in info.items():
                if d[kkk]:
                    assert iii["episode_length"] > 1
            if d["__all__"]:
                o = env.reset()
                print("Finish {} steps.".format(i))
    finally:
        env.close()
    assert max_num > old_num_of_vehicles


if __name__ == '__main__':
    test_infinite_agents()
