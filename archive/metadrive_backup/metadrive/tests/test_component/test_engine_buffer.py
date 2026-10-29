from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv


def test_engine_buffer():
    env = MultiAgentRoundaboutEnv({"num_buffering_objects": 0, "debug": True})
    try:
        for i in range(1, 30):
            env.reset()
    finally:
        env.close()
    env = MultiAgentRoundaboutEnv({"num_buffering_objects": 200, "debug": True})
    try:
        for i in range(1, 30):
            env.reset()
    finally:
        env.close()


if __name__ == '__main__':
    test_engine_buffer()
