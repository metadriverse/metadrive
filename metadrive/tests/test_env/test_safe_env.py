from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv


def test_safe_env():
    env = SafeMetaDriveEnv({"environment_num": 100, "start_seed": 75})
    try:
        o = env.reset()
        total_cost = 0
        for i in range(1, 100):
            o, r, d, info = env.step([0, 1])
            total_cost += info["cost"]
            assert env.observation_space.contains(o)
            if d:
                total_cost = 0
                print("Reset")
                env.reset()
        env.close()
    finally:
        env.close()


if __name__ == '__main__':
    test_safe_env()
