from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv


def test_episode_release():
    try:
        env = SafeMetaDriveEnv(
            {
                "use_render": False,
                "environment_num": 100,
                "accident_prob": .8,
                "traffic_density": 0.5,
                "debug": True
            }
        )
        o = env.reset()
        for i in range(1, 10):
            env.step([1.0, 1.0])
            env.step([1.0, 1.0])
            env.step([1.0, 1.0])
            env.step([1.0, 1.0])
            env.reset()
            env.reset()
    finally:
        env.close()
