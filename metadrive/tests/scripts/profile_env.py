from metadrive.envs.metadrive_env import MetaDriveEnv

if __name__ == "__main__":
    env = MetaDriveEnv(
        {
            "map": 30,
            "environment_num": 1,
            "traffic_density": 0.1,
            "pstats": True,
            "traffic_mode": "respawn"
        }
    )

    o = env.reset()
    for i in range(1, 10000):
        print(i)
        o, r, d, info = env.step([0, 0])
    env.close()
