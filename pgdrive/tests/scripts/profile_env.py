from pgdrive.envs.pgdrive_env import PGDriveEnv


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "map": 30,
                "environment_num": 1,
                "traffic_density": 0.1,
                "pg_world_config": {
                    "pstats": True
                },
                "traffic_mode": "respawn"
            }
        )


if __name__ == "__main__":
    env = TestEnv()

    o = env.reset()
    for i in range(1, 10000):
        print(i)
        o, r, d, info = env.step([0, 0])
    env.close()
