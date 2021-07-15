from pgdrive.envs.pgdrive_env_v2 import PGDriveEnvV2


def test_seeding():
    env = PGDriveEnvV2()
    try:
        env.seed(999999)
        assert env.pgdrive_engine is None
        assert env.current_seed != 999999
        env.reset()
        assert env.current_seed == 999999
        assert env.pgdrive_engine is not None
    finally:
        env.close()


if __name__ == '__main__':
    test_seeding()
