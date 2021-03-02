from pgdrive.envs.remote_pgdrive_env import RemotePGDrive


def test_remote_pgdrive_env():
    # Test
    envs = [RemotePGDrive(dict(map=7)) for _ in range(3)]
    ret = [env.reset() for env in envs]
    print(ret)
    ret = [env.step(env.action_space.sample()) for env in envs]
    print(ret)
    [env.reset() for env in envs]
    [env.close() for env in envs]
    print('Success!')


if __name__ == '__main__':
    test_remote_pgdrive_env()
