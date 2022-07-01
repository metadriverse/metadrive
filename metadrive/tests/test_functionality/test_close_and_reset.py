from metadrive.envs import MetaDriveEnv


# Related issue:
# https://github.com/metadriverse/metadrive/issues/191
def test_close_and_reset():
    env = MetaDriveEnv(MetaDriveEnv.default_config())
    env.reset()
    env.close()
    env.reset()


if __name__ == '__main__':
    test_close_and_reset()
