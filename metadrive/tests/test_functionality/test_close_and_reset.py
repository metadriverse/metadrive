from metadrive.envs import MetaDriveEnv


# Related issue:
# https://github.com/metadriverse/metadrive/issues/191
def test_close_and_reset():

    env = MetaDriveEnv({"start_seed": 1000, "environment_num": 1})
    eval_env = MetaDriveEnv()
    env.reset()
    for i in range(100):
        env.step(env.action_space.sample())
    env.close()
    env.reset()
    for i in range(100):
        env.step(env.action_space.sample())
    env.close()

    eval_env.reset()
    for i in range(100):
        eval_env.step(eval_env.action_space.sample())
    eval_env.close()
    env.reset()
    for i in range(100):
        env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    test_close_and_reset()
