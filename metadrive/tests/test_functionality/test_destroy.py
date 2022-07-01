from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger

setup_logger(debug=True)


def test_destroy(obs="state"):
    # Close and reset
    config = {"environment_num": 1, "start_seed": 3, "manual_control": False}
    if obs == "state":
        pass
    elif obs == "rgb":
        config["offscreen_render"] = True
    else:
        config["use_render"] = True
    env = MetaDriveEnv(config)
    try:
        env.reset()
        for i in range(1, 20):
            env.step([1, 1])

        env.close()
        env.reset()
        env.close()

        # Again!
        env = MetaDriveEnv(config)
        env.reset()
        for i in range(1, 20):
            env.step([1, 1])
        env.reset()
        env.close()
    finally:
        env.close()


if __name__ == "__main__":
    test_destroy("state")
    test_destroy("rgb")
    test_destroy("online_rgb")
