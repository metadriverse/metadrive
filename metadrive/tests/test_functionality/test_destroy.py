from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger

setup_logger(debug=True)


def _test_destroy(config):
    env = MetaDriveEnv(config)
    try:
        env.reset()
        for i in range(1, 20):
            env.step([1, 1])

        env.close()
        env.reset()
        env.close()
        env.close()
        env.reset()
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


def _test_destroy_rgb(obs="rgb"):
    # Close and reset
    config = {"num_scenarios": 1, "start_seed": 3, "manual_control": False}
    if obs == "state":
        pass
    elif obs == "rgb":
        config["image_observation"] = True
    else:
        config["use_render"] = True
    _test_destroy(config)


def test_destroy_state(obs="state"):
    # Close and reset
    config = {"num_scenarios": 1, "start_seed": 3, "manual_control": False}
    if obs == "state":
        pass
    elif obs == "rgb":
        config["image_observation"] = True
        config["use_render"] = False
        config["norm_pixel"] = True
    _test_destroy(config)


if __name__ == "__main__":
    # test_destroy_rgb()
    test_destroy_state()
