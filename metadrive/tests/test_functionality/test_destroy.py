from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger

setup_logger(debug=True)


def test_destroy():
    # Close and reset
    env = MetaDriveEnv({"environment_num": 1, "start_seed": 3, "manual_control": False})
    try:
        env.reset()
        for i in range(1, 20):
            env.step([1, 1])

        env.close()
        env.reset()
        env.close()

        # Again!
        env = MetaDriveEnv({"environment_num": 1, "start_seed": 3, "manual_control": False})
        env.reset()
        for i in range(1, 20):
            env.step([1, 1])
        env.reset()
        env.close()
    finally:
        env.close()


if __name__ == "__main__":
    test_destroy()
