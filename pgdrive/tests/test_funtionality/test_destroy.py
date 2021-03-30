from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import setup_logger

setup_logger(debug=True)


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "start_seed": 3,
                "target_vehicle_configs": {
                    "default_agent": {
                        "use_image": False,
                        "image_source": "depth_cam"
                    }
                },
                "pg_world_config": {
                    "debug": False
                },
                "manual_control": False
            }
        )
        # self.pg_world.cam.setPos(0, 0, 1500)
        # self.pg_world.cam.lookAt(0, 0, 0)


def test_destroy():
    # Close and reset
    env = TestEnv()
    try:
        env.reset()
        for i in range(1, 20):
            env.step([1, 1])

        env.close()
        env.reset()
        env.close()

        # Again!
        env = TestEnv()
        env.reset()
        for i in range(1, 20):
            env.step([1, 1])
        env.reset()
        env.close()
    finally:
        env.close()


if __name__ == "__main__":
    test_destroy()
