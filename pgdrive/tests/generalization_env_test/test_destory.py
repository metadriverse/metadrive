from pgdrive.envs.generalization_racing import GeneralizationRacing
from pgdrive.utils import setup_logger

setup_logger(debug=True)


class TestEnv(GeneralizationRacing):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "start_seed": 3,
                "pg_world_config": {
                    "debug": False
                },
                "manual_control": True
            }
        )
        # self.pg_world.cam.setPos(0, 0, 1500)
        # self.pg_world.cam.lookAt(0, 0, 0)


if __name__ == "__main__":
    # Close and reset
    env = TestEnv()
    env.reset()
    for i in range(1, 20):
        env.step([1, 1])

    env.close()
    env.reset()
    env.close()

    # Again!
    env2 = TestEnv()
    env2.reset()
    for i in range(1, 20):
        env2.step([1, 1])
    env2.reset()
    env2.close()
