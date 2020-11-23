from pg_drive.envs.generalization_racing import GeneralizationRacing
from pg_drive.utils import setup_logger

setup_logger(debug=True)


class ResetEnv(GeneralizationRacing):
    def __init__(self):
        super(ResetEnv, self).__init__(
            {
                "environment_num": 1,
                "start_seed": 3,
                "bt_world_config": {
                    "debug": False
                },
                "manual_control": True
            }
        )
        # self.bullet_world.cam.setPos(0, 0, 1500)
        # self.bullet_world.cam.lookAt(0, 0, 0)


if __name__ == "__main__":
    env = ResetEnv()
    env.reset()

    for i in range(1, 20):
        env.step([1, 1])
        # env.render(text={"can you see me": i})
    env.reset()
    # env.close()
