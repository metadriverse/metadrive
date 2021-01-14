from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.map import Map, MapGenerateMethod
from pgdrive.utils import setup_logger

# setup_logger(True)


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 5,
                "traffic_density": 0.1,
                "traffic_mode": "reborn",
                "start_seed": 5,
                # "controller": "joystick",
                "manual_control": True,
                "use_render": True,
                "use_saver": True,
                "map": 30
            }
        )


if __name__ == "__main__":
    env = TestEnv()

    o = env.reset()
    # env.pg_world.force_fps.toggle()
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        text = {"save": env.save_mode}
        env.render(text=text)
        # if d:
        #     print("Reset")
        #     env.reset()
    env.close()
