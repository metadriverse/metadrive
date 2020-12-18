from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.map import Map, MapGenerateMethod
from pgdrive.utils import setup_logger

setup_logger(debug=True)


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 4,
                "traffic_density": 0.0,
                "use_render": True,
                "map_config": {
                    Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                    Map.GENERATE_PARA: "SSSSSSSSSSSSS",
                }
            }
        )


if __name__ == "__main__":
    import numpy as np

    env = TestEnv()
    acc = [0, 1]
    brake = [-1, -np.nan]
    env.reset()
    for i in range(1, 100000):
        if i < 110:
            a = acc
        elif 110 < i < 120:
            a = brake
        else:
            a = [-1, -1]
        o, r, d, info = env.step(a)
        env.render(
            text="Old speed: {:.3f}\nnew speed: {:.3f}\ndiff: {:.3f}".format(
                env.vehicle.system.get_current_speed_km_hour(), env.vehicle.speed,
                env.vehicle.system.get_current_speed_km_hour() - env.vehicle.speed
            )
        )
    env.close()
