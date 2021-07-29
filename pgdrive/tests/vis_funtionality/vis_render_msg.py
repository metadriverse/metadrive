from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.map.base_map import BaseMap, MapGenerateMethod
from pgdrive.utils import setup_logger

setup_logger(debug=True)


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 4,
                "traffic_density": 0.1,
                "start_seed": 3,
                "pg_world_config": {
                    "debug": False,
                },
                "image_source": "mini_map",
                "manual_control": True,
                "use_render": True,
                "use_image": False,
                "steering_penalty": 0.0,
                "decision_repeat": 5,
                "rgb_clip": True,
                "map_config": {
                    BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
                    BaseMap.GENERATE_CONFIG: 12,
                    BaseMap.LANE_WIDTH: 3.5,
                    BaseMap.LANE_NUM: 3,
                }
            }
        )


if __name__ == "__main__":
    env = TestEnv()

    env.reset()
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        env.render(text={"Frame": i, "Speed": env.vehicle.speed})
    env.close()
