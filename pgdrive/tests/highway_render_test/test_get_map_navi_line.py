from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.map import Map, MapGenerateMethod
from pgdrive.utils import setup_logger

setup_logger(debug=True)


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.1,
                "start_seed": 3,
                "pg_world_config": {
                    "debug": False,
                    "highway_render": False
                },
                "image_source": "mini_map",
                "manual_control": False,
                "use_render": False,
                "use_image": False,
                "steering_penalty": 0.0,
                "decision_repeat": 5,
                "rgb_clip": True,
                "map_config": {
                    Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                    Map.GENERATE_PARA: "OCrRCTXRCCCCrOr",
                    Map.LANE_WIDTH: 3.5,
                    Map.LANE_NUM: 4,
                }
            }
        )


if __name__ == "__main__":
    env = TestEnv()
    env.reset()
    env.current_map.draw_map_with_navi_lines(env.vehicle, dest_resolution=(4096, 4096), save=True)
