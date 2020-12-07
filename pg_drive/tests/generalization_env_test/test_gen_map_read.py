import os

from pg_drive.envs.generalization_racing import GeneralizationRacing
from pg_drive.scene_creator.map import Map, MapGenerateMethod

from pg_drive.utils import setup_logger

setup_logger(debug=True)


class TestEnv(GeneralizationRacing):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.1,
                "start_seed": 4,
                "pg_world_config": {
                    "debug": False,
                },
                "vehicle_config": {
                    #     "mini_map": (256, 256, 100),
                    # "rgb_cam": (256, 256),
                },
                "image_source": "mini_map",
                "manual_control": True,
                "use_render": True,
                "use_image": False,
                "steering_penalty": 0.0,
                "decision_repeat": 5,
                "rgb_clip": True,
                "map_config": {
                    Map.GENERATE_METHOD: MapGenerateMethod.PG_MAP_FILE,
                    Map.GENERATE_PARA: VisLoader.file_path(os.path.dirname(__file__), "map_1.json"),
                }
            }
        )


if __name__ == "__main__":
    env = TestEnv()

    env.reset()
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])

        env.render(text={"can you see me": i})
        # if d:
        #     print("Reset")
        #     env.reset()
    env.close()
