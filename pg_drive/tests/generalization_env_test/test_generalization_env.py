from pg_drive.envs.generalization_racing import GeneralizationRacing
from pg_drive.scene_creator.map import Map, MapGenerateMethod
from pg_drive.utils import setup_logger
from pg_drive.scene_manager.traffic_manager import TrafficMode

setup_logger(debug=True)


class TestEnv(GeneralizationRacing):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 4,
                "traffic_density": 0.5,
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
                    Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                    Map.GENERATE_PARA: "TTTTTTT",
                    Map.LANE_WIDTH: 3.5,
                    Map.LANE_NUM: 3,
                }
            }
        )


if __name__ == "__main__":
    env = TestEnv()

    env.reset()
    for i in range(1, 100000):
        # start = time.time()
        # print("Step: ", i)
        o, r, d, info = env.step([0, 1])
        # print(r)
        # print(o)
        # print(time.time() - start)
        # print(len(o), "Vs.", env.observation_space.shape[0])
        # print(info)
        env.render(text={"can you see me": i})
        # if d:
        #     print("Reset")
        #     env.reset()
    env.close()
