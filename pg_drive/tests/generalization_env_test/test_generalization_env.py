from pg_drive.envs.generalization_racing import GeneralizationRacing
from pg_drive.scene_creator.map import Map, MapGenerateMethod
from pg_drive.utils import setup_logger

setup_logger(debug=True)


class ResetEnv(GeneralizationRacing):
    def __init__(self):
        super(ResetEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.1,
                "start_seed": 0,
                "pg_world_config": {
                    "debug": False,
                },
                "image_buffer_name": "mini_map",
                "manual_control": True,
                "use_render": True,
                "use_rgb": False,
                "steering_penalty": 0.0,
                "decision_repeat": 5,
                "rgb_clip": True,
                "map_config": {
                    Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_NUM,
                    Map.GENERATE_PARA: 12,
                    Map.LANE_WIDTH: 3.5,
                    Map.LANE_NUM: 2,
                }
            }
        )


if __name__ == "__main__":
    env = ResetEnv()

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
        if d:
            print("Reset")
            env.reset()
    env.close()
